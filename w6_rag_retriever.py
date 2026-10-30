"""
W6 本地 RAG 检索器（零依赖 · 离线可用 · 不耗 Token）
=================================================
**本文件是 W6 的主线实现**（唯一默认启用的检索方案）。

设计目标（呼应资源评估）：
  - HR 政策问答最大的 Token 陷阱是「每次把整份政策当 system prompt 端到端调用」。
  - 本检索器在本地用 BM25 只召回最相关的 k 个片段，拼进 prompt，
    让每次 API 调用只携带必要上下文 → 大幅省 Token、省钱、降延迟。
  - 纯标准库实现，免模型下载、免 API、跨设备零配置，无 GPU 也能跑。

为什么主线选 BM25 而不是向量检索：
  - 零依赖：不装 chromadb / sentence-transformers，不需要额外下载几百 MB 模型
  - 零外网：不需要 embedding API（对比：向量方案需注册硅基流动等第三方拿 Key）
  - 零阻塞：克隆下来 `pip install -r requirements.txt` 即可跑，无待办事项
  - 政策类文档术语固定（"年假""试用期""报销"），BM25 关键词匹配命中率已足够
  - 选型依据见 `framework_model_selection.md`：阶段一不引入重量级框架

进阶（可选，非必需）：
  - 若后续需要语义匹配（用户问法与文档用词差异大，如"休假"vs"年假"），
    可改用 `w6_hr_qa/` 子目录的向量检索方案（bge-m3 + Chroma）。
  - 该方案需额外安装 chromadb 并注册硅基流动获取 embedding Key，
    详见 `w6_hr_qa/README.md`。**默认不启用**，主线不受影响。

用法：
  from w6_rag_retriever import PolicyRetriever
  r = PolicyRetriever()                 # 默认读 policy_context.txt，空则回退 w6_sample_policy.txt
  hits = r.retrieve("试用期多久？", k=4)
  for text, score in hits:
      print(score, text[:60])

  # 或直接生成可直接塞进 prompt 的上下文
  ctx = r.build_context("年假怎么请？", k=4)

数据来源：
  - 优先读 `policy_context.txt`（放真实 HR 政策，文件默认为空）
  - 若为空则自动回退示例数据 `w6_sample_policy.txt` 并在终端提示
"""
from __future__ import annotations

import re
from math import log
from pathlib import Path

_DEFAULT_POLICY = "policy_context.txt"
_SAMPLE_POLICY = "w6_sample_policy.txt"

_CJK = re.compile(r"[\u4e00-\u9fff]+|[a-z0-9]+")

# 查询同义词扩展表（2026-09-06 评估发现：用户问法与语料用词不一致时 BM25 会失效，
# 如"标准工时"vs 语料的"工时时间"，且"规定"等高频词会把无关片段顶到 top-1）。
# 只在查询侧追加扩展词，不改动索引与打分算法；映射保守，仅覆盖实测失效的问法。
_SYNONYMS: dict[str, str] = {
    "标准工时": "工时 作息",
    "上班时间": "工时",
    "下班时间": "工时",
    "工作时间": "工时 作息",
    "病假条": "诊断书 病休",
    "病假证明": "诊断书 病休",
    "辞退": "解除劳动合同",
}


class PolicyRetriever:
    def __init__(self, policy_path: str | Path | None = None,
                 chunk_size: int = 320, overlap: int = 60):
        base = Path(__file__).resolve().parent
        if policy_path is None:
            policy_path = base / _DEFAULT_POLICY
        self.policy_path = Path(policy_path)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: list[str] = []
        self._tf: list[dict[str, int]] = []
        self._idf: dict[str, float] = {}
        self.avgdl = 1.0
        self._load()

    # ---- 文本处理 ----
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """中文：字符 unigram + bigram；英文/数字：整词。零依赖。"""
        text = text.lower()
        tokens: list[str] = []
        for m in _CJK.finditer(text):
            seg = m.group(0)
            if re.match(r"^[\u4e00-\u9fff]+$", seg):
                chars = list(seg)
                tokens.extend(chars)
                tokens.extend(chars[i] + chars[i + 1] for i in range(len(chars) - 1))
            else:
                tokens.append(seg)
        return tokens

    def _split(self, text: str) -> list[str]:
        """按段落切分；短标题行并入下一段（标题挂载）；超长段落用滑动窗口切块（带重叠）。"""
        paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
        # 标题挂载（2026-09-06）：形如"### 6.3.1 工时时间"的短标题行单独成块会变成
        # "标题空壳块"——占掉一个召回名次却无内容，且正文块失去章节语义。
        # 将连续短行（标题/小节号）并入其后第一个正文段。
        merged: list[str] = []
        buf = ""
        for p in paras:
            if len(p) <= 40:
                buf = (buf + "\n" + p) if buf else p
                continue
            merged.append((buf + "\n" + p) if buf else p)
            buf = ""
        if buf:
            merged.append(buf)
        paras = merged
        out: list[str] = []
        for p in paras:
            if len(p) <= self.chunk_size:
                out.append(p)
            else:
                step = max(1, self.chunk_size - self.overlap)
                for i in range(0, len(p), step):
                    piece = p[i:i + self.chunk_size].strip()
                    if piece:
                        out.append(piece)
        return out

    def _load(self) -> None:
        base = Path(__file__).resolve().parent
        if not self.policy_path.exists() or self.policy_path.stat().st_size == 0:
            sample = base / _SAMPLE_POLICY
            if sample.exists() and sample.stat().st_size > 0:
                print(f"[检索器] 未找到有效政策文件，回退使用示例数据：{sample.name}")
                self.policy_path = sample
            else:
                print("[检索器] 警告：policy_context.txt 与示例均为空，检索将无结果。"
                      "请把真实 HR 政策粘贴进 policy_context.txt。")
                self.chunks = []
                return
        text = self.policy_path.read_text(encoding="utf-8")
        self.chunks = self._split(text)
        df: dict[str, int] = {}
        self._tf = []
        for c in self.chunks:
            cnt: dict[str, int] = {}
            for t in self._tokenize(c):
                cnt[t] = cnt.get(t, 0) + 1
            self._tf.append(cnt)
            for t in cnt:
                df[t] = df.get(t, 0) + 1
        n = len(self.chunks)
        self.avgdl = sum(len(c) for c in self.chunks) / max(n, 1)
        self._idf = {t: log((n - f + 0.5) / (f + 0.5) + 1.0) for t, f in df.items()}

    @staticmethod
    def _expand_query(query: str) -> str:
        """查询侧同义词扩展：命中映射键则追加扩展词（只加 token，不改打分）。"""
        extra = [dst for src, dst in _SYNONYMS.items() if src in query]
        return query + (" " + " ".join(extra) if extra else "")

    # ---- 检索 ----
    def retrieve(self, query: str, k: int = 4) -> list[tuple[str, float]]:
        """返回 [(片段文本, 得分), ...] 按得分降序，最多 k 条。"""
        if not self.chunks:
            return []
        q = self._tokenize(self._expand_query(query))
        if not q:
            return []
        k1, b = 1.5, 0.75
        scored: list[tuple[float, int]] = []
        for i, cnt in enumerate(self._tf):
            dl = len(self.chunks[i])
            score = 0.0
            for t in q:
                if t in cnt:
                    tf = cnt[t]
                    idf = self._idf.get(t, 0.0)
                    score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / self.avgdl))
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [(self.chunks[i], score) for score, i in scored[:k]]

    def build_context(self, query: str, k: int = 4, max_chars: int = 2000) -> str:
        """把检索到的片段拼成可直接塞进 prompt 的上下文文本。"""
        hits = self.retrieve(query, k=k)
        parts, total = [], 0
        for idx, (text, _) in enumerate(hits, 1):
            if total + len(text) > max_chars:
                break
            parts.append(f"[政策片段 {idx}]\n{text}")
            total += len(text)
        return "\n\n".join(parts)


if __name__ == "__main__":
    r = PolicyRetriever()
    print(f"索引片段数：{len(r.chunks)}")
    for q in ["试用期是多久？", "加班怎么算？", "年假怎么请？"]:
        print(f"\n=== 查询：{q} ===")
        for text, score in r.retrieve(q, k=3):
            print(f"  {score:.3f} | {text[:50]}")
