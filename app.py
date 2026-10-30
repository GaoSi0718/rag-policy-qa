"""
企业制度问答 · Web 界面（Streamlit）
=====================================
只保留 RAG 问答主链路：检索 → 拼 context → 生成 → 显示引用。
多模态、会话持久化等模块与本文无关，未纳入本仓库。

运行：
    pip install -r requirements.txt
    在 .env 中写入 DEEPSEEK_API_KEY=sk-xxxx
    streamlit run app.py

设计要点：
    1. 防幻觉 —— system prompt 强制「只依据上下文回答，无依据即拒答」
    2. 可溯源 —— 答案末尾附命中的政策片段编号，用户可回查原文
    3. 省成本 —— 每次只送检索命中的片段，不送整本手册
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from w6_rag_retriever import PolicyRetriever

_ENV = Path(__file__).with_name(".env")
load_dotenv(_ENV)

SYSTEM_PROMPT = """你是企业内部制度问答助手。请严格遵守：
1. 只依据【政策片段】中的内容回答，不得使用你自己的知识补充。
2. 如果政策片段中找不到答案，直接回答「制度中未找到相关规定」，不要猜测。
3. 回答末尾用【政策片段 N】标注依据来源。
4. 涉及数字（天数、金额、期限）必须与原文完全一致，不得四舍五入或改写。"""


@st.cache_resource(show_spinner="正在建立索引…")
def get_retriever() -> PolicyRetriever:
    return PolicyRetriever()


def get_client():
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        return None
    from openai import OpenAI
    return OpenAI(api_key=key, base_url="https://api.deepseek.com")


def main() -> None:
    st.set_page_config(page_title="企业制度问答", page_icon="📘", layout="centered")
    st.title("企业制度问答助手")
    st.caption("基于 BM25 检索 + LLM 生成 · 答案只依据制度原文 · 无依据即拒答")

    retriever = get_retriever()

    with st.sidebar:
        st.subheader("检索设置")
        k = st.slider("召回片段数 (top-k)", 1, 10, 6,
                      help="k 越大召回越全，但噪声也越多。本项目实测 k=6 为最优。")
        show_ctx = st.checkbox("显示命中的政策片段", value=True)
        st.divider()
        st.metric("索引片段总数", getattr(retriever, "n_chunks", "—"))
        st.caption("纯标准库 BM25 · 离线可跑 · 不消耗模型调用")

    client = get_client()
    if client is None:
        st.warning("未找到 DEEPSEEK_API_KEY。请在项目目录放置 .env 文件，内容：\n\n"
                   "`DEEPSEEK_API_KEY=sk-xxxxxxxx`\n\n"
                   "（检索部分仍可离线验证：`python w6_rag_retriever.py`）")

    if "history" not in st.session_state:
        st.session_state.history = []

    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["q"])
        with st.chat_message("assistant"):
            st.write(turn["a"])
            if show_ctx and turn.get("ctx"):
                with st.expander(f"命中的 {len(turn['ctx'])} 个片段"):
                    for i, (text, score) in enumerate(turn["ctx"], 1):
                        st.markdown(f"**【政策片段 {i}】** · BM25 得分 `{score:.3f}`")
                        st.text(text[:400] + ("…" if len(text) > 400 else ""))

    question = st.chat_input("例如：试用期是多久？年假怎么请？加班怎么算？")
    if not question:
        return

    with st.chat_message("user"):
        st.write(question)

    hits = retriever.retrieve(question, k=k)
    ctx_parts = [f"【政策片段 {i}】\n{text}" for i, (text, _) in enumerate(hits, 1)]
    context = "\n\n".join(ctx_parts)[:6000]

    with st.chat_message("assistant"):
        if client is None:
            st.info("（无 API Key，仅展示检索结果）")
            answer = "—"
        else:
            placeholder = st.empty()
            buf = ""
            try:
                stream = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"【政策片段】\n{context}\n\n【问题】\n{question}"},
                    ],
                    temperature=0,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    buf += delta
                    placeholder.markdown(buf + "▌")
                placeholder.markdown(buf)
                answer = buf
            except Exception as exc:  # noqa: BLE001
                answer = f"调用失败：{exc}"
                st.error(answer)

        if show_ctx:
            with st.expander(f"命中的 {len(hits)} 个片段"):
                for i, (text, score) in enumerate(hits, 1):
                    st.markdown(f"**【政策片段 {i}】** · BM25 得分 `{score:.3f}`")
                    st.text(text[:400] + ("…" if len(text) > 400 else ""))

    st.session_state.history.append({"q": question, "a": answer, "ctx": hits})


if __name__ == "__main__":
    main()
