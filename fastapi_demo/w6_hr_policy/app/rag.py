"""Versioned clauses indexed by the existing BM25 scorer."""
import json
import re
from functools import lru_cache

from w6_rag_retriever import PolicyRetriever
from .config import CORPUS_PATH

CATEGORIES = ("假勤", "入离职", "差旅", "薪酬福利", "其他")
FIRST_TOPIC = re.compile(r"第一轮|最早那个问题|最开始那个问题")
FOLLOW_UP = re.compile(r"^(那|那么|这个|这种情况|它|还有呢)|第一轮|最早那个问题|最开始那个问题")
TOPIC_PREFIX = re.compile(r"^(?:第一轮(?:那个问题)?|最早那个问题|最开始那个问题)[，,：:\s]*")
OUTSIDE_SCOPE = re.compile(r"外包|劳务派遣|实习生|其他公司|另一家|旧版|历史版本|上个版本|新旧|去年版本|外地分公司")
INJECTION = re.compile(r"忽略.{0,8}(制度|规则|指令)|编造|系统提示词|ignore.{0,8}instructions", re.I)

@lru_cache(maxsize=1)
def corpus():
    data = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    clauses = data["clauses"]
    if len({item["id"] for item in clauses}) != len(clauses):
        raise ValueError("duplicate clause IDs")
    index = PolicyRetriever.from_chunks([
        " ".join([item["title"], *item["aliases"][1:], item["text"]]) for item in clauses
    ])
    return data["version"], clauses, index

def match_topics(query):
    _, clauses, _ = corpus()
    matches = []
    for item in clauses:
        for alias in item["aliases"]:
            for match in re.finditer(re.escape(alias), query):
                matches.append((match.start(), match.end(), item["id"]))
    return {topic for start, end, topic in matches if not any(
        left <= start and end <= right and (left, right) != (start, end)
        for left, right, _ in matches
    )}

def resolve(question, previous, version):
    current_version, _, _ = corpus()
    if INJECTION.search(question):
        return "refused", question
    follow_up = bool(FOLLOW_UP.search(question))
    prior = previous if follow_up else ""
    if version not in (None, current_version) or OUTSIDE_SCOPE.search(question) or OUTSIDE_SCOPE.search(prior):
        return "clarify", question
    if prior and not match_topics(question):
        topics = match_topics(prior)
        if len(topics) != 1:
            return "clarify", question
        _, clauses, _ = corpus()
        topic = next(item["title"] for item in clauses if item["id"] in topics)
        current = TOPIC_PREFIX.sub("", question)
        return "ready", f"已确认主题：{topic}。本轮问题：{current}"
    if follow_up and not match_topics(question):
        return "clarify", question
    return "ready", question

async def retrieve(query, top_k=3, category="其他"):
    _, clauses, index = corpus()
    topic_ids = match_topics(query)
    hits = []
    for position, score in index.rank(query):
        clause = clauses[position]
        if clause["id"] in topic_ids and category in ("其他", clause["category"]):
            hits.append({**clause, "score": round(score, 4)})
            if len(hits) == top_k:
                break
    return hits

def pack(hits, budget=1100):
    selected = []
    for hit in hits:
        item = {key: hit[key] for key in ("id", "title", "doc_id", "version", "scope", "text")}
        if len(json.dumps([*selected, item], ensure_ascii=False)) <= budget:
            selected.append(item)
    return selected
