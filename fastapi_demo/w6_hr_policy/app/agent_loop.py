"""Bounded selection loop. Only validated evidence becomes a policy answer."""
import json

from .config import MAX_STEPS, TOOL_TIMEOUT
from .dispatch import run_call
from .llm import SYSTEM
from .tools import IMPL, TOOLS

MESSAGES = {
    "refused": "当前合成制度没有足够依据回答该问题。",
    "clarify": "请确认适用对象、制度版本或要继续询问的事项。",
    "degraded": "本轮未得到可确认的回答。下方引用为候选原文，请核对是否适用。",
    "error": "问答处理失败，请稍后重试。",
}

def answer(status, evidence=(), reason=None):
    text = MESSAGES.get(status, "")
    if status == "answered":
        clause, = evidence
        text = f"【合成制度】{clause['text']} [{clause['id']}]"
    return {"status": status, "answer": text, "citations": list(evidence), "reason": reason}

async def run_agent(question, history, evidence, category, llm, emit):
    by_id = {item["id"]: item for item in evidence}
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps({
            "question": question, "history": history, "evidence": evidence
        }, ensure_ascii=False)},
    ]
    seen = set()
    for step in range(1, MAX_STEPS + 1):
        emit("generating", step=step)
        try:
            choice, calls = await llm.chat(messages, TOOLS, emit)
        except Exception as exc:
            return answer("degraded", evidence, type(exc).__name__)
        if not calls:
            if choice in ("REFUSE", "CLARIFY"):
                return answer("refused" if choice == "REFUSE" else "clarify", reason="model_selection")
            token = choice.removeprefix("ANSWER ")
            if token in by_id:
                return answer("answered", [by_id[token]])
            return answer("degraded", evidence, "invalid_selector")
        if step == MAX_STEPS:
            return answer("degraded", evidence, "max_steps")
        ids = [call.id for call in calls]
        if any(not cid for cid in ids) or len(ids) != len(set(ids)):
            return answer("degraded", evidence, "invalid_tool_call_ids")
        messages.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.raw}}
            for c in calls
        ]})
        for call in calls:
            key = (call.name, json.dumps(call.args, sort_keys=True))
            name = call.name if call.name in IMPL else "<unknown>"
            if key in seen:
                emit("tool_finished", tool=name, status="blocked", error="repeated_action")
                return answer("degraded", evidence, "repeated_action")
            seen.add(key)
            emit("tool_running", tool=name, argument_keys=sorted(
                {"query", "category"}.intersection(call.args) if isinstance(call.args, dict) else []
            ))
            result = await run_call(call, category, by_id, TOOL_TIMEOUT)
            emit("tool_finished", tool=name, **{key: value for key, value in result.items() if key != "items"})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False)})
    return answer("degraded", evidence, "max_steps")
