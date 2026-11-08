"""Raw turns stay in SQLite; only bounded context is injected into a model."""
import json

def model_history(turns, budget=1800):
    # 早期问题保留完整原句，避免摘要改写数字或否定条件。
    tail = []
    split = len(turns)
    for question, answer in reversed(turns[-2:]):
        item = {"question": question, "previous_answer": answer}
        if len(json.dumps([item, *tail], ensure_ascii=False)) > budget // 2:
            break
        tail.insert(0, item)
        split -= 1
    earlier = []
    for question, _ in turns[:split]:
        if question not in earlier and len(json.dumps([*earlier, question], ensure_ascii=False)) <= budget // 2 - 64:
            earlier.append(question)
    return {"earlier_questions": earlier, "recent_turns": tail}
