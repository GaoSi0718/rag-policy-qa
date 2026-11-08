"""Evaluate this ASGI application with synthetic data and a fresh SQLite database."""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from httpx import ASGITransport, AsyncClient
from w6_hr_policy.app import main, storage
from w6_hr_policy.app.config import CORPUS_PATH, MODEL, ROOT
from w6_hr_policy.app.rate_limit import limiter

def parse_events(text):
    events = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        event = next((line[7:] for line in block.splitlines() if line.startswith("event: ")), None)
        data = next((line[6:] for line in block.splitlines() if line.startswith("data: ")), None)
        if event and data:
            events.append({"event": event, **json.loads(data)})
    return events

class TopHitFixture:
    async def chat(self, messages, tools, emit):
        payload = json.loads(messages[1]["content"])
        emit("model_delta", chars=6)
        return "ANSWER " + payload["evidence"][0]["id"], []

    async def close(self):
        pass

class Recorder:
    def __init__(self, provider, calls):
        self.provider = provider
        self.calls = calls

    async def chat(self, messages, tools, emit):
        request = json.loads(messages[1]["content"])
        self.calls.append(request)
        choice, calls = await self.provider.chat(messages, tools, emit)
        request["selector_output"] = choice
        request["proposed_tools"] = [call.name for call in calls]
        return choice, calls

    async def close(self):
        await self.provider.close()

def evaluate(case, events, inputs):
    final = next((item for item in events if item["event"] in {"message", "error"}), {})
    retrieved = next((item for item in events if item.get("stage") == "retrieved"), {})
    status = final.get("status", "empty")
    citations = final.get("citations", [])
    citations_ids = [item["id"] for item in citations]
    expected_sources = [case["target_only"]] if "target_only" in case else case["sources"]
    hit = set(expected_sources) <= set(retrieved.get("matched_ids", []))
    evidence = inputs[0]["evidence"] if inputs else []
    source_correct = citations_ids == expected_sources and all(item in evidence for item in citations)
    content_correct = source_correct and bool(citations) and all(
        item["text"] in final.get("answer", "") for item in citations
    )
    expected = case["status"]
    passed = expected == status and (content_correct if expected == "answered" else not citations)
    return {"status": status, "reason": final.get("reason"), "support_hit": hit if expected == "answered" else None,
            "correct_with_citation": expected == "answered" and status == "answered" and content_correct,
            "passed": passed}

async def run(args):
    source = ROOT / "eval" / args.dataset
    cases = json.loads(source.read_text(encoding="utf-8"))["questions"]
    if args.limit:
        cases = cases[:args.limit]
    provider_factory = main.make_llm if args.mode == "real" else TopHitFixture
    if args.mode == "real" and not os.getenv("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY is required for --mode real")
    os.environ["W6_API_KEY"] = "isolated-evaluation"
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        storage.DB = Path(directory) / "eval.sqlite3"
        storage.initialize()
        limiter.reset()
        async with AsyncClient(transport=ASGITransport(app=main.app), base_url="http://local-eval",
                               headers={"X-API-Key": "isolated-evaluation"}) as client:
            for case in cases:
                inputs = []
                main.make_llm = lambda: Recorder(provider_factory(), inputs)
                params = {"q": case["question"], "session_id": case.get("session", f"eval-{case['id']}")}
                if "category" in case:
                    params["category"] = case["category"]
                setup = None
                if "history" in case:
                    response = await client.get("/chat/stream", params={**params, "q": case["history"]})
                    setup = {"events": parse_events(response.text), "model_inputs": list(inputs)}
                    inputs.clear()
                started = time.monotonic()
                response = await client.get("/chat/stream", params=params)
                events = parse_events(response.text)
                rows.append({"case": case, "http_status": response.status_code, "setup": setup,
                             "events": events, "model_inputs": inputs,
                             "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
                             "score": evaluate(case, events, inputs)})
                if args.mode == "real" and rows[-1]["score"]["status"] in {"degraded", "error", "empty"}:
                    # Record the first operational failure and stop real spending on the batch.
                    break
    answerable = [row for row in rows if row["case"]["status"] == "answered"]
    refused = [row for row in rows if row["case"]["status"] == "refused"]
    usage = [event for row in rows for event in row["events"] + (row["setup"] or {}).get("events", [])
             if event.get("stage") == "usage"]
    report = {
        "mode": args.mode, "model": MODEL if args.mode == "real" else "top-hit fixture",
        "dataset": args.dataset, "corpus_sha256": hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "summary": {
            "planned": len(cases), "executed": len(rows), "unexecuted": len(cases) - len(rows),
            "pass": sum(row["score"]["passed"] for row in rows),
            "answerable_executed": len(answerable), "refusal_expected_executed": len(refused),
            "support_hit": sum(row["score"]["support_hit"] for row in answerable),
            "correct_with_citation": sum(row["score"]["correct_with_citation"] for row in answerable),
            "expected_refusal_actual_refused": sum(row["score"]["status"] == "refused" for row in refused),
            "expected_refusal_actual_answer": sum(row["score"]["status"] == "answered" for row in refused),
            "expected_answer_actual_refused": sum(row["score"]["status"] == "refused" for row in answerable),
            "terminal_status_counts": Counter(row["score"]["status"] for row in rows),
            "usage_reports": len(usage),
            "prompt_tokens_reported": sum(item["prompt_tokens"] for item in usage),
            "completion_tokens_reported": sum(item["completion_tokens"] for item in usage),
            "cost": "zero external calls" if args.mode == "fixture" else "see provider usage/billing",
        },
        "rows": rows,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=True))
    return 0 if report["summary"]["pass"] == len(cases) else 1

def main_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fixture", "real"), default="fixture")
    parser.add_argument("--dataset", choices=("questions-v2.json", "diagnostics-v2.json", "session-12-v2.json"),
                        default="questions-v2.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--replay", type=Path, help="score a saved report without requesting a model")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.replay:
        saved = json.loads(args.replay.read_text(encoding="utf-8"))
        checks = []
        for row in saved["rows"]:
            score = evaluate(row["case"], row["events"], row["model_inputs"])
            checks.append({"id": row["case"]["id"], "score": score, "same_as_saved": score == row["score"]})
        report = {"mode": "saved response replay", "model_requests": 0, "input_file": args.replay.name,
                  "input_sha256": hashlib.sha256(args.replay.read_bytes()).hexdigest(),
                  "matched": sum(row["same_as_saved"] for row in checks), "rows": checks}
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"saved verdicts matched: {report['matched']}/{len(checks)}")
        raise SystemExit(0 if checks and report["matched"] == len(checks) else 1)
    raise SystemExit(asyncio.run(run(args)))

if __name__ == "__main__":
    main_cli()
