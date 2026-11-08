"""Enforce callable scope, JSON Schema and cooperative tool timeout."""
import asyncio
import time

from jsonschema import Draft202012Validator
from .tools import TOOLS, IMPL

SCHEMAS = {item["function"]["name"]: Draft202012Validator(item["function"]["parameters"])
           for item in TOOLS}

async def run_call(call, category, allowed_ids, timeout):
    start = time.monotonic()
    if call.name not in IMPL or call.name not in SCHEMAS:
        return {"status": "blocked", "error": "unknown_tool", "latency_ms": 0}
    if not SCHEMAS[call.name].is_valid(call.args):
        return {"status": "blocked", "error": "invalid_arguments", "latency_ms": 0}
    args = dict(call.args)
    requested_category = args.get("category", "其他")
    if category != "其他" and requested_category not in ("其他", category):
        return {"status": "blocked", "error": "category_outside_request", "latency_ms": 0}
    if category != "其他":
        args["category"] = category
    try:
        async with asyncio.timeout(timeout):
            items = await IMPL[call.name](**args)
        result = {"status": "ok", "items": [item for item in items if item["id"] in allowed_ids]}
    except Exception as exc:
        result = {"status": "error", "error": type(exc).__name__}
    result["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
    return result
