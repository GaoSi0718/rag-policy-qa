"""FastAPI entry point for a single-user local reproduction."""
import asyncio
from contextlib import asynccontextmanager, suppress
import hashlib
import json
import logging
import os
import sqlite3
import time
from typing import Literal
from uuid import uuid4

from anyio import CancelScope
from fastapi import Depends, FastAPI, Query
from sse_starlette.sse import EventSourceResponse

from . import storage
from .agent_loop import answer, run_agent
from .auth import require_api_key
from .config import TOTAL_TIMEOUT
from .context import model_history
from .llm import Provider
from .pii_guard import mask_identifiers
from .rag import FIRST_TOPIC, corpus, match_topics, pack, resolve, retrieve
from .rate_limit import rate_limit

logger = logging.getLogger(__name__)
Category = Literal["假勤", "入离职", "差旅", "薪酬福利", "其他"]

@asynccontextmanager
async def lifespan(app):
    corpus()
    storage.initialize()
    yield

app = FastAPI(title="青岚合成制度问答", lifespan=lifespan)
active_sessions = set()

def make_llm():
    return Provider() if os.getenv("DEEPSEEK_API_KEY") else None

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/chat/stream", dependencies=[Depends(rate_limit), Depends(require_api_key)])
async def chat_stream(
    q: str = Query(min_length=1, max_length=500),
    session_id: str = Query(default="default", pattern=r"^[A-Za-z0-9_-]{1,64}$"),
    category: Category = "其他",
    version: str | None = Query(default=None, max_length=40),
):
    q = mask_identifiers(q)
    run_id = uuid4().hex
    session_ref = hashlib.sha256(session_id.encode()).hexdigest()[:16]
    trace = []
    queue = asyncio.Queue()
    start = time.monotonic()
    evidence = []
    result = None
    persisted = False
    terminal = "client_disconnect"

    def emit(stage, **fields):
        entry = {"stage": stage, "ms": round((time.monotonic() - start) * 1000, 1),
                 "session_ref": session_ref, **fields}
        # 增量进度可以高频到达；日志保留首片与最终统计，其余原始增量不落盘。
        if stage != "model_delta" or not any(t["stage"] == stage for t in trace):
            trace.append(entry)
        queue.put_nowait(event("status", entry))

    def event(name, data):
        return {"event": name, "data": json.dumps({"run_id": run_id, **data}, ensure_ascii=False)}

    async def process():
        nonlocal evidence
        llm = None
        try:
            async with asyncio.timeout(TOTAL_TIMEOUT):
                turns = storage.history(session_id)
                ordered = turns if FIRST_TOPIC.search(q) else reversed(turns)
                previous = next((question for question, prior in ordered
                                 if prior.startswith("【合成制度】") and match_topics(question)), "")
                state, query = resolve(q.strip(), previous, version)
                if state != "ready":
                    return answer(state, reason="query_scope")
                hits = await retrieve(query, category=category)
                evidence = pack(hits)
                emit("retrieved", matched_ids=[item["id"] for item in hits],
                     injected_ids=[item["id"] for item in evidence])
                if not evidence:
                    return answer("refused", reason="no_evidence")
                llm = make_llm()
                if llm is None:
                    return answer("degraded", evidence, "model_unconfigured")
                return await run_agent(query, model_history(turns), evidence, category, llm, emit)
        except TimeoutError:
            return answer("degraded", evidence, "total_timeout")
        except Exception as exc:
            return answer("error", reason=type(exc).__name__)
        finally:
            if llm is not None:
                try:
                    await llm.close()
                except Exception as exc:
                    emit("model_close_error", error=type(exc).__name__)

    def save_terminal():
        trace.append({"stage": "terminal", "session_ref": session_ref, "status": terminal,
                      "reason": result["reason"] if result else terminal,
                      "ms": round((time.monotonic() - start) * 1000, 1)})
        try:
            storage.complete(run_id, session_id, q, result, trace)
            return True
        except (OSError, sqlite3.Error) as exc:
            logger.error("run=%s trace_persist_failed error=%s", run_id, type(exc).__name__)
            return False

    async def generate():
        nonlocal result, terminal, persisted
        if session_id in active_sessions:
            result = answer("error", reason="session_busy")
            terminal = "error"
            persisted = save_terminal()
            yield event("error", result)
            yield event("done", {"status": terminal, "trace_persisted": persisted})
            return
        active_sessions.add(session_id)
        task = None
        try:
            emit("accepted")
            task = asyncio.create_task(process())
            task.add_done_callback(lambda _: queue.put_nowait(None))
            while (item := await queue.get()) is not None:
                yield item
            result = task.result()
            terminal = result["status"]
            persisted = save_terminal()
            yield event("error" if terminal == "error" else "message", result)
            yield event("done", {"status": terminal, "trace_persisted": persisted})
        finally:
            try:
                if task is not None:
                    task.cancel()
                    with CancelScope(shield=True), suppress(asyncio.CancelledError):
                        await task
            finally:
                if not persisted and result is None:
                    save_terminal()
                active_sessions.discard(session_id)

    return EventSourceResponse(generate(), ping=10, send_timeout=5)

@app.get("/runs/{run_id}", dependencies=[Depends(require_api_key)])
async def trace(run_id: str):
    return {"run_id": run_id, "events": storage.read_trace(run_id)}
