import asyncio
import json
import os
from pathlib import Path
import socket
import tempfile
import time
import unittest
from unittest.mock import patch

from httpx import AsyncClient
import uvicorn

from w6_hr_policy.app import main, storage

class DelayedProvider:
    def __init__(self):
        self.done = False
        self.closed = asyncio.Event()

    async def chat(self, messages, tools, emit):
        await asyncio.sleep(0.08)
        emit("model_delta", chars=5)
        await asyncio.sleep(0.3)
        self.done = True
        return "ANSWER HR-V2-02", []

    async def close(self):
        self.closed.set()

class NetworkTest(unittest.IsolatedAsyncioTestCase):
    async def test_uvicorn_sse_order_and_disconnected_turn(self):
        with tempfile.TemporaryDirectory() as folder, socket.socket() as sock, patch.object(
            storage, "DB", Path(folder) / "network.sqlite3"
        ), patch.dict(os.environ, {"W6_API_KEY": "local-test"}):
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            server = uvicorn.Server(uvicorn.Config(main.app, log_level="warning", access_log=False))
            running = asyncio.create_task(server.serve(sockets=[sock]))
            notes = {}
            try:
                async with asyncio.timeout(3):
                    while not server.started:
                        await asyncio.sleep(0.01)
                async with AsyncClient(base_url=f"http://127.0.0.1:{port}",
                                       timeout=2, trust_env=False, headers={"X-API-Key":"local-test"}) as client:
                    for disconnect in (False, True):
                        model = DelayedProvider()
                        start = time.monotonic()
                        params = {"q": "试用期一般多久？", "session_id": "abort" if disconnect else "ok"}
                        entries = []
                        with patch.object(main, "make_llm", return_value=model):
                            async with client.stream("GET", "/chat/stream", params=params) as response:
                                self.assertEqual(response.status_code, 200)
                                async for line in response.aiter_lines():
                                    if not line.startswith("data: "):
                                        continue
                                    item = json.loads(line[6:])
                                    entries.append(item)
                                    if item.get("stage") == "model_delta":
                                        self.assertFalse(model.done)
                                        elapsed = round((time.monotonic() - start) * 1000, 1)
                                        notes["first_model_progress_abort_ms" if disconnect else "first_model_progress_normal_ms"] = elapsed
                                        if disconnect:
                                            break
                            await asyncio.wait_for(model.closed.wait(), timeout=1.5)
                        if not disconnect:
                            self.assertTrue(model.done)
                            self.assertEqual(entries[-1]["status"], "answered")
                            self.assertTrue(entries[-1]["trace_persisted"])
                            self.assertEqual(len(storage.history("ok")), 1)
                            notes["done_ms"] = round((time.monotonic() - start) * 1000, 1)
                        else:
                            self.assertFalse(model.done)
                            self.assertEqual(storage.history("abort"), [])
                            for _ in range(50):
                                traces = storage.read_trace(entries[0]["run_id"])
                                if traces:
                                    break
                                await asyncio.sleep(0.01)
                            self.assertEqual(traces[-1]["status"], "client_disconnect")
                            notes["disconnect_terminal"] = traces[-1]["status"]
            finally:
                server.should_exit = True
                await asyncio.wait_for(running, timeout=3)
            if os.getenv("W6_TEST_EVIDENCE"):
                path = Path(os.environ["W6_TEST_EVIDENCE"]) / "fixture-loopback.json"
                path.write_text(json.dumps(notes, indent=2), encoding="utf-8")
