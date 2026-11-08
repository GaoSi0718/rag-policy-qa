import json
import unittest

from httpx import AsyncByteStream, AsyncClient, MockTransport, Response
from openai import AsyncOpenAI
from w6_hr_policy.app.llm import Provider

def line(delta=None, finish=None):
    return ("data: " + json.dumps({
        "id": "stub", "object": "chat.completion.chunk", "created": 1, "model": "fixture",
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]
    }) + "\n\n").encode()

class Chunks(AsyncByteStream):
    def __init__(self, parts):
        self.parts = parts
        self.closed = False

    async def __aiter__(self):
        for part in self.parts:
            yield part

    async def aclose(self):
        self.closed = True

class ProviderTests(unittest.IsolatedAsyncioTestCase):
    async def check_stream(self, responses, expected_attempts):
        recorded = []
        requests = []
        def handler(request):
            requests.append(json.loads(request.content))
            return responses.pop(0)
        http = AsyncClient(transport=MockTransport(handler))
        provider = Provider.__new__(Provider)
        provider.client = AsyncOpenAI(api_key="fixture", base_url="https://mock.invalid", max_retries=0, http_client=http)
        try:
            result = await provider.chat([{"role": "user", "content": "fixture"}], [],
                                         lambda stage, **fields: recorded.append((stage, fields)))
        finally:
            await provider.close()
        self.assertTrue(all(body["stream"] for body in requests))
        self.assertEqual(len(requests), expected_attempts)
        return result, recorded

    async def test_retry_and_partial_sdk_response(self):
        stream = Chunks([line({"content": "ANSWER "}), line({"content": "HR-V2-02"}, "stop"), b"data: [DONE]\n\n"])
        result, events = await self.check_stream([Response(429, json={"error": {"message": "test"}}),
                                                  Response(200, stream=stream)], 2)
        self.assertEqual(result, ("ANSWER HR-V2-02", []))
        self.assertTrue(stream.closed)
        self.assertEqual([stage for stage, _ in events],
                         ["model_attempt", "model_error", "model_attempt", "model_delta", "model_delta"])

    async def test_assemble_tool_arguments(self):
        first = {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                 "function": {"name": "search_policy", "arguments": "{\"query\": "}}]}
        second = {"tool_calls": [{"index": 0, "function": {"arguments": "\"试用期\"}"}}]}
        stream = Chunks([line(first), line(second), line(finish="tool_calls"), b"data: [DONE]\n\n"])
        result, _ = await self.check_stream([Response(200, stream=stream)], 1)
        self.assertEqual(result[1][0].args, {"query": "试用期"})
        self.assertEqual(result[1][0].name, "search_policy")
