from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "evaluation" / "harness"
sys.path.insert(0, str(HARNESS))

from async_client import AsyncChatClient  # noqa: E402


class AsyncClientTests(unittest.TestCase):
    def test_completion_budget_is_forwarded_unchanged(self):
        async def run():
            client = AsyncChatClient("http://127.0.0.1:30000/v1", "test-model")
            calls: list[tuple[str, dict]] = []

            async def post(path: str, payload: dict) -> tuple[dict, float]:
                calls.append((path, payload))
                return (
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": "answer",
                                    "reasoning_content": "reasoning",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 200000,
                            "completion_tokens": 1,
                        },
                    },
                    0.01,
                )

            client._post = post
            try:
                result = await client.chat_raw(
                    [{"role": "user", "content": "large prompt"}],
                    max_completion_tokens=65536,
                    temperature=1.0,
                    top_p=0.95,
                    seed=7,
                    request_id="fixed-budget",
                )
            finally:
                await client.aclose()

            self.assertEqual(len(calls), 1)
            path, payload = calls[0]
            self.assertEqual(path, "/chat/completions")
            self.assertEqual(payload["max_completion_tokens"], 65536)
            self.assertEqual(result["requested_max_completion_tokens"], 65536)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
