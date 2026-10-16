"""Scripted LiteLLM stand-in. Only the network call is replaced: the real ADK
Runner -> LlmAgent -> LiteLlm path runs in tests, without credentials."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
import litellm
from google.adk.models.lite_llm import LiteLLMClient


@dataclass
class RecordedCall:
    model: str
    payload: dict[str, Any]  # the JSON user message sent to the agent
    kwargs: dict[str, Any]
    tools: Any


# A responder receives the recorded call and returns a report dict, raw text, or an exception.
Responder = Callable[[RecordedCall], "dict[str, Any] | str | Exception"]


class FakeLiteLLMClient(LiteLLMClient):
    def __init__(self, responder: Responder, *, prompt_tokens: int = 100, completion_tokens: int = 20):
        self.responder = responder
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.calls: list[RecordedCall] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def acompletion(self, model, messages, tools, **kwargs):  # noqa: ANN001 - LiteLLMClient signature
        user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        call = RecordedCall(model, json.loads(user), kwargs, tools)
        self.calls.append(call)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0.01)  # lets concurrent agents overlap
            result = self.responder(call)
        finally:
            self.in_flight -= 1
        if isinstance(result, Exception):
            raise result
        content = result if isinstance(result, str) else json.dumps(result)
        return litellm.ModelResponse(
            model=model,
            choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            usage={
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
            },
        )


def verdict(
    item_id: str,
    classification: str,
    *,
    confidence: float = 0.9,
    evidence: list[tuple[str, int, str]] | None = None,
    reason: str = "LLM reasoning",
    suggested_action: str = "",
) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "classification": classification,
        "confidence": confidence,
        "reason": reason,
        "evidence": [{"file": f, "line": ln, "snippet": s} for f, ln, s in evidence or []],
        "suggested_action": suggested_action,
    }


def for_all_items(classification: str, **kwargs: Any) -> Responder:
    """Responder answering every item in the request with the same verdict."""
    return lambda call: {
        "verdicts": [verdict(item["item_id"], classification, **kwargs) for item in call.payload["items"]]
    }


def rate_limit_error(retry_after: float | None = None) -> litellm.RateLimitError:
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    return litellm.RateLimitError(
        message="Rate limit reached",
        llm_provider="groq",
        model="openai/gpt-oss-20b",
        response=httpx.Response(429, headers=headers, request=httpx.Request("POST", "https://api.groq.com")),
    )


def auth_error() -> litellm.AuthenticationError:
    return litellm.AuthenticationError(message="Invalid API Key", llm_provider="groq", model="openai/gpt-oss-20b")
