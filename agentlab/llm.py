"""LLM client interface with a scriptable mock (offline) and an Anthropic backend."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

DEFAULT_MODEL = "claude-sonnet-5-5"

# USD per million tokens (input, output). Placeholder values: edit to match your current pricing.
PRICING: dict[str, tuple[float, float]] = {
    "default": (3.0, 15.0),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    inp, out = PRICING.get(model, PRICING["default"])
    return (input_tokens * inp + output_tokens * out) / 1_000_000


@dataclass
class ToolUse:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    text: str = ""
    tool_uses: list[ToolUse] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0

    def to_blocks(self) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        if self.text:
            blocks.append({"type": "text", "text": self.text})
        for tu in self.tool_uses:
            blocks.append({"type": "tool_use", "id": tu.id, "name": tu.name, "input": tu.input})
        return blocks


class LLMClient(Protocol):
    model: str

    def complete(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse: ...


class MockLLM:
    """Deterministic client for tests and offline demos.

    `script` is either a list of LLMResponse (returned in order, last one repeats)
    or a callable (system, messages, tools) -> LLMResponse.
    """

    model = "mock"

    def __init__(self, script: list[LLMResponse] | Callable[..., LLMResponse]):
        self.script = script
        self.calls = 0

    def complete(self, system, messages, tools=None) -> LLMResponse:
        self.calls += 1
        if callable(self.script):
            return self.script(system, messages, tools)
        idx = min(self.calls - 1, len(self.script) - 1)
        return self.script[idx]


class AnthropicLLM:
    """Thin wrapper over the Anthropic Messages API. Requires `pip install anthropic` and ANTHROPIC_API_KEY."""

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 4096):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Install the SDK first: pip install 'agentlab[anthropic]'") from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()

    def complete(self, system, messages, tools=None) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=self.model, max_tokens=self.max_tokens, system=system, messages=messages
        )
        if tools:
            kwargs["tools"] = tools
        resp = self._client.messages.create(**kwargs)
        text_parts: list[str] = []
        tool_uses: list[ToolUse] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_uses.append(ToolUse(id=block.id, name=block.name, input=dict(block.input)))
        return LLMResponse(
            text="".join(text_parts),
            tool_uses=tool_uses,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )


def _demo_script(system, messages, tools) -> LLMResponse:
    """Offline script for `--model mock`: solves the bundled bugfix and feature tasks, leaves the rest failing."""
    first = messages[0]["content"]
    first = first if isinstance(first, str) else str(first)
    done = sum(1 for m in messages if m["role"] == "assistant")
    writes: list[tuple[str, str]] = []
    if "last_n" in first:
        writes = [("utils.py", "def last_n(items, n):\n    return items[-n:] if n > 0 else []\n")]
    elif "is_prime" in first:
        writes = [("mathx.py", "def is_prime(n):\n    return n > 1 and all(n % i for i in range(2, int(n**.5) + 1))\n")]
    if done < len(writes):
        path, content = writes[done]
        return LLMResponse(
            tool_uses=[ToolUse(id=f"mock-{done}", name="write_file", input={"path": path, "content": content})],
            input_tokens=100,
            output_tokens=20,
        )
    return LLMResponse(text="done", input_tokens=100, output_tokens=10)


def make_client(name: str) -> LLMClient:
    """`mock` is an offline demo client; anything else is an Anthropic model id."""
    if name == "mock":
        return MockLLM(_demo_script)
    return AnthropicLLM(model=name)
