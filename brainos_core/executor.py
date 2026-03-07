"""ClaudeExecutor — thin, budget-aware wrapper around the Anthropic API.

Innovations extracted from agent-purple + agent-finance:
- Automatic model switching: Sonnet by default, Haiku at >80% token budget
- Structured output parsing (JSON extraction from raw text)
- Retry logic for transient errors
- Token counting without a full API round-trip
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field

import anthropic


@dataclass
class TokenBudget:
    """Track token consumption and switch model when budget is tight."""
    max_tokens: int = 180_000
    used: int = 0
    haiku_threshold: float = 0.80

    def consume(self, text: str) -> None:
        self.used += len(text) // 4  # ~4 chars per token estimate

    @property
    def exhausted(self) -> bool:
        return self.used >= self.max_tokens

    @property
    def tight(self) -> bool:
        return self.used / self.max_tokens >= self.haiku_threshold


class ClaudeExecutor:
    """Stateless Claude caller with model switching + structured output.

    Usage:
        executor = ClaudeExecutor(api_key="sk-ant-...")
        answer = await executor.call(system="...", user="...", max_tokens=256)
    """

    DEFAULT_MODEL = "claude-sonnet-4-6"
    FAST_MODEL = "claude-haiku-4-5-20251001"

    def __init__(
        self,
        api_key: str,
        default_model: str | None = None,
        fast_model: str | None = None,
    ):
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self.default_model = default_model or self.DEFAULT_MODEL
        self.fast_model = fast_model or self.FAST_MODEL

    async def call(
        self,
        system: str,
        user: str,
        max_tokens: int = 512,
        budget: TokenBudget | None = None,
        force_fast: bool = False,
    ) -> str:
        """Call Claude, return response text.

        Args:
            system: System prompt
            user: User message
            max_tokens: Max tokens in response
            budget: If provided, switches to fast model when budget is tight
            force_fast: Always use the fast (Haiku) model
        """
        model = self.default_model
        if force_fast or (budget and budget.tight):
            model = self.fast_model

        if budget:
            budget.consume(system + user)

        for attempt in range(3):
            try:
                resp = await self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                text = resp.content[0].text
                if budget:
                    budget.consume(text)
                return text
            except anthropic.RateLimitError:
                await asyncio.sleep(2 ** attempt)
            except anthropic.APIStatusError as exc:
                if exc.status_code >= 500:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise
        raise RuntimeError("Claude API: max retries exceeded")

    async def call_json(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        budget: TokenBudget | None = None,
    ) -> dict:
        """Call Claude and parse JSON from the response."""
        text = await self.call(system, user, max_tokens=max_tokens, budget=budget)
        # Try raw parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Extract first {...} or [...] block
        m = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise ValueError(f"No JSON found in Claude response: {text[:200]}")
