"""ClaudeExecutor — smart, budget-aware Claude API wrapper.

ALL LLM calls in BrainOS agents go through this executor.
The model is never hardcoded at the call site — it is selected upstream by
DAAO (difficulty-aware routing) and passed in explicitly.

Features:
- Explicit model override: every call accepts the DAAO-selected model
- Budget-aware fallback: switches to fast model at >80% token budget
- Retry on transient errors (rate limit, 5xx)
- JSON extraction from raw response text
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
    """Smart Claude caller: model is set by DAAO upstream, never hardcoded here.

    Usage::

        daao = DAAO(fast_model=FAST_MODEL, main_model=MAIN_MODEL)
        executor = ClaudeExecutor(api_key=..., default_model=MAIN_MODEL, fast_model=FAST_MODEL)

        # DAAO selects model — executor just executes
        model = daao.route(task_text)
        answer = await executor.call(system="...", user="...", model=model)
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
        model: str | None = None,
    ) -> str:
        """Call Claude and return response text.

        Model selection priority (highest wins):
        1. force_fast=True → always use fast model
        2. budget.tight    → switch to fast model (budget pressure)
        3. model param     → explicit model from DAAO
        4. self.default_model → fallback default

        Args:
            system:     System prompt
            user:       User message
            max_tokens: Max tokens in response
            budget:     If provided, switches to fast model when budget is tight
            force_fast: Always use the fast (Haiku) model regardless of model param
            model:      Explicit model ID from DAAO — the preferred way to set model
        """
        # Resolve model: explicit > default, then override if fast conditions met
        resolved = model or self.default_model
        if force_fast or (budget and budget.tight):
            resolved = self.fast_model

        if budget:
            budget.consume(system + user)

        for attempt in range(3):
            try:
                resp = await self._client.messages.create(
                    model=resolved,
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
        model: str | None = None,
    ) -> dict:
        """Call Claude and parse JSON from the response.

        Args:
            model: Explicit model ID from DAAO (same as call())
        """
        text = await self.call(system, user, max_tokens=max_tokens, budget=budget, model=model)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        m = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise ValueError(f"No JSON found in Claude response: {text[:200]}")
