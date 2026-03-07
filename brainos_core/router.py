"""router.py — Smart outer model: pure strategy selection + reward attribution.

The Router is a ZERO-LLM orchestration layer. It:
  1. Queries Brain layers to select the optimal strategy
  2. Applies hard rules (privacy → always llm_direct)
  3. Computes proxy reward from answer signals (no LLM judge needed)
  4. Records the outcome back to all Brain layers

No LLM calls. No I/O beyond the Brain's file layer.
This is the "meta-agent" that sits above all task handlers.

Usage:
    brain = Brain()
    router = Router(brain)

    # Select strategy
    strategy = router.select(category, force_strategy="llm_direct")

    # After handler returns:
    reward = router.reward(answer, strategy)
    router.record(task_text, category, strategy, model, reward, session_id)
"""
from __future__ import annotations

import re
import time
from typing import Any

from .brain import Brain, STRATEGIES


# ── Proxy reward signal (no LLM) ─────────────────────────────────────────────

def compute_reward(
    answer: str | None,
    strategy: str,
    *,
    expected_type: str = "text",  # "text" | "number" | "id" | "bool"
) -> float:
    """Compute proxy reward 0.0–1.0 from answer quality signals.

    Rules (deterministic, no LLM):
    - Empty / None answer → 0.0
    - Error / exception text → 0.1
    - Privacy refusal ("I cannot share") → 0.9 (correct behavior)
    - Code executor returned output → +0.3 bonus
    - LLM direct gave substantive answer → base 0.6
    - Short answers (<10 chars) for non-privacy tasks → 0.2
    """
    if not answer:
        return 0.0

    a = answer.strip()

    # Error patterns
    err = bool(re.search(r"\b(error|exception|failed|traceback|task failed)\b", a, re.I))
    if err and len(a) < 200:
        return 0.1

    # Privacy refusal — always correct
    if "cannot share this information" in a or "I cannot share" in a:
        return 0.9

    # Empty/too-short non-refusal
    if len(a) < 8:
        return 0.2

    # Base score by strategy
    if strategy == "code_exec":
        # Code ran, returned something meaningful
        base = 0.85 if len(a) > 0 else 0.3
    elif strategy == "mcp_tools":
        base = 0.75 if len(a) > 20 else 0.4
    else:  # llm_direct
        base = 0.65 if len(a) > 15 else 0.35

    # Boost for structured numeric/ID answers (often analytical tasks)
    if expected_type == "number" and re.search(r"\d", a):
        base = min(1.0, base + 0.1)
    elif expected_type == "id" and re.search(r"[A-Z0-9]{4,}", a):
        base = min(1.0, base + 0.1)

    # Penalty for apology/uncertainty patterns in supposed factual answers
    if re.search(r"\b(I don.t know|I.m not sure|I cannot determine|I.m unable)\b", a, re.I):
        base = max(0.0, base - 0.25)

    return round(min(1.0, max(0.0, base)), 3)


# ── Hard routing rules (no LLM) ──────────────────────────────────────────────

# These override UCB1 when the category semantics make one strategy obviously correct.
_FORCED_STRATEGY: dict[str, str] = {
    # Privacy: must use llm_direct (returns I cannot share)
    "private_customer_information": "llm_direct",
    "confidential_company_knowledge": "llm_direct",
}

# Categories where code_exec is strongly preferred as first attempt
_CODE_FIRST: frozenset[str] = frozenset({
    "monthly_trend_analysis", "lead_routing", "case_routing",
    "transfer_count", "sales_amount_understanding", "handle_time",
    "conversion_rate_comprehension", "best_region_identification",
    "lead_qualification", "activity_priority", "wrong_stage_rectification",
    "sales_cycle_understanding", "sales_insight_mining",
    "top_issue_identification", "named_entity_disambiguation",
    "invalid_config", "internal_operation_data", "quote_approval",
})


# ── Router ────────────────────────────────────────────────────────────────────

class Router:
    """Smart outer model — pure strategy selection and reward attribution.

    No LLM calls. Works entirely from Brain state + deterministic rules.

    Usage:
        router = Router(brain)
        strategy = router.select(category)
        # ... run handler ...
        reward = router.reward(answer, strategy)
        router.record(task_text, category, strategy, model, reward, session_id)
    """

    def __init__(self, brain: Brain) -> None:
        self.brain = brain

    def select(
        self,
        category: str,
        candidates: list[str] | None = None,
        force_strategy: str | None = None,
    ) -> str:
        """Select optimal strategy for this category.

        Priority order (highest to lowest):
          1. force_strategy argument
          2. Hard routing rules (_FORCED_STRATEGY)
          3. UCB1 bandit (Brain L4) informed by EMA (Brain L3)
          4. Domain bias (code_exec first for analytical categories)
          5. Default: llm_direct
        """
        if force_strategy:
            return force_strategy

        forced = _FORCED_STRATEGY.get(category)
        if forced:
            return forced

        arms = candidates or STRATEGIES

        # Narrow arms: code_exec preferred for analytical categories
        if category in _CODE_FIRST and "code_exec" in arms:
            # Still UCB1, but only between code_exec and llm_direct
            arms = ["code_exec", "llm_direct"]

        return self.brain.recommend_strategy(category, candidates=arms)

    def reward(
        self,
        answer: str | None,
        strategy: str,
        expected_type: str = "text",
    ) -> float:
        """Compute proxy reward from answer signals (no LLM)."""
        return compute_reward(answer, strategy, expected_type=expected_type)

    def record(
        self,
        task_text: str,
        category: str,
        strategy: str,
        model: str,
        reward: float,
        session_id: str = "",
        hint: str = "",
    ) -> None:
        """Persist outcome to all Brain layers (L2–L5)."""
        self.brain.record(
            task_text=task_text,
            category=category,
            strategy=strategy,
            model=model,
            reward=reward,
            session_id=session_id,
            hint=hint,
        )

    def enrich_system_prompt(
        self,
        base_system: str,
        task_text: str,
        category: str,
        session_id: str,
    ) -> str:
        """Inject brain context into the system prompt (no LLM).

        Appends working memory, episodic recall, and semantic hints
        to the base system prompt string.
        """
        ctx = self.brain.build_context(task_text, category, session_id)
        if not ctx:
            return base_system
        return base_system + "\n\n" + ctx

    def diagnostics(self) -> dict[str, Any]:
        """Return full routing diagnostics for monitoring."""
        return {
            "brain": self.brain.diagnostics(),
            "forced_strategies": _FORCED_STRATEGY,
            "code_first_categories": len(_CODE_FIRST),
        }
