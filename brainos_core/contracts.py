"""contracts.py — Deterministic completion contracts (L2 checkout, L3 citation, depth retry).

Completion contracts enforce mechanical guarantees after LLM execution.
Each contract is a pure function: no API calls, no side effects, instant.

Contracts:
  CheckoutContract — L2: verify a purchase/booking tool was called
  CitationContract — L3: verify academic/news answers contain citations
  DepthContract    — verify answer is substantive (not a one-liner)
  MutationContract — verify write tools were called when task requires mutation
"""
from __future__ import annotations

import re
from typing import List


# ---------------------------------------------------------------------------
# L2: Checkout / booking completion
# ---------------------------------------------------------------------------

_CHECKOUT_PREFIXES = (
    "checkout", "purchase", "buy", "order", "confirm", "book",
    "complete_purchase", "place_order", "finalize",
)


class CheckoutContract:
    """L2: verify that a checkout/purchase/booking tool was actually called.

    The most common shopping agent failure: completing the search-and-add phase
    but missing the final purchase step. Caught here at zero LLM cost.

    Usage::

        if not CheckoutContract.satisfied(tool_calls_made):
            # inject forced-checkout directive and re-run
            system_prompt += CheckoutContract.directive()
    """

    @classmethod
    def satisfied(cls, tool_calls_made: List[str]) -> bool:
        """True if at least one checkout-type tool was called."""
        return any(
            tc.lower().startswith(p) or p in tc.lower()
            for tc in tool_calls_made
            for p in _CHECKOUT_PREFIXES
        )

    @classmethod
    def directive(cls) -> str:
        """Forced-checkout directive to inject on contract failure."""
        return (
            "CRITICAL: You must call the checkout/purchase/confirm tool now "
            "to complete the transaction. The task is not complete until you do."
        )


# ---------------------------------------------------------------------------
# L3: Citation requirement for academic / news tasks
# ---------------------------------------------------------------------------

_CITATION_PATTERNS = [
    re.compile(r"\[[A-Z][a-z]+ \d{4}\]"),       # [Author YYYY]
    re.compile(r"https?://\S+"),                  # bare URL
    re.compile(r"Source:", re.IGNORECASE),
    re.compile(r"\(\d{4}\)"),                     # (YYYY)
    re.compile(r"doi:\s*10\.", re.IGNORECASE),    # DOI
    re.compile(r"et al\.", re.IGNORECASE),        # et al.
    re.compile(r"according to", re.IGNORECASE),
]


class CitationContract:
    """L3: verify academic/news answers contain at least one citation signal.

    On failure: one targeted retry that adds exactly what was missing.
    On passing cases: pure string scan — zero API cost.

    Usage::

        if not CitationContract.satisfied(answer):
            # append directive to last user turn and re-run
            last_user_turn += "\\n" + CitationContract.directive()
    """

    @classmethod
    def satisfied(cls, answer: str, min_length: int = 100) -> bool:
        """True if answer contains a citation signal (or is too short to need one)."""
        if len(answer.strip()) < min_length:
            return True  # depth check handles short answers separately
        return any(p.search(answer) for p in _CITATION_PATTERNS)

    @classmethod
    def directive(cls) -> str:
        """Citation injection directive to append on contract failure."""
        return (
            "CRITICAL: Include citations for every claim. "
            "Format: [Author YYYY] or source URL after each claim."
        )


# ---------------------------------------------------------------------------
# Depth retry: answer must be substantive
# ---------------------------------------------------------------------------

_SHALLOW_INDICATORS = re.compile(
    r"^(I (don'?t|cannot|can'?t) (find|locate|access)|"
    r"No (results|data|information)|"
    r"I (was|am) unable|"
    r"Sorry,? I|"
    r"Unfortunately)",
    re.IGNORECASE,
)


class DepthContract:
    """Depth retry: verify the answer is substantive, not a one-liner cop-out.

    Fires after any tool-use execution. If the LLM summarized everything into
    one sentence despite having tool results, re-run with a depth directive.

    Usage::

        if not DepthContract.satisfied(answer):
            system_prompt += DepthContract.directive()
    """

    @classmethod
    def satisfied(cls, answer: str, min_chars: int = 100) -> bool:
        """True if answer meets minimum depth requirements."""
        stripped = answer.strip()
        if len(stripped) < min_chars:
            return False
        if _SHALLOW_INDICATORS.match(stripped):
            return False
        return True

    @classmethod
    def directive(cls) -> str:
        """Depth directive to inject when answer is too shallow."""
        return (
            "Your answer is too brief. Use the tool results you already have. "
            "Add specific details, supporting evidence, and cite your sources."
        )


# ---------------------------------------------------------------------------
# L2: Mutation verification (write tools were called when required)
# ---------------------------------------------------------------------------

_WRITE_PREFIXES = (
    "update", "create", "delete", "set", "write", "save",
    "post", "send", "submit", "insert", "modify", "patch",
)

_MUTATION_TASK_PATTERN = re.compile(
    r"\b(update|create|delete|remove|change|set|assign|move|transfer"
    r"|cancel|rebook|send|submit|book|purchase|order|schedule)\b",
    re.IGNORECASE,
)


class MutationContract:
    """L2: verify write tools were called when the task requires mutation.

    Usage::

        requires = MutationContract.task_requires_mutation(task_text)
        if requires and not MutationContract.any_mutation(tool_calls_made):
            # task likely incomplete — retry or flag
            pass
    """

    @classmethod
    def task_requires_mutation(cls, task_text: str) -> bool:
        """Heuristic: does this task description require a write operation?"""
        return bool(_MUTATION_TASK_PATTERN.search(task_text))

    @classmethod
    def any_mutation(cls, tool_calls_made: List[str]) -> bool:
        """True if at least one write-type tool was called."""
        return any(
            tc.lower().startswith(p) or p in tc.lower()
            for tc in tool_calls_made
            for p in _WRITE_PREFIXES
        )
