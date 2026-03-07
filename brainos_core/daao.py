"""daao.py — Difficulty-Aware Adaptive Orchestration (zero-LLM model routing).

Routes every task to the cheapest capable model before a single token is spent
on execution. Routing is deterministic — keyword pattern + task length — so it
costs nothing to evaluate.

Fast path (Haiku):   simple lookups, single-step navigation, short tasks.
Deep path (Sonnet):  multi-source synthesis, reasoning, analysis, long tasks.
"""
from __future__ import annotations

import re

# Keywords that suggest a task is simple enough for the fast model
_FAST_PATTERNS = re.compile(
    r"\b(what is|what are|define|definition|who is|where is|when was"
    r"|navigate to|open|go to|how many|list the|name the|look up"
    r"|find the address|find the phone|get the url)\b",
    re.IGNORECASE,
)

# Keywords that signal deep reasoning regardless of task length
_DEEP_PATTERNS = re.compile(
    r"\b(analyz|synthes|compar|evaluat|optimiz|recommend|forecast"
    r"|summarize.*multiple|across sources|multi.step|portfolio"
    r"|schedul|debug|research|investigat|rebalanc|allocat)\b",
    re.IGNORECASE,
)

# Default word-count threshold for short = fast
_DEFAULT_FAST_WORD_LIMIT = 12


class DAAO:
    """Zero-LLM model selector: routes tasks to the cheapest capable model.

    Usage::

        daao = DAAO(fast_model="claude-haiku-4-5-20251001",
                    main_model="claude-sonnet-4-6")
        model = daao.route(task_text)
    """

    def __init__(
        self,
        fast_model: str,
        main_model: str,
        fast_word_limit: int = _DEFAULT_FAST_WORD_LIMIT,
    ) -> None:
        self.fast_model = fast_model
        self.main_model = main_model
        self.fast_word_limit = fast_word_limit

    def route(self, task_text: str) -> str:
        """Return model ID for this task. Zero API cost.

        Priority:
        1. Deep-signal keyword → main model (always)
        2. Short task AND fast-path keyword → fast model
        3. Short task (under word limit) AND no deep signals → fast model
        4. Everything else → main model
        """
        if _DEEP_PATTERNS.search(task_text):
            return self.main_model

        word_count = len(task_text.split())

        if _FAST_PATTERNS.search(task_text):
            return self.fast_model

        if word_count <= self.fast_word_limit:
            return self.fast_model

        return self.main_model

    def is_fast_path(self, task_text: str) -> bool:
        """True when Haiku is sufficient for this task."""
        return self.route(task_text) == self.fast_model

    def is_deep_path(self, task_text: str) -> bool:
        """True when Sonnet is needed for this task."""
        return self.route(task_text) == self.main_model
