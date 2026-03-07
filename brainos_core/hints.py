"""hints.py — Prefix-based sequence hints and recovery cascade.

Zero-LLM, protocol-agnostic. Works against any MCP server regardless of
whether the tool is called search_arxiv, search_web, or search_pubmed.

Two primitives:
  SequenceHints   — inject ordered tool-call directives per domain/category
  RecoveryCascade — append a broadening hint when a tool returns no results

Sequence directives use prefixes, not tool names. The LLM matches the prefix
against the actual tool names available at runtime. This makes the hints
portable across any green agent or MCP server configuration.
"""
from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Domain → tool-call sequence (prefix-based)
# ---------------------------------------------------------------------------

# Web / shopping domains
_WEB_SEQUENCES: dict[str, str] = {
    "shopping":  "search_/find_ → view_/click_ → add_ → checkout_",
    "booking":   "search_/find_ → view_/check_ → select_ → confirm_/book_",
    "search":    "search_/find_ → view_/get_ → compare_ → return_",
    "task":      "open_/navigate_ → fill_/enter_ → submit_/click_",
    "navigate":  "go_/open_/navigate_ → click_/select_",
}

# Research domains
_RESEARCH_SEQUENCES: dict[str, str] = {
    "academic":   "search_/query_ → get_/fetch_ → cite_/references_ → synthesize",
    "news":       "search_/find_ → get_/fetch_ → verify_/check_ → verdict",
    "technical":  "list_/check_ → run_/execute_ → debug_/fix_ → verify_",
    "code":       "search_/find_ → read_/fetch_ → analyze_/review_ → suggest",
    "general":    "search_ → get_/fetch_ → summarize_",
}

# CRM / business process domains
_CRM_SEQUENCES: dict[str, str] = {
    "case_management":      "get_/search_ → update_/set_ → verify_",
    "lead_management":      "search_/find_ → update_/assign_ → verify_",
    "opportunity_tracking": "get_ → update_/set_ → log_",
    "activity_logging":     "get_ → create_/log_ → verify_",
    "email_communication":  "get_ → send_/create_ → verify_",
}

_ALL_SEQUENCES: dict[str, str] = {
    **_WEB_SEQUENCES,
    **_RESEARCH_SEQUENCES,
    **_CRM_SEQUENCES,
}


class SequenceHints:
    """Inject ordered tool-call directives into system prompts.

    Directives use tool-name *prefixes* so they are portable across any MCP
    server regardless of the actual tool name convention.

    Usage::

        directive = SequenceHints.directive("shopping")
        # → "For shopping tasks follow this tool sequence: search_/find_ → ..."
    """

    @classmethod
    def get(cls, domain: str, fallback: str = "") -> str:
        """Return raw sequence string for a domain, or fallback if unknown."""
        return _ALL_SEQUENCES.get(domain.lower(), fallback)

    @classmethod
    def directive(cls, domain: str) -> str:
        """Return formatted directive for injection into system prompt."""
        seq = cls.get(domain)
        if not seq:
            return ""
        return (
            f"For {domain} tasks, follow this tool-call sequence (prefixes, not exact names):\n"
            f"  {seq}\n"
            "Match tool names by prefix. Complete all steps in order."
        )

    @classmethod
    def known_domains(cls) -> list[str]:
        """Return all known domain names."""
        return list(_ALL_SEQUENCES.keys())


# ---------------------------------------------------------------------------
# Recovery cascade — empty/no-result tool outputs
# ---------------------------------------------------------------------------

_EMPTY_RESULT_PATTERN = re.compile(
    r"no results?|0 items?|not found|empty|nothing found|no match"
    r"|returned 0|no data|unavailable",
    re.IGNORECASE,
)

_DEFAULT_RECOVERY_HINT = (
    "[RECOVERY HINT: Try broader search terms. "
    "Remove specific constraints and search by category. "
    "Try alternative product names or synonyms.]"
)


class RecoveryCascade:
    """Append a broadening hint to tool results that return no data.

    The most common dead-end in web and research agents: an overspecific
    query returning nothing. This catches it at zero API cost by appending
    a recovery hint to the tool result before the LLM's next turn.

    Usage::

        enriched = RecoveryCascade.check(tool_result)
        # → original result + hint if it looks empty
    """

    @classmethod
    def is_empty(cls, tool_result: str) -> bool:
        """True if tool result signals no data was found."""
        return bool(_EMPTY_RESULT_PATTERN.search(tool_result))

    @classmethod
    def check(cls, tool_result: str, custom_hint: Optional[str] = None) -> str:
        """Return tool_result with recovery hint appended if result is empty."""
        if cls.is_empty(tool_result):
            hint = custom_hint or _DEFAULT_RECOVERY_HINT
            return tool_result + "\n" + hint
        return tool_result
