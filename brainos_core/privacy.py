"""PrivacyGuard — detect PII and refuse before any tool/LLM cost.

Extracted from agent-purple + agent-finance.
Runs deterministically (no LLM) in <1ms.
Always runs BEFORE calling Claude to avoid wasting tokens on PII requests.
"""
from __future__ import annotations

import re

# Patterns that indicate PII requests
_PII_PATTERNS = [
    r'\bssn\b', r'\bsocial security\b', r'\bpassport\b',
    r'\bcredit card\b', r'\bbank account\b', r'\brouting number\b',
    r'\bdate of birth\b', r'\bmedical record\b', r'\bprescription\b',
    r'\bhome address\b', r'\bprivate email\b',
]

_PII_RE = re.compile('|'.join(_PII_PATTERNS), re.IGNORECASE)

# Standard refusal message
REFUSAL = "I cannot share this information."


class PrivacyGuard:
    """Fast, deterministic PII detection and refusal.

    Usage:
        guard = PrivacyGuard()
        refused = guard.check(task_text)
        if refused:
            return refused  # Return the refusal string directly
    """

    def check(self, text: str) -> str | None:
        """Check if text contains a PII request.

        Returns:
            Refusal string if PII detected, None if safe to proceed.
        """
        if _PII_RE.search(text):
            return REFUSAL
        return None

    def is_privacy_task(self, category: str) -> bool:
        """Return True if the task category is inherently about private data."""
        return category in {
            "private_customer_information",
            "confidential_company_knowledge",
        }

    def build_refusal_prompt(self, category: str) -> str:
        """Build a system prompt that instructs Claude to refuse PII sharing."""
        return (
            "PRIVACY POLICY: You must NEVER reveal private customer information, "
            "confidential company data, personally identifiable information (PII), "
            "or any restricted data.\n"
            f"This task is categorized as '{category}' which involves sensitive data.\n"
            f"If asked to reveal such data, respond with exactly: {REFUSAL}"
        )
