"""Task format detector — routes A2A messages to the right handler.

Extracted from agent-purple. Each benchmark sends tasks in a different format:
- CRMArenaPro: JSON with type=crm_task + embedded required_context
- τ²-Bench: text with embedded tool schemas (function definitions)
- OfficeQA: plain text Q&A
- ResearchToolBench: tool-use task format
- CausalRivers: time-series data JSON

detect_task_format() returns a string that agents use to route to the right handler.
"""
from __future__ import annotations


def detect_task_format(text: str) -> str:
    """Identify the benchmark task format from message text.

    Returns one of: "crm", "tau2", "officeqa", "causal", "research", "generic"
    """
    stripped = text.strip()
    lower = stripped.lower()

    # CRMArenaPro: JSON with type field
    if stripped.startswith('{') and ('"crm_task"' in stripped[:300] or "'crm_task'" in stripped[:300]):
        return "crm"

    # τ²-Bench: embedded tool schemas in message text
    if (
        "here's a list of tools you can use" in lower
        or ('"type": "function"' in stripped and '"name": "respond"' in stripped)
    ):
        return "tau2"

    # CausalRivers: time series JSON
    if stripped.startswith('{') and '"time_series"' in stripped[:500]:
        return "causal"

    # OfficeQA: Q&A format (plain questions)
    if lower.startswith("question:") or lower.startswith("q:"):
        return "officeqa"

    return "generic"


def is_analytical_category(category: str) -> bool:
    """Return True if a CRM task category requires code execution."""
    return category in {
        "monthly_trend_analysis", "lead_routing", "case_routing",
        "transfer_count", "sales_amount_understanding", "handle_time",
        "conversion_rate_comprehension", "best_region_identification",
        "lead_qualification", "activity_priority", "wrong_stage_rectification",
        "sales_cycle_understanding", "sales_insight_mining",
        "top_issue_identification", "named_entity_disambiguation",
        "invalid_config", "internal_operation_data", "quote_approval",
        "knowledge_qa",
    }


def is_privacy_category(category: str) -> bool:
    """Return True if a CRM task category involves PII/confidential data."""
    return category in {
        "private_customer_information",
        "confidential_company_knowledge",
    }
