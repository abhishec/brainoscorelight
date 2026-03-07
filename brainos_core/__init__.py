"""BrainOS Core Light — shared AI worker primitives.

Minimal, composable building blocks for AgentBeats benchmark agents.
Each module solves exactly one problem, has zero hidden state,
and can be imported independently.

Quick start:
    from brainos_core import BrainOSWorker
    worker = BrainOSWorker(anthropic_api_key="sk-ant-...")
    answer = await worker.handle(task_text, task_category="monthly_trend_analysis")
"""
from .executor import ClaudeExecutor
from .detector import detect_task_format
from .mcp import MCPBridge
from .code_exec import CodeExecutor
from .memory import RLPrimer
from .privacy import PrivacyGuard
from .worker import BrainOSWorker

__all__ = [
    "BrainOSWorker",
    "ClaudeExecutor",
    "MCPBridge",
    "CodeExecutor",
    "RLPrimer",
    "PrivacyGuard",
    "detect_task_format",
]
__version__ = "0.1.0"
