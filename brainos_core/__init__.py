"""BrainOS Core Light — shared AI worker primitives.

Minimal, composable building blocks for AgentBeats benchmark agents.
Each module solves exactly one problem, has zero hidden state,
and can be imported independently.

Core layers:
    Brain     — 5-layer file-based memory (working/episodic/semantic/strategic/meta)
    Router    — zero-LLM orchestrator: UCB1 strategy selection + reward attribution
    DAAO      — zero-LLM model router: Haiku for simple tasks, Sonnet for complex
    Contracts — deterministic completion guarantees (checkout, citation, depth)
    Hints     — prefix-based sequence hints + recovery cascade
    BrainOSWorker — abstract base wiring all primitives into a single handle()
"""
from .executor import ClaudeExecutor
from .detector import detect_task_format
from .mcp import MCPBridge
from .code_exec import CodeExecutor
from .memory import RLPrimer
from .privacy import PrivacyGuard
from .brain import Brain, WorkingMemory, EpisodicMemory, SemanticMemory, StrategicMemory, MetaMemory
from .router import Router, compute_reward
from .worker import BrainOSWorker
from .daao import DAAO
from .contracts import CheckoutContract, CitationContract, DepthContract, MutationContract
from .hints import SequenceHints, RecoveryCascade

__all__ = [
    # Core orchestration
    "Brain",
    "Router",
    "compute_reward",
    # Memory layers
    "WorkingMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "StrategicMemory",
    "MetaMemory",
    # Primitives
    "BrainOSWorker",
    "ClaudeExecutor",
    "MCPBridge",
    "CodeExecutor",
    "RLPrimer",
    "PrivacyGuard",
    "detect_task_format",
    # Smart routing
    "DAAO",
    # Completion contracts
    "CheckoutContract",
    "CitationContract",
    "DepthContract",
    "MutationContract",
    # Sequence hints + recovery
    "SequenceHints",
    "RecoveryCascade",
]
__version__ = "0.3.0"
