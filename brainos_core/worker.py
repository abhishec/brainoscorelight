"""BrainOSWorker — the composable base class for all BrainOS Mini AI Workers.

Usage:
    class MyCRMWorker(BrainOSWorker):
        async def handle_domain(self, task_text, task_category, context):
            # domain-specific logic here
            pass

    worker = MyCRMWorker(api_key="sk-ant-...")
    answer = await worker.handle(task_text)
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod

from .code_exec import CodeExecutor
from .detector import detect_task_format, is_analytical_category, is_privacy_category
from .executor import ClaudeExecutor, TokenBudget
from .mcp import MCPBridge
from .memory import Outcome, RLPrimer
from .privacy import PrivacyGuard


# CRM schema drift aliases (hardcoded by CRMArenaPro at medium level)
CRM_FIELD_ALIASES = {
    "AssignedAgent": "OwnerId",
    "ClientId": "AccountId",
    "PersonRef": "ContactId",
    "StatusCode": "Status",
    "Title": "Subject",
    "Details": "Description",
}


class BrainOSWorker(ABC):
    """Base class for all BrainOS Mini AI Workers.

    Provides:
    - Task format detection
    - Privacy guard
    - Code execution for analytical tasks
    - RL primer for learned patterns
    - MCP bridge for tool-using benchmarks

    Subclasses implement handle_domain() for benchmark-specific logic.
    """

    def __init__(
        self,
        api_key: str,
        mcp_endpoint: str = "http://localhost:9009",
        default_model: str = "claude-sonnet-4-6",
        fast_model: str = "claude-haiku-4-5-20251001",
        rl_cache: str = "~/.brainos/rl_primer.json",
    ):
        self.executor = ClaudeExecutor(
            api_key=api_key,
            default_model=default_model,
            fast_model=fast_model,
        )
        self.code_exec = CodeExecutor(claude=self.executor)
        self.mcp = MCPBridge(endpoint=mcp_endpoint)
        self.privacy = PrivacyGuard()
        self.rl = RLPrimer(cache_path=rl_cache)

    async def handle(
        self,
        task_text: str,
        task_category: str = "",
        session_id: str = "",
    ) -> str:
        """Route and handle any A2A task.

        Routing priority:
        1. Privacy fast-fail (no LLM cost for obvious PII requests)
        2. CRM tasks with code execution for analytical categories
        3. Domain-specific handling via handle_domain()
        4. Generic LLM fallback
        """
        # 1. Privacy fast-fail
        refused = self.privacy.check(task_text)
        if refused:
            return refused

        fmt = detect_task_format(task_text)

        # 2. CRM task handling
        if fmt == "crm":
            return await self._handle_crm(task_text)

        # 3. Domain-specific handling
        return await self.handle_domain(task_text, task_category, session_id)

    async def _handle_crm(self, task_text: str) -> str:
        """Handle CRMArenaPro tasks with code execution for analytical categories."""
        try:
            task = json.loads(task_text)
        except Exception:
            task = {"prompt": task_text, "required_context": "", "task_category": ""}

        prompt = task.get("prompt", task_text)
        context = task.get("required_context", "")
        persona = task.get("persona", "CRM agent")
        category = task.get("task_category", "")

        # RL primer: inject learned patterns
        primer = self.rl.build_prompt(prompt, category)
        best_strategy = self.rl.best_strategy(category)

        if self.privacy.is_privacy_task(category):
            system = self.privacy.build_refusal_prompt(category)
            answer = await self.executor.call(
                system=system,
                user=f"Question: {prompt}\n\nCRM Context:\n{context}",
                max_tokens=64,
                force_fast=True,
            )
            success = "cannot" in answer.lower() or "refuse" in answer.lower()
            self.rl.record(Outcome(category, prompt[:100], "llm_direct", success, 1.0 if success else 0.0))
            return answer

        if is_analytical_category(category):
            answer = await self.code_exec.solve(
                question=prompt,
                data=context,
                category=category,
                field_aliases=CRM_FIELD_ALIASES,
                extra_instructions=primer,
            )
            if answer:
                self.rl.record(Outcome(category, prompt[:100], "code_exec", True, 0.8))
                return answer
            # Fallback to LLM
            self.rl.record(Outcome(category, prompt[:100], "code_exec", False, 0.0))

        # LLM direct
        drift_note = ", ".join(f"{k}={v}" for k, v in CRM_FIELD_ALIASES.items())
        system = (
            f"You are a {persona}. Answer using ONLY the provided CRM data.\n"
            f"Field aliases: {drift_note}\n"
            f"{primer}\n"
            "Return ONLY the exact answer. No explanation."
        )
        answer = await self.executor.call(
            system=system,
            user=f"Question: {prompt}\n\nData:\n{context}",
            max_tokens=128,
            force_fast=True,
        )
        self.rl.record(Outcome(category, prompt[:100], "llm_direct", True, 0.5))
        return answer

    @abstractmethod
    async def handle_domain(
        self,
        task_text: str,
        task_category: str,
        session_id: str,
    ) -> str:
        """Implement domain-specific task handling.

        Called for non-CRM tasks. Return the answer string.
        """
        ...
