"""BrainOSWorker — the composable base class for all BrainOS Mini AI Workers.

THE COMMON LOOP (all 5 agents do this exact sequence):

    message = receive_a2a()            # A2A JSON-RPC 2.0
    task_type = detect_format(message) # crm / tau2 / officeqa / research / generic
    model = daao.route(message)        # DAAO: smart model selection — ZERO LLM cost
    strategy = router.select(category) # Router: UCB1 bandit — ZERO LLM cost
    tools = discover_mcp(endpoint)     # MCP bridge
    answer = call_claude(context,      # LLM executes with DAAO-selected model
                         model=model)
    reward = router.reward(answer)     # proxy reward — ZERO LLM cost
    brain.record(category, reward)     # update all 5 memory layers
    return format_a2a(answer)          # A2A response

KEY PRINCIPLE: DAAO and Router are ALWAYS upstream of LLM calls.
No model is ever hardcoded at the call site — DAAO selects it.
No strategy is ever hardcoded at the call site — Router selects it.

Usage::

    class MyCRMWorker(BrainOSWorker):
        async def handle_domain(self, task_text, task_category, session_id):
            model = self.daao.route(task_text)
            return await self.executor.call(
                system="You are a helpful CRM agent.",
                user=task_text,
                model=model,
            )

    worker = MyCRMWorker(api_key="sk-ant-...", mcp_endpoint="http://green-agent:9009")
    answer = await worker.handle(task_text)
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod

from .brain import Brain
from .code_exec import CodeExecutor
from .daao import DAAO
from .detector import detect_task_format, is_analytical_category, is_privacy_category
from .executor import ClaudeExecutor, TokenBudget
from .mcp import MCPBridge
from .memory import Outcome, RLPrimer
from .privacy import PrivacyGuard
from .router import Router, compute_reward


# CRM schema drift aliases (hardcoded by CRMArenaPro evaluation)
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

    Wires the complete cognitive loop:
      privacy guard → detect format → DAAO model routing →
      Router strategy selection → execute → Brain RL update

    Subclasses implement handle_domain() for benchmark-specific logic.
    All LLM calls from subclasses should use self.executor.call(model=self.daao.route(...)).
    """

    def __init__(
        self,
        api_key: str,
        mcp_endpoint: str = "http://localhost:9009",
        default_model: str = "claude-sonnet-4-6",
        fast_model: str = "claude-haiku-4-5-20251001",
        rl_cache: str = "~/.brainos/rl_primer.json",
        brain_dir: str | None = None,
    ):
        # LLM executor — model is always set by DAAO, not here
        self.executor = ClaudeExecutor(
            api_key=api_key,
            default_model=default_model,
            fast_model=fast_model,
        )
        self.code_exec = CodeExecutor(claude=self.executor)
        self.mcp = MCPBridge(endpoint=mcp_endpoint)
        self.privacy = PrivacyGuard()

        # Smart routing — ZERO LLM cost
        self.daao = DAAO(fast_model=fast_model, main_model=default_model)
        self.brain = Brain(base_dir=brain_dir) if brain_dir else Brain()
        self.router = Router(self.brain)

        # Legacy RL primer (kept for backward compat with older agent code)
        self.rl = RLPrimer(cache_path=rl_cache)

    async def handle(
        self,
        task_text: str,
        task_category: str = "",
        session_id: str = "",
    ) -> str:
        """THE COMMON LOOP: route and handle any A2A task.

        Routing priority:
        1. Privacy fast-fail — zero LLM cost
        2. DAAO model selection — zero LLM cost
        3. Router strategy selection — zero LLM cost (for CRM)
        4. Execute with selected model
        5. Record outcome to Brain — zero LLM cost
        """
        # 1. Privacy fast-fail (no LLM cost for obvious PII requests)
        refused = self.privacy.check(task_text)
        if refused:
            return refused

        fmt = detect_task_format(task_text)

        # 2. DAAO: select model BEFORE any LLM call
        model = self.daao.route(task_text)

        # 3. CRM task: Router selects strategy, DAAO-selected model goes to executor
        if fmt == "crm":
            return await self._handle_crm(task_text, session_id=session_id, model=model)

        # 4. Domain-specific handling (subclass implements this)
        return await self.handle_domain(task_text, task_category, session_id)

    async def _handle_crm(
        self,
        task_text: str,
        session_id: str = "",
        model: str | None = None,
    ) -> str:
        """Handle CRMArenaPro tasks.

        Strategy: Router.select(category) → UCB1 bandit (zero LLM)
        Model:    DAAO-selected, passed in from handle() or explicitly provided
        Execute:  code_exec (analytical) | llm_direct (lookup/privacy)
        Record:   Router.reward() → Brain.record() (all 5 layers, zero LLM)
        """
        try:
            task = json.loads(task_text)
        except Exception:
            task = {"prompt": task_text, "required_context": "", "persona": "CRM agent", "task_category": ""}

        prompt = task.get("prompt", task_text)
        context = task.get("required_context", "")
        persona = task.get("persona", "CRM agent")
        category = task.get("task_category", "")

        # Resolve model: use DAAO-selected if not provided explicitly
        if model is None:
            model = self.daao.route(task_text)

        # ── Router: UCB1 strategy selection (zero LLM) ────────────────────────
        strategy = self.router.select(category)
        print(f"[worker] cat={category} strategy={strategy} model={model}")

        # ── RL primer from episodic memory ────────────────────────────────────
        primer = self.rl.build_prompt(prompt, category)

        # ── Execute chosen strategy with DAAO-selected model ──────────────────
        answer: str = ""

        if strategy == "code_exec":
            # Code generation → subprocess sandbox → exact answer
            answer = await self.code_exec.solve(
                question=prompt,
                data=context,
                category=category,
                field_aliases=CRM_FIELD_ALIASES,
                extra_instructions=primer,
            ) or ""
            if not answer:
                # Code exec failed → fallback to llm_direct with DAAO model
                print(f"[worker] code_exec failed for cat={category}, falling back to llm_direct")
                strategy = "llm_direct"

        if strategy == "llm_direct" or not answer:
            if self.privacy.is_privacy_task(category):
                system = self.privacy.build_refusal_prompt(category)
                answer = await self.executor.call(
                    system=system,
                    user=f"Question: {prompt}\n\nCRM Context:\n{context}",
                    max_tokens=64,
                    force_fast=True,  # Privacy refusals always use fast model
                )
            else:
                drift_note = ", ".join(f"{k}={v}" for k, v in CRM_FIELD_ALIASES.items())
                system = (
                    f"You are a {persona}. Answer using ONLY the provided CRM data.\n"
                    f"Field aliases: {drift_note}\n"
                    f"{primer}\n"
                    "Return ONLY the exact answer. No explanation."
                )
                # DAAO model used — not hardcoded
                answer = await self.executor.call(
                    system=system,
                    user=f"Question: {prompt}\n\nData:\n{context}",
                    max_tokens=128,
                    model=model,
                )

        # ── Record outcome to Brain (all 5 layers, zero LLM) ──────────────────
        reward = self.router.reward(answer, strategy)
        self.router.record(
            task_text=prompt[:120],
            category=category,
            strategy=strategy,
            model=model,
            reward=reward,
            session_id=session_id,
        )
        print(f"[worker] cat={category} strategy={strategy} model={model} reward={reward:.2f}")

        return answer

    @abstractmethod
    async def handle_domain(
        self,
        task_text: str,
        task_category: str,
        session_id: str,
    ) -> str:
        """Implement domain-specific task handling.

        Called for non-CRM tasks. Always use self.daao.route() to select model::

            async def handle_domain(self, task_text, task_category, session_id):
                model = self.daao.route(task_text)
                return await self.executor.call(
                    system="...", user=task_text, model=model
                )
        """
        ...
