# BrainOS Core Light

**Minimal, composable AI worker primitives for AgentBeats benchmark agents.**

Build a new benchmark agent in ~40 lines. No boilerplate.

```python
from brainos_core import BrainOSWorker

class MyCRMAgent(BrainOSWorker):
    async def handle_domain(self, task_text, task_category, session_id):
        return await self.executor.call(
            system="You are a helpful business agent.",
            user=task_text,
        )

worker = MyCRMAgent(api_key="sk-ant-...", mcp_endpoint="http://green-agent:9009")
answer = await worker.handle(task_text)
```

## What's Inside

| Module | Purpose |
|--------|---------|
| `ClaudeExecutor` | Anthropic API wrapper — model switching, retries, budget awareness |
| `CodeExecutor` | LLM → Python code → subprocess sandbox → stdout answer |
| `MCPBridge` | JSON-RPC 2.0 MCP tool discovery + calling |
| `RLPrimer` | UCB1 bandit for strategy selection per task category |
| `PrivacyGuard` | Deterministic PII detection + refusal (no LLM, <1ms) |
| `detect_task_format` | Routes A2A messages: CRM / τ²-Bench / OfficeQA / CausalRivers |
| `BrainOSWorker` | Abstract base — wires all above into a single `handle()` call |

## Install

```bash
pip install brainos-core-light
```

Or directly from GitHub:

```bash
pip install git+https://github.com/abhishec/brainoscorelight.git
```

## Key Innovation: Code Execution for Analytical Tasks

Analytical CRM categories (monthly trend analysis, lead routing, case routing, etc.)
need **computation**, not LLM reasoning. The `CodeExecutor` solves this:

1. Claude Sonnet generates Python code from the question + raw CRM data
2. Code runs in a subprocess sandbox (8s timeout, /tmp isolation)
3. stdout is the answer — deterministic, fast, accurate
4. Falls back to LLM direct if code fails

This moves analytical categories from ~5% → ~80%+ accuracy.

## Benchmarks Supported

- **CRMArenaPro** — 22 task categories, synthetic CRM data, schema drift
- **τ²-Bench** — airline customer service with embedded tool schemas
- **OfficeQA** — plain text Q&A over business documents
- **CausalRivers** — time-series causal inference

## Example: New CRM Agent in 40 Lines

See [`examples/new_crm_agent.py`](examples/new_crm_agent.py).

## Architecture

```
BrainOSWorker.handle(task_text)
    │
    ├─ PrivacyGuard.check()      → refuse PII immediately (no LLM)
    ├─ detect_task_format()      → "crm" | "tau2" | "officeqa" | "generic"
    │
    ├─ CRM path:
    │   ├─ privacy category      → ClaudeExecutor (refuse)
    │   ├─ analytical category   → CodeExecutor → fallback ClaudeExecutor
    │   └─ other                 → ClaudeExecutor (Haiku, fast)
    │
    └─ handle_domain()           → your custom logic
```

## License

MIT
