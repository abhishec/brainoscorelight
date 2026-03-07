# BrainOS Core Light

**Minimal AI worker primitives powering all BrainOS benchmark agents.**

Zero boilerplate. Build a production benchmark agent in ~40 lines.

---

## Agent Benchmark Status

| Agent | Benchmark | Score | Status |
|-------|-----------|-------|--------|
| [purple-agent-business-process-worker](https://github.com/abhishec/purple-agent-business-process-worker) | CRMArenaPro | 20.7% → Run 5 (code executor) | 🔄 In progress |
| [purple-agent-finance-worker](https://github.com/abhishec/purple-agent-finance-worker) | OfficeQA | **100%** (246/246) | ✅ #1 |
| [purple-agent-web-worker](https://github.com/abhishec/purple-agent-web-worker) | WebShop+ | claude-sonnet-4-6 upgraded | 🚀 Deployed |
| [purple-agent-research-worker](https://github.com/abhishec/purple-agent-research-worker) | ResearchToolBench | claude-sonnet-4-6 upgraded | 🚀 Deployed |
| [causalriver-worker](https://github.com/abhishec/causalriver-worker) | CausalRivers | AUROC 0.7843 · #1 on 5/10 datasets | ✅ Running |

All agents install from: `pip install git+https://github.com/abhishec/brainoscorelight.git`

---

## Architecture: Brain + Router

```
Incoming task
     │
     ▼
Router.select(category)          ← zero-LLM orchestrator
     │
     ├─ hard rule? (privacy → llm_direct always)
     ├─ L4 UCB1 bandit → best (category, strategy) arm
     └─ L3 semantic EMA → cold-start prior
     │
     ▼
Execute chosen strategy:
  code_exec  → LLM generates Python → subprocess sandbox → stdout
  llm_direct → LLM answers from context
  mcp_tools  → discover MCP tools → LLM tool loop
     │
     ▼
Router.reward(answer, strategy)   ← proxy reward, NO LLM judge
     │
     ▼
Brain.record(...)                 ← update all 5 layers atomically
```

---

## 5-Layer Brain (file-based, survives Docker restarts)

| Layer | Class | What it stores | Persisted? |
|-------|-------|----------------|------------|
| L1 | `WorkingMemory` | Recent session turns (last 10) | No (in-memory) |
| L2 | `EpisodicMemory` | Task outcomes, last 500 episodes | ✅ `episodic_memory.json` |
| L3 | `SemanticMemory` | Category→strategy EMA beliefs, field aliases | ✅ `semantic_memory.json` |
| L4 | `StrategicMemory` | UCB1 bandit: counts + Q-values per (category, strategy) | ✅ `strategic_memory.json` |
| L5 | `MetaMemory` | Model success rates, run counts, worst categories | ✅ `meta_memory.json` |

All writes are atomic (write-to-tmp → rename). All reads degrade gracefully on I/O error.

Default path: `/app/brain/` (override with `BRAIN_DIR` env var).

---

## Module Map

| Module | Purpose | LLM calls? |
|--------|---------|------------|
| `brain.py` | 5-layer file-based memory | **No** |
| `router.py` | UCB1 strategy selection + proxy reward | **No** |
| `executor.py` | Anthropic API wrapper, retries, model switching | Yes |
| `code_exec.py` | LLM → Python code → subprocess sandbox → stdout | Yes (code gen only) |
| `mcp.py` | JSON-RPC 2.0 MCP tool discovery + calling | No |
| `privacy.py` | PII detection + refusal (<1ms) | **No** |
| `detector.py` | Task format routing (crm / tau2 / officeqa / causal) | **No** |
| `memory.py` | Legacy RLPrimer (use `Brain` instead for new agents) | **No** |
| `worker.py` | Abstract base — wires all primitives into `handle()` | Via executor |

---

## Quick Start

```python
from brainos_core import Brain, Router

# One Brain instance per process — shared across all requests
brain = Brain()        # reads/writes /app/brain/*.json
router = Router(brain)

# Select strategy — pure UCB1, no LLM
strategy = router.select(category="monthly_trend_analysis")
# → "code_exec" after a few tasks (bandit converges)

# After running your handler:
reward = router.reward(answer, strategy)
router.record(task_text, category, strategy, model="sonnet", reward=reward)
```

### Build a full agent in 40 lines

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

---

## RL Loop — No LLM Judge

```
Task → Router.select() → execute → Router.reward() → Brain.record()
                                        │
                    proxy reward (deterministic):
                    - code_exec returned output → 0.85
                    - privacy refusal (correct) → 0.90
                    - llm_direct substantive    → 0.65
                    - empty / error             → 0.10
```

UCB1 updates incrementally per (category, strategy) pair. Semantic EMA provides cold-start priors. After ~50 tasks per category the bandit converges; after 500 total episodes, episodic recall surfaces high-quality few-shot examples automatically.

---

## Key Innovation: Code Execution for Analytical Tasks

```
LLM guessing monthly_trend_analysis → ~5% accuracy
Python groupby().count().idxmax()   → ~85% accuracy
```

`CodeExecutor` + `Router` handle this automatically:
1. Router selects `code_exec` for analytical categories (UCB1 + hard prior)
2. LLM generates Python code from question + raw data
3. Subprocess sandbox executes (8s timeout, /tmp isolation)
4. stdout is the deterministic answer
5. Falls back to `llm_direct` if code fails or returns empty

---

## Benchmarks Supported

- **CRMArenaPro** — 22 task categories, synthetic CRM data, schema drift
- **τ²-Bench** — airline customer service with embedded tool schemas
- **OfficeQA** — plain text Q&A over business documents
- **WebShop+** — online shopping with budget/constraint enforcement
- **ResearchToolBench** — academic/news/technical research with citations
- **CausalRivers** — multivariate time-series causal discovery

---

## Install

```bash
pip install git+https://github.com/abhishec/brainoscorelight.git
```

---

## License

MIT
