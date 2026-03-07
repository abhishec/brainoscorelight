# BrainOS Core Light

**Shared AI worker primitives powering all BrainOS benchmark agents.**

Every innovation that is common across agents lives here — not duplicated per agent.
Zero-LLM orchestration, 5-layer memory, smart model routing, completion contracts,
sequence hints, RL loop. Build a production benchmark agent in ~40 lines.

---

## Agent Benchmark Status

| Agent | Benchmark | Score | Status |
|-------|-----------|-------|--------|
| [purple-agent-business-process-worker](https://github.com/abhishec/purple-agent-business-process-worker) | τ²-Bench | **3/3 · 100%** | ✅ #1 globally |
| [purple-agent-business-process-worker](https://github.com/abhishec/purple-agent-business-process-worker) | CRMArenaPro | Run 5 in progress (baseline: 20.7%) | 🔄 In progress |
| [purple-agent-finance-worker](https://github.com/abhishec/purple-agent-finance-worker) | OfficeQA | **246/246 · 100%** | ✅ #1 globally |
| [purple-agent-web-worker](https://github.com/abhishec/purple-agent-web-worker) | WebShop+ | claude-sonnet-4-6 · DAAO + contracts | 🚀 Deployed |
| [purple-agent-research-worker](https://github.com/abhishec/purple-agent-research-worker) | ResearchToolBench | claude-sonnet-4-6 · DAAO + contracts | 🚀 Deployed |
| [causalriver-worker](https://github.com/abhishec/causalriver-worker) | CausalRivers | AUROC 0.7843 · #1 on 5/10 datasets | ✅ Running |

All agents install from: `pip install git+https://github.com/abhishec/brainoscorelight.git`

---

## Architecture: Full Cognitive Stack

```
Incoming task
     │
     ├─ PrivacyGuard.check()           ← PII guard, zero API cost, fires first
     ├─ detect_task_format()           ← route to right domain handler
     ├─ DAAO.route(task_text)          ← zero-LLM model selection (Haiku vs Sonnet)
     │
     ▼
Router.select(category)               ← zero-LLM strategy selection
     │
     ├─ hard rule? (privacy → llm_direct always)
     ├─ L4 UCB1 bandit → best (category, strategy) arm
     └─ L3 semantic EMA → cold-start prior
     │
     ▼
Execute chosen strategy:
  code_exec  → LLM generates Python → subprocess sandbox → stdout
  llm_direct → LLM answers from context (DAAO-selected model)
  mcp_tools  → SequenceHints injected → discover MCP tools → LLM tool loop
                  └─ RecoveryCascade.check() on every empty result
     │
     ▼
Completion contracts (post-execution, zero LLM cost):
  CheckoutContract.satisfied()        ← shopping: was checkout tool called?
  CitationContract.satisfied()        ← research: are citations present?
  DepthContract.satisfied()           ← all: is answer substantive?
  MutationContract.any_mutation()     ← BP: were write tools called?
     │
     ▼
Router.reward(answer, strategy)       ← proxy reward, NO LLM judge
     │
     ▼
Brain.record(...)                     ← update all 5 layers atomically
```

---

## Module Map

| Module | Purpose | LLM calls? |
|--------|---------|------------|
| `brain.py` | 5-layer file-based memory | **No** |
| `router.py` | UCB1 strategy selection + proxy reward | **No** |
| `daao.py` | Difficulty-Aware Adaptive Orchestration (model routing) | **No** |
| `contracts.py` | CheckoutContract, CitationContract, DepthContract, MutationContract | **No** |
| `hints.py` | SequenceHints (prefix-based tool directives) + RecoveryCascade | **No** |
| `privacy.py` | PII detection + refusal (<1ms) | **No** |
| `detector.py` | Task format routing (crm / tau2 / officeqa / causal / research) | **No** |
| `executor.py` | Anthropic API wrapper, retries, budget-aware model switching | Yes |
| `code_exec.py` | LLM → Python code → subprocess sandbox → stdout | Yes (code gen only) |
| `mcp.py` | JSON-RPC 2.0 MCP tool discovery + calling | No |
| `memory.py` | RLPrimer — case log, UCB1, RL primer injection | **No** |
| `worker.py` | Abstract base — wires all primitives into `handle()` | Via executor |

**Zero-LLM modules power all orchestration decisions.** LLM is called only for task execution.

---

## DAAO — Difficulty-Aware Adaptive Orchestration

Routes every task to the cheapest capable model before a single token is spent.

```python
from brainos_core import DAAO

daao = DAAO(
    fast_model="claude-haiku-4-5-20251001",
    main_model="claude-sonnet-4-6",
)

model = daao.route("navigate to the settings page")  # → haiku (fast path)
model = daao.route("synthesize findings across 5 research papers")  # → sonnet
```

**Fast path (Haiku):** `"what is"`, `"define"`, `"navigate to"`, `"open"`, `"list"`, tasks ≤12 words with no deep signals.

**Deep path (Sonnet):** `"analyze"`, `"synthesize"`, `"optimize"`, `"research"`, multi-source, portfolio, long tasks.

Routing is deterministic — keyword regex + word count. Zero API cost.

---

## Completion Contracts

Mechanical guarantees that fire after LLM execution. Zero API cost on pass.

```python
from brainos_core import CheckoutContract, CitationContract, DepthContract

# L2: Shopping / booking
if not CheckoutContract.satisfied(tool_calls_made):
    system_prompt += CheckoutContract.directive()   # → force checkout retry

# L3: Research / academic
if not CitationContract.satisfied(answer):
    last_user_turn += CitationContract.directive()  # → force citation retry

# Depth: All domains
if not DepthContract.satisfied(answer):
    system_prompt += DepthContract.directive()      # → force depth retry
```

| Contract | What it checks | Failure action |
|----------|---------------|----------------|
| `CheckoutContract` | Was a checkout/purchase tool called? | Inject forced-checkout directive + retry |
| `CitationContract` | Does academic/news answer contain citations? | Inject citation directive + retry |
| `DepthContract` | Is answer ≥100 chars and substantive? | Inject depth directive + retry |
| `MutationContract` | Were write tools called when task requires it? | Flag incomplete task |

---

## Sequence Hints + Recovery Cascade

```python
from brainos_core import SequenceHints, RecoveryCascade

# Inject sequence directive into system prompt
system_prompt += SequenceHints.directive("shopping")
# → "For shopping tasks, follow: search_/find_ → view_/click_ → add_ → checkout_"

# Automatically append recovery hint to empty tool results
enriched_result = RecoveryCascade.check(tool_result)
# → original result + "[RECOVERY HINT: Try broader search terms...]" if empty
```

Directives use **prefixes**, not tool names — works against any MCP server regardless of naming convention.

| Domain | Sequence |
|--------|---------|
| `shopping` | `search_/find_` → `view_/click_` → `add_` → `checkout_` |
| `booking` | `search_/find_` → `view_/check_` → `select_` → `confirm_/book_` |
| `academic` | `search_/query_` → `get_/fetch_` → `cite_/references_` → synthesize |
| `news` | `search_/find_` → `get_/fetch_` → `verify_/check_` → verdict |
| `technical` | `list_/check_` → `run_/execute_` → `debug_/fix_` → `verify_` |
| `code` | `search_/find_` → `read_/fetch_` → `analyze_/review_` → suggest |

---

## 5-Layer Brain (file-based, survives Docker restarts)

| Layer | Class | What it stores | Persisted? |
|-------|-------|----------------|------------|
| L1 | `WorkingMemory` | Recent session turns (last 10) | No (in-memory) |
| L2 | `EpisodicMemory` | Task outcomes, last 500 episodes, Jaccard recall | ✅ `episodic_memory.json` |
| L3 | `SemanticMemory` | Category→strategy EMA beliefs (α=0.3), field aliases | ✅ `semantic_memory.json` |
| L4 | `StrategicMemory` | UCB1 bandit: counts + Q-values per (category, strategy) | ✅ `strategic_memory.json` |
| L5 | `MetaMemory` | Model success rates, run counts, worst categories | ✅ `meta_memory.json` |

All writes are atomic (write-to-tmp → rename). All reads degrade gracefully on I/O error.
Default path: `/app/brain/` (override with `BRAIN_DIR` env var).

---

## RL Loop — No LLM Judge

```
Task → Router.select() → execute → Router.reward() → Brain.record()
                                        │
                    proxy reward (deterministic, zero API):
                    - code_exec returned output → 0.85
                    - privacy refusal (correct) → 0.90
                    - mcp_tools answer          → 0.75
                    - llm_direct substantive    → 0.65
                    - empty / error             → 0.10
                    - uncertainty signals       → −0.15 penalty
```

UCB1 updates incrementally per (category, strategy) pair. SemanticMemory EMA provides cold-start priors. After ~50 tasks per category the bandit converges; after 500 total episodes, episodic recall surfaces high-quality few-shot examples automatically.

---

## Quick Start

```python
from brainos_core import Brain, Router, DAAO, SequenceHints, RecoveryCascade
from brainos_core import CheckoutContract, CitationContract, DepthContract

# One Brain instance per process — shared across all requests
brain = Brain()
router = Router(brain)
daao = DAAO(fast_model="claude-haiku-4-5-20251001", main_model="claude-sonnet-4-6")

# 1. Route model (zero LLM cost)
model = daao.route(task_text)

# 2. Build system prompt with sequence hints + brain context
system = SequenceHints.directive(domain) + "\n"
system += router.enrich_system_prompt(base_system, task_text, category, session_id)

# 3. Select strategy (zero LLM cost)
strategy = router.select(category)

# 4. Execute (LLM call with DAAO-selected model)
answer = await executor.call(system=system, user=task_text, model=model)

# 5. Recovery cascade on tool results
tool_result = RecoveryCascade.check(raw_tool_result)

# 6. Post-execution contracts (zero LLM cost)
if not DepthContract.satisfied(answer):
    answer = await executor.call(system + DepthContract.directive(), ...)
if not CitationContract.satisfied(answer):
    answer = await executor.call(... + CitationContract.directive(), ...)

# 7. RL update (zero LLM cost)
reward = router.reward(answer, strategy)
router.record(task_text, category, strategy, model, reward, session_id)
```

### Minimal agent (40 lines)

```python
from brainos_core import BrainOSWorker

class MyAgent(BrainOSWorker):
    async def handle_domain(self, task_text, task_category, session_id):
        return await self.executor.call(
            system="You are a helpful agent.",
            user=task_text,
        )

worker = MyAgent(api_key="sk-ant-...", mcp_endpoint="http://green-agent:9009")
answer = await worker.handle(task_text)
```

---

## Install

```bash
pip install git+https://github.com/abhishec/brainoscorelight.git
```

---

## License

MIT
