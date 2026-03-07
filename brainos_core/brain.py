"""brain.py — 5-layer file-based brain for BrainOS agents.

Each layer serves a distinct cognitive function:
  L1 WorkingMemory   — in-session turns (ephemeral, in-memory)
  L2 EpisodicMemory  — recent task outcomes, last 500 (file-based JSON)
  L3 SemanticMemory  — distilled category→strategy beliefs (file-based JSON)
  L4 StrategicMemory — UCB1 bandit per (category, strategy) (file-based JSON)
  L5 MetaMemory      — model/run aggregate stats (file-based JSON)

All file writes are atomic (write-to-tmp then rename) to survive Docker restarts.
All reads degrade gracefully — any file I/O error returns safe defaults.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

_BRAIN_DIR = Path(os.environ.get("BRAIN_DIR", "/app/brain"))
STRATEGIES = ["code_exec", "llm_direct", "mcp_tools"]


# ── Atomic file I/O ───────────────────────────────────────────────────────────

def _write_json(path: Path, data: Any) -> None:
    """Atomic write: temp file → rename. Safe against mid-write crashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.rename(path)
    except Exception:
        pass  # never crash the agent on memory write failure


def _read_json(path: Path, default: Any) -> Any:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default if not callable(default) else default()


# ── Layer 1: Working Memory ───────────────────────────────────────────────────

class WorkingMemory:
    """In-memory session context. Holds last N turns per session_id.

    Not persisted — cleared on process restart. Used to give the
    outer model awareness of the current conversation flow.
    """

    _MAX_TURNS = 10

    def __init__(self) -> None:
        self._sessions: dict[str, list[dict]] = {}

    def push(self, session_id: str, role: str, content: str) -> None:
        turns = self._sessions.setdefault(session_id, [])
        turns.append({"role": role, "content": content[:500], "ts": time.time()})
        if len(turns) > self._MAX_TURNS:
            self._sessions[session_id] = turns[-self._MAX_TURNS:]

    def get(self, session_id: str, max_turns: int = 5) -> list[dict]:
        return self._sessions.get(session_id, [])[-max_turns:]

    def context_str(self, session_id: str, max_turns: int = 3) -> str:
        turns = self.get(session_id, max_turns)
        if not turns:
            return ""
        lines = ["[WORKING MEMORY — recent session context]"]
        for t in turns:
            lines.append(f"  [{t['role']}] {t['content'][:200]}")
        return "\n".join(lines)

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


# ── Layer 2: Episodic Memory ─────────────────────────────────────────────────

class EpisodicMemory:
    """Recent task outcomes. File-based, survives restarts.

    Each episode: {task_summary, category, strategy, model, reward, keywords, hint, ts}
    Recall: Jaccard keyword overlap + category match + recency bias.
    """

    _MAX_EPISODES = 500

    def __init__(self, brain_dir: Path = _BRAIN_DIR) -> None:
        self._path = brain_dir / "episodic_memory.json"

    def _load(self) -> list[dict]:
        return _read_json(self._path, [])

    def record(self, episode: dict) -> None:
        episodes = self._load()
        episodes.append(episode)
        if len(episodes) > self._MAX_EPISODES:
            episodes = episodes[-self._MAX_EPISODES:]
        _write_json(self._path, episodes)

    def recall(
        self,
        task_text: str,
        category: str,
        top_k: int = 3,
        min_reward: float = 0.0,
    ) -> list[dict]:
        """Return top-k most relevant episodes above min_reward."""
        episodes = [e for e in self._load() if e.get("reward", 0) >= min_reward]
        if not episodes:
            return []

        task_words = set(re.findall(r"\b\w{4,}\b", task_text.lower()))

        def _score(ep: dict) -> float:
            kw = set(ep.get("keywords", []))
            overlap = len(task_words & kw) / max(len(task_words | kw), 1)
            cat_bonus = 0.4 if ep.get("category") == category else 0.0
            quality = ep.get("reward", 0.5) * 0.4
            recency = max(0.0, 1.0 - (time.time() - ep.get("ts", 0)) / 86400) * 0.1
            return overlap + cat_bonus + quality + recency

        return sorted(episodes, key=_score, reverse=True)[:top_k]

    def recall_str(self, task_text: str, category: str, top_k: int = 3) -> str:
        episodes = self.recall(task_text, category, top_k=top_k, min_reward=0.5)
        if not episodes:
            return ""
        lines = ["[EPISODIC MEMORY — similar past tasks]"]
        for ep in episodes:
            sym = "✓" if ep.get("reward", 0) >= 0.7 else "~"
            strat = ep.get("strategy", "?")
            model = ep.get("model", "?")[:6]
            hint = ep.get("hint", "")
            summary = ep.get("task_summary", "")[:90]
            lines.append(f"  {sym} [{ep.get('category','?')}|{strat}|{model}] {summary}")
            if hint:
                lines.append(f"     Hint: {hint[:120]}")
        return "\n".join(lines)

    def stats(self) -> dict:
        episodes = self._load()
        if not episodes:
            return {"count": 0}
        rewards = [e.get("reward", 0) for e in episodes]
        return {
            "count": len(episodes),
            "avg_reward": round(sum(rewards) / len(rewards), 3),
            "recent_avg": round(sum(rewards[-20:]) / len(rewards[-20:]), 3),
        }


# ── Layer 3: Semantic Memory ──────────────────────────────────────────────────

class SemanticMemory:
    """Distilled beliefs: category→strategy→EMA reward, field aliases, domain facts.

    Updated via exponential moving average (α=0.3) on each new observation.
    Provides the semantic prior that informs UCB1 initialization.
    """

    _ALPHA = 0.3

    def __init__(self, brain_dir: Path = _BRAIN_DIR) -> None:
        self._path = brain_dir / "semantic_memory.json"

    def _load(self) -> dict:
        d = _read_json(self._path, {})
        d.setdefault("category_strategy_ema", {})
        d.setdefault("field_aliases", {})
        d.setdefault("domain_facts", {})
        return d

    def update(self, category: str, strategy: str, reward: float) -> None:
        data = self._load()
        cat_map = data["category_strategy_ema"].setdefault(category, {})
        current = cat_map.get(strategy, 0.5)
        cat_map[strategy] = (1 - self._ALPHA) * current + self._ALPHA * reward
        _write_json(self._path, data)

    def best_strategy(self, category: str) -> str | None:
        """Return the strategy with highest EMA reward for this category."""
        data = self._load()
        cat_map = data["category_strategy_ema"].get(category, {})
        if not cat_map:
            return None
        return max(cat_map, key=cat_map.get)

    def get_strategy_ema(self, category: str) -> dict[str, float]:
        data = self._load()
        return data["category_strategy_ema"].get(category, {})

    def set_field_aliases(self, aliases: dict[str, str]) -> None:
        data = self._load()
        data["field_aliases"].update(aliases)
        _write_json(self._path, data)

    def get_field_aliases(self) -> dict[str, str]:
        return self._load().get("field_aliases", {})

    def set_fact(self, key: str, value: Any) -> None:
        data = self._load()
        data["domain_facts"][key] = value
        _write_json(self._path, data)

    def get_fact(self, key: str, default: Any = None) -> Any:
        return self._load().get("domain_facts", {}).get(key, default)

    def context_str(self, category: str) -> str:
        aliases = self.get_field_aliases()
        ema = self.get_strategy_ema(category)
        parts: list[str] = []
        if aliases:
            alias_str = ", ".join(f"{k}→{v}" for k, v in list(aliases.items())[:8])
            parts.append(f"[SEMANTIC] Field aliases: {alias_str}")
        if ema:
            ranked = sorted(ema.items(), key=lambda x: -x[1])
            ema_str = ", ".join(f"{s}={v:.2f}" for s, v in ranked)
            parts.append(f"[SEMANTIC] Category '{category}' strategy EMA: {ema_str}")
        return "\n".join(parts)


# ── Layer 4: Strategic Memory (UCB1 bandit) ───────────────────────────────────

class StrategicMemory:
    """Multi-armed UCB1 bandit: one arm per (category, strategy) pair.

    Exploration constant C=1.5 — balances exploitation of known good
    strategies vs exploration of untried ones.

    Bootstrapped from SemanticMemory EMA when count=0 to avoid cold start.
    """

    C = 1.5  # UCB1 exploration constant

    def __init__(self, brain_dir: Path = _BRAIN_DIR) -> None:
        self._path = brain_dir / "strategic_memory.json"

    def _load(self) -> dict:
        d = _read_json(self._path, {})
        d.setdefault("counts", {})
        d.setdefault("values", {})
        d.setdefault("total_pulls", 0)
        return d

    @staticmethod
    def _key(category: str, strategy: str) -> str:
        return f"{category}::{strategy}"

    def best_strategy(
        self,
        category: str,
        candidates: list[str] | None = None,
        semantic: SemanticMemory | None = None,
    ) -> str:
        """UCB1 arm selection. Uses semantic EMA as optimistic init for unseen arms."""
        arms = candidates or STRATEGIES
        data = self._load()
        total = max(data["total_pulls"], 1)

        ema_priors: dict[str, float] = {}
        if semantic:
            ema_priors = semantic.get_strategy_ema(category)

        best_arm, best_score = arms[0], -1.0
        for strategy in arms:
            k = self._key(category, strategy)
            n = data["counts"].get(k, 0)
            # Init Q from semantic EMA if available, else optimistic 0.6
            q_init = ema_priors.get(strategy, 0.6)
            q = data["values"].get(k, q_init)

            if n == 0:
                score = float("inf")  # try all arms at least once
            else:
                score = q + self.C * math.sqrt(math.log(total) / n)

            if score > best_score:
                best_score, best_arm = score, strategy

        return best_arm

    def update(self, category: str, strategy: str, reward: float) -> None:
        data = self._load()
        k = self._key(category, strategy)
        n = data["counts"].get(k, 0) + 1
        q_old = data["values"].get(k, 0.5)
        # Incremental mean
        data["counts"][k] = n
        data["values"][k] = q_old + (reward - q_old) / n
        data["total_pulls"] = data.get("total_pulls", 0) + 1
        _write_json(self._path, data)

    def stats(self) -> dict[str, dict]:
        data = self._load()
        return {
            k: {
                "n": data["counts"].get(k, 0),
                "q": round(data["values"].get(k, 0.5), 3),
            }
            for k in data.get("counts", {})
        }


# ── Layer 5: Meta Memory ──────────────────────────────────────────────────────

class MetaMemory:
    """Aggregate performance stats: model success rates, category totals, run counts.

    Answers questions like:
    - "Which model performs best overall?"
    - "What is our average reward across all tasks?"
    - "Which categories are we failing most often?"
    """

    def __init__(self, brain_dir: Path = _BRAIN_DIR) -> None:
        self._path = brain_dir / "meta_memory.json"

    def _load(self) -> dict:
        d = _read_json(self._path, {})
        d.setdefault("model_stats", {})
        d.setdefault("category_stats", {})
        d.setdefault("run_count", 0)
        d.setdefault("total_reward", 0.0)
        return d

    def record(self, model: str, category: str, strategy: str, reward: float) -> None:
        data = self._load()

        # Model stats
        ms = data["model_stats"].setdefault(model, {"n": 0, "total": 0.0})
        ms["n"] += 1
        ms["total"] += reward
        ms["avg"] = round(ms["total"] / ms["n"], 3)

        # Category stats
        cs = data["category_stats"].setdefault(category, {})
        st = cs.setdefault(strategy, {"n": 0, "total": 0.0})
        st["n"] += 1
        st["total"] += reward
        st["avg"] = round(st["total"] / st["n"], 3)

        data["run_count"] += 1
        data["total_reward"] += reward

        _write_json(self._path, data)

    def best_model(self, candidates: list[str] | None = None) -> str:
        data = self._load()
        ms = data.get("model_stats", {})
        pool = {k: v for k, v in ms.items() if not candidates or k in candidates}
        if not pool:
            return (candidates or ["sonnet"])[0]
        return max(pool, key=lambda m: pool[m].get("avg", 0))

    def worst_categories(self, top_n: int = 5) -> list[tuple[str, float]]:
        """Return categories with lowest average reward — focus improvement here."""
        data = self._load()
        cat_avgs: list[tuple[str, float]] = []
        for cat, strategies in data.get("category_stats", {}).items():
            rewards = [v["avg"] for v in strategies.values() if v.get("n", 0) > 0]
            if rewards:
                cat_avgs.append((cat, sum(rewards) / len(rewards)))
        return sorted(cat_avgs, key=lambda x: x[1])[:top_n]

    def summary(self) -> dict:
        data = self._load()
        n = max(data.get("run_count", 0), 1)
        return {
            "run_count": data.get("run_count", 0),
            "avg_reward": round(data.get("total_reward", 0.0) / n, 3),
            "model_stats": data.get("model_stats", {}),
            "worst_categories": self.worst_categories(3),
        }


# ── Composite Brain ───────────────────────────────────────────────────────────

class Brain:
    """5-layer composite brain. One instance per process — share across requests.

    Usage:
        brain = Brain()   # or Brain(brain_dir="/custom/path")

        # In task handler:
        ctx = brain.build_context(task_text, category, session_id)
        strategy = brain.recommend_strategy(category)

        # After task:
        brain.record(task_text, category, strategy, model, reward, session_id)
    """

    def __init__(self, brain_dir: str | Path = _BRAIN_DIR) -> None:
        d = Path(brain_dir)
        self.working = WorkingMemory()          # L1: in-memory
        self.episodic = EpisodicMemory(d)       # L2: file
        self.semantic = SemanticMemory(d)       # L3: file
        self.strategic = StrategicMemory(d)     # L4: file
        self.meta = MetaMemory(d)              # L5: file

    def recommend_strategy(
        self,
        category: str,
        candidates: list[str] | None = None,
    ) -> str:
        """Ask L4 (UCB1) for best strategy, informed by L3 (semantic EMA)."""
        return self.strategic.best_strategy(
            category,
            candidates=candidates,
            semantic=self.semantic,
        )

    def build_context(
        self,
        task_text: str,
        category: str,
        session_id: str,
    ) -> str:
        """Build enriched context from all relevant brain layers.

        Returns a string to inject into the system prompt.
        """
        parts: list[str] = []

        # L1: recent session context
        working = self.working.context_str(session_id)
        if working:
            parts.append(working)

        # L2: episodic recall
        episodic = self.episodic.recall_str(task_text, category)
        if episodic:
            parts.append(episodic)

        # L3: semantic context (aliases + EMA)
        semantic = self.semantic.context_str(category)
        if semantic:
            parts.append(semantic)

        return "\n\n".join(parts)

    def record(
        self,
        task_text: str,
        category: str,
        strategy: str,
        model: str,
        reward: float,
        session_id: str = "",
        hint: str = "",
    ) -> None:
        """Update all persistent layers (L2–L5) after task completion."""
        keywords = list(set(re.findall(r"\b\w{5,}\b", task_text.lower())))[:15]

        self.episodic.record({
            "task_summary": task_text[:120],
            "category": category,
            "strategy": strategy,
            "model": model,
            "reward": round(reward, 3),
            "keywords": keywords,
            "hint": hint,
            "ts": time.time(),
        })
        self.semantic.update(category, strategy, reward)
        self.strategic.update(category, strategy, reward)
        self.meta.record(model, category, strategy, reward)

        if session_id:
            self.working.push(
                session_id,
                "brain",
                f"[{strategy}|{model[:6]}] reward={reward:.2f}",
            )

    def diagnostics(self) -> dict:
        """Return full brain state for debugging/monitoring."""
        return {
            "episodic": self.episodic.stats(),
            "strategic": self.strategic.stats(),
            "meta": self.meta.summary(),
        }
