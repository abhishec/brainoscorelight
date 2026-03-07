"""RLPrimer — lightweight RL-from-experience without retraining.

Innovation from agent-purple: after each task, record what worked.
Before each task, inject the 3 most similar past successes as examples.
Zero additional training. Zero API cost (reads from a local JSON file).

Architecture: UCB1 bandit over (task_category, strategy) pairs.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Outcome:
    task_category: str
    task_snippet: str  # first 100 chars for similarity matching
    strategy: str      # "code_exec" | "llm_direct" | "mcp_tools"
    success: bool
    score: float       # 0.0–1.0
    timestamp: float = field(default_factory=time.time)


class RLPrimer:
    """Learn from past task outcomes and inject relevant examples.

    Usage:
        primer = RLPrimer(cache_path="~/.brainos/rl_primer.json")
        primer.record(outcome)

        # Before next task:
        examples = primer.build_prompt(task_text, task_category)
        # Inject examples into system prompt
    """

    def __init__(self, cache_path: str | Path = "~/.brainos/rl_primer.json"):
        self.cache_path = Path(cache_path).expanduser()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._outcomes: list[Outcome] = []
        self._load()

    def _load(self) -> None:
        if self.cache_path.exists():
            try:
                data = json.loads(self.cache_path.read_text())
                self._outcomes = [Outcome(**o) for o in data]
            except Exception:
                self._outcomes = []

    def _save(self) -> None:
        try:
            data = [o.__dict__ for o in self._outcomes[-500:]]  # keep last 500
            self.cache_path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def record(self, outcome: Outcome) -> None:
        """Record a task outcome for future learning."""
        self._outcomes.append(outcome)
        self._save()

    def build_prompt(self, task_text: str, task_category: str, top_k: int = 3) -> str:
        """Build a primer string with the most relevant past successes.

        Returns empty string if no relevant history.
        """
        relevant = [
            o for o in self._outcomes
            if o.success and o.task_category == task_category
        ]
        if not relevant:
            return ""

        # Sort by score descending, take top_k
        relevant.sort(key=lambda o: o.score, reverse=True)
        examples = relevant[:top_k]

        lines = [f"[LEARNED PATTERNS for {task_category}]"]
        for i, ex in enumerate(examples, 1):
            lines.append(
                f"{i}. Strategy '{ex.strategy}' succeeded (score={ex.score:.2f}): "
                f"{ex.task_snippet[:80]}..."
            )
        return "\n".join(lines)

    def best_strategy(self, task_category: str) -> str:
        """UCB1 bandit: pick the best strategy for this category."""
        strategies = ["code_exec", "llm_direct", "mcp_tools"]
        counts = {s: 0 for s in strategies}
        rewards = {s: 0.0 for s in strategies}

        for o in self._outcomes:
            if o.task_category == task_category:
                counts[o.strategy] += 1
                rewards[o.strategy] += o.score

        total = sum(counts.values())
        if total == 0:
            return "code_exec"  # default: try code execution first

        ucb_scores = {}
        for s in strategies:
            if counts[s] == 0:
                ucb_scores[s] = float("inf")
            else:
                avg = rewards[s] / counts[s]
                exploration = math.sqrt(2 * math.log(total) / counts[s])
                ucb_scores[s] = avg + exploration

        return max(ucb_scores, key=ucb_scores.get)
