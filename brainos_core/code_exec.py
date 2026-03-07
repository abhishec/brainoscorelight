"""CodeExecutor — the key innovation for analytical benchmark tasks.

First principles: some questions need computation, not language understanding.
"Which month had the highest case count?" → Python groupby, not LLM guessing.

Architecture:
1. Claude Sonnet generates Python code to answer the question
2. Subprocess sandbox executes with timeout
3. Caller gets exact stdout or None (caller decides fallback)

This module is domain-agnostic — works for CRM, finance, research, any data.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .executor import ClaudeExecutor


_CODE_GEN_SYSTEM = """\
You are a Python expert solving data analytics questions from structured data.

Write Python code that:
1. Reads from the variable `context_data` (already defined as a string)
2. Parses the data (JSON, CSV, or mixed text — handle all formats gracefully)
3. Computes the exact answer
4. Prints ONLY the final answer — no labels, no explanation, no extra text

Output format rules:
- IDs: print exact ID string (e.g., 005Wt000003NIiTIAW)
- State abbreviations: print 2-letter code (e.g., CA)
- Month names: print full name (e.g., September)
- Names: print exact string match
- Numbers: print plain number
- If multiple candidates, print the single best match

Always wrap code in ```python ... ``` fences.
Import what you need (pandas, json, csv, re, datetime, etc.).
"""


def _extract_code(text: str) -> str:
    """Pull Python code from ```python...``` or bare code."""
    m = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # No fence — heuristic: lines with print( or import
    if "print(" in text or "import " in text:
        return text.strip()
    return ""


def _run_sandbox(code: str, timeout: int = 10) -> str | None:
    """Execute Python code in a subprocess, return stdout or None."""
    fname = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix='.py', mode='w', delete=False, dir='/tmp'
        ) as f:
            f.write(code)
            fname = f.name

        result = subprocess.run(
            ["python3", fname],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = result.stdout.strip()
        return stdout if stdout else None
    except (subprocess.TimeoutExpired, Exception):
        return None
    finally:
        if fname:
            try:
                os.unlink(fname)
            except OSError:
                pass


class CodeExecutor:
    """Generate and execute Python code to answer analytical questions.

    Usage:
        executor = CodeExecutor(claude=ClaudeExecutor(api_key="..."))
        answer = await executor.solve(
            question="Which state has the fastest case closure?",
            data=context_data_string,
            field_aliases={"AssignedAgent": "OwnerId"},
        )
        # Returns "CA" or None if code execution failed
    """

    def __init__(self, claude: "ClaudeExecutor", sandbox_timeout: int = 10):
        self._claude = claude
        self._timeout = sandbox_timeout

    async def solve(
        self,
        question: str,
        data: str,
        category: str = "",
        field_aliases: dict[str, str] | None = None,
        extra_instructions: str = "",
    ) -> str | None:
        """Generate Python code and execute it to answer the question.

        Args:
            question: The analytics question to answer
            data: Raw data string (JSON, CSV, text)
            category: Task category hint (e.g., "monthly_trend_analysis")
            field_aliases: Dict of {renamed_field: original_field} for schema drift
            extra_instructions: Additional instructions for code generation

        Returns:
            The answer string, or None if code execution failed
        """
        alias_note = ""
        if field_aliases:
            pairs = ", ".join(f"{k}={v}" for k, v in field_aliases.items())
            alias_note = f"\nField aliases (renamed → original): {pairs}\nHandle both names."

        user_msg = (
            f"Category: {category}\n"
            f"Question: {question}\n"
            f"{alias_note}"
            f"{extra_instructions}\n\n"
            f"Data:\n{data[:12000]}"
        )

        try:
            raw = await self._claude.call(
                system=_CODE_GEN_SYSTEM,
                user=user_msg,
                max_tokens=1024,
                # Always use Sonnet for code generation — reliability matters more than speed
                force_fast=False,
            )
        except Exception:
            return None

        code = _extract_code(raw)
        if not code:
            return None

        # Prepend the data so `context_data` is available in user code
        full_code = f'context_data = """{data[:12000]}"""\n\n{code}'
        return _run_sandbox(full_code, timeout=self._timeout)
