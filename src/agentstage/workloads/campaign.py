"""Campaign indexer — walks an outputs root, parses per-run summary.json
files, and exposes a queryable view of all runs.

A `Campaign` is the queryable index. A `RunResult` wraps one run's
output directory (summary.json, stream.jsonl, prediction.json,
byte_metrics_v1.json) with lazy property accessors.

Used by paper_evals tests to iterate cells: filter by task, model,
provider, prompt mode, turn, presence of thinking content, etc.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path


@dataclass
class RunResult:
    """One run's output directory + lazy parsed metadata."""

    run_dir: Path

    # ----- Lazy file access -----

    @cached_property
    def _summary(self) -> dict:
        return json.loads((self.run_dir / "summary.json").read_text())

    @cached_property
    def byte_metrics_v1(self) -> dict | None:
        """v1-rescored byte metrics, if `rescore_run` has been run on this dir."""
        p = self.run_dir / "byte_metrics_v1.json"
        return json.loads(p.read_text()) if p.is_file() else None

    @cached_property
    def prediction_v1(self) -> dict | None:
        p = self.run_dir / "prediction_v1.json"
        return json.loads(p.read_text()) if p.is_file() else None

    # ----- Summary fields -----

    @cached_property
    def task(self) -> str:
        return self._summary.get("task") or ""

    @cached_property
    def model(self) -> str:
        return self._summary.get("model") or ""

    @cached_property
    def provider(self) -> str:
        return self._summary.get("provider") or ""

    @cached_property
    def provider_family(self) -> str:
        """One of: 'anthropic' / 'gemini' / 'openrouter' / 'oss' / ''."""
        p = self.provider.lower()
        m = self.model.lower()
        if "anthropic" in p or "claude" in m or "azure" in p and "anthropic" in m:
            return "anthropic"
        if "gemini" in p or "google" in p or "gemini" in m:
            return "gemini"
        if "openrouter" in p or "deepseek" in m:
            return "openrouter"
        return "oss"

    @cached_property
    def turn(self) -> int:
        return int(self._summary.get("turn") or 1)

    @cached_property
    def seed(self) -> int:
        return int(self._summary.get("seed") or 0)

    @cached_property
    def planning_prompt(self) -> bool:
        return bool(self._summary.get("planning_prompt"))

    @cached_property
    def thinking_budget(self) -> int:
        return int(self._summary.get("thinking_budget") or 0)

    @cached_property
    def via_azure(self) -> bool:
        return bool(self._summary.get("via_azure"))

    @cached_property
    def wall_ms(self) -> float | None:
        return self._summary.get("wall_ms")

    @cached_property
    def blocks(self) -> list[dict]:
        return self._summary.get("blocks") or []

    # ----- Derived signals -----

    @cached_property
    def has_thinking(self) -> bool:
        """True iff any thinking block has non-zero text."""
        for b in self.blocks:
            if b.get("type") == "thinking" and (b.get("text_len") or 0) > 0:
                return True
        return False

    @cached_property
    def slack_ms(self) -> float | None:
        """First-thinking-chunk → first-tool-use-block latency. None if
        the agent never emitted a tool_use or never thought."""
        first_thinking_t: float | None = None
        first_tool_t: float | None = None
        for b in self.blocks:
            t = b.get("t_first")
            if b.get("type") == "thinking" and first_thinking_t is None:
                first_thinking_t = t
            if b.get("type") == "tool_use" and first_tool_t is None:
                first_tool_t = t
                break
        if first_thinking_t is None or first_tool_t is None:
            return None
        return first_tool_t - first_thinking_t

    @cached_property
    def thinking_chars(self) -> int:
        return sum(
            int(b.get("text_len") or 0)
            for b in self.blocks
            if b.get("type") == "thinking"
        )

    @cached_property
    def is_well_defined(self) -> bool:
        """False for aiob_101 (structural edge case — no single first file
        by design; §TLDR excludes it from headline byte recall)."""
        return self.task != "aiob_101"


# ---------------------------------------------------------------------------
# Campaign — collection of RunResults with filtering
# ---------------------------------------------------------------------------

@dataclass
class Campaign:
    """Collection of RunResult records, queryable by task / model / etc."""

    runs: list[RunResult] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.runs)

    def __iter__(self):
        return iter(self.runs)

    def filter(
        self,
        *,
        task: str | None = None,
        exclude_tasks: Iterable[str] = (),
        model: str | None = None,
        provider_family: str | None = None,
        planning_prompt: bool | None = None,
        turn: int | None = None,
        well_defined_only: bool = False,
        with_thinking: bool | None = None,
        has_byte_metrics_v1: bool | None = None,
    ) -> Campaign:
        excl = set(exclude_tasks)

        def keep(r: RunResult) -> bool:
            if task is not None and r.task != task:
                return False
            if r.task in excl:
                return False
            if model is not None and model not in r.model:
                return False
            if provider_family is not None and r.provider_family != provider_family:
                return False
            if planning_prompt is not None and r.planning_prompt != planning_prompt:
                return False
            if turn is not None and r.turn != turn:
                return False
            if well_defined_only and not r.is_well_defined:
                return False
            if with_thinking is True and not r.has_thinking:
                return False
            if with_thinking is False and r.has_thinking:
                return False
            if has_byte_metrics_v1 is True and r.byte_metrics_v1 is None:
                return False
            if has_byte_metrics_v1 is False and r.byte_metrics_v1 is not None:
                return False
            return True

        return Campaign(runs=[r for r in self.runs if keep(r)])

    def group_by(self, *keys: str) -> dict[tuple, list[RunResult]]:
        """Group runs by tuples of attribute values."""
        out: dict[tuple, list[RunResult]] = {}
        for r in self.runs:
            k = tuple(getattr(r, key) for key in keys)
            out.setdefault(k, []).append(r)
        return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_campaign(outputs_root: Path) -> Campaign:
    """Walk `outputs_root` and load every directory that contains a
    `summary.json`. Descends one level (so `outputs/poc/<dirname>/`
    works alongside `outputs/<dirname>/`)."""
    root = Path(outputs_root)
    if not root.is_dir():
        return Campaign(runs=[])

    runs: list[RunResult] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if (path / "summary.json").is_file():
            runs.append(RunResult(run_dir=path))
            continue
        # One level of nesting (e.g. outputs/poc/<run_dirs>)
        for sub in sorted(path.iterdir()):
            if sub.is_dir() and (sub / "summary.json").is_file():
                runs.append(RunResult(run_dir=sub))
    return Campaign(runs=runs)
