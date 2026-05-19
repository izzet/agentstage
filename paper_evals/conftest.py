"""Shared fixtures for AgentStage paper evaluations.

Each test_h<N>_*.py file encodes one hypothesis from AGENTSTAGE.md §2. The
assertions in each file verify the quantitative claims from §3 (C1-C10) and
the evaluations from §11.5 (E1-E11) against captured data.

Two campaign types feed this suite:
  - trace-only campaigns: 88+ streaming probes living under --trace-root,
    each with stream.jsonl + summary.json + prediction.json + byte_metrics.json
  - end-to-end staging campaigns: future Ares-testbed runs living under
    --staging-root, each with staging_report.json + io_report.json

Tests record numeric artifacts via the `report` fixture; sessionfinish writes
the aggregated dict to .results/report.json for figure-plotting scripts.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("paper_evals", "AgentStage paper evaluation options")
    group.addoption(
        "--trace-root",
        type=str,
        default="poc/runs",
        help="Root directory containing trace-only probe runs (default: poc/runs)",
    )
    group.addoption(
        "--staging-root",
        type=str,
        default=None,
        help="Root directory containing end-to-end staging runs (default: none — "
             "tests requiring staging data will skip)",
    )
    group.addoption(
        "--ground-truth-root",
        type=str,
        default=None,
        help="Root directory containing static + empirical workload ground truth. "
             "Empirical truths come from completed io_report.json files in the "
             "AgentIOBench corpus (default: resolve from agentstage.workloads).",
    )
    group.addoption(
        "--min-seeds",
        type=int,
        default=3,
        help="Minimum seeds per (task, model, prompt) cell to include in assertions "
             "(default: 3)",
    )
    group.addoption(
        "--rule-library-version",
        type=str,
        default=None,
        help="Expected frozen rule library version. If set, H6/H7 tests assert the "
             "loaded rules match this version (genericity defense).",
    )
    group.addoption(
        "--report-dir",
        type=str,
        default=None,
        help="Directory for JSON report output (default: paper_evals/.results)",
    )


# ---------------------------------------------------------------------------
# Session-scoped path fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def trace_root(request: pytest.FixtureRequest) -> Path:
    p = Path(request.config.getoption("--trace-root"))
    if not p.is_dir():
        pytest.skip(f"--trace-root not found: {p}")
    return p


@pytest.fixture(scope="session")
def staging_root(request: pytest.FixtureRequest) -> Path | None:
    opt = request.config.getoption("--staging-root")
    if opt is None:
        return None
    p = Path(opt)
    if not p.is_dir():
        pytest.skip(f"--staging-root not found: {p}")
    return p


@pytest.fixture(scope="session")
def ground_truth_root(request: pytest.FixtureRequest) -> Path | None:
    opt = request.config.getoption("--ground-truth-root")
    if opt is None:
        return None
    p = Path(opt)
    if not p.is_dir():
        pytest.skip(f"--ground-truth-root not found: {p}")
    return p


@pytest.fixture(scope="session")
def min_seeds(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--min-seeds")


@pytest.fixture(scope="session")
def rule_library_version(request: pytest.FixtureRequest) -> str | None:
    return request.config.getoption("--rule-library-version")


# ---------------------------------------------------------------------------
# Report collector — gathers structured data from tests, writes JSON at end
# ---------------------------------------------------------------------------

class ReportCollector:
    """Accumulates structured data from tests for JSON report output.

    Tests call `report.record(key, data)` to store results that figure-plotting
    scripts will consume. At session end, the dict is written to a single
    JSON file under the report directory.
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def record(self, key: str, data: Any) -> None:
        """Store data under a key. Overwrites if the key already exists."""
        self._data[key] = data

    def append(self, key: str, entry: Any) -> None:
        """Append to a list under a key. Creates the list if needed."""
        self._data.setdefault(key, []).append(entry)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)


@pytest.fixture(scope="session")
def report() -> ReportCollector:
    """Shared report collector. Tests record data; conftest writes it at end."""
    return ReportCollector()


@pytest.fixture(autouse=True, scope="session")
def _register_report_collector(
    request: pytest.FixtureRequest, report: ReportCollector
) -> None:
    """Make the report collector accessible from session hooks."""
    request.config._report_collector = report  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Hooks — capture per-test outcomes, write report at session end
# ---------------------------------------------------------------------------

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if not hasattr(item, "_report"):
        item._report = {}
    item._report[f"{rep.when}_{rep.outcome}"] = True


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    report_dir_opt = session.config.getoption("--report-dir", default=None)
    report_dir = (
        Path(report_dir_opt) if report_dir_opt
        else Path(__file__).parent / ".results"
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    collected = getattr(session.config, "_report_collector", None)
    data = collected.to_dict() if collected is not None else {}

    outcomes: list[dict[str, Any]] = []
    for item in session.items:
        rep = getattr(item, "_report", None)
        if rep and rep.get("call_passed"):
            outcome = "passed"
        elif rep and rep.get("call_failed"):
            outcome = "failed"
        elif rep and rep.get("call_skipped"):
            outcome = "skipped"
        else:
            outcome = "unknown"
        outcomes.append({"nodeid": item.nodeid, "outcome": outcome})

    summary: dict[str, int] = {}
    tr = session.config.pluginmanager.get_plugin("terminalreporter")
    if tr is not None:
        for status in ("passed", "failed", "skipped", "error"):
            summary[status] = len(tr.stats.get(status, []))

    report_blob = {
        "timestamp": datetime.now().isoformat(),
        "exitstatus": exitstatus,
        "summary": summary,
        "tests": outcomes,
        "data": data,
    }

    with (report_dir / "report.json").open("w") as f:
        json.dump(report_blob, f, indent=2, default=str)
