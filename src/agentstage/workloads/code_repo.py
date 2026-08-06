"""code_repo workload — coding agent on the AgentIOBench source tree.

A "find the dispatch bug" task with static ground truth: runner.py +
tools.py are the immediate-need files; llm.py is NOT relevant per the
reading of 14 failing seeds' thinking texts.

The repo we point at is `$CODE_REPO_ROOT` (defaults to the AIOB
submodule under `external/benchmarks/agentiobench/`). The workspace
prior is built by walking the source tree at load time.

Ported from `poc/probe_reasoning_slack.py` on 2026-05-19.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from agentstage.workloads.aiob import TaskConfig, Workload

_THIS_FILE = Path(__file__).resolve()
DEFAULT_CODE_REPO_ROOT = (
    _THIS_FILE.parent.parent.parent.parent / "external" / "benchmarks" / "agentiobench"
)


def code_repo_root() -> Path:
    """Root of the codebase the code_repo task operates on. Override via
    $CODE_REPO_ROOT (e.g. point at an alternate repo for evaluation)."""
    return Path(os.environ.get("CODE_REPO_ROOT", str(DEFAULT_CODE_REPO_ROOT)))


CODE_REPO_FILE_EXTENSIONS = (".py", ".yaml", ".sh", ".md")
CODE_REPO_TOP_BUCKETS = {
    "agentiobench": ("core_runner", "core_tools", "core_llm", "core_other"),
    "paper_evals": ("tests",),
    "scripts": ("scripts_dir",),
    "dftracer": ("tracing_files",),
}


def load_code_repo() -> Workload:
    """Build a Workload by walking the source tree at $CODE_REPO_ROOT."""
    real_base = code_repo_root()
    logical_base = "/repo"

    buckets: dict[str, list[str]] = {
        "core_runner": [], "core_tools": [], "core_llm": [],
        "core_other": [], "config_files": [], "tests": [],
        "scripts_dir": [], "tracing_files": [], "all_files": [],
    }

    for top in CODE_REPO_TOP_BUCKETS:
        top_dir = real_base / top
        if not top_dir.is_dir():
            continue
        for root, _, files in os.walk(top_dir):
            if "__pycache__" in root or ".git" in root:
                continue
            for fname in files:
                if not fname.endswith(CODE_REPO_FILE_EXTENSIONS):
                    continue
                full = Path(root) / fname
                rel = full.relative_to(real_base)
                rel_str = str(rel)
                logical = f"{logical_base}/{rel_str}"
                buckets["all_files"].append(logical)
                # bucket by file role
                if rel_str == "agentiobench/runner.py":
                    buckets["core_runner"].append(logical)
                elif rel_str == "agentiobench/tools.py":
                    buckets["core_tools"].append(logical)
                elif rel_str == "agentiobench/llm.py":
                    buckets["core_llm"].append(logical)
                elif rel_str.startswith("agentiobench/config/"):
                    buckets["config_files"].append(logical)
                elif rel_str.startswith("agentiobench/"):
                    buckets["core_other"].append(logical)
                elif rel_str.startswith("paper_evals/"):
                    buckets["tests"].append(logical)
                elif rel_str.startswith("scripts/"):
                    buckets["scripts_dir"].append(logical)
                elif rel_str.startswith("dftracer/"):
                    buckets["tracing_files"].append(logical)

    workspace_prior: dict[str, tuple[str, ...]] = {
        **{k: tuple(v) for k, v in buckets.items()},
        "output_fix": ("/repo/result/fix.md",),
        "output_report": ("/repo/result/report.md",),
    }

    # GT: runner.py + tools.py are the relevant files (per §6.4 PoC reading).
    gt_full = workspace_prior["core_runner"] + workspace_prior["core_tools"]
    gt_first = gt_full  # immediate need == full set for this task

    # Synthetic TaskConfig — code_repo is not an AIOB task.
    task = TaskConfig(
        name="code_repo_fix_double_write",
        instance_id=0,
        domain="Coding / software engineering",
        dataset_subdir="",  # not under AGENTIOBENCH_DATA_ROOT
        task_inst="Find the cause of write_file being called twice; propose a one-line fix.",
        output_fname="result/fix.md",
        io_tier="light",
    )

    return Workload(
        task_id="code_repo",
        task=task,
        workspace_prior=workspace_prior,
        ground_truth_full=gt_full,
        ground_truth_first_inspect=gt_first,
        prefix_map=((f"{logical_base}/", f"{real_base}/"),),
    )
