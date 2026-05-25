"""DSBench (Jing et al., ICLR 2025) data_modeling workload loader.

Each task is a Kaggle/ModelOff competition with pre-split CSV data under
`outputs/dsbench-data/data_modeling/data/data_resplit/<task>/` and a
human-readable task description under `.../data/task/<task>.txt`.

Files per task:
    data_resplit/<task>/train.csv
    data_resplit/<task>/test.csv
    data_resplit/<task>/sample_submission.csv

Used by the full-agentic-loop runner (dsbench_multiturn.py) and the
script-only orchestrator (dsbench_e2e.py). The Workload shape mirrors
agentstage.workloads.aiob.Workload and .kramabench.KramaWorkload so the
SessionDetector + AutoRuleGenerator + Stager all see a uniform interface.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DSBENCH_DATA_ROOT = Path(os.environ.get(
    "DSBENCH_DATA_ROOT",
    Path(__file__).resolve().parents[3] / "outputs" / "dsbench-data"
    / "data_modeling" / "data",
))


@dataclass(frozen=True)
class DSBTask:
    """One DSBench data_modeling task."""

    task_id: str        # e.g. "santander-value-prediction-challenge"
    description: str    # contents of task/<task_id>.txt
    train_csv: Path
    test_csv: Path
    sample_csv: Path | None

    @property
    def task_inst(self) -> str:
        """AIOB-style task_inst for AutoRuleGenerator."""
        return self.description


@dataclass(frozen=True)
class DSBWorkload:
    """DSBench analog of AIOB's Workload."""

    task_id: str
    task: DSBTask
    workspace_prior: dict[str, tuple[str, ...]]
    ground_truth_full: tuple[str, ...]
    ground_truth_first_inspect: tuple[str, ...]
    prefix_map: tuple[tuple[str, str], ...]

    @property
    def all_workspace_paths(self) -> tuple[str, ...]:
        return tuple(p for paths in self.workspace_prior.values() for p in paths)


def _logical_root(task_id: str) -> str:
    return f"/data/{task_id}"


def load_dsbench_task(task_id: str) -> DSBWorkload:
    """Load a DSBench data_modeling task by name."""
    resplit_dir = DSBENCH_DATA_ROOT / "data_resplit" / task_id
    task_file = DSBENCH_DATA_ROOT / "task" / f"{task_id}.txt"
    if not resplit_dir.is_dir():
        raise FileNotFoundError(f"data_resplit dir missing: {resplit_dir}")
    if not task_file.is_file():
        raise FileNotFoundError(f"task description missing: {task_file}")

    train_csv = resplit_dir / "train.csv"
    test_csv = resplit_dir / "test.csv"
    sample_csv = resplit_dir / "sample_submission.csv"
    sample_csv = sample_csv if sample_csv.is_file() else None

    description = task_file.read_text(errors="replace")

    task = DSBTask(
        task_id=task_id,
        description=description,
        train_csv=train_csv.resolve(),
        test_csv=test_csv.resolve(),
        sample_csv=sample_csv.resolve() if sample_csv else None,
    )

    log_root = _logical_root(task_id)

    # Workspace prior buckets — one per file class plus aggregates.
    # Keys follow AIOB's `{class}_{instance}` convention so AutoRuleGenerator
    # emits sensible per-class regexes.
    workspace_prior: dict[str, tuple[str, ...]] = {
        "train_csv":  (f"{log_root}/train.csv",),
        "test_csv":   (f"{log_root}/test.csv",),
    }
    if sample_csv:
        workspace_prior["sample_csv"] = (f"{log_root}/sample_submission.csv",)
    workspace_prior["all_files"] = tuple(
        p for v in workspace_prior.values() for p in v
    )
    workspace_prior["output_submission"] = (f"{log_root}/submission.csv",)

    # Ground truth: any solution will read train+test (+ sample_submission).
    gt = (f"{log_root}/train.csv", f"{log_root}/test.csv")
    if sample_csv:
        gt = gt + (f"{log_root}/sample_submission.csv",)

    # Logical → physical: /data/<task>/foo.csv → .../data_resplit/<task>/foo.csv
    prefix_map = ((log_root + "/", str(resplit_dir) + "/"),)

    return DSBWorkload(
        task_id=f"dsb_{task_id}",
        task=task,
        workspace_prior=workspace_prior,
        ground_truth_full=gt,
        ground_truth_first_inspect=gt,
        prefix_map=prefix_map,
    )


# Curated 3-task slice for FULL-AGENTIC-LOOP experiments (E-040).
# Selected for I/O-heavy + compute-light profile so the agent's solution
# finishes fast and the I/O cost is the dominant wall-time fraction:
#   - ventilator-pressure-prediction       : 336 MB train,   8 columns
#   - tabular-playground-series-may-2022   : 243 MB train,  33 columns
#   - lmsys-chatbot-arena                  : 141 MB train,   9 columns
# (For the script-only orchestrator E-039, the original heavy-compute picks
#  santander/tabular-feb/microsoft-malware remain in scripts/microbench/dsbench_e2e.py.)
DSB_AGENTIC_SLICE = (
    "ventilator-pressure-prediction",
    "tabular-playground-series-may-2022",
    "lmsys-chatbot-arena",
)

# Original slice for script-only E2E (heavy compute, big files)
DSB_E2E_SLICE = (
    "santander-value-prediction-challenge",
    "tabular-playground-series-feb-2022",
    "microsoft-malware-prediction",
)
