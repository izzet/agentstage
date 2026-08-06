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


_DSB_MULTITASK_ANALYSIS_TASKS = (
    "tabular-playground-series-oct-2021",
    "tabular-playground-series-dec-2021",
    "tabular-playground-series-nov-2021",
    "tabular-playground-series-sep-2021",
)

_DSB_MULTITASK_ANALYSIS_INST = (
    "You are doing comparative data-analysis across FOUR tabular Kaggle "
    "competitions. For EACH of the following tasks (each has its own "
    "train.csv and test.csv under /data/<task-name>/), load the train "
    "set via pandas and compute summary statistics, then write a single "
    "consolidated comparison report.\n\n"
    "Tasks (full train.csv path under /data/):\n"
    "  - /data/tabular-playground-series-oct-2021/train.csv\n"
    "  - /data/tabular-playground-series-dec-2021/train.csv\n"
    "  - /data/tabular-playground-series-nov-2021/train.csv\n"
    "  - /data/tabular-playground-series-sep-2021/train.csv\n\n"
    "For each task: load the FULL train.csv via "
    "`pd.read_csv(path)` (do NOT subsample, do NOT use nrows or chunksize), "
    "then record: row count, column count, target-column mean and std "
    "(the target is typically named 'target' or 'claim'), and the "
    "missing-rate of the most-missing column.\n\n"
    "Write the consolidated report to result/analysis_summary.csv with "
    "columns: task, n_rows, n_cols, target_mean, target_std, "
    "max_missing_rate, max_missing_col. One row per task, four rows total.\n\n"
    "Read EVERY train.csv fully; this is a data-quality audit, not a "
    "modeling exercise, so pandas read_csv on each file is the dominant "
    "I/O step. Use one pandas pass per file."
)


def load_dsbench_multitask_analysis() -> DSBWorkload:
    """Custom DSBench workload: combined analysis across 4 tabular tasks.

    Combines ~3.7 GB of CSV data so the I/O share is large enough relative
    to LLM time to actually demonstrate session-level speedup on a
    compute-bound benchmark family.
    """
    log_root = "/data"
    workspace_prior: dict[str, tuple[str, ...]] = {}
    prefix_map_entries: list[tuple[str, str]] = []
    all_files: list[str] = []
    for tid in _DSB_MULTITASK_ANALYSIS_TASKS:
        resplit_dir = DSBENCH_DATA_ROOT / "data_resplit" / tid
        if not resplit_dir.is_dir():
            raise FileNotFoundError(f"data_resplit dir missing: {resplit_dir}")
        train_logical = f"{log_root}/{tid}/train.csv"
        test_logical = f"{log_root}/{tid}/test.csv"
        key_safe = tid.replace("-", "_")
        workspace_prior[f"train_{key_safe}"] = (train_logical,)
        workspace_prior[f"test_{key_safe}"] = (test_logical,)
        all_files.extend([train_logical, test_logical])
        prefix_map_entries.append(
            (f"{log_root}/{tid}/", str(resplit_dir) + "/"))
    workspace_prior["all_files"] = tuple(all_files)
    workspace_prior["output_submission"] = (
        f"{log_root}/result/analysis_summary.csv",)
    task = DSBTask(
        task_id="multitask_analysis",
        description=_DSB_MULTITASK_ANALYSIS_INST,
        train_csv=(DSBENCH_DATA_ROOT / "data_resplit"
                   / _DSB_MULTITASK_ANALYSIS_TASKS[0] / "train.csv").resolve(),
        test_csv=(DSBENCH_DATA_ROOT / "data_resplit"
                  / _DSB_MULTITASK_ANALYSIS_TASKS[0] / "test.csv").resolve(),
        sample_csv=None,
    )
    return DSBWorkload(
        task_id="dsb_multitask_analysis",
        task=task,
        workspace_prior=workspace_prior,
        ground_truth_full=tuple(all_files),
        ground_truth_first_inspect=tuple(all_files),
        prefix_map=tuple(prefix_map_entries),
    )


_DSB_INTEGRITY_MANIFEST_INST = (
    "You are computing a DATA INTEGRITY MANIFEST for four tabular Kaggle "
    "competition datasets staged on a shared filesystem. This is a "
    "standard data-publishing preprocessing step: every file in the "
    "archive must be checksummed so downstream consumers can verify "
    "integrity after transfer.\n\n"
    "Files to checksum (full paths under /data/):\n"
    "  - /data/tabular-playground-series-oct-2021/train.csv\n"
    "  - /data/tabular-playground-series-oct-2021/test.csv\n"
    "  - /data/tabular-playground-series-dec-2021/train.csv\n"
    "  - /data/tabular-playground-series-dec-2021/test.csv\n"
    "  - /data/tabular-playground-series-nov-2021/train.csv\n"
    "  - /data/tabular-playground-series-nov-2021/test.csv\n"
    "  - /data/tabular-playground-series-sep-2021/train.csv\n"
    "  - /data/tabular-playground-series-sep-2021/test.csv\n\n"
    "For EACH file compute the xxh64 hex digest of the ENTIRE file "
    "contents using the `xxhash` package "
    "(`xxhash.xxh64(open(p,'rb').read()).hexdigest()`). Also record the "
    "file size in bytes (via os.path.getsize). Use a streaming approach "
    "if the file is too large for memory: read in 8 MB chunks and update "
    "the hasher.\n\n"
    "Write result/integrity_manifest.csv with columns: dataset, "
    "filename, size_bytes, xxh64. Sort by dataset then filename. "
    "Include all 8 files, no subsampling."
)


_DSB_INTEGRITY_SINGLE_INST = (
    "You are computing a DATA INTEGRITY MANIFEST for the "
    "tabular-playground-series-oct-2021 Kaggle competition dataset. "
    "Standard data-publishing preprocessing step.\n\n"
    "For EACH file under /data/tabular-playground-series-oct-2021/ "
    "(specifically train.csv and test.csv), compute the xxh64 hex "
    "digest of the ENTIRE file contents using the `xxhash` package. "
    "Use a streaming approach: read in 8 MB chunks and update the "
    "hasher (xxhash.xxh64()).update(chunk). Also record the file size "
    "in bytes via os.path.getsize.\n\n"
    "Write result/integrity_manifest.csv with columns: filename, "
    "size_bytes, xxh64. Sort by filename. Include both train.csv and "
    "test.csv."
)


def load_dsbench_integrity_single() -> DSBWorkload:
    """DSBench single-task integrity manifest. Just train.csv + test.csv
    of tabular-playground-series-oct-2021 (~2.2 GB). Few-large-files
    workload — less shim overhead than the 8-file multi-task variant."""
    tid = "tabular-playground-series-oct-2021"
    log_root = "/data"
    resplit_dir = DSBENCH_DATA_ROOT / "data_resplit" / tid
    if not resplit_dir.is_dir():
        raise FileNotFoundError(f"data_resplit dir missing: {resplit_dir}")
    train_logical = f"{log_root}/{tid}/train.csv"
    test_logical = f"{log_root}/{tid}/test.csv"
    workspace_prior: dict[str, tuple[str, ...]] = {
        "train_csv": (train_logical,),
        "test_csv": (test_logical,),
        "all_files": (train_logical, test_logical),
        "output_submission": (f"{log_root}/result/integrity_manifest.csv",),
    }
    task = DSBTask(
        task_id="integrity_single",
        description=_DSB_INTEGRITY_SINGLE_INST,
        train_csv=(resplit_dir / "train.csv").resolve(),
        test_csv=(resplit_dir / "test.csv").resolve(),
        sample_csv=None,
    )
    return DSBWorkload(
        task_id="dsb_integrity_single",
        task=task,
        workspace_prior=workspace_prior,
        ground_truth_full=(train_logical, test_logical),
        ground_truth_first_inspect=(train_logical, test_logical),
        prefix_map=((f"{log_root}/{tid}/", str(resplit_dir) + "/"),),
    )


def load_dsbench_integrity_manifest() -> DSBWorkload:
    """DSBench integrity-manifest task: xxh64 every train/test CSV
    across the 4 tabular-playground tasks. Real third-party benchmark
    data, AgentStage-style I/O-bound task design."""
    log_root = "/data"
    workspace_prior: dict[str, tuple[str, ...]] = {}
    prefix_map_entries: list[tuple[str, str]] = []
    all_files: list[str] = []
    for tid in _DSB_MULTITASK_ANALYSIS_TASKS:
        resplit_dir = DSBENCH_DATA_ROOT / "data_resplit" / tid
        if not resplit_dir.is_dir():
            raise FileNotFoundError(f"data_resplit dir missing: {resplit_dir}")
        train_logical = f"{log_root}/{tid}/train.csv"
        test_logical = f"{log_root}/{tid}/test.csv"
        key_safe = tid.replace("-", "_")
        workspace_prior[f"train_{key_safe}"] = (train_logical,)
        workspace_prior[f"test_{key_safe}"] = (test_logical,)
        all_files.extend([train_logical, test_logical])
        prefix_map_entries.append(
            (f"{log_root}/{tid}/", str(resplit_dir) + "/"))
    workspace_prior["all_files"] = tuple(all_files)
    workspace_prior["output_submission"] = (
        f"{log_root}/result/integrity_manifest.csv",)
    task = DSBTask(
        task_id="integrity_manifest",
        description=_DSB_INTEGRITY_MANIFEST_INST,
        train_csv=(DSBENCH_DATA_ROOT / "data_resplit"
                   / _DSB_MULTITASK_ANALYSIS_TASKS[0] / "train.csv").resolve(),
        test_csv=(DSBENCH_DATA_ROOT / "data_resplit"
                  / _DSB_MULTITASK_ANALYSIS_TASKS[0] / "test.csv").resolve(),
        sample_csv=None,
    )
    return DSBWorkload(
        task_id="dsb_integrity_manifest",
        task=task,
        workspace_prior=workspace_prior,
        ground_truth_full=tuple(all_files),
        ground_truth_first_inspect=tuple(all_files),
        prefix_map=tuple(prefix_map_entries),
    )


def load_dsbench_task(task_id: str) -> DSBWorkload:
    """Load a DSBench data_modeling task by name."""
    # Special-case for the multi-task analysis workload — combines 4 tasks
    # into one session to exercise more cumulative I/O.
    if task_id == "multitask_analysis":
        return load_dsbench_multitask_analysis()
    if task_id == "integrity_manifest":
        return load_dsbench_integrity_manifest()
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


# Curated 3-task slice for full-agentic-loop experiments.
# Selected for I/O-heavy + compute-light profile so the agent's solution
# finishes fast and the I/O cost is the dominant wall-time fraction:
#   - ventilator-pressure-prediction       : 336 MB train,   8 columns
#   - tabular-playground-series-may-2022   : 243 MB train,  33 columns
#   - lmsys-chatbot-arena                  : 141 MB train,   9 columns
# (For the script-only orchestrator, the original heavy-compute picks
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
