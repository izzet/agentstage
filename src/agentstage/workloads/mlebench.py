"""MLE-bench (OpenAI 2024) competition loader for AgentStage experiments.

Loads competitions prepared via `mlebench prepare -c <id>`, which writes
to `<data-dir>/<competition>/prepared/public/`. Files vary by competition
(images, csv, audio, zip), so the loader inspects the public dir and
classifies files into workspace_prior buckets by extension/role.

Used by the full-agentic-loop runner (mlebench_multiturn.py) and the
script-only orchestrator (mlebench_e2e.py). The Workload shape mirrors
agentstage.workloads.{aiob,dsbench,kramabench} so SessionDetector +
AutoRuleGenerator + Stager work unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

MLEBENCH_DATA_ROOT = Path(os.environ.get(
    "MLEBENCH_DATA_ROOT",
    Path(__file__).resolve().parents[3] / "outputs" / "mlebench-data",
))


@dataclass(frozen=True)
class MLETask:
    competition_id: str
    description: str
    public_dir: Path

    @property
    def task_inst(self) -> str:
        """AIOB-style task_inst for AutoRuleGenerator format-token scanning."""
        return self.description


@dataclass(frozen=True)
class MLEWorkload:
    task_id: str
    task: MLETask
    workspace_prior: dict[str, tuple[str, ...]]
    ground_truth_full: tuple[str, ...]
    ground_truth_first_inspect: tuple[str, ...]
    prefix_map: tuple[tuple[str, str], ...]

    @property
    def all_workspace_paths(self) -> tuple[str, ...]:
        return tuple(p for paths in self.workspace_prior.values() for p in paths)


def _bucket_for_file(fname: str) -> str | None:
    """Map a filename to a workspace_prior bucket key (or None to skip).

    Buckets follow AIOB's `{class}_{instance}` convention so the
    AutoRuleGenerator emits sensible per-class regex patterns.
    """
    lower = fname.lower()
    if lower == "description.md":
        return None  # excluded — task spec, not data
    if lower == "sample_submission.csv":
        return "sample_csv"
    if lower == "train.csv":
        return "train_csv"
    if lower == "test.csv":
        return "test_csv"
    if lower.endswith(".csv"):
        return "extra_csv"
    if lower.endswith(".zip"):
        if "train" in lower:
            return "train_zip"
        if "test" in lower:
            return "test_zip"
        return "data_zip"
    if lower.endswith((".tsv", ".parquet", ".feather", ".jsonl",
                        ".npy", ".npz", ".h5", ".hdf5")):
        return "extra_data"
    if lower.endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff",
                        ".wav", ".mp3", ".flac")):
        return "media"
    return None  # silently skip txt readmes, etc.


def load_mle_competition(competition_id: str) -> MLEWorkload:
    """Load a prepared MLE-bench competition by id."""
    comp_dir = MLEBENCH_DATA_ROOT / competition_id / "prepared" / "public"
    if not comp_dir.is_dir():
        raise FileNotFoundError(
            f"prepared/public dir missing: {comp_dir}\n"
            f"Run: mlebench prepare -c {competition_id} "
            f"--data-dir {MLEBENCH_DATA_ROOT}"
        )

    description_file = comp_dir / "description.md"
    description = (description_file.read_text(errors="replace")
                   if description_file.is_file() else "")

    log_root = f"/data/{competition_id}"

    # Build workspace_prior by walking the public dir
    buckets: dict[str, list[str]] = {}
    for root, _, files in os.walk(comp_dir):
        for fname in files:
            bucket = _bucket_for_file(fname)
            if bucket is None:
                continue
            phys = Path(root) / fname
            rel = phys.relative_to(comp_dir)
            logical = f"{log_root}/{rel}"
            buckets.setdefault(bucket, []).append(logical)

    workspace_prior: dict[str, tuple[str, ...]] = {
        k: tuple(v) for k, v in buckets.items()
    }
    workspace_prior["all_files"] = tuple(
        p for v in workspace_prior.values() for p in v
    )
    workspace_prior["output_submission"] = (f"{log_root}/submission.csv",)

    # Ground truth = every public input file (a complete solution
    # reads at least the labels + data archives).
    gt = workspace_prior["all_files"]

    prefix_map = ((log_root + "/", str(comp_dir.resolve()) + "/"),)

    task = MLETask(
        competition_id=competition_id,
        description=description,
        public_dir=comp_dir.resolve(),
    )

    return MLEWorkload(
        task_id=f"mle_{competition_id}",
        task=task,
        workspace_prior=workspace_prior,
        ground_truth_full=gt,
        ground_truth_first_inspect=gt,
        prefix_map=prefix_map,
    )


# Curated 3-competition slice for full-agentic E-041 experiments.
# Picked for I/O-heavy + compute-light profile:
#   - histopathologic-cancer-detection : 7 GB, 220K image files (massive open-fanout)
#   - new-york-city-taxi-fare-prediction : 5 GB tabular CSV (sequential read)
#   - dogs-vs-cats-redux-kernels-edition : 800 MB image classification (mid-fanout)
MLE_AGENTIC_SLICE = (
    "histopathologic-cancer-detection",
    "new-york-city-taxi-fare-prediction",
    "dogs-vs-cats-redux-kernels-edition",
)

# Smoke-test only (too small for I/O headline but pipeline-validation)
MLE_SMOKE_SLICE = (
    "aerial-cactus-identification",
)
