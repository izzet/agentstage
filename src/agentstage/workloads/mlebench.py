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


def _has_zip_for(dir_name: str, top_level_files: set[str]) -> bool:
    """Returns True if a sibling .zip file with a matching name exists.
    E.g. train/ + train.zip → True; test/ + test.zip → True."""
    return f"{dir_name}.zip" in top_level_files


def load_mle_competition(competition_id: str) -> MLEWorkload:
    """Load a prepared MLE-bench competition by id.

    Workspace-prior policy:
      - Always include top-level files (description.md, *.csv, *.zip).
      - For subdirectories: include each file IF the dir has < 2000 files
        AND no sibling .zip exists. Rationale:
          * If a sibling .zip exists, the agent naturally reads from the
            zip — staging the unpacked dir is redundant work.
          * If the unpacked dir has > 2000 files, enumerating each one
            into the prior makes the Stager prefetch impractical (each
            file copied via thread pool). Better to skip the dir entirely
            and let the script either read from a non-existent zip
            (graceful error → agent rewrites) or hit the cold tier
            per-file (the failure case we report honestly).
    """
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

    # First, enumerate top-level entries to decide which subdirs to skip
    top_level_files: set[str] = set()
    for entry in comp_dir.iterdir():
        if entry.is_file():
            top_level_files.add(entry.name)

    # Decide which subdirectories to include based on file count + zip presence
    SUBDIR_MAX_FILES = 2000
    skip_subdirs: set[str] = set()
    for entry in comp_dir.iterdir():
        if not entry.is_dir():
            continue
        n_files = sum(1 for _ in entry.iterdir())
        has_zip = _has_zip_for(entry.name, top_level_files)
        if has_zip:
            # Use the zip; skip the unpacked dir to avoid double-staging
            skip_subdirs.add(entry.name)
        elif n_files > SUBDIR_MAX_FILES:
            # Too many files for individual prefetch — skip this dir.
            # Honest tradeoff: agent's script will hit cold tier per-file
            # if it iterates these. For our experiments we should pick
            # competitions where this isn't the dominant access pattern.
            skip_subdirs.add(entry.name)

    # Build workspace_prior by walking the public dir, honoring skip_subdirs
    buckets: dict[str, list[str]] = {}
    for root, dirs, files in os.walk(comp_dir):
        # Mutate dirs in place to skip subdirectories we've excluded
        rel_root = Path(root).relative_to(comp_dir)
        if rel_root == Path("."):
            dirs[:] = [d for d in dirs if d not in skip_subdirs]
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


# Curated slice for full-agentic E-041 experiments — picked for
# I/O-heavy + tractable-prefetch profile:
#   - new-york-city-taxi-fare-prediction : 5.3 GB single labels.csv (perfect)
#   - dogs-vs-cats-redux-kernels-edition  : 490 MB train.zip + 54 MB test.zip
#     (zips ship alongside unpacked dirs; loader uses zips to bound prefetch)
MLE_AGENTIC_SLICE = (
    "new-york-city-taxi-fare-prediction",
    "dogs-vs-cats-redux-kernels-edition",
)

# Edge case for the paper's honest-negative section. histopathologic ships
# 220,025 individual TIF files with no zip alternative. Per-file staging
# can't fit inside the agent's reasoning slack; even bulk `cp -r` takes
# 76s. The loader skips the unpacked dirs (SUBDIR_MAX_FILES gate), so only
# train_labels.csv + sample_submission.csv get staged. Result: agent
# scripts that iterate over individual images hit the cold tier file-by-
# file regardless of mode → AgentStage shows little/no speedup. We
# include this run honestly to bound the regime where AgentStage helps.
MLE_NEGATIVE_SLICE = (
    "histopathologic-cancer-detection",
)

# Smoke-test only (too small for I/O headline but pipeline-validation)
MLE_SMOKE_SLICE = (
    "aerial-cactus-identification",
)
