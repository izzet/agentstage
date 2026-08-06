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


_MLE_INTEGRITY_MANIFEST_INST = (
    "You are computing a DATA INTEGRITY MANIFEST for the NYC taxi-fare "
    "competition dataset (MLE-bench / Kaggle). This is a standard "
    "data-publishing preprocessing step: every CSV file in the public "
    "release must be checksummed for downstream verification.\n\n"
    "For EACH file under /data/new-york-city-taxi-fare-prediction/ that "
    "ends in .csv (recursively across all subdirectories), compute the "
    "xxh64 hex digest of the ENTIRE file contents using the `xxhash` "
    "package (`xxhash.xxh64(open(p,'rb').read()).hexdigest()` for small "
    "files, or stream-update for large ones in 8MB chunks). Also record "
    "the file size in bytes (via os.path.getsize).\n\n"
    "Write result/integrity_manifest.csv with columns: relpath, "
    "size_bytes, xxh64. Sort by relpath. Include every CSV file in the "
    "archive (no subsampling). The dataset contains roughly 3-5 CSV "
    "files including the bulk train/test data."
)


def load_mle_integrity_manifest() -> MLEWorkload:
    """MLE-bench integrity-manifest task: xxh64 every CSV in the NYC
    taxi competition. AgentStage-style I/O-bound task on real
    third-party benchmark data."""
    competition_id = "new-york-city-taxi-fare-prediction"
    comp_dir = MLEBENCH_DATA_ROOT / competition_id / "prepared" / "public"
    if not comp_dir.is_dir():
        raise FileNotFoundError(
            f"prepared/public dir missing: {comp_dir}")
    log_root = f"/data/{competition_id}"
    workspace_prior: dict[str, tuple[str, ...]] = {}
    all_files: list[str] = []
    for entry in sorted(comp_dir.iterdir()):
        if entry.is_file() and entry.suffix.lower() == ".csv":
            logical = f"{log_root}/{entry.name}"
            key = entry.stem.replace("-", "_").replace(".", "_")
            workspace_prior[f"csv_{key}"] = (logical,)
            all_files.append(logical)
    workspace_prior["all_files"] = tuple(all_files)
    workspace_prior["output_submission"] = (
        f"{log_root}/result/integrity_manifest.csv",)
    task = MLETask(
        competition_id=competition_id,
        description=_MLE_INTEGRITY_MANIFEST_INST,
        public_dir=comp_dir.resolve(),
    )
    prefix_map = ((log_root + "/", str(comp_dir.resolve()) + "/"),)
    return MLEWorkload(
        task_id=f"mle_integrity_manifest",
        task=task,
        workspace_prior=workspace_prior,
        ground_truth_full=tuple(all_files),
        ground_truth_first_inspect=tuple(all_files),
        prefix_map=prefix_map,
    )


_MLE_DOGSVCATS_INTEGRITY_INST = (
    "You are computing a DATA INTEGRITY MANIFEST for the dogs-vs-cats "
    "Kaggle competition dataset. This is a standard data-publishing "
    "preprocessing step: every archive file must be checksummed for "
    "downstream verification.\n\n"
    "For EACH file under "
    "/data/dogs-vs-cats-redux-kernels-edition/ that ends in .zip, "
    ".csv, or .md (top-level only), compute the xxh64 hex digest of "
    "the ENTIRE file contents using the `xxhash` package "
    "(`xxhash.xxh64(open(p,'rb').read()).hexdigest()` for small "
    "files, or stream-update in 8 MB chunks for large ones). Also "
    "record the file size in bytes via os.path.getsize.\n\n"
    "Write result/integrity_manifest.csv with columns: relpath, "
    "size_bytes, xxh64. Sort by relpath. Include every file at the "
    "top level (the dataset includes train.zip, test.zip, "
    "sample_submission.csv, and description.md). Do NOT recurse into "
    "the train/ or test/ directories — use the zips."
)


def load_mle_dogsvcats_integrity() -> MLEWorkload:
    """MLE-bench dogs-vs-cats: xxh64 the 2 zip files + small CSVs.
    ~544 MB total. AgentStage-style I/O-bound task on real MLE-bench
    data (different competition from NYC taxi for cross-task coverage)."""
    competition_id = "dogs-vs-cats-redux-kernels-edition"
    comp_dir = MLEBENCH_DATA_ROOT / competition_id / "prepared" / "public"
    if not comp_dir.is_dir():
        raise FileNotFoundError(
            f"prepared/public dir missing: {comp_dir}")
    log_root = f"/data/{competition_id}"
    workspace_prior: dict[str, tuple[str, ...]] = {}
    all_files: list[str] = []
    # Just top-level files: .zip, .csv, .md
    for entry in sorted(comp_dir.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in (".zip", ".csv", ".md"):
            continue
        logical = f"{log_root}/{entry.name}"
        key = entry.stem.replace("-", "_").replace(".", "_")
        bucket = ("train_zip" if "train" in entry.name.lower()
                  else "test_zip" if "test" in entry.name.lower()
                  else "extra_file")
        workspace_prior.setdefault(bucket, []).append(logical)
        all_files.append(logical)
    workspace_prior = {k: tuple(v) for k, v in workspace_prior.items()}
    workspace_prior["all_files"] = tuple(all_files)
    workspace_prior["output_submission"] = (
        f"{log_root}/result/integrity_manifest.csv",)
    task = MLETask(
        competition_id=competition_id,
        description=_MLE_DOGSVCATS_INTEGRITY_INST,
        public_dir=comp_dir.resolve(),
    )
    prefix_map = ((log_root + "/", str(comp_dir.resolve()) + "/"),)
    return MLEWorkload(
        task_id="mle_dogsvcats_integrity",
        task=task,
        workspace_prior=workspace_prior,
        ground_truth_full=tuple(all_files),
        ground_truth_first_inspect=tuple(all_files),
        prefix_map=prefix_map,
    )


_MLE_DOGSVCATS_THUMBHASH_INST = (
    "You are computing a per-image integrity manifest for the dogs-vs-cats "
    "Kaggle competition's TRAINING SET (the already-extracted train/ "
    "directory, not the train.zip archive). Every cat.*.jpg and dog.*.jpg "
    "file in the train directory needs an xxh64 digest for a downstream "
    "image-dedup pipeline.\n\n"
    "For EACH file under /data/dogs-vs-cats-redux-kernels-edition/train/ "
    "(every cat.N.jpg and dog.N.jpg, 25000 files total), compute the xxh64 "
    "hex digest of the file's bytes using the `xxhash` package "
    "(`xxhash.xxh64(open(p,'rb').read()).hexdigest()`). Also record the "
    "file size via os.path.getsize.\n\n"
    "Write result/train_thumbnail_manifest.csv with columns: filename, "
    "size_bytes, xxh64. Sort by filename. Use a simple os.listdir loop "
    "over the train/ directory; do NOT recurse into subdirectories."
)


def load_mle_dogsvcats_thumbhash() -> MLEWorkload:
    """MLE-bench dogs-vs-cats THUMBNAIL HASH task: xxh64 every .jpg in the
    extracted train/ directory (25000 files, ~537 MB total). Designed to
    stress many-small-file prefetch — the workload that AgentStage's
    bulk-stage-to-/dev/shm pattern targets most directly. The bucket cap
    must be raised via STAGER_BUCKET_CAP env var (e.g. 30000) for the
    full corpus to be prefetched."""
    competition_id = "dogs-vs-cats-redux-kernels-edition"
    comp_dir = MLEBENCH_DATA_ROOT / competition_id / "prepared" / "public"
    if not comp_dir.is_dir():
        raise FileNotFoundError(
            f"prepared/public dir missing: {comp_dir}")
    train_dir = comp_dir / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"train/ dir missing (run zip extraction first): {train_dir}")
    log_root = f"/data/{competition_id}"
    workspace_prior: dict[str, tuple[str, ...]] = {}
    cat_files: list[str] = []
    dog_files: list[str] = []
    for entry in sorted(train_dir.iterdir()):
        if not entry.is_file() or entry.suffix.lower() != ".jpg":
            continue
        logical = f"{log_root}/train/{entry.name}"
        if entry.name.startswith("cat."):
            cat_files.append(logical)
        elif entry.name.startswith("dog."):
            dog_files.append(logical)
    workspace_prior["train_cats"] = tuple(cat_files)
    workspace_prior["train_dogs"] = tuple(dog_files)
    workspace_prior["all_train_images"] = tuple(cat_files + dog_files)
    workspace_prior["output_manifest"] = (
        f"{log_root}/result/train_thumbnail_manifest.csv",)
    task = MLETask(
        competition_id=competition_id,
        description=_MLE_DOGSVCATS_THUMBHASH_INST,
        public_dir=comp_dir.resolve(),
    )
    prefix_map = ((log_root + "/", str(comp_dir.resolve()) + "/"),)
    return MLEWorkload(
        task_id="mle_dogsvcats_thumbhash",
        task=task,
        workspace_prior=workspace_prior,
        ground_truth_full=tuple(cat_files + dog_files),
        ground_truth_first_inspect=tuple(cat_files[:50]),
        prefix_map=prefix_map,
    )


_MLE_HISTOPATH_THUMBHASH_INST = (
    "You are computing a per-image integrity manifest for the histopathologic "
    "cancer detection Kaggle competition's TEST SET (the already-extracted "
    "test/ directory of .tif files). Every test image needs an xxh64 digest "
    "for a downstream patch-dedup pipeline before model inference.\n\n"
    "For EACH .tif file under /data/histopathologic-cancer-detection/test/ "
    "(45561 files total, ~28 KB each, ~1.3 GB total), compute the xxh64 hex "
    "digest using `xxhash.xxh64(open(p,'rb').read()).hexdigest()`. Also "
    "record the file size via os.path.getsize.\n\n"
    "Write result/test_thumbnail_manifest.csv with columns: filename, "
    "size_bytes, xxh64. Sort by filename. Use a simple os.listdir loop over "
    "the test/ directory; do NOT recurse into subdirectories."
)


def load_mle_histopath_thumbhash() -> MLEWorkload:
    """MLE-bench histopathologic-cancer-detection THUMBHASH task: xxh64 every
    .tif in the extracted test/ directory (45561 files, ~1.3 GB). Many-small-
    files workload pattern, 2× the file count of dogvscats. Requires
    STAGER_BUCKET_CAP=50000 or higher to prefetch the full corpus."""
    competition_id = "histopathologic-cancer-detection"
    comp_dir = MLEBENCH_DATA_ROOT / competition_id / "prepared" / "public"
    if not comp_dir.is_dir():
        raise FileNotFoundError(
            f"prepared/public dir missing: {comp_dir}")
    test_dir = comp_dir / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(
            f"test/ dir missing (run zip extraction first): {test_dir}")
    log_root = f"/data/{competition_id}"
    workspace_prior: dict[str, tuple[str, ...]] = {}
    tif_files: list[str] = []
    for entry in sorted(test_dir.iterdir()):
        if entry.is_file() and entry.suffix.lower() == ".tif":
            tif_files.append(f"{log_root}/test/{entry.name}")
    workspace_prior["all_test_images"] = tuple(tif_files)
    workspace_prior["output_manifest"] = (
        f"{log_root}/result/test_thumbnail_manifest.csv",)
    task = MLETask(
        competition_id=competition_id,
        description=_MLE_HISTOPATH_THUMBHASH_INST,
        public_dir=comp_dir.resolve(),
    )
    prefix_map = ((log_root + "/", str(comp_dir.resolve()) + "/"),)
    return MLEWorkload(
        task_id="mle_histopath_thumbhash",
        task=task,
        workspace_prior=workspace_prior,
        ground_truth_full=tuple(tif_files),
        ground_truth_first_inspect=tuple(tif_files[:50]),
        prefix_map=prefix_map,
    )


def load_mle_competition_dispatch(task_id: str) -> MLEWorkload:
    """Top-level loader: dispatch to integrity-manifest task or fall back
    to the standard MLE-bench competition loader."""
    if task_id == "integrity_manifest":
        return load_mle_integrity_manifest()
    if task_id == "dogsvcats_integrity":
        return load_mle_dogsvcats_integrity()
    if task_id == "dogsvcats_thumbhash":
        return load_mle_dogsvcats_thumbhash()
    if task_id == "histopath_thumbhash":
        return load_mle_histopath_thumbhash()
    return load_mle_competition(task_id)


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
    # SUBDIR_MAX_FILES caps how many files in a single subdir we'll enumerate
    # into workspace_prior. Override via env so experiments with big image
    # directories (histopath = 220k PNGs) can opt in to per-file prefetching
    # while preserving the default safety net for unintended runs.
    SUBDIR_MAX_FILES = int(os.environ.get("MLEBENCH_SUBDIR_MAX_FILES", "2000"))
    SUBDIR_SAMPLE_CAP = os.environ.get("MLEBENCH_SUBDIR_SAMPLE_CAP")
    sample_cap = int(SUBDIR_SAMPLE_CAP) if SUBDIR_SAMPLE_CAP else None
    skip_subdirs: set[str] = set()
    truncate_subdirs: dict[str, int] = {}
    for entry in comp_dir.iterdir():
        if not entry.is_dir():
            continue
        n_files = sum(1 for _ in entry.iterdir())
        has_zip = _has_zip_for(entry.name, top_level_files)
        if has_zip:
            # Use the zip; skip the unpacked dir to avoid double-staging
            skip_subdirs.add(entry.name)
        elif n_files > SUBDIR_MAX_FILES:
            if sample_cap and sample_cap < n_files:
                # Enumerate a deterministic prefix of files; agent's task
                # prompt must direct workload onto this subset.
                truncate_subdirs[entry.name] = sample_cap
            else:
                # Too many files for individual prefetch — skip this dir.
                skip_subdirs.add(entry.name)

    # Build workspace_prior by walking the public dir, honoring skip_subdirs
    buckets: dict[str, list[str]] = {}
    # Per-subdir counter for truncation
    truncated_seen: dict[str, int] = {n: 0 for n in truncate_subdirs}
    for root, dirs, files in os.walk(comp_dir):
        # Mutate dirs in place to skip subdirectories we've excluded
        rel_root = Path(root).relative_to(comp_dir)
        if rel_root == Path("."):
            dirs[:] = [d for d in dirs if d not in skip_subdirs]
        # Determine which top-level subdir this root belongs to (for trunc)
        rel_str = str(rel_root)
        top_subdir = rel_str.split("/", 1)[0] if rel_str != "." else None
        for fname in sorted(files):
            if (top_subdir and top_subdir in truncate_subdirs
                    and truncated_seen[top_subdir]
                    >= truncate_subdirs[top_subdir]):
                continue
            bucket = _bucket_for_file(fname)
            if bucket is None:
                continue
            if top_subdir and top_subdir in truncate_subdirs:
                truncated_seen[top_subdir] += 1
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


# Curated slice for full-agentic experiments — picked for
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
