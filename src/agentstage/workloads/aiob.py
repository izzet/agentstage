"""AgentIOBench workload definitions for AgentStage.

For each AIOB workload we use end-to-end (aiob_101 / aiob_104 / aiob_107 /
aiob_110), this module produces a `Workload` value carrying:

- The AIOB `TaskConfig` (loaded from the submodule's YAML) — gives us
  `dataset_subdir`, `output_fname`, `task_inst`, validator, etc.
- A bucketed **workspace prior** (`dict[str, tuple[str, ...]]`) whose keys
  match the `target_keys` referenced by `agentstage.detector.rules`.
- A **static ground truth** in two flavors: `ground_truth_full` (eventual
  working set) and `ground_truth_first_inspect` (immediate need).
- A **prefix map** translating logical paths (`/data/<task>/raw/...`) used
  in the workspace prior to real on-disk paths under
  `$AGENTIOBENCH_DATA_ROOT/<dataset_subdir>/...`.

The workspace prior + GT come from us (AIOB doesn't carry that concept);
the TaskConfig fields come from AIOB.

`TaskConfig` is replicated locally instead of imported from `agentiobench`
because AIOB's `__init__.py` eagerly imports `runner` → `validation` →
numpy, which we don't want as a runtime dependency on Day 1. When T13b
lands the `feat/agentstage-integration` branch with a lazy `__init__` +
public re-export, switch the import to:

    from agentiobench import TaskConfig

Ported from `poc/probe_reasoning_slack.py` on 2026-05-19.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from agentstage.detector.rules import AIOB_104_SAMPLES, AIOB_110_SUBJECTS

# ---------------------------------------------------------------------------
# Curated task instructions (AgentStage-authored bulk-read tasks on AIOB data)
# ---------------------------------------------------------------------------
# We pair upstream AIOB datasets with our own task descriptions that target
# standard scientific workflows (cohort-wide coverage QC, diurnal-cycle
# climatology, population PSTH) — each naturally requires bulk reads of
# the underlying data without index-aware slicing shortcuts.

_CURATED_TASK_INST: dict[str, str] = {
    "aiob_104": (
        "Compute the per-base read-depth histogram for chromosome 20 across all "
        "50 GBR samples in the IGSR Phase 3 exome subset.\n\n"
        "For each sample's chr20-restricted BAM, tabulate read coverage at every "
        "position on chromosome 20 and aggregate into a histogram with bins "
        "[0, 1, 2, ..., 99, >=100].\n\n"
        "Save:\n"
        "1. result/chr20_coverage_histogram.parquet — one row per (sample_id, "
        "coverage_bin) with columns sample_id, coverage_bin (int 0-100 where 100 "
        "means >=100), n_bases.\n"
        "2. result/per_sample_summary.parquet — one row per sample with columns "
        "sample_id, mean_coverage, median_coverage, n_bases_at_min_1x, "
        "n_bases_at_min_20x, chr20_length_bp.\n"
        "3. result/report.md — cohort, method, total positions analyzed, "
        "aggregate coverage statistics.\n"
    ),
    "aiob_107": (
        "Compute the global hourly diurnal cycle of brightness temperature for "
        "each band across the GOES-16 ABI-L2-CMIPC 7-day bundle "
        "(2024-05-01 through 2024-05-07 UTC, bands 08, 09, 10).\n\n"
        "For each NetCDF file: read the CMI (brightness temperature) and DQF "
        "(data quality flag) arrays, mask pixels where DQF != 0, and compute "
        "the spatial mean over valid CONUS pixels. Parse band and UTC timestamp "
        "from the filename (GOES ABI standard naming). Aggregate by "
        "(band, hour-of-day) across the full 7-day record.\n\n"
        "Save:\n"
        "1. result/goes_global_hourly.parquet — band, hour_utc, "
        "mean_brightness_temp_K, std_brightness_temp_K, n_files, "
        "n_total_valid_pixels.\n"
        "2. result/goes_global_hourly_diurnal.png — multi-panel figure showing "
        "the diurnal cycle per band.\n"
        "3. result/report.md — bands, file count per band, quality-filter "
        "statistics, method.\n"
    ),
    "aiob_110": (
        "Compute the population peri-stimulus time histogram (PSTH) across all "
        "spike-sorted units in all 39 sessions of the Steinmetz Neuropixels "
        "dataset (DANDI 000017).\n\n"
        "For each session, for each unit, bin its spike_times at 10 ms "
        "resolution in a [-0.5 s, +1.5 s] window around each trial's stimulus "
        "onset and average across trials to obtain a unit-level PSTH. Map units "
        "to brain regions via the electrodes table (each unit's peak_channel "
        "links to an electrode with a location field).\n\n"
        "Save:\n"
        "1. result/population_psth.parquet — session_path, subject, unit_idx, "
        "brain_region, time_bin_ms (-500..1490 in 10 ms steps), mean_rate_hz "
        "(trial-averaged).\n"
        "2. result/session_summary.parquet — session_path, subject, "
        "n_units_total, n_trials, total_spike_count, recording_duration_s, "
        "unique_brain_regions (semicolon-joined).\n"
        "3. result/report.md — sessions processed, unit/trial counts, brain "
        "region coverage, method.\n"
    ),
}


def _curate_task(task: "TaskConfig", task_id: str) -> "TaskConfig":
    """Override task_inst with the curated bulk-read instruction if defined."""
    if task_id not in _CURATED_TASK_INST:
        return task
    return replace(task, task_inst=_CURATED_TASK_INST[task_id])

# ---------------------------------------------------------------------------
# Paths and env
# ---------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
AIOB_SUBMODULE_ROOT = _THIS_FILE.parent.parent.parent.parent / "external" / "benchmarks" / "agentiobench"
TASK_YAML_DIR = AIOB_SUBMODULE_ROOT / "agentiobench" / "config" / "task"

DEFAULT_DATA_ROOT = "/mnt/common/datasets-staging/agentiobench/datasets"


def data_root() -> Path:
    """Root of the AIOB dataset tree. Override via $AGENTIOBENCH_DATA_ROOT."""
    return Path(os.environ.get("AGENTIOBENCH_DATA_ROOT", DEFAULT_DATA_ROOT))


# ---------------------------------------------------------------------------
# TaskConfig — field-compatible mirror of agentiobench.config.TaskConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskConfig:
    """Field-compatible mirror of agentiobench.config.TaskConfig.

    Avoids importing `agentiobench` directly (its __init__ eagerly pulls
    numpy). Replace with `from agentiobench import TaskConfig` after the
    feat/agentstage-integration branch lands the public re-export.
    """

    name: str
    instance_id: int
    domain: str
    dataset_subdir: str
    task_inst: str
    output_fname: str
    io_tier: str = ""
    dataset_folder_tree: str = ""
    dataset_preview: str = ""
    gold_program_name: str = ""
    eval_script_name: str = ""

    @classmethod
    def from_yaml(cls, path: Path) -> TaskConfig:
        raw = yaml.safe_load(path.read_text())
        fields = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in fields})

    @property
    def dataset_root(self) -> Path:
        return data_root() / self.dataset_subdir


def _task_yaml(stem: str) -> Path:
    matches = sorted(TASK_YAML_DIR.glob(f"{stem}_*.yaml"))
    if not matches:
        raise FileNotFoundError(
            f"No AIOB task YAML matching {stem}_* under {TASK_YAML_DIR}. "
            "Did you `git submodule update --init`?"
        )
    return matches[0]


# ---------------------------------------------------------------------------
# Workload definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Workload:
    """An AgentStage workload: AIOB task + workspace prior + GT + prefix map."""

    task_id: str
    task: TaskConfig
    workspace_prior: dict[str, tuple[str, ...]]
    ground_truth_full: tuple[str, ...]
    ground_truth_first_inspect: tuple[str, ...]
    prefix_map: tuple[tuple[str, str], ...]
    """Logical→real prefix pairs for resolving paths during byte-size lookup
    and stager dispatch. Format: ((logical_prefix, real_prefix), ...) —
    matched in order; first hit wins."""

    @property
    def all_workspace_paths(self) -> tuple[str, ...]:
        return tuple(p for paths in self.workspace_prior.values() for p in paths)


# ---------------------------------------------------------------------------
# aiob_101 — ERA5 climate (36 monthly NetCDFs + shapefile)
# ---------------------------------------------------------------------------

def load_aiob_101() -> Workload:
    task = TaskConfig.from_yaml(_task_yaml("aiob_101"))
    workspace_prior = {
        "input_netcdfs": tuple(
            f"/data/era5_heatwave/raw/era5_single_levels_{y}_{m:02d}.nc"
            for y in (2022, 2023, 2024) for m in range(1, 13)
        ),
        "input_shapefile": ("/data/era5_heatwave/raw/cb_2023_us_county_500k.zip",),
        "output_stage_a_zarr": ("/output/stage_a/daily_wetbulb_rechunked.zarr",),
        "output_chunk_schema": ("/output/stage_a/chunk_schema.json",),
        "output_result_zarr": ("/output/result/daily_wetbulb.zarr",),
        "output_events_parquet": ("/output/result/heatwave_events.parquet",),
        "output_report_md": ("/output/result/report.md",),
    }
    gt_full = (
        *workspace_prior["input_netcdfs"],
        *workspace_prior["input_shapefile"],
        *workspace_prior["output_stage_a_zarr"],
    )
    return Workload(
        task_id="aiob_101",
        task=task,
        workspace_prior=workspace_prior,
        ground_truth_full=gt_full,
        ground_truth_first_inspect=gt_full,  # structural edge case — no single first
        prefix_map=(("/data/era5_heatwave/", f"{data_root()}/{task.dataset_subdir}/"),),
    )


# ---------------------------------------------------------------------------
# aiob_103 — Sentinel-2 NDVI mosaics (16 scenes × 5 GeoTIFFs each)
# ---------------------------------------------------------------------------
# I/O profile: every band TIF is fully read by rasterio (libgdal C backend),
# tests the LD_PRELOAD shim across a C subprocess code path. Compute is
# vectorized numpy on float32 arrays.

_AIOB_103_BANDS: tuple[str, ...] = ("B02", "B03", "B04", "B08", "SCL")


def _aiob_103_month(scene_name: str) -> str:
    """Extract YYYY-MM from scene name like S2A_T10SFG_20240612T185531_L2A."""
    parts = scene_name.split("_")
    for p in parts:
        if len(p) >= 8 and p[:8].isdigit():
            return f"{p[:4]}-{p[4:6]}"
    return "unknown"


def load_aiob_103() -> Workload:
    task = _curate_task(TaskConfig.from_yaml(_task_yaml("aiob_103")), "aiob_103")
    real_base = data_root() / task.dataset_subdir / "raw"
    logical_base = "/data/sentinel2_ndvi/raw"

    scenes: list[str] = []
    per_scene: dict[str, list[str]] = {}
    per_band: dict[str, list[str]] = {f"band_{b.lower()}": [] for b in _AIOB_103_BANDS}
    per_month: dict[str, list[str]] = {}
    all_files: list[str] = []

    if real_base.is_dir():
        for scene_dir in sorted(real_base.iterdir()):
            if not scene_dir.is_dir():
                continue
            scene_name = scene_dir.name
            scenes.append(scene_name)
            triplet: list[str] = []
            for f in sorted(scene_dir.iterdir()):
                if f.suffix.lower() not in (".tif", ".tiff"):
                    continue
                logical = f"{logical_base}/{scene_name}/{f.name}"
                triplet.append(logical)
                all_files.append(logical)
                for b in _AIOB_103_BANDS:
                    if f.name.startswith(b):
                        per_band[f"band_{b.lower()}"].append(logical)
                        break
            per_scene[f"scene_{scene_name}"] = triplet
            mkey = _aiob_103_month(scene_name)
            per_month.setdefault(f"month_{mkey.replace('-', '_')}", []).extend(triplet)

    workspace_prior: dict[str, tuple[str, ...]] = {
        **{k: tuple(v) for k, v in per_scene.items()},
        **{k: tuple(v) for k, v in per_band.items()},
        **{k: tuple(v) for k, v in per_month.items()},
        "all_scenes": tuple(all_files),
        "first_scene": tuple(per_scene[f"scene_{scenes[0]}"]) if scenes else (),
        "output_ndvi_jun": ("/output/result/ndvi_2024_06.tif",),
        "output_ndvi_jul": ("/output/result/ndvi_2024_07.tif",),
        "output_ndvi_aug": ("/output/result/ndvi_2024_08.tif",),
        "output_figure": ("/output/result/sentinel2_monthly_ndvi.png",),
        "output_report": ("/output/result/report.md",),
    }

    return Workload(
        task_id="aiob_103",
        task=task,
        workspace_prior=workspace_prior,
        ground_truth_full=workspace_prior["all_scenes"],
        ground_truth_first_inspect=workspace_prior["first_scene"],
        prefix_map=((f"{logical_base}/", f"{real_base}/"),),
    )


# ---------------------------------------------------------------------------
# aiob_104 — IGSR genomics (50 GBR samples × BAM/BAI/BAS + reference)
# ---------------------------------------------------------------------------

def load_aiob_104() -> Workload:
    task = _curate_task(TaskConfig.from_yaml(_task_yaml("aiob_104")), "aiob_104")
    real_base = data_root() / task.dataset_subdir / "raw"
    logical_base = "/data/igsr_coverage_qc/raw"

    workspace_prior: dict[str, tuple[str, ...]] = {
        "reference": (
            f"{logical_base}/reference/20130108.exome.targets.bed",
            f"{logical_base}/reference/human_g1k_v37.fasta.fai",
            f"{logical_base}/reference/README.human_g1k_v37.fasta.txt",
        ),
    }
    for s in AIOB_104_SAMPLES:
        d = real_base / "data" / s / "exome_alignment"
        triplet: list[str] = []
        if d.is_dir():
            for f in sorted(p.name for p in d.iterdir()):
                triplet.append(f"{logical_base}/data/{s}/exome_alignment/{f}")
        workspace_prior[f"sample_{s}"] = tuple(triplet)

    workspace_prior["all_samples"] = tuple(
        p for s in AIOB_104_SAMPLES for p in workspace_prior.get(f"sample_{s}", ())
    )
    workspace_prior["output_histogram"] = ("/output/result/chr20_coverage_histogram.parquet",)
    workspace_prior["output_summary"] = ("/output/result/per_sample_summary.parquet",)
    workspace_prior["output_report"] = ("/output/result/report.md",)

    gt_full = workspace_prior["all_samples"] + workspace_prior["reference"]
    # Sonnet inspects a BAM sample first (HG00096); Gemini inspects reference.
    # See AGENTSTAGE.md §6.4.1 — model-strategy variance.
    gt_first = workspace_prior.get("sample_HG00096", ()) + workspace_prior["reference"]

    return Workload(
        task_id="aiob_104",
        task=task,
        workspace_prior=workspace_prior,
        ground_truth_full=gt_full,
        ground_truth_first_inspect=gt_first,
        prefix_map=((f"{logical_base}/", f"{real_base}/"),),
    )


# ---------------------------------------------------------------------------
# aiob_107 — GOES meteorology (6042 NetCDFs across 3 bands × 7 days × 24 hours)
# ---------------------------------------------------------------------------

def load_aiob_107_s3(
    s3_mount: str | Path = "/tmp/s3-noaa-goes16",
    s3_prefix: str = "ABI-L2-CMIPC",
) -> Workload:
    """S3 variant of aiob_107. Sources GOES data directly from NOAA's
    public Open Data bucket (`s3://noaa-goes16/`) via a mountpoint-s3
    mount. Workspace_prior reuses the logical paths from the local
    aiob_107 — the detector sees identical `/data/goes_cmi_composites/raw/...`
    paths in thinking text — but the prefix_map maps them onto the
    bucket's native `ABI-L2-CMIPC/YYYY/DDD/HH/...` layout.

    Prereq:
        mount-s3 --no-sign-request --read-only --region us-east-1 \
            noaa-goes16 /tmp/s3-noaa-goes16

    For tasks where the local AIOB data may not exist (e.g. an
    Ares clone without the staged GOES corpus), this loader still
    works because workspace_prior is reused from load_aiob_107's
    logical path list — falls back to a small hard-coded probe set
    if local AIOB data is absent.
    """
    # Reuse local aiob_107's workspace_prior structure (logical paths).
    # The detector matches against text mentions of files; what changes
    # for the S3 variant is purely the physical-path mapping.
    local = load_aiob_107()
    s3_mount = Path(s3_mount)

    return Workload(
        task_id="aiob_107_s3",
        task=local.task,
        workspace_prior=local.workspace_prior,
        ground_truth_full=local.ground_truth_full,
        ground_truth_first_inspect=local.ground_truth_first_inspect,
        # /data/goes_cmi_composites/raw/  ->  /tmp/s3-noaa-goes16/ABI-L2-CMIPC/
        prefix_map=(
            ("/data/goes_cmi_composites/raw/", f"{s3_mount}/{s3_prefix}/"),
        ),
    )


def load_aiob_107() -> Workload:
    task = _curate_task(TaskConfig.from_yaml(_task_yaml("aiob_107")), "aiob_107")
    real_base = data_root() / task.dataset_subdir / "raw"
    logical_base = "/data/goes_cmi_composites/raw"

    by_band: dict[str, list[str]] = {"band_08": [], "band_09": [], "band_10": []}
    by_day: dict[str, list[str]] = {f"day_{d}": [] for d in range(122, 129)}
    all_files: list[str] = []

    if real_base.is_dir():
        for year_dir in sorted(real_base.iterdir()):
            if not year_dir.is_dir():
                continue
            for day_dir in sorted(year_dir.iterdir()):
                if not day_dir.is_dir():
                    continue
                day_key = f"day_{day_dir.name}"
                for hour_dir in sorted(day_dir.iterdir()):
                    if not hour_dir.is_dir():
                        continue
                    for f in sorted(hour_dir.iterdir()):
                        if f.suffix != ".nc":
                            continue
                        logical = f"{logical_base}/{year_dir.name}/{day_dir.name}/{hour_dir.name}/{f.name}"
                        all_files.append(logical)
                        if "M6C08" in f.name:
                            by_band["band_08"].append(logical)
                        elif "M6C09" in f.name:
                            by_band["band_09"].append(logical)
                        elif "M6C10" in f.name:
                            by_band["band_10"].append(logical)
                        if day_key in by_day:
                            by_day[day_key].append(logical)

    workspace_prior: dict[str, tuple[str, ...]] = {
        **{k: tuple(v) for k, v in by_band.items()},
        **{k: tuple(v) for k, v in by_day.items()},
        "all_files": tuple(all_files),
        "first_file": tuple(sorted(all_files)[:1]),
        "first_hour_all_bands": tuple(f for f in all_files if "/122/00/" in f),
        "output_hourly": ("/output/result/goes_global_hourly.parquet",),
        "output_diurnal": ("/output/result/goes_global_hourly_diurnal.png",),
        "output_report": ("/output/result/report.md",),
    }

    return Workload(
        task_id="aiob_107",
        task=task,
        workspace_prior=workspace_prior,
        ground_truth_full=workspace_prior["all_files"],
        ground_truth_first_inspect=workspace_prior["first_file"],
        prefix_map=((f"{logical_base}/", f"{real_base}/"),),
    )


# ---------------------------------------------------------------------------
# aiob_110 — Steinmetz NWB (39 NWBs across 10 subjects)
# ---------------------------------------------------------------------------

# Hardcoded subject/session mapping ported from PoC. The Steinmetz dataset has
# a fixed schedule of recording sessions per subject that doesn't change.
_AIOB_110_SESSIONS: dict[str, tuple[str, ...]] = {
    "sub-Cori":      ("20161214", "20161217", "20161218"),
    "sub-Forssmann": ("20171101", "20171102", "20171104", "20171105"),
    "sub-Hench":     ("20170615", "20170616", "20170617", "20170618"),
    "sub-Lederberg": ("20171205", "20171206", "20171207", "20171208", "20171209", "20171210", "20171211"),
    "sub-Moniz":     ("20170515", "20170516", "20170518"),
    "sub-Muller":    ("20170107", "20170108", "20170109"),
    "sub-Radnitz":   ("20170108", "20170109", "20170110", "20170111", "20170112"),
    "sub-Richards":  ("20171029", "20171030", "20171031", "20171101", "20171102"),
    "sub-Tatum":     ("20171206", "20171207", "20171208", "20171209"),
    "sub-Theiler":   ("20171011",),
}


def _nwb_path(subj: str, date: str) -> str:
    return f"/data/steinmetz_neuropixels/raw/{subj}/{subj}_ses-{date}T120000.nwb"


def load_aiob_110() -> Workload:
    task = _curate_task(TaskConfig.from_yaml(_task_yaml("aiob_110")), "aiob_110")
    real_base = data_root() / task.dataset_subdir / "raw"
    logical_base = "/data/steinmetz_neuropixels/raw"

    workspace_prior: dict[str, tuple[str, ...]] = {
        **{
            f"subject_{subj}": tuple(_nwb_path(subj, d) for d in dates)
            for subj, dates in _AIOB_110_SESSIONS.items()
        },
        "all_subjects": tuple(
            _nwb_path(subj, d)
            for subj, dates in _AIOB_110_SESSIONS.items()
            for d in dates
        ),
        "output_psth": ("/output/result/population_psth.parquet",),
        "output_session_summary": ("/output/result/session_summary.parquet",),
        "output_report_md": ("/output/result/report.md",),
    }

    # Verify subject set matches the detector rule library.
    assert set(_AIOB_110_SESSIONS) == set(AIOB_110_SUBJECTS), (
        "aiob_110 subject list drifted between rules.py and workloads/aiob.py"
    )

    return Workload(
        task_id="aiob_110",
        task=task,
        workspace_prior=workspace_prior,
        ground_truth_full=workspace_prior["all_subjects"],
        ground_truth_first_inspect=workspace_prior["subject_sub-Cori"],
        prefix_map=((f"{logical_base}/", f"{real_base}/"),),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_AIOB_WORKLOADS: dict[str, "callable[[], Workload]"] = {
    "aiob_101": load_aiob_101,
    "aiob_103": load_aiob_103,
    "aiob_104": load_aiob_104,
    "aiob_107": load_aiob_107,
    "aiob_110": load_aiob_110,
}


def get_aiob_workload(task_id: str) -> Workload:
    if task_id not in ALL_AIOB_WORKLOADS:
        raise KeyError(
            f"Unknown AIOB workload {task_id!r}. "
            f"Known: {sorted(ALL_AIOB_WORKLOADS)}"
        )
    return ALL_AIOB_WORKLOADS[task_id]()
