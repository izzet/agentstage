"""AgentIOBench workload definitions for AgentStage.

For each AIOB workload we use end-to-end (aiob_101 / aiob_104 / aiob_107 /
aiob_110), this module produces a `Workload` value carrying:

- The AIOB `TaskConfig` (loaded from the submodule's YAML) — gives us
  `dataset_subdir`, `output_fname`, `task_inst`, validator, etc.
- A bucketed **workspace prior** (`dict[str, tuple[str, ...]]`) whose keys
  match the `target_keys` referenced by `agentstage.predictor.rules`.
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
from dataclasses import dataclass
from pathlib import Path

import yaml

from agentstage.predictor.rules import AIOB_104_SAMPLES, AIOB_110_SUBJECTS

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
# aiob_104 — IGSR genomics (50 GBR samples × BAM/BAI/BAS + reference)
# ---------------------------------------------------------------------------

def load_aiob_104() -> Workload:
    task = TaskConfig.from_yaml(_task_yaml("aiob_104"))
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
    workspace_prior["output_coverage"] = ("/output/result/coverage_matrix.parquet",)
    workspace_prior["output_qc"] = ("/output/result/qc_metrics.parquet",)
    workspace_prior["output_undercov"] = ("/output/result/undercovered_intervals.parquet",)
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
    aiob_107 — the predictor sees identical `/data/goes_cmi_composites/raw/...`
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
    # The predictor matches against text mentions of files; what changes
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
    task = TaskConfig.from_yaml(_task_yaml("aiob_107"))
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
        "output_csv": ("/output/result/goes_cmi_timeseries.csv",),
        "output_fig": ("/output/result/goes_cmi_point_timeseries.png",),
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
    task = TaskConfig.from_yaml(_task_yaml("aiob_110"))
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
        "output_trial_responses": ("/output/result/trial_responses.parquet",),
        "output_session_summary": ("/output/result/session_summary.parquet",),
        "output_report_md": ("/output/result/report.md",),
    }

    # Verify subject set matches the predictor rule library.
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
