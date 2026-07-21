"""KramaBench workload loader.

Builds AgentStage Workload-shaped objects from KramaBench's
external/benchmarks/kramabench/workload/<domain>.json + the matching
data/<domain>/input/ tree.

Used for cross-benchmark generalization evidence (H6) — same detector
+ same auto-rule generator, different benchmark, different domain.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

KRAMA_ROOT = Path(
    os.environ.get(
        "AGENTSTAGE_KRAMABENCH_ROOT",
        Path(__file__).resolve().parents[3] / "external" / "benchmarks" / "kramabench",
    )
)


@dataclass(frozen=True)
class KramaTask:
    """A single KramaBench task (one of the 21 wildfire / 12 astronomy / 9 biomedical)."""

    task_id: str                # e.g. "wildfire-easy-1"
    domain: str                 # e.g. "wildfire"
    query: str
    answer: object
    data_sources: tuple[str, ...]
    subtasks: tuple[dict, ...] = ()

    @property
    def task_inst(self) -> str:
        """AIOB-style task_inst (so AutoRuleGenerator can find format tokens)."""
        return self.query


@dataclass(frozen=True)
class KramaWorkload:
    """KramaBench analog of agentstage.workloads.aiob.Workload.

    Differences from AIOB:
      - All files live under one logical root /data/<domain>/ (no
        per-bucket logical prefixes).
      - workspace_prior is built from listing data/<domain>/input/
        (everything the agent could plausibly read).
      - ground_truth_full is data_sources from the task spec
        (gold pipeline references).
    """

    task_id: str
    task: KramaTask
    workspace_prior: dict[str, tuple[str, ...]]
    ground_truth_full: tuple[str, ...]
    ground_truth_first_inspect: tuple[str, ...]
    prefix_map: tuple[tuple[str, str], ...]

    @property
    def all_workspace_paths(self) -> tuple[str, ...]:
        return tuple(p for paths in self.workspace_prior.values() for p in paths)


def _enumerate_input(domain_input: Path) -> list[str]:
    """Recursively enumerate every readable file under data/<domain>/input/.
    Returns physical absolute paths."""
    out: list[str] = []
    for root, _dirs, files in os.walk(domain_input):
        for f in files:
            out.append(str(Path(root) / f))
    return sorted(out)


def _build_workspace_prior(
    files_phys: list[str], logical_root: str,
) -> dict[str, tuple[str, ...]]:
    """Group physical files into bucket keys by their top-level subdir
    under input/. Returns LOGICAL paths."""
    buckets: dict[str, list[str]] = {}
    for phys in files_phys:
        # Strip path prefix to get the subpath under input/
        # phys = .../data/<domain>/input/<group>/<sub>/<name>
        try:
            i = phys.split("/input/", 1)[1]
        except IndexError:
            continue
        parts = i.split("/", 1)
        group = parts[0] if len(parts) > 1 else "root"
        # Sanitize bucket key
        bucket_key = "files_" + group.replace("-", "_").replace(".", "_")
        logical = f"{logical_root}/{i}"
        buckets.setdefault(bucket_key, []).append(logical)
    # Add an aggregate
    all_files: list[str] = []
    for v in buckets.values():
        all_files.extend(v)
    out: dict[str, tuple[str, ...]] = {k: tuple(v) for k, v in buckets.items()}
    out["all_files"] = tuple(all_files)
    return out


def load_kramabench_task(domain: str, task_id: str) -> KramaWorkload:
    """Load one KramaBench task as a Workload."""
    spec_path = KRAMA_ROOT / "workload" / f"{domain}.json"
    tasks = json.loads(spec_path.read_text())
    matching = [t for t in tasks if t["id"] == task_id]
    if not matching:
        raise KeyError(f"task {task_id} not in {spec_path}")
    raw = matching[0]
    task = KramaTask(
        task_id=raw["id"],
        domain=domain,
        query=raw["query"],
        answer=raw.get("answer"),
        data_sources=tuple(raw.get("data_sources", ())),
        subtasks=tuple(raw.get("subtasks", ())),
    )

    data_input = KRAMA_ROOT / "data" / domain / "input"
    files_phys = _enumerate_input(data_input)
    logical_root = f"/data/{domain}"

    workspace_prior = _build_workspace_prior(files_phys, logical_root)

    # GT: the agent should read the task's named data_sources
    gt_logical = tuple(f"{logical_root}/{src}" for src in task.data_sources)

    prefix_map = ((logical_root + "/", str(data_input) + "/"),)

    return KramaWorkload(
        task_id=f"kb_{domain}_{task_id}",
        task=task,
        workspace_prior=workspace_prior,
        ground_truth_full=gt_logical,
        ground_truth_first_inspect=gt_logical,
        prefix_map=prefix_map,
    )


# Convenience loaders for the 3-task minimal cross-benchmark slice
def load_kb_astronomy_easy_1() -> KramaWorkload:
    return load_kramabench_task("astronomy", "astronomy-easy-1")


def load_kb_biomedical_easy_6() -> KramaWorkload:
    return load_kramabench_task("biomedical", "biomedical-easy-6")


def load_kb_wildfire_easy_1() -> KramaWorkload:
    return load_kramabench_task("wildfire", "wildfire-easy-1")


KB_MINIMAL_SLICE = {
    "kb_astronomy_easy_1": load_kb_astronomy_easy_1,
    "kb_biomedical_easy_6": load_kb_biomedical_easy_6,
    "kb_wildfire_easy_1": load_kb_wildfire_easy_1,
}


# ---------------------------------------------------------------------------
# Custom "data inventory" task — built on KramaBench astronomy raw data.
#
# Motivation: KramaBench's stock tasks are Q&A style and each reads only
# 5-10 files, so they are LLM-dominated and don't exercise bulk I/O.
# Real data-engineering pipelines often start with a data-inventory step
# that walks every file. This custom task exercises that pattern on
# KramaBench's actual astronomy raw archive (1538 files / 486 MB) and
# becomes a transferability data point for AgentStage on a third-party
# scientific-data benchmark.
# ---------------------------------------------------------------------------

_KB_ASTRONOMY_INVENTORY_INST = (
    "You are building a data inventory for the astronomy raw archive in "
    "/data/astronomy/. The full archive contains every file across the "
    "following subgroups: geomag_forecast, STORM-AI, swarm, omni2, and "
    "any other top-level directory under /data/astronomy/. Each file must "
    "appear in the manifest exactly once.\n\n"
    "For every file under /data/astronomy/ (recursively, across all "
    "subdirectories) record three fields: the relative path, the file size "
    "in bytes (via os.path.getsize), and the first 64 bytes of the file "
    "content encoded as a lowercase hex string (open the file in binary, "
    "read up to 64 bytes, .hex()). Do not subsample or skip directories.\n\n"
    "Save the result as result/astronomy_inventory.csv with columns: "
    "relpath, size_bytes, head_hex. Sort by relpath. Include every file "
    "found in the recursive walk (typical scientific data archives in "
    "this domain ship roughly 1500 files at this stage of the pipeline).\n"
)


_KB_WILDFIRE_INVENTORY_INST = (
    "You are building a data inventory for the wildfire raw archive "
    "in /data/wildfire/. The archive contains ~22 files covering "
    "wildfire counts, suppression costs, weather data, and "
    "geographic boundaries across multiple sources (NIFC, NOAA, US "
    "Census, RAWS stations).\n\n"
    "For every file under /data/wildfire/ (recursively across all "
    "subdirectories) record three fields: relative path, file size in "
    "bytes (via os.path.getsize), and the first 64 bytes of the file "
    "content encoded as a lowercase hex string (open in binary, read "
    "up to 64 bytes, .hex()). Do not subsample.\n\n"
    "Save the result as result/wildfire_inventory.csv with columns: "
    "relpath, size_bytes, head_hex. Sort by relpath. Include every "
    "file from the recursive walk."
)


def load_kb_wildfire_inventory() -> KramaWorkload:
    """KramaBench wildfire data inventory task (~22 files / 31 MB).
    Smaller than astronomy by 16× — explicitly tests the lower bound
    of AgentStage's operating envelope on small-data workloads."""
    data_input = KRAMA_ROOT / "data" / "wildfire" / "input"
    files_phys = _enumerate_input(data_input)
    logical_root = "/data/wildfire"
    workspace_prior = _build_workspace_prior(files_phys, logical_root)
    gt_logical = workspace_prior["all_files"]
    task = KramaTask(
        task_id="wildfire-inventory",
        domain="wildfire",
        query=_KB_WILDFIRE_INVENTORY_INST,
        answer=None,
        data_sources=tuple(p[len("/data/wildfire/"):] for p in gt_logical),
    )
    prefix_map = ((logical_root + "/", str(data_input) + "/"),)
    return KramaWorkload(
        task_id="kb_wildfire_inventory",
        task=task,
        workspace_prior=workspace_prior,
        ground_truth_full=gt_logical,
        ground_truth_first_inspect=gt_logical,
        prefix_map=prefix_map,
    )


def load_kb_astronomy_inventory() -> KramaWorkload:
    """KramaBench astronomy: comprehensive data-inventory task over all
    1538 input files. Stresses many-small-files metadata + first-byte
    read latency, which is the regime where OrangeFS cold-tier RPCs
    dominate session time."""
    data_input = KRAMA_ROOT / "data" / "astronomy" / "input"
    files_phys = _enumerate_input(data_input)
    logical_root = "/data/astronomy"
    workspace_prior = _build_workspace_prior(files_phys, logical_root)
    # All-files first-inspect: a 'first-look' over the entire archive,
    # matching the inventory task pattern.
    gt_logical = workspace_prior["all_files"]
    task = KramaTask(
        task_id="astronomy-inventory",
        domain="astronomy",
        query=_KB_ASTRONOMY_INVENTORY_INST,
        answer=None,
        data_sources=tuple(p[len("/data/astronomy/"):] for p in gt_logical),
    )
    prefix_map = ((logical_root + "/", str(data_input) + "/"),)
    return KramaWorkload(
        task_id="kb_astronomy_inventory",
        task=task,
        workspace_prior=workspace_prior,
        ground_truth_full=gt_logical,
        ground_truth_first_inspect=gt_logical,
        prefix_map=prefix_map,
    )
