"""Workload definitions — per-workload workspace prior + ground truth +
AIOB TaskConfig + prefix map.

Public surface:
    from agentstage.workloads import get_workload, Workload, TaskConfig

A `Workload` carries everything the detector + scorer + runner need to
process one (task, model, seed) cell. The rule library
(`agentstage.detector.rules`) references workspace-prior bucket keys
by name; this module defines what files those keys resolve to.
"""

from agentstage.workloads.aiob import (
    ALL_AIOB_WORKLOADS,
    TaskConfig,
    Workload,
    data_root,
    get_aiob_workload,
    load_aiob_101,
    load_aiob_104,
    load_aiob_107,
    load_aiob_110,
)
from agentstage.workloads.code_repo import code_repo_root, load_code_repo
from agentstage.workloads.dsbench import load_dsbench_task
from agentstage.workloads.mlebench import (
    load_mle_competition,
    load_mle_dogsvcats_thumbhash,
)
from agentstage.workloads.kramabench import (
    load_kb_astronomy_inventory,
)

ALL_WORKLOADS: dict[str, "callable[[], Workload]"] = {
    **ALL_AIOB_WORKLOADS,
    "code_repo": load_code_repo,
    "mle_dogsvcats_thumbhash": load_mle_dogsvcats_thumbhash,
    "kb_astronomy_inventory": load_kb_astronomy_inventory,
}

# Result-trio DSBench + MLE-bench tasks. Loaded via parameterized factories
# rather than hardcoded ALL_WORKLOADS entries because the loaders walk
# on-disk dataset trees that vary per host.
_DSBENCH_TASKS = frozenset({
    "lmsys-chatbot-arena",
    "ventilator-pressure-prediction",
    "tabular-playground-series-may-2022",
    "tabular-playground-series-oct-2021",
})
_MLEBENCH_TASKS = frozenset({
    "dogs-vs-cats-redux-kernels-edition",
    "new-york-city-taxi-fare-prediction",
    "histopathologic-cancer-detection",
    "aerial-cactus-identification",
})


def get_workload(task_id: str) -> Workload:
    """Load a Workload by task_id. Walks the filesystem (workspace priors
    are built lazily from on-disk dataset trees)."""
    if task_id in ALL_WORKLOADS:
        return ALL_WORKLOADS[task_id]()
    if task_id in _DSBENCH_TASKS:
        return load_dsbench_task(task_id)
    if task_id in _MLEBENCH_TASKS:
        return load_mle_competition(task_id)
    raise KeyError(
        f"Unknown workload {task_id!r}. Known: "
        f"{sorted(ALL_WORKLOADS) + sorted(_DSBENCH_TASKS) + sorted(_MLEBENCH_TASKS)}"
    )


__all__ = [
    "ALL_AIOB_WORKLOADS",
    "ALL_WORKLOADS",
    "TaskConfig",
    "Workload",
    "code_repo_root",
    "data_root",
    "get_aiob_workload",
    "get_workload",
    "load_aiob_101",
    "load_aiob_104",
    "load_aiob_107",
    "load_aiob_110",
    "load_code_repo",
]
