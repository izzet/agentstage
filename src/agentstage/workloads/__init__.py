"""Workload definitions — per-workload workspace prior + ground truth +
AIOB TaskConfig + prefix map.

Public surface:
    from agentstage.workloads import get_workload, Workload, TaskConfig

A `Workload` carries everything the predictor + scorer + runner need to
process one (task, model, seed) cell. The rule library
(`agentstage.predictor.rules`) references workspace-prior bucket keys
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

ALL_WORKLOADS: dict[str, "callable[[], Workload]"] = {
    **ALL_AIOB_WORKLOADS,
    "code_repo": load_code_repo,
}


def get_workload(task_id: str) -> Workload:
    """Load a Workload by task_id. Walks the filesystem (workspace priors
    are built lazily from on-disk dataset trees)."""
    if task_id not in ALL_WORKLOADS:
        raise KeyError(
            f"Unknown workload {task_id!r}. Known: {sorted(ALL_WORKLOADS)}"
        )
    return ALL_WORKLOADS[task_id]()


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
