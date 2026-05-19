"""AgentStage stager: in-process file pre-stager + LD_PRELOAD path-rewriting shim.

See STAGER_DESIGN.md (spec) and STAGER_WALKTHROUGH.md (explainer) at the
project root for the full design.

Quick start:
    from agentstage.stager import Stager, DataHint, StagingReport

    stager = Stager(
        hot_root="/dev/shm/agentstage",
        cold_roots=["/mnt/common/datasets-staging"],
        capacity_bytes=32 * 1024**3,
    )
    hint = DataHint(
        predicted_files=("/mnt/common/datasets-staging/foo.nc",),
        tier=1,
        fired_at_ms=1200.0,
        rule_id="first_inspect_goes",
    )
    stager.prefetch(hint)
    # ... LLM continues thinking ...
    # ... agent eventually opens foo.nc ...
    # LD_PRELOAD shim redirects open() to /dev/shm/agentstage/mnt/.../foo.nc
"""

from .daemon import GB, Stager, StagerOutOfSpace
from .report import DataHint, StageEvent, StagingReport, now_ms

__all__ = [
    "DataHint",
    "GB",
    "StageEvent",
    "Stager",
    "StagerOutOfSpace",
    "StagingReport",
    "now_ms",
]
