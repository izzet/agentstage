"""Re-score a run's turns/ trajectory against the auto-rule generator.

Parallel to `rescore.py` but invokes `AutoRuleGenerator` per workload
instead of `get_ruleset(task_id)`. Writes `byte_metrics_auto.json` +
`detection_auto.json` so the auto and frozen scores can coexist.

Used by the paper-figure pipeline once we move §IV.B to characterize
the auto-generator's output rather than the curated frozen library.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from agentstage.detector.auto_rules import AutoRuleGenerator
from agentstage.detector.engine import (
    Detection,
    StreamBlock,
    parse_stream,
    run_detector,
)
from agentstage.metrics.byte_metrics import byte_score
from agentstage.metrics.rescore import blocks_from_turns
from agentstage.workloads import Workload, get_workload


def _score_all_tiers(pred: Detection, workload: Workload) -> dict[str, dict]:
    """Same shape as rescore._score_all_tiers but without rule-library
    version pins (auto-rules are workload-derived, not frozen)."""
    gt_first = workload.ground_truth_first_inspect
    gt_full = workload.ground_truth_full

    out: dict[str, dict] = {}

    for tier_name, tier in (
        ("tier_1", pred.tier_1),
        ("tier_2", pred.tier_2),
        ("tier_3", pred.tier_3),
    ):
        for gt_name, gt in (("first", gt_first), ("full", gt_full)):
            score = byte_score(tier.detected_files, gt, workload.prefix_map)
            out[f"{tier_name}_{gt_name}"] = score.to_dict()

    hot_paths = tuple(pred.hot.keys())
    for gt_name, gt in (("first", gt_first), ("full", gt_full)):
        score = byte_score(hot_paths, gt, workload.prefix_map)
        out[f"hot_{gt_name}"] = score.to_dict()

    naive = workload.all_workspace_paths
    for gt_name, gt in (("first", gt_first), ("full", gt_full)):
        score = byte_score(naive, gt, workload.prefix_map)
        out[f"naive_{gt_name}"] = score.to_dict()

    return out


def rescore_run(run_dir: Path, force: bool = False) -> Path | None:
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return None

    summary = json.loads(summary_path.read_text())
    task_id = summary.get("task")
    if not task_id:
        return None
    try:
        workload = get_workload(task_id)
    except (KeyError, FileNotFoundError):
        return None

    out_metrics = run_dir / "byte_metrics_auto.json"
    out_detection = run_dir / "detection_auto.json"
    if out_metrics.is_file() and not force:
        return out_metrics

    stream_path = run_dir / "stream.jsonl"
    turns_dir = run_dir / "turns"
    if stream_path.is_file():
        provider = summary.get("provider")
        blocks = parse_stream(stream_path, provider=provider)
        per_char = True
    elif turns_dir.is_dir():
        blocks = blocks_from_turns(turns_dir)
        per_char = False
    else:
        return None

    ruleset = AutoRuleGenerator(
        workload_id=task_id,
        task_instruction=workload.task.task_inst,
        workspace_prior_keys=tuple(workload.workspace_prior.keys()),
    ).generate()

    detection = run_detector(
        blocks, workload.workspace_prior, ruleset, per_char=per_char,
    )
    metrics = _score_all_tiers(detection, workload)

    out_metrics.write_text(json.dumps(metrics, indent=2, default=float))
    out_detection.write_text(json.dumps(detection.to_dict(), indent=2, default=float))
    return out_metrics


def rescore_outputs_root(
    outputs_root: Path,
    force: bool = False,
    skip_tasks: Iterable[str] = (),
    require_task_in: Iterable[str] | None = None,
) -> dict[str, int]:
    skip = set(skip_tasks)
    require = set(require_task_in) if require_task_in is not None else None
    root = Path(outputs_root)
    if not root.is_dir():
        return {"processed": 0, "skipped": 0, "errored": 0}

    candidates: list[Path] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith("_smoke") or d.name.startswith("_archive"):
            continue
        if (d / "summary.json").is_file():
            candidates.append(d)
            continue
        for sub in sorted(d.iterdir()):
            if not sub.is_dir():
                continue
            if (sub / "summary.json").is_file():
                candidates.append(sub)

    processed = 0
    skipped = 0
    errored = 0

    for d in candidates:
        try:
            summary = json.loads((d / "summary.json").read_text())
            task_id = summary.get("task")
            if task_id in skip:
                skipped += 1
                continue
            if require is not None and task_id not in require:
                skipped += 1
                continue
            result = rescore_run(d, force=force)
            if result is None:
                skipped += 1
            else:
                processed += 1
        except Exception:  # noqa: BLE001
            errored += 1

    return {"processed": processed, "skipped": skipped, "errored": errored}
