"""Re-score a run's stream.jsonl against the frozen v1 rule library.

For every run dir under an outputs root, this module:

1. Parses `stream.jsonl` into blocks (via `predictor.engine.parse_stream`)
2. Runs the v1 predictor → `Prediction`
3. Computes byte recall + overfetch against the workload's static GT
   (both "first inspect" and "full working set" flavors) for each
   tier and the HOT layer
4. Writes `byte_metrics_v1.json` + `prediction_v1.json` alongside the
   originals (the PoC's `byte_metrics.json` is kept untouched as a
   pre-freeze snapshot)

Re-scoring is idempotent (re-runs over the same dir overwrite the v1
files but leave originals alone). Skips dirs that already have
`byte_metrics_v1.json` unless `force=True`.

Used by:
- paper_evals (H3, H7) which expect byte_metrics_v1.json on every run
- the CLI `agentstage-rescore` (when it lands)
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from agentstage.metrics.byte_metrics import ByteScore, byte_score
from agentstage.predictor.engine import (
    Prediction,
    parse_stream,
    run_predictor,
)
from agentstage.predictor.rules import RULE_LIBRARY_HASH, RULE_LIBRARY_VERSION, get_ruleset
from agentstage.workloads import Workload, get_workload


def _score_all_tiers(
    pred: Prediction,
    workload: Workload,
) -> dict[str, dict]:
    """Compute byte_score per (tier, ground-truth flavor)."""
    gt_first = workload.ground_truth_first_inspect
    gt_full = workload.ground_truth_full

    out: dict[str, dict] = {
        "rule_library_version": RULE_LIBRARY_VERSION,
        "rule_library_hash": RULE_LIBRARY_HASH,
    }

    # Tier scores against immediate-need + eventual-working-set
    for tier_name, tier in (("tier_1", pred.tier_1), ("tier_2", pred.tier_2), ("tier_3", pred.tier_3)):
        for gt_name, gt in (("first", gt_first), ("full", gt_full)):
            score = byte_score(tier.predicted_files, gt, workload.prefix_map)
            out[f"{tier_name}_{gt_name}"] = score.to_dict()

    # HOT scan score (high-precision, low-recall layer)
    hot_paths = tuple(pred.hot.keys())
    for gt_name, gt in (("first", gt_first), ("full", gt_full)):
        score = byte_score(hot_paths, gt, workload.prefix_map)
        out[f"hot_{gt_name}"] = score.to_dict()

    # Naive stage-all baseline (every file in the workspace prior)
    naive = workload.all_workspace_paths
    for gt_name, gt in (("first", gt_first), ("full", gt_full)):
        score = byte_score(naive, gt, workload.prefix_map)
        out[f"naive_{gt_name}"] = score.to_dict()

    return out


def rescore_run(run_dir: Path, force: bool = False) -> Path | None:
    """Re-score one run dir's stream.jsonl against the frozen v1 rule lib.

    Writes `byte_metrics_v1.json` + `prediction_v1.json` next to the
    originals. Returns the path to `byte_metrics_v1.json`, or None if
    the run can't be re-scored (no thinking, unknown task, etc.).
    """
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
        ruleset = get_ruleset(task_id)
    except KeyError:
        return None

    out_metrics = run_dir / "byte_metrics_v1.json"
    out_prediction = run_dir / "prediction_v1.json"
    if out_metrics.is_file() and not force:
        return out_metrics

    stream_path = run_dir / "stream.jsonl"
    if not stream_path.is_file():
        return None

    provider = summary.get("provider")
    blocks = parse_stream(stream_path, provider=provider)
    prediction = run_predictor(blocks, workload.workspace_prior, ruleset)
    metrics = _score_all_tiers(prediction, workload)

    out_metrics.write_text(json.dumps(metrics, indent=2, default=float))
    out_prediction.write_text(json.dumps(prediction.to_dict(), indent=2, default=float))
    return out_metrics


def rescore_outputs_root(
    outputs_root: Path,
    force: bool = False,
    skip_tasks: Iterable[str] = (),
) -> dict[str, int]:
    """Re-score every run dir under `outputs_root`. Returns counts."""
    skip = set(skip_tasks)
    root = Path(outputs_root)
    if not root.is_dir():
        return {"processed": 0, "skipped": 0, "errored": 0}

    processed = 0
    skipped = 0
    errored = 0

    candidates: list[Path] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if (d / "summary.json").is_file():
            candidates.append(d)
            continue
        for sub in sorted(d.iterdir()):
            if sub.is_dir() and (sub / "summary.json").is_file():
                candidates.append(sub)

    for d in candidates:
        try:
            summary = json.loads((d / "summary.json").read_text())
            if summary.get("task") in skip:
                skipped += 1
                continue
            result = rescore_run(d, force=force)
            if result is None:
                skipped += 1
            else:
                processed += 1
        except Exception:  # noqa: BLE001 — best-effort scan
            errored += 1

    return {"processed": processed, "skipped": skipped, "errored": errored}
