"""Re-score a run's stream.jsonl against the frozen v1 rule library.

For every run dir under an outputs root, this module:

1. Parses `stream.jsonl` into blocks (via `detector.engine.parse_stream`)
2. Runs the v1 detector → `Detection`
3. Computes byte recall + overfetch against the workload's static GT
   (both "first inspect" and "full working set" flavors) for each
   tier and the HOT layer
4. Writes `byte_metrics_v1.json` + `detection_v1.json` alongside the
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
from agentstage.detector.engine import (
    Detection,
    StreamBlock,
    parse_stream,
    run_detector,
)
from agentstage.detector.rules import RULE_LIBRARY_HASH, RULE_LIBRARY_VERSION, get_ruleset
from agentstage.workloads import Workload, get_workload


# ---------------------------------------------------------------------------
# Multi-turn (turns/) format support
#
# The PoC single-turn runs record a live SSE `stream.jsonl`. The agentic
# multiturn runs (scripts/microbench/aiob_multiturn*.py) instead record a
# `turns/turn_NN/{thinking,text,tool_use,tool_result}.jsonl` tree and have
# no stream.jsonl. This builder reconstructs the same `StreamBlock`s the
# detector scans (thinking + text + tool_result) from that tree so the
# identical `run_detector` / `_score_all_tiers` pipeline applies. Recall
# depends only on the detected *set*, so synthesized per-turn timings (as
# in engine.blocks_from_messages) are sufficient.
# ---------------------------------------------------------------------------

def _read_deltas(path: Path) -> str:
    """Concatenate streamed {'delta': ...} chunks from a turns/*.jsonl."""
    if not path.is_file():
        return ""
    out: list[str] = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and "delta" in d:
            out.append(str(d["delta"]))
    return "".join(out)


def _read_tool_results(path: Path) -> str:
    """Concatenate tool_result `content` (the listings/outputs the model
    sees on the next turn — they carry literal paths the HOT scan needs)."""
    if not path.is_file():
        return ""
    parts: list[str] = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(d, dict):
            continue
        c = d.get("content")
        if isinstance(c, list):
            parts.append("\n".join(
                str(s.get("text", "")) for s in c
                if isinstance(s, dict) and s.get("type") == "text"))
        elif c is not None:
            parts.append(str(c))
    return "\n".join(p for p in parts if p)


def blocks_from_turns(turns_dir: Path, per_turn_ms: float = 1000.0) -> list[StreamBlock]:
    """Build detector-scannable StreamBlocks from a turns/ directory.

    Per turn N: a thinking block and a text block at t∈[N, N+1)*per_turn_ms,
    and a tool_result block at the back half of the turn (so it precedes the
    next turn's reasoning in stream order). tool_use blocks are not emitted
    because the detector does not scan them (`_SCANNABLE_BLOCK_TYPES`)."""
    blocks: list[StreamBlock] = []
    for tdir in sorted(turns_dir.glob("turn_*")):
        try:
            turn = int(tdir.name.split("_", 1)[1])
        except (ValueError, IndexError):
            turn = 0
        tf = turn * per_turn_ms
        ts = (turn + 1) * per_turn_ms
        thinking = _read_deltas(tdir / "thinking.jsonl")
        text = _read_deltas(tdir / "text.jsonl")
        tool_result = _read_tool_results(tdir / "tool_result.jsonl")
        if thinking:
            blocks.append(StreamBlock("thinking", tf, ts, thinking, 1, turn=turn))
        if text:
            blocks.append(StreamBlock("text", tf, ts, text, 1, turn=turn))
        if tool_result:
            blocks.append(StreamBlock(
                "tool_result", tf + 0.5 * per_turn_ms, ts, tool_result, 1, turn=turn))
    return blocks


def _score_all_tiers(
    pred: Detection,
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
            score = byte_score(tier.detected_files, gt, workload.prefix_map)
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

    Writes `byte_metrics_v1.json` + `detection_v1.json` next to the
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
    out_detection = run_dir / "detection_v1.json"
    if out_metrics.is_file() and not force:
        return out_metrics

    stream_path = run_dir / "stream.jsonl"
    turns_dir = run_dir / "turns"
    if stream_path.is_file():
        provider = summary.get("provider")
        blocks = parse_stream(stream_path, provider=provider)
        per_char = True
    elif turns_dir.is_dir():
        # Multiturn agentic run (no live stream) — reconstruct blocks from
        # the turns/ tree and score through the identical pipeline. Atomic
        # thinking scan (per_char=False): identical detected set, avoids the
        # O(n²) per-char loop degenerating on long transcripts.
        blocks = blocks_from_turns(turns_dir)
        per_char = False
    else:
        return None

    detection = run_detector(blocks, workload.workspace_prior, ruleset, per_char=per_char)
    metrics = _score_all_tiers(detection, workload)

    out_metrics.write_text(json.dumps(metrics, indent=2, default=float))
    out_detection.write_text(json.dumps(detection.to_dict(), indent=2, default=float))
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
