"""Detection ablation: naive hand rules vs auto-generator vs auto+list_dir.

For each of the 9 result-trio tasks, we author 3-5 "naive" hand rules from
the task description text alone — no peeking at the workspace inventory,
no per-instance enumeration. Each rule targets an aggregate workspace_prior
bucket (e.g., all_samples, all_files, train_csv) using regex patterns over
file types and domain keywords a human reader would identify.

The ablation runs three detector variants on the 108 sessions of the
36-cell matrix:
  (H)  Naive hand rules, with thinking + text + prior-turn tool_result
  (A1) Auto-generator output, thinking + text only (no list_dir refinement)
  (A2) Auto-generator output, with thinking + text + prior-turn tool_result
       (the §IV.B headline configuration)

Output:
  Pooled and per-benchmark median tier-1 byte recall + no-detection counts
  for each variant. Used by paper §IV.B's ablation paragraph.

This script is a one-off ablation runner, not part of the production
detector. Production AgentStage uses the auto-generator (A2 path).
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from agentstage.detector.auto_rules import AutoRuleGenerator  # noqa: E402
from agentstage.detector.engine import run_detector  # noqa: E402
from agentstage.detector.rules import Rule, RuleSet  # noqa: E402
from agentstage.metrics.byte_metrics import byte_score  # noqa: E402
from agentstage.metrics.rescore import blocks_from_turns  # noqa: E402
from agentstage.workloads import get_workload  # noqa: E402


# ---------------------------------------------------------------------------
# Naive hand rules: what a researcher might write after reading the task
# description text only, with no workspace-inventory traversal.
# ---------------------------------------------------------------------------
HAND_RULES: dict[str, list[tuple[str, str, tuple[str, ...]]]] = {
    "aiob_201": [
        ("bam_pattern", r"\bBAM\b|alignment|samples", ("all_samples",)),
        ("reference_pattern", r"reference|chromosome", ("reference",)),
        ("output_manifest", r"manifest|integrity", ("output_manifest",)),
    ],
    "aiob_202": [
        ("fits_pattern", r"\bFITS\b|JWST|coadd|calibration", ("all_fits",)),
        ("detector_pattern", r"detector|nrca|nrcb", ("all_fits",)),
        ("output_manifest", r"manifest|integrity", ("output_manifest",)),
    ],
    "aiob_205": [
        ("fits_pattern", r"\bFITS\b|JWST", ("all_files",)),
        ("geotiff_pattern", r"GeoTIFF|tiff|Sentinel|scene", ("all_files",)),
        ("output_manifest", r"manifest|integrity", ("output_manifest",)),
    ],
}


def hand_ruleset(task_id: str) -> RuleSet:
    spec = HAND_RULES.get(task_id, [])
    rules = tuple(
        Rule(name=n, pattern=p, target_keys=tk, origin="general")
        for (n, p, tk) in spec
    )
    return RuleSet(workload=task_id, rules=rules)


# ---------------------------------------------------------------------------
# Result-trio scope (mirrors paper_figures/build_fig_detection.py)
# ---------------------------------------------------------------------------
PATH_EXCLUDE = ("poc/", "surgical_", "_sweep_pathful", "_smoke", "_archive")
RESULT_TRIO = set(HAND_RULES.keys())
TASK_BENCH = {t: "aiob" for t in ("aiob_201", "aiob_202", "aiob_205")}
REPS_PER_CELL = 3


def model_key(m: str | None) -> str | None:
    m = (m or "").lower()
    if "haiku" in m:
        return "haiku"
    if "sonnet" in m:
        return "sonnet"
    if "flash" in m and "gemini" in m:
        return "flash"
    if "qwen" in m:
        return "qwen3"
    return None


def score_one(blocks, workload, ruleset):
    detection = run_detector(
        blocks, workload.workspace_prior, ruleset, per_char=False,
    )
    # Score against ground_truth_full to match §IV.B's recall metric
    gt = workload.ground_truth_full
    s = byte_score(detection.tier_1.detected_files, gt, workload.prefix_map)
    return s.to_dict()


def main() -> int:
    rows: list[dict] = []
    for sf in (REPO / "outputs").rglob("summary.json"):
        rel = str(sf.relative_to(REPO / "outputs"))
        if any(f in rel for f in PATH_EXCLUDE):
            continue
        if "_archive" in sf.parts:
            continue
        try:
            s = json.loads(sf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        task = s.get("task")
        if task not in RESULT_TRIO:
            continue
        mk = model_key(s.get("model"))
        if mk is None:
            continue
        rd = sf.parent
        if not (rd / "turns").is_dir():
            continue
        try:
            workload = get_workload(task)
            auto_rs = AutoRuleGenerator(
                workload_id=task,
                task_instruction=workload.task.task_inst,
                workspace_prior_keys=tuple(workload.workspace_prior.keys()),
            ).generate()
            hand_rs = hand_ruleset(task)
        except Exception:  # noqa: BLE001
            continue
        blocks = blocks_from_turns(rd / "turns")
        blocks_full = [b for b in blocks if b.type in ("thinking", "text", "tool_result")]
        blocks_noldir = [b for b in blocks if b.type in ("thinking", "text")]
        if not blocks_full:
            continue
        try:
            score_h = score_one(blocks_full, workload, hand_rs)
            score_anl = score_one(blocks_noldir, workload, auto_rs)
            score_al = score_one(blocks_full, workload, auto_rs)
            if score_h.get("n_ground_truth", 0) == 0:
                continue
        except Exception:  # noqa: BLE001
            continue
        rows.append({
            "rel": str(rd), "task": task, "mk": mk, "bench": TASK_BENCH[task],
            "h_rec": float(score_h.get("byte_recall", 0)),
            "anl_rec": float(score_anl.get("byte_recall", 0)),
            "al_rec": float(score_al.get("byte_recall", 0)),
        })

    # Cap to REPS_PER_CELL per (task, model) cell
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_cell[(r["task"], r["mk"])].append(r)
    capped: list[dict] = []
    for _, cell_rows in by_cell.items():
        cell_rows.sort(key=lambda r: r["rel"])
        capped.extend(cell_rows[-REPS_PER_CELL:])

    print(f"\nn = {len(capped)} (capped to {REPS_PER_CELL} reps per cell)\n")
    print("Per-benchmark median tier-1 byte recall:")
    print(f"  {'Variant':22s}  {'AIOB':>6s}  {'DSBench':>8s}  {'MLE':>6s}")
    for label, key in [
        ("Naive hand", "h_rec"),
        ("Auto, no list_dir", "anl_rec"),
        ("Auto + list_dir", "al_rec"),
    ]:
        cells = []
        for b in ["aiob", "dsbench", "mle"]:
            vals = [r[key] for r in capped if r["bench"] == b]
            cells.append(statistics.median(vals) if vals else 0.0)
        print(f"  {label:22s}  {cells[0]:>6.2f}  {cells[1]:>8.2f}  {cells[2]:>6.2f}")

    print("\nPooled no-detection counts:")
    for label, key in [
        ("Naive hand", "h_rec"),
        ("Auto, no list_dir", "anl_rec"),
        ("Auto + list_dir", "al_rec"),
    ]:
        nd = sum(1 for r in capped if r[key] == 0.0)
        print(f"  {label:22s}  {nd}/{len(capped)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
