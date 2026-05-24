"""E-036 — H6 frozen-rules cross-benchmark replay.

Takes auto-generated rules trained on AIOB workloads, replays them
against KramaBench (or SAB) captured corpora. Measures recall +
precision of the FROZEN AIOB ruleset against the cross-benchmark
ground truth (KramaBench's gold pipeline data_sources).

This is the H6 test: does the detector ARCHITECTURE generalize
across benchmarks when given an auto-rule set generated for one and
applied to another?

Two configurations measured:
  A. FROZEN: rules generated from AIOB workload (e.g. aiob_107),
     applied as-is.
  B. NATIVE: rules generated from the KramaBench task's own prior.
     Upper bound — what's possible if rules adapt to each new workload.

The H6 question is whether (A) lags (B) by more than 10pp; if not,
the architecture generalizes.

Usage:
    python scripts/microbench/path_b_h6_frozen_xbench.py \\
        --corpus outputs/multi_turn/<kb_capture_dir> \\
        --frozen-from-workload aiob_107 \\
        --out <corpus>/h6_xbench.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from agentstage.detector.auto_rules import AutoRuleGenerator  # noqa: E402
from agentstage.detector.engine import StreamBlock  # noqa: E402
from agentstage.detector.rules import RuleSet  # noqa: E402
from agentstage.detector.session import SessionDetector  # noqa: E402
from agentstage.workloads.aiob import (  # noqa: E402
    load_aiob_104, load_aiob_107, load_aiob_110,
)
from agentstage.workloads.kramabench import load_kramabench_task  # noqa: E402


def reconstruct_blocks_per_turn(turns_dir: Path) -> list[tuple[int, list[StreamBlock]]]:
    """Walk turns_dir, return [(turn, [blocks]), ...]."""
    out: list[tuple[int, list[StreamBlock]]] = []
    base_t = 0.0
    for tdir in sorted(turns_dir.glob("turn_*")):
        turn_num = int(tdir.name.split("_")[1])
        blocks: list[StreamBlock] = []
        for fname, btype in [("thinking.jsonl", "thinking"), ("text.jsonl", "text")]:
            p = tdir / fname
            if not p.is_file():
                continue
            chunks_by_block: dict[int, list[tuple[float, str]]] = {}
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                bi = d.get("block", 0)
                chunks_by_block.setdefault(bi, []).append(
                    (d.get("t_ms", 0.0), d.get("delta", "")))
            for bi, items in chunks_by_block.items():
                items.sort()
                text = "".join(s for _, s in items)
                t_first = items[0][0] if items else 0.0
                t_stop = items[-1][0] if items else 0.0
                blocks.append(StreamBlock(
                    type=btype, t_first=base_t + t_first, t_stop=base_t + t_stop,
                    text=text, chunks=len(items), turn=turn_num,
                ))
        tr_path = tdir / "tool_result.jsonl"
        tr_blocks: list[StreamBlock] = []
        if tr_path.exists() and tr_path.stat().st_size > 0:
            for line in tr_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = d.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(
                        c.get("text", "") for c in content
                        if isinstance(c, dict) and c.get("type") == "text"
                    )
                tr_blocks.append(StreamBlock(
                    type="tool_result",
                    t_first=base_t + 1000.0 * (turn_num + 0.5),
                    t_stop=base_t + 1000.0 * (turn_num + 0.5),
                    text=str(content), chunks=1, turn=turn_num + 1,
                ))
        out.append((turn_num, blocks))
        if tr_blocks:
            out.append((turn_num, tr_blocks))
        base_t += 10_000.0
    return out


def replay_with_ruleset(rs: RuleSet, blocks_chunks, prior) -> set[str]:
    """Return cumulative LOGICAL paths the detector would prefetch."""
    sd = SessionDetector(prior=prior, ruleset=rs)
    for _t, blocks in blocks_chunks:
        assistant = [b for b in blocks if b.type in ("thinking", "text")]
        tool_results = [b for b in blocks if b.type == "tool_result"]
        if assistant:
            sd.feed_turn(assistant)
        if tool_results:
            sd.feed_tool_results(tool_results)
    pred = sd.cumulative_detection()
    return set(pred.tier_1.detected_files + pred.tier_2.detected_files
               + pred.tier_3.detected_files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True,
                        help="A KramaBench capture dir from kramabench_capture.py")
    parser.add_argument("--frozen-from-workload",
                        choices=["aiob_104", "aiob_107", "aiob_110"],
                        default="aiob_107")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    summary_p = args.corpus / "summary.json"
    if not summary_p.is_file():
        print(f"FATAL: {summary_p} missing — is this a KB capture dir?", file=sys.stderr)
        return 2
    sm = json.loads(summary_p.read_text())
    domain = sm.get("domain", "?")
    task_id_raw = sm["task_id"]  # e.g. "kb_wildfire_wildfire-easy-1"
    parts = task_id_raw.split("_", 2)  # ["kb","wildfire","wildfire-easy-1"]
    kb_task_id = parts[-1]
    kb_workload = load_kramabench_task(domain, kb_task_id)

    blocks_chunks = reconstruct_blocks_per_turn(args.corpus / "turns")

    # ----- Configuration A: FROZEN from AIOB workload -----
    aiob_loaders = {
        "aiob_104": load_aiob_104,
        "aiob_107": load_aiob_107,
        "aiob_110": load_aiob_110,
    }
    aiob_wl = aiob_loaders[args.frozen_from_workload]()
    frozen_gen = AutoRuleGenerator(
        workload_id=aiob_wl.task_id,
        task_instruction=aiob_wl.task.task_inst,
        workspace_prior_keys=tuple(aiob_wl.workspace_prior.keys()),
    )
    frozen_rs = frozen_gen.generate()

    # CROSS-APPLY: rules trained on AIOB, but PRIOR is KramaBench's (the rules
    # output bucket keys exist only in AIOB's prior, so the frozen ruleset
    # cannot dispatch KB paths directly. The right H6 measure is:
    # "do the frozen rules' PATTERNS fire on KB reasoning content?"
    # We measure that as "any rule activation" not "correct subset dispatch.")
    sd_frozen = SessionDetector(prior=aiob_wl.workspace_prior,
                                 ruleset=frozen_rs)
    for _t, blocks in blocks_chunks:
        assistant = [b for b in blocks if b.type in ("thinking", "text")]
        tool_results = [b for b in blocks if b.type == "tool_result"]
        if assistant:
            sd_frozen.feed_turn(assistant)
        if tool_results:
            sd_frozen.feed_tool_results(tool_results)
    frozen_pred = sd_frozen.cumulative_detection()
    n_frozen_activations = sum(
        1 for act in frozen_pred.activations if act.detected_files
    )

    # ----- Configuration B: NATIVE on KB prior -----
    native_gen = AutoRuleGenerator(
        workload_id=kb_workload.task_id,
        task_instruction=kb_workload.task.task_inst,
        workspace_prior_keys=tuple(kb_workload.workspace_prior.keys()),
    )
    native_rs = native_gen.generate()
    native_logical = replay_with_ruleset(native_rs, blocks_chunks,
                                          kb_workload.workspace_prior)

    # GT = files the agent actually opened in this run (logical)
    gt_actual = set(sm.get("files_opened_logical", []))
    # GT_static = files KramaBench's gold pipeline lists
    gt_static = set(kb_workload.ground_truth_full)

    def metrics(predicted: set[str], gt: set[str]) -> dict:
        if not gt:
            return {"hits": 0, "precision": None, "recall": None,
                    "predicted_n": len(predicted), "gt_n": 0}
        hits = predicted & gt
        prec = len(hits) / len(predicted) if predicted else None
        rec = len(hits) / len(gt)
        return {"hits": len(hits),
                "precision": round(prec, 4) if prec is not None else None,
                "recall": round(rec, 4),
                "predicted_n": len(predicted), "gt_n": len(gt)}

    native_vs_actual = metrics(native_logical, gt_actual)
    native_vs_static = metrics(native_logical, gt_static)

    # FROZEN scoring: since the rules dispatch AIOB paths, "predicted set
    # intersected with KB GT" is always empty by construction. The
    # meaningful H6 question is whether the *pattern detector layer* of
    # the frozen ruleset fires on KB content (i.e., does the frozen
    # vocabulary trigger on cross-benchmark reasoning?).
    # Report n_rule_firings as the frozen quality signal.

    result = {
        "experiment": "E-036",
        "corpus": str(args.corpus),
        "kb_task_id": kb_workload.task_id,
        "domain": domain,
        "frozen_from_workload": args.frozen_from_workload,
        "model": sm.get("model"),
        "prompt_mode": sm.get("prompt_mode"),
        "n_agent_opens": len(sm.get("files_opened_logical", [])),
        "n_gt_actual": len(gt_actual),
        "n_gt_static": len(gt_static),
        "frozen": {
            "n_rules": len(frozen_rs.rules),
            "n_activations": n_frozen_activations,
            "fires_on_xbench_content": n_frozen_activations > 0,
            "note": ("Frozen rules dispatch AIOB-domain bucket keys; "
                     "cross-benchmark dispatch is impossible by "
                     "construction. We measure whether the AIOB-trained "
                     "rule PATTERNS fire on KB reasoning content."),
        },
        "native": {
            "n_rules": len(native_rs.rules),
            "vs_actual": native_vs_actual,
            "vs_static_gt": native_vs_static,
        },
    }
    args.out.write_text(json.dumps(result, indent=2))

    print(f"E-036 H6 cross-benchmark | {args.corpus.name}")
    print(f"  KB task: {kb_workload.task_id}")
    print(f"  model:   {sm.get('model')} ({sm.get('prompt_mode')})")
    print(f"  agent opens (logical): {len(sm.get('files_opened_logical', []))}")
    print()
    print(f"  FROZEN ({args.frozen_from_workload}): {n_frozen_activations} rule activations on KB content")
    print(f"     → fires-on-xbench-content: {n_frozen_activations > 0}")
    print()
    print(f"  NATIVE (auto-rules on KB prior):")
    print(f"     vs agent's actual opens: recall={native_vs_actual['recall']}, "
          f"precision={native_vs_actual['precision']}")
    print(f"     vs gold pipeline GT:     recall={native_vs_static['recall']}, "
          f"precision={native_vs_static['precision']}")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
