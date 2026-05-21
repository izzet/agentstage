"""Path B replay — run different detector configurations against a
captured multi-turn corpus.

For each captured run under outputs/multi_turn/<exp>/turns/turn_NN/, we
have stream.jsonl (events), tool_use.jsonl, tool_result.jsonl, and
thinking.txt. We reconstruct the StreamBlock list and run it through
four detector variants:

  A. thinking_only          — old behavior (text/tool_result ignored)
  B. thinking+tool_result   — engine.py extension, no session state
  C. thinking+text+tool_result    — full extension, single-shot
  D. session                — full extension via SessionDetector (multi-turn deltas)

Outputs a comparison table per run. Used for E-012 and E-013.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentstage.detector.engine import (
    RuleActivation,
    StreamBlock,
    run_detector,
)
from agentstage.detector.rules import get_ruleset
from agentstage.detector.session import SessionDetector
from agentstage.workloads.aiob import (
    load_aiob_107,
    load_aiob_107_s3,
    load_aiob_110,
)


def reconstruct_blocks(turns_dir: Path) -> list[StreamBlock]:
    """Walk turns_dir/turn_NN/ in order, emit StreamBlocks in chronological order."""
    blocks: list[StreamBlock] = []
    turn_dirs = sorted(turns_dir.glob("turn_*"))
    base_t_ms = 0.0
    for tdir in turn_dirs:
        turn_num = int(tdir.name.split("_")[1])
        # Parse stream.jsonl: accumulate thinking/text by block_idx, keep turn=turn_num
        thinking_by_idx: dict[int, list[str]] = {}
        text_by_idx: dict[int, list[str]] = {}
        block_kind: dict[int, str] = {}
        first_chunk_ms_by_idx: dict[int, float] = {}
        last_chunk_ms_by_idx: dict[int, float] = {}
        with (tdir / "stream.jsonl").open() as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = d.get("type")
                if etype == "content_block_start":
                    idx = d.get("block_idx", -1)
                    block_kind[idx] = d.get("block_type") or ""
                    first_chunk_ms_by_idx[idx] = d.get("t_ms", 0.0)
                elif etype == "content_block_delta":
                    idx = d.get("block_idx", -1)
                    last_chunk_ms_by_idx[idx] = d.get("t_ms", 0.0)
                    dtype = d.get("delta_type")
                    chunk = d.get("chunk", "")
                    if dtype == "thinking_delta":
                        thinking_by_idx.setdefault(idx, []).append(chunk)
                    elif dtype == "text_delta":
                        text_by_idx.setdefault(idx, []).append(chunk)
        for idx in sorted(block_kind.keys()):
            kind = block_kind[idx]
            tf = base_t_ms + first_chunk_ms_by_idx.get(idx, 0.0)
            ts = base_t_ms + last_chunk_ms_by_idx.get(idx, tf)
            if kind == "thinking":
                t_text = "".join(thinking_by_idx.get(idx, []))
                if t_text:
                    blocks.append(StreamBlock(
                        type="thinking", t_first=tf, t_stop=ts,
                        text=t_text, chunks=1, turn=turn_num,
                    ))
            elif kind == "text":
                tt = "".join(text_by_idx.get(idx, []))
                if tt:
                    blocks.append(StreamBlock(
                        type="text", t_first=tf, t_stop=ts,
                        text=tt, chunks=1, turn=turn_num,
                    ))
        # Add tool_result blocks BETWEEN turn N and turn N+1
        # (they're recorded in turn N's tool_result.jsonl, but apply to
        # the next turn's context — stamp turn=turn_num+1 to match the
        # live runner's stamping convention)
        tr_path = tdir / "tool_result.jsonl"
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
                blocks.append(StreamBlock(
                    type="tool_result",
                    t_first=base_t_ms + 1000.0 * (turn_num + 0.5),
                    t_stop=base_t_ms + 1000.0 * (turn_num + 0.5),
                    text=str(content), chunks=1, turn=turn_num + 1,
                ))
        base_t_ms += 10_000.0  # rough per-turn synthetic spacing
    return blocks


def filter_blocks(blocks: list[StreamBlock], *,
                  include_text: bool, include_tool_result: bool) -> list[StreamBlock]:
    out: list[StreamBlock] = []
    for b in blocks:
        if b.type == "thinking":
            out.append(b)
        elif b.type == "text" and include_text:
            out.append(b)
        elif b.type == "tool_result" and include_tool_result:
            out.append(b)
    return out


def summarize(activations: list[RuleActivation]) -> dict:
    by_source: dict[str, int] = {}
    by_turn: dict[int, list[str]] = {}
    for a in activations:
        by_source[a.source] = by_source.get(a.source, 0) + 1
        by_turn.setdefault(a.turn, []).append(a.rule_name)
    return {
        "n_activations": len(activations),
        "by_source": by_source,
        "by_turn": {t: rs for t, rs in sorted(by_turn.items())},
        "rule_names": sorted({a.rule_name for a in activations}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True,
                        help="Multi-turn run directory under outputs/multi_turn/")
    parser.add_argument("--workload",
                        choices=["aiob_107", "aiob_107_s3", "aiob_110"],
                        default="aiob_107_s3")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output JSON path for the replay comparison")
    args = parser.parse_args()

    loaders = {
        "aiob_107": load_aiob_107,
        "aiob_107_s3": load_aiob_107_s3,
        "aiob_110": load_aiob_110,
    }
    workload = loaders[args.workload]()
    rules_key = args.workload.replace("_s3", "")
    ruleset = get_ruleset(rules_key)
    prior = workload.workspace_prior

    turns_dir = args.corpus / "turns"
    if not turns_dir.exists():
        print(f"FATAL: no turns/ under {args.corpus}")
        return 2

    blocks = reconstruct_blocks(turns_dir)
    print(f"Reconstructed {len(blocks)} StreamBlocks from {args.corpus}")
    counts = {}
    for b in blocks:
        counts[b.type] = counts.get(b.type, 0) + 1
    print(f"  Counts: {counts}")

    # Variant A: thinking only (legacy behavior)
    A_blocks = filter_blocks(blocks, include_text=False, include_tool_result=False)
    A_pred = run_detector(A_blocks, prior, ruleset)

    # Variant B: thinking + tool_result (no text)
    B_blocks = filter_blocks(blocks, include_text=False, include_tool_result=True)
    B_pred = run_detector(B_blocks, prior, ruleset)

    # Variant C: thinking + text + tool_result (full)
    C_blocks = filter_blocks(blocks, include_text=True, include_tool_result=True)
    C_pred = run_detector(C_blocks, prior, ruleset)

    # Variant D: SessionDetector — feed each turn separately to confirm
    # delta tracking works and produces the same total activation set.
    sp = SessionDetector(prior=prior, ruleset=ruleset)
    # Bucket the blocks by turn for streaming-style feed
    blocks_by_turn: dict[int, list[StreamBlock]] = {}
    for b in blocks:
        blocks_by_turn.setdefault(b.turn, []).append(b)
    for turn in sorted(blocks_by_turn.keys()):
        turn_blocks = blocks_by_turn[turn]
        # Split assistant blocks (thinking/text) vs user tool_result
        assistant = [b for b in turn_blocks if b.type in ("thinking", "text")]
        tool_results = [b for b in turn_blocks if b.type == "tool_result"]
        if assistant:
            sp.feed_turn(assistant)
        if tool_results:
            sp.feed_tool_results(tool_results)
    D_pred = sp.cumulative_detection()

    result = {
        "corpus": str(args.corpus),
        "n_blocks": {
            "all": len(blocks),
            "thinking": counts.get("thinking", 0),
            "text": counts.get("text", 0),
            "tool_result": counts.get("tool_result", 0),
        },
        "variants": {
            "A_thinking_only": summarize(list(A_pred.activations)),
            "B_thinking_plus_tool_result": summarize(list(B_pred.activations)),
            "C_full": summarize(list(C_pred.activations)),
            "D_session_detector": summarize(list(D_pred.activations)),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    # Pretty-print
    print()
    print(f"{'Variant':<32} {'rules':<6} {'sources':<40}")
    print("-" * 80)
    for name, summary in result["variants"].items():
        srcs = ", ".join(f"{k}={v}" for k, v in summary["by_source"].items())
        print(f"{name:<32} {summary['n_activations']:<6} {srcs}")
    print()
    print(f"Variant A rules: {result['variants']['A_thinking_only']['rule_names']}")
    print(f"Variant C rules: {result['variants']['C_full']['rule_names']}")
    print()
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
