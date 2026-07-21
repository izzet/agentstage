"""H12 path-compliance analysis (READ-ONLY).

Operationalizes the "Claim A" mechanism behind H12: even when explicitly
instructed to enumerate absolute file paths before each tool call, do
models actually emit usable literal paths in their reasoning?

The H12 pytest measures the OUTCOME (does pathful prompting lift the
predictor's HOT/tier-1 recall). This script measures the BEHAVIOR that
explains it: per run, how many literal file paths the model wrote in its
reasoning (thinking + assistant text) vs how many files it actually
touched (open_file calls + paths referenced in run_shell_command bodies).

  path_compliance = |reasoning-emitted file paths ∩ accessed files|
                    / |accessed files|

A compliance near 0 with accessed > 0 is the signal: the model ignored
the enumeration instruction, so the literal-path (HOT) scan gains nothing
— hence auto-rules over semantic intent are required, not prompting.

READ-ONLY: scans turns/ under a sweep dir, writes a single
compliance_report.json INTO that sweep dir (which we own). Touches
nothing the live campaign reads or writes.

Usage:
  uv run python scripts/microbench/analyze_h12_compliance.py \
      --sweep-dir outputs/aiob_mt/_sweep_pathful_<TS>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Path-like token starting at an absolute /data/ or relative data/ root.
# Stop at whitespace, quotes, parens, brackets, backslash, comma, semicolon.
_ABS_TOKEN = re.compile(r"/data/[^\s\"'`)\]\},;]+")
_REL_TOKEN = re.compile(r"(?<![\w/])data/[^\s\"'`)\]\},;]+")
# Looks like a file if it ends in a short extension (handles .bam.bai etc.
# by matching the final extension only).
_HAS_EXT = re.compile(r"\.[A-Za-z0-9]{1,6}$")


def _strip_trailing_punct(tok: str) -> str:
    return tok.rstrip(".:>)\"'`")


def _canonical(p: str) -> str:
    """Normalize absolute and relative forms to a comparable suffix:
    /data/ds/x.bam and data/ds/x.bam -> data/ds/x.bam."""
    p = _strip_trailing_punct(p)
    i = p.find("data/")
    return p[i:] if i != -1 else p


def _is_file(p: str) -> bool:
    return bool(_HAS_EXT.search(_strip_trailing_punct(p)))


def _read_deltas(path: Path) -> str:
    """Reconstruct streamed text from a *.jsonl of {'delta': ...} chunks."""
    if not path.is_file():
        return ""
    out = []
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


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _file_paths(text: str) -> set[str]:
    """Canonical file-looking paths (abs or rel) found in a blob."""
    found: set[str] = set()
    for m in _ABS_TOKEN.findall(text) + _REL_TOKEN.findall(text):
        tok = _strip_trailing_punct(m)
        if _is_file(tok):
            found.add(_canonical(tok))
    return found


def _reasoning_chars(run_dir: Path) -> int:
    """Total thinking + assistant-text chars across all turns. Works on the
    multiturn turns/ format (the only place reasoning is recorded for these
    runs — summary.json has no thinking_chars field)."""
    turns_dir = run_dir / "turns"
    if not turns_dir.is_dir():
        return 0
    total = 0
    for tdir in turns_dir.glob("turn_*"):
        total += len(_read_deltas(tdir / "thinking.jsonl"))
        total += len(_read_deltas(tdir / "text.jsonl"))
    return total


def _scan_baselines(baseline_root: Path) -> dict[tuple, list[int]]:
    """Map (task, model) -> [reasoning_chars] for every baseline+hinted run
    under baseline_root (one or two levels deep)."""
    out: dict[tuple, list[int]] = {}
    if not baseline_root.is_dir():
        return out
    candidates: list[Path] = []
    for p in baseline_root.iterdir():
        if not p.is_dir():
            continue
        if (p / "summary.json").is_file():
            candidates.append(p)
        else:
            for sub in p.iterdir():
                if sub.is_dir() and (sub / "summary.json").is_file():
                    candidates.append(sub)
    for run_dir in candidates:
        try:
            s = json.loads((run_dir / "summary.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if s.get("mode") != "baseline" or s.get("prompt_mode") != "hinted":
            continue
        key = (s.get("task"), s.get("model"))
        out.setdefault(key, []).append(_reasoning_chars(run_dir))
    return out


def _hot_recall(run_dir: Path, workload, reasoning_only: bool) -> float | None:
    """HOT (literal-path) byte-recall vs immediate-need GT for one run.

    reasoning_only=True scans thinking+text only (what the model actually
    reasoned/emitted); False also scans tool_result listings (the frozen
    byte_metrics_v1 behavior, which is saturated by directory-listing
    leakage). Returns None if the run has no turns/."""
    from agentstage.metrics.rescore import blocks_from_turns
    from agentstage.detector.engine import hot_path_scan
    from agentstage.metrics.byte_metrics import byte_score

    turns = run_dir / "turns"
    if not turns.is_dir():
        return None
    blocks = blocks_from_turns(turns)
    if reasoning_only:
        blocks = [b for b in blocks if b.type != "tool_result"]
    hot = hot_path_scan(blocks, workload.workspace_prior)
    return byte_score(tuple(hot.keys()),
                      workload.ground_truth_first_inspect,
                      workload.prefix_map).byte_recall


def _recall_pairs(rows: list[dict], sweep: Path, baseline_root: Path) -> dict | None:
    """For each pathful run, pair against matched baseline+hinted runs and
    compute HOT-recall deltas, both reasoning-only and tool_result-inclusive."""
    from agentstage.workloads import get_workload

    # index baseline+hinted run dirs by (task, model)
    base_dirs: dict[tuple, list[Path]] = {}
    for p in baseline_root.iterdir():
        if not p.is_dir():
            continue
        cands = [p] if (p / "summary.json").is_file() else [
            s for s in p.iterdir() if s.is_dir() and (s / "summary.json").is_file()]
        for run_dir in cands:
            try:
                s = json.loads((run_dir / "summary.json").read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if s.get("mode") == "baseline" and s.get("prompt_mode") == "hinted":
                base_dirs.setdefault((s.get("task"), s.get("model")), []).append(run_dir)

    pairs = []
    wcache: dict[str, object] = {}
    for r in rows:
        task, model = r["task"], r["model"]
        try:
            w = wcache.get(task) or wcache.setdefault(task, get_workload(task))
        except KeyError:
            continue
        pf_dir = sweep / r["run"]
        bdirs = base_dirs.get((task, model), [])
        if not bdirs:
            continue
        pf_reas = _hot_recall(pf_dir, w, reasoning_only=True)
        pf_all = _hot_recall(pf_dir, w, reasoning_only=False)
        b_reas = [v for d in bdirs if (v := _hot_recall(d, w, True)) is not None]
        b_all = [v for d in bdirs if (v := _hot_recall(d, w, False)) is not None]
        if pf_reas is None or not b_reas:
            continue
        bm_reas = sorted(b_reas)[len(b_reas) // 2]
        bm_all = sorted(b_all)[len(b_all) // 2]
        pairs.append({
            "task": task, "model": model, "baseline_n": len(bdirs),
            "reasoning_only": {
                "pathful": round(pf_reas, 4), "baseline_median": round(bm_reas, 4),
                "delta": round(pf_reas - bm_reas, 4)},
            "with_tool_result": {
                "pathful": round(pf_all, 4), "baseline_median": round(bm_all, 4),
                "delta": round(pf_all - bm_all, 4)},
        })
    if not pairs:
        return None
    dr = sorted(p["reasoning_only"]["delta"] for p in pairs)
    da = sorted(p["with_tool_result"]["delta"] for p in pairs)
    return {
        "n_pairs": len(pairs),
        "median_delta_reasoning_only": dr[len(dr) // 2],
        "median_delta_with_tool_result": da[len(da) // 2],
        "per_pair": pairs,
    }


def analyze_run(run_dir: Path) -> dict | None:
    summary_p = run_dir / "summary.json"
    turns_dir = run_dir / "turns"
    if not summary_p.is_file() or not turns_dir.is_dir():
        return None
    summary = json.loads(summary_p.read_text())

    reasoning = []           # thinking + assistant text (what the prompt addressed)
    toolcall_blob = []       # open_file paths + shell cmd bodies
    accessed: set[str] = set()

    for tdir in sorted(turns_dir.glob("turn_*")):
        reasoning.append(_read_deltas(tdir / "thinking.jsonl"))
        reasoning.append(_read_deltas(tdir / "text.jsonl"))
        for tu in _read_jsonl(tdir / "tool_use.jsonl"):
            name = tu.get("name")
            pin = tu.get("parsed_input") or {}
            if name == "open_file":
                p = pin.get("path")
                if p:
                    toolcall_blob.append(p)
                    if _is_file(p):
                        accessed.add(_canonical(p))
            elif name == "run_shell_command":
                cmd = pin.get("cmd") or ""
                toolcall_blob.append(cmd)
                # files the script actually reads/writes
                accessed |= _file_paths(cmd)
            elif name == "list_dir":
                # directory traversal — not a file access, but record the
                # path blob so we can see abs paths still only appear here
                p = pin.get("path")
                if p:
                    toolcall_blob.append(p)

    reasoning_text = "\n".join(reasoning)
    toolcall_text = "\n".join(toolcall_blob)

    reasoning_files = _file_paths(reasoning_text)
    # strict: absolute-only emission in reasoning (what was literally demanded)
    abs_reasoning_files = {
        _canonical(_strip_trailing_punct(m))
        for m in _ABS_TOKEN.findall(reasoning_text)
        if _is_file(_strip_trailing_punct(m))
    }

    matched = reasoning_files & accessed
    compliance = (len(matched) / len(accessed)) if accessed else None

    return {
        "run": run_dir.name,
        "task": summary.get("task"),
        "model": summary.get("model"),
        "prompt_mode": summary.get("prompt_mode"),
        "n_turns": summary.get("n_turns"),
        "reasoning_chars": len(reasoning_text),
        "n_files_accessed": len(accessed),
        "n_file_paths_in_reasoning": len(reasoning_files),
        "n_abs_file_paths_in_reasoning": len(abs_reasoning_files),
        "n_reasoning_accessed_match": len(matched),
        "path_compliance": (round(compliance, 4)
                            if compliance is not None else None),
        "files_accessed_sample": sorted(accessed)[:8],
        "reasoning_files_sample": sorted(reasoning_files)[:8],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=Path, required=True)
    ap.add_argument("--baseline-root", type=Path, default=None,
                    help="Root to scan for baseline+hinted runs to compute a "
                         "real reasoning-token overhead (e.g. outputs/aiob_mt). "
                         "Replaces the vacuous pytest thinking_chars=0 metric.")
    ap.add_argument("--out", type=Path, default=None,
                    help="JSON report path (default: <sweep-dir>/compliance_report.json)")
    args = ap.parse_args()

    sweep = args.sweep_dir
    if not sweep.is_dir():
        print(f"FATAL: sweep dir not found: {sweep}")
        return 2

    rows = []
    for run_dir in sorted(sweep.glob("*_pathful_r1")):
        r = analyze_run(run_dir)
        if r is not None:
            rows.append(r)

    if not rows:
        print("No analyzable runs found.")
        return 1

    # Per-run table
    print(f"\nH12 path-compliance — {sweep.name}\n")
    hdr = f"{'task':<10}{'model':<22}{'turns':>5}{'reas_ch':>9}{'accessed':>9}{'reas_files':>11}{'abs_in_reas':>12}{'compliance':>11}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        comp = "n/a" if r["path_compliance"] is None else f"{r['path_compliance']:.2f}"
        print(f"{(r['task'] or '?'):<10}{(r['model'] or '?'):<22}"
              f"{(r['n_turns'] or 0):>5}{r['reasoning_chars']:>9}"
              f"{r['n_files_accessed']:>9}{r['n_file_paths_in_reasoning']:>11}"
              f"{r['n_abs_file_paths_in_reasoning']:>12}{comp:>11}")

    # Aggregates
    def agg(subset):
        with_access = [r for r in subset if r["path_compliance"] is not None]
        n = len(subset)
        tot_abs = sum(r["n_abs_file_paths_in_reasoning"] for r in subset)
        comps = [r["path_compliance"] for r in with_access]
        med = (sorted(comps)[len(comps)//2] if comps else None)
        return {
            "n_runs": n,
            "n_runs_with_access": len(with_access),
            "total_abs_file_paths_in_reasoning": tot_abs,
            "runs_emitting_zero_abs_paths": sum(
                1 for r in subset if r["n_abs_file_paths_in_reasoning"] == 0),
            "median_compliance": (round(med, 4) if med is not None else None),
        }

    overall = agg(rows)
    by_model = {}
    for m in sorted({r["model"] for r in rows}):
        by_model[m] = agg([r for r in rows if r["model"] == m])

    print("\nAggregate:")
    print(f"  runs: {overall['n_runs']}  "
          f"(with file access: {overall['n_runs_with_access']})")
    print(f"  total absolute file paths emitted in reasoning across all runs: "
          f"{overall['total_abs_file_paths_in_reasoning']}")
    print(f"  runs emitting ZERO absolute file paths in reasoning: "
          f"{overall['runs_emitting_zero_abs_paths']}/{overall['n_runs']}")
    print(f"  median path-compliance: {overall['median_compliance']}")
    print("  by model:")
    for m, a in by_model.items():
        print(f"    {m:<22} zero-abs={a['runs_emitting_zero_abs_paths']}/{a['n_runs']}  "
              f"median_compliance={a['median_compliance']}")

    # Real reasoning-token overhead from turns/ (the pytest token test reads
    # thinking_chars from summary.json, which is absent on multiturn runs and
    # so reports a vacuous 0). Here we pair each pathful run against the
    # median reasoning size of matched baseline+hinted runs.
    overhead = None
    if args.baseline_root is not None:
        base = _scan_baselines(args.baseline_root)
        pairs = []
        for r in rows:
            key = (r["task"], r["model"])
            bvals = base.get(key, [])
            if not bvals:
                continue
            bmed = sorted(bvals)[len(bvals) // 2]
            pct = round(100 * (r["reasoning_chars"] - bmed) / max(1, bmed), 1)
            pairs.append({
                "task": r["task"], "model": r["model"],
                "pathful_reasoning_chars": r["reasoning_chars"],
                "baseline_reasoning_chars_median": bmed,
                "baseline_n": len(bvals),
                "overhead_pct": pct,
            })
        if pairs:
            ov = sorted(p["overhead_pct"] for p in pairs)
            overhead = {
                "n_pairs": len(pairs),
                "median_reasoning_overhead_pct": ov[len(ov) // 2],
                "min_overhead_pct": ov[0],
                "max_overhead_pct": ov[-1],
                "note": "reasoning_chars (thinking+text) is a char-count proxy "
                        "for tokens; baselines are heterogeneous across sweeps.",
                "per_pair": pairs,
            }
            print("\nReasoning-token overhead (pathful vs baseline+hinted, from turns/):")
            print(f"  {'task':<10}{'model':<22}{'pathful':>9}{'base_med':>10}"
                  f"{'base_n':>7}{'overhead%':>11}")
            for p in pairs:
                print(f"  {p['task']:<10}{p['model']:<22}"
                      f"{p['pathful_reasoning_chars']:>9}"
                      f"{p['baseline_reasoning_chars_median']:>10}"
                      f"{p['baseline_n']:>7}{p['overhead_pct']:>10}%")
            print(f"  median overhead: {overhead['median_reasoning_overhead_pct']}% "
                  f"(range {overhead['min_overhead_pct']}..{overhead['max_overhead_pct']}%)")

    # Paired HOT-recall deltas. The frozen byte_metrics_v1 HOT metric scans
    # tool_result listings and is ceiling-saturated by directory-listing
    # leakage; the reasoning-only variant is the meaningful predictive signal.
    recall = None
    if args.baseline_root is not None:
        recall = _recall_pairs(rows, sweep, args.baseline_root)
        if recall is not None:
            print("\nPaired HOT byte-recall delta (pathful − baseline median, vs gt_first):")
            print(f"  {'task':<10}{'model':<22}{'reas:base→path':>20}{'allTR:base→path':>20}")
            for p in recall["per_pair"]:
                ro, tr = p["reasoning_only"], p["with_tool_result"]
                print(f"  {p['task']:<10}{p['model']:<22}"
                      f"{ro['baseline_median']:.2f}→{ro['pathful']:.2f} ({ro['delta']:+.2f})".rjust(20)
                      + f"{tr['baseline_median']:.2f}→{tr['pathful']:.2f} ({tr['delta']:+.2f})".rjust(20))
            print(f"  median delta reasoning-only:   "
                  f"{recall['median_delta_reasoning_only']:+.4f}")
            print(f"  median delta with tool_result: "
                  f"{recall['median_delta_with_tool_result']:+.4f}  "
                  f"(ceiling-saturated)")

    out = args.out or (sweep / "compliance_report.json")
    out.write_text(json.dumps(
        {"sweep": sweep.name, "overall": overall, "by_model": by_model,
         "reasoning_token_overhead": overhead, "hot_recall_pairs": recall,
         "per_run": rows}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
