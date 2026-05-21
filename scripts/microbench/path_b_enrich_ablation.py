"""Precision-tuning ablation for dynamic prior enrichment (E-023).

Reproduces a captured corpus's tool_result extraction under several
enrichment policies and reports recall vs precision vs byte_overfetch.
The original `enrich_prior_from_tool_result` adds ALL files in each
list_dir result. This script measures the trade-off as we tighten:

  A. no_enrich           — baseline (rules + static prior only)
  B. all_files           — current default
  C. cap_N <K>           — keep only first K files per listing
  D. pattern_scope       — only files whose name shares ≥4 chars with
                           any token the LLM has mentioned in this session
  E. ext_subset          — restrict to a subset of extensions (e.g. only .nc)

Replays offline against a captured run's tool_results and tool_use
trail. Computes per-policy:
  - prior size delta (how much we added)
  - recall@actual: did agent's opened file end up in the enriched prior?
  - byte_overfetch: bytes_in_enriched_set / bytes_agent_opened

Usage:
    python scripts/microbench/path_b_enrich_ablation.py \\
        --corpus outputs/multi_turn/<E-021 run> \\
        --workload aiob_107_s3 \\
        --out <corpus>/enrich_ablation.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from agentstage.runners.path_b_multiturn import _resolve_logical_to_physical
from agentstage.workloads.aiob import (
    load_aiob_104,
    load_aiob_107,
    load_aiob_107_s3,
    load_aiob_110,
)

LOADERS = {
    "aiob_104": load_aiob_104,
    "aiob_107": load_aiob_107,
    "aiob_107_s3": load_aiob_107_s3,
    "aiob_110": load_aiob_110,
}

# Match "  FILE  <path>  (N bytes)" lines in the tool_result text
_FILE_LINE = re.compile(
    r"^\s*FILE\s+(\S+)\s+\((\d+)\s+bytes\)\s*$",
    re.MULTILINE,
)


def extract_files_from_tool_result(text: str) -> list[tuple[str, int]]:
    """Return [(path, size_bytes), ...] from a list_dir tool_result."""
    out: list[tuple[str, int]] = []
    for m in _FILE_LINE.finditer(text):
        try:
            out.append((m.group(1), int(m.group(2))))
        except ValueError:
            continue
    return out


def collect_corpus_evidence(
    corpus: Path,
    prefix_map: tuple[tuple[str, str], ...],
    cold_root: str,
) -> dict:
    """Walk turns_dir; per turn return tool_result file lists + the
    agent's open targets across all turns (PHYSICAL paths)."""
    per_turn_listings: list[list[tuple[str, int]]] = []
    agent_opens: set[str] = set()
    llm_mentions: set[str] = set()  # tokens from thinking/text — used by D
    for tdir in sorted((corpus / "turns").glob("turn_*")):
        # Collect tool_result text for this turn
        tr_path = tdir / "tool_result.jsonl"
        listings: list[tuple[str, int]] = []
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
                listings.extend(extract_files_from_tool_result(str(content)))
        per_turn_listings.append(listings)

        # Collect agent opens
        tu_path = tdir / "tool_use.jsonl"
        if tu_path.exists():
            for line in tu_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("name") not in ("open_file", "read_file"):
                    continue
                logical = (d.get("parsed_input") or {}).get("path", "")
                if logical:
                    agent_opens.add(logical)

        # Collect LLM mentions (token bag from thinking + text)
        for sf in ("thinking.txt",):
            f = tdir / sf
            if f.is_file():
                txt = f.read_text()
                for tok in re.findall(r"[A-Za-z0-9_]{4,}", txt):
                    llm_mentions.add(tok)
        # Also scan stream.jsonl for text_delta chunks
        stream = tdir / "stream.jsonl"
        if stream.is_file():
            for line in stream.read_text().splitlines():
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("delta_type") in ("text_delta", "thinking_delta"):
                    for tok in re.findall(r"[A-Za-z0-9_]{4,}", d.get("chunk") or ""):
                        llm_mentions.add(tok)

    return {
        "per_turn_listings": per_turn_listings,
        "agent_opens": agent_opens,
        "llm_mentions": llm_mentions,
    }


def policy_no_enrich(per_turn_listings):
    """Baseline: keep nothing."""
    return []


def policy_all(per_turn_listings):
    """Current default: keep everything."""
    out: list[tuple[str, int]] = []
    for L in per_turn_listings:
        out.extend(L)
    return out


def policy_cap_n(per_turn_listings, k: int):
    """Keep first K files per listing."""
    out: list[tuple[str, int]] = []
    for L in per_turn_listings:
        out.extend(L[:k])
    return out


def policy_pattern(per_turn_listings, llm_mentions: set[str]):
    """Keep only files whose basename shares ≥4-character substring with
    any token in llm_mentions."""
    mentions_lc = {m.lower() for m in llm_mentions if len(m) >= 4}
    out: list[tuple[str, int]] = []
    for L in per_turn_listings:
        for path, size in L:
            base = path.rsplit("/", 1)[-1].lower()
            kept = False
            for m in mentions_lc:
                if m in base:
                    kept = True
                    break
            if kept:
                out.append((path, size))
    return out


def policy_ext(per_turn_listings, exts: tuple[str, ...]):
    """Keep only files with extension in `exts`."""
    out: list[tuple[str, int]] = []
    for L in per_turn_listings:
        for path, size in L:
            if any(path.endswith(e) for e in exts):
                out.append((path, size))
    return out


def evaluate(kept: list[tuple[str, int]], agent_opens: set[str]) -> dict:
    paths = {p for p, _ in kept}
    sizes = {p: s for p, s in kept}
    hits = paths & agent_opens
    opened_size = sum(sizes.get(p, 0) for p in agent_opens & paths)
    # Approximate "agent opened size" with sum of hit sizes (we don't know
    # sizes of agent opens that aren't in kept). For overfetch we use
    # kept_bytes / opened_bytes_of_hits.
    total_size = sum(sizes.values())
    return {
        "kept_files": len(paths),
        "kept_bytes": total_size,
        "agent_opens": len(agent_opens),
        "hits": len(hits),
        "recall": (len(hits) / len(agent_opens)) if agent_opens else None,
        "precision": (len(hits) / len(paths)) if paths else None,
        "byte_overfetch_vs_hits": (total_size / opened_size) if opened_size > 0 else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--workload",
                        choices=list(LOADERS.keys()), default="aiob_107_s3")
    parser.add_argument("--cold-root",
                        default="/tmp/s3-noaa-goes16/ABI-L2-CMIPC")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    workload = LOADERS[args.workload]()
    evidence = collect_corpus_evidence(
        args.corpus, workload.prefix_map, args.cold_root)
    per_turn = evidence["per_turn_listings"]
    agent_opens = evidence["agent_opens"]
    mentions = evidence["llm_mentions"]

    print(f"Corpus: {args.corpus.name}")
    print(f"  Tool_result listings: {sum(len(L) for L in per_turn)} files across "
          f"{len(per_turn)} turns")
    print(f"  Agent opens (logical paths): {len(agent_opens)}")
    print(f"  Distinct LLM tokens (≥4 chars): {len(mentions)}")

    policies = {
        "A_no_enrich":       policy_no_enrich(per_turn),
        "B_all_files":       policy_all(per_turn),
        "C_cap_5":           policy_cap_n(per_turn, 5),
        "C_cap_10":          policy_cap_n(per_turn, 10),
        "C_cap_25":          policy_cap_n(per_turn, 25),
        "D_pattern_scoped":  policy_pattern(per_turn, mentions),
        "E_ext_nc_only":     policy_ext(per_turn, (".nc",)),
    }

    results = {}
    print(f"  {'policy':<22} {'kept':>6} {'hits':>5} {'recall':>8} {'precision':>11} {'byte_over':>10}")
    for name, kept in policies.items():
        m = evaluate(kept, agent_opens)
        results[name] = m
        rec = m["recall"]
        prec = m["precision"]
        over = m["byte_overfetch_vs_hits"]
        print(f"  {name:<22} {m['kept_files']:>6} {m['hits']:>5} "
              f"{rec*100 if rec is not None else 0:>7.1f}% "
              f"{prec*100 if prec is not None else 0:>10.1f}% "
              f"{over if over else 0:>9.1f}×")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "corpus": str(args.corpus),
        "workload": args.workload,
        "n_turns": len(per_turn),
        "n_tool_result_files": sum(len(L) for L in per_turn),
        "n_agent_opens": len(agent_opens),
        "n_llm_mentions": len(mentions),
        "policies": results,
    }, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
