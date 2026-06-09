"""Larger-N reasoning-pathfulness scan over the 72-cell true-cold pool.

For every cell in outputs/amdahl_analysis.csv (the 72-cell paper pool), find
the source agentic run dir (the _sweep_*/<cell> dir that owns the turns/
reasoning streams) and measure how often the agent's REASONING TEXT
(thinking.jsonl + text.jsonl, concatenated across turns) contains a COMPLETE
file path.

"Complete file path" definition (verbatim regex from analyze_h12_compliance.py):
  a token rooted at /data/... (absolute) or data/... (relative-to-CWD) that ends
  in a short file extension (\\.[A-Za-z0-9]{1,6}$). A bare directory root like
  /data/astronomy/ does NOT count (no extension); neither does prose or a glob
  like /data/astronomy/raw/*.fits.

READ-ONLY. Prints a per-cell table and the aggregate. Writes nothing.
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AMDAHL = REPO / "outputs" / "amdahl_analysis.csv"
ROOTS = [REPO / "outputs" / "aiob_mt",
         REPO / "outputs" / "dsbench_mt",
         REPO / "outputs" / "mlebench_mt"]

# --- path tokens: copied verbatim from analyze_h12_compliance.py ---
_ABS_TOKEN = re.compile(r"/data/[^\s\"'`)\]\},;]+")
_REL_TOKEN = re.compile(r"(?<![\w/])data/[^\s\"'`)\]\},;]+")
_HAS_EXT = re.compile(r"\.[A-Za-z0-9]{1,6}$")


def _strip(t): return t.rstrip(".:>)\"'`")
def _canon(p):
    p = _strip(p); i = p.find("data/"); return p[i:] if i != -1 else p
def _is_file(p): return bool(_HAS_EXT.search(_strip(p)))


def _deltas(path: Path) -> str:
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


def _complete_paths(text: str) -> set[str]:
    found = set()
    for m in _ABS_TOKEN.findall(text) + _REL_TOKEN.findall(text):
        tok = _strip(m)
        if _is_file(tok):
            found.add(_canon(tok))
    return found


def _abs_complete_paths(text: str) -> set[str]:
    return {_canon(_strip(m)) for m in _ABS_TOKEN.findall(text) if _is_file(_strip(m))}


def find_run_dir(campaign: str, cell: str) -> Path | None:
    for root in ROOTS:
        rd = root / f"_sweep_{campaign}" / cell
        if rd.is_dir():
            return rd
    for root in ROOTS:
        if not root.is_dir():
            continue
        for d in root.iterdir():
            if d.is_dir() and campaign in d.name and (d / cell).is_dir():
                return d / cell
    return None


def reasoning_for(run_dir: Path) -> str:
    turns = run_dir / "turns"
    if not turns.is_dir():
        return ""
    parts = []
    for tdir in sorted(turns.glob("turn_*")):
        parts.append(_deltas(tdir / "thinking.jsonl"))
        parts.append(_deltas(tdir / "text.jsonl"))
    return "\n".join(parts)


def main() -> int:
    rows = list(csv.DictReader(AMDAHL.open()))
    per = []
    for r in rows:
        rd = find_run_dir(r["campaign"], r["cell"])
        if rd is None:
            per.append({**r, "found": False})
            continue
        txt = reasoning_for(rd)
        cp = _complete_paths(txt)
        ap = _abs_complete_paths(txt)
        per.append({
            "bench": r["bench"], "cell": r["cell"],
            "found": True, "reasoning_chars": len(txt),
            "n_complete": len(cp), "n_abs_complete": len(ap),
        })

    ok = [p for p in per if p.get("found")]
    N = len(ok)
    zero_complete = sum(1 for p in ok if p["n_complete"] == 0)
    zero_abs = sum(1 for p in ok if p["n_abs_complete"] == 0)
    counts = sorted(p["n_complete"] for p in ok)
    med = counts[len(counts) // 2]

    print(f"{'bench':<6}{'cell':<46}{'reas_ch':>9}{'complete':>9}{'abs':>5}")
    print("-" * 75)
    for p in ok:
        print(f"{p['bench']:<6}{p['cell']:<46}{p['reasoning_chars']:>9}"
              f"{p['n_complete']:>9}{p['n_abs_complete']:>5}")

    print(f"\nN (cells with reasoning streams) = {N} / {len(per)}")
    print(f"cells emitting ZERO complete file paths in reasoning (abs OR rel): "
          f"{zero_complete}/{N} = {100*zero_complete/N:.1f}%")
    print(f"cells emitting ZERO absolute complete paths (strict):             "
          f"{zero_abs}/{N} = {100*zero_abs/N:.1f}%")
    print(f"median complete paths per reasoning trace: {med}")
    print(f"max complete paths in any trace: {max(counts)}")

    byb = defaultdict(list)
    for p in ok:
        byb[p["bench"]].append(p)
    print("\nby bench (zero-complete-paths / total):")
    for b, lst in sorted(byb.items()):
        z = sum(1 for p in lst if p["n_complete"] == 0)
        print(f"  {b:<5} {z}/{len(lst)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
