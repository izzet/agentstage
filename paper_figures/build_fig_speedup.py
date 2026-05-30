"""Fig 4 - session speedup across the result trio.

Single panel, full-column wide (FULL_COL_W = 3.85"). Per-(benchmark,
model) grouped bar chart of session speedup; per-cell speedup is the
ratio of baseline-mode median elapsed time to staged-mode median
elapsed time, computed on 3 baseline + 3 staged sessions per cell.

Bars are colored by model family; benchmarks are grouped on the x-axis.
A 1.5x reference line marks the headline win threshold. Hero cells are
labeled inline above the bar where they cross 2.5x.

Outputs:
    paper/figures/fig_speedup.{pdf,png}
    paper/figures/data/fig_speedup.csv
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _style import (  # noqa: E402
    DATA_DIR,
    FIG_DIR,
    FULL_COL_W,
    dump_csv,
    style_axis,
)

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

_BENCH_ORDER = ["aiob", "dsbench", "mlebench"]
_BENCH_LABELS = {
    "aiob": "AIOB",
    "dsbench": "DSBench",
    "mlebench": "MLE-bench",
}
_MODEL_ORDER = ["haiku", "sonnet", "flash", "qwen3"]
_MODEL_LABELS = {
    "haiku": "Haiku",
    "sonnet": "Sonnet",
    "flash": "Flash",
    "qwen3": "Qwen3",
}
_MODEL_COLOR = {
    "haiku": "#d62728",   # red
    "sonnet": "#ff7f0e",  # orange
    "flash": "#1f77b4",   # blue
    "qwen3": "#2ca02c",   # green
}


def _model_key(m: str) -> str | None:
    m = (m or "").lower()
    if "claude-haiku" in m:
        return "haiku"
    if "claude-sonnet" in m:
        return "sonnet"
    if "gemini-2.5-flash" in m or "gemini-flash" in m:
        return "flash"
    if "qwen" in m:
        return "qwen3"
    return None


def load_cells() -> list[dict]:
    """Walk outputs/*_mt/, exclude smoke sweeps, keep the most recent 3
    sessions per (bench, model, task, mode) by file mtime, and emit one
    per-cell row with median(baseline) / median(staged) session speedup."""
    raw: list[dict] = []
    for sf in (REPO / "outputs").rglob("summary.json"):
        if "_archive" in sf.parts:
            continue
        rel = sf.relative_to(REPO / "outputs")
        if len(rel.parts) < 2 or not rel.parts[0].endswith("_mt"):
            continue
        if rel.parts[1].startswith("_smoke"):
            continue
        try:
            s = json.loads(sf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        mk = _model_key(s.get("model") or "")
        if mk is None:
            continue
        mode = s.get("mode")
        if mode not in ("baseline", "staged"):
            continue
        elapsed = s.get("session_elapsed_s")
        if elapsed is None or float(elapsed) < 5.0:
            continue
        bench = rel.parts[0].replace("_mt", "")
        raw.append({
            "bench": bench,
            "model": mk,
            "task": s.get("task"),
            "mode": mode,
            "elapsed_s": float(elapsed),
            "mtime": sf.stat().st_mtime,
        })

    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for r in raw:
        by_key[(r["bench"], r["model"], r["task"], r["mode"])].append(r)
    latest3: list[dict] = []
    for k, rs in by_key.items():
        rs_sorted = sorted(rs, key=lambda x: -x["mtime"])[:3]
        latest3.extend(rs_sorted)

    by_cell: dict[tuple, dict] = defaultdict(lambda: {"baseline": [], "staged": []})
    for r in latest3:
        by_cell[(r["bench"], r["model"], r["task"])][r["mode"]].append(r["elapsed_s"])

    cells: list[dict] = []
    for (b, mk, t), d in by_cell.items():
        if len(d["baseline"]) < 2 or len(d["staged"]) < 2:
            continue
        bs = statistics.median(d["baseline"])
        ss = statistics.median(d["staged"])
        if bs <= 0 or ss <= 0:
            continue
        cells.append({
            "bench": b,
            "model": mk,
            "task": t,
            "baseline_med_s": bs,
            "staged_med_s": ss,
            "speedup": bs / ss,
            "n_baseline": len(d["baseline"]),
            "n_staged": len(d["staged"]),
        })
    return cells


def _gmean(vals: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def build(cells: list[dict], out_name: str = "fig_speedup") -> Path:
    by_bm: dict[tuple, list[dict]] = defaultdict(list)
    for c in cells:
        by_bm[(c["bench"], c["model"])].append(c)

    fig, ax = plt.subplots(figsize=(FULL_COL_W, 2.4))

    n_benches = len(_BENCH_ORDER)
    n_models = len(_MODEL_ORDER)
    bar_width = 0.18
    group_pad = 0.4

    bench_positions: list[float] = []
    handles: list[mpatches.Patch] = []
    labels: list[str] = []

    x_cursor = 0.0
    for bi, bench in enumerate(_BENCH_ORDER):
        group_left = x_cursor
        for mi, mk in enumerate(_MODEL_ORDER):
            speedups = [c["speedup"] for c in by_bm.get((bench, mk), [])]
            if not speedups:
                x_cursor += bar_width
                continue
            am = statistics.mean(speedups)
            x = x_cursor + bar_width / 2
            ax.bar(
                x, am, width=bar_width * 0.92,
                color=_MODEL_COLOR[mk],
                edgecolor="white", linewidth=0.4, zorder=3,
            )
            for s in speedups:
                ax.scatter(x, s, s=9, color="black",
                           alpha=0.55, zorder=4,
                           edgecolors="none")
            x_cursor += bar_width

            if bi == 0:
                handles.append(mpatches.Patch(
                    facecolor=_MODEL_COLOR[mk],
                    edgecolor="white", linewidth=0.4,
                    label=_MODEL_LABELS[mk],
                ))
                labels.append(_MODEL_LABELS[mk])
        bench_positions.append(group_left + bar_width * n_models / 2)
        x_cursor += group_pad

    ax.set_xticks(bench_positions)
    ax.set_xticklabels([_BENCH_LABELS[b] for b in _BENCH_ORDER])
    ax.set_xlim(-0.15, x_cursor - group_pad + 0.15)

    ax.axhline(1.0, color="#444444", linewidth=0.6,
               linestyle="-", zorder=2, alpha=0.6)
    ax.axhline(1.5, color="#888888", linewidth=0.6,
               linestyle="--", zorder=2, alpha=0.7)
    ax.text(x_cursor - group_pad - 0.05, 1.5, r"1.5$\times$",
            va="center", ha="right", color="#444444",
            bbox=dict(boxstyle="round,pad=0.12",
                      facecolor="white", edgecolor="none", alpha=0.9))

    y_max = max([c["speedup"] for c in cells]) * 1.10
    ax.set_ylim(0.0, y_max)
    style_axis(ax, ylabel=r"Session Speedup ($\times$)")

    ax.legend(handles, labels, loc="upper left",
              bbox_to_anchor=(0.0, 1.04), ncol=4, frameon=False,
              handlelength=0.9, columnspacing=0.7, handletextpad=0.3)

    fig.tight_layout(pad=0.3)
    pdf = FIG_DIR / f"{out_name}.pdf"
    png = FIG_DIR / f"{out_name}.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=200)
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return pdf


def report_stats(cells: list[dict]) -> None:
    print("\nFig 4 summary statistics")
    print("=" * 78)
    sess = [c["speedup"] for c in cells]
    print(f"All {len(cells)} cells: mean={statistics.mean(sess):.2f}x  "
          f"gmean={_gmean(sess):.2f}x  median={statistics.median(sess):.2f}x  "
          f"max={max(sess):.2f}x  >=1.5x: "
          f"{sum(1 for v in sess if v >= 1.5)}/{len(sess)}")
    print()
    print("Per-benchmark:")
    for b in _BENCH_ORDER:
        bc = [c["speedup"] for c in cells if c["bench"] == b]
        if not bc:
            continue
        print(f"  {_BENCH_LABELS[b]:10s}  n={len(bc):2d}  "
              f"mean={statistics.mean(bc):.2f}x  gmean={_gmean(bc):.2f}x  "
              f"median={statistics.median(bc):.2f}x  max={max(bc):.2f}x")
    print()
    print("Per-model:")
    for mk in _MODEL_ORDER:
        mc = [c["speedup"] for c in cells if c["model"] == mk]
        if not mc:
            continue
        print(f"  {_MODEL_LABELS[mk]:10s}  n={len(mc):2d}  "
              f"mean={statistics.mean(mc):.2f}x  gmean={_gmean(mc):.2f}x  "
              f"median={statistics.median(mc):.2f}x  max={max(mc):.2f}x")
    print()
    print("Top 5 cells:")
    for c in sorted(cells, key=lambda c: -c["speedup"])[:5]:
        print(f"  {c['bench']:9s}/{c['model']:7s}/{c['task'][:40]:40s}  "
              f"{c['speedup']:.2f}x  ({c['baseline_med_s']:.0f}s -> "
              f"{c['staged_med_s']:.0f}s)")


def main() -> int:
    cells = load_cells()
    if not cells:
        print("ERROR: no cells loaded", file=sys.stderr)
        return 2
    print(f"Loaded {len(cells)} cells")

    build(cells)
    report_stats(cells)

    dump_csv(
        "fig_speedup",
        [
            {
                "bench": c["bench"],
                "model": c["model"],
                "task": c["task"],
                "baseline_med_s": round(c["baseline_med_s"], 2),
                "staged_med_s": round(c["staged_med_s"], 2),
                "speedup": round(c["speedup"], 4),
                "n_baseline": c["n_baseline"],
                "n_staged": c["n_staged"],
            }
            for c in sorted(cells, key=lambda c: (c["bench"], c["model"], c["task"]))
        ],
        ["bench", "model", "task", "baseline_med_s", "staged_med_s",
         "speedup", "n_baseline", "n_staged"],
    )

    print(f"\nOutputs written to {FIG_DIR}/")
    print(f"CSV data written to {DATA_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
