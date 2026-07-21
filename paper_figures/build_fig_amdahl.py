"""Build Fig:amdahl — observed session speedup vs Amdahl-predicted ceiling
across the 72 paper-grade cells, colored by benchmark.

Data source: outputs/amdahl_analysis.csv (produced by scripts/microbench/
amdahl_analysis.py with the PAPER_GRADE_CELLS filter).

Each point is one (benchmark, task, model, rep) cell:
  x = amdahl_ceiling (predicted session speedup ceiling)
  y = session_sp (observed session speedup)

The diagonal y = x is plotted as reference; points above the diagonal mean
observed exceeds the proxy ceiling (which happens for AIOB cells where the
shim+stager also reduce metadata-RPC time the io_share proxy doesn't count).
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from _style import FULL_COL_W, save, dump_csv

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "outputs" / "amdahl_analysis.csv"

# Bench palette — distinct from the model palette so the two figure
# families are visually disjoint at a glance.
BENCH_ORDER = ["AIOB", "MLE", "KB", "DSB"]
BENCH_LABELS = ["Curated", "MLE-bench", "KramaBench", "DSBench"]
BENCH_COLORS = {
    "AIOB": "#d62728",  # red
    "MLE":  "#1f77b4",  # blue
    "KB":   "#2ca02c",  # green
    "DSB":  "#9467bd",  # purple
}
BENCH_MARKERS = {
    "AIOB": "o",
    "MLE":  "s",
    "KB":   "^",
    "DSB":  "D",
}


def main() -> None:
    rows = list(csv.DictReader(open(CSV)))
    points = [(float(r["amdahl_ceiling"]), float(r["session_sp"]), r["bench"])
              for r in rows]

    # Rectangle (wider than tall): keep equal data aspect for the y=x
    # diagonal to read cleanly, but stop the figure from dominating the
    # column with a tall square.
    fig, ax = plt.subplots(figsize=(FULL_COL_W, FULL_COL_W * 0.55))

    # Plot points per bench
    for bench, label in zip(BENCH_ORDER, BENCH_LABELS):
        xs = [p[0] for p in points if p[2] == bench]
        ys = [p[1] for p in points if p[2] == bench]
        if not xs:
            continue
        ax.scatter(xs, ys,
                   c=BENCH_COLORS[bench],
                   marker=BENCH_MARKERS[bench],
                   s=28, alpha=0.75,
                   edgecolors="white", linewidths=0.4,
                   label=f"{label} (n={len(xs)})")

    # Reference: y = x diagonal. Independent (untied) x/y limits so the
    # scatter fills the wide panel: the ceiling tops out near 1.95 while
    # observed reaches ~2.41, so matched limits would leave a large empty
    # band on the right. Without equal-aspect the diagonal is not a literal
    # 45 degrees, but it is still the observed==ceiling locus (a point lies
    # above it iff observed beats the predicted ceiling).
    x_lo, x_hi = 0.9, 2.05
    y_lo, y_hi = 0.9, 2.5
    ax.plot([y_lo, y_hi], [y_lo, y_hi],
            linestyle="--", color="#666666", linewidth=0.8, zorder=0,
            label=r"$y = x$ (ceiling)")

    # x = 1.0 / y = 1.0 references (no-speedup baselines)
    ax.axhline(1.0, color="#aaaaaa", linewidth=0.5, linestyle=":", zorder=0)
    ax.axvline(1.0, color="#aaaaaa", linewidth=0.5, linestyle=":", zorder=0)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.grid(True, axis="both", linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, pad=2)
    ax.set_xlabel("Amdahl-Predicted Ceiling")
    ax.set_ylabel("Observed Session Speedup")
    ax.legend(loc="upper left", frameon=True, facecolor="white",
              framealpha=0.7, edgecolor="none", handletextpad=0.4,
              borderaxespad=0.3, labelspacing=0.3)

    # Annotation: R^2
    ax.text(0.96, 0.04,
            r"$\mathrm{R}^2 = 0.901$, Pearson $r = 0.949$, $n = 72$",
            transform=ax.transAxes,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.10",
                      facecolor="white", edgecolor="none"))

    pdf, png = save(fig, "fig_amdahl")
    dump_csv("fig_amdahl",
             [{"bench": p[2], "ceiling": p[0], "observed": p[1]}
              for p in points],
             fieldnames=["bench", "ceiling", "observed"])
    print(f"  wrote {pdf}")
    print(f"  wrote {png}")


if __name__ == "__main__":
    main()
