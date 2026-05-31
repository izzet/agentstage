"""Fig 6 - Trajectory-controlled I/O replay.

Paired bar chart, log-y. For each of three replayed AIOB cells, shows
cumulative cold-baseline and staged read times (median of 3 reps),
with the per-cell I/O speedup annotated above the pair.

Data source: outputs/trajectory_replay.json (produced by
scripts/microbench/trajectory_replay.py).

Outputs:
    paper/figures/fig_replay.{pdf,png}
    paper/figures/data/fig_replay.csv

Run:
    uv run python paper_figures/build_fig_replay.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _style import (  # noqa: E402
    DATA_DIR,
    FIG_DIR,
    FULL_COL_W,
    dump_csv,
    style_axis,
)

REPO = Path(__file__).resolve().parent.parent

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_COLD_COLOR = "#5b5b5b"
_STAGED_COLOR = "#2ca02c"


def load_results() -> list[dict]:
    p = REPO / "outputs" / "trajectory_replay.json"
    data = json.loads(p.read_text())
    return [r for r in data if "error" not in r]


def build(results: list[dict], out_name: str = "fig_replay") -> Path:
    fig, ax = plt.subplots(figsize=(FULL_COL_W, 1.8))

    n = len(results)
    x = np.arange(n)
    bar_w = 0.36
    cold_meds = [r["cold_median_ms"] / 1000.0 for r in results]
    staged_meds = [r["staged_median_ms"] / 1000.0 for r in results]
    cold_reps = [[v / 1000.0 for v in r["cold_ms"]] for r in results]
    staged_reps = [[v / 1000.0 for v in r["staged_ms"]] for r in results]

    bars_c = ax.bar(x - bar_w / 2, cold_meds, width=bar_w,
                     color=_COLD_COLOR, alpha=0.85,
                     edgecolor="white", linewidth=0.4, zorder=3,
                     label="Cold-baseline")
    bars_s = ax.bar(x + bar_w / 2, staged_meds, width=bar_w,
                     color=_STAGED_COLOR, alpha=0.85,
                     edgecolor="white", linewidth=0.4, zorder=3,
                     label="Staged (tmpfs)")

    # Overlay individual reps as small dots
    for i, reps in enumerate(cold_reps):
        for v in reps:
            ax.scatter(x[i] - bar_w / 2, v, color="black", s=6,
                        alpha=0.6, zorder=4, edgecolors="none")
    for i, reps in enumerate(staged_reps):
        for v in reps:
            ax.scatter(x[i] + bar_w / 2, v, color="black", s=6,
                        alpha=0.6, zorder=4, edgecolors="none")

    # Speedup annotation above each pair
    for i, r in enumerate(results):
        speedup = r["speedup"]
        ymax = max(cold_meds[i], staged_meds[i])
        ax.text(x[i], ymax * 3.0, f"{speedup:.1f}$\\times$",
                ha="center", va="bottom", fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18",
                          facecolor="white", edgecolor="#888888",
                          linewidth=0.4, alpha=0.95))

    # Annotation: numeric times on each bar
    for i, (cv, sv) in enumerate(zip(cold_meds, staged_meds)):
        # Cold time label
        ax.text(x[i] - bar_w / 2, cv * 1.10,
                f"{cv:.2f} s", ha="center", va="bottom", fontsize=8,
                color=_COLD_COLOR)
        # Staged time label
        ax.text(x[i] + bar_w / 2, sv * 1.10,
                f"{sv:.2f} s", ha="center", va="bottom", fontsize=8,
                color=_STAGED_COLOR)

    # x-axis labels (matplotlib doesn't interpret LaTeX escapes; use plain)
    _TASK_SLUG = {
        "aiob_104": "igsr-cov-qc",
        "aiob_107": "goes-r",
        "aiob_110": "steinmetz-nwb",
    }
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_TASK_SLUG.get(r["task"], r["task"]) for r in results]
    )
    ax.set_xlim(-0.5, n - 0.5)

    ax.set_yscale("log")
    ax.set_ylim(0.01, 80.0)
    style_axis(ax, xlabel="Workload", ylabel="Read Time (s, log)")

    # Legend in lower-left (clear of bars in steinmetz column which
    # are all above y=4) — keeps the plot clean and the xlabel free.
    ax.legend(loc="lower left", frameon=True, facecolor="white",
              framealpha=0.85, edgecolor="none",
              handlelength=1.0, handletextpad=0.4, fontsize=9)

    fig.tight_layout(pad=0.3)
    pdf = FIG_DIR / f"{out_name}.pdf"
    png = FIG_DIR / f"{out_name}.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=200)
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return pdf


def main() -> int:
    results = load_results()
    if not results:
        print("ERROR no replay results", file=sys.stderr)
        return 2
    print(f"Loaded {len(results)} replay cells")
    for r in results:
        print(f"  {r['task']:10s}: {r['n_files']:3d} files, "
              f"{r['total_bytes']/1e9:.2f} GB,  "
              f"cold {r['cold_median_ms']/1000:.2f}s -> "
              f"staged {r['staged_median_ms']/1000:.2f}s,  "
              f"speedup {r['speedup']:.2f}x")
    build(results)
    dump_csv(
        "fig_replay",
        [{"task": r["task"],
          "n_files": r["n_files"],
          "total_bytes": r["total_bytes"],
          "cold_median_s": round(r["cold_median_ms"] / 1000, 3),
          "staged_median_s": round(r["staged_median_ms"] / 1000, 3),
          "speedup": round(r["speedup"], 3),
          "cold_reps_s": ",".join(f"{v/1000:.3f}" for v in r["cold_ms"]),
          "staged_reps_s": ",".join(f"{v/1000:.3f}" for v in r["staged_ms"])}
         for r in results],
        ["task", "n_files", "total_bytes",
         "cold_median_s", "staged_median_s", "speedup",
         "cold_reps_s", "staged_reps_s"],
    )
    print(f"\nOutputs to {FIG_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
