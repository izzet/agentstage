"""Fig 6 — Community-benchmark session speedup: grouped bar chart.

Half-column subfigure (HALF_COL = 1.85x1.85). For each of the three
community benchmarks (MLE-bench, KramaBench, DSBench), four bars
(one per reasoning model) show the per-(benchmark, model)
arithmetic-mean session speedup across the 3 reps. Per-rep cells
are overlaid as dots above each bar.

Inputs: outputs/replay/_ofs_full/ (true-cold OrangeFS campaign: MLE, KB, DSB).
Outputs:
    paper/figures/fig_speedup_community.{pdf,png}
    paper/figures/data/fig_speedup_community.csv
"""
from __future__ import annotations

import glob
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _style import save, dump_csv, style_axis, label_bbox  # noqa: E402

# Raw size matches the curated figure (and the detection two-panel
# size) so on-paper font is identical across Fig 3/5/6. Extra height
# leaves room for the legend strip above the plot.
_SIZE = (2.0, 2.05)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FULL = REPO / "outputs" / "replay" / "_ofs_full"
_MODEL_FROM_ID = {
    "claude-haiku-4-5": "Haiku", "claude-sonnet-4-5": "Sonnet",
    "gemini-2.5-flash": "Flash", "Qwen/Qwen3.6-27B": "Qwen",
}
_BENCH_FROM_TASK = {
    "mle_dogsvcats_thumbhash": "MLE",
    "kb_astronomy_inventory": "KB",
    "tabular-playground-series-oct-2021": "DSB",
}

# Display order + labels: community benchmarks. Short x-tick labels
# (full names spelled out in the caption) keep the axis horizontal and
# uncrowded, matching the curated figure.
_BENCH_ORDER = ["MLE", "KB", "DSB"]
_BENCH_LABELS = {
    "MLE": "MLE",
    "KB":  "KB",
    "DSB": "DSB",
}

# Data keys match the model column in amdahl_analysis.csv; display
# labels match the detection figure (Fig 3) so legends agree across
# the paper.
_MODEL_ORDER = ["Haiku", "Sonnet", "Flash", "Qwen"]
_MODEL_LABEL = {
    "Haiku":  "Haiku",
    "Sonnet": "Sonnet",
    "Flash":  "Flash",
    "Qwen":   "Qwen3",
}
_MODEL_COLOR = {
    "Haiku":  "#d62728",
    "Sonnet": "#ff7f0e",
    "Flash":  "#1f77b4",
    "Qwen":   "#2ca02c",
}


def load() -> dict:
    """Returns {(bench, model): [sess_sp, ...]} from the true-cold campaign."""
    manifest = {r["cell"]: r for r in json.load(open(FULL / "manifest.json"))}
    cells = defaultdict(list)
    for sf in glob.glob(str(FULL / "*_staged.json")):
        cell = Path(sf).name[:-12]
        if cell not in manifest:
            continue
        cf = FULL / f"{cell}_cold.json"
        if not cf.exists():
            continue
        bench = _BENCH_FROM_TASK.get(manifest[cell]["task"])
        model = _MODEL_FROM_ID.get(manifest[cell]["model"])
        if bench is None or model is None:
            continue
        c = json.loads(cf.read_text())
        s = json.loads(Path(sf).read_text())
        if c["session_elapsed_s"] <= 1:
            continue
        cells[(bench, model)].append(c["session_elapsed_s"] / s["session_elapsed_s"])
    return cells


def main() -> None:
    cells = load()
    fig, ax = plt.subplots(figsize=_SIZE)

    n_models = len(_MODEL_ORDER)
    width = 0.22
    x = np.arange(len(_BENCH_ORDER))

    for i, model in enumerate(_MODEL_ORDER):
        offset = (i - (n_models - 1) / 2) * width
        means = []
        for bench in _BENCH_ORDER:
            vals = cells.get((bench, model), [])
            means.append(statistics.mean(vals) if vals else 0.0)
        ax.bar(
            x + offset, means, width=width * 0.92,
            color=_MODEL_COLOR[model], label=_MODEL_LABEL[model],
            linewidth=0, zorder=2,
        )

    # Per-benchmark mean line: one horizontal segment spanning each
    # benchmark's 4-bar cluster at the benchmark's mean speedup (mean
    # over all models and reps).
    mean_half = (n_models - 1) / 2 * width + (width * 0.92) / 2
    for bi, bench in enumerate(_BENCH_ORDER):
        gvals = [v for m in _MODEL_ORDER for v in cells.get((bench, m), [])]
        if not gvals:
            continue
        gm = statistics.mean(gvals)
        ax.plot([x[bi] - mean_half, x[bi] + mean_half], [gm, gm],
                color="black", linewidth=1.4, solid_capstyle="round",
                zorder=6)
        ax.text(x[bi], gm + 0.05, rf"{gm:.2f}$\times$",
                ha="center", va="bottom", zorder=7,
                bbox=label_bbox(alpha=0.75))

    ax.axhline(1.0, color="#888888", linewidth=0.7, linestyle="--", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([_BENCH_LABELS[b] for b in _BENCH_ORDER])
    # Tight x-limits so the bar groups fill the axes width edge-to-edge.
    ax.set_xlim(-0.5, len(_BENCH_ORDER) - 0.5)
    ax.set_ylim(0.0, 2.6)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
    style_axis(ax, xlabel="Community Benchmark", ylabel=r"Session Speedup ($\times$)")
    # Legend ABOVE the plot in 2 columns so it stays within the plot
    # width (a single 4-wide row would stretch the figure and leave the
    # plot floating with side gaps).
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False, handlelength=1.0, handletextpad=0.4,
              columnspacing=1.0, labelspacing=0.3, borderaxespad=0.0)

    save(fig, "fig_speedup_community")
    rows = []
    for (bench, model), vals in sorted(cells.items()):
        rows.append({
            "bench": _BENCH_LABELS[bench],
            "model": model,
            "n": len(vals),
            "mean": round(statistics.mean(vals), 3) if vals else 0,
            "per_rep": ",".join(f"{v:.3f}" for v in vals),
        })
    dump_csv("fig_speedup_community", rows,
             fieldnames=["bench", "model", "n", "mean", "per_rep"])
    print(f"  wrote {len(rows)} (bench, model) cells")


if __name__ == "__main__":
    main()
