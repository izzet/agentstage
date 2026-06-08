"""Fig 5 — Curated session speedup: grouped bar chart.

Half-column subfigure (HALF_COL = 1.85x1.85). For each of the three
curated tasks (igsr-integrity, jwst-integrity, cross-archive-integrity),
four bars (one per reasoning model) show the per-(task, model)
arithmetic-mean session speedup across the 3 reps. Per-rep cells are
overlaid as dots above each bar.

Inputs: outputs/replay/_ofs_full/ (true-cold OrangeFS campaign, curated tasks).
Outputs:
    paper/figures/fig_speedup_curated.{pdf,png}
    paper/figures/data/fig_speedup_curated.csv
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

# Raw size matches the detection two-panel size (2.0 wide) so text
# downscales by the same ~0.83 factor when placed at 0.48\columnwidth,
# keeping on-paper font identical across Fig 3 and Fig 5/6. Extra height
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

# Display order + labels: curated tasks. Short labels match the
# detection figure (fig_detection_b) so the two figures cross-reference
# cleanly without rotation.
_TASK_ORDER = ["aiob_201", "aiob_202", "aiob_205"]
_TASK_LABELS = {
    "aiob_201": "igsr",
    "aiob_202": "jwst",
    "aiob_205": "cross",
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
    """Returns {(task, model): [sess_sp, ...]} from the true-cold campaign."""
    manifest = {r["cell"]: r for r in json.load(open(FULL / "manifest.json"))}
    cells = defaultdict(list)
    for sf in glob.glob(str(FULL / "*_staged.json")):
        cell = Path(sf).name[:-12]
        if cell not in manifest:
            continue
        cf = FULL / f"{cell}_cold.json"
        if not cf.exists():
            continue
        task = manifest[cell]["task"]
        model = _MODEL_FROM_ID.get(manifest[cell]["model"])
        if task not in _TASK_ORDER or model is None:
            continue
        c = json.loads(cf.read_text())
        s = json.loads(Path(sf).read_text())
        if c["session_elapsed_s"] <= 1:
            continue
        cells[(task, model)].append(c["session_elapsed_s"] / s["session_elapsed_s"])
    return cells


def main() -> None:
    cells = load()
    fig, ax = plt.subplots(figsize=_SIZE)

    n_models = len(_MODEL_ORDER)
    width = 0.22
    x = np.arange(len(_TASK_ORDER))

    # Plot per-(task, model) mean bars.
    for i, model in enumerate(_MODEL_ORDER):
        offset = (i - (n_models - 1) / 2) * width
        means = []
        for task in _TASK_ORDER:
            vals = cells.get((task, model), [])
            means.append(statistics.mean(vals) if vals else 0.0)
        ax.bar(
            x + offset, means, width=width * 0.92,
            color=_MODEL_COLOR[model], label=_MODEL_LABEL[model],
            linewidth=0, zorder=2,
        )

    # Per-task mean line: one horizontal segment spanning each task's
    # 4-bar cluster at the task's mean speedup (mean over all models and
    # reps), so the per-task average reads straight off the y-axis.
    mean_half = (n_models - 1) / 2 * width + (width * 0.92) / 2
    for ti, task in enumerate(_TASK_ORDER):
        gvals = [v for m in _MODEL_ORDER for v in cells.get((task, m), [])]
        if not gvals:
            continue
        gm = statistics.mean(gvals)
        ax.plot([x[ti] - mean_half, x[ti] + mean_half], [gm, gm],
                color="black", linewidth=1.4, solid_capstyle="round",
                zorder=6)
        ax.text(x[ti], gm + 0.05, rf"{gm:.2f}$\times$",
                ha="center", va="bottom", zorder=7,
                bbox=label_bbox(alpha=0.75))

    ax.axhline(1.0, color="#888888", linewidth=0.7, linestyle="--", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([_TASK_LABELS[t] for t in _TASK_ORDER])
    # Tight x-limits so the bar groups fill the axes width edge-to-edge
    # instead of floating with wide autoscale margins.
    ax.set_xlim(-0.5, len(_TASK_ORDER) - 0.5)
    ax.set_ylim(0.0, 2.6)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
    style_axis(ax, xlabel="Curated Task", ylabel=r"Session Speedup ($\times$)")
    # Legend ABOVE the plot in 2 columns: a single 4-wide row is wider
    # than the bar axes and would stretch the figure (leaving the plot
    # floating with side gaps). A 2x2 block stays within the plot width.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False, handlelength=1.0, handletextpad=0.4,
              columnspacing=1.0, labelspacing=0.3, borderaxespad=0.0)

    save(fig, "fig_speedup_curated")
    # CSV companion
    rows = []
    for (task, model), vals in sorted(cells.items()):
        rows.append({
            "task": _TASK_LABELS[task],
            "model": model,
            "n": len(vals),
            "mean": round(statistics.mean(vals), 3) if vals else 0,
            "per_rep": ",".join(f"{v:.3f}" for v in vals),
        })
    dump_csv("fig_speedup_curated", rows,
             fieldnames=["task", "model", "n", "mean", "per_rep"])
    print(f"  wrote {len(rows)} (task, model) cells")


if __name__ == "__main__":
    main()
