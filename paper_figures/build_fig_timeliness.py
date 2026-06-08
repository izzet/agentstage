"""Fig:timeliness — prefetch time vs thinking-phase budget, per workload.

Message: the thinking phase is a sufficient overlap window to stage the data
for bandwidth-bound workloads (points above the y=x line = timely); only the
many-small-file metadata case (dogs-vs-cats, 22.5k opens) overruns the budget.

Data: outputs/replay/_ofs_full/ (cold sessions, non-Qwen to avoid the replay
stream cap inflating thinking) + measured tier constants (parallel prefetch BW
1179 MB/s, per-file open 4.48 ms).
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from _style import HALF_COL, save, dump_csv, style_axis, label_bbox

REPO = Path(__file__).resolve().parents[1]
FULL = REPO / "outputs" / "replay" / "_ofs_full"

# (bytes, n_files, regime, label) per workload
WL = {
    "aiob_201": (10.69e9, 153, "bandwidth", "igsr"),
    "aiob_202": (11.29e9, 96, "bandwidth", "jwst"),
    "aiob_205": (25.95e9, 176, "bandwidth", "cross"),
    "mle_dogsvcats_thumbhash": (0.537e9, 22500, "metadata", "dogs"),
    "kb_astronomy_inventory": (0.51e9, 1538, "metadata", "kb"),
    "tabular-playground-series-oct-2021": (2.33e9, 3, "compute", "tabular"),
}
BW_PAR = 1179e6   # measured 4-worker parallel prefetch bandwidth (MB/s)
T_OPEN = 4.48e-3  # measured per-file open latency (s)
REG_COLOR = {"bandwidth": "#1f77b4", "metadata": "#d62728", "compute": "#7f7f7f"}
REG_LABEL = {"bandwidth": "Bandwidth-bound", "metadata": "Metadata-bound", "compute": "Compute-bound"}


def main() -> None:
    manifest = {r["cell"]: r for r in json.load(open(FULL / "manifest.json"))}
    think = {t: [] for t in WL}
    for sf in glob.glob(str(FULL / "*_staged.json")):
        cell = Path(sf).name[:-12]
        if cell not in manifest:
            continue
        cf = FULL / f"{cell}_cold.json"
        if not cf.exists():
            continue
        if "Qwen" in manifest[cell]["model"]:   # exclude cap-inflated thinking
            continue
        c = json.loads(cf.read_text())
        cold = c["session_elapsed_s"]
        shell = c["shell_io_aggregate"]["total_shell_elapsed_s"]
        if cold > 1:
            think[manifest[cell]["task"]].append(cold - shell)

    rows = []
    for t, (by, nf, reg, lab) in WL.items():
        pf = max(by / BW_PAR, nf * T_OPEN)
        th = float(np.mean(think[t]))
        rows.append(dict(workload=lab, regime=reg, prefetch_s=pf, thinking_s=th))

    fig, ax = plt.subplots(figsize=HALF_COL)
    # y = x boundary: above => thinking budget exceeds prefetch time => timely
    lim = (1.0, 220.0)
    ax.plot(lim, lim, ls="--", color="#666666", lw=0.8, zorder=1)
    ax.fill_between(lim, lim, lim[1], color="#2ca02c", alpha=0.06, zorder=0)
    ax.text(1.4, 150, "timely", color="#2ca02c", va="top", ha="left")

    for r in rows:
        ax.scatter(r["prefetch_s"], r["thinking_s"], s=46, zorder=3,
                   color=REG_COLOR[r["regime"]], edgecolors="white", linewidths=0.5)
    # annotate only the exception: dogs-vs-cats (22.5k-file metadata) overruns the budget
    d = next(r for r in rows if r["workload"] == "dogs")
    ax.annotate("dogs-vs-cats\n(22.5k files)", (d["prefetch_s"], d["thinking_s"]),
                xytext=(38, 121), ha="center", va="center",
                fontsize=8, bbox=label_bbox(alpha=0.9),
                arrowprops=dict(arrowstyle="->", color="#666666", lw=0.6))
    ax.text(80, 14, "late", color="#d62728", va="center", ha="center")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    style_axis(ax, xlabel="Prefetch Time (s)", ylabel="Thinking-Phase Budget (s)")
    # regime legend
    handles = [plt.Line2D([], [], marker="o", ls="", color=REG_COLOR[k],
                          markeredgecolor="white", markersize=5, label=REG_LABEL[k])
               for k in ["bandwidth", "metadata", "compute"]]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.03),
              frameon=True, facecolor="white", framealpha=0.75, edgecolor="none",
              handletextpad=0.15, labelspacing=0.12, handlelength=0.7, borderpad=0.25)

    pdf, png = save(fig, "fig_timeliness")
    dump_csv("fig_timeliness", rows, ["workload", "regime", "prefetch_s", "thinking_s"])
    print(f"  wrote {pdf}\n  wrote {png}")
    for r in rows:
        print(f"    {r['workload']:8s} pf={r['prefetch_s']:.1f}s think={r['thinking_s']:.1f}s "
              f"{'TIMELY' if r['prefetch_s']<r['thinking_s'] else 'LATE'}")


if __name__ == "__main__":
    main()
