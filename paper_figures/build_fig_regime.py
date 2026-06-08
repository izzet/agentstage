"""Fig:regime — predicted I/O-share vs observed session speedup, with the
Amdahl ceiling 1/(1-io_share).

Message: speedup rises with the a-priori predicted I/O-share and stays under
the Amdahl ceiling. Crucially, bandwidth- and metadata-bound workloads have
similar I/O-share but different speedup: I/O-share sets the ceiling, timeliness
(Fig:timeliness) decides how close you get. Bandwidth-bound reaches ~86% of the
ceiling, metadata-bound only ~65%, compute-bound has near-zero I/O-share.

Data: outputs/replay/_ofs_full/ (72 cells) + I/O cost model (BW_cold 340 MB/s,
t_open 4.48 ms) for the a-priori I/O-share.
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from _style import HALF_COL, save, dump_csv, style_axis

REPO = Path(__file__).resolve().parents[1]
FULL = REPO / "outputs" / "replay" / "_ofs_full"
WL = {
    "aiob_201": (10.69e9, 153, "bandwidth"), "aiob_202": (11.29e9, 96, "bandwidth"),
    "aiob_205": (25.95e9, 176, "bandwidth"), "mle_dogsvcats_thumbhash": (0.537e9, 22500, "metadata"),
    "kb_astronomy_inventory": (0.51e9, 1538, "metadata"),
    "tabular-playground-series-oct-2021": (2.33e9, 3, "compute"),
}
BW_C = 340e6; T_OPEN = 4.48e-3
REG_COLOR = {"bandwidth": "#1f77b4", "metadata": "#d62728", "compute": "#7f7f7f"}
REG_LABEL = {"bandwidth": "Bandwidth-bound", "metadata": "Metadata-bound", "compute": "Compute-bound"}
REG_MARKER = {"bandwidth": "o", "metadata": "s", "compute": "D"}


def main() -> None:
    manifest = {r["cell"]: r for r in json.load(open(FULL / "manifest.json"))}
    pts = []
    for sf in glob.glob(str(FULL / "*_staged.json")):
        cell = Path(sf).name[:-12]
        if cell not in manifest:
            continue
        cf = FULL / f"{cell}_cold.json"
        if not cf.exists():
            continue
        c = json.loads(cf.read_text()); s = json.loads(Path(sf).read_text())
        cold = c["session_elapsed_s"]
        if cold <= 1:
            continue
        by, nf, reg = WL[manifest[cell]["task"]]
        # use the cold run's ACTUAL read volume (captures re-reads) for an
        # accurate predicted I/O cost; static dataset bytes under-counts.
        rchar = c["shell_io_aggregate"]["total_rchar"]
        iocost = rchar / BW_C + nf * T_OPEN
        ioshare = min(iocost, 0.97 * cold) / cold
        pts.append(dict(io_share=ioshare, speedup=cold / s["session_elapsed_s"], regime=reg))

    fig, ax = plt.subplots(figsize=HALF_COL)
    # Amdahl ceiling curve
    xs = np.linspace(0.0, 0.72, 200)
    ax.plot(xs, 1.0 / (1.0 - xs), ls="--", color="#666666", lw=0.9, zorder=1,
            label="Amdahl ceiling")
    ax.axhline(1.0, color="#aaaaaa", lw=0.5, ls=":", zorder=0)

    for reg in ["bandwidth", "metadata", "compute"]:
        xs_ = [p["io_share"] for p in pts if p["regime"] == reg]
        ys_ = [p["speedup"] for p in pts if p["regime"] == reg]
        ax.scatter(xs_, ys_, s=22, alpha=0.8, marker=REG_MARKER[reg],
                   color=REG_COLOR[reg], edgecolors="white", linewidths=0.4,
                   label=REG_LABEL[reg], zorder=3)

    ax.set_xlim(0.0, 0.72); ax.set_ylim(0.9, 2.6)
    style_axis(ax, xlabel="Predicted I/O Share", ylabel=r"Session Speedup ($\times$)")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.65,
              edgecolor="none", handletextpad=0.15, borderaxespad=0.25, labelspacing=0.12,
              handlelength=1.0, borderpad=0.25)

    pdf, png = save(fig, "fig_regime")
    dump_csv("fig_regime", pts, ["io_share", "speedup", "regime"])
    print(f"  wrote {pdf}\n  wrote {png}")
    for reg in ["bandwidth", "metadata", "compute"]:
        ss = [p for p in pts if p["regime"] == reg]
        ish = np.mean([p["io_share"] for p in ss]); sp = np.mean([p["speedup"] for p in ss])
        print(f"    {reg:10s} io_share={ish:.2f} sp={sp:.2f} bound={1/(1-ish):.2f} "
              f"({100*sp/(1/(1-ish)):.0f}% of ceiling)")


if __name__ == "__main__":
    main()
