"""Slide-scale charts for the AgentStage eScience'26 talk.

Rebuilt from paper/figures/data/*.csv for projector readability:
IBM Plex Sans, >=18pt text at 200 dpi (renders >=24px when the PNG is
shown at half its pixel width on a 1920x1080 slide), thick marks,
GRC-blue accent + neutral grays, shape/label redundancy instead of hue.
"""
import glob, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for f in glob.glob(os.path.expanduser("~/.fonts/*.ttf")):
    fm.fontManager.addfont(f)

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
OUT = sys.argv[2] if len(sys.argv) > 2 else "out"
os.makedirs(OUT, exist_ok=True)

BG = "#FAFBFC"
INK = "#16191D"
SOFT = "#3A434D"
MUTED = "#5B6670"
GRID = "#D5DCE2"
BLUE = "#0B6E99"      # deepened GRC blue (5.6:1 on BG)
BLUE_L = "#9FD3EA"    # light fill (areas only, never text)
GRAY = "#8A949E"

plt.rcParams.update({
    "font.family": "IBM Plex Sans",
    "font.size": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 19,
    "ytick.labelsize": 19,
    "legend.fontsize": 19,
    "axes.edgecolor": SOFT,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 1.6,
    "xtick.major.width": 1.6,
    "ytick.major.width": 1.6,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
DPI = 200


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def ygrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=1.4)
    ax.set_axisbelow(True)


def xgrid(ax):
    ax.xaxis.grid(True, color=GRID, lw=1.4)
    ax.set_axisbelow(True)


# 1. Tool-execution share of wall time -----------------------------------
d = pd.read_csv(f"{DATA}/fig_motivation_decomp.csv")
names = {"Curated": "Curated", "KramaBench": "KramaBench", "MLE": "MLE-bench", "DSBench": "DSBench"}
d["label"] = d["benchmark"].map(names)
d = d.sort_values("tool_exec_pct")
fig, ax = plt.subplots(figsize=(10, 5.6))
y = np.arange(len(d))
ax.barh(y, 100, color="#E3E8ED", height=0.62)
ax.barh(y, d["tool_exec_pct"], color=BLUE, height=0.62)
for yi, v in zip(y, d["tool_exec_pct"]):
    ax.text(v - 2, yi, f"{v:.0f}%", va="center", ha="right", color="white", fontsize=24, fontweight="bold")
ax.set_yticks(y, d["label"], fontsize=22)
ax.set_xlim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
ax.set_xlabel("Share of median session wall time spent in tool execution")
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
save(fig, "c_toolshare")

# 2. Bytes stageable within one thinking phase ----------------------------
b = pd.read_csv(f"{DATA}/fig_motivation_bytes_moveable.csv")
tier = {"local_ssd": "Local SSD", "orangefs": "OrangeFS", "shared_xfs": "Shared XFS", "local_nvme": "Local NVMe"}
b["label"] = b["backend"].map(tier)
b["gb"] = b["bytes_moveable_mb"] / 1000
b = b.sort_values("gb")
fig, ax = plt.subplots(figsize=(10, 5.6))
y = np.arange(len(b))
ax.barh(y, b["gb"], color=BLUE, height=0.62)
for yi, v in zip(y, b["gb"]):
    ax.text(v + 0.8, yi, f"{v:.0f} GB", va="center", ha="left", color=INK, fontsize=22, fontweight="bold")
# workload dataset sizes (Table I): 0.5 to 26 GB
ax.axvspan(0.5, 26, color="#E3E8ED", zorder=0)
ax.text(13.25, len(b) - 0.35, "Workload datasets\n0.5–26 GB", ha="center", va="bottom", color=SOFT, fontsize=19)
ax.set_yticks(y, b["label"], fontsize=22)
ax.set_xlim(0, 58)
ax.set_ylim(-0.6, len(b) + 0.45)
ax.set_xlabel("Data stageable during one thinking phase (GB)")
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
save(fig, "c_stageable")

# 3. Timeline: naive vs AgentStage session --------------------------------
t = pd.read_csv(f"{DATA}/fig_motivation_timeline.csv")
fig, ax = plt.subplots(figsize=(10, 3.4))
rows = [("Without staging", t.iloc[0]), ("AgentStage", t.iloc[1])]
for i, (lab, r) in enumerate(rows[::-1]):
    ax.barh(i, r["floor_s"], color="#5B6670", height=0.6)
    ax.barh(i, r["tool_s"], left=r["floor_s"], color=BLUE if lab == "AgentStage" else SOFT, height=0.6)
    ax.text(r["total_s"] + 1.5, i, f"{r['total_s']:.0f} s", va="center", fontsize=22, fontweight="bold", color=INK)
    ax.text(r["floor_s"] / 2, i, "LLM + other", va="center", ha="center", fontsize=18, color="white")
    ax.text(r["floor_s"] + r["tool_s"] / 2, i, f"tools {r['tool_s']:.0f} s", va="center", ha="center", fontsize=18, color="white", fontweight="bold")
ax.set_yticks([0, 1], ["AgentStage", "Without staging"], fontsize=21)
ax.set_xlim(0, 110)
ax.set_xlabel("Mean curated session wall time (s)")
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
save(fig, "c_timeline")

# 4. Detection recall per model -------------------------------------------
det = pd.read_csv(f"{DATA}/fig_detection.csv")
mname = {"claude-haiku-4-5": "Haiku 4.5", "claude-sonnet-4-5": "Sonnet 4.5",
         "gemini-2.5-flash": "Gemini 2.5\nFlash", "Qwen/Qwen3.6-27B": "Qwen3.6\n27B"}
order = ["claude-haiku-4-5", "claude-sonnet-4-5", "gemini-2.5-flash", "Qwen/Qwen3.6-27B"]
fig, ax = plt.subplots(figsize=(10, 6.2))
rng = np.random.default_rng(3)
for i, m in enumerate(order):
    v = det.loc[det["model"] == m, "byte_recall"].clip(upper=1.0).values
    med = np.median(v)
    ax.bar(i, med, color=BLUE, width=0.62, zorder=2)
    ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=110, color="white", edgecolor=INK, lw=2, zorder=3)
    ax.text(i, med + 0.035 if med < 0.95 else 1.06, f"{med:.2f}", ha="center", va="bottom", fontsize=23, fontweight="bold", color=INK)
ax.set_xticks(range(4), [mname[m] for m in order])
ax.set_ylim(0, 1.18)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_ylabel("Tier-1 byte recall")
ygrid(ax)
save(fig, "c_detection")
print("detection medians", det.groupby("model")["byte_recall"].median().to_dict())

# 5. Activation latency CDF (pooled) --------------------------------------
act = pd.read_csv(f"{DATA}/fig_activation.csv")
lat = np.sort(act["latency_s"].values)
p = np.arange(1, len(lat) + 1) / len(lat)
fig, ax = plt.subplots(figsize=(10, 6.2))
ax.step(np.r_[0, lat], np.r_[0, p], where="post", color=BLUE, lw=4.5)
med = np.median(lat)
ax.axvline(med, color=SOFT, lw=2, ls="--")
ax.text(med + 0.08, 0.08, f"median {med:.2f} s", color=INK, fontsize=21, fontweight="bold")
f1 = (lat <= 1).mean(); f2 = (lat <= 2).mean()
for x, f in [(1, f1), (2, f2)]:
    ax.plot([x], [f], "o", ms=13, color="white", mec=INK, mew=2.5, zorder=4)
    ax.text(x + 0.12, f - 0.07, f"{f*100:.0f}% within {x} s", fontsize=21, color=INK)
ax.set_xlim(0, 5)
ax.set_ylim(0, 1.03)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
ax.set_xlabel("Time from first thinking token to first correct detection (s)")
ax.set_ylabel("Sessions")
ygrid(ax)
save(fig, "c_activation")
print("activation n", len(lat), "median", med, "<=1", f1, "<=2", f2)


# 6/7. Speedup bars with per-cell dots -------------------------------------
def speedup_chart(csv, key, order, labels, name, ylim, color=BLUE, sub=None):
    s = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(10, 6.2))
    rng = np.random.default_rng(7)
    means = []
    for i, k in enumerate(order):
        rows = s[s[key] == k]
        reps = np.concatenate([np.array(r.split(","), float) for r in rows["per_rep"]])
        m = reps.mean(); means.append(m)
        ax.bar(i, m, color=color, width=0.6, zorder=2)
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(reps)), reps, s=95, color="white", edgecolor=INK, lw=2, zorder=3)
        ax.text(i, max(reps.max(), m) + 0.06, f"{m:.2f}×", ha="center", va="bottom", fontsize=26, fontweight="bold", color=INK)
    ax.axhline(1.0, color=SOFT, lw=2, ls="--", zorder=1)
    ax.set_xticks(range(len(order)), labels)
    ax.set_ylim(0, ylim)
    ax.set_ylabel("Session speedup (×)")
    if sub:
        for i, t_ in enumerate(sub):
            ax.text(i, -0.2 * ylim / 2.8, t_, ha="center", va="top", fontsize=18, color=SOFT, transform=ax.transData)
    ygrid(ax)
    save(fig, name)
    return means


print("curated", speedup_chart(f"{DATA}/fig_speedup_curated.csv", "task", ["igsr", "jwst", "cross"],
                               ["igsr-cov-qc\nGenomics", "jwst-nircam\nAstronomy", "cross-archive\nMulti-domain"],
                               "c_speedup_curated", 2.85))
print("community", speedup_chart(f"{DATA}/fig_speedup_community.csv", "bench", ["MLE", "KB", "DSB"],
                                 ["MLE-bench\ndogs-vs-cats", "KramaBench\nastronomy-inv.", "DSBench\ntabular"],
                                 "c_speedup_community", 2.0))

# 8. Regime scatter ---------------------------------------------------------
r = pd.read_csv(f"{DATA}/fig_regime.csv")
fig, ax = plt.subplots(figsize=(9, 7))
xs = np.linspace(0, 0.66, 200)
ax.plot(xs * 100, 1 / (1 - xs), color=SOFT, lw=2.5, ls="--", zorder=1)
AMDAHL_LABEL = True  # placed on the curve after limits are set (below)
style = {"bandwidth": ("o", BLUE, "Bandwidth-bound"), "metadata": ("s", INK, "Metadata-bound"),
         "compute": ("D", GRAY, "Compute-bound")}
for k, (mk, c, lab) in style.items():
    q = r[r["regime"] == k]
    ax.scatter(q["io_share"] * 100, q["speedup"], marker=mk, s=150, color=c, edgecolor="white", lw=1.5, label=lab, zorder=3)
ax.set_xlim(-1, 72)
ax.set_ylim(0.9, 2.6)
ax.set_xticks([0, 20, 40, 60], ["0%", "20%", "40%", "60%"])
ax.set_xlabel("Predicted I/O share of session")
ax.set_ylabel("Session speedup (×)")
# label sits on the dashed ceiling, rotated to its local slope
fig.canvas.draw()
x0 = 0.40
p1 = ax.transData.transform((x0 * 100 - 2, 1 / (1 - (x0 - 0.02))))
p2 = ax.transData.transform((x0 * 100 + 2, 1 / (1 - (x0 + 0.02))))
ang = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
ax.annotate("Amdahl ceiling\n1 / (1 − I/O share)", xy=(x0 * 100, 1 / (1 - x0)), xytext=(-16, 4), textcoords="offset points",
            rotation=ang, rotation_mode="anchor", ha="center", va="bottom", multialignment="center", linespacing=1.15, color=SOFT, fontsize=19, fontweight="bold",
            transform_rotates_text=False)
ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3, frameon=False, handletextpad=0.2, columnspacing=0.8, borderaxespad=0.3)
ygrid(ax)
save(fig, "c_regime")

# 9. Timeliness: prefetch time vs thinking window -------------------------
tm = pd.read_csv(f"{DATA}/fig_timeliness.csv")
lab = {"igsr": "igsr-cov-qc", "jwst": "jwst-nircam", "cross": "cross-archive", "dogs": "MLE-bench", "kb": "KramaBench", "tabular": "DSBench"}
tm["label"] = tm["workload"].map(lab)
tm = tm.iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9, 7))
y = np.arange(len(tm))
ax.barh(y, tm["thinking_s"], color="#E3E8ED", height=0.7, edgecolor=GRAY, lw=1.5, label="Thinking window")
for yi, row in tm.iterrows():
    late = row["prefetch_s"] > row["thinking_s"]
    ax.barh(yi, row["prefetch_s"], color=INK if late else BLUE, height=0.36, label=None)
    txt = f"{row['prefetch_s']:.0f} s" + ("  overruns" if late else "")
    ax.text(max(row["prefetch_s"], row["thinking_s"]) + 2, yi, txt, va="center", fontsize=19,
            color=INK, fontweight="bold" if late else "normal")
ax.set_yticks(y, tm["label"], fontsize=20)
ax.set_xlim(0, 150)
ax.set_xlabel("Seconds")
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#E3E8ED", edgecolor=GRAY, label="Thinking window"),
                   Patch(facecolor=BLUE, label="Prefetch time")], loc="lower right", frameon=False)
xgrid(ax)
save(fig, "c_timeliness")
print("done")
