"""Shared matplotlib style + sizing helpers for AgentStage paper figures.

Conventions inherited from the AgentIOBench paper (IEEE eScience target,
IEEEtran double-column 10pt 8.5x11). Single source of truth — every
`build_fig_*.py` script must `from _style import *` at the top instead of
copying rcParams blocks.

Rules (do NOT override per-script unless explicitly approved):
  * Font: serif 10pt across every text element (labels, ticks, legend,
    annotations). matplotlib's default `axes.titlesize = 'large'` resolves
    to 12pt when font.size=10 — we override it explicitly.
  * NEVER use per-element fontsize kwargs (`ax.set_xticklabels(..., fontsize=9)`)
    without explicit user approval. If text doesn't fit, fix the SPACE
    (figsize, wspace, rotation, xlim) — not the font size.
  * Sizes hold scale factor 0.86 between raw figsize and final on-paper
    width, so text across figures renders at the same ~8.6pt regardless
    of column placement.
  * Two-panel: `\\begin{subfigure}{0.48\\columnwidth}` with auto (a)/(b).
  * No panel titles inside the figure — caption + subcaption carry framing.
  * Axis labels in Title Case ("Tier-1 Byte Recall", not "tier-1 byte recall").
  * y-axis grid only: linestyle=':', alpha=0.35, axisbelow=True.
  * Tick params: length=2.5, pad=2.
  * Annotation bboxes: round-pad 0.10, facecolor white, no edge.
  * Save BOTH .pdf and .png. Dump per-panel data to figures/data/<name>.csv
    so the user can re-plot from CSV independently.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Font
# ---------------------------------------------------------------------------
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [
    "Times New Roman",
    "Nimbus Roman",
    "Liberation Serif",
    "DejaVu Serif",
]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10

# ---------------------------------------------------------------------------
# Sizes — raw figsize in inches; LaTeX scales down to final column width
# ---------------------------------------------------------------------------
# Half-column subfigure inside `0.48\columnwidth` (~1.66" final).
HALF_COL = (1.85, 1.85)
# Full single-column figure (~3.45" final width on IEEE 8.5x11 10pt).
FULL_COL_W = 3.85
# Full-page two-column figure* (~7.16" final width).
FIGURE_STAR_W = 7.5

# ---------------------------------------------------------------------------
# Output roots
# ---------------------------------------------------------------------------
PAPER_DIR = Path(__file__).resolve().parent.parent / "paper"
FIG_DIR = PAPER_DIR / "figures"
DATA_DIR = FIG_DIR / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Canonical model order + palette
# Trio: Claude Haiku 4.5, Gemini 2.5 Flash, OSS reasoning.
# PoC corpus models (Sonnet 4.5, Gemini 2.5 Pro, DeepSeek-R1) appear as
# cross-cost-tier validation in the appendix; keep colors disjoint so the
# two corpora are visually distinguishable.
# ---------------------------------------------------------------------------
MODEL_ORDER = [
    "haiku_4_5",
    "gemini_2_5_flash",
    "oss_reasoning",
    "sonnet_4_5",
    "gemini_2_5_pro",
    "deepseek_r1",
]
MODEL_LABELS = [
    "Haiku 4.5",
    "Gemini 2.5 Flash",
    "OSS Reasoning",
    "Sonnet 4.5",
    "Gemini 2.5 Pro",
    "DeepSeek-R1",
]
COLORS = [
    "#d62728",  # haiku — primary red
    "#1f77b4",  # gemini flash — blue
    "#2ca02c",  # OSS — green
    "#ff7f0e",  # sonnet — orange (PoC corpus)
    "#9467bd",  # gemini pro — purple (PoC corpus)
    "#8c564b",  # deepseek-r1 — brown (PoC corpus)
]

# Backend palette (cold-tier substrate) — used in E2E speedup + bandwidth-sweep plots.
BACKEND_ORDER = ["xfs", "s3", "lustre"]
BACKEND_LABELS = ["Local XFS", "Amazon S3", "Lustre"]
BACKEND_COLORS = ["#444444", "#1f77b4", "#d62728"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save(fig, name: str, dpi: int = 200) -> tuple[Path, Path]:
    """Save figure to paper/figures/<name>.{pdf,png}. Returns both paths."""
    pdf = FIG_DIR / f"{name}.pdf"
    png = FIG_DIR / f"{name}.png"
    fig.tight_layout(pad=0.3)
    fig.savefig(pdf, bbox_inches="tight", dpi=dpi)
    fig.savefig(png, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return pdf, png


def dump_csv(name: str, rows: list[dict], fieldnames: list[str]) -> Path:
    """Write per-figure data CSV to paper/figures/data/<name>.csv."""
    path = DATA_DIR / f"{name}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return path


def style_axis(ax, *, ylabel: str | None = None, xlabel: str | None = None) -> None:
    """Apply tick/grid/spine conventions to one axis."""
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=2.5, pad=2)


def label_bbox(facecolor: str = "white", alpha: float = 1.0) -> dict:
    """Standard white-bg round bbox for inline annotations on lines."""
    return dict(
        boxstyle="round,pad=0.10",
        facecolor=facecolor,
        edgecolor="none",
        alpha=alpha,
    )
