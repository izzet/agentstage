"""Fig 1 - motivation (three panels: 1+2+1 widths, equal heights).

Layout (sciiobench square convention from feedback-figure-style memory):
    (a) 1.85" x 1.85"  square          time-decomposition stacked bars
    (b) 3.70" x 1.85"  2x-wide rectangle  bytes-moveable per backend
                                          with workload dataset reference lines
    (c) 1.85" x 1.85"  square          mechanism timeline (real x/y axes)

All three panels carry: x-label, y-tick category labels, legend, no panel
title (caption/subcaption do the framing per the established convention).

Outputs:
    paper/figures/fig_motivation_a.{pdf,png}
    paper/figures/fig_motivation_b.{pdf,png}
    paper/figures/fig_motivation_c.{pdf,png}
    paper/figures/fig_motivation.{pdf,png}                   combined preview
    paper/figures/data/fig_motivation_decomp.csv
    paper/figures/data/fig_motivation_bytes_moveable.csv
    paper/figures/data/fig_motivation_timeline.csv

Inputs:
    outputs/{aiob_mt,dsbench_mt,mlebench_mt}/_sweep_*/
    outputs/bench_tiers/<latest>/summary.csv

Run:
    uv run python paper_figures/build_fig_motivation.py
"""
from __future__ import annotations

import csv
import glob
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _style import (  # noqa: E402
    DATA_DIR,
    FIG_DIR,
    dump_csv,
    label_bbox,
    save,
    style_axis,
)

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Wall-time decomposition (inlined from scripts/microbench/decompose_wall_time.py)
# ---------------------------------------------------------------------------

def _turn_first_last_ms(turn_dir: Path) -> tuple[float | None, float | None]:
    t_min: float | None = None
    t_max: float | None = None
    for stream in ("thinking.jsonl", "text.jsonl"):
        p = turn_dir / stream
        if not p.is_file():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = r.get("t_ms")
            if t is None:
                continue
            if t_min is None or t < t_min:
                t_min = t
            if t_max is None or t > t_max:
                t_max = t
    return t_min, t_max


def _decompose_session(session_dir: Path) -> dict | None:
    """Decompose a session into five mutually exclusive buckets summing
    to session_elapsed_s:

      comm_s       = per-turn time from turn start to first streaming delta
                     (API roundtrip + network)
      stream_s     = per-turn time from first to last streaming delta
                     (LLM actively emitting tokens — strict Reasoning)
      tool_shell_s = per-turn post-streaming time on turns whose tool_names
                     include `run_shell_command` (strict shell execution)
      tool_other_s = per-turn post-streaming time on non-shell tool turns
                     (list_dir / read_file / write_file / open_file)
      harness_s    = per-turn post-streaming time on no-tool turns +
                     session-level inter-turn gaps + any residual

    Tool Exec (broad, for Fig 1a) = tool_shell_s + tool_other_s.
    Fig 1c Thinking Phase = total - tool_shell_s = the parallel-copyable window.
    """
    sf = session_dir / "summary.json"
    if not sf.is_file():
        return None
    s = json.loads(sf.read_text())
    if s.get("crash"):
        return None
    total = s.get("session_elapsed_s")
    if total is None:
        return None
    comm = stream = tool_shell = tool_other = harness = 0.0
    for t in s.get("per_turn", []):
        idx = t.get("turn", 0)
        dur = float(t.get("duration_s", 0) or 0)
        td = session_dir / "turns" / f"turn_{idx:02d}"
        t_first, t_last = (None, None)
        if td.is_dir():
            t_first, t_last = _turn_first_last_ms(td)
        names = t.get("tool_names") or []
        has_shell = "run_shell_command" in names
        has_other_tools = any(n != "run_shell_command" for n in names)
        if t_first is None or t_last is None:
            # No streaming deltas — attribute entire turn duration.
            if has_shell:
                tool_shell += dur
            elif has_other_tools:
                tool_other += dur
            else:
                harness += dur
            continue
        comm += t_first / 1000.0
        stream += (t_last - t_first) / 1000.0
        rest = dur - (t_last / 1000.0)
        if has_shell:
            tool_shell += rest
        elif has_other_tools:
            tool_other += rest
        else:
            harness += rest
    # Inter-turn gaps + session-level residual roll into harness so the
    # five buckets sum to session_elapsed_s.
    accounted = comm + stream + tool_shell + tool_other + harness
    harness += max(0.0, float(total) - accounted)
    return {
        "task": s.get("task"),
        "model": s.get("model"),
        "mode": s.get("mode"),
        "total_s": total,
        "comm_s": comm,
        "stream_s": stream,
        "tool_shell_s": tool_shell,
        "tool_other_s": tool_other,
        "harness_s": harness,
    }


_BENCH_SWEEP_GLOBS = {
    "AIOB": ["outputs/aiob_mt/_sweep_*_curated*"],
    "DSBench": [
        "outputs/dsbench_mt/_sweep_haiku_*",
        "outputs/dsbench_mt/_sweep_sonnet_*",
        "outputs/dsbench_mt/_sweep_gemini_*",
    ],
    "MLE-bench": [
        "outputs/mlebench_mt/_sweep_haiku_*",
        "outputs/mlebench_mt/_sweep_sonnet_*",
        "outputs/mlebench_mt/_sweep_gemini_*",
    ],
}


def load_decomp_by_bench_mode() -> tuple[dict, list[dict]]:
    """Aggregate sessions per (benchmark, mode). Returns
    ({(bench, mode): {comm_s, stream_s, ...}}, raw_rows)."""
    by: dict[tuple[str, str], dict] = {}
    raw: list[dict] = []
    for bench, patterns in _BENCH_SWEEP_GLOBS.items():
        sweep_dirs: list[Path] = []
        for pattern in patterns:
            sweep_dirs.extend(Path(p) for p in glob.glob(str(REPO / pattern)))
        rows_per_mode: dict[str, list[dict]] = {"baseline": [], "staged": []}
        for sweep in sorted(set(sweep_dirs)):
            for sess in sorted(sweep.iterdir()):
                if not sess.is_dir():
                    continue
                r = _decompose_session(sess)
                if r is None:
                    continue
                if r["total_s"] < 10.0:
                    continue
                mode = r["mode"]
                if mode not in rows_per_mode:
                    continue
                rows_per_mode[mode].append(r)
                raw.append({**r, "benchmark": bench})
        for mode, rs in rows_per_mode.items():
            if not rs:
                continue
            # Per-session shares — each session sums to 100% by construction
            # so median-of-shares is meaningful (medians of raw seconds are
            # not additive, which would make Fig 1a misleading).
            for r in rs:
                t = float(r["total_s"])
                r["tool_exec_pct"] = (
                    (float(r["tool_shell_s"]) + float(r["tool_other_s"]))
                    / t * 100.0
                )
                r["reasoning_pct"] = float(r["stream_s"]) / t * 100.0
                r["comm_pct"] = float(r["comm_s"]) / t * 100.0
                r["harness_pct"] = float(r["harness_s"]) / t * 100.0
            med = statistics.median
            by[(bench, mode)] = {
                "n_sessions": len(rs),
                "total_s": med([r["total_s"] for r in rs]),
                "comm_s": med([r["comm_s"] for r in rs]),
                "stream_s": med([r["stream_s"] for r in rs]),
                "tool_shell_s": med([r["tool_shell_s"] for r in rs]),
                "tool_other_s": med([r["tool_other_s"] for r in rs]),
                "harness_s": med([r["harness_s"] for r in rs]),
                # Median of per-session shares (these are the load-bearing
                # numbers for Fig 1a — guaranteed to sum to ~100%).
                "tool_exec_pct": med([r["tool_exec_pct"] for r in rs]),
                "reasoning_pct": med([r["reasoning_pct"] for r in rs]),
                "comm_pct": med([r["comm_pct"] for r in rs]),
                "harness_pct": med([r["harness_pct"] for r in rs]),
            }
    return by, raw


# ---------------------------------------------------------------------------
# IOR backend BW
# ---------------------------------------------------------------------------

def load_backend_bw() -> list[dict]:
    runs = sorted((REPO / "outputs" / "bench_tiers").glob("*/summary.csv"))
    if not runs:
        return []
    latest = runs[-1]
    rows: list[dict] = []
    with open(latest) as f:
        for r in csv.DictReader(f):
            if r["tool"] != "ior" or r["op"] != "read":
                continue
            rows.append({
                "tier": r["tier"],
                "bw_mibps": float(r["mean_mibps"]),
                "min_mibps": float(r["min_mibps"]),
                "max_mibps": float(r["max_mibps"]),
            })
    return rows


WORKLOAD_REFS = [
    ("DSBench tabular-feb-2022", 30),
    ("DSBench ventilator", 400),
    ("AIOB-110 NWB", 1311),
    ("MLE NYC", 5401),
    ("AIOB-107 GOES", 18000),
]


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

_TOOL_COLOR = "#d62728"
_STREAM_COLOR = "#1f77b4"
_COMM_COLOR = "#7f7f7f"
_OTHER_COLOR = "#cccccc"
_HARNESS_COLOR = "#bcbcbc"
_HOT_COLOR = "#2ca02c"
_COLD_COLOR = "#5b5b5b"
_REF_COLOR = "#a83232"
_COPY_COLOR = "#9ecae1"


# ---------------------------------------------------------------------------
# (a) Time decomposition  (square 1.85 x 1.85)
# ---------------------------------------------------------------------------

def _draw_panel_a(ax, per_bench_baseline: dict) -> None:
    """Stacked horizontal bars per benchmark. Three categories summing
    to 100% by construction:

      Tool Exec (red)   = tool_shell + tool_other  (every tool call)
      Reasoning (blue)  = stream + comm            (LLM-side wall time:
                                                    token emission + API
                                                    roundtrip)
      Harness (light)   = harness                  (inter-turn gaps and
                                                    no-tool turns; near-
                                                    zero on tight loops)

    Stack order (left to right): Tool Exec | Reasoning | Harness.
    """
    benches = list(per_bench_baseline.keys())
    n = len(benches)
    y = np.arange(n)

    tool_exec_pct: list[float] = []
    reasoning_pct: list[float] = []
    harness_pct: list[float] = []
    totals: list[float] = []
    for b in benches:
        d = per_bench_baseline[b]
        totals.append(float(d["total_s"]))
        te = float(d["tool_exec_pct"])
        rs = float(d["reasoning_pct"]) + float(d["comm_pct"])
        h = float(d["harness_pct"])
        # Renormalize so the three categories exactly fill 100% (medians
        # of per-session shares sum close to but not exactly 100%).
        total_pct = te + rs + h
        if total_pct > 0:
            scale = 100.0 / total_pct
            te *= scale
            rs *= scale
            h *= scale
        tool_exec_pct.append(te)
        reasoning_pct.append(rs)
        harness_pct.append(h)

    left = np.zeros(n)
    categories = [
        ("Tool Exec", _TOOL_COLOR, tool_exec_pct),
        ("Reasoning", _STREAM_COLOR, reasoning_pct),
        ("Harness", _HARNESS_COLOR, harness_pct),
    ]
    handles: list[mpatches.Patch] = []
    labels: list[str] = []
    for label, color, vals_list in categories:
        vals = np.array(vals_list)
        ax.barh(y, vals, left=left, height=0.62, color=color,
                edgecolor="white", linewidth=0.4)
        left += vals
        handles.append(mpatches.Patch(facecolor=color, edgecolor="white",
                                       linewidth=0.4))
        labels.append(label)
    ax._legend_handles = (handles, labels)  # type: ignore

    ax.set_yticks(y)
    ax.set_yticklabels(benches)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(["0", "50", "100"])
    ax.invert_yaxis()
    style_axis(ax, xlabel="Share of Session Wall Time (%)")

    # Total session time INSIDE the Tool Exec block (right-aligned, white).
    for i, (te, t) in enumerate(zip(tool_exec_pct, totals)):
        ax.text(te - 1.5, i, f"{int(t)} s", va="center", ha="right",
                color="white", fontweight="bold")

    # Legend below x-axis label (uniform 10pt, inherited from rcParams).
    ax.legend(handles, labels, loc="upper center",
              bbox_to_anchor=(0.5, -0.30), ncol=3, frameon=False,
              handlelength=1.0, columnspacing=0.6, handletextpad=0.3)


def _save_floating(fig, name: str) -> Path:
    """Save without `tight_layout()` — the legend stays anchored outside
    the figsize and `bbox_inches='tight'` crops the output to include
    whatever floats below. The plot box keeps its full figsize space."""
    pdf = FIG_DIR / f"{name}.pdf"
    png = FIG_DIR / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=200)
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return pdf


def build_fig_a(per_bench_baseline: dict,
                out_name: str = "fig_motivation_a") -> Path:
    # Square plot box. Legend floats below — does not compress the plot.
    fig, ax = plt.subplots(1, 1, figsize=(1.85, 1.85))
    _draw_panel_a(ax, per_bench_baseline)
    _save_floating(fig, out_name)
    return FIG_DIR / f"{out_name}.pdf"


# ---------------------------------------------------------------------------
# (b) Bytes-moveable per backend  (rectangle 3.70 x 1.85)
# ---------------------------------------------------------------------------

def _draw_panel_b(ax, backends: list[dict], prefetch_window_s: float) -> None:
    cold = [b for b in backends if b["tier"] != "tmpfs"]
    cold.sort(key=lambda x: x["bw_mibps"])

    y = np.arange(len(cold))
    bw = np.array([c["bw_mibps"] for c in cold])
    mb = bw * prefetch_window_s

    HOT_TIERS = {"local_nvme"}
    colors = [_HOT_COLOR if c["tier"] in HOT_TIERS else _COLD_COLOR
              for c in cold]

    ax.barh(y, mb, height=0.55, color=colors,
            edgecolor="black", linewidth=0.4)

    label_map = {
        "local_ssd": "Local SSD",
        "shared_xfs": "Shared XFS",
        "orangefs": "OrangeFS",
        "local_nvme": "Local NVMe",
    }
    ax.set_yticks(y)
    ax.set_yticklabels([label_map.get(c["tier"], c["tier"]) for c in cold])

    ax.set_xscale("log")
    ax.set_xlim(10, 300_000)
    ax.set_xticks([10, 100, 1000, 10_000, 100_000])
    ax.set_xticklabels(["10", "100", "1G", "10G", "100G"])
    ax.set_xlabel(f"Bytes-Moveable in {prefetch_window_s:.0f} s Floor (MB)")
    style_axis(ax)

    # Dataset reference verticals + rotated labels inside plot
    n_cold = len(cold)
    for name, mb_size in WORKLOAD_REFS:
        ax.axvline(mb_size, color=_REF_COLOR, linestyle="--",
                   linewidth=0.7, alpha=0.65, zorder=2)
        ax.text(mb_size, n_cold - 0.55, name,
                rotation=90, ha="center", va="top",
                color=_REF_COLOR, bbox=label_bbox(alpha=0.9))

    # Bar-end value annotation
    for i, v in enumerate(mb):
        label = f"{v/1000:.1f} GB" if v >= 1000 else f"{v:.0f} MB"
        ax.text(v * 1.20, i, label, va="center", ha="left")

    handles = [
        mpatches.Patch(facecolor=_HOT_COLOR, edgecolor="black",
                       linewidth=0.4, label="Hot tier (stage target)"),
        mpatches.Patch(facecolor=_COLD_COLOR, edgecolor="black",
                       linewidth=0.4, label="Cold tier (stage source)"),
        plt.Line2D([0], [0], color=_REF_COLOR, linestyle="--", linewidth=0.7,
                   label="Workload dataset size"),
    ]
    ax._legend_handles = (handles, [h.get_label() for h in handles])  # type: ignore

    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.30), ncol=3, frameon=False,
              handlelength=1.2, columnspacing=0.8, handletextpad=0.3)


def build_fig_b(backends: list[dict], prefetch_window_s: float,
                out_name: str = "fig_motivation_b") -> Path:
    # Same plot-box height as Fig 1a; double the width per the 1:2:1
    # row layout. Legend floats below — no tight_layout compression.
    fig, ax = plt.subplots(1, 1, figsize=(3.70, 1.85))
    _draw_panel_b(ax, backends, prefetch_window_s)
    _save_floating(fig, out_name)
    return FIG_DIR / f"{out_name}.pdf"


# ---------------------------------------------------------------------------
# (c) Mechanism timeline  (square 1.85 x 1.85, real x/y axes)
# ---------------------------------------------------------------------------

def _draw_panel_c(ax, naive_total: float, naive_floor: float,
                  staged_total: float, staged_floor: float) -> None:
    """Real timeline plot. Two rows (Naive, AgentStage) with bars at
    actual second positions; copy stripe overlays the AgentStage row.

    Inputs are median session decomposition for a representative cell
    (we use AIOB since it shows the dominant-tool regime AgentStage
    targets). naive_total / staged_total are wall-clock seconds;
    *_floor is the comm+stream+other window inside which the copy fits."""

    naive_tool = naive_total - naive_floor
    staged_tool = staged_total - staged_floor
    # Visualize copy as filling exactly the floor (best case: copy stays
    # entirely within the LLM-side window). Real campaigns show the copy
    # finishes before the floor closes on most cells.
    copy_dur = min(staged_floor, naive_floor)

    # y positions: Naive at top, AgentStage below
    y_naive = 1.0
    y_agent = 0.0
    bar_h = 0.38

    # Naive row
    ax.barh(y_naive, naive_floor, height=bar_h, left=0,
            color=_COMM_COLOR, edgecolor="black", linewidth=0.4)
    ax.barh(y_naive, naive_tool, height=bar_h, left=naive_floor,
            color=_TOOL_COLOR, edgecolor="black", linewidth=0.4)

    # AgentStage row
    ax.barh(y_agent, staged_floor, height=bar_h, left=0,
            color=_COMM_COLOR, edgecolor="black", linewidth=0.4)
    ax.barh(y_agent, staged_tool, height=bar_h, left=staged_floor,
            color=_TOOL_COLOR, edgecolor="black", linewidth=0.4)

    # Copy stripe overlaid below the AgentStage LLM block
    copy_y = y_agent - bar_h / 2 - 0.18
    ax.barh(copy_y, copy_dur, height=0.20, left=0,
            color=_COPY_COLOR, edgecolor="black", linewidth=0.3,
            hatch="////")

    # "saved" double-arrow between staged tool-end and naive tool-end
    saved_x0 = staged_total
    saved_x1 = naive_total
    arrow_y = (y_naive + y_agent) / 2
    ax.annotate(
        "", xy=(saved_x1, arrow_y), xytext=(saved_x0, arrow_y),
        arrowprops=dict(arrowstyle="<->", color="black", lw=0.7),
    )
    ax.text((saved_x0 + saved_x1) / 2, arrow_y + 0.05,
            f"saved {int(naive_total - staged_total)} s",
            ha="center", va="bottom",
            fontweight="bold", color="black",
            bbox=label_bbox(alpha=0.9))

    # y-axis: categorical
    ax.set_yticks([y_agent, y_naive])
    ax.set_yticklabels(["AgentStage", "Naive"])

    # x-axis: time in seconds — pick a tick step that keeps the labels
    # readable at uniform 10pt within a 1.85" panel (typically 3 ticks).
    ax.set_xlim(0, naive_total * 1.02)
    if naive_total > 250:
        step = 150
    elif naive_total > 120:
        step = 75
    else:
        step = 50
    ticks = list(range(0, int(naive_total) + 1, step))
    if ticks[-1] < naive_total:
        ticks.append(int(round(naive_total / step) * step))
    ax.set_xticks(ticks)
    style_axis(ax, xlabel="Time (s)")

    # widen y-limits so copy stripe + saved annotation fit
    ax.set_ylim(copy_y - 0.20, y_naive + bar_h / 2 + 0.15)

    handles = [
        mpatches.Patch(facecolor=_COMM_COLOR, edgecolor="black",
                       linewidth=0.4, label="Thinking Phase"),
        mpatches.Patch(facecolor=_TOOL_COLOR, edgecolor="black",
                       linewidth=0.4, label="Tool Exec"),
        mpatches.Patch(facecolor=_COPY_COLOR, edgecolor="black",
                       linewidth=0.3, hatch="///", label="Copy"),
    ]
    ax._legend_handles = (handles, [h.get_label() for h in handles])  # type: ignore

    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.30), ncol=3, frameon=False,
              handlelength=1.0, columnspacing=0.6, handletextpad=0.3)


def build_fig_c(naive_total: float, naive_floor: float,
                staged_total: float, staged_floor: float,
                out_name: str = "fig_motivation_c") -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(1.85, 1.85))
    _draw_panel_c(ax, naive_total, naive_floor, staged_total, staged_floor)
    _save_floating(fig, out_name)
    return FIG_DIR / f"{out_name}.pdf"


# ---------------------------------------------------------------------------
# Combined preview (single PDF, 7.4 x 1.85, gridspec 1:2:1)
# ---------------------------------------------------------------------------

def build_combined(per_bench_baseline: dict, backends: list[dict],
                   prefetch_window_s: float,
                   c_args: tuple,
                   out_name: str = "fig_motivation") -> Path:
    # Combined preview matches the standalone plot-box height of 1.85";
    # each panel's legend floats below via the same _save_floating path.
    fig = plt.figure(figsize=(7.40, 1.85))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 2.0, 1.0],
                          wspace=0.65)

    ax_a = fig.add_subplot(gs[0, 0])
    _draw_panel_a(ax_a, per_bench_baseline)

    ax_b = fig.add_subplot(gs[0, 1])
    _draw_panel_b(ax_b, backends, prefetch_window_s)

    ax_c = fig.add_subplot(gs[0, 2])
    _draw_panel_c(ax_c, *c_args)

    _save_floating(fig, out_name)
    return FIG_DIR / f"{out_name}.pdf"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("Loading decomposition (baseline + staged, headline cells)...")
    by, _raw = load_decomp_by_bench_mode()
    per_bench_baseline = {
        b: d for (b, mode), d in by.items() if mode == "baseline"
    }
    per_bench_staged = {
        b: d for (b, mode), d in by.items() if mode == "staged"
    }
    def _floor(d: dict) -> float:
        # Fig 1c "Thinking Phase" = parallel-copyable window = everything
        # except strict shell execution. Includes Reasoning + Comm +
        # Harness + non-shell tool ops.
        return float(d["total_s"]) - float(d["tool_shell_s"])

    for b, d in per_bench_baseline.items():
        tool_exec = d["tool_shell_s"] + d["tool_other_s"]
        print(f"  {b:10s} baseline  n={d['n_sessions']:3d}  "
              f"total={d['total_s']:.0f}s  toolExec={tool_exec:.0f}s "
              f"({tool_exec/d['total_s']:.0%})  reasoning={d['stream_s']:.0f}s "
              f"({d['stream_s']/d['total_s']:.0%})  "
              f"floor={_floor(d):.0f}s")
    for b, d in per_bench_staged.items():
        tool_exec = d["tool_shell_s"] + d["tool_other_s"]
        print(f"  {b:10s} staged    n={d['n_sessions']:3d}  "
              f"total={d['total_s']:.0f}s  toolExec={tool_exec:.0f}s "
              f"({tool_exec/d['total_s']:.0%})  reasoning={d['stream_s']:.0f}s "
              f"({d['stream_s']/d['total_s']:.0%})  "
              f"floor={_floor(d):.0f}s")
    if not per_bench_baseline:
        print("ERROR: no baseline sessions found", file=sys.stderr)
        return 2

    print("\nLoading backend BW (latest bench_tiers run)...")
    backends = load_backend_bw()
    for b in backends:
        print(f"  {b['tier']:12s}  {b['bw_mibps']:>8.1f} MiB/s")
    if not backends:
        print("ERROR: no IOR backend data found", file=sys.stderr)
        return 2

    prefetch_window_s = float(statistics.median(
        [_floor(d) for d in per_bench_baseline.values()]
    ))
    print(f"\nMedian prefetch window (thinking phase, across baseline benches): "
          f"{prefetch_window_s:.1f} s")

    # Fig 1c source data: AIOB representative (largest tool dominance)
    aiob_baseline = per_bench_baseline.get("AIOB")
    aiob_staged = per_bench_staged.get("AIOB")
    if aiob_baseline is None or aiob_staged is None:
        print("WARNING: AIOB staged or baseline missing — using illustrative numbers")
        c_args = (200.0, 50.0, 100.0, 50.0)
    else:
        c_args = (
            float(aiob_baseline["total_s"]),
            _floor(aiob_baseline),
            float(aiob_staged["total_s"]),
            _floor(aiob_staged),
        )
    print(f"Fig 1c timeline source (AIOB):")
    print(f"  Naive  total={c_args[0]:.0f}s  floor={c_args[1]:.0f}s")
    print(f"  Staged total={c_args[2]:.0f}s  floor={c_args[3]:.0f}s")

    print("\nBuilding panel (a) - time decomposition...")
    build_fig_a(per_bench_baseline)
    print("Building panel (b) - bytes-moveable per backend...")
    build_fig_b(backends, prefetch_window_s)
    print("Building panel (c) - mechanism timeline...")
    build_fig_c(*c_args)
    print("Building combined preview...")
    build_combined(per_bench_baseline, backends, prefetch_window_s, c_args)

    print("\nWriting CSV companions...")
    dump_csv(
        "fig_motivation_decomp",
        [
            {
                "benchmark": b,
                "mode": mode,
                "n_sessions": d["n_sessions"],
                "total_s": round(d["total_s"], 2),
                "comm_s": round(d["comm_s"], 2),
                "stream_s": round(d["stream_s"], 2),
                "tool_shell_s": round(d["tool_shell_s"], 2),
                "tool_other_s": round(d["tool_other_s"], 2),
                "harness_s": round(d["harness_s"], 2),
                "tool_exec_pct": round(
                    100 * (d["tool_shell_s"] + d["tool_other_s"])
                    / d["total_s"], 1),
                "reasoning_pct": round(
                    100 * d["stream_s"] / d["total_s"], 1),
                "floor_s": round(d["total_s"] - d["tool_shell_s"], 2),
            }
            for (b, mode), d in by.items()
        ],
        ["benchmark", "mode", "n_sessions", "total_s", "comm_s", "stream_s",
         "tool_shell_s", "tool_other_s", "harness_s",
         "tool_exec_pct", "reasoning_pct", "floor_s"],
    )
    dump_csv(
        "fig_motivation_bytes_moveable",
        [
            {
                "backend": b["tier"],
                "ior_read_mibps_mean": round(b["bw_mibps"], 2),
                "ior_read_mibps_min": round(b["min_mibps"], 2),
                "ior_read_mibps_max": round(b["max_mibps"], 2),
                "prefetch_window_s": round(prefetch_window_s, 2),
                "bytes_moveable_mb": round(b["bw_mibps"] * prefetch_window_s, 1),
            }
            for b in backends if b["tier"] != "tmpfs"
        ],
        ["backend", "ior_read_mibps_mean", "ior_read_mibps_min",
         "ior_read_mibps_max", "prefetch_window_s", "bytes_moveable_mb"],
    )
    dump_csv(
        "fig_motivation_timeline",
        [
            {"row": "Naive", "total_s": round(c_args[0], 2),
             "floor_s": round(c_args[1], 2),
             "tool_s": round(c_args[0] - c_args[1], 2)},
            {"row": "AgentStage", "total_s": round(c_args[2], 2),
             "floor_s": round(c_args[3], 2),
             "tool_s": round(c_args[2] - c_args[3], 2)},
        ],
        ["row", "total_s", "floor_s", "tool_s"],
    )

    print(f"\nOutputs written to {FIG_DIR}/")
    print(f"CSV data written to {DATA_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
