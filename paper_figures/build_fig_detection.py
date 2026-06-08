"""Fig 3 — Tier-1 predictor accuracy across active campaign models.

Two side-by-side panels (~2.0x1.85 raw):
  (a) Tier-1 byte recall   (linear y, 0 to 1.02, ref line at 0.85)
  (b) Tier-1 byte overfetch (log y, ref line at 1.5x)

Per panel: strip plot + median tick, four columns, one per active model
(Haiku / Sonnet / Flash / Qwen3).

Scoring scope: This figure runs the detector LIVE per session against
thinking + text blocks only (tool_result blocks excluded). That matches
the headline H3 claim — "predictor during the LLM-side reasoning floor"
— which scores the staging decision BEFORE the tool fires. The byte_
metrics_v1.json files written by rescore.py currently include tool_result
blocks, which inflates tier_1_first overfetch on multiturn runs because
prior-turn list_dir outputs cause every per-class rule to fire (workspace
fanout). See PAPER_DEFENSE.md §5b.4b for the irreducible floor framing.

Source data: outputs/**/{stream.jsonl or turns/} + summary.json.
Active campaign only: claude-{haiku,sonnet}-4-5, gemini-2.5-flash, Qwen3.
Archived models (outputs/_archive_models/) are excluded by the path filter.

Outputs:
    paper/figures/fig_detection_a.{pdf,png}     tier-1 byte recall
    paper/figures/fig_detection_b.{pdf,png}     tier-1 byte overfetch
    paper/figures/fig_detection.{pdf,png}       combined preview
    paper/figures/data/fig_detection.csv        per-seed scored values

Run:
    uv run python paper_figures/build_fig_detection.py
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
    HALF_COL,
    dump_csv,
    style_axis,
)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from agentstage.detector.auto_rules import AutoRuleGenerator  # noqa: E402
from agentstage.detector.engine import parse_stream, run_detector  # noqa: E402
from agentstage.metrics.byte_metrics import byte_score  # noqa: E402
from agentstage.metrics.rescore import blocks_from_turns  # noqa: E402
from agentstage.workloads import get_workload  # noqa: E402

# Result-trio task allowlist. Detection scoring is restricted to the same
# (benchmark × task × model) cells §IV.D reports speedups on, so §IV.B and
# §IV.D characterize the same campaign.
_RESULT_TRIO = frozenset({
    # Curated only — §IV.B reports detection on the targeted regime
    # (per-instance bucket-decomposable workloads). Community benchmarks
    # are discussed separately in §IV.F (generalizability), where the
    # staging mechanism uses tier-2/tier-3 for monolithic-bucket workloads.
    "aiob_201", "aiob_202", "aiob_205",
})

_TASK_BENCH = {
    **{t: "aiob" for t in ("aiob_201", "aiob_202", "aiob_205")},
}

# Path-fragment exclusions — drop PoC-era runs, surgical campaigns, and
# H12 pathful-prompt sweeps that aren't part of the §IV.D 36-cell matrix.
_PATH_EXCLUDE_FRAGMENTS = ("poc/", "surgical_", "_sweep_pathful", "_smoke")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Model palette (one column per active campaign model). Order goes cheap to
# expensive within each provider, then provider order Anthropic / Google /
# Open. Colors follow paper_figures/_style.py.
# ---------------------------------------------------------------------------
_MODEL_ORDER = ["haiku", "sonnet", "flash", "qwen3"]
_MODEL_LABELS = {
    "haiku": "Haiku",
    "sonnet": "Sonnet",
    "flash": "Flash",
    "qwen3": "Qwen3",
}
_MODEL_COLOR = {
    "haiku": "#d62728",   # red (Anthropic, cheap)
    "sonnet": "#ff7f0e",  # orange (Anthropic, bigger)
    "flash": "#1f77b4",   # blue (Google)
    "qwen3": "#2ca02c",   # green (open-weight)
}


def _model_key(model: str) -> str | None:
    m = (model or "").lower()
    if "claude-haiku" in m:
        return "haiku"
    if "claude-sonnet" in m:
        return "sonnet"
    if "gemini-2.5-flash" in m or "gemini-flash" in m:
        return "flash"
    if "qwen" in m:
        return "qwen3"
    return None


# ---------------------------------------------------------------------------
# Load data — live-detector scoring, scope = thinking + text only
# ---------------------------------------------------------------------------
def _score_predictor_floor(blocks, workload, ruleset) -> dict | None:
    """Run the detector against the signals available before a tool fires:
    thinking + visible text from the current turn plus tool_result blocks
    from prior turns (the agent's own list_dir feedback, per §III.C). This
    matches what the live SessionDetector actually sees at decision time.

    Returns {byte_recall, byte_overfetch, n_predicted, n_ground_truth,
    bytes_predicted, bytes_ground_truth} or None if there is nothing to
    score (no scannable content, or empty ground truth)."""
    scan_blocks = [b for b in blocks
                   if b.type in ("thinking", "text", "tool_result")]
    if not scan_blocks:
        return None
    detection = run_detector(
        scan_blocks, workload.workspace_prior, ruleset, per_char=False,
    )
    # Score tier-1 staging against the workload's full read set
    # (every file the agent's session ultimately opens), not just the
    # first-inspect probe. On integrity-manifest workloads the agent
    # reads the whole corpus in one shell call, so the eligible
    # staging target equals the full set, and tier-1 byte overfetch
    # against that target is the meaningful precision measurement.
    gt = workload.ground_truth_full
    score = byte_score(detection.tier_1.detected_files, gt, workload.prefix_map)
    d = score.to_dict()
    return {
        "byte_recall": float(d.get("byte_recall", 0.0)),
        "byte_overfetch": float(d.get("byte_overfetch", 0.0)),
        "n_predicted": int(d.get("n_predicted",
                                  d.get("n_detected", 0))),
        "n_ground_truth": int(d.get("n_ground_truth", 0)),
        "bytes_predicted": int(d.get("bytes_predicted",
                                       d.get("bytes_detected", 0))),
        "bytes_ground_truth": int(d.get("bytes_ground_truth", 0)),
    }


def _blocks_for_run(run_dir: Path, summary: dict):
    """Build the StreamBlock list for a run dir, picking the right source
    depending on whether the run is single-turn (stream.jsonl) or
    multiturn (turns/)."""
    stream_path = run_dir / "stream.jsonl"
    turns_dir = run_dir / "turns"
    if stream_path.is_file():
        provider = summary.get("provider")
        return parse_stream(stream_path, provider=provider)
    if turns_dir.is_dir():
        return blocks_from_turns(turns_dir)
    return None


REPS_PER_CELL = 3   # latest N reps per (task, model_key) cell


def load_seed_records() -> list[dict]:
    """Walk outputs/**/summary.json. For each result-trio multiturn session,
    auto-generate its rule set from workload metadata, score tier_1_first
    against thinking+text blocks, and emit a row.

    Returns at most REPS_PER_CELL rows per (task, model_key) cell, taking
    the lexicographically-last paths (timestamps in directory names sort
    correctly, so this picks the most recent reps)."""
    rows: list[dict] = []
    skipped_no_blocks = 0
    skipped_unknown_task = 0
    skipped_out_of_trio = 0
    skipped_path_excluded = 0
    skipped_no_predict = 0
    for sf in (REPO / "outputs").rglob("summary.json"):
        if "_archive" in sf.parts:
            continue
        rel_path = str(sf.relative_to(REPO / "outputs"))
        if any(frag in rel_path for frag in _PATH_EXCLUDE_FRAGMENTS):
            skipped_path_excluded += 1
            continue
        try:
            s = json.loads(sf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        model_key = _model_key(s.get("model") or "")
        if model_key is None:
            continue
        rd = sf.parent
        task_id = s.get("task")
        if not task_id:
            continue
        if task_id not in _RESULT_TRIO:
            skipped_out_of_trio += 1
            continue
        try:
            workload = get_workload(task_id)
        except (KeyError, FileNotFoundError):
            skipped_unknown_task += 1
            continue
        # Auto-generate the rule set from workload metadata (no model traces
        # or evaluation outputs are consumed — see §IV.B leakage guard).
        try:
            ruleset = AutoRuleGenerator(
                workload_id=task_id,
                task_instruction=workload.task.task_inst,
                workspace_prior_keys=tuple(workload.workspace_prior.keys()),
            ).generate()
        except Exception:  # noqa: BLE001
            skipped_unknown_task += 1
            continue
        blocks = _blocks_for_run(rd, s)
        if blocks is None or not blocks:
            skipped_no_blocks += 1
            continue
        try:
            scored = _score_predictor_floor(blocks, workload, ruleset)
        except Exception:  # noqa: BLE001 — keep one bad seed from killing the build
            continue
        if scored is None:
            skipped_no_blocks += 1
            continue
        if scored["n_predicted"] == 0 and scored["n_ground_truth"] > 0:
            # Rule library didn't fire — well_defined=False slot.
            skipped_no_predict += 1
            continue
        rows.append({
            "rel_dir": str(rd.relative_to(REPO / "outputs")),
            "model": s.get("model"),
            "model_key": model_key,
            "task": task_id,
            "bench": _TASK_BENCH[task_id],
            "byte_recall": scored["byte_recall"],
            "byte_overfetch": scored["byte_overfetch"],
            "n_predicted": scored["n_predicted"],
            "n_ground_truth": scored["n_ground_truth"],
        })
    print(f"  filtered: out_of_trio={skipped_out_of_trio}, "
          f"path_excluded={skipped_path_excluded}, "
          f"unknown_task={skipped_unknown_task}, "
          f"no_blocks={skipped_no_blocks}, "
          f"no_predict={skipped_no_predict}")

    # Cap at REPS_PER_CELL latest reps per (task, model_key) cell so §IV.B
    # aligns with §IV.D's 36-cell matrix (3 benchmarks × 3 tasks × 4 models
    # × REPS_PER_CELL reps).
    from collections import defaultdict
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_cell[(r["task"], r["model_key"])].append(r)
    capped: list[dict] = []
    dropped_excess = 0
    for cell, cell_rows in by_cell.items():
        # Lexicographic sort on rel_dir — directory names contain timestamps
        # like ...T174525... so the last N entries are the most recent.
        cell_rows.sort(key=lambda r: r["rel_dir"])
        kept = cell_rows[-REPS_PER_CELL:]
        capped.extend(kept)
        dropped_excess += max(0, len(cell_rows) - REPS_PER_CELL)
    print(f"  capped to latest {REPS_PER_CELL} reps per cell: "
          f"kept={len(capped)}, dropped_excess={dropped_excess}, "
          f"cells_seen={len(by_cell)}")
    return capped


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _strip_box(ax, values_per_model: dict[str, list[float]],
               *, jitter: float = 0.10, marker_size: float = 12.0,
               ylabel: str = "", yref: float | None = None,
               yref_label: str | None = None,
               ylim: tuple[float, float] | None = None,
               log_y: bool = False, show_legend: bool = False) -> None:
    """Draw a strip plot with a median tick per model column.
    Models on x-axis at integer positions 0..N-1. When show_legend is
    set, models are identified by a 2-row legend above the plot (matching
    the per-task panel and Fig 5/6) and the x-tick text is dropped, since
    color already encodes the model and 4 names do not fit horizontally.
    """
    rng = np.random.RandomState(7)

    for i, mk in enumerate(_MODEL_ORDER):
        vals = values_per_model.get(mk, [])
        if not vals:
            continue
        xs = i + (rng.rand(len(vals)) - 0.5) * 2 * jitter
        ax.scatter(
            xs, vals,
            s=marker_size, color=_MODEL_COLOR[mk], alpha=0.55,
            edgecolors="white", linewidths=0.4, zorder=3,
            label=_MODEL_LABELS[mk],
        )

    # Median tick (short horizontal bar through each column). We skip the
    # box-and-whisker because the tier-1 overfetch distribution is bimodal
    # (clusters near 1.0 = precise match, and near 12 = workspace fanout),
    # which makes a box centered on the median misleading. Strip plot +
    # median tick lets the bimodality speak for itself.
    for i, mk in enumerate(_MODEL_ORDER):
        vals = values_per_model.get(mk, [])
        if not vals:
            continue
        med = float(np.percentile(vals, 50))
        ax.plot([i - 0.24, i + 0.24], [med, med],
                color="black", linewidth=1.4, zorder=6,
                solid_capstyle="round")

    # Reference line. Anchor the label at the left edge above the line so
    # it never collides with data clusters in the right-hand columns.
    n_models = len(_MODEL_ORDER)
    if yref is not None:
        ax.axhline(yref, linestyle="--", color="#666666", linewidth=0.7,
                   alpha=0.85, zorder=2)
        if yref_label:
            ax.text(-0.45, yref, yref_label,
                    va="center", ha="left", color="#444444",
                    bbox=dict(boxstyle="round,pad=0.10",
                              facecolor="white", edgecolor="none", alpha=0.9))

    # x-axis.
    ax.set_xticks(list(range(n_models)))
    if show_legend:
        # Model identity comes from the 2-row legend above; drop the
        # x-tick text (4 names do not fit horizontally) but keep ticks.
        ax.set_xticklabels([""] * n_models)
    else:
        # Slight rotation keeps 4 column labels from touching when narrow.
        ax.set_xticklabels(
            [_MODEL_LABELS[mk] for mk in _MODEL_ORDER],
            rotation=20, ha="right",
        )
    ax.set_xlim(-0.50, n_models - 1 + 0.50)

    # y-axis
    if log_y:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    style_axis(ax, xlabel="Model", ylabel=ylabel)

    if show_legend:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                  frameon=False, handlelength=1.0, handletextpad=0.4,
                  columnspacing=1.0, labelspacing=0.3, borderaxespad=0.0)


# ---------------------------------------------------------------------------
# Build each panel
# ---------------------------------------------------------------------------
def _bucket_by_model(rows: list[dict], key: str,
                     drop_nonfinite: bool = False
                     ) -> dict[str, list[float]]:
    by: dict[str, list[float]] = {mk: [] for mk in _MODEL_ORDER}
    for r in rows:
        v = r[key]
        if drop_nonfinite and (v <= 0 or v == float("inf")):
            continue
        by[r["model_key"]].append(v)
    return by


def build_panel_a(rows: list[dict], out_name: str = "fig_detection_a") -> Path:
    """Tier-1 byte recall: 0..1 linear, ref line at 0.85."""
    # Slightly wider than HALF_COL because 4 column labels need the room
    # at 10pt serif. Combined preview keeps the HALF_COL convention.
    fig, ax = plt.subplots(figsize=(2.0, 1.85))
    # Panel a: models on the x-axis (no legend; that would be redundant
    # with the axis). Keeps the bolder median line and the spread-out
    # dots from the pile-up fix.
    _strip_box(
        ax, _bucket_by_model(rows, "byte_recall"),
        ylabel="Tier-1 Byte Recall",
        yref=0.85, yref_label="0.85",
        ylim=(-0.04, 1.06),
        jitter=0.18, marker_size=46.0,
    )
    pdf = FIG_DIR / f"{out_name}.pdf"
    png = FIG_DIR / f"{out_name}.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=200)
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return pdf


def build_panel_b(rows: list[dict], out_name: str = "fig_detection_b") -> Path:
    """Per-task recall distribution: strip plot of byte_recall per task
    (3 columns for the 3 curated tasks), colored by model. Decomposes
    the per-model headline (panel a) by task so the reader sees on
    which tasks each model achieves full coverage."""
    rng = np.random.RandomState(7)

    # Shorter plot area than panel a (~20% shorter axes): the recall
    # distribution is concentrated at 1.0 with a few low outliers, so a
    # squatter axes reads fine and keeps 3b's total height (it carries a
    # bottom legend) close to 3a's.
    fig, ax = plt.subplots(figsize=(2.0, 1.44))

    # Task ordering matches the workload table
    task_order = ["aiob_201", "aiob_202", "aiob_205"]
    task_labels = {
        "aiob_201": "igsr",
        "aiob_202": "jwst",
        "aiob_205": "cross",
    }

    # bucket by (task, model_key)
    by_task_model = {(t, mk): [] for t in task_order for mk in _MODEL_ORDER}
    for r in rows:
        if r["task"] in task_order:
            by_task_model[(r["task"], r["model_key"])].append(r["byte_recall"])

    # Plot per-(task) groups; within each task, scatter per model with jitter
    n_models = len(_MODEL_ORDER)
    for ti, task in enumerate(task_order):
        for mi, mk in enumerate(_MODEL_ORDER):
            vals = by_task_model[(task, mk)]
            if not vals:
                continue
            # Offset per model within the task group
            offset = (mi - (n_models - 1) / 2) * 0.16
            xs = ti + offset + (rng.rand(len(vals)) - 0.5) * 0.06
            ax.scatter(xs, vals, s=46, color=_MODEL_COLOR[mk], alpha=0.75,
                       edgecolors="white", linewidths=0.4, zorder=3,
                       label=_MODEL_LABELS[mk] if ti == 0 else None)

    # Per-task median line (matches Fig 5/6's bold rounded group line).
    cluster_half = (n_models - 1) / 2 * 0.16 + 0.10
    for ti, task in enumerate(task_order):
        tvals = [v for mk in _MODEL_ORDER for v in by_task_model[(task, mk)]]
        if not tvals:
            continue
        med = float(np.percentile(tvals, 50))
        ax.plot([ti - cluster_half, ti + cluster_half], [med, med],
                color="black", linewidth=1.4, zorder=6,
                solid_capstyle="round")

    # 0.85 reference
    ax.axhline(0.85, linestyle="--", color="#666666", linewidth=0.7,
               alpha=0.85, zorder=2)
    ax.text(-0.45, 0.85, "0.85", va="center", ha="left", color="#444444",
            bbox=dict(boxstyle="round,pad=0.10",
                      facecolor="white", edgecolor="none", alpha=0.9))

    ax.set_xticks(list(range(len(task_order))))
    ax.set_xticklabels([task_labels[t] for t in task_order])
    ax.set_xlim(-0.50, len(task_order) - 1 + 0.50)
    ax.set_ylim(-0.04, 1.06)
    style_axis(ax, xlabel="Curated Task", ylabel="Tier-1 Byte Recall")
    # Legend BELOW the plot in 2 columns. Placing it at the bottom (rather
    # than on top) leaves both panels' plot areas top-aligned, and a 2-row
    # block stays within the plot width (a single 4-wide row would stretch
    # the panel).
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2,
              frameon=False, handlelength=1.0, handletextpad=0.4,
              columnspacing=1.0, labelspacing=0.3, borderaxespad=0.0)

    pdf = FIG_DIR / f"{out_name}.pdf"
    png = FIG_DIR / f"{out_name}.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=200)
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return pdf


def build_combined(rows: list[dict], out_name: str = "fig_detection") -> Path:
    """Side-by-side preview (full single-column width)."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(3.85, 1.85))
    _strip_box(ax_a, _bucket_by_model(rows, "byte_recall"),
               ylabel="Tier-1 Byte Recall",
               yref=0.85, yref_label="0.85", ylim=(-0.04, 1.06))
    _strip_box(ax_b, _bucket_by_model(rows, "byte_overfetch", drop_nonfinite=True),
               ylabel=r"Tier-1 Byte Overfetch ($\times$)",
               yref=1.5, yref_label=r"1.5$\times$", ylim=(0.9, 100.0),
               log_y=True)
    fig.tight_layout(pad=0.3)
    pdf = FIG_DIR / f"{out_name}.pdf"
    png = FIG_DIR / f"{out_name}.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=200)
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return pdf


# ---------------------------------------------------------------------------
# Summary stats (for paper text)
# ---------------------------------------------------------------------------
def _quantiles(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "median": None, "p25": None, "p75": None,
                "p95": None, "max": None, "frac_ge_0_85": None,
                "frac_le_1_5x": None}
    return {
        "n": len(vals),
        "median": float(statistics.median(vals)),
        "p25": float(np.percentile(vals, 25)),
        "p75": float(np.percentile(vals, 75)),
        "p95": float(np.percentile(vals, 95)),
        "max": float(max(vals)),
    }


def report_stats(rows: list[dict]) -> None:
    print("\nFig 3 distribution summary")
    print("=" * 72)
    for metric_key, metric_name, threshold, op in [
        ("byte_recall", "Tier-1 byte recall", 0.85, "ge"),
        ("byte_overfetch", "Tier-1 byte overfetch", 1.5, "le"),
    ]:
        print(f"\n{metric_name}")
        print("-" * 72)
        for mk in _MODEL_ORDER:
            vals = [r[metric_key] for r in rows if r["model_key"] == mk
                    and r[metric_key] != float("inf")]
            q = _quantiles(vals)
            if not vals:
                print(f"  {_MODEL_LABELS[mk]:14s}  n=0")
                continue
            frac = (sum(1 for v in vals if v >= threshold) / len(vals)
                    if op == "ge"
                    else sum(1 for v in vals if v <= threshold) / len(vals))
            print(
                f"  {_MODEL_LABELS[mk]:14s}  "
                f"n={q['n']:3d}  "
                f"median={q['median']:.3f}  "
                f"p25={q['p25']:.3f}  p75={q['p75']:.3f}  "
                f"p95={q['p95']:.3f}  max={q['max']:.3f}  "
                f"|  frac_{op}_{threshold:g}={frac:.0%}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    rows = load_seed_records()
    if not rows:
        print("ERROR: no scored sessions found", file=sys.stderr)
        return 2
    print(f"Loaded {len(rows)} scored seeds from outputs/**/byte_metrics_v1.json")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["model_key"]] = counts.get(r["model_key"], 0) + 1
    for mk in _MODEL_ORDER:
        print(f"  {_MODEL_LABELS[mk]:14s}  n={counts.get(mk, 0)}")

    print("\nBuilding panel (a) - byte recall...")
    build_panel_a(rows)
    print("Building panel (b) - byte overfetch...")
    build_panel_b(rows)
    print("Building combined preview...")
    build_combined(rows)

    report_stats(rows)

    print("\nWriting CSV companion...")
    dump_csv(
        "fig_detection",
        [
            {
                "rel_dir": r["rel_dir"],
                "model_key": r["model_key"],
                "model": r["model"],
                "task": r["task"],
                "byte_recall": round(r["byte_recall"], 4),
                "byte_overfetch": round(r["byte_overfetch"], 4),
                "n_predicted": r["n_predicted"],
                "n_ground_truth": r["n_ground_truth"],
            }
            for r in rows
        ],
        ["rel_dir", "model_key", "model", "task",
         "byte_recall", "byte_overfetch",
         "n_predicted", "n_ground_truth"],
    )

    print(f"\nOutputs written to {FIG_DIR}/")
    print(f"CSV data written to {DATA_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
