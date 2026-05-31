"""Fig 6 - per-session case studies.

Three submitted staged sessions across two benchmarks and two models,
visualized as horizontal session-time timelines with annotated events:

  - t=0   : agent's first thinking delta starts
  - t_d   : detector first dispatches a prefetch (rule fires)
  - t_s   : first shell-tool fires (i.e. agent's tool starts reading)
  - t_end : session ends

The shaded region between t_d and t_s is the staging window AgentStage
gets for the cold-to-hot copy in this session.

Outputs:
    paper/figures/fig_cases.{pdf,png}
    paper/figures/data/fig_cases.csv
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
)

REPO = Path(__file__).resolve().parent.parent

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

_CASES = [
    {
        "label": "AIOB / aiob_110 / Haiku 4.5",
        "subdir": "outputs/aiob_mt",
        "task": "aiob_110",
        "model_substr": "claude-haiku-4-5",
        "context": "Steinmetz NWB, 14.7 GB",
    },
    {
        "label": "AIOB / aiob_104 / Sonnet 4.5",
        "subdir": "outputs/aiob_mt",
        "task": "aiob_104",
        "model_substr": "claude-sonnet-4-5",
        "context": "1000 Genomes BAM",
    },
    {
        "label": "DSBench / lmsys / Sonnet 4.5",
        "subdir": "outputs/dsbench_mt",
        "task": "lmsys-chatbot-arena",
        "model_substr": "claude-sonnet-4-5",
        "context": "LLM-eval tabular",
    },
]


def _find_median_submitted(subdir: str, task: str, model_substr: str) -> dict | None:
    runs: list[tuple[float, Path, dict]] = []
    for sf in (REPO / subdir).rglob("summary.json"):
        if "_archive" in sf.parts:
            continue
        if sf.parent.parent.name.startswith("_smoke"):
            continue
        try:
            s = json.loads(sf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if s.get("mode") != "staged":
            continue
        if model_substr not in (s.get("model") or ""):
            continue
        if s.get("task") != task:
            continue
        if not s.get("submitted"):
            continue
        elapsed = s.get("session_elapsed_s")
        if elapsed is None or elapsed < 5:
            continue
        runs.append((float(elapsed), sf.parent, s))
    if not runs:
        return None
    runs.sort()
    picked_elapsed, picked_dir, picked_s = runs[len(runs) // 2]
    return {"run_dir": picked_dir, "summary": picked_s,
            "elapsed_s": picked_elapsed,
            "median_elapsed_s": statistics.median([r[0] for r in runs]),
            "n_runs": len(runs)}


def _extract_events(summary: dict) -> dict:
    cum = 0.0
    first_disp_t = None
    first_disp_n = 0
    first_tool_t = None
    for t in summary.get("per_turn") or []:
        dur = float(t.get("duration_s", 0) or 0)
        if t.get("dispatched_prefetches") and first_disp_t is None:
            first_disp_t = cum
            first_disp_n = sum(d.get("n_files", 0)
                                for d in t["dispatched_prefetches"])
        if "run_shell_command" in (t.get("tool_names") or []) and first_tool_t is None:
            first_tool_t = cum
        cum += dur
    return {
        "t_dispatch_s": first_disp_t,
        "n_dispatched_files": first_disp_n,
        "t_first_tool_s": first_tool_t,
        "t_end_s": cum,
    }


_COLOR_FLOOR = "#7f7f7f"
_COLOR_TOOL = "#d62728"
_COLOR_STAGE = "#9ecae1"
_COLOR_DISPATCH = "#1f77b4"
_COLOR_FIRE = "#222222"


def build(cases: list[dict], out_name: str = "fig_cases") -> Path:
    n = len(cases)
    fig, ax = plt.subplots(figsize=(FULL_COL_W, 0.55 * n + 0.7))

    bar_h = 0.42
    label_pad = 0.06
    y_positions = list(range(n - 1, -1, -1))  # top-down

    # Determine common x range
    x_max = max(c["events"]["t_end_s"] for c in cases) * 1.05

    for y, c in zip(y_positions, cases):
        e = c["events"]
        t_disp = e["t_dispatch_s"] if e["t_dispatch_s"] is not None else 0.0
        t_fire = e["t_first_tool_s"] if e["t_first_tool_s"] is not None else 0.0
        t_end = e["t_end_s"]

        # The "LLM floor" portion (before tool fires)
        ax.barh(y, t_fire - 0.0, height=bar_h, left=0.0,
                color=_COLOR_FLOOR, alpha=0.55, edgecolor="white",
                linewidth=0.4, zorder=3)
        # The "Tool exec" portion (after first tool fires)
        ax.barh(y, t_end - t_fire, height=bar_h, left=t_fire,
                color=_COLOR_TOOL, alpha=0.45, edgecolor="white",
                linewidth=0.4, zorder=3)

        # Staging window (between dispatch and first tool fire)
        if t_disp is not None and t_fire is not None and t_fire > t_disp:
            ax.barh(y, t_fire - t_disp, height=bar_h * 0.45,
                    left=t_disp, color=_COLOR_STAGE,
                    edgecolor="black", linewidth=0.4,
                    hatch="//", zorder=4)

        # Detector fire mark
        ax.scatter(t_disp, y + bar_h / 2 + 0.05, marker="v",
                   color=_COLOR_DISPATCH, s=22, zorder=5,
                   edgecolors="none")
        # Tool fire mark
        ax.scatter(t_fire, y - bar_h / 2 - 0.05, marker="^",
                   color=_COLOR_FIRE, s=22, zorder=5,
                   edgecolors="none")

        # Annotation: dispatched files count
        ax.text(t_disp + 1.0, y + bar_h / 2 + 0.15,
                f"{e['n_dispatched_files']} files staged",
                fontsize=9, color=_COLOR_DISPATCH, va="bottom",
                bbox=dict(boxstyle="round,pad=0.10",
                           facecolor="white", edgecolor="none", alpha=0.85))

        # Row label (left, two lines)
        ax.text(-x_max * 0.005, y, c["label"] + "\n" + c["context"],
                ha="right", va="center", fontsize=9)

    # Legend with patches and markers
    handles = [
        mpatches.Patch(facecolor=_COLOR_FLOOR, alpha=0.55,
                       edgecolor="white", linewidth=0.4,
                       label="LLM Floor"),
        mpatches.Patch(facecolor=_COLOR_TOOL, alpha=0.45,
                       edgecolor="white", linewidth=0.4,
                       label="Tool Exec"),
        mpatches.Patch(facecolor=_COLOR_STAGE, hatch="//",
                       edgecolor="black", linewidth=0.4,
                       label="Staging Window"),
        plt.Line2D([0], [0], marker="v", color="white",
                   markerfacecolor=_COLOR_DISPATCH,
                   markeredgecolor="none", markersize=6,
                   label="Detector Fires"),
        plt.Line2D([0], [0], marker="^", color="white",
                   markerfacecolor=_COLOR_FIRE,
                   markeredgecolor="none", markersize=6,
                   label="Tool Fires"),
    ]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.20), ncol=5, frameon=False,
              handlelength=1.0, columnspacing=0.6, handletextpad=0.3,
              fontsize=9)

    # x-axis: session time
    ax.set_xlim(0.0, x_max)
    ax.set_xlabel("Session Time (s)")
    ax.set_yticks([])
    ax.set_ylim(-0.6, n - 0.4)
    # Light vertical grid for orientation
    ax.grid(axis="x", linestyle=":", alpha=0.35, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fig.tight_layout(pad=0.3)
    pdf = FIG_DIR / f"{out_name}.pdf"
    png = FIG_DIR / f"{out_name}.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=200)
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return pdf


def main() -> int:
    cases: list[dict] = []
    for c in _CASES:
        pick = _find_median_submitted(c["subdir"], c["task"], c["model_substr"])
        if pick is None:
            print(f"WARN no submitted run for {c['task']} / {c['model_substr']}",
                  file=sys.stderr)
            continue
        events = _extract_events(pick["summary"])
        cases.append({**c, "pick": pick, "events": events})

    if not cases:
        print("ERROR no cases", file=sys.stderr)
        return 2

    for c in cases:
        p = c["pick"]
        e = c["events"]
        print(f"  {c['task']:32s}  {c['model_substr'][:14]:14s}  "
              f"elapsed={p['elapsed_s']:.1f}s  "
              f"dispatch@{e['t_dispatch_s'] or 0:.2f}s (n={e['n_dispatched_files']}); "
              f"tool@{e['t_first_tool_s'] or 0:.2f}s; "
              f"window={(e['t_first_tool_s'] or 0) - (e['t_dispatch_s'] or 0):.1f}s")

    build(cases)
    dump_csv(
        "fig_cases",
        [{"label": c["label"], "context": c["context"],
          "task": c["task"], "model": c["model_substr"],
          "elapsed_s": round(c["pick"]["elapsed_s"], 2),
          "t_dispatch_s": round(c["events"]["t_dispatch_s"] or 0, 3),
          "n_dispatched_files": c["events"]["n_dispatched_files"],
          "t_first_tool_s": round(c["events"]["t_first_tool_s"] or 0, 3),
          "t_end_s": round(c["events"]["t_end_s"], 3),
          "staging_window_s": round(
              (c["events"]["t_first_tool_s"] or 0)
              - (c["events"]["t_dispatch_s"] or 0), 3)}
         for c in cases],
        ["label", "context", "task", "model", "elapsed_s",
         "t_dispatch_s", "n_dispatched_files",
         "t_first_tool_s", "t_end_s", "staging_window_s"],
    )

    print(f"\nOutputs written to {FIG_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
