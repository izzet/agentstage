"""Fig 5 - detector activation latency CDF.

Per staged AIOB session: time from first thinking delta to first
detector activation whose target files overlap with the workload's
first-inspect ground truth. Absolute session time is reconstructed
from per_turn[].duration_s in summary.json plus per-turn thinking
delta t_ms (which is per-turn relative for the multiturn corpus).

Outputs:
    paper/figures/fig_activation.{pdf,png}
    paper/figures/data/fig_activation.csv

Inputs:
    outputs/aiob_mt/*/aiob_{104,107,110}_staged_r*/summary.json
        plus turns/turn_NN/thinking.jsonl (real per-delta timestamps)
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
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
sys.path.insert(0, str(REPO / "src"))

from agentstage.detector.rules import get_ruleset  # noqa: E402
from agentstage.workloads import get_workload  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

VALID_TASKS = ("aiob_104", "aiob_107", "aiob_110")

_MODEL_ORDER = ["haiku", "sonnet", "flash", "qwen3"]
_MODEL_LABELS = {"haiku": "Haiku", "sonnet": "Sonnet",
                  "flash": "Flash", "qwen3": "Qwen3"}
_MODEL_COLOR = {
    "haiku": "#d62728",
    "sonnet": "#ff7f0e",
    "flash": "#1f77b4",
    "qwen3": "#2ca02c",
}


def _model_key(m: str) -> str | None:
    m = (m or "").lower()
    if "claude-haiku" in m:
        return "haiku"
    if "claude-sonnet" in m:
        return "sonnet"
    if "gemini-2.5-flash" in m or "gemini-flash" in m:
        return "flash"
    if "qwen" in m:
        return "qwen3"
    return None


def _thinking_deltas(turn_dir: Path) -> list[tuple[float, str]]:
    p = turn_dir / "thinking.jsonl"
    if not p.is_file():
        return []
    out: list[tuple[float, str]] = []
    for line in p.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = r.get("t_ms")
        d = r.get("delta", "")
        if t is None:
            continue
        out.append((float(t), str(d)))
    return out


def _latency_s(rd: Path, summary: dict, task: str) -> float | None:
    """Return time from first thinking delta to first correct detector
    activation, in seconds (absolute session time)."""
    try:
        workload = get_workload(task)
        ruleset = get_ruleset(task)
    except KeyError:
        return None
    gt = set(workload.ground_truth_first_inspect)
    prior = workload.workspace_prior
    compiled = [(r, re.compile(r.pattern, flags=re.IGNORECASE))
                for r in ruleset.rules]

    turn_offsets: dict[int, float] = {0: 0.0}
    cum_s = 0.0
    for t in summary.get("per_turn") or []:
        idx = t.get("turn", 0)
        dur = float(t.get("duration_s", 0) or 0)
        cum_s += dur
        turn_offsets[idx + 1] = cum_s * 1000.0

    acc = ""
    char_t_ms: list[float] = []
    first_t: float | None = None
    correct_fire_t: list[float] = []
    fired: set[str] = set()

    for tdir in sorted((rd / "turns").glob("turn_*")):
        try:
            tid = int(tdir.name.split("_", 1)[1])
        except (ValueError, IndexError):
            tid = 0
        offset_ms = turn_offsets.get(tid, tid * 1000.0)
        deltas = _thinking_deltas(tdir)
        if not deltas:
            continue
        for t_ms, txt in deltas:
            abs_t = offset_ms + t_ms
            if first_t is None:
                first_t = abs_t
            for ch in txt:
                acc += ch
                char_t_ms.append(abs_t)
            for rule, regex in compiled:
                if rule.name in fired:
                    continue
                m = regex.search(acc)
                if m:
                    fired.add(rule.name)
                    idx = min(m.end() - 1, len(char_t_ms) - 1)
                    t_fire = char_t_ms[idx]
                    det = set()
                    for k in rule.target_keys:
                        det.update(prior.get(k, ()))
                    if det & gt:
                        correct_fire_t.append(t_fire)

    if first_t is None or not correct_fire_t:
        return None
    return max(0.0, (min(correct_fire_t) - first_t) / 1000.0)


def load_latencies() -> list[dict]:
    rows: list[dict] = []
    for sf in (REPO / "outputs").rglob("summary.json"):
        if "_archive" in sf.parts:
            continue
        rel = sf.relative_to(REPO / "outputs")
        if len(rel.parts) < 2 or not str(rel).startswith("aiob_mt"):
            continue
        if rel.parts[1].startswith("_smoke"):
            continue
        try:
            s = json.loads(sf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if s.get("task") not in VALID_TASKS:
            continue
        if s.get("mode") != "staged":
            continue
        mk = _model_key(s.get("model") or "")
        if mk is None:
            continue
        try:
            lat = _latency_s(sf.parent, s, s.get("task"))
        except Exception:  # noqa: BLE001
            continue
        if lat is None:
            continue
        rows.append({
            "rel_dir": str(rel.parent),
            "task": s.get("task"),
            "model_key": mk,
            "model": s.get("model"),
            "latency_s": lat,
        })
    return rows


def build_cdf(rows: list[dict], out_name: str = "fig_activation") -> Path:
    fig, ax = plt.subplots(figsize=(FULL_COL_W, 2.0))

    # Pooled CDF (black)
    all_lats = sorted(r["latency_s"] for r in rows)
    n = len(all_lats)
    xs = [0.001] + all_lats + [all_lats[-1] * 1.5]
    ys = [0.0] + [(i + 1) / n for i in range(n)] + [1.0]
    ax.step(xs, ys, where="post", color="black",
            linewidth=1.6, zorder=5, label="All")

    # Per-model CDFs
    for mk in _MODEL_ORDER:
        ml = sorted(r["latency_s"] for r in rows if r["model_key"] == mk)
        if len(ml) < 3:
            continue
        mn = len(ml)
        mxs = [0.001] + ml + [max(all_lats) * 1.5]
        mys = [0.0] + [(i + 1) / mn for i in range(mn)] + [1.0]
        ax.step(mxs, mys, where="post",
                color=_MODEL_COLOR[mk], linewidth=1.0, alpha=0.85,
                zorder=4, label=_MODEL_LABELS[mk])

    ax.set_xscale("log")
    ax.set_xlim(0.05, 100.0)
    ax.set_xticks([0.1, 1.0, 10.0, 100.0])
    ax.set_xticklabels(["0.1", "1", "10", "100"])
    ax.set_xlabel("Time From Reasoning Start to First Correct Fire (s, log)")
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    style_axis(ax, ylabel="Cumulative % of Sessions")

    ax.axvline(1.0, color="#444444", linestyle="--",
               linewidth=0.6, alpha=0.7, zorder=2)
    ax.text(1.05, 0.05, "1 s", color="#444444",
            bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                      edgecolor="none", alpha=0.9))

    ax.legend(loc="lower right", frameon=False, ncol=2,
              handlelength=1.4, columnspacing=0.6, handletextpad=0.4)

    fig.tight_layout(pad=0.3)
    pdf = FIG_DIR / f"{out_name}.pdf"
    png = FIG_DIR / f"{out_name}.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=200)
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return pdf


def report_stats(rows: list[dict]) -> None:
    lats = sorted(r["latency_s"] for r in rows)
    n = len(lats)
    print("\nActivation latency summary")
    print("=" * 72)
    print(f"All n={n}  median={statistics.median(lats):.2f}s  "
          f"mean={statistics.mean(lats):.2f}s  "
          f"p75={lats[3*n//4]:.2f}s  p95={lats[int(n*0.95)]:.2f}s  "
          f"max={lats[-1]:.2f}s")
    for thr in [1.0, 2.0, 5.0, 10.0]:
        print(f"  fraction <= {thr:>4.1f}s : {sum(1 for v in lats if v <= thr) / n:.0%}")
    print()
    for mk in _MODEL_ORDER:
        ml = sorted(r["latency_s"] for r in rows if r["model_key"] == mk)
        if not ml:
            continue
        mn = len(ml)
        print(f"  {_MODEL_LABELS[mk]:7s}  n={mn:>3d}  "
              f"median={statistics.median(ml):.2f}s  "
              f"p95={ml[int(mn*0.95)] if mn > 5 else max(ml):.2f}s  "
              f"max={max(ml):.2f}s")


def main() -> int:
    rows = load_latencies()
    if not rows:
        print("ERROR: no latency rows", file=sys.stderr)
        return 2
    print(f"Loaded {len(rows)} staged AIOB sessions with measurable latency")
    build_cdf(rows)
    report_stats(rows)
    dump_csv(
        "fig_activation",
        [{"rel_dir": r["rel_dir"], "task": r["task"],
          "model_key": r["model_key"], "model": r["model"],
          "latency_s": round(r["latency_s"], 4)}
         for r in rows],
        ["rel_dir", "task", "model_key", "model", "latency_s"],
    )
    print(f"\nOutputs written to {FIG_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
