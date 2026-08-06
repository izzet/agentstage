"""Audit per-cell shell-timeout activity across the 27-cell matrix.

Identifies cells where one mode hit the per-turn shell timeout more often
than the other — those cells have inflated reported speedup. See
PAPER_DEFENSE.md §5b for the methodology discussion.

Usage:
    python scripts/microbench/audit_timeouts.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SWEEPS = {
    ("DSBench", "haiku"):  ROOT / "outputs/dsbench_mt/_sweep_20260525T113623",
    ("DSBench", "sonnet"): ROOT / "outputs/dsbench_mt/_sweep_sonnet_20260526T150919",
    ("DSBench", "gemini"): ROOT / "outputs/dsbench_mt/_sweep_gemini_20260526T151730",
    ("MLE-bench", "haiku"):  ROOT / "outputs/mlebench_mt/_sweep_20260525T165123",
    ("MLE-bench", "sonnet"): ROOT / "outputs/mlebench_mt/_sweep_sonnet_20260526T150922_mle",
    ("MLE-bench", "gemini"): ROOT / "outputs/mlebench_mt/_sweep_gemini_20260526T151731_mle",
    ("AIOB", "haiku"):  ROOT / "outputs/aiob_mt/_sweep_haiku_20260527T164353_v3",
    ("AIOB", "sonnet"): ROOT / "outputs/aiob_mt/_sweep_sonnet_20260527T164353_v3",
    ("AIOB", "gemini"): ROOT / "outputs/aiob_mt/_sweep_gemini_20260527T164353_v3",
}

TASKS = {
    "DSBench": ["lmsys-chatbot-arena", "tabular-playground-series-may-2022",
                "ventilator-pressure-prediction"],
    "MLE-bench": ["dogs-vs-cats-redux-kernels-edition",
                  "histopathologic-cancer-detection",
                  "new-york-city-taxi-fare-prediction"],
    "AIOB": ["aiob_103", "aiob_107", "aiob_110"],
}

# Per-turn shell timeout in seconds. DSBench/MLE-bench use 180 s; AIOB v3 uses 300 s.
TIMEOUTS = {"DSBench": 180, "MLE-bench": 180, "AIOB": 300}

CLEAN, SYMMETRIC, ASYMMETRIC = [], [], []

print(f"{'Cell':<58} {'baseTO':>7} {'staTO':>7} {'asym':>6} {'class':<10}")
print("-" * 90)

for (bench, model), sweep_dir in SWEEPS.items():
    timeout_s = TIMEOUTS[bench]
    cliff = timeout_s - 5
    for task in TASKS[bench]:
        base_to = staged_to = 0
        base_n = staged_n = 0
        for f in sorted(sweep_dir.glob(f"{task}_*/summary.json")):
            s = json.load(open(f))
            if s.get("crash"):
                continue
            tos = sum(1 for t in s.get("per_turn", [])
                      if "run_shell_command" in t.get("tool_names", [])
                      and t.get("duration_s", 0) >= cliff)
            if s.get("mode") == "baseline":
                base_to += tos; base_n += 1
            else:
                staged_to += tos; staged_n += 1
        asym = abs(base_to - staged_to)
        if base_to == 0 and staged_to == 0:
            klass = "CLEAN"; CLEAN.append((bench, model, task))
        elif asym <= 1 and (base_to + staged_to) > 0:
            klass = "SYMMETRIC"; SYMMETRIC.append((bench, model, task, base_to, staged_to))
        else:
            klass = "ASYMMETRIC"; ASYMMETRIC.append((bench, model, task, base_to, staged_to))
        cell = f"{bench:<10} {model:<7} {task[:32]:<32}"
        print(f"{cell:<58} {base_to:>4}/{base_n}    {staged_to:>4}/{staged_n}    {asym:>4}    {klass}")

print()
print(f"Classification: clean={len(CLEAN)}, symmetric={len(SYMMETRIC)}, "
      f"asymmetric={len(ASYMMETRIC)}")
print()
print("ASYMMETRIC cells (reported speedup likely inflated):")
for bench, model, task, b, s in ASYMMETRIC:
    print(f"  {bench:<10} {model:<7} {task:<40}  base_TO={b}  staged_TO={s}")
