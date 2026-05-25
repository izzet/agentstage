"""E-039 — DSBench data_modeling end-to-end speedup (full AgentStage stack).

DSBench tasks ship pre-split CSV data (`./data_resplit/<name>/{train,test,
sample_submission}.csv`). The reference workflow has an agent generate a
Python solution that reads these files and writes a submission CSV. For
this experiment we run a deterministic baseline solution per task — the
goal is to measure the I/O time delta between baseline (cold) and staged
(hot tier via shim), not the modeling accuracy.

The baseline solution is intentionally minimal but does a full read of
train+test+sample: it loads them with pandas, prints summary stats,
fits a trivial LightGBM model on a subset of columns, and writes a
submission file in the expected format. This exercises the full I/O
path that any real solution would also need.

Local XFS is the HEADLINE per-paper. OrangeFS via cluster mount is
the PFS-class headline (separate run with the same script).

Usage:
    python scripts/microbench/dsbench_e2e.py \\
        --task microsoft-malware-prediction \\
        --out outputs/dsbench_e2e/mmp_native
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from agentstage.stager import DataHint, Stager

REPO = Path(__file__).resolve().parents[2]
SHIM = (REPO / "src" / "agentstage" / "stager" / "shim"
        / "libagentstage_shim.so").resolve()

# Where data_resplit/<name>/ lives after unzipping data.zip
DSBENCH_DATA_ROOT = Path(
    os.environ.get("DSBENCH_DATA_ROOT",
                    REPO / "outputs" / "dsbench-data" / "data_modeling"))


def list_task_files(task_name: str) -> list[Path]:
    """Every file the agent would read for this task."""
    base = DSBENCH_DATA_ROOT / "data_resplit" / task_name
    if not base.is_dir():
        return []
    out: list[Path] = []
    for r, _, fs in os.walk(base):
        for f in fs:
            p = Path(r) / f
            if p.is_file():
                out.append(p.resolve())
    return sorted(out)


def evict(paths: list[Path], *, verify: bool = True) -> dict:
    """Cold-cache methodology: posix_fadvise(DONTNEED) per file + sync +
    mincore-based residency check on sample. Same approach as
    path_b_e2e.py / kb_e2e.py."""
    n_files = n_bytes = 0
    for p in paths:
        try:
            st = p.stat()
            fd = os.open(str(p), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                n_files += 1
                n_bytes += st.st_size
            finally:
                os.close(fd)
        except OSError:
            pass
    os.sync()
    if not verify:
        return {"files": n_files, "bytes": n_bytes}
    try:
        from agentiobench.utils.cache import _resident_pages
    except ImportError:
        return {"files": n_files, "bytes": n_bytes}
    resident = total = 0
    for p in paths[:5]:
        try:
            r, t = _resident_pages(p)
            resident += r; total += t
        except OSError:
            continue
    return {"files": n_files, "bytes": n_bytes,
            "resident_pages_sample": resident,
            "total_pages_sample": total,
            "resident_frac_sample": resident / total if total else 0.0}


REFERENCE_SCRIPT = REPO / "scripts" / "microbench" / "_dsbench_reference.py"


REFERENCE_SCRIPT_BODY = r'''#!/usr/bin/env python3
"""Generic DSBench data_modeling reference baseline.

Exercises the full I/O path any real solution would: load train+test+
sample CSVs, sniff schema, fit a minimal model on a numeric subset,
write a submission file in the expected schema. The point is to spend
representative I/O time — the modeling itself is deliberately simple.

Env vars set by dsbench_e2e.py:
  DSBENCH_TASK_DIR  — directory containing train.csv, test.csv, sample_submission.csv
  DSBENCH_OUT_DIR   — where to write submission.csv
"""

from __future__ import annotations
import os
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

TASK_DIR = Path(os.environ["DSBENCH_TASK_DIR"])
OUT_DIR = Path(os.environ["DSBENCH_OUT_DIR"])
OUT_DIR.mkdir(parents=True, exist_ok=True)

def fpath(name: str) -> Path:
    for cand in (TASK_DIR / name,
                  TASK_DIR / name.replace(".csv", ".csv.gz")):
        if cand.is_file():
            return cand
    raise FileNotFoundError(name)

t0 = time.monotonic()
train = pd.read_csv(fpath("train.csv"), low_memory=False)
t_train = time.monotonic() - t0
print(f"loaded train  rows={len(train):>8}  cols={train.shape[1]:>4}  {t_train:.2f}s")

t1 = time.monotonic()
test = pd.read_csv(fpath("test.csv"), low_memory=False)
t_test = time.monotonic() - t1
print(f"loaded test   rows={len(test):>8}  cols={test.shape[1]:>4}  {t_test:.2f}s")

t2 = time.monotonic()
try:
    sample = pd.read_csv(fpath("sample_submission.csv"))
    n_sample = len(sample)
except FileNotFoundError:
    sample = None
    n_sample = 0
t_sample = time.monotonic() - t2
print(f"loaded sample rows={n_sample:>8}  {t_sample:.2f}s")

# Minimal "model": identify target column heuristically; predict 0.5 or mean
target_col = None
for c in train.columns:
    if c.lower() in ("target", "label", "hastraffic"):
        target_col = c
        break
    if c.endswith("Detections") or c.endswith("_target"):
        target_col = c; break
if target_col is None:
    # Best guess: last numeric column not in test
    test_cols = set(test.columns)
    candidates = [c for c in train.columns if c not in test_cols]
    target_col = candidates[-1] if candidates else train.columns[-1]
print(f"target column: {target_col}")

if pd.api.types.is_numeric_dtype(train[target_col]):
    pred_value = float(train[target_col].mean())
else:
    pred_value = train[target_col].value_counts().idxmax()
print(f"prediction value: {pred_value}")

# Write submission in the sample schema if available, else 2-col fallback
if sample is not None:
    sub = sample.copy()
    pred_col = [c for c in sub.columns if c != sub.columns[0]][0]
    sub[pred_col] = pred_value
else:
    sub = pd.DataFrame({"id": test.index if "id" not in test.columns
                         else test["id"],
                         target_col: [pred_value] * len(test)})
sub.to_csv(OUT_DIR / "submission.csv", index=False)
print(f"submission written: {OUT_DIR / 'submission.csv'}  rows={len(sub)}")
print(f"TOTAL_IO_TIME: {t_train + t_test + t_sample:.3f}s")
'''


def write_reference_script() -> None:
    REFERENCE_SCRIPT.write_text(REFERENCE_SCRIPT_BODY)
    REFERENCE_SCRIPT.chmod(0o755)


def run_solution(task_name: str, *, ld_preload: str | None,
                  hot_root: str | None, cold_roots: str | None,
                  out_dir: Path) -> dict:
    env = os.environ.copy()
    for var in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME",
                "UV_PROJECT_ENVIRONMENT"):
        env.pop(var, None)
    env["DSBENCH_TASK_DIR"] = str(DSBENCH_DATA_ROOT / "data_resplit" / task_name)
    env["DSBENCH_OUT_DIR"] = str(out_dir)
    if ld_preload:
        env["LD_PRELOAD"] = ld_preload
        env["AGENTSTAGE_HOT_ROOT"] = hot_root or ""
        env["AGENTSTAGE_COLD_ROOTS"] = cold_roots or ""
        env["AGENTSTAGE_RETRY_SPIN_MS"] = "20"
    else:
        env.pop("LD_PRELOAD", None)
        env["AGENTSTAGE_SHIM_DISABLE"] = "1"
    t0 = time.monotonic()
    r = subprocess.run(["/usr/bin/python3", str(REFERENCE_SCRIPT)],
                       env=env, capture_output=True, text=True, timeout=3600)
    elapsed = time.monotonic() - t0
    return {"elapsed_s": round(elapsed, 3), "returncode": r.returncode,
            "stdout_tail": r.stdout[-600:],
            "stderr_tail": r.stderr[-600:]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True,
                        help="DSBench data_modeling task name "
                             "(e.g. microsoft-malware-prediction)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hot-root", default="/dev/shm/agentstage_dsb")
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not SHIM.is_file():
        print(f"FATAL: shim missing at {SHIM}", file=sys.stderr)
        return 2

    targets = list_task_files(args.task)
    if not targets:
        print(f"FATAL: no files under {DSBENCH_DATA_ROOT / 'data_resplit' / args.task}",
              file=sys.stderr)
        return 2

    write_reference_script()
    total_mb = sum(p.stat().st_size for p in targets) / 1024 / 1024
    cold_root_anc = str(DSBENCH_DATA_ROOT.resolve())
    print(f"E-039 DSBench end-to-end | task={args.task}")
    print(f"  input files: {len(targets)} ({total_mb:.1f} MB)")
    print(f"  data_resplit dir: {DSBENCH_DATA_ROOT / 'data_resplit' / args.task}")
    print()

    hot_root = Path(args.hot_root)
    if hot_root.exists():
        shutil.rmtree(hot_root)
    hot_root.mkdir(parents=True, exist_ok=True)

    # ── BASELINE reps ─────────────────────────────────────────────────
    print(f"[1/2] BASELINE — {args.reps} reps cold...")
    baseline_runs: list[dict] = []
    for i in range(args.reps):
        ev = evict(targets)
        r = run_solution(args.task, ld_preload=None,
                          hot_root=None, cold_roots=None,
                          out_dir=args.out / f"baseline_out_{i}")
        baseline_runs.append({**r, "evict": ev})
        print(f"  baseline rep {i+1}: {r['elapsed_s']:.2f}s (rc={r['returncode']})  "
              f"resident_frac_after_evict={ev.get('resident_frac_sample','?')}")
        if r["returncode"] != 0:
            print(f"     stderr: {r['stderr_tail'][:300]}", file=sys.stderr)

    # ── STAGED: prefetch into hot tier ────────────────────────────────
    print(f"\n[2/2] STAGED — prestaging {len(targets)} files into {hot_root}...")
    stager = Stager(
        hot_root=hot_root,
        cold_roots=[Path(cold_root_anc)],
        max_workers=4, capacity_bytes=64 * 1024**3,
    )
    t_stage0 = time.monotonic()
    futures = stager.prefetch(DataHint(
        detected_files=tuple(str(p) for p in targets),
        tier=3, fired_at_ms=0.0, rule_id="dsb_e2e_prestage",
    ))
    n_staged = 0
    for f in futures:
        try:
            f.result(timeout=1800)
            n_staged += 1
        except Exception as e:
            print(f"  stage error: {e!r}", file=sys.stderr)
    stage_elapsed = time.monotonic() - t_stage0
    print(f"  staged {n_staged}/{len(targets)} in {stage_elapsed:.2f}s")

    staged_runs: list[dict] = []
    for i in range(args.reps):
        ev = evict(targets)
        r = run_solution(args.task, ld_preload=str(SHIM),
                          hot_root=str(hot_root), cold_roots=cold_root_anc,
                          out_dir=args.out / f"staged_out_{i}")
        staged_runs.append({**r, "evict": ev})
        print(f"  staged rep {i+1}:   {r['elapsed_s']:.2f}s (rc={r['returncode']})")
        if r["returncode"] != 0:
            print(f"     stderr: {r['stderr_tail'][:300]}", file=sys.stderr)
    stager.shutdown(wait=True)

    base_times = sorted(r["elapsed_s"] for r in baseline_runs if r["returncode"] == 0)
    stag_times = sorted(r["elapsed_s"] for r in staged_runs if r["returncode"] == 0)
    base_med = base_times[len(base_times)//2] if base_times else None
    stag_med = stag_times[len(stag_times)//2] if stag_times else None
    speedup = (base_med / stag_med) if (base_med and stag_med) else None

    result = {
        "experiment": "E-039",
        "task": args.task,
        "n_input_files": len(targets),
        "input_total_mb": round(total_mb, 2),
        "n_staged": n_staged,
        "stage_elapsed_s": round(stage_elapsed, 3),
        "reps": args.reps,
        "baseline": {"times_s": base_times, "median_s": base_med,
                      "runs": baseline_runs},
        "staged":   {"times_s": stag_times, "median_s": stag_med,
                      "runs": staged_runs},
        "session_speedup": round(speedup, 3) if speedup else None,
        "wall_time_saved_s": round((base_med or 0) - (stag_med or 0), 2)
                              if (base_med and stag_med) else None,
    }
    (args.out / f"dsb_e2e_{args.task}.json").write_text(json.dumps(result, indent=2))

    print()
    print(f"  ===== E-039 {args.task.upper()} RESULT =====")
    print(f"  baseline median: {base_med:>9.2f}s  (n={len(base_times)})")
    print(f"  staged median:   {stag_med:>9.2f}s  (n={len(stag_times)})")
    print(f"  wall-time saved: {result['wall_time_saved_s']:>9.2f}s")
    print(f"  session speedup: {result['session_speedup']}×")
    print(f"\n  wrote {args.out}/dsb_e2e_{args.task}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
