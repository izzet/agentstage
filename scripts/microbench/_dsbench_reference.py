#!/usr/bin/env python3
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
