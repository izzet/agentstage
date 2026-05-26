"""Thorough baseline for dogs-vs-cats-redux-kernels-edition.

The agent solutions in MLE-bench read individual images from the
unpacked train/ and test/ directories. Our workload loader skips
those for staging (zip-only policy when both exist). So a 'thorough'
baseline that exercises the STAGED files should read the zips
directly — a realistic ML preprocessing step in many production
pipelines.

This script:
  1. Fully reads train.zip + test.zip (bytes through, no extract).
     This is real I/O the cold/hot tier delta covers.
  2. Loads sample_submission.csv to know the test IDs.
  3. Writes a constant submission.
"""

from __future__ import annotations

import time
import pandas as pd
from pathlib import Path


TASK_DIR = Path("data/dogs-vs-cats-redux-kernels-edition")

# Full byte read of train.zip — exercises the cold->hot delta on the
# 490 MB compressed image archive (the file the workload loader staged).
t0 = time.monotonic()
n_train_bytes = 0
with open(TASK_DIR / "train.zip", "rb") as f:
    while chunk := f.read(1 << 20):
        n_train_bytes += len(chunk)
t_train = time.monotonic() - t0
print(f"read train.zip: {n_train_bytes/1024/1024:.1f} MB  {t_train:.2f}s")

t1 = time.monotonic()
n_test_bytes = 0
with open(TASK_DIR / "test.zip", "rb") as f:
    while chunk := f.read(1 << 20):
        n_test_bytes += len(chunk)
t_test = time.monotonic() - t1
print(f"read test.zip:  {n_test_bytes/1024/1024:.1f} MB  {t_test:.2f}s")

t2 = time.monotonic()
sample = pd.read_csv(TASK_DIR / "sample_submission.csv")
t_sample = time.monotonic() - t2
print(f"loaded sample:  rows={len(sample)}  {t_sample:.2f}s")

# Trivial constant predictor (valid submission format)
sample["label"] = 0.5
sample.to_csv("submission.csv", index=False)
print(f"wrote submission.csv  rows={len(sample)}")
print(f"TOTAL_IO_TIME: {t_train + t_test + t_sample:.3f}s")
