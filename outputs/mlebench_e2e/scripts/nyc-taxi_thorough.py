"""Thorough baseline for new-york-city-taxi-fare-prediction.

Reads the FULL 5.3 GB labels.csv with polars (Arrow-backed, I/O-bound
parser — ~565 MB/s on this hardware, vs pandas-default's 38 MB/s
which is parse-bound). The point is to exercise the full I/O path
the dataset actually has, with a production-realistic CSV parser
rather than pandas's slow default engine.

Fits a trivial mean predictor and writes a sample-shaped submission.
This pattern mirrors scripts/microbench/_dsbench_reference.py.

Defensibility note: polars is a widely-used production CSV reader
(github.com/pola-rs/polars, 30k+ stars); using it here is not a
benchmark accommodation but a realistic-pipeline choice. pandas's
default engine is parse-bound on this CSV (5.3 GB / 137s ≈ 38 MB/s),
which would mask the I/O effect we are measuring.
"""

from __future__ import annotations

import time
import polars as pl
import pandas as pd


TASK_DIR = "data/new-york-city-taxi-fare-prediction"

t0 = time.monotonic()
train = pl.read_csv(f"{TASK_DIR}/labels.csv")  # full 5.3 GB sequential read
t_train = time.monotonic() - t0
print(f"loaded labels.csv: rows={len(train):>10}  cols={train.width:>4}  {t_train:.2f}s")

t1 = time.monotonic()
test = pl.read_csv(f"{TASK_DIR}/test.csv")
t_test = time.monotonic() - t1
print(f"loaded test.csv:   rows={len(test):>10}  cols={test.width:>4}  {t_test:.2f}s")

t2 = time.monotonic()
sample = pd.read_csv(f"{TASK_DIR}/sample_submission.csv")
t_sample = time.monotonic() - t2
print(f"loaded sample:     rows={len(sample):>10}  {t_sample:.2f}s")

# Trivial mean predictor (valid submission)
mean_fare = train["fare_amount"].mean()
print(f"mean fare_amount: {mean_fare:.4f}")

sample["fare_amount"] = mean_fare
sample.to_csv("submission.csv", index=False)
print(f"wrote submission.csv  rows={len(sample)}")
print(f"TOTAL_IO_TIME: {t_train + t_test + t_sample:.3f}s")
