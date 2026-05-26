"""Throughput-only baseline for nyc-taxi: pure I/O measurement.

Streams labels.csv + test.csv through `read()` calls with NO per-row
parsing or Python loop. This measures the pure cold/hot delta on
this competition's bulk data, isolating I/O from any CPU work.

Defensibility: this is the same pattern dogs-vs-cats_thorough.py uses
(byte-through read of zip files). For nyc-taxi, the CSV is read with
identical I/O work to what any Arrow-backed columnar reader would do
on the bytes side; the difference is the CPU-side decode, which we
skip here to isolate the I/O contribution.

This isn't a useful ML solution on its own — it's a controlled
I/O-only measurement that bounds AgentStage's ceiling on this
competition's data. Pair with the polars 'thorough' result (which
adds the parse cost) to see how much of the I/O delta survives
into a real ML pipeline.

Writes a valid (constant) submission.
"""

from __future__ import annotations

import time
import pandas as pd
from pathlib import Path


TASK_DIR = Path("data/new-york-city-taxi-fare-prediction")

t0 = time.monotonic()
n_bytes = 0
with open(TASK_DIR / "labels.csv", "rb") as f:
    while chunk := f.read(4 << 20):
        n_bytes += len(chunk)
t_labels = time.monotonic() - t0
print(f"read labels.csv: {n_bytes/1024/1024/1024:.2f} GB  {t_labels:.2f}s  "
      f"({n_bytes/1024/1024/t_labels:.0f} MB/s)")

t1 = time.monotonic()
n_test_bytes = 0
with open(TASK_DIR / "test.csv", "rb") as f:
    while chunk := f.read(4 << 20):
        n_test_bytes += len(chunk)
t_test = time.monotonic() - t1
print(f"read test.csv:   {n_test_bytes/1024/1024:.1f} MB  {t_test:.2f}s")

t2 = time.monotonic()
sample = pd.read_csv(TASK_DIR / "sample_submission.csv")
t_sample = time.monotonic() - t2
print(f"loaded sample:   rows={len(sample)}  {t_sample:.2f}s")

sample["fare_amount"] = 11.36  # arbitrary constant — valid submission shape
sample.to_csv("submission.csv", index=False)
print(f"wrote submission.csv  rows={len(sample)}")
print(f"TOTAL_IO_TIME: {t_labels + t_test + t_sample:.3f}s")
