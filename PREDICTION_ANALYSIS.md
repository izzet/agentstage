# Speedup / I-O Prediction Analysis (true-cold OrangeFS, 72 cells)

Date: 2026-06-07. Data: `outputs/replay/_ofs_full/` (72/72 cells, true-cold OrangeFS,
3 reps x 4 models x {3 curated + 3 community tasks}). Calibration: `io_calib.json`.

## TL;DR
- **I/O cost is a-priori predictable: R² = 0.998.** A calibrated `bytes/BW_seq + n_files·t_open`
  model predicts cold-tier read time across a 10–425 MB/s effective-bandwidth span.
- **Session/tool speedup is NOT impressively a-priori predictable: R² ≈ 0.57 ceiling**
  (learned, leave-one-out; same for linear and random forest, with/without model identity).
  The residual is trajectory-dependent (compute + thinking time) and system effects
  (shim per-open overhead, prefetch↔thinking overlap), none knowable from data layout.
- **Speedup IS tightly explained post-hoc by an Amdahl decomposition using the MEASURED
  shell speedup: R² = 0.994** — but that is descriptive, not a-priori.

## 1. Calibrated first-principles I/O cost model  (the strong predictive result)
`cold_io_time = total_bytes / BW_seq + n_files · t_open`, calibrated once on the storage tier.

Calibration on 6 workloads (clean true-cold full-dataset reads):
```
BW_seq = 340 MB/s,  t_open = 4.48 ms/file        calibration R² = 0.998
  igsr   meas 30.7s pred 32.2s     dogs meas 115.5s pred 115.3s
  jwst   meas 32.2s pred 33.7s     kb   meas   5.9s pred   8.4s
  sentinel 46.0s    pred 43.5s     tab  meas   5.5s pred   6.9s
```
Effective BW per workload spanned 10 MB/s (dogs, 25k files, metadata-bound) to
425 MB/s (tabular, 3 contiguous files). The single-BW model failed (R²=0.42) because it
lacked the metadata term; adding the static `n_files` term fixes it. Parallel (4-worker)
prefetch BW = 1179 MB/s vs 348 single-stream → prefetch stages ~3.4x faster than the agent
reads cold, which is why bulk reads are fully hidden inside the thinking window.

This is genuinely a-priori (tier calibration + static data layout) and generalizes.

## 2. Why a-priori SPEEDUP prediction caps at ~0.57
session_sp = (thinking + compute + io_cold) / (thinking + compute + io_hot).
Only `io_cold` is predictable from data. `thinking` and `compute` depend on the agent's
trajectory (which model, what it decides to do), not the data layout.

Tested predictors (n=70, leave-one-out where learned):
| model | target | R² |
|---|---|---|
| `1/(1 - io_pred/cold_baseline)` (hide-all-I/O physics) | session sp | **−41** (overpredicts) |
| `1/(1 - io_pred/shell_baseline)` | tool sp | −40 |
| learned linear `[log bytes, log nfiles, io_pred]` | session sp | **0.54** |
| learned linear `+ mean_file_size` | session sp | 0.54 |
| learned linear `+ model one-hot` | session sp | 0.55 |
| random forest, all a-priori feats | session sp | **0.58** |

The physics "hide-all-I/O" model overpredicts badly because staging does NOT eliminate the
full single-stream cold read:
- metadata-bound (dogs): full prefetch to hot, yet staged shell only drops 170→130s
  (~40s saved of ~115s I/O). The residual is the **shim's per-open overhead** (25k intercepted
  opens) + hot-tier metadata, not cold I/O. → io_hidden ≪ io_cold.
- compute-bound (tabular): io_cold ~6s of a ~300s LightGBM session → sp ~1.0 (correct).

Also tried a shim-overhead-corrected physical model `io_hidden = io_cold_pred - k·n_files`
(cold runs with shim disabled, staged runs pay per-open shim cost): best-fit k=0, R²=−9,
LOO R²=−74. The `cold/(cold-io_hidden)` form is too sensitive and the per-cell hidden I/O
does not track any static feature combination. Confirms ~0.57 is the a-priori ceiling.

## 3. Amdahl decomposition (descriptive, post-hoc) — R² = 0.994
`session_sp = 1 / (1 - shell_share·(1 - 1/r))`, r = MEASURED shell speedup.
n=66 (excludes 6 no-speedup cells with r≤1). R²=0.994, Pearson 0.999.
NOTE: replaces the paper's current claim "predicts at R²=0.901", which was (a) a mislabeled
Pearson r² and (b) computed from a double-counted ceiling. Uses measured r, so NOT a-priori.

## 4. Final corrected true-cold speedups (for reference)
Curated mean **1.74** (igsr 1.58 / jwst 1.79 / cross 1.84), per-cell max 2.34.
Community: MLE dogs **1.36**, KB **1.20**, DSB tabular **1.03** (compute-bound).

## 5. Recommendation for the paper
The empirical speedup (1.74x curated) is the contribution; it is demonstrated, not predicted.
For the prediction angle, the honest + strong framing is two parts:
1. **A-priori I/O cost model (R²=0.998)** — predicts the I/O staging targets; motivates the system.
2. **Amdahl decomposition (R²=0.994)** — explains how that I/O benefit becomes session speedup.
Do NOT claim a-priori session-speedup prediction (ceiling ~0.57; trajectory-dominated).
Reframe C4 from "predicts speedup at R²=0.901" → "predicts I/O cost (0.998) and decomposes
speedup (0.994)."

OPEN: if a-priori *speedup* prediction is required, it needs a model of (a) the non-I/O
session time and (b) the prefetch↔thinking overlap + shim per-open overhead — both
trajectory/system dependent. Not achievable from data layout alone at useful accuracy.
