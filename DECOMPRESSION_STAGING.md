# Decompression-Staging — Design Brainstorm

> Extends "staging" from data *movement* to data *movement + preparation*.
> The hot tier can hold data in a cheaper-to-consume representation
> (uncompressed), not just at a faster-to-reach location.
>
> Brainstormed 2026-05-22. Experiment: E-029.

---

## 1. The observation

E-028 (end-to-end, local tier) measured: baseline 110.3 s, plain-staged
97.7 s — only 1.13×. The staged run is still ~98 s because the agent's
generated script spends most of its time **not** on cold-tier read
latency (which staging eliminates) but on **decompression CPU**, which
plain staging does not touch.

The agent's script (`outputs/e2e/task_script.py`, lines 117-119):

```python
with nc.Dataset(filepath, 'r') as ds:
    cmi = ds.variables['CMI'][:]   # reads the ENTIRE 1500x2500 grid
    dqf = ds.variables['DQF'][:]   # reads the ENTIRE grid
```

It reads the **whole grid** then slices 10x10 boxes in numpy. Every
file: all ~120 zlib chunks decompressed, even though ~5 are needed.
The AIOB task's own knowledge hint warns against this; this Sonnet
agent ignored it. So decompression is the dominant cost.

## 2. The reframe

Current definition — staging = **data movement**: cold → hot,
byte-identical.

Proposed — staging = **data movement + data preparation**: the hot
tier holds the data in whatever representation is cheapest for the
agent's compute to consume. Decompressed-in-RAM beats
compressed-on-disk for a consumer that will decompress anyway.

The slack-window principle is unchanged. AgentStage already moves
**read latency** off the critical path (turn-12 script execution) into
the staging window (turns 0-11 + racing ahead). Decompression-staging
moves **decompression CPU** off the critical path the same way.

## 3. The math

Per aiob_107 GOES file:
- Compressed on disk: ~2.8 MB
- Uncompressed (CMI 1500x2500 float32 ~15 MB + DQF ~4 MB + coords): **~19 MB**
- zlib decompress ~250 MB/s/core -> **~76 ms/file** of pure decompression

| Scope | Files | Decompression CPU (currently on critical path) |
|---|---:|---:|
| day 122 (E-028/E-029 scope) | 864 | **~66 s** |
| full aiob_107 task | 6,042 | **~460 s** (7.5 min) |

E-028 staged run (97.7 s) decomposes roughly as:
~8 s hot-tier I/O + **~66 s decompression** + ~24 s other compute.

If decompression moves into staging (hot tier holds uncompressed .nc):
- Staged run -> ~3 s tmpfs I/O (bigger files) + ~24 s compute ~= **~30-35 s**
- vs baseline 110 s -> projected **~3.2x on local NFS** (was 1.13x)
- On S3 the baseline is far higher, so the combined ratio compounds.

## 4. Architecture

Minimal. The shim does not change at all.

```
DataHint  + transform: none | decompress | recodec:lz4
                │
                ▼
Stager._stage(cold, hint):
  transform == none       ->  shutil.copy(cold -> hot)        [today]
  transform == decompress ->  nc.Dataset(cold) ->
                              nc.Dataset(hot, compression=None)
  (atomic rename either way)
                │
                ▼
hot tier holds an UNCOMPRESSED .nc (same data, no zlib filter)
                │
                ▼
shim redirects open() -> hot copy   [UNCHANGED — just rewrites the
                                     path; the bytes differ but the
                                     DATA the agent reads is identical]
```

Who sets `transform`: the **detector**. It already classifies file
semantic classes; it would additionally know "this workload's files
are zlib-compressed NetCDF" from workspace-prior metadata or the
task's chunking hint (which the leakage audit already parses).
Compressed scientific files -> `transform=decompress`.

### Transform options, increasing aggressiveness

1. **`decompress`** — rewrite NetCDF with `compression=None`.
   Transparent; `ds.variables['CMI'][:]` returns the same array, HDF5
   reads uncompressed chunks -> zero zlib. **Recommended.**
2. **`recodec:lz4`/`zstd`** — recompress with a codec that decompresses
   5-10x faster than zlib. Smaller hot footprint; needs the HDF5 codec
   plugin in the agent's runtime.
3. **`region-extract`** — detector sees grid coords in the prompt
   ("Houston 457,690"); stager pre-extracts just the 10x10 boxes. Tiny
   hot footprint, but task-specific and fragile.

## 5. Costs (honest)

| Cost | Detail |
|---|---|
| Hot-tier capacity | Uncompressed ~7x bigger (19 MB vs 2.8 MB). day-122 = ~16 GB (fits 32 GB tmpfs). Full task ~121 GB — does NOT fit; needs LRU stage-ahead/evict-behind or a warm NVMe tier. |
| Transcode CPU | Stager workers decompress instead of memcpy. ~460 s CPU for the full task — parallel across workers (~60 s wall), overlapped with agent turns + racing ahead. Off the critical path. |
| Byte-identity broken | Hot file != cold file byte-for-byte (data-identical, not byte-identical). Shim doesn't care; checksumming consumers would. Note it. |
| Codec availability | `decompress` needs nothing extra; `recodec` needs the HDF5 plugin. |

## 6. Honest scope of the benefit

Decompression-staging's benefit is **largest for I/O-naive agent
scripts** (full-grid reads, like this Sonnet one) and **smallest for
I/O-efficient ones** (chunk-aligned slicing — which decompress only the
needed chunks anyway). Since AIOB explicitly grades I/O efficiency, the
honest framing is: this is a "rescues the naive agent" optimization.
Report the benefit conditioned on the agent script's read pattern.

## 7. Experiment — E-029

`scripts/microbench/path_b_e2e_decompress.py`:
- BASELINE: cold tier, no shim (= E-028 baseline)
- DECOMP-STAGED: transcode all 864 files to uncompressed via
  `outputs/e2e/transcode.py`, run the agent script with the shim
- Both tiers (local + S3)
- Three-way comparison vs E-028's plain-staged number

Result recorded in [`EXPERIMENTS.md`](EXPERIMENTS.md).
