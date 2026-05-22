# Foundations — Why S3, Why These Benchmarks, What "Speedup" Means

> Records the design rationale a reviewer will ask about. Companion to
> [`IO_LEAKAGE_AUDIT.md`](IO_LEAKAGE_AUDIT.md) (prompt-bias audit) and
> [`EXPERIMENTS.md`](EXPERIMENTS.md) (chronological measurement log).
>
> Written 2026-05-22 to capture the "back why we used S3" discussion.

---

## 1. Why an S3-backed cold tier?

### 1.1 The honest status

AgentIOBench's own datasets are **not** on S3 — they ship on
local/NFS storage. The `aiob_107_s3` task variant is **our addition**
(upstream commit `dea5686` on the `feat/agentstage-integration`
branch, made during this work). So the honest answer to "did the
benchmark use S3?" is: **no, we added the S3 backing ourselves.**

The same is true of the other agent benchmarks we considered:
- **ScienceAgentBench** ships datasets locally with the harness.
- **KramaBench** distributes its data via HuggingFace `datasets`;
  it frames tasks as "data-to-insight pipelines over data lakes"
  but the reference data is downloaded locally for evaluation.

So **S3 is not how published agent benchmarks fetch data.** They
ship data locally for reproducibility and evaluation convenience.

### 1.2 Why we use it anyway — and why that is defensible

The justification is **not** "benchmarks use S3." It is: *the
real-world scientific workflows these benchmarks emulate increasingly
read from cloud object storage, and that is the deployment scenario
where data staging matters most.* Concretely:

1. **The exact dataset we use is genuinely on S3.** The GOES-16 ABI
   imagery for `aiob_107` is published in the public S3 bucket
   `noaa-goes16` (us-east-1), part of the **NOAA Open Data
   Dissemination (NODD)** program. We point AWS's `mountpoint-s3`
   FUSE driver at the real bucket — no synthesis, no mock. NODD's
   own language: *"NOAA makes data openly available to ensure
   maximum use of our data."* GOES-16/17/18/19 are all in NODD S3
   buckets accessible with `aws s3 --no-sign-request`.

2. **Cloud object storage is a primary locus of scientific data.**
   The **AWS Open Data program** (and its Open Data Sponsorship
   Program, which "covers storage costs for publicly available
   high-value cloud-optimized datasets") hosts climate, genomics,
   satellite, and geospatial datasets. NASA (Earthdata on S3), USGS,
   and the Pangeo/Zarr community publish analysis-ready data on S3.
   A scientific agent deployed in a cloud environment reads from
   object storage, not a local NVMe.

3. **`mountpoint-s3` is the established access path.** It is AWS's
   official, generally-available FUSE client for exposing S3 as a
   POSIX filesystem — exactly the integration surface an LD_PRELOAD
   shim or a staging daemon would sit in front of. Reading S3 through
   a FUSE mount is a real, supported deployment, not a contrivance.

4. **The cold/hot latency gap is what makes staging a research
   problem at all.** Local NVMe first-byte latency is ~0.5 ms; an S3
   object's first byte through `mountpoint-s3` is ~750 ms (measured,
   E-010). A staging system is only interesting when the cold tier
   is genuinely slow. S3 is the honest "slow cold tier" of a cloud
   scientific deployment; local NFS is the "fast cold tier" of an
   on-prem HPC cluster. **We report both** (see §3).

### 1.3 How to state this in the paper

> "We evaluate two cold-tier configurations. The *on-prem* tier is
> local NFS/XFS storage, matching how AgentIOBench, ScienceAgentBench,
> and KramaBench ship their reference data. The *cloud* tier is
> Amazon S3, accessed via AWS's `mountpoint-s3` FUSE client against
> the public `noaa-goes16` bucket (NOAA Open Data Dissemination
> program). The S3 configuration is our addition — published agent
> benchmarks ship data locally for evaluation convenience — but it
> reflects the deployment reality of cloud-hosted scientific
> computing, where analysis-ready datasets (NOAA, NASA Earthdata,
> AWS Open Data) live in object storage. Data staging is most
> valuable precisely when the cold tier is slow, so the S3
> configuration is the stress test and the NFS configuration is the
> conservative baseline."

This is honest: it does not claim S3 is the benchmark norm, it
explains why S3 is the *realistic deployment* the benchmark norm
abstracts away.

---

## 2. Why these benchmarks (AIOB / SAB / KramaBench)?

- **AgentIOBench (AIOB)** — purpose-built for I/O-aware scientific
  agents. 12 tasks across climate, genomics, meteorology,
  neuroscience, etc. The task prompts describe dataset structure;
  the agent writes + executes analysis code. This is the primary
  benchmark because it is *about* the I/O behavior our system
  targets.
- **ScienceAgentBench (SAB)** — published (Chen et al., ICLR 2025),
  used as an external-generality probe. Workloads are small
  (1-3 files); not a staging stress test, but confirms the rule
  library transfers.
- **KramaBench** — recent (Lai et al., 2025; MIT DB Lab), framed as
  "data-to-insight pipelines over data lakes." Used as a second
  external-generality probe and as the "naturally sparse prompt"
  regime (it does not leak folder trees / counts the way AIOB does
  — see [`IO_LEAKAGE_AUDIT.md`](IO_LEAKAGE_AUDIT.md) §3).

Triangulating one purpose-built benchmark (AIOB) with two published
ones (SAB, KramaBench) is the standard defense against "you built
your own benchmark to look good."

---

## 3. What "speedup" means — three units, do not conflate them

A recurring confusion. AgentStage produces speedup numbers in three
different units. They are all real; they answer different questions.

### 3.1 Per-file (read-latency) speedup — ~10⁴×

The latency of a single `open()+read()` on a file: cold-tier (S3
through `mountpoint-s3`) vs hot-tier (tmpfs via the LD_PRELOAD shim).
Measured live, in-process: ~754 ms cold → ~0.05 ms hot ≈ **10⁴×**
(E-010, E-021, E-023, E-026).

This is the **mechanism demonstration**. It proves the shim
intercepts, the stager populates, the read is served from RAM. It is
*not* a claim about how much faster an agent's task finishes.

### 3.2 Per-session (wall-time) speedup — workload-dependent

The fraction of an agent's *total session wall-time* that is POSIX
I/O, which staging can eliminate. From 30 real AIOB production runs
(E-027):

| Workload | I/O fraction (local NFS) | Session speedup if I/O eliminated |
|---|---:|---:|
| aiob_104 (genomics) | 1.4% | 1.01× |
| aiob_107 (meteorology) | 30.6% | 2.08× |
| aiob_110 (neuroscience) | 17.1% | 1.30× |

Most of an agent session is LLM inference and compute, which staging
does **not** touch. The per-session speedup is therefore much smaller
than the per-file number, and varies by how I/O-bound the workload is.

### 3.3 Per-session speedup projected to S3 — 1.3× to 24×

The E-027 numbers above are measured on local NFS, which is already
fast. Projected to an S3 cold tier (latency ratio from E-010), the
I/O fraction grows and the session speedup rises to **1.3×–24×**
depending on workload.

### 3.4 The rule for the paper

- Report **per-file 10⁴×** as the *mechanism* result.
- Report **per-session 1.0–2.1× (NFS) / 1.3–24× (S3)** as the
  *user-visible* result.
- **Never** report 10⁴× as if it were the session-level speedup.
  That is the single most likely reviewer trap.

The end-to-end experiment (E-028, in progress) measures the
per-session number *directly* by running the agent's actual
generated analysis script with vs without staging, on both cold
tiers — closing the gap between the measured-NFS and projected-S3
numbers with a real side-by-side.

---

## 4. Conversation record (2026-05-21 → 2026-05-22)

Condensed log of the design decisions made in discussion, so the
reasoning is not lost:

1. **Per-file vs per-session** — flagged that the 10⁴× per-file
   number is not a session-level claim. Smoke runs showed ~1%
   session speedup because they are exploration-heavy (1 file read
   in 8 turns). Resolved by E-025 (decomposition) + E-027 (real
   production-run analysis).

2. **Real production runs** — pointed at
   `/mnt/common/datasets-staging/agentiobench/outputs/`. The
   `replay.yaml` + `io_report.json` files show real agent I/O
   behavior: the agent does ~11 discovery turns then writes + runs
   one Python script that performs all 6,000+ file reads in a single
   677 s execution turn. This is the I/O pattern staging must serve.

3. **S3 as baseline** — questioned whether S3-backed access is
   established. Answer: not for the *benchmarks* (they ship data
   locally), but yes for the *real scientific workflows* they
   emulate. Documented in §1 above.

4. **End-to-end demo** — agreed we must show a real speedup from
   running the agent's actual generated script with vs without
   staging, on both local NFS and S3. That is E-028.
