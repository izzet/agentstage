# AgentStage

Thinking-phase data staging for scientific LLM agents.

An LLM agent alternates between a *thinking phase*, where it plans the next
tool call, and a *tool phase*, where that tool reads data. The thinking phase
is idle storage time. AgentStage reads the model's streaming reasoning and the
agent's own filesystem probes to infer which files the next tool will open, and
stages them from cold storage into a local hot tier before the tool fires, so
the reads land hot instead of cold.

## Paper

> Izzet Yildirim, Xian-He Sun, Anthony Kougkas.
> **AgentStage: Exploiting LLM Thinking for Data Staging in Scientific Agents.**
> IEEE International Conference on e-Science (eScience'26), Naples, Italy,
> 28 September – 2 October 2026.

Across three curated I/O-bearing scientific tasks plus one data-intensive task
from each of MLE-bench, KramaBench, and DSBench, over four reasoning models,
AgentStage delivers an average **1.74× end-to-end session speedup** on the
curated tasks (up to 2.34× per run) with no per-model or per-task tuning.

Use [`CITATION.cff`](CITATION.cff) to cite this work.

## How it works

| Component | Role |
|---|---|
| Capture proxy | In-process SDK wrapper. Forwards each stream event unchanged while firing the detector over the accumulated thinking text |
| Auto-rule generator | Mines each workload's metadata into a per-workload regex rule library, with no hand-written rules |
| Tiered detector | Matches rules against the streaming reasoning, refines against filesystem-probe results, and classifies each rule by target-set size into eager, opportunistic, and on-demand tiers |
| Staging daemon | Thread pool that copies cold-tier files into the hot tier, publishing each atomically (copy to a temporary path, then `rename`) so a reader never sees a partial file |
| `LD_PRELOAD` shim | Small C library that redirects path-taking syscalls under managed cold roots to the staged hot copy. Writes pass through, leaving the hot tier read-only from the agent's view |

## Layout

| Path | Purpose |
|---|---|
| `src/agentstage/` | The package: `client/` (capture proxy), `detector/`, `stager/` (daemon + C shim), `workloads/`, `runners/`, `metrics/` |
| `tests/` | Unit tests |
| `paper_evals/` | Claim-verification suite, run separately from `tests/` |
| `paper_figures/` | Figure builders plus version-controlled data snapshots |
| `scripts/microbench/` | Measurement and analysis scripts behind the paper's numbers |
| `scripts/BENCH_TIERS.md` | Storage-tier bandwidth measurements (paper §IV.A) |
| `PREDICTION_ANALYSIS.md` | I/O-share and timeliness analysis (paper §IV.F) |
| `external/` | Benchmark and tracing submodules |

## Requirements

- Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/)
- Linux with glibc, for the `LD_PRELOAD` shim
- A cold tier (parallel filesystem, network export, or local disk) and a hot
  tier (tmpfs or local NVMe)

## Install

```bash
git clone https://github.com/izzet/agentstage && cd agentstage
git submodule update --init --recursive
uv sync
```

> **Known limitation.** The curated workload definitions in
> `src/agentstage/workloads/aiob.py` depend on AgentIOBench, an unpublished
> sibling project wired in as a private submodule and a uv workspace member.
> Without access to that repository, `git submodule update --init --recursive`
> and `uv sync` will not complete. The detector, staging daemon, and shim do
> not otherwise depend on it, and the community-benchmark workloads
> (MLE-bench, KramaBench, DSBench) use their own public submodules.

Build the shim:

```bash
make -C src/agentstage/stager/shim
```

Run the tests:

```bash
uv run pytest              # unit tests
uv run pytest paper_evals/ # claim-verification suite
```

## Configuration

Copy `.env.example` to `.env` and fill in provider credentials and data roots.
The shim is configured entirely through the environment:

| Variable | Meaning |
|---|---|
| `AGENTSTAGE_HOT_ROOT` | Hot-tier mount (required) |
| `AGENTSTAGE_COLD_ROOTS` | Colon-separated cold roots (required) |
| `AGENTSTAGE_HOT_OVERFLOW` | Overflow tier used when the primary hot tier fills |
| `AGENTSTAGE_RETRY_SPIN_MS` | How long to wait on an in-flight stage (default 20) |
| `AGENTSTAGE_SHIM_LOG` | Write per-call events to this path |
| `AGENTSTAGE_SHIM_DISABLE` | Set to `1` to pass every call through unchanged |

Integrating AgentStage into an agent harness means replacing the provider's
streaming client with the drop-in equivalent in `src/agentstage/client/` and
exporting `LD_PRELOAD` for the tool subprocesses.

## License

MIT. See [`LICENSE`](LICENSE).
