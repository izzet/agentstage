# AgentStage

AgentStage stages scientific data during an LLM agent's *thinking phase*, the
interval where the model plans its next tool call and the storage layer sits
idle. It reads the model's streaming reasoning and the agent's own filesystem
probes to infer which files the next tool will open, then copies them from cold
storage into a local hot tier before the tool fires, so the reads land hot
instead of cold.

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
| Staging daemon | Thread pool that copies cold-tier files into the hot tier, expanding compressed sources and pulling in companion indices. Each file is published atomically (write to a temporary path, then `rename`) so a reader never sees a partial one |
| `LD_PRELOAD` shim | Small C library that redirects path-taking syscalls under managed cold roots to the staged hot copy. Writes pass through, leaving the hot tier read-only from the agent's view |

## Layout

| Path | Purpose |
|---|---|
| `src/agentstage/` | The package: `client/` (capture proxy), `detector/`, `stager/` (daemon + C shim), `workloads/`, `runners/`, `metrics/` |
| `tests/` | Unit tests |
| `paper_evals/` | Claim-verification suite, run separately from `tests/` |
| `paper_figures/` | Figure builders plus version-controlled data snapshots |
| `outputs/` | Committed run artifacts that the eval suite and figure builders read |
| `scripts/microbench/` | Measurement and analysis scripts behind the paper's numbers |
| `scripts/BENCH_TIERS.md` | Storage-tier bandwidth measurements (paper §IV.A) |
| `external/` | Benchmark and tracing submodules |

## Requirements

- Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/)
- Linux with glibc, for the `LD_PRELOAD` shim
- A cold tier (parallel filesystem, network export, or local disk) and a hot
  tier (tmpfs or local NVMe)

## Install

```bash
git clone https://github.com/izzet/agentstage && cd agentstage
uv sync
```

The package installs and the test suite runs from a bare clone, with no
submodules checked out. Fetch the benchmark submodules only if you want to
run the community workloads:

```bash
git submodule update --init external/benchmarks/mle-bench \
    external/benchmarks/kramabench external/benchmarks/dsbench
```

The three curated tasks additionally need AgentIOBench, a sibling project.
It is intentionally not a dependency, so its absence costs only those tasks:

```bash
uv pip install -e external/benchmarks/agentiobench   # once available
```

Build the shim:

```bash
make -C src/agentstage/stager/shim
```

Run the tests:

```bash
uv run pytest              # unit tests
uv run pytest paper_evals/ # claim-verification suite; tests whose input
                           # artifacts are not committed will skip
```

## Usage

AgentStage has two halves: a drop-in client that detects and stages during the
thinking phase, and an `LD_PRELOAD` shim that redirects the tool's reads to the
staged copy.

```python
import os
from agentstage.client.anthropic import AnthropicClient
from agentstage.detector.auto_rules import AutoRuleGenerator
from agentstage.stager import Stager

# What exists in the workspace, grouped into the buckets rules match against.
workspace_prior = {
    "sample_HG00096": ("/cold/igsr/HG00096/aln.bam", "/cold/igsr/HG00096/aln.bai"),
    "reference": ("/cold/igsr/ref/human_g1k_v37.fasta.fai",),
}

# Mine a rule library from workload metadata. No hand-written rules.
ruleset = AutoRuleGenerator(
    workload_id="igsr-cov-qc",
    task_instruction="Compute an xxh64 manifest for every BAM under each sample.",
    workspace_prior_keys=tuple(workspace_prior),
).generate()

# Background thread pool that copies cold to hot.
stager = Stager(hot_root="/dev/shm/agentstage", cold_roots=["/cold"])

# Drop-in replacement for the vendor client: same call, plus staging.
client = AnthropicClient(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    stager=stager,
    workspace_prior=workspace_prior,
    ruleset=ruleset,
)

response = client.stream(
    model="claude-haiku-4-5",
    messages=[{"role": "user", "content": "..."}],
)
for event in response:
    ...  # forwarded unchanged; the detector fires on each thinking delta
```

`GeminiClient` (`agentstage.client.gemini`) and `OpenAIClient`
(`agentstage.client.openai`) take the same arguments. `OpenAIClient` points
at any OpenAI-compatible endpoint via `base_url` and stages when the server
surfaces reasoning text on `delta.reasoning` or `delta.reasoning_content`, as
vLLM and DeepSeek do. OpenAI's own API returns no reasoning tokens, so it
streams normally but stages nothing.

Run the agent's tool subprocesses under the shim so their `open()` calls land
on the staged copies:

```bash
LD_PRELOAD=src/agentstage/stager/shim/libagentstage_shim.so \
AGENTSTAGE_HOT_ROOT=/dev/shm/agentstage \
AGENTSTAGE_COLD_ROOTS=/cold \
  python analysis.py
```

### Compressed inputs and companion indices

Staging is keyed to **the path the tool opens**, not to the file sitting on
disk. If the tool opens `x.csv` and the cold tier holds only `x.csv.gz`, the
daemon expands it into the hot tier at `x.csv`. If the tool opens `x.csv.gz`
itself, it gets a plain copy and does its own decompression. The hot copy is
therefore always byte-identical to what the tool expects, which is why the
shim needs no knowledge of compression at all.

`.gz`, `.bz2`, and `.xz` expand with the standard library. `.zst` needs an
optional extra; without it, sources the stager cannot decode are skipped
rather than failing:

```bash
uv sync --extra codecs
```

Formats with an external index (BAM/BAI, CRAM/CRAI, VCF/TBI, FASTA/FAI) also
pull their companion file in. Staging a BAM without its index leaves the tool
doing random access against a cold index, which is the read pattern staging
exists to remove.

Archives are not expanded: one `.tar.gz` becoming many files breaks the
one-target-one-file model the capacity accounting and the shim's per-file
mapping rely on.

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
