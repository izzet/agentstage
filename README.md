# AgentStage

Streaming-intent-driven tiered data staging for scientific LLM agents.
Uses an LLM's own streaming thinking content as an early signal of which
files the agent will read next, and stages those files from cold storage
(PFS / object store / NFS) into local NVMe during the wall-clock slack
between "first thinking chunk" and "first tool dispatch" — moving only
data, not pre-executing the tool.

**Authoritative research doc:** [`AGENTSTAGE.md`](AGENTSTAGE.md).
Read it first. Everything else here is operational scaffolding.

## Where things live

| Path | Purpose |
|---|---|
| `AGENTSTAGE.md` | Research notes, hypotheses (H1-H5), claims (C1-C10), evaluations (E1-E11), execution plan (§11.8) |
| `CAMPAIGN.md` | Experimental campaign matrix, model choice, cost ceiling |
| `TASKS.md` | Persistent task list across sessions |
| `src/agentstage/` | uv-managed src-layout package (Python ≥3.12) |
| `tests/` | Unit tests over `src/agentstage/`. Run with `uv run pytest`. |
| `paper_evals/` | Claim-verification suite — one file per hypothesis H1-H10. Run with `uv run pytest paper_evals/ --trace-root <path>`. See `paper_evals/README.md`. |
| `external/benchmarks/` | Git submodules: ScienceAgentBench (pinned `72220ee8`), SWE-bench |
| `external/datasets/` | Gitignored. Populated by `scripts/fetch_datasets.sh`. |
| `poc/` | Gitignored. Trace-only PoC script + 88 captured probe runs (referenced by `AGENTSTAGE.md` §6). |
| `paper/` | Gitignored. Working directory for paper drafts. |
| `results/` | Gitignored. Experiment outputs. |

## Setup

```bash
# 1. Clone with submodules
git clone <repo> agentstage && cd agentstage
git submodule update --init --recursive

# 2. Install uv (if not already present)
# uv lives at ~/.local/bin/uv on Ares; install per https://astral.sh/uv if elsewhere

# 3. Sync the project venv
~/.local/bin/uv sync

# 4. Copy + populate the env file
cp .env.example .env
# Edit .env to set:
#   AZURE_FOUNDRY_KEY, AZURE_FOUNDRY_ANTHROPIC_URL  (Claude Haiku 4.5)
#   GOOGLE_GEMINI_API_KEY                          (Gemini 2.5 Flash)
#   OSS_MODEL_BASE_URL                             (self-hosted Qwen/DeepSeek vLLM)
#   AGENTIOBENCH_DATA_ROOT                         (workload data path)

# 5. Verify
~/.local/bin/uv run pytest                          # unit tests
~/.local/bin/uv run pytest paper_evals/ --trace-root poc/runs   # claim-verification stubs
```

## Hard rules

- **Never write into `/home/iyildirim/projects/sciiobench/`.** That's the
  adjacent AgentIOBench project; treat as read-only. Go there for `.env`
  reference, prior `io_report.json` files, or workload definitions, but
  do not modify or commit there.
- **Never add `Co-Authored-By: Claude ...` to commits or PRs.** No AI
  attribution trailers, no `🤖 Generated with ...` markers.
- **Frozen rules cannot be quietly updated.** Bumping
  `RULE_LIBRARY_VERSION` in `src/agentstage/predictor/__init__.py` is a
  deliberate, commit-message-documented event. The version hash is pinned
  in `tests/test_rules_freeze.py` and consumed by the genericity defense
  (H6, H7).
- **`uv run pytest` covers `tests/` only.** Paper evals run explicitly:
  `uv run pytest paper_evals/`. Do not migrate paper evals into `tests/`
  — the discovery separation is intentional.

## Daily workflow

1. Open `TASKS.md`, find the next unchecked task for today
2. Make changes
3. Commit (no Co-Authored-By trailer)
4. Mark `[x]` in `TASKS.md` and append `— _commit `<hash>`_`
5. If discovering a new task, add it inline rather than holding it in head

## Paper submission target

eScience '26 full paper, 8 pages IEEE, **deadline 2026-06-01 AoE**. See
`AGENTSTAGE.md` §11 for the 14-day execution plan and §11.9 for risks.
