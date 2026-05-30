# paper_figures/

Reproducible figure pipeline for the AgentStage eScience '26 paper. Pattern lifted from `sciiobench/scripts/build_fig_*.py`, with the rcParams block factored into a shared `_style.py` so future style tweaks are single-source.

## Convention

| Aspect | Rule |
|---|---|
| Font | serif 10pt across every text element; never override per-element |
| Sizes | half-col raw 1.85"×1.85"; full-col raw 3.85"; figure* raw 7.5" (scale 0.86) |
| Layout | two-panel via `\begin{subfigure}{0.48\columnwidth}` — auto (a)/(b) |
| Titles | no panel titles inside the figure; caption + subcaption do the framing |
| Axis labels | Title Case ("Tier-1 Byte Recall") |
| Grid | y-axis only, linestyle=':', alpha=0.35, axisbelow=True |
| Annotation bbox | round-pad 0.10, white, no edge |
| Output | both `.pdf` and `.png` to `paper/figures/<name>.<ext>` |
| Data dump | per-figure CSV to `paper/figures/data/<name>.csv` for replay |
| Build location | `paper_figures/build_fig_*.py` at repo root, NOT inside `paper/` |

All scripts must start with:

```python
from _style import (
    HALF_COL, FULL_COL_W, FIGURE_STAR_W,
    MODEL_ORDER, MODEL_LABELS, COLORS,
    BACKEND_ORDER, BACKEND_LABELS, BACKEND_COLORS,
    save, dump_csv, style_axis, label_bbox,
)
```

`_style.py` already calls `matplotlib.use('Agg')` and applies the rcParams block — importing it is enough.

## Figures planned for eScience '26 paper (8 body pages, 6 figures)

| # | Name | Section | Type | Size convention |
|---|---|---|---|---|
| 1 | `fig_motivation` | §1 P3 teaser | 2-panel: slack distribution (left) + measured speedup per benchmark (right) | half-col subfig 1.85"×1.85" each |
| 2 | `fig_taxonomy` | §2.5 gap | 4-corner schematic: compute-side vs data-side × latency-hiding vs observability; AgentStage in the empty cell | full-col 3.85"×~1.8" |
| 3 | `fig_architecture` | §3 design | AgentStage architecture: 4 components + data flow arrows | likely TikZ directly in paper.tex (no Python script) |
| 4 | `fig_recall_overfetch` | §5 eval | Predictor byte recall + overfetch across AIOB / SAB / KB / DSBench / MLE-bench | full-col 3.85"×~2.4" |
| 5 | `fig_e2e_speedup` | §5 eval | End-to-end speedup per benchmark × model × backend (XFS / S3 / Lustre) | full-col 3.85"×~2.4" |
| 6 | `fig_bandwidth_sensitivity` | §5 eval | Speedup vs cold-tier BW (50 / 200 / 1000 / 3000 MB/s, 1 measured + 3 simulated per E6) | half-col 1.85"×1.85" |

## Build

```bash
# All figures
for f in paper_figures/build_fig_*.py; do uv run python "$f"; done

# Single figure
uv run python paper_figures/build_fig_motivation.py
```

Each script writes `paper/figures/<name>.pdf`, `paper/figures/<name>.png`, and `paper/figures/data/<name>.csv`.

## Prose conventions (carried over from sciiobench memory)

These apply to every `.tex` file under `paper/sections/`, not just figures, but listed here as the discoverable convention doc:

- **`\looseness=-1`** at the end of every paragraph (TeX primitive, works in IEEEtran)
- **No em dashes** (`---` in LaTeX, "—" anywhere) — use `:`, `,`, or `(...)` per context
- **Build to `paper/out/`** — `latexmk -output-directory=out paper.tex` from inside `paper/`

## Data sources

| Figure | Reads from |
|---|---|
| fig_motivation (left) | `outputs/*/stream.jsonl` + `outputs/*/summary.json` from PoC + Haiku/Flash trace probes — slack window column |
| fig_motivation (right) | E2E run dirs (e.g., `outputs/dsbench_mt/*/verdict.json`) — speedup vs baseline |
| fig_recall_overfetch | `paper_evals/.results/report.json` — keys `table_tier1_recall`, `table_tier3_recall`, `table_overfetch` |
| fig_e2e_speedup | E2E run dirs across benchmarks; aggregate via `agentstage.metrics` |
| fig_bandwidth_sensitivity | Bandwidth-sweep run dirs (E6) — 1 measured + 3 simulated per backend |

Each script should resolve data via `paper_evals` or `agentstage.workloads` public APIs, not by re-implementing aggregators inside `paper_figures/` (matches the [[project-paper-evals]] data-loading discipline).
