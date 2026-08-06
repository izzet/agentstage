"""H11: Agents reveal workspace structure before substantive reads.

The agent's first tool call is overwhelmingly `list_dir` (or an equivalent
filesystem probe). This is the empirical anchor for the subset-selection
challenge: the predictor doesn't have to commit on first signal — it can
combine streaming reasoning with the active filesystem evidence the agent
is materializing turn by turn.

Across the E-033 measurement (n=638 runs spanning 11 AIOB workloads × 7
models), 96.1% of first tool calls were `list_dir`. The remainder were
`run_shell_command` invocations (typically `ls`, `find`, or `head` — also
filesystem probes by another name).

Serves: C2 (subset selection), supports the §1 P3 empirical claim and
        the §4 predictor design.
Origin: outputs/microbench/first_tool_stat.json
Required data: outputs/microbench/first_tool_stat.json (E-033) OR live
               campaign with `blocks` populated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.h11


_PROBE_TOOLS = {"list_dir", "run_shell_command"}


def _first_tool_stat(outputs_root: Path) -> dict | None:
    """Load the E-033 pre-aggregated artifact if present."""
    p = outputs_root / "microbench" / "first_tool_stat.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _first_tool_name(run) -> str | None:
    """First tool_use block's tool_name, or None if the agent never used a tool."""
    for b in run.blocks:
        if b.get("type") == "tool_use":
            return b.get("tool_name") or b.get("tool") or None
    return None


class TestFirstToolIsListDir:
    """Across the campaign, the dominant first tool is `list_dir`."""

    def test_first_tool_distribution_from_e033(self, outputs_root, report):
        """Pre-aggregated E-033 result: ≥ 90% of first tool calls across 638
        runs are filesystem probes (list_dir + run_shell_command).

        §1 P3 anchor — supports the subset-selection challenge framing.
        """
        stat = _first_tool_stat(outputs_root)
        if stat is None:
            pytest.skip(
                "H11.e033: outputs/microbench/first_tool_stat.json missing — "
                "run scripts/microbench/first_tool_stat.py against the campaign."
            )

        n_total = stat.get("n_runs_with_first_tool")
        dist = stat.get("overall_distribution", {})
        list_dir_pct = dist.get("list_dir", 0.0)
        probe_pct = sum(v for k, v in dist.items() if k in _PROBE_TOOLS)

        report.record("h11_first_tool_distribution_e033", {
            "n_runs_total": stat.get("n_runs_scanned"),
            "n_runs_with_first_tool": n_total,
            "overall_distribution": dist,
            "list_dir_pct": list_dir_pct,
            "any_probe_pct": probe_pct,
        })

        # The §1 P3 claim is the 96.1% figure. We assert ≥ 90% so a small
        # campaign shift doesn't redden the suite without warning.
        assert list_dir_pct >= 90.0, (
            f"H11.e033: list_dir share is {list_dir_pct:.1f}% across "
            f"{n_total} runs — below the 90% threshold (§1 P3 anchor: 96.1%)."
        )

    def test_first_tool_is_probe_live_campaign(self, campaign, min_seeds, report):
        """Live recomputation from the campaign object: ≥ 80% of runs that
        emitted ANY tool call started with a filesystem probe (list_dir or
        run_shell_command).

        Tighter than the E-033 pre-aggregate because campaign data may
        include configs that bias toward direct reads (e.g., pathful-prompt
        variants from H12). Threshold loosened to 80% to allow that bias
        without spurious failure.
        """
        runs_with_tool: list[tuple] = []
        for r in campaign.runs:
            name = _first_tool_name(r)
            if name is not None:
                runs_with_tool.append((r, name))

        if len(runs_with_tool) < min_seeds:
            pytest.skip(
                f"H11.live: only {len(runs_with_tool)} runs with a tool "
                f"call, need ≥ {min_seeds}"
            )

        n_probe = sum(1 for _, n in runs_with_tool if n in _PROBE_TOOLS)
        n_list_dir = sum(1 for _, n in runs_with_tool if n == "list_dir")
        n_total = len(runs_with_tool)
        probe_frac = n_probe / n_total
        list_dir_frac = n_list_dir / n_total

        report.record("h11_first_tool_live_campaign", {
            "n_runs_with_tool": n_total,
            "list_dir_count": n_list_dir,
            "list_dir_frac": round(list_dir_frac, 3),
            "any_probe_count": n_probe,
            "any_probe_frac": round(probe_frac, 3),
        })

        assert probe_frac >= 0.80, (
            f"H11.live: only {probe_frac:.0%} of {n_total} live campaign runs "
            f"opened with a filesystem probe — the subset-selection anchor "
            f"is weaker than expected (§1 P3 target: 96.1% via list_dir alone)."
        )


class TestPerModelListDirRate:
    """Per-model breakdown — some models (e.g., Haiku) substitute
    `run_shell_command` for `list_dir`, so the per-model rate matters
    for understanding the heterogeneity behind the headline 96.1%."""

    def test_no_model_below_60pct_probe(self, outputs_root, report):
        """For every model in the E-033 aggregate, the share of first-tool
        calls that are filesystem probes (list_dir OR run_shell_command)
        is at least 60%. Captures the Haiku behavior in §6 (73% list_dir
        + 27% run_shell_command = 100% probe) without forcing list_dir
        specifically.
        """
        stat = _first_tool_stat(outputs_root)
        if stat is None:
            pytest.skip("H11.per_model: first_tool_stat.json missing")

        per_model = stat.get("per_model", {})
        if not per_model:
            pytest.skip("H11.per_model: no per_model section")

        rows = []
        violators = []
        for model_key, m in per_model.items():
            n = m.get("n", 0)
            if n == 0:
                continue
            probe_pct = sum(
                t["pct"] for t in m.get("top", [])
                if t.get("tool") in _PROBE_TOOLS
            )
            rows.append({"model": model_key, "n": n,
                         "any_probe_pct": round(probe_pct, 1)})
            if probe_pct < 60.0:
                violators.append((model_key, probe_pct))

        report.record("h11_per_model_probe_rate", rows)

        assert not violators, (
            f"H11.per_model: {len(violators)} models below 60% probe rate: "
            f"{violators[:5]}"
        )
