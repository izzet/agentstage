"""H1: Reasoning slack windows are real, large, and reliable.

LLM agents with thinking enabled produce a wall-clock gap between first
thinking chunk and first tool dispatch that is large enough at typical
NVMe ingest rates (3–7 GB/s) to stage useful amounts of data (≥ 100 MB)
per tool call.

Serves: C1, C6
Origin: slack distribution
Required data: trace-only (--outputs-root)
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.h1


def _thinking_seeds_with_slack(campaign):
    """Seeds that produced thinking AND emitted a tool_use (so slack_ms is
    well-defined). Includes aiob_101 — slack is not affected by the
    structural ambiguity that makes its byte recall an edge case."""
    return [
        r for r in campaign.runs
        if r.has_thinking and r.slack_ms is not None
    ]


class TestSlackDistribution:
    """The slack window's CDF clears the thresholds in §6.1."""

    def test_median_slack_above_5s(self, campaign, min_seeds, report):
        """Median slack across thinking-bearing seeds ≥ 5 000 ms.

        §6.1 reports median 6 268 ms across 44 thinking seeds. The
        threshold here is 5 s — a budget that comfortably stages 100 MB
        at 50 MB/s S3-class cold tier per Figure 1.
        """
        seeds = _thinking_seeds_with_slack(campaign)
        if len(seeds) < min_seeds:
            pytest.skip(f"need ≥ {min_seeds} thinking seeds w/ slack, got {len(seeds)}")

        slacks = sorted(r.slack_ms for r in seeds)
        median = slacks[len(slacks) // 2]
        p25 = slacks[len(slacks) // 4]
        p75 = slacks[(3 * len(slacks)) // 4]
        report.record("table_slack_distribution", {
            "n_seeds": len(slacks),
            "median_ms": median,
            "p25_ms": p25,
            "p75_ms": p75,
            "max_ms": max(slacks),
            "min_ms": min(slacks),
        })
        assert median >= 5000.0, (
            f"Median slack {median:.0f} ms on {len(slacks)} thinking seeds "
            f"is below the 5 s threshold (§6.1 target: 6.3 s)"
        )

    def test_fraction_seeds_above_thresholds(self, campaign, min_seeds, report):
        """Slack ≥ 2 s on ≥ 80% of seeds; ≥ 5 s on ≥ 50%.

        §6.1 reports 86% ≥ 2 s, 64% ≥ 5 s; §TLDR reports 98% / 67% on the
        cleaned subset. We assert the conservative 80% / 50% thresholds
        so the test stays green across the full re-scored corpus.
        """
        seeds = _thinking_seeds_with_slack(campaign)
        if len(seeds) < min_seeds:
            pytest.skip(f"need ≥ {min_seeds} thinking seeds w/ slack, got {len(seeds)}")

        slacks = [r.slack_ms for r in seeds]
        frac_2s = sum(1 for s in slacks if s >= 2000) / len(slacks)
        frac_5s = sum(1 for s in slacks if s >= 5000) / len(slacks)
        report.record("table_slack_threshold_fractions", {
            "n_seeds": len(slacks),
            "frac_ge_2s": frac_2s,
            "frac_ge_5s": frac_5s,
        })
        assert frac_2s >= 0.80, (
            f"Only {frac_2s:.0%} of seeds clear 2s slack (§6.1 target: 86%)"
        )
        assert frac_5s >= 0.50, (
            f"Only {frac_5s:.0%} of seeds clear 5s slack (§6.1 target: 64%)"
        )

    def test_slack_per_provider_consistency(self, campaign, min_seeds, report):
        """Every provider family with ≥ 3 thinking seeds has median slack
        ≥ 2 s — the floor below which prestaging stops paying off.

        DeepSeek-R1 may be an outlier on the HIGH side (248 s thinking in
        the PoC); the floor is what we care about, not the ceiling.
        """
        all_seeds = _thinking_seeds_with_slack(campaign)
        if not all_seeds:
            pytest.skip("no thinking seeds available")

        per_family: dict[str, list[float]] = {}
        for r in all_seeds:
            per_family.setdefault(r.provider_family, []).append(r.slack_ms)

        record = {}
        for fam, slacks in per_family.items():
            slacks_sorted = sorted(slacks)
            record[fam] = {
                "n_seeds": len(slacks),
                "median_ms": slacks_sorted[len(slacks_sorted) // 2],
            }
        report.record("table_slack_per_provider", record)

        failures = []
        for fam, slacks in per_family.items():
            if len(slacks) < 3:
                continue  # too few to assert
            med = sorted(slacks)[len(slacks) // 2]
            if med < 2000.0:
                failures.append((fam, med, len(slacks)))
        assert not failures, (
            "Provider families below 2 s median slack: "
            + ", ".join(f"{fam}={med:.0f}ms (n={n})" for fam, med, n in failures)
        )
