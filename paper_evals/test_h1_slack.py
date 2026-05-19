"""H1: Reasoning slack windows are real, large, and reliable.

LLM agents with thinking enabled produce a wall-clock gap between first
thinking chunk and first tool dispatch that is large enough at typical NVMe
ingest rates (3-7 GB/s) to stage useful amounts of data (≥ 100 MB) per
tool call.

Serves: C1, C6
Origin: AGENTSTAGE.md §3 (claims table), §6.1 (slack distribution)
Required data: trace-only (--trace-root)
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.h1


class TestSlackDistribution:
    """The slack window's CDF clears the thresholds in §6.1."""

    def test_median_slack_above_5s(self, outputs_root, min_seeds, report):
        """Median slack across thinking-bearing seeds ≥ 5 000 ms.

        §6.1 reports median 6 268 ms across 44 thinking seeds. The threshold
        here is 5 s — a budget that comfortably stages 100 MB at 50 MB/s
        S3-class cold tier per Figure 1.

        Will record: `table_slack_distribution` (per-config median, p25, p75, max).
        """
        pytest.skip(
            "H1.median_slack: pending — needs agentstage.workloads.Campaign to "
            "index trace-root run directories and pull `slack_ms` out of "
            "summary.json. Land alongside the E2 re-score on Day 1."
        )

    def test_fraction_seeds_above_thresholds(self, outputs_root, min_seeds):
        """Slack ≥ 2 s on ≥ 80% of seeds; ≥ 5 s on ≥ 60%.

        §6.1 reports 86% ≥ 2 s, 64% ≥ 5 s; §TLDR reports 98% / 67% on the
        47-seed cleaned subset. We pick the more conservative thresholds
        here so the suite stays green across the full 88-probe corpus.
        """
        pytest.skip(
            "H1.threshold_fractions: pending — depends on same loader as "
            "test_median_slack_above_5s."
        )

    def test_slack_per_provider_consistency(self, outputs_root, min_seeds, report):
        """Each provider family (Anthropic, Gemini, DeepSeek-R1) clears the 2 s
        floor on its median slack.

        Cross-provider story in TLDR + §6.4 + §6.4.1. DeepSeek-R1 is allowed
        to be an outlier on the high side (248 s thinking) but not below 2 s.
        """
        pytest.skip(
            "H1.per_provider: pending — needs provider-family extraction from "
            "the run-dir filename convention (see agentstage.workloads)."
        )
