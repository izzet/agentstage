"""H3: A tiered semantic-class predictor over the workspace prior achieves
high byte recall and low overfetch against the agent's actual file accesses.

THIS IS THE PAPER'S HEADLINE HYPOTHESIS. Tier-1 covers the immediate-need
set; tier-3 covers the eventual working set. Both must clear the reviewer-
stated thresholds (≥ 0.85 byte recall, ≤ 1.5× / ≤ 2.0× overfetch) across
provider families and workloads.

Serves: C2, C3 (and through them, the entire paper's central claim)
Origin: AGENTSTAGE.md §3, §6.2 (predictor accuracy), §6.4 (per-config)
Required data: trace-only (--trace-root) + ground-truth (--ground-truth-root)
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.h3


class TestTier1ImmediateNeed:
    """Tier-1 stages a small, precisely-correct set for immediate execution."""

    def test_tier1_byte_recall(
        self, trace_root, ground_truth_root, min_seeds, report
    ):
        """≥ 0.85 byte recall on ≥ 90% of well-defined seeds (i.e. excluding
        aiob_101's structurally-ambiguous 36-NetCDF bucket).

        §TLDR: 94% (44/47). §6.2: 98% at the lower 0.70 threshold.
        Records: `table_tier1_byte_recall_per_config`.
        """
        pytest.skip(
            "H3.tier1_recall: pending — primary Day 1 assertion. Needs "
            "byte_metrics.json loader + per-seed ground-truth join + "
            "aiob_101 exclusion filter."
        )

    def test_tier1_byte_overfetch(
        self, trace_root, ground_truth_root, min_seeds, report
    ):
        """≤ 1.5× byte overfetch on ≥ 95% of well-defined seeds.

        §TLDR: 98% (46/47). The single miss is the DeepSeek-R1 two-subject
        commit at 2.12× — flagged in the data record but not the assertion.
        """
        pytest.skip(
            "H3.tier1_overfetch: pending — same loader as recall test."
        )


class TestTier3EventualWorkingSet:
    """Tier-3 stages the eventual working set with bounded overfetch."""

    def test_tier3_byte_recall(
        self, trace_root, ground_truth_root, min_seeds, report
    ):
        """≥ 0.85 byte recall on ≥ 95% of well-defined seeds against the
        eventual-working-set ground truth.

        §TLDR: 100% (47/47) on the 4 well-defined workloads.
        """
        pytest.skip(
            "H3.tier3_recall: pending — needs eventual-working-set ground "
            "truth (per-task static + empirical-via-io_report join)."
        )

    def test_tier3_byte_overfetch(
        self, trace_root, ground_truth_root, min_seeds, report
    ):
        """≤ 2.0× byte overfetch on ≥ 95% of well-defined seeds.

        §TLDR: 98% (46/47); the miss is code_repo where mention-rules fired
        for many modules.
        """
        pytest.skip(
            "H3.tier3_overfetch: pending — same loader as tier-3 recall."
        )


class TestCrossProviderConsistency:
    """The tier-1 result holds across LLM provider families (§6.4.1, TLDR)."""

    def test_anthropic_family_perfect_tier1(
        self, trace_root, ground_truth_root, report
    ):
        """Anthropic (Sonnet 4.5 + Haiku 4.5) reaches 100% tier-1 byte recall
        on all well-defined seeds (n = 34 in the §TLDR breakdown).
        """
        pytest.skip(
            "H3.anthropic_family: pending — needs per-provider rollup."
        )

    def test_gemini_overfetch_holds(self, trace_root, ground_truth_root, report):
        """Gemini 2.5 Pro: byte overfetch ≤ 1.5× holds on 100% of seeds even
        when recall misses fire (the misses are strategy-variance, not
        predictor failures; see §6.4.1).
        """
        pytest.skip(
            "H3.gemini_overfetch: pending — same loader."
        )
