"""H5: Planning prompts are a free 2-10× slack multiplier.

Inserting explicit thinking instructions ("think step-by-step about which
files you'll read") into the user message multiplies slack and increases
intent-extraction precision without changing the model or budget. This
is a paired comparison: same (task, model, turn) with prompt variant.

Serves: C6
Origin: AGENTSTAGE.md §2 (H5), §3 (C6), §7.1 (free lever)
Required data: trace-only (--trace-root) — needs paired no-PP / PP configs
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.h5


class TestPlanningPromptSlackMultiplier:
    """Paired comparison: with-PP slack ≥ 2× without-PP slack."""

    def test_paired_slack_ratio_per_config(
        self, trace_root, min_seeds, report
    ):
        """For every (task, model, turn) that has both no-PP and PP runs,
        median(PP slack) / median(no-PP slack) ≥ 2.0.

        §3 C6 evidence: aiob_101 t=1 sonnet no-PP 3.9 s vs PP 9.6 s
        (2.5×); gemini-pro aiob_110 t=2 no-PP 0.8 s vs PP 7.9 s (~10×).

        Records: `table_planning_prompt_multipliers` (per-config ratio).
        """
        pytest.skip(
            "H5.paired_ratio: pending — needs Campaign.pair_by_prompt_mode "
            "helper to join configs that differ only in the PP flag."
        )

    def test_strict_pp_does_not_regress(self, trace_root, min_seeds):
        """The 'strict-PP' variant (force literal absolute paths) does not
        REDUCE slack compared to the vanilla planning prompt — at worst
        it ties. This was a concern in §5.4 / §5.5 (strict-PP is one of
        the matrix configs); we lock the result in here.
        """
        pytest.skip(
            "H5.strict_pp: pending — needs the strict-PP slice in the "
            "Campaign indexer."
        )


class TestPlanningPromptIntentPrecision:
    """Planning prompts also tighten the predicted set (lower overfetch),
    not just lengthen slack. This is the part of C6 the paper claims but
    has not separately measured."""

    def test_pp_reduces_tier1_overfetch(
        self, trace_root, ground_truth_root, min_seeds, report
    ):
        """For paired configs, tier-1 overfetch with PP ≤ tier-1 overfetch
        without PP. Direction-only test — magnitude varies by workload.
        """
        pytest.skip(
            "H5.pp_overfetch: pending — depends on H3 byte_metrics + H5 "
            "pairing helper."
        )
