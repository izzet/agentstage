"""H7: Within-corpus leave-one-out generalization.

Tune the rule library on 3 of the 4 well-defined AgentIOBench workloads,
evaluate on the held-out 4th. Tier-1 byte recall ≥ 0.85 must hold on the
held-out workload. This is the L1 level of the §11.6 genericity defense.

Note: in practice the rule library is FROZEN from Day 1 (see H6); the
leave-one-out here re-scores existing trace data against a rule library
that simply doesn't include rules referencing the held-out workload's
specific vocabulary. The mechanism is rule-tagging, not rule-retraining.

Serves: L1 genericity (E3)
Origin: AGENTSTAGE.md §11.6
Required data: trace-only (--trace-root) — works against the existing
88-probe PoC corpus today.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.h7


class TestLeaveOneOut:
    """For each well-defined workload (aiob_104, aiob_107, aiob_110,
    code_repo), holding it out from the rule-source set still yields
    tier-1 byte recall ≥ 0.85 on that workload."""

    @pytest.mark.parametrize(
        "held_out", ["aiob_104", "aiob_107", "aiob_110", "code_repo"]
    )
    def test_held_out_tier1_recall(
        self, held_out, trace_root, ground_truth_root, min_seeds, report
    ):
        """Re-score the held-out workload's traces using only rules tagged
        as originating from the other 3 workloads. Tier-1 byte recall on
        the held-out workload must clear 0.85 (or 0.70 for code_repo,
        which is the §6.4 known-weak case).
        """
        pytest.skip(
            f"H7.loo[{held_out}]: pending — needs (a) rule-origin tagging in "
            "agentstage.predictor.rules and (b) a re-score driver. Both "
            "land alongside Day 2's leave-one-out work in §11.8."
        )

    def test_loo_summary_table(
        self, trace_root, ground_truth_root, min_seeds, report
    ):
        """Aggregate the 4 leave-one-out results into a single table for the
        paper's §11.6 Level-1 row.

        Records: `table_loo_tier1_recall` (per held-out workload).
        """
        pytest.skip(
            "H7.summary: pending — aggregator over the parametrized results."
        )
