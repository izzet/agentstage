"""H4: Tiered staging is the right architecture. The worst-case workload
collapses from naive stage-all overfetch to ≈ 1× under tier-1.

The aiob_107 (GOES meteorology) case is the visceral selling point: 6 042
files in workspace, 18 GB total, immediate need is one 3 MB NetCDF. Naive
stage-all overfetches 6 078×; tier-1 overfetches 1.00×. Both decisions
happen within the slack window.

Serves: C2/C3 case study, supports the architectural decision in §4.2
Origin: AGENTSTAGE.md §6.3 (the GOES collapse table)
Required data: trace-only (--trace-root) + ground-truth (--ground-truth-root)
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.h4


class TestGoesCollapse:
    """The aiob_107 case study — paper's Section 6.3 reproduces here."""

    def test_aiob107_tier1_overfetch_near_1(
        self, outputs_root, io_report_root, report
    ):
        """Tier-1 byte overfetch on aiob_107 ≤ 1.2× across all seeds.

        §6.3 reports 1.00× on the sonnet+PP, seed 0 trace; the 11-seed
        sonnet config in §TLDR is 100% at ≤ 1.5×.

        Records: `figure_goes_collapse` (detector → overfetch table).
        """
        pytest.skip(
            "H4.aiob107_collapse: pending — depends on H3's byte_metrics "
            "loader; filter on task_id == aiob_107."
        )

    def test_naive_baseline_is_dramatically_worse(
        self, outputs_root, io_report_root
    ):
        """The stage-all baseline overfetches ≥ 1 000× on aiob_107 — without
        a baseline this dramatic, the tiering story doesn't sell.

        This is computed analytically: workspace bytes / immediate-need
        bytes for the workload, not from a probe.
        """
        pytest.skip(
            "H4.naive_baseline: pending — needs workload static inventory "
            "(workspace size + immediate-need bytes) in "
            "agentstage.workloads.aiob."
        )


class TestTieringIsLoadBearing:
    """Across all workloads, removing the tier-1 layer materially worsens
    overfetch versus the union (tier-3) of all rules."""

    def test_union_overfetch_exceeds_tier1_on_high_fanout_workloads(
        self, outputs_root, io_report_root, report
    ):
        """On workloads where workspace ≫ working set (aiob_107, code_repo),
        the union-of-rules overfetch is ≥ 10× the tier-1 overfetch. This
        is what justifies the tiering choice over a single-set detector.
        """
        pytest.skip(
            "H4.union_vs_tier1: pending — needs both tier-1 and union "
            "(legacy WARM) byte metrics in byte_metrics.json."
        )
