"""H8: End-to-end staging materially reduces per-tool first-read latency.

With the full proxy + detector + stager running, per-tool-call first-read
P95 latency on file-I/O-bound scientific agent workloads is reduced by ≥ 2×
versus the no-prestaging baseline at S3-class cold-tier bandwidth.

This is the load-bearing measured contribution. Without an H8 pass, C4 in
the paper degrades to a design proposal.

Serves: C8 (status: unverified → verified after E5), E5
Origin: AGENTSTAGE.md §11.5 (E5 row), §11.9 (risk row "E5 measured speedup")
Required data: end-to-end staging campaigns (--staging-root) from Day 5-7+.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.h8


class TestFirstReadLatencyReduction:
    """Per-tool first-read P95 with stager ≤ 0.5 × baseline."""

    @pytest.mark.parametrize("task_id", ["aiob_107", "aiob_110"])
    def test_first_read_p95_speedup(
        self, task_id, outputs_root, min_seeds, report
    ):
        """First-read P95 with AgentStage / first-read P95 without ≤ 0.5
        on the primary speedup-favorable workloads.

        §11.9 risk note: if all bandwidth regimes fail the 1.3× wall-clock
        target, P95 first-read reduction is still a publishable result
        even if total wall-clock is flat. This test pins the P95 measure.

        Records: `figure_first_read_p95_per_task` (with vs without).
        """
        if outputs_root is None:
            pytest.skip("H8 requires --staging-root (end-to-end stager output)")
        pytest.skip(
            f"H8.first_read_p95[{task_id}]: pending — requires Day 5-7 stager "
            "build + staging_report.json schema."
        )


class TestEndToEndWallClock:
    """Total wall-clock reduction is the headline number. Pass threshold
    1.3× per §11.9 — this is the risk-managed minimum, not the 2-5×
    aspirational claim."""

    def test_wall_clock_speedup_ge_1_3x(
        self, outputs_root, min_seeds, report
    ):
        """median(wall_clock without) / median(wall_clock with) ≥ 1.3 at
        50 MB/s cold-tier bandwidth (S3-class).

        Records: `table_wall_clock_speedup_per_config`.
        """
        if outputs_root is None:
            pytest.skip("H8 requires --staging-root")
        pytest.skip(
            "H8.wall_clock: pending — Day 7 measured end-to-end target."
        )


class TestStagerCorrectness:
    """The stager must not be cheating: prestaged bytes count toward the
    same checksum the agent would read directly."""

    def test_no_byte_divergence_between_staged_and_direct(self, outputs_root):
        """For every staged file, sha256(staged copy) == sha256(direct read)
        — caught early so we don't quietly serve stale or partial data.
        """
        if outputs_root is None:
            pytest.skip("H8 requires --staging-root")
        pytest.skip(
            "H8.byte_correctness: pending — needs stager checksum hook."
        )
