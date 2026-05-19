"""H9: End-to-end speedup is monotone-decreasing in cold-tier bandwidth.

At 50 MB/s (S3-class), the slack window covers more useful data than the
agent's tool can consume during reasoning, so prestaging wins big. At 3 GB/s
(local NVMe-on-NVMe), the cold tier is already fast enough that prestaging
margin shrinks. The curve shape — and the regime where AgentStage is most
valuable — is the headline of §3 Figure 1 / Figure 2.

Serves: E6
Origin: AGENTSTAGE.md §11.5 (E6 row), §11.2 (figures 1 & 2)
Required data: end-to-end staging campaigns (--staging-root) at multiple
bandwidth regimes. 1 measured point + 3 simulator points per §11.7.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.h9


class TestBandwidthCurve:
    """Speedup curve is monotone in cold-tier BW, and the shape matches the
    Figure 1 / Figure 2 selling story."""

    def test_speedup_monotone_decreasing_in_bw(self, outputs_root, report):
        """For BW points {50, 200, 1000, 3000} MB/s, speedup(BW_i) ≥
        speedup(BW_j) when BW_i < BW_j (strict monotonicity not required —
        flat regions allowed within ±5% margin).

        Records: `figure_bandwidth_curve` (BW, measured speedup, simulator
        speedup, source).
        """
        if outputs_root is None:
            pytest.skip("H9 requires --staging-root")
        pytest.skip(
            "H9.monotone: pending — Day 10 bandwidth sweep. Needs tc/cgroup "
            "rate-limit on the NFS source for the measured point + simulator "
            "interpolation for the rest."
        )

    def test_s3_class_speedup_above_threshold(self, outputs_root):
        """At 50 MB/s (S3-class cold tier), end-to-end speedup ≥ 2×. This is
        the regime AgentStage is designed for — the curve must show its
        biggest win here.
        """
        if outputs_root is None:
            pytest.skip("H9 requires --staging-root")
        pytest.skip(
            "H9.s3_class: pending — needs Day 10 measurement at the 50 MB/s "
            "tc cap."
        )


class TestSimulatorMeasuredAgreement:
    """The simulator is complementary to real-stager numbers; where they
    overlap (the one measured BW point), they must agree to within 20%."""

    def test_simulator_within_20pct_of_measured(self, outputs_root, report):
        """abs(speedup_measured - speedup_simulator) / speedup_measured ≤ 0.20
        at the measured BW point. Validates the simulator as a credible
        sensitivity tool for the other BW points.
        """
        if outputs_root is None:
            pytest.skip("H9 requires --staging-root")
        pytest.skip(
            "H9.sim_vs_measured: pending — simulator built Days 3-4, "
            "cross-check on Day 10."
        )
