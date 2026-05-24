"""H9: End-to-end speedup is monotone-decreasing in cold-tier bandwidth.

At 10 MB/s (PFS-class), the slack window covers more useful data than the
agent's tool can consume during reasoning, so prestaging wins big. At
native XFS-SSD (≈140 MB/s) and especially local NVMe (≈3 GB/s), the cold
tier is already fast enough that prestaging margin shrinks. The curve
shape — and the regime where AgentStage is most valuable — is the headline
of §3 Figure 1 / Figure 2.

Backed by E-007 throttle sweep (aiob_110, 10/30/50 MB/s tc-throttled
NFS + native + with-stager) and E-028/E-030 end-to-end measurements on
local NVMe XFS and S3-class FUSE-S3.

Serves: E6
Origin: AGENTSTAGE.md §11.5 (E6 row), §11.2 (figures 1 & 2)
Required data: outputs/microbench/throttle_sweep_aiob_110_*/ +
               outputs/e2e/{local,s3}/e2e_*.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.h9


def _load_throttle_sweep(outputs_root: Path) -> dict | None:
    """Find most-recent throttle sweep dir + load its JSON points."""
    sweeps = sorted((outputs_root / "microbench").glob("throttle_sweep_aiob_*"))
    if not sweeps:
        return None
    latest = sweeps[-1]
    points: dict[str, dict] = {}
    for jf in latest.glob("*.json"):
        try:
            points[jf.stem] = json.loads(jf.read_text())
        except (OSError, json.JSONDecodeError):
            continue
    if not points or "with_stager" not in points:
        return None
    return {"dir": str(latest), "points": points}


def _load_e2e(outputs_root: Path, tier: str) -> dict:
    p = outputs_root / "e2e" / tier / f"e2e_{tier}.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def _wall_speedup_at(points: dict, key: str) -> float | None:
    """Wall-time speedup at a throttle point vs with_stager."""
    if key not in points or "with_stager" not in points:
        return None
    bw_mean = points[key]["aggregate"]["full_read_ms"]["mean"]
    hot_mean = points["with_stager"]["aggregate"]["full_read_ms"]["mean"]
    return bw_mean / hot_mean if hot_mean > 0 else None


class TestBandwidthCurve:
    """Speedup curve is monotone in cold-tier BW: as cold BW decreases,
    speedup increases. Built from E-007's 4 measured points + E-028's
    local-NVMe and S3 end-to-end points."""

    def test_speedup_monotone_decreasing_in_bw(self, outputs_root, report):
        sweep = _load_throttle_sweep(outputs_root)
        if sweep is None:
            pytest.skip("H9.monotone: no throttle_sweep_aiob_* under outputs/microbench/")
        pts = sweep["points"]
        # Throttle points in increasing BW order:
        #   10 → 30 → 50 → native (~141 MB/s)
        ordered = [
            ("10mbps", 10.0),
            ("30mbps", 30.0),
            ("50mbps", 50.0),
            ("native", 141.0),
        ]
        curve = []
        for label, mbps_nominal in ordered:
            key = f"baseline_{label}"
            sp = _wall_speedup_at(pts, key)
            if sp is None:
                continue
            mbps_actual = pts[key]["aggregate"]["throughput_mbps"]["mean"]
            curve.append({"mbps": mbps_actual, "speedup": round(sp, 2),
                          "label": label})

        if len(curve) < 3:
            pytest.skip(f"H9.monotone: need ≥3 sweep points, got {len(curve)}")
        report.record("figure_bandwidth_curve", curve)

        # Monotone: speedup strictly decreases as BW increases. Allow ±5%
        # noise tolerance (flat ranges within ±5% are OK).
        for prev, nxt in zip(curve, curve[1:]):
            if prev["speedup"] < 0.95 * nxt["speedup"]:
                pytest.fail(
                    f"H9 monotonicity violated: at {prev['mbps']:.1f} MB/s "
                    f"speedup={prev['speedup']:.2f}×, but at "
                    f"{nxt['mbps']:.1f} MB/s speedup={nxt['speedup']:.2f}× "
                    f"(higher BW should have ≤ speedup)."
                )

    def test_s3_class_speedup_above_threshold(self, outputs_root, report):
        """At S3-class cold tier (mountpoint-s3 over noaa-goes16), the
        end-to-end agent-script speedup must clear 2×. This is the regime
        AgentStage is designed for — must show its biggest win here."""
        e2e_s3 = _load_e2e(outputs_root, "s3")
        if not e2e_s3:
            pytest.skip("H9.s3_class: outputs/e2e/s3/e2e_s3.json missing")
        sp = e2e_s3["session_speedup"]
        report.record("h9_s3_class_speedup", {
            "tier": "s3", "speedup": sp,
            "baseline_s": e2e_s3["baseline"]["elapsed_s"],
            "staged_s": e2e_s3["staged"]["elapsed_s"],
            "experiment": e2e_s3["experiment"],
        })
        assert sp >= 2.0, (
            f"H9.s3_class: only {sp:.2f}× at S3-class — must be ≥ 2×. "
            f"baseline {e2e_s3['baseline']['elapsed_s']:.1f}s, "
            f"staged {e2e_s3['staged']['elapsed_s']:.1f}s."
        )

    def test_throttled_pfs_speedup_above_threshold(self, outputs_root, report):
        """At 10 MB/s (PFS-class throttled), wall-time speedup must clear 5×."""
        sweep = _load_throttle_sweep(outputs_root)
        if sweep is None:
            pytest.skip("H9.throttled_pfs: no throttle sweep artifacts")
        sp = _wall_speedup_at(sweep["points"], "baseline_10mbps")
        if sp is None:
            pytest.skip("H9.throttled_pfs: baseline_10mbps missing")
        report.record("h9_throttled_10mbps_speedup", {
            "throttle_mbps": 10.0, "wall_speedup": round(sp, 2),
        })
        assert sp >= 5.0, (
            f"H9.throttled_pfs: only {sp:.2f}× at 10 MB/s, expected ≥ 5×."
        )


class TestLocalNvmeLowerBound:
    """Local NVMe XFS is the lower bound of the speedup spectrum. We
    don't require a big win here — just that we don't regress below
    1×. This pins the worst-case end of the H9 curve in our claim."""

    def test_local_nvme_not_regressing(self, outputs_root, report):
        e2e_local = _load_e2e(outputs_root, "local")
        if not e2e_local:
            pytest.skip("H9.local_nvme: outputs/e2e/local/e2e_local.json missing")
        sp = e2e_local["session_speedup"]
        report.record("h9_local_nvme_lower_bound", {
            "tier": "local_nvme_xfs", "speedup": sp,
            "baseline_s": e2e_local["baseline"]["elapsed_s"],
            "staged_s": e2e_local["staged"]["elapsed_s"],
        })
        assert sp >= 1.0, (
            f"H9.local_nvme: {sp:.2f}× — staging made local NVMe SLOWER. "
            f"Investigate shim overhead."
        )
