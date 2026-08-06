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
Origin: bandwidth-sensitivity requirement
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


class TestBytesMoveablePerBackend:
    """Figure 1b: bytes-moveable per slack window by cold-tier backend.

    This is the §1 P3 anchor — pairing slack with bandwidth shows that
    the budget for moving data spans 2+ orders of magnitude depending on
    the cold tier. The compute is `slack_s × backend_bandwidth_MBps`.

    Backends covered (from existing artifacts):
      - local NVMe XFS (~native throttle point, ~140 MB/s effective for
        the staging mount; faster for true NVMe page cache)
      - throttled PFS classes (10 / 30 / 50 MB/s — the H9 sweep points)
      - S3 (from outputs/e2e/s3 measured throughput)
      - OrangeFS (sourced from a storage profile if available)
    """

    @staticmethod
    def _slack_seconds_from_report(outputs_root: Path) -> float:
        """Use H1's recorded median slack if it ran; otherwise fall back
        to the §6.1 PoC value of 6.3 s."""
        report_path = (
            Path(__file__).parent / ".results" / "report.json"
        )
        if report_path.is_file():
            try:
                d = json.loads(report_path.read_text())
                rec = d.get("data", {}).get("table_slack_distribution")
                if isinstance(rec, dict) and rec.get("median_ms"):
                    return rec["median_ms"] / 1000.0
            except (OSError, json.JSONDecodeError):
                pass
        return 6.3

    def test_bytes_moveable_table_recorded(self, outputs_root, report):
        """Record the per-backend bytes-moveable table for Fig 1b. Asserts
        that the range spans at least 50× across the available backends —
        without that spread, the figure wouldn't tell a story."""
        slack_s = self._slack_seconds_from_report(outputs_root)
        rows: list[dict] = []

        # Throttle sweep points (PFS-class simulations)
        sweep = _load_throttle_sweep(outputs_root)
        if sweep:
            for label in ("10mbps", "30mbps", "50mbps", "native"):
                key = f"baseline_{label}"
                pts = sweep["points"].get(key)
                if not pts:
                    continue
                mbps = pts["aggregate"]["throughput_mbps"]["mean"]
                rows.append({
                    "backend": f"Throttled PFS ({label})"
                               if label != "native" else "Ares cold mount (native)",
                    "bandwidth_mbps": round(mbps, 2),
                    "bytes_moveable_mb": round(mbps * slack_s, 1),
                    "source": "throttle_sweep",
                })
            # Hot tier (NVMe page cache) — bytes-moveable upper bound
            hot = sweep["points"].get("with_stager")
            if hot:
                mbps = hot["aggregate"]["throughput_mbps"]["mean"]
                rows.append({
                    "backend": "Ares NVMe (hot, post-stage)",
                    "bandwidth_mbps": round(mbps, 2),
                    "bytes_moveable_mb": round(mbps * slack_s, 1),
                    "source": "throttle_sweep_with_stager",
                })

        # S3 — derive effective bandwidth from the e2e measurement
        s3 = _load_e2e(outputs_root, "s3")
        if s3 and s3.get("baseline", {}).get("elapsed_s"):
            # Approximate effective throughput from the baseline read time.
            total_b = s3.get("total_input_bytes") or 0
            elapsed_s = s3["baseline"]["elapsed_s"]
            if total_b and elapsed_s:
                mbps = (total_b / 1e6) / elapsed_s
                rows.append({
                    "backend": "Amazon S3 (mountpoint-s3)",
                    "bandwidth_mbps": round(mbps, 2),
                    "bytes_moveable_mb": round(mbps * slack_s, 1),
                    "source": "e2e_s3_baseline",
                })

        # OrangeFS — optional, pulled from a storage profile
        # at outputs/storage_profile_orangefs.json if the user dropped one.
        ofs_path = outputs_root / "storage_profile_orangefs.json"
        if ofs_path.is_file():
            try:
                ofs = json.loads(ofs_path.read_text())
                mbps = (
                    ofs.get("aggregate_throughput_mbps")
                    or ofs.get("throughput_mbps")
                )
                if mbps:
                    rows.append({
                        "backend": "OrangeFS",
                        "bandwidth_mbps": round(float(mbps), 2),
                        "bytes_moveable_mb": round(float(mbps) * slack_s, 1),
                        "source": "storage_profile_orangefs",
                    })
            except (OSError, json.JSONDecodeError, TypeError):
                pass

        if len(rows) < 2:
            pytest.skip(
                f"H9.bytes_moveable: only {len(rows)} backend rows — need "
                f"≥ 2 to compute a range. Drop throttle sweep data + S3 e2e."
            )

        report.record("figure_bytes_moveable_per_backend", {
            "slack_seconds_used": slack_s,
            "rows": rows,
        })

        mb_vals = [r["bytes_moveable_mb"] for r in rows]
        spread = max(mb_vals) / min(mb_vals) if min(mb_vals) > 0 else 0
        assert spread >= 50.0, (
            f"H9.bytes_moveable: backend range only spans {spread:.1f}× — "
            f"§1 P3 story claims 2+ orders of magnitude. Investigate "
            f"whether slow-tier data is missing."
        )
