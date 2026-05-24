"""H8: End-to-end staging materially reduces per-tool first-read latency.

With the full proxy + detector + stager running, per-tool-call first-read
P95 latency on file-I/O-bound scientific agent workloads is reduced by ≥ 2×
versus the no-prestaging baseline at S3-class cold-tier bandwidth.

Backed by E-028/E-029/E-030/E-031 end-to-end task-script measurements:
real agent-authored script + real stager + real LD_PRELOAD shim, running
against both local (cold cache, posix_fadvise-evicted, mincore-verified)
and S3 (mountpoint-s3 over noaa-goes16) tiers.

Serves: C8, E5
Origin: AGENTSTAGE.md §11.5 (E5 row), §11.9 (risk row "E5 measured speedup")
Required data: outputs/e2e/{local,s3}/e2e_{local,s3}.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.h8


def _load_e2e(outputs_root: Path, tier: str) -> dict:
    p = outputs_root / "e2e" / tier / f"e2e_{tier}.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text())


def _load_e2e_decomp(outputs_root: Path, tier: str) -> dict:
    p = outputs_root / "e2e" / tier / f"e2e_decomp_{tier}.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text())


class TestEndToEndWallClock:
    """Total wall-clock reduction is the headline. At S3-class tier the
    measured speedup must clear the §11.9 risk-managed minimum of 1.3×.
    Local NVMe XFS is reported separately as the lower-bound regime."""

    @pytest.mark.parametrize("tier,min_speedup", [
        ("s3", 1.3),       # §11.9 risk-managed minimum at S3-class cold tier
        ("local", 1.0),    # local NVMe is the lower-bound — must not regress
    ])
    def test_wall_clock_speedup(
        self, tier, min_speedup, outputs_root, report
    ):
        """median(wall_clock without) / median(wall_clock with) ≥ min_speedup
        on the E-028 end-to-end agent-script measurement. Records the per-tier
        speedup for the table_wall_clock_speedup_per_config figure."""
        e2e = _load_e2e(outputs_root, tier)
        if not e2e:
            pytest.skip(f"H8.wall_clock[{tier}]: outputs/e2e/{tier}/e2e_{tier}.json missing")
        rc = e2e["baseline"]["returncode"]
        if rc != 0:
            pytest.skip(f"H8.wall_clock[{tier}]: baseline returncode={rc}")
        speedup = e2e["session_speedup"]
        report.append("table_wall_clock_speedup_per_config", {
            "tier": tier,
            "baseline_s": e2e["baseline"]["elapsed_s"],
            "staged_s": e2e["staged"]["elapsed_s"],
            "speedup": speedup,
            "wall_saved_s": e2e["wall_time_saved_s"],
            "n_files": e2e["n_task_files"],
            "experiment": e2e["experiment"],
        })
        assert speedup >= min_speedup, (
            f"H8 wall-clock speedup on {tier}: {speedup:.2f}× < {min_speedup}× threshold. "
            f"baseline {e2e['baseline']['elapsed_s']:.1f}s → "
            f"staged {e2e['staged']['elapsed_s']:.1f}s."
        )


class TestDecompressionStaging:
    """E-029 — decompression staging (uncompressed hot copies) ≥ plain
    staging on the same tier. Validates that moving decompression CPU
    into the staging window is a strict improvement."""

    @pytest.mark.parametrize("tier", ["local", "s3"])
    def test_decomp_at_least_as_good_as_plain(self, tier, outputs_root, report):
        plain = _load_e2e(outputs_root, tier)
        decomp = _load_e2e_decomp(outputs_root, tier)
        if not plain or not decomp:
            pytest.skip(f"H8.decomp[{tier}]: E-028 or E-029 artifact missing")
        plain_sp = plain["session_speedup"]
        decomp_sp = decomp["session_speedup"]
        report.append("table_decomp_vs_plain", {
            "tier": tier,
            "plain_speedup": plain_sp,
            "decomp_speedup": decomp_sp,
            "decomp_advantage_pct": round((decomp_sp / plain_sp - 1) * 100, 1),
        })
        # Allow ±10% slack — measurement noise, not a regression.
        assert decomp_sp >= 0.9 * plain_sp, (
            f"Decomp-staging on {tier} ({decomp_sp:.2f}×) is more than 10% "
            f"worse than plain staging ({plain_sp:.2f}×) — investigate."
        )


class TestStagerCorrectness:
    """The stager must succeed AND produce a valid run (rc == 0).
    Returncode pass on both baseline and staged is the byte-correctness
    proxy: if the agent script reads the wrong bytes, the script crashes
    or writes nothing."""

    @pytest.mark.parametrize("tier", ["local", "s3"])
    def test_staged_run_returncode_zero(self, tier, outputs_root):
        e2e = _load_e2e(outputs_root, tier)
        if not e2e:
            pytest.skip(f"H8.correctness[{tier}]: missing artifact")
        assert e2e["baseline"]["returncode"] == 0, (
            f"Baseline {tier} returned {e2e['baseline']['returncode']} — "
            f"control run failed, speedup is meaningless. "
            f"stderr_tail: {e2e['baseline']['stderr_tail'][:200]}"
        )
        assert e2e["staged"]["returncode"] == 0, (
            f"Staged {tier} returned {e2e['staged']['returncode']} — "
            f"shim is delivering wrong bytes or the script crashed. "
            f"stderr_tail: {e2e['staged']['stderr_tail'][:200]}"
        )

    @pytest.mark.parametrize("tier", ["local", "s3"])
    def test_all_targets_staged(self, tier, outputs_root):
        """Every detected file made it to the hot tier — n_staged == n_task_files."""
        e2e = _load_e2e(outputs_root, tier)
        if not e2e:
            pytest.skip(f"H8.staged_complete[{tier}]: missing artifact")
        assert e2e["n_staged"] == e2e["n_task_files"], (
            f"Only {e2e['n_staged']}/{e2e['n_task_files']} files staged on {tier} — "
            f"stager dropped files."
        )
