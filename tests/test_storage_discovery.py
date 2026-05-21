"""Tests for storage discovery + tier probing."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from agentstage.stager.discovery import (
    MountEntry,
    StorageHierarchy,
    StorageTier,
    _classify_device,
    discover_candidate_tiers,
    parse_proc_mounts,
    probe_tier,
    save_hierarchy,
    load_cached_hierarchy,
)


def test_parse_proc_mounts_returns_records():
    """On Linux, /proc/mounts always has at least the root filesystem."""
    mounts = parse_proc_mounts()
    assert len(mounts) > 0
    # Find the root mount
    roots = [m for m in mounts if m.mount == "/"]
    assert len(roots) >= 1
    assert roots[0].fs_type != ""


def test_classify_device_tmpfs():
    """tmpfs maps to 'memory'."""
    e = MountEntry(device="tmpfs", mount="/dev/shm", fs_type="tmpfs", opts="rw")
    assert _classify_device(e) == "memory"


def test_classify_device_nfs():
    """NFS maps to 'network'."""
    e = MountEntry(device="server:/exp", mount="/mnt/nfs", fs_type="nfs4", opts="rw")
    assert _classify_device(e) == "network"


def test_classify_device_unknown_fallback():
    """Unknown fs types without /sys/block info return 'unknown'."""
    e = MountEntry(device="/dev/imaginary99", mount="/x",
                   fs_type="exotic_fs", opts="rw")
    assert _classify_device(e) == "unknown"


def test_discover_candidate_tiers_finds_tmpdir_or_shm():
    """Discovery should find at least one tmpfs-class mount on Linux test hosts."""
    tiers = discover_candidate_tiers()
    assert len(tiers) > 0
    # At least one usable mount should be writable
    for entry, _cls in tiers:
        assert os.access(entry.mount, os.W_OK)


def test_probe_tier_on_tmpfs():
    """Probing /dev/shm (if writable) should return a finite first_byte_ms."""
    if not os.access("/dev/shm", os.W_OK):
        pytest.skip("/dev/shm not writable")
    tier = probe_tier("/dev/shm", probe_size_mb=1,
                      fs_type="tmpfs", device_class="memory")
    assert tier is not None
    assert tier.first_byte_ms >= 0
    assert tier.throughput_mbps > 0
    # tmpfs should be fast — first byte under 5 ms in practice
    assert tier.first_byte_ms < 50.0, f"tmpfs first_byte_ms={tier.first_byte_ms}"


def test_storage_hierarchy_sorting():
    """tiers list is interpreted hot-to-cold by first_byte_ms ascending."""
    tiers = [
        StorageTier(mount="/cold", fs_type="nfs", device_class="network",
                    free_bytes=10**12, first_byte_ms=20.0,
                    throughput_mbps=100.0, write_ms=15.0),
        StorageTier(mount="/hot", fs_type="tmpfs", device_class="memory",
                    free_bytes=10**9, first_byte_ms=0.05,
                    throughput_mbps=5000.0, write_ms=0.05),
        StorageTier(mount="/warm", fs_type="xfs", device_class="nvme",
                    free_bytes=10**11, first_byte_ms=0.5,
                    throughput_mbps=1500.0, write_ms=0.4),
    ]
    # Caller normally sorts; we simulate that
    tiers.sort(key=lambda t: t.first_byte_ms)
    hier = StorageHierarchy(tiers=tiers)
    assert hier.hot.mount == "/hot"
    assert hier.cold.mount == "/cold"
    assert len(hier.warm) == 1
    assert hier.warm[0].mount == "/warm"


def test_tier_for_path_longest_prefix():
    """tier_for_path returns the longest matching mount prefix."""
    tiers = [
        StorageTier(mount="/data", fs_type="nfs", device_class="network",
                    free_bytes=10**12, first_byte_ms=20.0,
                    throughput_mbps=100.0, write_ms=15.0),
        StorageTier(mount="/data/staging", fs_type="xfs", device_class="nvme",
                    free_bytes=10**11, first_byte_ms=0.5,
                    throughput_mbps=1500.0, write_ms=0.4),
    ]
    hier = StorageHierarchy(tiers=tiers)
    # /data/staging/x.csv should match /data/staging (longer prefix), not /data
    t = hier.tier_for_path("/data/staging/x.csv")
    assert t.mount == "/data/staging"
    t = hier.tier_for_path("/data/raw/y.nc")
    assert t.mount == "/data"


def test_should_stage_rejects_hot_files():
    """If the file is already on the hot tier, should_stage returns False."""
    tiers = [
        StorageTier(mount="/hot", fs_type="tmpfs", device_class="memory",
                    free_bytes=10**9, first_byte_ms=0.05,
                    throughput_mbps=5000.0, write_ms=0.05),
        StorageTier(mount="/cold", fs_type="nfs", device_class="network",
                    free_bytes=10**12, first_byte_ms=200.0,
                    throughput_mbps=100.0, write_ms=15.0),
    ]
    hier = StorageHierarchy(tiers=tiers)
    decision, reason = hier.should_stage("/hot/foo.csv", 1_000_000)
    assert decision is False
    assert "hot tier" in reason


def test_should_stage_rejects_when_cost_exceeds_slack():
    """If estimated stage cost exceeds slack window, refuse."""
    tiers = [
        StorageTier(mount="/hot", fs_type="tmpfs", device_class="memory",
                    free_bytes=10**9, first_byte_ms=0.05,
                    throughput_mbps=5000.0, write_ms=0.05),
        StorageTier(mount="/cold", fs_type="nfs", device_class="network",
                    free_bytes=10**12, first_byte_ms=200.0,
                    throughput_mbps=10.0, write_ms=15.0),
    ]
    hier = StorageHierarchy(tiers=tiers)
    # 100 MB file on a 10 MB/s tier = ~10 s. Slack window 2 s → refuse.
    decision, reason = hier.should_stage("/cold/big.nc",
                                          byte_estimate=100 * 1024 * 1024,
                                          slack_ms=2000.0)
    assert decision is False
    assert "stage cost" in reason


def test_should_stage_accepts_when_cost_fits():
    """If cost ≤ slack window, accept."""
    tiers = [
        StorageTier(mount="/hot", fs_type="tmpfs", device_class="memory",
                    free_bytes=10**9, first_byte_ms=0.05,
                    throughput_mbps=5000.0, write_ms=0.05),
        StorageTier(mount="/cold", fs_type="nfs", device_class="network",
                    free_bytes=10**12, first_byte_ms=20.0,
                    throughput_mbps=200.0, write_ms=15.0),
    ]
    hier = StorageHierarchy(tiers=tiers)
    # 5 MB file on a 200 MB/s tier = ~25 ms. Slack window 1000 ms → accept.
    decision, reason = hier.should_stage("/cold/small.nc",
                                          byte_estimate=5 * 1024 * 1024,
                                          slack_ms=1000.0)
    assert decision is True


def test_hierarchy_cache_roundtrip(tmp_path: Path):
    """save_hierarchy + load_cached_hierarchy roundtrip preserves tiers."""
    tiers = [
        StorageTier(mount="/hot", fs_type="tmpfs", device_class="memory",
                    free_bytes=10**9, first_byte_ms=0.05,
                    throughput_mbps=5000.0, write_ms=0.05),
    ]
    hier = StorageHierarchy(tiers=tiers)
    save_hierarchy(hier, tmp_path)
    loaded = load_cached_hierarchy(tmp_path)
    assert loaded is not None
    assert len(loaded.tiers) == 1
    assert loaded.tiers[0].mount == "/hot"
    assert loaded.tiers[0].first_byte_ms == pytest.approx(0.05)
