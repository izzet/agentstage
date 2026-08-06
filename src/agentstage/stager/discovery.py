"""Self-configuring storage-tier discovery for the stager.

Replaces the hard-coded `hot_root` / `cold_roots` configuration with
runtime discovery: walk `/proc/mounts`, classify each writable
filesystem (tmpfs / nvme / sata-ssd / hdd / network-fs / object-store),
probe each one for first-byte latency + sequential throughput, then
rank by measured latency to assemble a `StorageHierarchy`.

The discovered profile is cached per-host under
`outputs/storage_profile_<hostname>.json` to amortize probe cost across
runs. The cache TTL is 7 days (regenerated if older).

Tier assignment follows the target-set-size policy in the detector.

Note: discovery is opt-in. If the user sets `AGENTSTAGE_HOT_ROOT` and
`AGENTSTAGE_COLD_ROOTS` explicitly, those override the discovered values.
This preserves backward compatibility with existing scripts.
"""

from __future__ import annotations

import json
import os
import platform
import re
import socket
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Mount discovery
# ---------------------------------------------------------------------------

_MOUNT_RE = re.compile(
    r"^(?P<device>\S+)\s+(?P<mount>\S+)\s+(?P<fs_type>\S+)\s+(?P<opts>\S+)"
)

# fs_type → device-class heuristic. tmpfs is RAM; squashfs/proc/sys are read-only
# pseudo-fs and skipped; fuse mounts are object/network depending on the device
# string; nfs/cifs/smb are network; everything else inspected via /sys/block.
_FS_TYPE_CLASS: dict[str, str] = {
    "tmpfs": "memory",
    "ramfs": "memory",
    "devtmpfs": "memory",
    "nfs": "network",
    "nfs4": "network",
    "cifs": "network",
    "smb3": "network",
    "smbfs": "network",
    "lustre": "network",
    "gpfs": "network",
    "9p": "network",
    "ceph": "network",
    "fuse.s3fs": "object",
    "fuse.mountpoint-s3": "object",
    "fuse.goofys": "object",
    "fuse": "object",  # heuristic: most user-space fuse is object-ish
}

_SKIP_FS_TYPES = {
    "proc", "sysfs", "devpts", "tracefs", "debugfs", "cgroup", "cgroup2",
    "securityfs", "pstore", "autofs", "binfmt_misc", "configfs", "fusectl",
    "hugetlbfs", "mqueue", "nsfs", "selinuxfs", "bpf", "rpc_pipefs",
    "squashfs", "overlay", "overlayfs",
}


@dataclass(frozen=True)
class MountEntry:
    device: str
    mount: str
    fs_type: str
    opts: str


def parse_proc_mounts(path: str = "/proc/mounts") -> list[MountEntry]:
    """Parse /proc/mounts into MountEntry records. Linux-only."""
    out: list[MountEntry] = []
    try:
        with open(path) as f:
            for line in f:
                m = _MOUNT_RE.match(line)
                if not m:
                    continue
                out.append(MountEntry(
                    device=m["device"],
                    mount=m["mount"],
                    fs_type=m["fs_type"],
                    opts=m["opts"],
                ))
    except OSError:
        pass
    return out


def _classify_device(entry: MountEntry) -> str:
    """Return a device-class label: memory | nvme | ssd | hdd | network | object | unknown."""
    if entry.fs_type in _FS_TYPE_CLASS:
        return _FS_TYPE_CLASS[entry.fs_type]
    # Fall through: inspect /sys/block for rotational/non-rotational
    dev_basename = entry.device.split("/")[-1].rstrip("0123456789")
    if not dev_basename:
        return "unknown"
    rotational_path = Path(f"/sys/block/{dev_basename}/queue/rotational")
    if rotational_path.is_file():
        try:
            val = rotational_path.read_text().strip()
            if val == "0":
                # Non-rotational — could be SSD or NVMe. NVMe device names start with nvme.
                if dev_basename.startswith("nvme"):
                    return "nvme"
                return "ssd"
            else:
                return "hdd"
        except OSError:
            pass
    return "unknown"


def _writable_and_large_enough(mount: str, min_free_bytes: int = 256 * 1024 * 1024) -> bool:
    """Skip mounts we can't write to or that are too small to be useful."""
    if not os.access(mount, os.W_OK):
        return False
    try:
        info = os.statvfs(mount)
    except OSError:
        return False
    free_bytes = info.f_bavail * info.f_bsize
    return free_bytes >= min_free_bytes


def discover_candidate_tiers(
    *,
    skip_paths: tuple[str, ...] = ("/proc", "/sys", "/boot", "/snap",
                                   "/var/lib/docker", "/var/lib/containers"),
    skip_exact: tuple[str, ...] = ("/dev", "/dev/pts", "/dev/mqueue",
                                   "/dev/hugepages", "/run", "/run/lock"),
) -> list[tuple[MountEntry, str]]:
    """Return [(MountEntry, device_class), ...] for usable mounts.

    Filters out:
      - read-only or unwritable mounts
      - mounts with < 256 MB free
      - system pseudo-filesystems (proc, sysfs, etc.)
      - virtualization overlay mounts (docker layers, snap)
    """
    out: list[tuple[MountEntry, str]] = []
    seen: set[str] = set()  # dedupe by mount point
    for entry in parse_proc_mounts():
        if entry.fs_type in _SKIP_FS_TYPES:
            continue
        if entry.mount in skip_exact:
            continue
        if any(entry.mount.startswith(skip + "/") or entry.mount == skip
               for skip in skip_paths):
            continue
        if entry.mount in seen:
            continue
        if not _writable_and_large_enough(entry.mount):
            continue
        seen.add(entry.mount)
        out.append((entry, _classify_device(entry)))
    return out


# ---------------------------------------------------------------------------
# Tier profiling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StorageTier:
    """One discovered + profiled storage tier."""
    mount: str
    fs_type: str
    device_class: str
    free_bytes: int
    first_byte_ms: float
    throughput_mbps: float
    write_ms: float


def _evict_cache(path: str) -> None:
    """Best-effort drop of OS page cache for a single file."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        os.sync()
    except OSError:
        pass


def _find_existing_file(mount: str, min_size: int = 4096) -> Path | None:
    """Find any existing regular file under `mount` with size >= min_size.
    Used for read-only probing of mounts we cannot write to (e.g., S3,
    read-only datasets). Walks at most a few directories to keep startup
    fast."""
    root = Path(mount)
    if not root.is_dir():
        return None
    # Breadth-first walk capped at ~50 directories to avoid scanning huge trees
    queue: list[Path] = [root]
    visited = 0
    while queue and visited < 50:
        d = queue.pop(0)
        visited += 1
        try:
            for entry in d.iterdir():
                try:
                    if entry.is_file():
                        sz = entry.stat().st_size
                        if sz >= min_size:
                            return entry
                    elif entry.is_dir() and not entry.is_symlink():
                        queue.append(entry)
                except OSError:
                    continue
        except (OSError, PermissionError):
            continue
    return None


def _probe_read_only(mount: str, fs_type: str, device_class: str) -> StorageTier | None:
    """Read-only probe for mounts we cannot write to (S3, read-only datasets).

    Finds an existing file, evicts, times first-byte read + sequential
    throughput. Skips the write timing (set to NaN). free_bytes reflects
    *total* free even though we won't be writing.
    """
    sample = _find_existing_file(mount)
    if sample is None:
        return None
    try:
        _evict_cache(str(sample))
        t0 = time.monotonic_ns()
        with open(sample, "rb") as f:
            f.read(4096)
        first_byte_ms = (time.monotonic_ns() - t0) / 1e6

        # Throughput probe over up to 16 MB of the same file
        _evict_cache(str(sample))
        target_bytes = min(sample.stat().st_size, 16 * 1024 * 1024)
        t0 = time.monotonic_ns()
        with open(sample, "rb") as f:
            read_so_far = 0
            while read_so_far < target_bytes:
                chunk = f.read(min(1024 * 1024, target_bytes - read_so_far))
                if not chunk:
                    break
                read_so_far += len(chunk)
        seq_ns = time.monotonic_ns() - t0
        throughput_mbps = (read_so_far / 1024 / 1024) / (seq_ns / 1e9) if seq_ns > 0 else 0.0

        try:
            info = os.statvfs(mount)
            free_bytes = info.f_bavail * info.f_bsize
        except OSError:
            free_bytes = 0

        return StorageTier(
            mount=mount,
            fs_type=fs_type,
            device_class=device_class,
            free_bytes=free_bytes,
            first_byte_ms=first_byte_ms,
            throughput_mbps=throughput_mbps,
            write_ms=float("nan"),  # read-only — write_ms is not measured
        )
    except OSError:
        return None


def probe_tier(
    mount: str,
    *,
    probe_size_mb: int = 16,
    fs_type: str = "",
    device_class: str = "",
) -> StorageTier | None:
    """Write a probe file, evict, time the first-byte read + sequential
    bandwidth, clean up. Returns None on any failure (mount is rejected).

    `probe_size_mb` defaults to 16 (small enough for tmpfs, large enough
    to amortize first-byte cost when computing throughput). For slow
    tiers (FUSE-S3) this may take seconds.

    If writing fails (read-only mount), automatically falls back to
    read-only probing via _probe_read_only which uses an existing file.
    """
    probe_dir = Path(mount) / ".agentstage_probe"
    try:
        probe_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Read-only mount or no permission to create dir at this level.
        # Fall straight to read-only probing.
        return _probe_read_only(mount, fs_type, device_class)
    probe_path = probe_dir / f"probe_{uuid.uuid4().hex}.bin"
    payload_bytes = probe_size_mb * 1024 * 1024

    try:
        # Write the probe
        write_start = time.monotonic_ns()
        with open(probe_path, "wb") as f:
            f.write(os.urandom(payload_bytes))
            f.flush()
            os.fsync(f.fileno())
        write_ms = (time.monotonic_ns() - write_start) / 1e6

        # First-byte read (after eviction)
        _evict_cache(str(probe_path))
        t0 = time.monotonic_ns()
        with open(probe_path, "rb") as f:
            f.read(4096)
        first_byte_ms = (time.monotonic_ns() - t0) / 1e6

        # Sequential throughput (after eviction)
        _evict_cache(str(probe_path))
        t0 = time.monotonic_ns()
        with open(probe_path, "rb") as f:
            total = 0
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
        seq_ns = time.monotonic_ns() - t0
        throughput_mbps = (total / 1024 / 1024) / (seq_ns / 1e9) if seq_ns > 0 else 0.0

        # Free space
        info = os.statvfs(mount)
        free_bytes = info.f_bavail * info.f_bsize

        return StorageTier(
            mount=mount,
            fs_type=fs_type,
            device_class=device_class,
            free_bytes=free_bytes,
            first_byte_ms=first_byte_ms,
            throughput_mbps=throughput_mbps,
            write_ms=write_ms,
        )
    except OSError:
        # Could not write — fall back to read-only probing
        try:
            probe_path.unlink(missing_ok=True)
            probe_dir.rmdir()
        except OSError:
            pass
        return _probe_read_only(mount, fs_type, device_class)
    finally:
        try:
            probe_path.unlink(missing_ok=True)
            probe_dir.rmdir()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------


@dataclass
class StorageHierarchy:
    """All discovered tiers sorted ascending by first_byte_ms (hot → cold)."""
    tiers: list[StorageTier] = field(default_factory=list)

    @property
    def hot(self) -> StorageTier | None:
        return self.tiers[0] if self.tiers else None

    @property
    def cold(self) -> StorageTier | None:
        return self.tiers[-1] if self.tiers else None

    @property
    def warm(self) -> list[StorageTier]:
        return self.tiers[1:-1] if len(self.tiers) > 2 else []

    def tier_for_path(self, path: str) -> StorageTier | None:
        """Return the tier that contains `path`, or None if no match."""
        # Find the LONGEST mount prefix matching the path
        best: StorageTier | None = None
        best_len = -1
        for t in self.tiers:
            if path.startswith(t.mount.rstrip("/") + "/") or path == t.mount:
                if len(t.mount) > best_len:
                    best = t
                    best_len = len(t.mount)
        return best

    def stage_cost_ms(self, tier: StorageTier, byte_estimate: int) -> float:
        """Estimate stage time from `tier` (source) into the hot tier.

        cost = first_byte_ms_of_src + (bytes / throughput_of_src)
        Assumes the destination tier (hot) is fast enough that the write
        is dominated by the read from cold.
        """
        bytes_mb = byte_estimate / 1024 / 1024
        if tier.throughput_mbps <= 0:
            return float("inf")
        return tier.first_byte_ms + (bytes_mb / tier.throughput_mbps) * 1000.0

    def should_stage(
        self,
        file_path: str,
        byte_estimate: int,
        slack_ms: float | None = None,
    ) -> tuple[bool, str]:
        """Decide whether `file_path` is worth staging.

        Returns (decision, reason). False decisions include:
          - file is already on the hot tier (waste)
          - estimated stage cost exceeds slack window
          - no hot tier to stage into
        """
        if not self.tiers:
            return False, "no tiers discovered"
        src = self.tier_for_path(file_path)
        if src is None:
            # Don't know which tier — be conservative and stage
            return True, "unknown source tier; staging by default"
        if src is self.hot:
            return False, f"already on hot tier ({src.mount})"
        if slack_ms is not None:
            cost = self.stage_cost_ms(src, byte_estimate)
            if cost > slack_ms:
                return False, (
                    f"stage cost {cost:.0f} ms > slack window {slack_ms:.0f} ms"
                )
        return True, "ok"

    def to_dict(self) -> dict:
        return {"tiers": [asdict(t) for t in self.tiers]}


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

_CACHE_TTL_S = 7 * 24 * 3600


def _cache_path(cache_dir: Path) -> Path:
    host = socket.gethostname() or "unknown"
    return cache_dir / f"storage_profile_{host}.json"


def load_cached_hierarchy(cache_dir: Path) -> StorageHierarchy | None:
    """Load a cached hierarchy if the file exists and is fresh."""
    p = _cache_path(cache_dir)
    if not p.is_file():
        return None
    try:
        age = time.time() - p.stat().st_mtime
        if age > _CACHE_TTL_S:
            return None
        data = json.loads(p.read_text())
        tiers = [StorageTier(**t) for t in data.get("tiers", [])]
        return StorageHierarchy(tiers=tiers)
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def save_hierarchy(hierarchy: StorageHierarchy, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = _cache_path(cache_dir)
    p.write_text(json.dumps({
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "tiers": [asdict(t) for t in hierarchy.tiers],
    }, indent=2))
    return p


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def discover_and_probe(
    *,
    cache_dir: Path = Path("outputs"),
    use_cache: bool = True,
    probe_size_mb: int = 16,
    extra_paths: Iterable[str] = (),
) -> StorageHierarchy:
    """Build a StorageHierarchy by discovering + probing tiers.

    Args:
      cache_dir: where to read/write the cached profile.
      use_cache: if True (default) and a fresh cache exists, return it.
      probe_size_mb: probe payload size. Default 16 MB.
      extra_paths: additional mount points to probe explicitly (e.g. an
        S3 mount that's mounted into a non-/proc/mounts-visible location).
    """
    if use_cache:
        cached = load_cached_hierarchy(cache_dir)
        if cached is not None:
            return cached

    profiles: list[StorageTier] = []
    seen_mounts: set[str] = set()
    for entry, dev_class in discover_candidate_tiers():
        if entry.mount in seen_mounts:
            continue
        prof = probe_tier(entry.mount, probe_size_mb=probe_size_mb,
                          fs_type=entry.fs_type, device_class=dev_class)
        if prof is not None:
            profiles.append(prof)
            seen_mounts.add(entry.mount)

    for extra_mount in extra_paths:
        if extra_mount in seen_mounts:
            continue
        # Skip the writable check for extra_paths — caller is explicitly
        # asking us to probe these (typical case: a read-only cold tier
        # like an S3 mount). probe_tier handles the read-only fallback.
        if not Path(extra_mount).is_dir():
            continue
        prof = probe_tier(extra_mount, probe_size_mb=probe_size_mb,
                          fs_type="extra", device_class="unknown")
        if prof is not None:
            profiles.append(prof)
            seen_mounts.add(extra_mount)

    profiles.sort(key=lambda p: p.first_byte_ms)
    hierarchy = StorageHierarchy(tiers=profiles)
    save_hierarchy(hierarchy, cache_dir)
    return hierarchy
