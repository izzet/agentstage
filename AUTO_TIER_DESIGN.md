# Auto Tier + Bandwidth Detection — Design Sketch

> Self-configuring stager: discovers storage tiers at startup, probes
> their latency/throughput, and decides dynamically per file whether
> staging is worth it.
>
> Status: design sketch (not implemented). Companion to the auto-rules
> work in `auto_rules.py`. Both share the same philosophy: replace
> hard-coded operator knowledge with measured behavior so the system
> generalizes to new environments without expert configuration.

---

## Why this matters for the paper

Today the stager has two hard-coded knobs:

```python
hot_root   = "/dev/shm/agentstage_path_b"     # tmpfs
cold_roots = ["/tmp/s3-noaa-goes16/ABI-L2-CMIPC"]  # FUSE-S3 mount
```

A reviewer can ask:

> "How does AgentStage decide what to call 'hot' vs 'cold'? What if my
> cluster has $HOME (NFS), $SCRATCH (Lustre), a local NVMe, AND `/dev/shm`?
> What if my 'cold tier' is actually fast enough that staging adds
> overhead instead of saving time?"

Today the answer is "the operator picks". That's a real configuration
burden and a real correctness risk (staging a hot file is pure waste).
Auto-tier detection makes the answer:

> "AgentStage probes each mounted filesystem at startup, ranks them by
> measured first-byte latency and sustained bandwidth, and chooses
> stage source/destination per file based on file size, available
> slack window, and the measured tier gap."

That's a defensible systems contribution: removes operator knobs,
generalizes across HPC sites, and avoids the "stage a hot file"
correctness bug.

---

## Algorithm

### Phase 1 — Discovery (startup, one-shot)

Walk the filesystem hierarchy and identify every candidate storage tier:

```python
def discover_storage_tiers() -> list[StorageTier]:
    candidates = []
    # 1. Read /proc/mounts to find all mounted filesystems
    for mount in parse_mounts():
        # 2. statvfs(mount) → fs type, total/free bytes, block size
        info = os.statvfs(mount.path)
        # 3. lsblk / sysfs → device type (rotational? SSD? NVMe? tmpfs? network?)
        device_type = classify_device(mount.device)
        # 4. Filter: skip mounts unsuitable for staging
        #    - read-only mounts
        #    - mounts <1 GB free
        #    - mounts the user can't write to
        if usable(mount, info):
            candidates.append(StorageTier(
                path=mount.path,
                fs_type=mount.fs_type,
                device_type=device_type,
                free_bytes=info.f_bavail * info.f_bsize,
            ))
    return candidates
```

Typical Ares result:
```
[
  StorageTier(path="/dev/shm", fs_type="tmpfs",  device_type="memory"),
  StorageTier(path="/scratch", fs_type="xfs",    device_type="nvme"),
  StorageTier(path="/data",    fs_type="nfs4",   device_type="network"),
  StorageTier(path="/tmp/s3-*", fs_type="fuse",  device_type="object"),
]
```

### Phase 2 — Probing (startup, < 1 second total)

For each discovered tier, run two micro-probes:

```python
def probe_tier(tier: StorageTier, probe_size_mb: int = 50) -> TierProfile:
    # Pick a probe file: use a temporary file we control
    probe_path = tier.path / f".agentstage_probe_{uuid()}"
    payload = os.urandom(probe_size_mb * 1024 * 1024)

    # Write the probe (best-effort timing; not the metric we care about)
    write_start = time.monotonic_ns()
    with open(probe_path, "wb") as f:
        f.write(payload)
    os.fsync(...)
    write_ms = (time.monotonic_ns() - write_start) / 1e6

    # Evict, then probe FIRST-BYTE LATENCY (4 KB cold read)
    evict_cache(probe_path)
    t0 = time.monotonic_ns()
    with open(probe_path, "rb") as f:
        f.read(4096)
    first_byte_ms = (time.monotonic_ns() - t0) / 1e6

    # Probe SEQUENTIAL THROUGHPUT (re-read the full payload)
    evict_cache(probe_path)
    t0 = time.monotonic_ns()
    with open(probe_path, "rb") as f:
        while f.read(1024 * 1024):
            pass
    throughput_mbps = probe_size_mb / ((time.monotonic_ns() - t0) / 1e9)

    os.remove(probe_path)
    return TierProfile(
        tier=tier,
        first_byte_ms=first_byte_ms,
        throughput_mbps=throughput_mbps,
        write_ms=write_ms,
    )
```

Expected values (Ares):
- tmpfs:    first_byte ≈ 0.05 ms, throughput ≈ 5000 MB/s
- NVMe:     first_byte ≈ 0.5 ms,  throughput ≈ 2000 MB/s
- NFS:      first_byte ≈ 5-20 ms, throughput ≈ 100-500 MB/s
- FUSE-S3:  first_byte ≈ 500-800 ms, throughput ≈ 5-50 MB/s

### Phase 3 — Hierarchy (one-shot after probing)

Sort tiers by `first_byte_ms` ascending. The fastest is `hot`, the
slowest is `cold`. Anything in between is a candidate intermediate
("warm").

```python
def build_hierarchy(profiles: list[TierProfile]) -> StorageHierarchy:
    sorted_p = sorted(profiles, key=lambda p: p.first_byte_ms)
    return StorageHierarchy(
        hot=sorted_p[0],
        warm=sorted_p[1:-1],
        cold=sorted_p[-1],
    )
```

Replaces the hard-coded `hot_root` / `cold_roots`.

### Phase 4 — Dynamic stage decision (per file)

Given a `DataHint` (file path, byte estimate, slack window), decide
whether to stage and to which destination tier:

```python
def should_stage(file_path: str, byte_estimate: int,
                slack_ms: float, hierarchy: StorageHierarchy) -> Decision:
    src_tier = identify_tier(file_path, hierarchy)
    if src_tier is hierarchy.hot:
        return Decision.SKIP  # already fastest; staging is waste
    # Estimated stage cost from src_tier to hot
    bytes_mb = byte_estimate / 1024 / 1024
    stage_cost_ms = (bytes_mb / src_tier.throughput_mbps) * 1000 + \
                    src_tier.first_byte_ms
    if stage_cost_ms > slack_ms:
        return Decision.SKIP  # can't finish in time anyway
    # Pick destination: hot if file fits in hot's free space, else warm
    if byte_estimate < 0.5 * hierarchy.hot.tier.free_bytes:
        return Decision.STAGE_TO(hierarchy.hot)
    elif hierarchy.warm:
        return Decision.STAGE_TO(hierarchy.warm[0])  # second-fastest
    else:
        return Decision.SKIP
```

This is the policy that the current hardcoded `Stager.prefetch` is
implicitly enforcing — but the current code can't reason about it
because no profiles exist.

---

## Reproducibility / paper hook

Per-experiment artifacts gain a new field:

```json
{
  "storage_profile": {
    "hot": {"path": "/dev/shm", "first_byte_ms": 0.05, "throughput_mbps": 5200},
    "warm": [{"path": "/scratch", "first_byte_ms": 0.6, "throughput_mbps": 1900}],
    "cold": {"path": "/tmp/s3-noaa-goes16/...", "first_byte_ms": 754.5, "throughput_mbps": 4.5}
  }
}
```

A reviewer can then read any experiment artifact and immediately see
*on what hardware* the speedup was measured. Reproducing on a different
machine produces a different profile and a different speedup — but the
methodology is identical.

---

## Scope of implementation

Estimated 1 day:

1. **`src/agentstage/stager/discovery.py`** (~150 LoC)
   - `discover_storage_tiers()` → walks /proc/mounts, classifies devices
   - `probe_tier()` → writes/evicts/reads a probe file
   - `StorageHierarchy` dataclass
   - Cache the profile in `outputs/storage_profile_<host>.json` so
     subsequent runs don't re-probe unnecessarily

2. **`Stager` constructor change** (~30 LoC)
   - Default `hot_root`/`cold_roots` to discovery results
   - Allow override via env vars (existing behavior preserved)

3. **`should_stage` decision** wired into `Stager.prefetch` (~50 LoC)
   - Reject DataHints that resolve to a file already on the hot tier
   - Reject DataHints whose stage cost > known slack window

4. **Tests** (~100 LoC)
   - Synthetic 3-tier scenario (tmpfs + 2 fake "slow" tiers via FUSE
     emulation or mount with artificial latency)
   - Verify hierarchy assembled correctly
   - Verify a hot-file hint is rejected (no waste staging)

5. **One paper figure**: per-tier first-byte distribution (CDF)
   measured on Ares + Delta + a third site if available, showing the
   automatic ranking.

---

## What it doesn't do (intentionally)

- **Cross-host migration**. Doesn't move files between hosts; only
  manages tiers within one node. Multi-node is future work.
- **Cache-coherence with writes**. Assumes data is read-mostly (the
  scientific-agent assumption). Write-through to cold is out of scope.
- **Bandwidth budgeting across concurrent stages**. The 3-MB-per-rule
  scenarios we measure don't need it; could be added if a workload's
  tier-3 dispatch produces GB-scale concurrent fetches.

---

## Connection to the rules-side auto-generation

Both pieces share a slogan: **measure rather than configure**.

- `auto_rules.py`: don't ask the operator to hand-tune rules; derive
  them from task spec + workspace prior.
- `auto_tier`: don't ask the operator to label hot/cold; derive ranking
  from measured first-byte latency.

Together they remove the two configuration burdens that would make
AgentStage hard to deploy on a new HPC site. The paper's contribution
shifts from "we built a clever staging system you'll need to tune" to
"we built a self-configuring staging system that adapts to the
deployment environment". Reviewer-defensible.
