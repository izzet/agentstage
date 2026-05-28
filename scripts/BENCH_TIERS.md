# Storage-tier bandwidth bench (`bench_tiers.sh`)

Single-node IOR sweep across all storage tiers available on Ares compute nodes,
plus optional `dd` cross-check baseline. Designed to be reproducible and
idempotent — re-running reuses an existing OrangeFS mount, IOR config, and
results skeleton.

Companion script `build_ior.sh` builds IOR from source against the OpenMPI
variant `orangefs/2.10` depends on, sidestepping an Lmod hierarchy conflict
that makes the Spack-provided `ior/3.3.0-fihmxeb` module unusable alongside
OrangeFS.

## Quick start

```bash
# One-time: build IOR into ~/.local/bin (clones, bootstrap, configure, install)
bash scripts/build_ior.sh

# Then in any salloc shell:
salloc -N 1 -t 02:00:00 --exclusive
bash scripts/bench_tiers.sh              # ~2 min, default mode
bash scripts/bench_tiers.sh --rigorous   # ~15-20 min, multi-task + dd baseline
```

Output lands in `outputs/bench_tiers/<timestamp>/`:
- `summary.csv` — one row per (tier, tool, op)
- `bench.log` — full timestamped log
- `ior_<tier>.txt` — per-tier raw IOR output
- `servers.txt` / `clients.txt` / `pvfs2-server.log` — OrangeFS deploy artefacts

## Tiers measured

| Name | Path | Backing | O_DIRECT |
|---|---|---|---|
| `tmpfs` | `/dev/shm/$USER/bench_tiers` | RAM | no (not applicable) |
| `local_nvme` | `/mnt/nvme/$USER/bench_tiers` | per-node NVMe | yes |
| `local_ssd` | `/mnt/ssd/$USER/bench_tiers` | per-node SATA SSD | yes |
| `shared_xfs` | `/mnt/common/$USER/bench_tiers` | network XFS export | yes |
| `orangefs` | `/mnt/ssd/$USER/orangefs/bench_tiers` | OrangeFS (FUSE) over `/mnt/nvme` | probed |

Subset via `TIERS="local_nvme orangefs" bash scripts/bench_tiers.sh`.

## Cold-cache discipline

- Every FS tier (all except tmpfs) opens with `--posix.odirect`, bypassing the
  kernel page cache for both write and read.
- `sync; echo 3 > /proc/sys/vm/drop_caches` is also attempted between phases as
  belt-and-suspenders. Without passwordless sudo this is a no-op and the script
  warns — O_DIRECT then carries the cold-cache guarantee on its own.
- tmpfs is RAM, so "cold cache" is undefined. The script runs the same IOR job
  without O_DIRECT and flags `odirect=no` in the summary.
- OrangeFS-via-FUSE O_DIRECT support is probed at runtime (a small aligned
  `dd oflag=direct` write); if the FUSE client rejects O_DIRECT the script
  drops back to non-O_DIRECT and relies on `drop_caches`.

## Modes

### Default (~2 min)
| Parameter | Value | Total I/O |
|---|---|---|
| `IOR_REPS` | 3 | |
| `IOR_BLOCK` | 2g | |
| `IOR_XFER` | 1m | |
| `IOR_TASKS` | 1 | ~60 GiB across the 5 tiers |
| `DD_REPS` | 0 (off) | |

Use for fast iteration / smoke. Single-thread numbers undersell tiers that
amortize per-call overhead with concurrency (NVMe writes, shared XFS, OrangeFS).

### `--rigorous` (~15-20 min)
| Parameter | Value | Total I/O |
|---|---|---|
| `IOR_REPS` | 5 | |
| `IOR_BLOCK` | 4g | |
| `IOR_XFER` | 1m | |
| `IOR_TASKS` | 4 (mpirun-launched) | ~800 GiB across the 5 tiers |
| `DD_REPS` | 3 | +30 GiB |

Adds two things over default:
1. **Multi-task IOR** — 4 MPI ranks per tier exposes per-request serialization
   that single-thread runs hide. Required to see actual NVMe / FUSE / network
   bandwidth ceilings.
2. **dd baseline** — 3 reps of 2 GiB single-thread `dd` with matching O_DIRECT
   flags before each IOR run. Reported alongside IOR plus a `dd_mean / ior_mean`
   ratio. Ratio ≈ 1.0 means single-thread is at the ceiling; ratio « 1.0 means
   the tier scales with concurrency.

### Custom
Any env var can override either mode:

```bash
IOR_TASKS=8 IOR_REPS=10 bash scripts/bench_tiers.sh --rigorous
TIERS="orangefs" IOR_BLOCK=8g bash scripts/bench_tiers.sh
```

## Idempotency

- **OrangeFS**: if the mount at `$ORANGEFS_MOUNT` is live (`pvfs2`/`orangefs`/
  `fuse.orangefs` fstype) the script reuses it. Otherwise it wipes
  `$STORAGE_BASE/orangefs/{data,metadata}` (pvfs2-server -f refuses to clobber
  existing storage) and runs `ares-orangefs-deploy`. The EXIT trap optionally
  tears down (`TEARDOWN=1`).
- **Config**: reuses `~/.orangefs.conf` if present; regenerate with
  `REBUILD_CONF=1`.
- **Modules**: `module purge` before loading, so leaked MODULEPATH state from
  the parent shell can't shadow the requested variants.
- **Results**: each run writes to a fresh timestamped directory; nothing is
  overwritten in place.

## Why a locally-built IOR?

The Spack-provided `ior/3.3.0-fihmxeb` lives at:

```
/mnt/repo/software/spack/spack/share/spack/lmod/.../openmpi/5.0.5-x2kvx5l/gcc/11.4.0/ior/3.3.0-fihmxeb
```

i.e. it's nested under the `openmpi/5.0.5-x2kvx5l` hierarchy. But
`orangefs/2.10` declares `depends_on("openmpi/5.0.5-cphqvsy")`. Loading
orangefs forces a version swap — Lmod logs `openmpi/5.0.5-x2kvx5l =>
openmpi/5.0.5-cphqvsy`, and ior becomes "Inactive" because its hierarchy is no
longer in `MODULEPATH`.

`build_ior.sh` resolves this by building IOR against `openmpi/5.0.5-cphqvsy`
directly, installing into `~/.local/bin/ior`. Now both orangefs and ior coexist
without a swap conflict, and `bench_tiers.sh` doesn't need to load the ior
module at all.

## Lmod gotchas (worth knowing)

Three traps tripped during development; all are now handled in the scripts but
worth knowing if you adapt them:

1. **Don't redirect stdout from `module`.** Lmod relies on the surrounding
   shell to `eval` the env mutations it prints. `module load X >>$LOG_FILE 2>&1`
   captures those mutations into the log file and they never apply. Use
   `module --redirect load X 2>>$LOG_FILE` instead — `--redirect` forces
   messages to stderr, keeping stdout clean for `eval`.
2. **`module purge` before any `module load`.** Ares login shells leak
   `openmpi/5.0.5-cphqvsy/gcc/11.4.0` into `MODULEPATH` even with empty
   `LOADEDMODULES`. Without purge, `module load openmpi/5.0.5-x2kvx5l` silently
   loads the cphqvsy variant instead.
3. **`set -uo pipefail` interacts with `if cmd1 | grep ...`.** If `cmd1` exits
   non-zero (e.g. `ior -h` exits 1 by design after printing help), pipefail
   makes the whole pipeline rc=1 and the `if` falls to the `else` branch even
   though grep matched. Either drop the pipe, capture into a variable first
   with `out=$(cmd1 2>&1 || true)`, or just check `[ -x "$binary" ]` as a
   simpler heuristic.

## Other operational gotchas

- **`~/.ssh/config` must be mode 600.** `ares-orangefs-deploy` shells into the
  local node via `parallel-ssh`; SSH refuses to start with `Bad owner or
  permissions on /home/$USER/.ssh/config` if the file is group/world readable.
- **Stale host keys.** If a node was reimaged the entry in `~/.ssh/known_hosts`
  may no longer match. Fix with `ssh-keygen -R <hostname>` then
  `ssh-keyscan -T 5 <hostname> >> ~/.ssh/known_hosts`.
- **`pvfs2-server -f` refuses non-empty storage dirs.** The script wipes
  `$STORAGE_BASE/orangefs/{data,metadata}` before deploy when no live mount
  exists. If `ares-orangefs-deploy` returns 0 but the mount isn't actually up,
  check `tail bench.log` for the pvfs2-server stderr.

## CSV schema

```
tier,tool,op,odirect,tasks,n_reps,xfer,block,max_mibps,min_mibps,mean_mibps,stdev_mibps,mean_s,test_file
```

- `tool`: `ior` or `dd`
- `op`: `write` or `read`
- `odirect`: `1` if `O_DIRECT` was set, `0` otherwise
- `tasks`: MPI rank count (1 for dd, $IOR_TASKS for ior)
- `mean_s`: mean wall time per rep (`-` for dd; not reported separately)

## Default-mode example output

```
tier        tool  write_MiB/s  read_MiB/s   odirect
tmpfs       ior   2059         4100         no
local_nvme  ior   1101         1246         yes
local_ssd   ior   465          501          yes
shared_xfs  ior   236          606          yes
orangefs    ior   265          566          yes
```

Single-thread numbers are bandwidth-floored by per-request serialization on
several tiers. Use `--rigorous` for true ceilings.

## `--rigorous` example output (4 tasks, dd cross-check)

```
tier         tool  write_MiB/s  read_MiB/s   odirect
local_nvme   dd     998         1240         yes
local_nvme   ior   1076         2330         yes
local_ssd    dd     427          500         yes
local_ssd    ior    503          532         yes
orangefs     dd     243          552         yes
orangefs     ior    789         1303         yes
shared_xfs   dd     154          616         yes
shared_xfs   ior    701         1091         yes
tmpfs        dd    1812         4164         no
tmpfs        ior   7906        12552         no

dd vs ior cross-check (ratio = dd_mean / ior_mean):
tier        op     dd_MiB/s     ior_MiB/s    ratio
local_nvme  write  998          1076         0.93   # at ceiling
local_nvme  read   1240         2330         0.53   # scales 2x with 4 threads
local_ssd   write  427          503          0.85   # at SATA ceiling
local_ssd   read   500          532          0.94   # at SATA ceiling
orangefs    write  243          789          0.31   # scales 3x — FUSE per-call serialization
orangefs    read   552          1303         0.42   # scales 2.4x
shared_xfs  write  154          701          0.22   # scales 4.5x — network FS amortizes
shared_xfs  read   616          1091         0.56   # scales 1.8x
tmpfs       write  1812         7906         0.23   # memcpy parallelizes across cores
tmpfs       read   4164         12552        0.33   # ditto
```

## Headline reading

- Ratios near 1.0 (local_ssd both ops; local_nvme write) → already at the
  device ceiling at 1 thread.
- Ratios well below 1.0 (orangefs, shared_xfs, tmpfs) → per-request /
  per-syscall serialization, not bandwidth, is the bottleneck. Concurrency
  scales these.
- OrangeFS at 1 thread (243/552 MiB/s) ≈ shared_xfs at 1 thread (154/616 MiB/s),
  despite radically different stacks (local FUSE-on-NVMe vs network XFS). At 4
  threads both pull away — but the single-stream parity is the relevant signal
  for agent-style workloads that issue one syscall at a time per process.
