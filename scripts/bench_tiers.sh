#!/usr/bin/env bash
# bench_tiers.sh - single-node bandwidth sweep across all storage tiers.
#
# Prereqs (just salloc, then run this — module loads are handled below):
#   bash scripts/build_ior.sh           # one-time: builds ior into ~/.local/bin
#   salloc -N 1 -t 02:00:00 --exclusive
#   bash scripts/bench_tiers.sh
#
# NOTE: We use a locally-built ior (against openmpi/5.0.5-cphqvsy, the variant
# orangefs/2.10 depends on) instead of the Spack `ior/3.3.0-fihmxeb` module.
# The Spack ior lives under the x2kvx5l openmpi hierarchy, but orangefs forces
# a swap to cphqvsy on load, which deactivates the Spack ior. Local build
# against cphqvsy makes both coexist cleanly.
#
# Tiers measured:
#   tmpfs       /dev/shm/$USER/bench_tiers          (RAM; cannot use O_DIRECT)
#   local_nvme  /mnt/nvme/$USER/bench_tiers         (per-node NVMe)
#   local_ssd   /mnt/ssd/$USER/bench_tiers          (per-node SATA SSD)
#   shared_xfs  /mnt/common/$USER/bench_tiers       (network-mounted XFS)
#   orangefs    /mnt/ssd/$USER/orangefs/bench_tiers (FUSE; deployed by this script)
#   s3          /tmp/s3-noaa-goes16                 (mountpoint-s3; read-only;
#                                                    same setup as path0_s3_run.sh
#                                                    for comparable numbers)
#
# Cold-cache discipline:
#   - Every tier (except tmpfs) opens with O_DIRECT, bypassing the kernel page
#     cache for both write and read. We also `echo 3 > /proc/sys/vm/drop_caches`
#     before each phase as belt-and-suspenders.
#   - tmpfs lives in RAM so "cold cache" is undefined; we run the same IOR job
#     with O_DIRECT disabled and flag the row in the summary.
#   - OrangeFS's FUSE client may or may not honor O_DIRECT; we probe at runtime
#     and fall back to non-O_DIRECT + drop_caches with a WARN in the summary.
#
# Idempotency: rerunning reuses an existing OrangeFS mount and config when
# possible; set TEARDOWN=1 to terminate OrangeFS on exit, REBUILD_CONF=1 to
# force config regeneration.
#
# Modes:
#   default        3 reps, 2 GiB block, 1 task, no dd baseline (~2 min)
#   --rigorous     5 reps, 4 GiB block, 4 MPI tasks, dd baseline x3 per tier
#                  (~15-25 min). Tighter stats + cross-tool validation.
#   --small-files  10 reps, 4 MiB block, 4 MiB xfer, 1 task (agent-scale).
#                  dd baseline x10 over 4 MiB files. Measures per-file
#                  bandwidth/latency for the agent access pattern (open small
#                  file, single-read, close) — different from rigorous's
#                  steady-state ceiling.
#   Pass via CLI flag or set RIGOROUS=1 / SMALL_FILES=1.
#
# Tuning (env vars, all optional — most overridden by --rigorous):
#   IOR_BLOCK     per-process block size      (default: 2g; rigorous: 4g)
#   IOR_XFER      transfer (request) size     (default: 1m)
#   IOR_REPS      repetitions per tier        (default: 3; rigorous: 5)
#   IOR_TASKS     MPI tasks                   (default: 1; rigorous: 4)
#   IOR_BIN       ior binary                  (default: ~/.local/bin/ior from build_ior.sh)
#   DD_REPS       dd baseline repetitions     (default: 0 = off; rigorous: 3)
#   DD_COUNT      dd block count (MiB blocks) (default: 2048 = 2 GiB)
#   MODULE_MPI    openmpi module              (default: openmpi/5.0.5-cphqvsy
#                                              -- matches orangefs/2.10's dep)
#   MODULE_ORANGEFS orangefs module           (default: orangefs/2.10)
#   SKIP_MODULES  1 to skip module loads      (default: 0)
#   TIERS         space-separated tier names  (default: all five)
#   RESULTS_DIR   output dir                  (default: $PWD/outputs/bench_tiers/<ts>)
#   STORAGE_BASE  orangefs backing storage    (default: /mnt/nvme/$USER)
#   ORANGEFS_MOUNT mount point                (default: /mnt/ssd/$USER/orangefs)
#   TCP_PORT      orangefs server port        (default: 3334)
#   TEARDOWN      1 to terminate orangefs     (default: 0; leaves it up)
#   REBUILD_CONF  1 to regen orangefs.conf    (default: 0)
#   S3_BUCKET     s3 bucket (no-sign-request) (default: noaa-goes16)
#   S3_REGION     s3 region                   (default: us-east-1)
#   S3_PREFIX     prefix under bucket          (default: ABI-L2-CMIPC/2024/122/00)
#   S3_MOUNT      mountpoint-s3 mount path    (default: /tmp/s3-noaa-goes16)
#   S3_FILE_GLOB glob for sample files       (default: OR_ABI-L2-CMIPC-M6C08_G16_*.nc)
#   S3_N_FILES    files to read per rep       (default: 5; matches path0_s3.py)
#   S3_REPS       s3 read reps                (default: IOR_REPS)
#   S3_TEARDOWN   1 to umount s3 on exit      (default: 0; leaves mount up)
#   MOUNT_S3_BIN  mount-s3 binary path        (default: ~/.local/bin/mount-s3)

set -uo pipefail

# ---------------------------------------------------------------------------
# CLI flag parsing (--rigorous)
# ---------------------------------------------------------------------------
RIGOROUS="${RIGOROUS:-0}"
SMALL_FILES="${SMALL_FILES:-0}"
for arg in "$@"; do
  case "$arg" in
    --rigorous|-r) RIGOROUS=1 ;;
    --small-files|-s) SMALL_FILES=1 ;;
    -h|--help)
      sed -n '2,65p' "$0"
      exit 0
      ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

if [ "$RIGOROUS" = "1" ] && [ "$SMALL_FILES" = "1" ]; then
  echo "error: --rigorous and --small-files are mutually exclusive" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Resolve repo root + config
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
USER="${USER:-$(id -un)}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"

RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/outputs/bench_tiers/$RUN_TS}"
mkdir -p "$RESULTS_DIR"
LOG_FILE="$RESULTS_DIR/bench.log"
SUMMARY_CSV="$RESULTS_DIR/summary.csv"
: > "$LOG_FILE"

# OrangeFS config
STORAGE_BASE="${STORAGE_BASE:-/mnt/nvme/$USER}"
ORANGEFS_MOUNT="${ORANGEFS_MOUNT:-/mnt/ssd/$USER/orangefs}"
STORAGE_ROOT="$STORAGE_BASE/orangefs"
STORAGE_DATA="$STORAGE_ROOT/data"
STORAGE_META="$STORAGE_ROOT/metadata"
CONF_FILE="${ORANGEFS_CONFIG:-$HOME/.orangefs.conf}"
TCP_PORT="${TCP_PORT:-3334}"
NET_SUFFIX="${NET_SUFFIX--40g}"
TEARDOWN="${TEARDOWN:-0}"
REBUILD_CONF="${REBUILD_CONF:-0}"

# IOR + dd config (mode bumps reps/block/tasks/xfer accordingly)
if [ "$SMALL_FILES" = "1" ]; then
  # Agent-scale: 4 MiB files, single read per file, 1 task, 10 reps.
  IOR_BLOCK="${IOR_BLOCK:-4m}"
  IOR_XFER="${IOR_XFER:-4m}"
  IOR_REPS="${IOR_REPS:-10}"
  IOR_TASKS="${IOR_TASKS:-1}"
  DD_REPS="${DD_REPS:-10}"
  DD_COUNT="${DD_COUNT:-4}"     # 4 MiB total per dd
elif [ "$RIGOROUS" = "1" ]; then
  IOR_BLOCK="${IOR_BLOCK:-4g}"
  IOR_XFER="${IOR_XFER:-1m}"
  IOR_REPS="${IOR_REPS:-5}"
  IOR_TASKS="${IOR_TASKS:-4}"
  DD_REPS="${DD_REPS:-3}"
  DD_COUNT="${DD_COUNT:-2048}"  # 2 GiB total per dd
else
  IOR_BLOCK="${IOR_BLOCK:-2g}"
  IOR_XFER="${IOR_XFER:-1m}"
  IOR_REPS="${IOR_REPS:-3}"
  IOR_TASKS="${IOR_TASKS:-1}"
  DD_REPS="${DD_REPS:-0}"
  DD_COUNT="${DD_COUNT:-2048}"
fi

# S3 config (matches scripts/microbench/path0_s3_run.sh)
S3_BUCKET="${S3_BUCKET:-noaa-goes16}"
S3_REGION="${S3_REGION:-us-east-1}"
S3_PREFIX="${S3_PREFIX:-ABI-L2-CMIPC/2024/122/00}"
S3_MOUNT="${S3_MOUNT:-/tmp/s3-noaa-goes16}"
S3_FILE_GLOB="${S3_FILE_GLOB:-OR_ABI-L2-CMIPC-M6C08_G16_*.nc}"
S3_N_FILES="${S3_N_FILES:-5}"
S3_REPS="${S3_REPS:-$IOR_REPS}"
S3_TEARDOWN="${S3_TEARDOWN:-0}"
MOUNT_S3_BIN="${MOUNT_S3_BIN:-$HOME/.local/bin/mount-s3}"

# Tier list: name:path:supports_odirect_default
# s3 is special-cased: read-only mount, no write/IOR, dd-read existing files.
ALL_TIERS=(
  "tmpfs:/dev/shm/$USER/bench_tiers:0"
  "local_nvme:/mnt/nvme/$USER/bench_tiers:1"
  "local_ssd:/mnt/ssd/$USER/bench_tiers:1"
  "shared_xfs:/mnt/common/$USER/bench_tiers:1"
  "orangefs:$ORANGEFS_MOUNT/bench_tiers:1"
  "s3:$S3_MOUNT:0"
)
TIERS="${TIERS:-tmpfs local_nvme local_ssd shared_xfs orangefs s3}"

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
log()   { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG_FILE"; }
phase() { printf '\n========== %s ==========\n' "$*" | tee -a "$LOG_FILE"; }
ok()    { printf '  PASS: %s\n'  "$*" | tee -a "$LOG_FILE"; }
warn()  { printf '  WARN: %s\n'  "$*" | tee -a "$LOG_FILE"; }
fail()  { printf '  FAIL: %s\n'  "$*" | tee -a "$LOG_FILE"; exit 1; }

# ---------------------------------------------------------------------------
# Module loads (Ares Spack-Lmod; idempotent)
#
# IMPORTANT: never redirect stdout from `module` — it relies on the surrounding
# shell to eval the env mutations it prints. We use `module --redirect ...`
# which forces messages to stderr and keeps stdout clean for the eval pipeline.
# IOR comes from $IOR_BIN (built by scripts/build_ior.sh against the same MPI
# variant orangefs depends on), so no `module load ior` is needed and there is
# no version-swap conflict.
# ---------------------------------------------------------------------------
MODULE_MPI="${MODULE_MPI:-openmpi/5.0.5-cphqvsy}"
MODULE_ORANGEFS="${MODULE_ORANGEFS:-orangefs/2.10}"
IOR_BIN="${IOR_BIN:-$HOME/.local/bin/ior}"
SKIP_MODULES="${SKIP_MODULES:-0}"

if [ "$SKIP_MODULES" != "1" ]; then
  if ! command -v module >/dev/null 2>&1; then
    for init in /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash; do
      [ -f "$init" ] && . "$init" >/dev/null 2>&1 && break
    done
  fi
  if command -v module >/dev/null 2>&1; then
    phase "Module loads"
    # purge first — Ares login shells leak module state into MODULEPATH.
    module --redirect purge 2>>"$LOG_FILE"
    for m in "$MODULE_MPI" "$MODULE_ORANGEFS"; do
      module --redirect load "$m" 2>>"$LOG_FILE"
      if [ $? -eq 0 ]; then
        ok "loaded $m"
      else
        warn "module load $m failed; continuing (set SKIP_MODULES=1 to silence)"
      fi
    done
    module --redirect list 2>>"$LOG_FILE" >>"$LOG_FILE"
  else
    warn "no module command and no Lmod init found; relying on inherited PATH"
  fi
fi

# ---------------------------------------------------------------------------
# Phase 0: Sanity
# ---------------------------------------------------------------------------
phase "Phase 0: Sanity checks"

log "REPO_ROOT=$REPO_ROOT"
log "RESULTS_DIR=$RESULTS_DIR"
log "ORANGEFS_MOUNT=$ORANGEFS_MOUNT"
log "STORAGE_BASE=$STORAGE_BASE"
log "TIERS=$TIERS"

if [ -z "${SLURM_JOB_ID:-}" ]; then
  warn "no SLURM_JOB_ID; running outside a Slurm allocation"
  warn "OrangeFS deploy will likely fail; non-orangefs tiers may still work"
fi

if [ -x "$IOR_BIN" ]; then
  ok "ior: $IOR_BIN"
else
  fail "ior not found or not executable at $IOR_BIN; run \`bash scripts/build_ior.sh\` first"
fi

if [ "$IOR_TASKS" -gt 1 ]; then
  command -v mpirun >/dev/null 2>&1 || fail "mpirun not in PATH but IOR_TASKS=$IOR_TASKS > 1"
  ok "mpirun: $(command -v mpirun)"
fi

if [ "$SMALL_FILES" = "1" ]; then mode_label="small-files"
elif [ "$RIGOROUS" = "1" ]; then mode_label="rigorous"
else mode_label="default"
fi
log "mode: $mode_label  reps=$IOR_REPS  block=$IOR_BLOCK  xfer=$IOR_XFER  tasks=$IOR_TASKS  dd_reps=$DD_REPS  dd_count=${DD_COUNT}M"

NEED_ORANGEFS=0
NEED_S3=0
for t in $TIERS; do
  [ "$t" = "orangefs" ] && NEED_ORANGEFS=1
  [ "$t" = "s3" ]       && NEED_S3=1
done

if [ "$NEED_ORANGEFS" = "1" ]; then
  command -v pvfs2-genconfig    >/dev/null || fail "pvfs2-genconfig not in PATH; \`module load orangefs\` first"
  command -v ares-orangefs-deploy >/dev/null || fail "ares-orangefs-deploy not in PATH; \`module load orangefs\` first"
  ok "pvfs2-genconfig:      $(command -v pvfs2-genconfig)"
  ok "ares-orangefs-deploy: $(command -v ares-orangefs-deploy)"
fi

if [ "$NEED_S3" = "1" ]; then
  if [ -x "$MOUNT_S3_BIN" ]; then
    ok "mount-s3: $MOUNT_S3_BIN"
  else
    fail "mount-s3 not found at $MOUNT_S3_BIN; download from https://s3.amazonaws.com/mountpoint-s3-release/latest/x86_64/mount-s3.tar.gz"
  fi
fi

HOSTNAME_SHORT="$(hostname -s)"
HOSTNAME_NET="${HOSTNAME_SHORT}${NET_SUFFIX}"
log "hostname (mgmt): $HOSTNAME_SHORT"
log "hostname (net):  $HOSTNAME_NET"

# ---------------------------------------------------------------------------
# Phase 1: Setup OrangeFS (idempotent)
# ---------------------------------------------------------------------------
phase "Phase 1: OrangeFS deploy"

orangefs_mounted() {
  mountpoint -q "$ORANGEFS_MOUNT" 2>/dev/null && \
    findmnt -rn -T "$ORANGEFS_MOUNT" -o FSTYPE | awk '$1 ~ /^(pvfs2|orangefs|fuse\.orangefs)$/ {found=1} END {exit found?0:1}'
}

if [ "$NEED_ORANGEFS" != "1" ]; then
  log "orangefs not in TIERS; skipping deploy"
elif orangefs_mounted; then
  ok "orangefs already mounted at $ORANGEFS_MOUNT (reusing)"
else
  log "preparing storage dirs..."
  mkdir -p "$STORAGE_DATA" "$STORAGE_META" "$ORANGEFS_MOUNT" 2>/dev/null || \
    fail "cannot mkdir under $STORAGE_BASE / $ORANGEFS_MOUNT - check sudoers or pre-create the dirs"

  # Wipe stale storage from prior runs (pvfs2-server -f refuses to clobber).
  if [ -n "$(ls -A "$STORAGE_DATA" 2>/dev/null)" ] || [ -n "$(ls -A "$STORAGE_META" 2>/dev/null)" ]; then
    log "wiping stale OrangeFS storage at $STORAGE_DATA, $STORAGE_META"
    rm -rf "$STORAGE_DATA"/* "$STORAGE_DATA"/.[!.]* 2>/dev/null
    rm -rf "$STORAGE_META"/* "$STORAGE_META"/.[!.]* 2>/dev/null
    ok "storage wiped"
  fi

  SERVERS_FILE="$RESULTS_DIR/servers.txt"
  CLIENTS_FILE="$RESULTS_DIR/clients.txt"
  printf '%s\n' "$HOSTNAME_NET" > "$SERVERS_FILE"
  cp "$SERVERS_FILE" "$CLIENTS_FILE"

  if [ "$REBUILD_CONF" = "1" ] || [ ! -s "$CONF_FILE" ]; then
    log "generating $CONF_FILE via pvfs2-genconfig..."
    rm -f "$CONF_FILE"
    set +e
    pvfs2-genconfig \
      --protocol tcp \
      --tcpport "$TCP_PORT" \
      --ioservers "$HOSTNAME_NET" \
      --metaservers "$HOSTNAME_NET" \
      --storage "$STORAGE_DATA" \
      --metadata "$STORAGE_META" \
      --logfile "$RESULTS_DIR/pvfs2-server.log" \
      --quiet \
      "$CONF_FILE" \
      >>"$LOG_FILE" 2>&1
    GEN_RC=$?
    set -e
    if [ $GEN_RC -ne 0 ] || [ ! -s "$CONF_FILE" ]; then
      log "non-interactive genconfig failed (rc=$GEN_RC); trying stdin-piped"
      rm -f "$CONF_FILE"
      {
        echo "tcp"
        echo "$TCP_PORT"
        echo "$HOSTNAME_NET"
        echo "$HOSTNAME_NET"
        echo "$STORAGE_DATA"
        echo "$STORAGE_META"
        echo "$RESULTS_DIR/pvfs2-server.log"
        echo "y"
      } | pvfs2-genconfig "$CONF_FILE" >>"$LOG_FILE" 2>&1 \
          || fail "pvfs2-genconfig failed both modes; see $LOG_FILE"
    fi
    ok "config written: $CONF_FILE ($(wc -l < "$CONF_FILE") lines)"
  else
    ok "reusing existing $CONF_FILE (set REBUILD_CONF=1 to regenerate)"
  fi

  log "deploying with ares-orangefs-deploy..."
  if ORANGEFS_CONFIG="$CONF_FILE" ORANGEFS_MOUNT="$ORANGEFS_MOUNT" \
     ares-orangefs-deploy "$SERVERS_FILE" "$CLIENTS_FILE" "$CONF_FILE" "$ORANGEFS_MOUNT" \
     >>"$LOG_FILE" 2>&1; then
    ok "ares-orangefs-deploy returned 0"
  else
    fail "ares-orangefs-deploy failed; see $LOG_FILE"
  fi

  sleep 3
  if orangefs_mounted; then
    ok "$ORANGEFS_MOUNT mounted ($(findmnt -rn -T "$ORANGEFS_MOUNT" -o FSTYPE | head -1))"
  else
    fail "deploy returned 0 but $ORANGEFS_MOUNT is not mounted; see $LOG_FILE"
  fi
fi

teardown() {
  rc=$?
  if [ "$NEED_ORANGEFS" = "1" ] && [ "$TEARDOWN" = "1" ] && orangefs_mounted; then
    phase "Teardown OrangeFS"
    if command -v ares-orangefs-terminate >/dev/null; then
      ORANGEFS_CONFIG="$CONF_FILE" ORANGEFS_MOUNT="$ORANGEFS_MOUNT" \
        ares-orangefs-terminate \
          "$RESULTS_DIR/servers.txt" "$RESULTS_DIR/clients.txt" "$CONF_FILE" "$ORANGEFS_MOUNT" \
          >>"$LOG_FILE" 2>&1 \
        || warn "terminate reported errors (often benign)"
    else
      warn "ares-orangefs-terminate not in PATH; leaving mount up"
    fi
  fi
  if [ "$NEED_S3" = "1" ] && [ "$S3_TEARDOWN" = "1" ] && mountpoint -q "$S3_MOUNT" 2>/dev/null; then
    phase "Teardown S3"
    fusermount -u "$S3_MOUNT" >>"$LOG_FILE" 2>&1 \
      || warn "fusermount -u $S3_MOUNT failed"
  fi
  log "exit rc=$rc; results: $RESULTS_DIR"
  exit $rc
}
trap teardown EXIT INT TERM

# ---------------------------------------------------------------------------
# Phase 1.5: Mount S3 (idempotent)
# ---------------------------------------------------------------------------
s3_mount_alive() {
  # Returns 0 iff the mount is up AND responding to a directory listing
  # (mount-s3 daemons die with their salloc; the mountpoint entry survives but
  # listings return "Transport endpoint is not connected").
  mountpoint -q "$S3_MOUNT" 2>/dev/null || return 1
  ls "$S3_MOUNT" >/dev/null 2>&1
}

if [ "$NEED_S3" = "1" ]; then
  phase "Phase 1.5: mountpoint-s3"
  if s3_mount_alive; then
    ok "s3 already mounted at $S3_MOUNT (reusing)"
  else
    if mountpoint -q "$S3_MOUNT" 2>/dev/null; then
      warn "stale s3 mount at $S3_MOUNT (dead daemon); fusermount -u and remounting"
      fusermount -u "$S3_MOUNT" >>"$LOG_FILE" 2>&1 \
        || fail "fusermount -u $S3_MOUNT failed; clear manually and retry"
    fi
    mkdir -p "$S3_MOUNT"
    log "mounting s3://$S3_BUCKET (region=$S3_REGION) at $S3_MOUNT"
    if "$MOUNT_S3_BIN" --no-sign-request --read-only --region "$S3_REGION" \
        "$S3_BUCKET" "$S3_MOUNT" >>"$LOG_FILE" 2>&1; then
      sleep 1
      if s3_mount_alive; then
        ok "mounted: $S3_MOUNT"
      else
        fail "mount-s3 returned 0 but $S3_MOUNT is not responding"
      fi
    else
      fail "mount-s3 failed; see $LOG_FILE"
    fi
  fi
  # Verify the prefix is reachable
  if [ -d "$S3_MOUNT/$S3_PREFIX" ]; then
    ok "prefix visible: $S3_MOUNT/$S3_PREFIX"
  else
    fail "$S3_MOUNT/$S3_PREFIX not visible; check S3_PREFIX or bucket access"
  fi
fi

# ---------------------------------------------------------------------------
# Phase 2: Per-tier prep + O_DIRECT probe
# ---------------------------------------------------------------------------
phase "Phase 2: Tier prep + O_DIRECT probe"

probe_odirect() {
  local path="$1"
  local probe="$path/.odirect_probe.$$"
  mkdir -p "$path" 2>/dev/null || return 2
  # Aligned 4K write with O_DIRECT; failure (EINVAL) means unsupported.
  if dd if=/dev/zero of="$probe" oflag=direct bs=4096 count=1 status=none 2>/dev/null; then
    rm -f "$probe"
    return 0
  fi
  rm -f "$probe"
  return 1
}

# Build active tier list: name|path|use_odirect
ACTIVE_TIERS=()
for spec in "${ALL_TIERS[@]}"; do
  IFS=':' read -r name path default_od <<< "$spec"
  case " $TIERS " in *" $name "*) ;; *) continue ;; esac

  # s3 is read-only; skip mkdir + writability + O_DIRECT probe.
  if [ "$name" = "s3" ]; then
    if mountpoint -q "$path" 2>/dev/null; then
      ok "$name: $path  (read-only; mountpoint-s3; no O_DIRECT)"
      ACTIVE_TIERS+=("$name|$path|0")
    else
      warn "$name: $path not mounted (skipping)"
    fi
    continue
  fi

  if ! mkdir -p "$path" 2>/dev/null; then
    warn "$name: cannot mkdir $path (skipping)"
    continue
  fi
  if ! touch "$path/.write_probe.$$" 2>/dev/null; then
    warn "$name: $path not writable (skipping)"
    continue
  fi
  rm -f "$path/.write_probe.$$"

  if [ "$default_od" = "0" ]; then
    use_od=0
    ok "$name: $path  (O_DIRECT not applicable; tmpfs lives in RAM)"
  else
    if probe_odirect "$path"; then
      use_od=1
      ok "$name: $path  (O_DIRECT supported)"
    else
      use_od=0
      warn "$name: $path  (O_DIRECT rejected; will use drop_caches between phases)"
    fi
  fi
  ACTIVE_TIERS+=("$name|$path|$use_od")
done

[ "${#ACTIVE_TIERS[@]}" -ge 1 ] || fail "no usable tiers"

# ---------------------------------------------------------------------------
# Phase 3: dd baseline (optional) + IOR sweep
# ---------------------------------------------------------------------------
phase "Phase 3: dd baseline + IOR write/read sweep"

drop_caches() {
  sync
  if [ -w /proc/sys/vm/drop_caches ]; then
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null && return 0
  fi
  sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null && return 0
  return 1
}

# CSV schema (rev 3: adds ms_per_4mib derived column for agent-scale interpretation).
echo "tier,tool,op,odirect,tasks,n_reps,xfer,block,max_mibps,min_mibps,mean_mibps,stdev_mibps,mean_s,ms_per_4mib,test_file" > "$SUMMARY_CSV"

# Aggregate dd reps with awk (mean/min/max/stdev).
dd_summary_csv_row() {
  local tier="$1" op="$2" use_od="$3" tasks="$4" reps="$5" test_file="$6"
  shift 6
  printf '%s\n' "$@" | awk -v tier="$tier" -v op="$op" -v od="$use_od" -v tasks="$tasks" -v reps="$reps" -v xfer="-" -v blk="$DD_COUNT" -v tf="$test_file" '
    { v[NR]=$1; s+=$1; if (NR==1||$1>max) max=$1; if (NR==1||$1<min) min=$1 }
    END {
      n=NR; mean=s/n; sd=0
      for (i=1;i<=n;i++) sd += (v[i]-mean)*(v[i]-mean)
      sd = (n>1) ? sqrt(sd/(n-1)) : 0
      ms4 = (mean>0) ? 4000.0/mean : 0
      printf "%s,dd,%s,%s,%d,%d,%s,%sM,%.2f,%.2f,%.2f,%.2f,-,%.2f,%s\n",
        tier, op, od, tasks, reps, xfer, blk, max, min, mean, sd, ms4, tf
    }
  '
}

# Parse one dd output line, return MiB/s as a float.
dd_parse_mibps() {
  # dd format: "... N bytes (M GB, K GiB) copied, T s, BW units"
  awk '/copied/ {
    for (i=1;i<=NF;i++) {
      if ($i ~ /MB\/s$/ || $i ~ /GB\/s$/) {
        bw=$(i-1)+0
        if ($i == "GB/s") bw = bw * 1000      # decimal GB → MiB-ish
        printf "%.2f\n", bw * 0.9537          # MB→MiB (approx); good enough at the precision dd reports
        exit
      }
    }
  }'
}

run_dd_tier() {
  local name="$1" path="$2" use_od="$3"
  [ "$DD_REPS" -gt 0 ] || return 0

  local test_file="$path/dd_test.dat"
  local w_args="" r_args=""
  if [ "$use_od" = "1" ]; then
    w_args="oflag=direct conv=fdatasync"
    r_args="iflag=direct"
  else
    w_args="conv=fdatasync"
    r_args=""
  fi

  local writes=() reads=()
  for ((i=1; i<=DD_REPS; i++)); do
    drop_caches >/dev/null 2>&1 || true
    rm -f "$test_file" 2>/dev/null

    local w_out r_out w_mb r_mb
    w_out=$(dd if=/dev/zero of="$test_file" bs=1M count="$DD_COUNT" $w_args 2>&1)
    w_mb=$(echo "$w_out" | dd_parse_mibps)
    writes+=("$w_mb")

    drop_caches >/dev/null 2>&1 || true
    r_out=$(dd if="$test_file" of=/dev/null bs=1M $r_args 2>&1)
    r_mb=$(echo "$r_out" | dd_parse_mibps)
    reads+=("$r_mb")
  done
  rm -f "$test_file" 2>/dev/null

  dd_summary_csv_row "$name" "write" "$use_od" 1 "$DD_REPS" "$test_file" "${writes[@]}" >> "$SUMMARY_CSV"
  dd_summary_csv_row "$name" "read"  "$use_od" 1 "$DD_REPS" "$test_file" "${reads[@]}"  >> "$SUMMARY_CSV"
}

run_ior_tier() {
  local name="$1" path="$2" use_od="$3"
  local out="$RESULTS_DIR/ior_${name}.txt"
  local test_file="$path/ior_test.dat"

  log "tier=$name  path=$path  odirect=$use_od  tasks=$IOR_TASKS"

  rm -f "$test_file"* 2>/dev/null || true
  mkdir -p "$path"

  if drop_caches; then
    log "  drop_caches: ok"
  else
    warn "  drop_caches not permitted (no sudo); O_DIRECT must carry the cold-cache guarantee"
  fi

  local odirect_args=""
  [ "$use_od" = "1" ] && odirect_args="--posix.odirect"

  # Launcher: mpirun for multi-task, direct for single.
  local launcher=()
  if [ "$IOR_TASKS" -gt 1 ]; then
    launcher=(mpirun --oversubscribe -np "$IOR_TASKS")
  fi

  # IOR flags:
  #   -a POSIX       POSIX backend
  #   -F             file-per-process
  #   -w -r          write then read
  #   -e             fsync at close (forces durable write)
  #   -k             keep test file across reps
  #   -b/-t          block / transfer size
  #   -i             repetitions
  #   --posix.odirect  set O_DIRECT
  set +e
  "${launcher[@]}" "$IOR_BIN" \
    -a POSIX \
    -F -w -r -e -k \
    -b "$IOR_BLOCK" \
    -t "$IOR_XFER" \
    -i "$IOR_REPS" \
    -o "$test_file" \
    $odirect_args \
    > "$out" 2>&1
  local rc=$?
  set -e

  if [ $rc -ne 0 ]; then
    warn "  ior exit rc=$rc; see $out"
  fi

  rm -f "$test_file"* 2>/dev/null

  awk -v tier="$name" -v od="$use_od" -v tasks="$IOR_TASKS" -v reps="$IOR_REPS" -v xfer="$IOR_XFER" -v blk="$IOR_BLOCK" -v tf="$test_file" '
    /^Summary of all tests:/ { in_sum=1; next }
    in_sum && /^(write|read) / {
      op=$1; max=$2; min=$3; mean=$4; sd=$5; meanS=$10
      ms4 = (mean+0 > 0) ? 4000.0/(mean+0) : 0
      printf "%s,ior,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%.2f,%s\n",
        tier, op, od, tasks, reps, xfer, blk, max, min, mean, sd, meanS, ms4, tf
    }
  ' "$out" >> "$SUMMARY_CSV"
}

run_s3_read_tier() {
  # Read $S3_N_FILES distinct files from $S3_MOUNT/$S3_PREFIX with dd,
  # $S3_REPS times. Each open() triggers a fresh S3 GET (mountpoint-s3
  # doesn't cache data by default), so reps are independent cold reads.
  local name="$1" path="$2"
  local prefix_dir="$path/$S3_PREFIX"

  if [ ! -d "$prefix_dir" ]; then
    warn "  $name: $prefix_dir not visible; skipping"
    return 0
  fi

  # Sample S3_N_FILES files matching the glob.
  mapfile -t s3_files < <(find "$prefix_dir" -maxdepth 1 -name "$S3_FILE_GLOB" 2>/dev/null | shuf -n "$S3_N_FILES")
  if [ "${#s3_files[@]}" -eq 0 ]; then
    warn "  $name: no files matching $S3_FILE_GLOB under $prefix_dir; skipping"
    return 0
  fi
  log "  s3 sample: ${#s3_files[@]} files from $S3_PREFIX (matching $S3_FILE_GLOB)"

  local reads_mibps=()
  for ((rep=1; rep<=S3_REPS; rep++)); do
    local total_bytes=0 t0 t1 elapsed_s mibps
    t0=$(date +%s.%N)
    for f in "${s3_files[@]}"; do
      # dd with iflag=nocache asks the kernel not to retain pages, though FUSE
      # bypasses page cache anyway for mountpoint-s3 data.
      local out
      out=$(dd if="$f" of=/dev/null bs=1M iflag=nocache 2>&1 || true)
      # Sum file sizes for an aggregate MiB/s number.
      local sz
      sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
      total_bytes=$((total_bytes + sz))
    done
    t1=$(date +%s.%N)
    elapsed_s=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b-a}')
    mibps=$(awk -v b="$total_bytes" -v s="$elapsed_s" 'BEGIN{printf "%.2f", (b/1048576)/s}')
    reads_mibps+=("$mibps")
    log "    rep $rep: $((total_bytes/1024/1024)) MiB in ${elapsed_s}s = $mibps MiB/s"
  done

  # Aggregate (mean/min/max/stdev) and write CSV row.
  printf '%s\n' "${reads_mibps[@]}" | awk -v tier="$name" -v tf="$prefix_dir" -v reps="$S3_REPS" -v nfiles="${#s3_files[@]}" '
    { v[NR]=$1; s+=$1; if (NR==1||$1>max) max=$1; if (NR==1||$1<min) min=$1 }
    END {
      n=NR; mean=s/n; sd=0
      for (i=1;i<=n;i++) sd += (v[i]-mean)*(v[i]-mean)
      sd = (n>1) ? sqrt(sd/(n-1)) : 0
      ms4 = (mean>0) ? 4000.0/mean : 0
      printf "%s,dd,read,0,1,%d,-,%d_files,%.2f,%.2f,%.2f,%.2f,-,%.2f,%s\n",
        tier, reps, nfiles, max, min, mean, sd, ms4, tf
    }
  ' >> "$SUMMARY_CSV"
}

for entry in "${ACTIVE_TIERS[@]}"; do
  IFS='|' read -r name path use_od <<< "$entry"
  if [ "$name" = "s3" ]; then
    log "tier=$name  s3 read-only baseline (n_files=$S3_N_FILES, reps=$S3_REPS)"
    run_s3_read_tier "$name" "$path"
    continue
  fi
  if [ "$DD_REPS" -gt 0 ]; then
    log "tier=$name  dd baseline (reps=$DD_REPS, count=${DD_COUNT}M)"
    run_dd_tier "$name" "$path" "$use_od"
  fi
  run_ior_tier "$name" "$path" "$use_od"
done

# ---------------------------------------------------------------------------
# Phase 4: Render summary
# ---------------------------------------------------------------------------
phase "Phase 4: Bandwidth summary"

log "csv: $SUMMARY_CSV"
echo "" | tee -a "$LOG_FILE"
column -s, -t < "$SUMMARY_CSV" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Compact wide-form table: tier | tool | write mean MiB/s | read mean MiB/s |
# read ms per 4 MiB (agent-scale latency) | odirect
log "compact (mean MiB/s; read_ms_4MiB = ms to read a 4 MiB file at this bandwidth):"
{
  printf "  %-12s %-5s %-12s %-12s %-12s %-7s\n" "tier" "tool" "write_MiB/s" "read_MiB/s" "read_ms_4MiB" "odirect"
  awk -F, '
    NR==1 { next }
    {
      # Columns: tier,tool,op,odirect,tasks,n_reps,xfer,block,max,min,mean,stdev,mean_s,ms_per_4mib,test_file
      key=$1"|"$2
      seen[key]=1
      tier[key]=$1; tool[key]=$2
      if ($3=="write") { w[key]=$11; w_od[key]=$4 }
      if ($3=="read")  { r[key]=$11; r_ms[key]=$14; r_od[key]=$4 }
    }
    END {
      for (k in seen)
        printf "  %-12s %-5s %-12.2f %-12.2f %-12.2f %-7s\n", tier[k], tool[k], w[k]+0, r[k]+0, r_ms[k]+0, (w_od[k]+0 ? "yes" : "no")
    }
  ' "$SUMMARY_CSV" | sort -k1,1 -k2,2
} | tee -a "$LOG_FILE"

# Cross-tool sanity: report dd/ior ratio per tier (rigorous mode only)
if [ "$DD_REPS" -gt 0 ]; then
  echo "" | tee -a "$LOG_FILE"
  log "dd vs ior cross-check (ratio = dd_mean / ior_mean; should be ~1.0):"
  awk -F, '
    NR==1 { next }
    { key=$1"|"$3; if ($2=="dd") dd[key]=$11; if ($2=="ior") ior[key]=$11 }
    END {
      printf "  %-12s %-6s %-12s %-12s %-7s\n", "tier", "op", "dd_MiB/s", "ior_MiB/s", "ratio"
      for (k in dd) {
        if (k in ior && ior[k]>0) {
          split(k, p, "|")
          printf "  %-12s %-6s %-12.2f %-12.2f %-7.2f\n", p[1], p[2], dd[k], ior[k], dd[k]/ior[k]
        }
      }
    }
  ' "$SUMMARY_CSV" | sort -k1,1 -k2,2 | tee -a "$LOG_FILE"
fi

log "done"
exit 0
