#!/usr/bin/env bash
# Restart the OrangeFS server in-place without re-running the full
# orangefs_smoke.sh setup. Use this when pvfs2-server died but the
# mount, config, and backing storage on /mnt/nvme are still intact.
#
# Typical recovery scenario:
#   - You ran `pkill` or `kill` that accidentally hit pvfs2-server
#   - SLURM stepd killed it on step exit (REMOTE_LAUNCHER=srun footgun)
#   - Node didn't reboot; OrangeFS data on /mnt/nvme is untouched
#
# If the backing storage is gone, use orangefs_smoke.sh (in sciiobench)
# to do a fresh bring-up.
set -euo pipefail

NODE="${1:-$(hostname)}"
CONF="${ORANGEFS_CONF:-/home/$USER/orangefs_aiob205/orangefs.conf}"
LOG="${ORANGEFS_RESTART_LOG:-/tmp/pvfs2-server-restart.log}"
PVFS_BIN="${PVFS_BIN:-/mnt/repo/software/modules-install/orangefs/2.10/sbin/pvfs2-server}"
MOUNT="${ORANGEFS_MOUNT:-/mnt/ssd/$USER/orangefs_aiob205}"

if pgrep -af pvfs2-server >/dev/null; then
    echo "pvfs2-server already running:"
    pgrep -af pvfs2-server
    exit 0
fi

[ -f "$CONF" ] || { echo "ERROR: config not found at $CONF" >&2; exit 1; }
[ -x "$PVFS_BIN" ] || { echo "ERROR: pvfs2-server not at $PVFS_BIN" >&2; exit 1; }

echo "Starting pvfs2-server -a $NODE $CONF ..."
nohup "$PVFS_BIN" -a "$NODE" "$CONF" > "$LOG" 2>&1 &
sleep 3

if ! pgrep -af pvfs2-server >/dev/null; then
    echo "ERROR: server did not start. Check $LOG" >&2
    tail -20 "$LOG" >&2
    exit 1
fi

echo "Server running:"
pgrep -af pvfs2-server
echo ""
echo "Mount probe ($MOUNT):"
ls "$MOUNT" 2>&1 | head -5 || true
