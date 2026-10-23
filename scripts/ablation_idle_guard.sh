#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# ablation_idle_guard.sh — safeguard against runaway idle billing on EC2 runner.
#
# Runs periodically (via ablation-idle-guard.timer) to detect "zombie" instances:
#   * Boots where no ablation tag was provided and instance was forgotten.
#   * Early crashes during cloud-init/bootstrap before a job starts.
#   * Completed jobs where automatic shutdown failed for any unexpected reason.
#
# Immune to Kaggle waiting periods:
#   As long as `ablation@<EXP>.service` is in the `active` state (even if CPU
#   is near 0% while waiting for Kaggle GPU execution), this guard exits immediately.
#
# Escape hatch:
#   Touch `/tmp/disable_idle_guard` to keep the VM alive during manual debugging.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

OVERRIDE_FILE="/tmp/disable_idle_guard"
GRACE_PERIOD_SEC="${IDLE_GRACE_PERIOD_SEC:-1200}" # 20 minutes default

log() { printf '[idle-guard %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# 1. Manual override check
if [[ -f "$OVERRIDE_FILE" ]]; then
  log "Override file $OVERRIDE_FILE exists; idle guard disabled."
  exit 0
fi

# 2. Boot grace period check (allows cloud-init, bootstrap and startup to run)
UPTIME_SEC=0
if [[ -f /proc/uptime ]]; then
  UPTIME_SEC=$(awk '{print int($1)}' /proc/uptime 2>/dev/null || echo 0)
fi

if [[ "$UPTIME_SEC" -lt "$GRACE_PERIOD_SEC" ]]; then
  # Still within initial grace period
  exit 0
fi

# 3. Check if any ablation job is actively running
# Matches any instance of the template unit (e.g. ablation@B1, ablation@B2)
if systemctl is-active --quiet "ablation@*.service" 2>/dev/null; then
  # Job is active (reprompting, waiting for Kaggle, scoring, or publishing)
  exit 0
fi

# 4. Check if autostart boot-trigger is still running
if systemctl is-active --quiet "ablation-autostart.service" 2>/dev/null; then
  exit 0
fi

# 5. Check if cloud-init is still running on first boot
if command -v cloud-init >/dev/null 2>&1; then
  if cloud-init status 2>/dev/null | grep -q "status: running"; then
    exit 0
  fi
fi

# No ablation job is running and grace period has elapsed -> shut down
log "No active ablation service detected after ${UPTIME_SEC}s uptime (> ${GRACE_PERIOD_SEC}s)."
log "Initiating safe VM shutdown to prevent runaway billing..."
sync
sleep 2

if command -v shutdown >/dev/null 2>&1; then
  sudo shutdown -h now || shutdown -h now || true
elif command -v poweroff >/dev/null 2>&1; then
  sudo poweroff || poweroff || true
fi
