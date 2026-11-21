#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# ablation_idle_guard.sh — safeguard against runaway idle billing on EC2 runner.
#
# Runs periodically (via ablation-idle-guard.timer) to detect "zombie" instances:
#   * Boots where no ablation job was triggered and instance was left idle.
#   * Early crashes during cloud-init/bootstrap before a job starts.
#   * Completed jobs where automatic shutdown failed for any unexpected reason.
#
# Inactivity tracking:
#   * Detects active, activating, or reloading systemd units (ablation@*.service).
#   * Detects active execution locks held on $LOCK_DIR (manual CLI runs).
#   * Detects ongoing cloud-init runs.
#   * Tracks true continuous inactivity by recording the last-active uptime
#     to $STATE_FILE (or fallback /tmp/ablation_last_active_uptime).
#   * Initiates VM shutdown only when continuous inactivity reaches or exceeds
#     $GRACE_PERIOD_SEC (default: 1800s / 30m).
#
# Escape hatch:
#   Touch `/tmp/disable_idle_guard` (or $ABLATION_OVERRIDE_FILE) to keep the
#   VM alive during manual debugging.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

OVERRIDE_FILE="${ABLATION_OVERRIDE_FILE:-/tmp/disable_idle_guard}"
LOCK_DIR="${ABLATION_LOCK_DIR:-/tmp/ablation_job.lock.d}"
UPTIME_FILE="${ABLATION_UPTIME_FILE:-/proc/uptime}"
STATE_FILE="${ABLATION_LAST_ACTIVE_FILE:-/run/ablation/last_active_uptime}"
FALLBACK_STATE_FILE="/tmp/ablation_last_active_uptime"
GRACE_PERIOD_SEC="${IDLE_GRACE_PERIOD_SEC:-1800}"

log() { printf '[idle-guard %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

flock_fd_nonblock() {
  local fd="$1"
  if command -v flock >/dev/null 2>&1; then
    flock -n "$fd"
  elif command -v python3 >/dev/null 2>&1; then
    python3 -c 'import fcntl, sys; fcntl.flock(int(sys.argv[1]), fcntl.LOCK_EX | fcntl.LOCK_NB)' "$fd" 2>/dev/null
  else
    return 0
  fi
}

write_last_active() {
  local dir
  dir="$(dirname "$STATE_FILE")"
  if mkdir -p "$dir" 2>/dev/null && printf '%s\n' "$UPTIME_SEC" > "$STATE_FILE" 2>/dev/null; then
    return 0
  fi
  printf '%s\n' "$UPTIME_SEC" > "$FALLBACK_STATE_FILE" 2>/dev/null || true
}

read_last_active() {
  local val=""
  if [[ -r "$STATE_FILE" ]]; then
    val="$(head -n 1 "$STATE_FILE" 2>/dev/null || true)"
  fi
  if [[ -z "$val" || ! "$val" =~ ^[0-9]+$ ]] && [[ -r "$FALLBACK_STATE_FILE" ]]; then
    val="$(head -n 1 "$FALLBACK_STATE_FILE" 2>/dev/null || true)"
  fi
  if [[ "$val" =~ ^[0-9]+$ ]]; then
    # Guard against stale pre-reboot state (e.g., if /tmp persisted across a stop/start cycle)
    if (( val > UPTIME_SEC )); then
      echo 0
    else
      echo "$val"
    fi
  else
    echo 0
  fi
}

# 1. Manual override check
if [[ -f "$OVERRIDE_FILE" ]]; then
  log "Override file $OVERRIDE_FILE exists; idle guard disabled."
  exit 0
fi

# 2. Read system uptime
UPTIME_SEC=0
if [[ -f "$UPTIME_FILE" ]]; then
  UPTIME_SEC=$(awk '{print int($1)}' "$UPTIME_FILE" 2>/dev/null || echo 0)
fi
[[ "$UPTIME_SEC" =~ ^[0-9]+$ ]] || UPTIME_SEC=0

# 3. Detect whether any ablation work is currently active
ACTIVE=0

# Check systemd for active, activating, or reloading instances (awk drains pipe to avoid SIGPIPE under pipefail)
if command -v systemctl >/dev/null 2>&1 && systemctl list-units --type=service --state=active,activating,reloading --plain --no-legend "ablation@*.service" 2>/dev/null | awk 'NF { found=1 } END { exit (found ? 0 : 1) }'; then
  ACTIVE=1
fi

# Check if LOCK_DIR is held by any process (covers manual CLI runs)
if [[ "$ACTIVE" -eq 0 && -d "$LOCK_DIR" ]] && ! (exec 201<"$LOCK_DIR" && flock_fd_nonblock 201) 2>/dev/null; then
  ACTIVE=1
fi

# Check if cloud-init is still running on first boot
if [[ "$ACTIVE" -eq 0 ]] && command -v cloud-init >/dev/null 2>&1 && cloud-init status 2>/dev/null | awk '/status: running/ { found=1 } END { exit (found ? 0 : 1) }'; then
  ACTIVE=1
fi

# 4. Handle active vs inactive state
if [[ "$ACTIVE" -eq 1 ]]; then
  write_last_active
  exit 0
fi

# 5. When inactive, compute true idle duration
LAST_ACTIVE="$(read_last_active)"
IDLE_SEC=$(( UPTIME_SEC - LAST_ACTIVE ))
if (( IDLE_SEC < 0 )); then
  IDLE_SEC=0
fi

if [[ "$IDLE_SEC" -lt "$GRACE_PERIOD_SEC" ]]; then
  exit 0
fi

# Continuous idle grace period has elapsed -> initiate safe shutdown
log "No active ablation job detected for ${IDLE_SEC}s (>= ${GRACE_PERIOD_SEC}s; uptime=${UPTIME_SEC}s)."
log "Initiating safe VM shutdown to prevent runaway billing..."
sync
sleep "${ABLATION_SHUTDOWN_SLEEP_SEC:-2}"

if command -v shutdown >/dev/null 2>&1; then
  sudo shutdown -h now || shutdown -h now || true
elif command -v poweroff >/dev/null 2>&1; then
  sudo poweroff || poweroff || true
fi
