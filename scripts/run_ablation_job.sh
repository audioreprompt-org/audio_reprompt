#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_ablation_job.sh — on-demand execution runner for ablation experiments.
#
# Designed to run unattended on a virtual machine (or local workstation):
#   1. Acquires a lock to prevent multiple local executions.
#   2. Loads environment variables (from .env or AWS Secrets Manager for RDS).
#   3. Syncs latest repository branch and DVC dependencies.
#   4. Runs the ablation experiment pipeline with per-run Kaggle isolation.
#   5. Publishes results concurrency-safely via ablation_publish.sh (DVC + git).
#   6. Optionally powers off / stops the VM when execution completes.
#
# Usage:
#   scripts/run_ablation_job.sh --experiment B2
#   scripts/run_ablation_job.sh -e B1 --sample-size 50 --seed 42 --shutdown
#   scripts/run_ablation_job.sh -e B2 --secret-name audio-db-secrets --shutdown
#
# Environment variables:
#   AUTO_SHUTDOWN=1       Automatically stop/poweroff VM when done
#   DB_SECRET_NAME=...     AWS Secrets Manager secret name for RDS credentials
#   KAGGLE_SECRET_NAME=... AWS Secrets Manager secret name for Kaggle credentials
#   KAGGLE_ISOLATE_RUNS=1  Force isolated Kaggle kernel/dataset per run (default: 1)
# ──────────────────────────────────────────────────────────────────────────────
set -Eeuo pipefail

EXPERIMENT="B2"
SAMPLE_SIZE=30
SEED=42
RUN_ID=""
BRANCH=""
REMOTE="origin"
SECRET_NAME="${DB_SECRET_NAME:-}"
KAGGLE_SECRET_NAME="${KAGGLE_SECRET_NAME:-}"
KAGGLE_USER_ARG=""
KAGGLE_KEY_ARG=""
SHUTDOWN="${AUTO_SHUTDOWN:-0}"
DO_PUBLISH=1
LOCK_FILE="/tmp/ablation_job.lock"

log() { printf '[job %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage: run_ablation_job.sh [OPTIONS]

Options:
  -e, --experiment EXP       Experiment group to run (B1, B2, B3, or variant, default: B2)
  -s, --sample-size N        Number of stratified samples (default: 30)
      --seed S               Random seed (default: 42)
      --run-id RID           Explicit run ID (default: auto-generated RYYYYMMDD_HHMMSS)
  -b, --branch BRANCH        Git branch to sync and publish to (default: current branch)
      --remote REMOTE        Git remote (default: origin)
      --secret-name NAME     AWS Secrets Manager secret name for RDS database credentials
      --kaggle-secret NAME   AWS Secrets Manager secret name for Kaggle credentials (multi-user)
      --kaggle-user USER     Kaggle username for this run (overrides environment)
      --kaggle-key KEY       Kaggle API key for this run (overrides environment)
      --shutdown             Shut down / power off the VM upon completion
      --no-publish           Skip publishing DVC outputs and git commit
  -h, --help                 Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -e | --experiment) EXPERIMENT="$2"; shift 2 ;;
    -s | --sample-size) SAMPLE_SIZE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    -b | --branch) BRANCH="$2"; shift 2 ;;
    --remote) REMOTE="$2"; shift 2 ;;
    --secret-name) SECRET_NAME="$2"; shift 2 ;;
    --kaggle-secret) KAGGLE_SECRET_NAME="$2"; shift 2 ;;
    --kaggle-user) KAGGLE_USER_ARG="$2"; shift 2 ;;
    --kaggle-key) KAGGLE_KEY_ARG="$2"; shift 2 ;;
    --shutdown) SHUTDOWN=1; shift ;;
    --no-publish) DO_PUBLISH=0; shift ;;
    -h | --help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [[ -z "$BRANCH" ]]; then
  BRANCH="$(git rev-parse --abbrev-ref HEAD)"
  if [[ "$BRANCH" == "HEAD" ]]; then
    BRANCH="ablations-experiments"
  fi
fi

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="R$(date +%Y%m%d_%H%M%S)"
fi

# ── 1. Acquire execution lock ────────────────────────────────────────────────
exec 200>"$LOCK_FILE"
if command -v flock >/dev/null 2>&1; then
  if ! flock -n 200; then
    die "Another ablation job is already running on this machine (locked by $LOCK_FILE)."
  fi
fi

# Handle post-execution actions (shutdown if requested)
cleanup_and_shutdown() {
  local exit_code=$?
  if [[ "$SHUTDOWN" -eq 1 ]]; then
    log "Job finished with status $exit_code. Initiating VM shutdown..."
    sync
    sleep 3
    if command -v shutdown >/dev/null 2>&1; then
      sudo shutdown -h now || shutdown -h now || true
    elif command -v poweroff >/dev/null 2>&1; then
      sudo poweroff || poweroff || true
    fi
  fi
  exit $exit_code
}
trap cleanup_and_shutdown EXIT INT TERM

log "============================================================"
log "Starting Ablation Job"
log "  Experiment:  $EXPERIMENT"
log "  Sample Size: $SAMPLE_SIZE"
log "  Seed:        $SEED"
log "  Run ID:      $RUN_ID"
log "  Branch:      $BRANCH"
log "  Shutdown:    $SHUTDOWN"
log "============================================================"

# ── 2. Environment & Credentials Configuration ────────────────────────────────
# Load .env if present
if [[ -f .env ]]; then
  log "Loading configuration from .env"
  set -a; source .env 2>/dev/null || true; set +a
fi

# Load DB and Kaggle credentials safely via models.infra_secrets
log "Resolving credentials..."
SECRETS_EXPORTS=$(uv run python -m models.infra_secrets \
  ${SECRET_NAME:+--db-secret "$SECRET_NAME"} \
  ${KAGGLE_SECRET_NAME:+--kaggle-secret "$KAGGLE_SECRET_NAME"} \
  ${KAGGLE_USER_ARG:+--kaggle-user "$KAGGLE_USER_ARG"} \
  ${KAGGLE_KEY_ARG:+--kaggle-key "$KAGGLE_KEY_ARG"})

if [[ -n "$SECRETS_EXPORTS" ]]; then
  eval "$SECRETS_EXPORTS"
fi

if [[ -n "${DB_HOST:-}" ]]; then
  log "✓ Database target: $DB_HOST (DB: ${DB_NAME:-default})"
else
  log "⚠ Note: DB_HOST is not set. Reprompt phase requires PostgreSQL."
fi

if [[ -n "${KAGGLE_USERNAME:-}" ]]; then
  log "✓ Kaggle account: $KAGGLE_USERNAME (GPU quota owner)"
fi

# ── 3. Code & DVC synchronization ─────────────────────────────────────────────
log "Syncing repository with $REMOTE/$BRANCH..."
git fetch "$REMOTE" "$BRANCH" --quiet || true
# If there are no unstaged modifications, align branch
if git diff --quiet && git diff --cached --quiet; then
  git checkout "$BRANCH" --quiet || true
fi

# Ensure DVC prompt inputs exist
if [[ -f data/prompts.dvc ]] && ! ls data/prompts/user/*.csv >/dev/null 2>&1; then
  log "Pulling input prompts via DVC..."
  dvc pull data/prompts.dvc || true
fi

# ── 4. Execution ─────────────────────────────────────────────────────────────
export KAGGLE_ISOLATE_RUNS="${KAGGLE_ISOLATE_RUNS:-1}"
export PYTHONUNBUFFERED=1

log "Running experiment $EXPERIMENT (Phase: full)..."
TARGET_CMD="ablation-${EXPERIMENT}-full"
# If standard group target doesn't exist, call ablation-full with EXPERIMENT
if ! grep -q "^${TARGET_CMD}:" Makefile; then
  uv run --env-file .env python -m models.ablation_runner \
    --experiment "$EXPERIMENT" \
    --sample-size "$SAMPLE_SIZE" \
    --seed "$SEED" \
    --phase full \
    --run-id "$RUN_ID"
else
  make "$TARGET_CMD" \
    SAMPLE_SIZE="$SAMPLE_SIZE" \
    SEED="$SEED" \
    RUN_ID="$RUN_ID"
fi

log "✓ Pipeline execution completed successfully for $EXPERIMENT ($RUN_ID)."

# ── 5. Publication ───────────────────────────────────────────────────────────
if [[ "$DO_PUBLISH" -eq 1 ]]; then
  log "Publishing results (updating data/ablations.dvc and git commit)..."
  ./scripts/ablation_publish.sh \
    --run-id "$RUN_ID" \
    --branch "$BRANCH" \
    --remote "$REMOTE" \
    --target data/ablations \
    -m "ablations: automated run $EXPERIMENT ($RUN_ID) [size=$SAMPLE_SIZE, seed=$SEED]"
  log "✓ Results successfully published."
fi

log "============================================================"
log "Ablation Job $RUN_ID Completed."
log "============================================================"
