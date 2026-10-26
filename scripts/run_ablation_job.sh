#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_ablation_job.sh — on-demand execution runner for ablation experiments.
#
# Designed to run unattended on a virtual machine (or local workstation):
#   1. Acquires a lock to prevent multiple local executions.
#   2. Hard-syncs the repository to $REMOTE/$BRANCH (+ uv sync --frozen) and
#      re-executes itself from the updated checkout.
#   3. Loads credentials (.env and/or AWS Secrets Manager) and fails fast if
#      the database, Kaggle or (when publishing) git push are not usable.
#   4. Pulls DVC inputs and runs the ablation pipeline with per-run Kaggle isolation.
#   5. Publishes results concurrency-safely via ablation_publish.sh (DVC + git).
#   6. Optionally powers off / stops the VM when execution completes.
#
# Usage:
#   scripts/run_ablation_job.sh --experiment B2
#   scripts/run_ablation_job.sh -e B1 --sample-size 50 --seed 42 --shutdown
#   scripts/run_ablation_job.sh -e B2 --secret-name audio-db-secrets --shutdown
#   scripts/run_ablation_job.sh -e B2 --no-sync --no-publish   # local dev
#
# Environment variables (on EC2 provided by /etc/ablation/env via systemd):
#   AUTO_SHUTDOWN=1          Automatically stop/poweroff VM when done
#   ABLATION_BRANCH=...      Branch to sync, run and publish to
#   DB_SECRET_NAME=...       AWS Secrets Manager secret name/ARN for RDS credentials
#   KAGGLE_SECRET_NAME=...   AWS Secrets Manager secret name for Kaggle credentials
#   LLM_SECRET_NAME=...      AWS Secrets Manager secret name for LLM API keys
#   GIT_SSH_SECRET_NAME=...  AWS Secrets Manager secret holding a GitHub deploy key
#   GIT_PUSH_URL=...         Push URL used with the deploy key (e.g. git@github.com:org/repo.git)
#   GIT_AUTHOR_NAME/EMAIL    Commit identity when git user.name/email are not configured
#   KAGGLE_ISOLATE_RUNS=1    Force isolated Kaggle kernel/dataset per run (default: 1)
# ──────────────────────────────────────────────────────────────────────────────
set -Eeuo pipefail

ORIG_ARGS=("$@")

EXPERIMENT="B2"
SAMPLE_SIZE=30
SEED=42
RUN_ID=""
BRANCH="${ABLATION_BRANCH:-}"
REMOTE="origin"
SECRET_NAME="${DB_SECRET_NAME:-}"
KAGGLE_SECRET_NAME="${KAGGLE_SECRET_NAME:-}"
LLM_SECRET_NAME="${LLM_SECRET_NAME:-}"
GIT_SSH_SECRET_NAME="${GIT_SSH_SECRET_NAME:-}"
KAGGLE_USER_ARG=""
KAGGLE_KEY_ARG=""
SHUTDOWN="${AUTO_SHUTDOWN:-0}"
DO_PUBLISH=1
DO_SYNC=1
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
  -b, --branch BRANCH        Git branch to sync and publish to
                             (default: $ABLATION_BRANCH, else current branch)
      --remote REMOTE        Git remote (default: origin)
      --secret-name NAME     AWS Secrets Manager secret name for RDS database credentials
      --kaggle-secret NAME   AWS Secrets Manager secret name for Kaggle credentials (multi-user)
      --llm-secret NAME      AWS Secrets Manager secret name for LLM API keys (OpenAI)
      --kaggle-user USER     Kaggle username for this run (overrides environment)
      --kaggle-key KEY       Kaggle API key for this run (overrides environment)
      --git-ssh-secret NAME  AWS Secrets Manager secret holding a GitHub deploy key (for publish)
      --shutdown             Shut down / power off the VM upon completion
      --no-publish           Skip publishing DVC outputs and git commit
      --no-sync              Don't hard-sync the checkout to REMOTE/BRANCH (local dev)
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
    --llm-secret) LLM_SECRET_NAME="$2"; shift 2 ;;
    --kaggle-user) KAGGLE_USER_ARG="$2"; shift 2 ;;
    --kaggle-key) KAGGLE_KEY_ARG="$2"; shift 2 ;;
    --git-ssh-secret) GIT_SSH_SECRET_NAME="$2"; shift 2 ;;
    --shutdown) SHUTDOWN=1; shift ;;
    --no-publish) DO_PUBLISH=0; shift ;;
    --no-sync) DO_SYNC=0; shift ;;
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

# ── 1. Acquire execution lock ────────────────────────────────────────────────
# After a self re-exec (step 2) fd 200 is inherited and still holds the lock.
if [[ -z "${ABLATION_JOB_REEXEC:-}" ]]; then
  exec 200>"$LOCK_FILE"
  if command -v flock >/dev/null 2>&1; then
    if ! flock -n 200; then
      die "Another ablation job is already running on this machine (locked by $LOCK_FILE)."
    fi
  fi
fi

# Handle post-execution actions (shutdown if requested)
cleanup_and_shutdown() {
  local exit_code=$?
  trap - EXIT  # disable trap to prevent re-entrancy
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
  exit "$exit_code"
}
trap cleanup_and_shutdown EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ── 2. Code synchronization ──────────────────────────────────────────────────
# Runs before anything else so the job always executes the latest code of
# $BRANCH (not whatever was cloned / checked out on the machine).
sync_code() {
  log "Syncing repository with $REMOTE/$BRANCH..."
  git fetch --quiet "$REMOTE" "+refs/heads/$BRANCH:refs/remotes/$REMOTE/$BRANCH" \
    || die "git fetch failed for $REMOTE/$BRANCH"

  if ! git diff --quiet || ! git diff --cached --quiet; then
    die "Working tree has uncommitted changes; commit/stash them or pass --no-sync."
  fi
  if git rev-parse --verify --quiet "refs/heads/$BRANCH" >/dev/null; then
    local ahead
    ahead="$(git rev-list --count "$REMOTE/$BRANCH..refs/heads/$BRANCH")"
    if [[ "$ahead" -ne 0 ]]; then
      die "Local $BRANCH has $ahead unpushed commit(s); push them or pass --no-sync."
    fi
  fi

  git checkout --quiet -B "$BRANCH" "$REMOTE/$BRANCH"
  log "✓ Checked out $BRANCH @ $(git rev-parse --short HEAD)"

  # --frozen: never rewrite uv.lock (a dirty tree would block publishing)
  uv sync --frozen --quiet || die "uv sync --frozen failed (is uv.lock up to date?)"
}

if [[ "$DO_SYNC" -eq 1 && -z "${ABLATION_JOB_REEXEC:-}" ]]; then
  OLD_HEAD="$(git rev-parse HEAD)"
  sync_code
  if [[ "$(git rev-parse HEAD)" != "$OLD_HEAD" ]]; then
    log "Code updated; re-executing job from the new checkout..."
    trap - EXIT INT TERM
    export ABLATION_JOB_REEXEC=1
    exec "$REPO_ROOT/scripts/run_ablation_job.sh" ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"}
  fi
fi

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="R$(date +%Y%m%d_%H%M%S)"
fi

log "============================================================"
log "Starting Ablation Job"
log "  Experiment:  $EXPERIMENT"
log "  Sample Size: $SAMPLE_SIZE"
log "  Seed:        $SEED"
log "  Run ID:      $RUN_ID"
log "  Branch:      $BRANCH @ $(git rev-parse --short HEAD)"
log "  Publish:     $DO_PUBLISH"
log "  Shutdown:    $SHUTDOWN"
log "============================================================"

# ── 3. Environment & Credentials Configuration ────────────────────────────────
# Load .env if present (local workstation; absent on a fresh EC2 clone)
if [[ -f .env ]]; then
  log "Loading configuration from .env"
  set -a; source .env 2>/dev/null || true; set +a
fi

# Load DB / Kaggle / git credentials via models.infra_secrets.
# It exits non-zero if any *requested* secret can't be resolved → fail fast.
log "Resolving credentials..."
SECRETS_EXPORTS=$(uv run --quiet python -m models.infra_secrets \
  ${SECRET_NAME:+--db-secret "$SECRET_NAME"} \
  ${KAGGLE_SECRET_NAME:+--kaggle-secret "$KAGGLE_SECRET_NAME"} \
  ${LLM_SECRET_NAME:+--llm-secret "$LLM_SECRET_NAME"} \
  ${KAGGLE_USER_ARG:+--kaggle-user "$KAGGLE_USER_ARG"} \
  ${KAGGLE_KEY_ARG:+--kaggle-key "$KAGGLE_KEY_ARG"} \
  ${GIT_SSH_SECRET_NAME:+--git-ssh-secret "$GIT_SSH_SECRET_NAME"}) \
  || die "Credential resolution failed (see messages above)."

if [[ -n "$SECRETS_EXPORTS" ]]; then
  eval "$SECRETS_EXPORTS"
fi

# Preflight: every `--phase full` run needs PostgreSQL (reprompt) and Kaggle.
[[ -n "${DB_HOST:-}" ]] \
  || die "DB_HOST is not set. Provide DB_SECRET_NAME / --secret-name or DB_* in .env."
log "✓ Database target: $DB_HOST:${DB_PORT:-5432} (DB: ${DB_NAME:-default})"

if [[ -z "${KAGGLE_USERNAME:-}" && ! -f "$HOME/.kaggle/kaggle.json" ]]; then
  die "No Kaggle credentials. Provide KAGGLE_SECRET_NAME / --kaggle-secret or ~/.kaggle/kaggle.json."
fi
log "✓ Kaggle account: ${KAGGLE_USERNAME:-<from ~/.kaggle/kaggle.json>} (GPU quota owner)"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  die "No LLM API key set (OPENAI_API_KEY). Provide LLM_SECRET_NAME / --llm-secret or set OPENAI_API_KEY in .env."
fi
log "✓ LLM credentials configured"

# Preflight: verify we can commit + push *before* spending hours on the run.
if [[ "$DO_PUBLISH" -eq 1 ]]; then
  if [[ -n "${GIT_PUSH_URL:-}" ]]; then
    git remote set-url --push "$REMOTE" "$GIT_PUSH_URL"
  fi
  # Commit identity: env vars override; fall back to a bot identity if git has none
  if [[ -z "${GIT_AUTHOR_NAME:-}" ]] && ! git config user.name >/dev/null; then
    export GIT_AUTHOR_NAME="ablation-bot"
  fi
  if [[ -z "${GIT_AUTHOR_EMAIL:-}" ]] && ! git config user.email >/dev/null; then
    export GIT_AUTHOR_EMAIL="ablation-bot@users.noreply.github.com"
  fi
  if [[ -n "${GIT_AUTHOR_NAME:-}" ]]; then export GIT_COMMITTER_NAME="${GIT_COMMITTER_NAME:-$GIT_AUTHOR_NAME}"; fi
  if [[ -n "${GIT_AUTHOR_EMAIL:-}" ]]; then export GIT_COMMITTER_EMAIL="${GIT_COMMITTER_EMAIL:-$GIT_AUTHOR_EMAIL}"; fi

  log "Checking push access to $REMOTE/$BRANCH..."
  git push --dry-run --quiet "$REMOTE" "HEAD:refs/heads/$BRANCH" \
    || die "Cannot push to $REMOTE/$BRANCH. Configure GIT_SSH_SECRET_NAME + GIT_PUSH_URL, or pass --no-publish."
  log "✓ Push access OK"
fi

# ── 4. DVC inputs ────────────────────────────────────────────────────────────
# dvc lives in the uv venv → always invoke it through `uv run`.
if [[ -f data/raw.dvc ]] && ! ls data/raw/user/*.csv >/dev/null 2>&1; then
  log "Pulling raw user prompts via DVC (data/raw.dvc)..."
  uv run --quiet dvc pull data/raw.dvc || die "dvc pull data/raw.dvc failed"
fi
if [[ -f data/prompts.dvc ]] && ! ls data/prompts/*.csv >/dev/null 2>&1; then
  log "Pulling prompts via DVC (data/prompts.dvc)..."
  uv run --quiet dvc pull data/prompts.dvc || die "dvc pull data/prompts.dvc failed"
fi

# ── 5. Execution ─────────────────────────────────────────────────────────────
export KAGGLE_ISOLATE_RUNS="${KAGGLE_ISOLATE_RUNS:-1}"
export PYTHONUNBUFFERED=1

# Invoke the runner directly (credentials are already in the environment);
# avoids depending on `uv run --env-file .env`, which doesn't exist on EC2.
log "Running experiment $EXPERIMENT (Phase: full)..."
uv run python -m models.ablation_runner \
  --experiment "$EXPERIMENT" \
  --sample-size "$SAMPLE_SIZE" \
  --seed "$SEED" \
  --phase full \
  --run-id "$RUN_ID"

log "✓ Pipeline execution completed successfully for $EXPERIMENT ($RUN_ID)."

# ── 6. Publication ───────────────────────────────────────────────────────────
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
