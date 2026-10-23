#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# ablation_autostart.sh — boot-time trigger for the EC2 ablation runner.
#
# Invoked by ablation-autostart.service on every boot (runs as root). If the
# instance carries an `AblationExperiment` tag (e.g. B2), starts
# ablation@<EXP>.service without blocking boot. No tag → no-op, so the VM can
# be started for manual maintenance by removing the tag first.
#
# Reads tags from IMDSv2, which requires `instance_metadata_tags = "enabled"`
# (set in terraform/modules/ec2_runner). No AWS CLI / IAM permission needed.
#
#   Change experiment:  aws ec2 create-tags --resources <id> --tags Key=AblationExperiment,Value=B1
#   Disable autostart:  aws ec2 delete-tags --resources <id> --tags Key=AblationExperiment
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

TAG_KEY="${ABLATION_TAG_KEY:-AblationExperiment}"
IMDS="http://169.254.169.254/latest"

log() { echo "[autostart] $*"; }

TOKEN=$(curl -fsS -X PUT "$IMDS/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 300" \
  --retry 5 --retry-connrefused --retry-delay 2) || {
  log "IMDS not reachable; skipping auto-trigger."
  exit 0
}

# 404 when the tag is absent (or metadata tags are disabled) → empty value
EXP=$(curl -fsS -H "X-aws-ec2-metadata-token: $TOKEN" \
  "$IMDS/meta-data/tags/instance/$TAG_KEY" 2>/dev/null || true)

if [[ -z "$EXP" ]]; then
  log "No '$TAG_KEY' tag on this instance; nothing to run."
  exit 0
fi

if [[ ! "$EXP" =~ ^[A-Za-z0-9]+$ ]]; then
  log "Invalid '$TAG_KEY' value '$EXP' (expected e.g. B1, B2, B3a)."
  exit 1
fi

log "Found $TAG_KEY=$EXP → starting ablation@$EXP.service"
systemctl start --no-block "ablation@$EXP.service"
