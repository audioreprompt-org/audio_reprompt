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

case "$EXP" in
  all|generation|B1|B2|B3|B1[a-d]|B2[a-b]|B3[a-e])
    ;;
  *)
    log "Invalid '$TAG_KEY' value '$EXP' (expected B1, B2, B3, B1a-B1d, B2a-B2b, B3a-B3e, all, or generation); skipping auto-trigger."
    exit 0
    ;;
esac

log "Found $TAG_KEY=$EXP → starting ablation@$EXP.service"
systemctl start --no-block "ablation@$EXP.service"
