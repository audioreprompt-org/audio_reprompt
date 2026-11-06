#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# ablation_publish.sh — concurrency-safe DVC and Git publisher
#
# Publishes new results under a DVC-tracked directory (data/ablations) and
# commits the updated pointer (data/ablations.dvc) with optimistic retry
# when multiple users publish to the same branch concurrently.
#
# Delegates to models.ablation_publish for robust, maintainable execution.
#
# Usage:
#   scripts/ablation_publish.sh --run-id R20261022_204000
#   scripts/ablation_publish.sh --branch main --target data/ablations
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

exec uv run python -m models.ablation_publish "$@"
