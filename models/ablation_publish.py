"""Concurrency-safe DVC and Git publisher for ablation experiments.

Publishes new results under a DVC-tracked directory (data/ablations) and commits
the updated pointer (md5, size, nfiles in data/ablations.dvc) safely when
multiple users or runners publish to the same git branch concurrently.

Workflow:
    1. Identify added/modified files under the target directory via `dvc data status`.
    2. Stash them in a temporary directory.
    3. Loop with optimistic concurrency:
       a. Fetch and reset to remote branch tip.
       b. `dvc pull --force` remote results.
       c. Restore stashed files on top (union).
       d. `dvc add` target directory (computes new hash, total size, file count).
       e. `dvc push` data to remote.
       f. Commit updated .dvc file and push to git.
       g. If git push is rejected due to a race, back off and retry.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[publish %(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def run_cmd(
    cmd: list[str],
    cwd: Path | None = None,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Execute a subprocess command with clear logging."""
    logger.debug(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=capture_output, check=False)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout, stderr=result.stderr)
    return result


def _scan_target_dir_files(target_dir: Path, repo_root: Path) -> list[Path]:
    """Walk repo_root / target_dir directly, skipping cache and system files."""
    full_target = target_dir if target_dir.is_absolute() else (repo_root / target_dir)
    if not full_target.exists() or not full_target.is_dir():
        return []

    collected: list[Path] = []
    ignored_dir_names = {".cache", "__pycache__", ".git", ".dvc"}
    ignored_file_names = {".DS_Store", ".gitkeep"}

    for root, dirs, files in os.walk(full_target):
        # Prune ignored directories in-place so os.walk does not descend
        dirs[:] = [d for d in dirs if d not in ignored_dir_names and not d.startswith(".")]
        for f in files:
            if f in ignored_file_names or f.startswith("._") or f.endswith(".tmp"):
                continue
            file_path = Path(root) / f
            if file_path.is_file():
                try:
                    rel_path = file_path.relative_to(repo_root)
                except ValueError:
                    rel_path = file_path
                collected.append(rel_path)

    return sorted(collected)


def get_changed_dvc_files(target_dir: Path, repo_root: Path, dvc_cmd: str = "dvc") -> list[Path]:
    """Identify new or modified files under target_dir using dvc data status,
    with a fallback to scanning target_dir directly if dvc data status fails
    or returns no changes while files exist on disk.
    """
    cmd = [dvc_cmd, "data", "status", "--granular", "--json"]
    changed: list[Path] = []
    seen: set[str] = set()

    try:
        proc = run_cmd(cmd, cwd=repo_root, check=True, capture_output=True)
        status = json.loads(proc.stdout.strip() or "{}")
        target_prefix = target_dir.as_posix().rstrip("/") + "/"

        # uncommitted: workspace vs .dvc file; committed: .dvc file vs git HEAD
        for section in ("uncommitted", "committed"):
            sec_data = status.get(section, {})
            for kind in ("added", "modified"):
                for p_str in sec_data.get(kind, []):
                    # Ignore directories itself or outside target
                    if p_str.endswith("/") or not p_str.startswith(target_prefix) or p_str in seen:
                        continue
                    seen.add(p_str)
                    full_path = repo_root / p_str
                    if full_path.is_file():
                        changed.append(Path(p_str))
    except Exception as exc:
        logger.warning(f"dvc data status failed ({exc}); falling back to direct scan of {target_dir}")
        changed = []

    if changed:
        return sorted(changed)

    # Fallback: if dvc data status returned empty or failed, but files exist in target_dir,
    # collect all files so fresh runs on uninitialized VMs never silently skip publishing.
    fallback_files = _scan_target_dir_files(target_dir, repo_root)
    if fallback_files:
        logger.info(
            f"dvc data status reported 0 changed files; collected {len(fallback_files)} file(s) via fallback scan of {target_dir}"
        )
        return fallback_files

    return []


def stash_files(file_rel_paths: list[Path], repo_root: Path, stash_dir: Path) -> None:
    """Copy changed files into a temporary stash directory maintaining relative hierarchy."""
    for rel_path in file_rel_paths:
        src = repo_root / rel_path
        dest = stash_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def restore_stash(stash_dir: Path, repo_root: Path) -> None:
    """Restore all files from stash directory into workspace."""
    for root, _, files in os.walk(stash_dir):
        for f in files:
            src = Path(root) / f
            rel = src.relative_to(stash_dir)
            dest = repo_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)


def publish(
    target: str = "data/ablations",
    branch: str = "",
    remote: str = "origin",
    run_id: str = "",
    message: str = "",
    max_attempts: int = 6,
    dvc_cmd: str = "dvc",
    force: bool = False,
) -> int:
    """Publish results to DVC and git with optimistic concurrency and retry."""
    # Find repo root
    rev_parse = run_cmd(["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True)
    repo_root = Path(rev_parse.stdout.strip())

    if not branch:
        branch_proc = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root, check=True, capture_output=True)
        branch = branch_proc.stdout.strip()
        if branch == "HEAD":
            raise RuntimeError("Detached HEAD state; please pass explicit --branch")

    target_path = Path(target).relative_to(repo_root) if Path(target).is_absolute() else Path(target)
    dvc_file_rel = f"{target_path.as_posix()}.dvc"
    dvc_file = repo_root / dvc_file_rel

    if not dvc_file.exists():
        raise FileNotFoundError(f"{dvc_file} does not exist. Is {target_path} tracked by DVC?")

    commit_msg = message or f"ablations: publish results {run_id or '(manual)'}"

    # Check for uncommitted tracked changes
    if not force:
        status_proc = run_cmd(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo_root, capture_output=True)
        if status_proc.stdout.strip():
            logger.error("Tracked git files have uncommitted changes; commit/stash them or pass --force:")
            logger.error(status_proc.stdout.strip())
            return 1

    # 1. Detect changed files
    logger.info(f"Detecting new or modified files under {target_path} ...")
    changed_files = get_changed_dvc_files(target_path, repo_root, dvc_cmd=dvc_cmd)

    if not changed_files:
        logger.info(f"No changes detected under {target_path}/. Nothing to publish.")
        return 0

    logger.info(f"Stashing {len(changed_files)} changed file(s)")

    with tempfile.TemporaryDirectory(prefix="dvc-publish-") as tmpdir:
        stash_dir = Path(tmpdir)
        stash_files(changed_files, repo_root, stash_dir)

        # 2. Concurrency retry loop
        for attempt in range(1, max_attempts + 1):
            logger.info(f"Attempt {attempt}/{max_attempts} — syncing with {remote}/{branch}")

            # Fetch and align with remote tip
            run_cmd(["git", "fetch", "--quiet", remote, branch], cwd=repo_root, check=True)
            run_cmd(["git", "checkout", "--quiet", "-f", "-B", branch, f"{remote}/{branch}"], cwd=repo_root, check=True)

            # Pull remote DVC assets
            run_cmd([dvc_cmd, "pull", "--force", dvc_file_rel], cwd=repo_root, check=True)

            # Restore stashed files on top
            restore_stash(stash_dir, repo_root)

            # Update DVC pointer (computes new md5, size, and nfiles)
            run_cmd([dvc_cmd, "add", target_path.as_posix()], cwd=repo_root, check=True)

            # Push data to DVC remote before committing git pointer
            run_cmd([dvc_cmd, "push", dvc_file_rel], cwd=repo_root, check=True)

            # Check if .dvc file changed
            run_cmd(["git", "add", dvc_file_rel], cwd=repo_root, check=True)
            diff_proc = run_cmd(["git", "diff", "--cached", "--quiet"], cwd=repo_root, check=False)
            if diff_proc.returncode == 0:
                logger.info("Remote already contains these results; nothing new to commit.")
                return 0

            # Commit pointer
            run_cmd(["git", "commit", "--quiet", "-m", commit_msg], cwd=repo_root, check=True)

            # Attempt push
            push_proc = run_cmd(["git", "push", "--quiet", remote, f"HEAD:refs/heads/{branch}"], cwd=repo_root, check=False)
            if push_proc.returncode == 0:
                head_proc = run_cmd(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root, check=True, capture_output=True)
                logger.info(f"✓ Published {head_proc.stdout.strip()} to {remote}/{branch}")
                # Print updated DVC entry
                if dvc_file.exists():
                    logger.info(f"Updated {dvc_file_rel}:\n{dvc_file.read_text().strip()}")
                return 0

            if attempt >= max_attempts:
                logger.error(f"Push still rejected after {max_attempts} attempts.")
                return 1

            backoff = (2**attempt) + random.randint(0, 4)
            logger.warning(f"Push rejected (concurrent commit detected); retrying in {backoff}s...")
            time.sleep(backoff)

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish DVC outputs and commit with concurrency handling")
    parser.add_argument("--target", default="data/ablations", help="DVC target directory (default: data/ablations)")
    parser.add_argument("--branch", default="", help="Git branch (default: current branch)")
    parser.add_argument("--remote", default="origin", help="Git remote (default: origin)")
    parser.add_argument("--run-id", default="", help="Run ID for commit message")
    parser.add_argument("-m", "--message", default="", help="Custom git commit message")
    parser.add_argument("--max-attempts", type=int, default=6, help="Maximum push retry attempts (default: 6)")
    parser.add_argument("--dvc", default=os.getenv("DVC", "dvc"), help="DVC binary command (default: dvc)")
    parser.add_argument("--force", action="store_true", help="Discard local tracked modifications before syncing")

    args = parser.parse_args()

    sys.exit(
        publish(
            target=args.target,
            branch=args.branch,
            remote=args.remote,
            run_id=args.run_id,
            message=args.message,
            max_attempts=args.max_attempts,
            dvc_cmd=args.dvc,
            force=args.force,
        )
    )


if __name__ == "__main__":
    main()
