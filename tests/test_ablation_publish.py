"""Unit tests for models/ablation_publish.py."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from models.ablation_publish import (
    _scan_target_dir_files,
    get_changed_dvc_files,
    publish,
    restore_stash,
    stash_files,
)


class TestAblationPublish(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.tmp_dir.name)
        self.target_dir = Path("data/ablations")
        self.target_full = self.repo_root / self.target_dir
        self.target_full.mkdir(parents=True, exist_ok=True)

        # Create dummy data/ablations.dvc
        self.dvc_file = self.repo_root / "data/ablations.dvc"
        self.dvc_file.write_text("outs:\n- md5: test123\n  path: ablations\n")

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_scan_target_dir_files_filters_junk_and_preserves_hierarchy(self) -> None:
        # Create valid files
        (self.target_full / "run1").mkdir(parents=True, exist_ok=True)
        (self.target_full / "run1" / "metrics.json").write_text('{"loss": 0.1}')
        (self.target_full / "summary.csv").write_text("id,val\n1,2\n")

        # Create files that should be ignored
        (self.target_full / ".DS_Store").write_text("junk")
        (self.target_full / "run1" / ".gitkeep").write_text("")
        (self.target_full / "temp.tmp").write_text("tmp")
        cache_dir = self.target_full / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "cached_item.bin").write_text("cache")

        found = _scan_target_dir_files(self.target_dir, self.repo_root)

        expected = [
            Path("data/ablations/run1/metrics.json"),
            Path("data/ablations/summary.csv"),
        ]
        self.assertEqual(found, expected)

    def test_get_changed_dvc_files_from_dvc_status(self) -> None:
        # Create the actual files so is_file() is True
        f1 = self.target_full / "run_a" / "eval.json"
        f1.parent.mkdir(parents=True, exist_ok=True)
        f1.write_text("{}")
        f2 = self.target_full / "run_b" / "result.csv"
        f2.parent.mkdir(parents=True, exist_ok=True)
        f2.write_text("1,2")

        dvc_status_output = json.dumps({
            "committed": {
                "modified": ["data/ablations/run_a/eval.json"],
                "added": [],
            },
            "uncommitted": {
                "modified": [],
                "added": [
                    "data/ablations/run_b/result.csv",
                    "data/other/ignore.csv",  # Outside target_dir
                    "data/ablations/run_b/",  # Directory path itself
                ],
            },
        })

        mock_proc = MagicMock(spec=subprocess.CompletedProcess)
        mock_proc.returncode = 0
        mock_proc.stdout = dvc_status_output

        with patch("models.ablation_publish.run_cmd", return_value=mock_proc) as mock_run:
            files = get_changed_dvc_files(self.target_dir, self.repo_root)
            mock_run.assert_called_once_with(
                ["dvc", "data", "status", "--granular", "--json"],
                cwd=self.repo_root,
                check=True,
                capture_output=True,
            )

        self.assertEqual(
            files,
            [
                Path("data/ablations/run_a/eval.json"),
                Path("data/ablations/run_b/result.csv"),
            ],
        )

    def test_get_changed_dvc_files_fallback_when_dvc_status_empty(self) -> None:
        # Create files on disk
        f1 = self.target_full / "B2_run" / "report.md"
        f1.parent.mkdir(parents=True, exist_ok=True)
        f1.write_text("# Report")

        mock_proc = MagicMock(spec=subprocess.CompletedProcess)
        mock_proc.returncode = 0
        mock_proc.stdout = "{}"

        with patch("models.ablation_publish.run_cmd", return_value=mock_proc):
            files = get_changed_dvc_files(self.target_dir, self.repo_root)

        self.assertEqual(files, [Path("data/ablations/B2_run/report.md")])

    def test_get_changed_dvc_files_fallback_when_dvc_command_fails(self) -> None:
        # Create files on disk
        f1 = self.target_full / "B1_run" / "scores.csv"
        f1.parent.mkdir(parents=True, exist_ok=True)
        f1.write_text("score\n0.85\n")

        with patch(
            "models.ablation_publish.run_cmd",
            side_effect=subprocess.CalledProcessError(1, ["dvc", "data", "status"]),
        ):
            files = get_changed_dvc_files(self.target_dir, self.repo_root)

        self.assertEqual(files, [Path("data/ablations/B1_run/scores.csv")])

    def test_get_changed_dvc_files_returns_empty_when_no_files_exist(self) -> None:
        mock_proc = MagicMock(spec=subprocess.CompletedProcess)
        mock_proc.returncode = 0
        mock_proc.stdout = "{}"

        with patch("models.ablation_publish.run_cmd", return_value=mock_proc):
            files = get_changed_dvc_files(self.target_dir, self.repo_root)

        self.assertEqual(files, [])

    def test_stash_and_restore_files(self) -> None:
        file1_rel = Path("data/ablations/nested/dir/test.txt")
        file2_rel = Path("data/ablations/root_file.json")

        f1 = self.repo_root / file1_rel
        f2 = self.repo_root / file2_rel
        f1.parent.mkdir(parents=True, exist_ok=True)
        f1.write_text("nested content 123")
        f2.write_text('{"status": "ok"}')

        with tempfile.TemporaryDirectory() as stash_tmp:
            stash_dir = Path(stash_tmp)
            stash_files([file1_rel, file2_rel], self.repo_root, stash_dir)

            self.assertTrue((stash_dir / file1_rel).is_file())
            self.assertEqual((stash_dir / file1_rel).read_text(), "nested content 123")
            self.assertTrue((stash_dir / file2_rel).is_file())
            self.assertEqual((stash_dir / file2_rel).read_text(), '{"status": "ok"}')

            # Modify or delete workspace files
            f1.unlink()
            f2.write_text('{"status": "clobbered"}')

            # Restore
            restore_stash(stash_dir, self.repo_root)

            self.assertTrue(f1.is_file())
            self.assertEqual(f1.read_text(), "nested content 123")
            self.assertEqual(f2.read_text(), '{"status": "ok"}')

    def test_publish_passes_relative_dvc_file_path(self) -> None:
        file_rel = Path("data/ablations/run_x/metrics.json")
        f = self.repo_root / file_rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text('{"accuracy": 0.95}')

        executed_cmds: list[list[str]] = []

        def fake_run_cmd(
            cmd: list[str],
            cwd: Path | None = None,
            check: bool = True,
            capture_output: bool = False,
        ) -> subprocess.CompletedProcess[str]:
            executed_cmds.append(cmd)
            stdout = ""
            returncode = 0

            if cmd[:3] == ["git", "rev-parse", "--show-toplevel"]:
                stdout = str(self.repo_root)
            elif cmd[:3] == ["git", "rev-parse", "--abbrev-ref"]:
                stdout = "main"
            elif cmd[:3] == ["git", "status", "--porcelain"]:
                stdout = ""
            elif cmd[:3] == ["dvc", "data", "status"]:
                stdout = json.dumps({"uncommitted": {"added": [file_rel.as_posix()]}})
            elif cmd[:3] == ["git", "diff", "--cached"]:
                # Non-zero returncode means diff exists, so commit + push proceeds
                returncode = 1
            elif cmd[:3] == ["git", "rev-parse", "--short"]:
                stdout = "abc1234"

            return subprocess.CompletedProcess(cmd, returncode=returncode, stdout=stdout, stderr="")

        with patch("models.ablation_publish.run_cmd", side_effect=fake_run_cmd):
            res = publish(
                target="data/ablations",
                branch="main",
                remote="origin",
                run_id="R_TEST",
            )

        self.assertEqual(res, 0)

        # Verify exact dvc commands: must use relative path data/ablations.dvc, NOT ablations.dvc!
        dvc_pull_cmds = [cmd for cmd in executed_cmds if len(cmd) >= 2 and cmd[0] == "dvc" and cmd[1] == "pull"]
        self.assertEqual(len(dvc_pull_cmds), 1)
        self.assertEqual(dvc_pull_cmds[0], ["dvc", "pull", "--force", "data/ablations.dvc"])

        dvc_push_cmds = [cmd for cmd in executed_cmds if len(cmd) >= 2 and cmd[0] == "dvc" and cmd[1] == "push"]
        self.assertEqual(len(dvc_push_cmds), 1)
        self.assertEqual(dvc_push_cmds[0], ["dvc", "push", "data/ablations.dvc"])

        # Also verify dvc add and git add
        dvc_add_cmds = [cmd for cmd in executed_cmds if len(cmd) >= 2 and cmd[0] == "dvc" and cmd[1] == "add"]
        self.assertEqual(dvc_add_cmds[0], ["dvc", "add", "data/ablations"])

        git_add_cmds = [cmd for cmd in executed_cmds if len(cmd) >= 3 and cmd[0] == "git" and cmd[1] == "add"]
        self.assertIn(["git", "add", "data/ablations.dvc"], git_add_cmds)

    def test_publish_no_changes_detected_exits_early(self) -> None:
        def fake_run_cmd(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            if cmd[:3] == ["git", "rev-parse", "--show-toplevel"]:
                return subprocess.CompletedProcess(cmd, 0, stdout=str(self.repo_root))
            if cmd[:3] == ["git", "rev-parse", "--abbrev-ref"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="main")
            if cmd[:3] == ["git", "status", "--porcelain"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="")
            if cmd[:3] == ["dvc", "data", "status"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="{}")
            return subprocess.CompletedProcess(cmd, 0, stdout="")

        with patch("models.ablation_publish.run_cmd", side_effect=fake_run_cmd):
            res = publish(target="data/ablations", branch="main")

        self.assertEqual(res, 0)

    def test_publish_missing_dvc_file_raises_error(self) -> None:
        self.dvc_file.unlink()

        def fake_run_cmd(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            if cmd[:3] == ["git", "rev-parse", "--show-toplevel"]:
                return subprocess.CompletedProcess(cmd, 0, stdout=str(self.repo_root))
            return subprocess.CompletedProcess(cmd, 0, stdout="main")

        with patch("models.ablation_publish.run_cmd", side_effect=fake_run_cmd):
            with self.assertRaises(FileNotFoundError):
                publish(target="data/ablations", branch="main")

    def test_publish_retries_on_concurrent_push_conflict(self) -> None:
        file_rel = Path("data/ablations/result.json")
        f = self.repo_root / file_rel
        f.write_text("{}")

        push_attempts = 0

        def fake_run_cmd(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            nonlocal push_attempts
            if cmd[:3] == ["git", "rev-parse", "--show-toplevel"]:
                return subprocess.CompletedProcess(cmd, 0, stdout=str(self.repo_root))
            if cmd[:3] == ["git", "rev-parse", "--abbrev-ref"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="main")
            if cmd[:3] == ["git", "status", "--porcelain"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="")
            if cmd[:3] == ["dvc", "data", "status"]:
                return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"uncommitted": {"added": [file_rel.as_posix()]}}))
            if cmd[:3] == ["git", "diff", "--cached"]:
                return subprocess.CompletedProcess(cmd, 1)  # has diff
            if cmd[:3] == ["git", "push", "--quiet"]:
                push_attempts += 1
                if push_attempts == 1:
                    return subprocess.CompletedProcess(cmd, 1, stderr="[rejected] non-fast-forward")
                return subprocess.CompletedProcess(cmd, 0, stdout="")
            if cmd[:3] == ["git", "rev-parse", "--short"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="def5678")
            return subprocess.CompletedProcess(cmd, 0, stdout="")

        with patch("models.ablation_publish.run_cmd", side_effect=fake_run_cmd):
            with patch("time.sleep", return_value=None):
                res = publish(target="data/ablations", branch="main", max_attempts=3)

        self.assertEqual(res, 0)
        self.assertEqual(push_attempts, 2)


if __name__ == "__main__":
    unittest.main()
