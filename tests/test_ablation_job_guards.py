"""Unit and integration tests for ablation job guards and Kaggle runner configuration."""

from __future__ import annotations

import fcntl
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
IDLE_GUARD_SCRIPT = SCRIPTS_DIR / "ablation_idle_guard.sh"
RUN_JOB_SCRIPT = SCRIPTS_DIR / "run_ablation_job.sh"


class TestKaggleRunnerConfig(unittest.TestCase):
    """Tests for Kaggle runner timeout and notebook generation cleanup."""

    def test_default_timeout_value(self) -> None:
        """Verify models.kaggle_runner.DEFAULT_TIMEOUT == 32400 when KAGGLE_TIMEOUT is unset."""
        import models.kaggle_runner

        orig_env = os.environ.get("KAGGLE_TIMEOUT")
        try:
            if "KAGGLE_TIMEOUT" in os.environ:
                del os.environ["KAGGLE_TIMEOUT"]
            importlib.reload(models.kaggle_runner)
            self.assertEqual(models.kaggle_runner.DEFAULT_TIMEOUT, 32400)
        finally:
            if orig_env is not None:
                os.environ["KAGGLE_TIMEOUT"] = orig_env
            else:
                os.environ.pop("KAGGLE_TIMEOUT", None)
            importlib.reload(models.kaggle_runner)

    def test_generate_notebook_cleanup_and_summary(self) -> None:
        """Verify _generate_notebook creates .ipynb tracking total_wavs, rmtree unzipped dirs, and prints summary."""
        from models.kaggle_runner import KaggleTarget, _generate_notebook

        with tempfile.TemporaryDirectory() as td:
            staging = Path(td)
            target = KaggleTarget(
                dataset_slug="testuser/test-dataset",
                kernel_slug="testuser/test-kernel",
                kernel_title="Test Kernel",
                kernel_id_no=None,
                staging_dir=staging,
            )
            nb_path = _generate_notebook(target)
            self.assertTrue(nb_path.exists())

            nb_data = json.loads(nb_path.read_text())
            cells = nb_data.get("cells", [])
            all_code = "\n".join(
                cell.get("source", "")
                for cell in cells
                if cell.get("cell_type") == "code"
            )

            # 1. Tracks total_wavs
            self.assertIn("total_wavs = 0", all_code)
            self.assertIn('total_wavs += len(list(sub_dir.glob("*.wav")))', all_code)

            # 2. Calls shutil.make_archive and then shutil.rmtree(sub_dir, ignore_errors=True)
            archive_call = 'shutil.make_archive(str(archive), "zip", str(sub_dir))'
            rmtree_call = "shutil.rmtree(sub_dir, ignore_errors=True)"
            self.assertIn(archive_call, all_code)
            self.assertIn(rmtree_call, all_code)

            archive_pos = all_code.find(archive_call)
            rmtree_pos = all_code.find(rmtree_call)
            self.assertGreater(
                rmtree_pos,
                archive_pos,
                "shutil.rmtree must be called after shutil.make_archive",
            )

            # 3. Prints {total_wavs} WAV files
            self.assertIn("{total_wavs} WAV files", all_code)


class TestIdleGuardAndJobLock(unittest.TestCase):
    """Integration tests for scripts/ablation_idle_guard.sh and scripts/run_ablation_job.sh."""

    def setUp(self) -> None:
        self.test_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.test_dir.name)
        self.fake_bin = self.temp_path / "bin"
        self.fake_bin.mkdir(parents=True, exist_ok=True)

        self.shutdown_marker = self.temp_path / "shutdown_marker"
        self.systemctl_output_file = self.temp_path / "systemctl_out"
        self.lock_dir = self.temp_path / "locks"
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self.uptime_file = self.temp_path / "uptime"
        self.last_active_file = self.temp_path / "last_active"
        self.override_file = self.temp_path / "override"

        # Create fake shutdown binary
        shutdown_script = self.fake_bin / "shutdown"
        shutdown_script.write_text(
            "#!/bin/sh\n"
            'if [ -n "$SHUTDOWN_MARKER" ]; then\n'
            '    touch "$SHUTDOWN_MARKER"\n'
            '    echo "shutdown called: $*" >> "$SHUTDOWN_MARKER"\n'
            "fi\n"
            "exit 0\n"
        )
        shutdown_script.chmod(0o755)

        # Create fake cloud-init binary
        cloud_init_script = self.fake_bin / "cloud-init"
        cloud_init_script.write_text("#!/bin/sh\nexit 0\n")
        cloud_init_script.chmod(0o755)

        # Create fake sudo binary
        sudo_script = self.fake_bin / "sudo"
        sudo_script.write_text('#!/bin/sh\nexec "$@"\n')
        sudo_script.chmod(0o755)

        # Create fake systemctl binary
        systemctl_script = self.fake_bin / "systemctl"
        systemctl_script.write_text(
            "#!/bin/sh\n"
            'if [ -n "$SYSTEMCTL_OUTPUT_FILE" ] && [ -f "$SYSTEMCTL_OUTPUT_FILE" ]; then\n'
            '    cat "$SYSTEMCTL_OUTPUT_FILE"\n'
            'elif [ -n "$FAKE_SYSTEMCTL_OUTPUT" ]; then\n'
            '    printf "%s\\n" "$FAKE_SYSTEMCTL_OUTPUT"\n'
            "fi\n"
            "exit 0\n"
        )
        systemctl_script.chmod(0o755)

        # If flock is not available natively in PATH (e.g. macOS), provide a Python flock CLI shim
        if shutil.which("flock") is None:
            flock_script = self.fake_bin / "flock"
            flock_script.write_text(
                "#!/usr/bin/env python3\n"
                "import sys, os, fcntl, time, subprocess\n"
                "def main():\n"
                "    args = sys.argv[1:]\n"
                "    nonblock = False\n"
                "    shared = False\n"
                "    unlock = False\n"
                "    timeout = None\n"
                "    target = None\n"
                "    cmd = []\n"
                "    i = 0\n"
                "    while i < len(args):\n"
                "        a = args[i]\n"
                "        if a in ('-n', '--nonblock', '--nb'):\n"
                "            nonblock = True\n"
                "        elif a in ('-x', '--exclusive'):\n"
                "            shared = False\n"
                "        elif a in ('-s', '--shared'):\n"
                "            shared = True\n"
                "        elif a in ('-u', '--unlock'):\n"
                "            unlock = True\n"
                "        elif a in ('-w', '--timeout'):\n"
                "            i += 1\n"
                "            timeout = float(args[i])\n"
                "        elif a.startswith('-'):\n"
                "            pass\n"
                "        else:\n"
                "            target = a\n"
                "            cmd = args[i+1:]\n"
                "            break\n"
                "        i += 1\n"
                "    if not target:\n"
                "        sys.exit(1)\n"
                "    op = fcntl.LOCK_UN if unlock else (fcntl.LOCK_SH if shared else fcntl.LOCK_EX)\n"
                "    if nonblock:\n"
                "        op |= fcntl.LOCK_NB\n"
                "    try:\n"
                "        fd = int(target)\n"
                "        close_fd = False\n"
                "    except ValueError:\n"
                "        fd = os.open(target, os.O_RDONLY)\n"
                "        close_fd = True\n"
                "    start = time.time()\n"
                "    while True:\n"
                "        try:\n"
                "            fcntl.flock(fd, op)\n"
                "            break\n"
                "        except (OSError, IOError):\n"
                "            if nonblock:\n"
                "                sys.exit(1)\n"
                "            if timeout is not None and (time.time() - start) >= timeout:\n"
                "                sys.exit(1)\n"
                "            time.sleep(0.05)\n"
                "    if cmd:\n"
                "        res = subprocess.run(cmd)\n"
                "        if close_fd:\n"
                "            fcntl.flock(fd, fcntl.LOCK_UN)\n"
                "            os.close(fd)\n"
                "        sys.exit(res.returncode)\n"
                "if __name__ == '__main__':\n"
                "    main()\n"
            )
            flock_script.chmod(0o755)

        self.base_env = os.environ.copy()
        self.base_env["PATH"] = f"{self.fake_bin}:{self.base_env.get('PATH', '')}"
        self.base_env["SHUTDOWN_MARKER"] = str(self.shutdown_marker)
        self.base_env["SYSTEMCTL_OUTPUT_FILE"] = str(self.systemctl_output_file)
        self.base_env["ABLATION_LOCK_DIR"] = str(self.lock_dir)
        self.base_env["ABLATION_UPTIME_FILE"] = str(self.uptime_file)
        self.base_env["ABLATION_LAST_ACTIVE_FILE"] = str(self.last_active_file)
        self.base_env["ABLATION_OVERRIDE_FILE"] = str(self.override_file)
        self.base_env["IDLE_GRACE_PERIOD_SEC"] = "1800"
        self.base_env["ABLATION_SHUTDOWN_SLEEP_SEC"] = "0"
        self.base_env["ABLATION_HANDOFF_SLEEP_SEC"] = "0"

    def tearDown(self) -> None:
        self.test_dir.cleanup()

    def _exec_script(
        self, script_path: Path, args: list[str], env: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        if os.access(script_path, os.X_OK):
            cmd = [str(script_path), *args]
        else:
            cmd = ["bash", str(script_path), *args]
        return subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_idle_guard_activating_systemd_unit(self) -> None:
        """Activating systemd unit (regression test for Type=oneshot/activating bug)."""
        self.systemctl_output_file.write_text(
            "ablation@B3.service loaded activating start Audio Reprompt Ablation Runner (B3)\n"
        )
        self.uptime_file.write_text("25000\n")

        res = self._exec_script(IDLE_GUARD_SCRIPT, [], self.base_env)

        self.assertEqual(res.returncode, 0)
        self.assertFalse(self.shutdown_marker.exists())
        self.assertTrue(self.last_active_file.exists())
        self.assertEqual(self.last_active_file.read_text().strip(), "25000")

    def test_idle_guard_held_lock_prevents_shutdown(self) -> None:
        """Held directory lock prevents shutdown even if no systemd unit is active."""
        self.systemctl_output_file.write_text("")
        self.uptime_file.write_text("25000\n")

        fd = os.open(str(self.lock_dir), os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            res = self._exec_script(IDLE_GUARD_SCRIPT, [], self.base_env)
            self.assertEqual(res.returncode, 0)
            self.assertFalse(self.shutdown_marker.exists())
            self.assertTrue(self.last_active_file.exists())
            self.assertEqual(self.last_active_file.read_text().strip(), "25000")
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def test_idle_guard_grace_window_vs_shutdown(self) -> None:
        """True inactivity grace window vs shutdown."""
        self.systemctl_output_file.write_text("")
        self.last_active_file.write_text("25000\n")

        # 1. Fake uptime 26000 (1000s < 1800s idle) -> assert no shutdown called
        self.uptime_file.write_text("26000\n")
        res1 = self._exec_script(IDLE_GUARD_SCRIPT, [], self.base_env)
        self.assertEqual(res1.returncode, 0)
        self.assertFalse(self.shutdown_marker.exists())

        # 2. Fake uptime 26900 (1900s >= 1800s idle) -> assert shutdown IS called
        self.uptime_file.write_text("26900\n")
        res2 = self._exec_script(IDLE_GUARD_SCRIPT, [], self.base_env)
        self.assertTrue(self.shutdown_marker.exists())

    def test_run_job_no_wait_fails_fast_when_lock_held(self) -> None:
        """--no-wait fails fast when lock is held."""
        fd = os.open(str(self.lock_dir), os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            res = self._exec_script(
                RUN_JOB_SCRIPT,
                ["--experiment", "B2", "--no-sync", "--no-publish", "--no-wait"],
                self.base_env,
            )
            self.assertNotEqual(res.returncode, 0)
            output = res.stdout + res.stderr
            self.assertIn("Another ablation job is already running", output)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def test_run_job_queueing_waits_for_lock_release(self) -> None:
        """Queueing waits for lock release, then acquires lock and executes."""
        lock_holder_code = f"""
import fcntl, os, sys, time
fd = os.open({repr(str(self.lock_dir))}, os.O_RDONLY)
fcntl.flock(fd, fcntl.LOCK_EX)
sys.stdout.write("LOCKED\\n")
sys.stdout.flush()
time.sleep(1.2)
fcntl.flock(fd, fcntl.LOCK_UN)
os.close(fd)
"""
        holder_proc = subprocess.Popen(
            [sys.executable, "-c", lock_holder_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            ready_line = holder_proc.stdout.readline() if holder_proc.stdout else ""
            self.assertEqual(ready_line.strip(), "LOCKED")

            env = self.base_env.copy()
            env["DB_HOST"] = ""

            res = self._exec_script(
                RUN_JOB_SCRIPT,
                ["--experiment", "B2", "--no-sync", "--no-publish"],
                env,
            )
            output = res.stdout + res.stderr
            self.assertIn("waiting in queue", output)
            self.assertIn("Execution lock acquired", output)
        finally:
            holder_proc.wait(timeout=10)
            if holder_proc.stdout:
                holder_proc.stdout.close()
            if holder_proc.stderr:
                holder_proc.stderr.close()

    def test_run_job_shutdown_skipped_when_another_unit_active(self) -> None:
        """Shutdown skipped when another systemd unit is active."""
        self.systemctl_output_file.write_text(
            "ablation@B2.service loaded active running Audio Reprompt Ablation Runner (B2)\n"
            "ablation@B3.service loaded active running Audio Reprompt Ablation Runner (B3)\n"
        )
        env = self.base_env.copy()
        env["ABLATION_UNIT_NAME"] = "ablation@B2.service"
        env["DB_HOST"] = ""

        res = self._exec_script(
            RUN_JOB_SCRIPT,
            ["--experiment", "B2", "--no-sync", "--no-publish", "--shutdown"],
            env,
        )
        output = res.stdout + res.stderr
        self.assertIn("skipping VM shutdown", output)
        self.assertFalse(self.shutdown_marker.exists())

    def test_idle_guard_stale_prereboot_state_is_ignored(self) -> None:
        """Stale pre-reboot last_active (> current uptime) is treated as 0."""
        self.systemctl_output_file.write_text("")
        # Simulate stale pre-reboot value 50000 while current boot uptime is 2000 (>= 1800 grace)
        self.last_active_file.write_text("50000\n")
        self.uptime_file.write_text("2000\n")

        res = self._exec_script(IDLE_GUARD_SCRIPT, [], self.base_env)
        self.assertEqual(res.returncode, 0)
        self.assertTrue(self.shutdown_marker.exists())

    def test_systemd_unit_and_makefile_ssm_json_validity(self) -> None:
        """Verify ablation@.service configuration and Makefile SSM JSON payload validity."""
        unit_text = (REPO_ROOT / "deploy" / "systemd" / "ablation@.service").read_text()
        self.assertIn("Type=exec", unit_text)
        self.assertIn("RuntimeMaxSec=86400", unit_text)
        self.assertIn("EnvironmentFile=-/run/ablation/%i.env", unit_text)
        self.assertIn('Environment="ABLATION_UNIT_NAME=%n"', unit_text)

        # Ensure ExecStopPost single-quoted bash script contains no nested single quotes
        stop_post_lines = [
            line for line in unit_text.splitlines() if line.startswith("ExecStopPost=")
        ]
        self.assertEqual(len(stop_post_lines), 1)
        self.assertEqual(stop_post_lines[0].count("'"), 2)

        # Verify Makefile ablation-ec2-run renders valid JSON commands array for AWS SSM
        make_dry = subprocess.run(
            ["make", "-n", "ablation-ec2-run", "EXPERIMENT=B3", "SAMPLE_SIZE=100", "SEED=42"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        # Evaluate the shell quoting of the --parameters argument to ensure AWS CLI receives valid JSON
        marker = "--parameters "
        idx = make_dry.stdout.index(marker) + len(marker)
        end_idx = make_dry.stdout.index(" --query", idx)
        raw_shell_arg = make_dry.stdout[idx:end_idx]
        unquoted = subprocess.run(
            ["/bin/sh", "-c", f"printf '%s' {raw_shell_arg}"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout
        self.assertTrue(unquoted.startswith("commands="))
        commands_list = json.loads(unquoted[len("commands="):])
        self.assertIsInstance(commands_list, list)
        self.assertTrue(any("/run/ablation/B3.env" in c for c in commands_list))


if __name__ == "__main__":
    unittest.main()
