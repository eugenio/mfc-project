import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import subprocess
from unittest.mock import patch, MagicMock
import pytest

from build_qlearning import run_command, main


class TestRunCommand:
    """Tests for run_command function."""

    def test_run_command_success(self):
        result = run_command("echo hello", "Echo test")
        assert result is True

    def test_run_command_failure(self):
        result = run_command("false", "Always-fail command")
        assert result is False

    def test_run_command_timeout(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 300)):
            result = run_command("sleep 999", "Timeout test")
            assert result is False

    def test_run_command_exception(self):
        with patch("subprocess.run", side_effect=OSError("No such file")):
            result = run_command("nonexistent_cmd", "Exception test")
            assert result is False

    def test_run_command_with_stdout(self):
        result = run_command("echo some_output", "Stdout test")
        assert result is True

    def test_run_command_nonzero_return(self):
        result = run_command("exit 1", "Non-zero return")
        assert result is False


class TestMain:
    """Tests for main function."""

    def test_main_no_odes_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_main_all_steps_succeed(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "odes.mojo").write_text("# fake")
        (tmp_path / "odes.so").write_text("")
        (tmp_path / "mfc_qlearning").write_text("")
        (tmp_path / "mfc_qlearning_demo.py").write_text("")
        with patch("build_qlearning.run_command", return_value=True):
            main()

    def test_main_some_steps_fail(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "odes.mojo").write_text("# fake")
        call_count = [0]
        def mock_run(cmd, desc):
            call_count[0] += 1
            return call_count[0] == 1
        with patch("build_qlearning.run_command", side_effect=mock_run):
            main()

    def test_main_all_steps_fail(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "odes.mojo").write_text("# fake")
        with patch("build_qlearning.run_command", return_value=False):
            main()

    def test_main_file_existence_checks(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "odes.mojo").write_text("# fake")
        (tmp_path / "odes.so").write_text("")
        with patch("build_qlearning.run_command", return_value=True):
            main()

    def test_main_no_output_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "odes.mojo").write_text("# fake")
        with patch("build_qlearning.run_command", return_value=True):
            main()
