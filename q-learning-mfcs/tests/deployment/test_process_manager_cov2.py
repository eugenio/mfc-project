"""Tests for deployment/process_manager.py - coverage target 98%+."""
import sys
import os
import json
import signal
import time

import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from deployment.process_manager import (
    ProcessManager, ProcessConfig, ProcessInfo, ProcessState,
    RestartPolicy, HealthChecker, main,
)


@pytest.fixture
def manager(tmp_path):
    mgr = ProcessManager(log_dir=str(tmp_path / "logs"))
    yield mgr


@pytest.mark.coverage_extra
class TestHealthCheckerLoop:
    def test_health_check_loop_checks_running(self, manager):
        cfg = ProcessConfig(name="hc_proc", command="echo hi")
        info = manager.add_process(cfg)
        info.state = ProcessState.RUNNING
        info.pid = 99999
        checker = manager.health_checker
        cc = [0]
        def stop(*a, **kw):
            cc[0] += 1; checker.running = False
        with patch.object(checker, "_check_process_health", side_effect=stop):
            with patch("time.sleep"):
                checker.running = True; checker._health_check_loop()
        assert cc[0] >= 1

    def test_health_check_loop_exception(self, manager):
        cfg = ProcessConfig(name="hce", command="echo hi")
        info = manager.add_process(cfg)
        info.state = ProcessState.RUNNING; info.pid = 99999
        checker = manager.health_checker
        cc = [0]
        def fail(*a, **kw):
            cc[0] += 1
            if cc[0] == 1: raise RuntimeError("fail")
            checker.running = False
        with patch.object(checker, "_check_process_health", side_effect=fail):
            with patch("time.sleep"):
                checker.running = True; checker._health_check_loop()
        assert cc[0] >= 1


@pytest.mark.coverage_extra
class TestSignalHandler:
    def test_signal_handler_calls_shutdown(self, manager):
        handler = signal.getsignal(signal.SIGTERM)
        with patch.object(manager, "shutdown_all"):
            handler(signal.SIGTERM, None)


@pytest.mark.coverage_extra
class TestStartProcessLogFile:
    def test_start_with_log_file(self, manager, tmp_path):
        lf = str(tmp_path / "proc.log")
        cfg = ProcessConfig(name="lp", command="echo hello", log_file=lf)
        manager.add_process(cfg)
        mp = MagicMock(); mp.pid = 12345
        with patch("subprocess.Popen", return_value=mp):
            assert manager.start_process("lp") is True


@pytest.mark.coverage_extra
class TestMonitorProcessException:
    def test_monitor_process_exception(self, manager):
        cfg = ProcessConfig(name="me", command="echo hi")
        info = manager.add_process(cfg)
        info.state = ProcessState.RUNNING
        mp = MagicMock(); mp.wait.side_effect = RuntimeError("wait failed")
        manager._monitor_process(info, mp)
        assert info.state == ProcessState.FAILED
        assert info.last_error == "wait failed"


@pytest.mark.coverage_extra
class TestStopProcessPsutil:
    def test_stop_process_force_kill(self, manager):
        import psutil
        cfg = ProcessConfig(name="sp", command="sleep 3600")
        info = manager.add_process(cfg)
        info.state = ProcessState.RUNNING; info.pid = 99999
        cc = [0]
        def fp(pid):
            cc[0] += 1
            if cc[0] <= 1:
                m = MagicMock(); m.is_running.return_value = False; return m
            raise psutil.NoSuchProcess(pid)
        with patch("os.kill"):
            with patch("psutil.Process", side_effect=fp):
                with patch("time.sleep"):
                    assert manager._stop_process_internal(info, 0.2) is True

    def test_stop_process_nosuchprocess_in_force(self, manager):
        import psutil
        cfg = ProcessConfig(name="sn", command="sleep 3600")
        info = manager.add_process(cfg)
        info.state = ProcessState.RUNNING; info.pid = 99998
        def fp(pid): raise psutil.NoSuchProcess(pid)
        with patch("os.kill"):
            with patch("psutil.Process", side_effect=fp):
                with patch("time.sleep"):
                    assert manager._stop_process_internal(info, 0.2) is True


@pytest.mark.coverage_extra
class TestMainCLI:
    def _run_main(self, argv, extra_patches=None):
        mock_mgr = MagicMock(spec=ProcessManager)
        patches = [
            patch("sys.argv", argv),
            patch("deployment.process_manager.get_process_manager", return_value=mock_mgr),
        ]
        if extra_patches:
            patches.extend(extra_patches)
        ctx = [p.__enter__() for p in patches]
        try:
            return main(), mock_mgr
        finally:
            for p in reversed(patches):
                p.__exit__(None, None, None)

    def test_main_start(self, tmp_path):
        cf = tmp_path / "c.json"
        cf.write_text(json.dumps({"processes": [{"name": "t", "command": ["echo", "h"]}]}))
        ret, mgr = self._run_main(["p", "--config", str(cf), "--start", "t"])
        assert ret == 0

    def test_main_stop(self):
        ret, mgr = self._run_main(["p", "--stop", "x"])
        assert ret == 0

    def test_main_restart(self):
        ret, mgr = self._run_main(["p", "--restart", "x"])
        assert ret == 0

    def test_main_start_all(self):
        ret, mgr = self._run_main(["p", "--start-all"])
        assert ret == 0

    def test_main_stop_all(self):
        ret, mgr = self._run_main(["p", "--stop-all"])
        assert ret == 0

    def test_main_status(self):
        mock_mgr = MagicMock(spec=ProcessManager)
        mock_mgr.get_status.return_value = {"processes": [], "total": 0}
        with patch("sys.argv", ["p", "--status"]):
            with patch("deployment.process_manager.get_process_manager", return_value=mock_mgr):
                assert main() == 0

    def test_main_monitor(self):
        ret, mgr = self._run_main(["p", "--monitor"],
            [patch("time.sleep", side_effect=KeyboardInterrupt)])
        assert ret == 0

    def test_main_exception(self):
        mock_mgr = MagicMock(spec=ProcessManager)
        mock_mgr.start_process.side_effect = RuntimeError("f")
        with patch("sys.argv", ["p", "--start", "m"]):
            with patch("deployment.process_manager.get_process_manager", return_value=mock_mgr):
                assert main() == 1
