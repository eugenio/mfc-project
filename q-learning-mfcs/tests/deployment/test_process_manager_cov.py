"""Tests for deployment/process_manager.py - coverage target 98%+."""
import sys
import os
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from deployment.process_manager import (
    ProcessState,
    RestartPolicy,
    ProcessConfig,
    ProcessInfo,
    HealthChecker,
    ProcessManager,
    get_process_manager,
)


class TestProcessState:
    def test_all_states(self):
        assert ProcessState.STARTING.value == "starting"
        assert ProcessState.RUNNING.value == "running"
        assert ProcessState.STOPPING.value == "stopping"
        assert ProcessState.STOPPED.value == "stopped"
        assert ProcessState.FAILED.value == "failed"
        assert ProcessState.RESTARTING.value == "restarting"
        assert ProcessState.UNKNOWN.value == "unknown"


class TestRestartPolicy:
    def test_all_policies(self):
        assert RestartPolicy.NEVER.value == "never"
        assert RestartPolicy.ON_FAILURE.value == "on_failure"
        assert RestartPolicy.ALWAYS.value == "always"
        assert RestartPolicy.UNLESS_STOPPED.value == "unless_stopped"


class TestProcessConfig:
    def test_defaults(self):
        cfg = ProcessConfig(name="test", command="echo hello")
        assert cfg.name == "test"
        assert cfg.command == ["echo", "hello"]
        assert cfg.restart_policy == RestartPolicy.ON_FAILURE
        assert cfg.max_restarts == 5

    def test_list_command(self):
        cfg = ProcessConfig(name="test", command=["ls", "-la"])
        assert cfg.command == ["ls", "-la"]

    def test_empty_command_raises(self):
        with pytest.raises(ValueError, match="Command cannot be empty"):
            ProcessConfig(name="test", command="")

    def test_custom_config(self):
        cfg = ProcessConfig(
            name="svc", command="python app.py",
            working_dir="/tmp", environment={"FOO": "bar"},
            restart_policy=RestartPolicy.ALWAYS, max_restarts=10,
            restart_delay=2.0, health_check_command="curl localhost",
            memory_limit=512, cpu_limit=80.0,
            log_file="/tmp/svc.log", user="app", group="app",
        )
        assert cfg.memory_limit == 512
        assert cfg.cpu_limit == 80.0


class TestProcessInfo:
    def test_defaults(self):
        cfg = ProcessConfig(name="test", command="echo")
        info = ProcessInfo(config=cfg)
        assert info.pid is None
        assert info.state == ProcessState.STOPPED
        assert info.restart_count == 0

    def test_uptime_running(self):
        cfg = ProcessConfig(name="test", command="echo")
        info = ProcessInfo(config=cfg)
        info.state = ProcessState.RUNNING
        info.start_time = datetime.now() - timedelta(hours=1)
        assert info.uptime is not None
        assert info.uptime.total_seconds() > 3500

    def test_uptime_stopped(self):
        cfg = ProcessConfig(name="test", command="echo")
        info = ProcessInfo(config=cfg)
        info.state = ProcessState.STOPPED
        assert info.uptime is None

    def test_uptime_no_start(self):
        cfg = ProcessConfig(name="test", command="echo")
        info = ProcessInfo(config=cfg)
        info.state = ProcessState.RUNNING
        info.start_time = None
        assert info.uptime is None


class TestHealthChecker:
    def test_start_stop(self):
        pm = MagicMock()
        pm._processes = {}
        hc = HealthChecker(pm)
        hc.start()
        assert hc.running is True
        hc.start()  # idempotent
        time.sleep(0.1)
        hc.stop()
        assert hc.running is False

    def test_is_process_alive_no_pid(self):
        pm = MagicMock()
        hc = HealthChecker(pm)
        cfg = ProcessConfig(name="t", command="echo")
        info = ProcessInfo(config=cfg, pid=None)
        assert hc._is_process_alive(info) is False

    def test_is_process_alive_ok(self):
        pm = MagicMock()
        hc = HealthChecker(pm)
        cfg = ProcessConfig(name="t", command="echo")
        info = ProcessInfo(config=cfg, pid=1234)
        mp = MagicMock()
        mp.is_running.return_value = True
        with patch("deployment.process_manager.psutil.Process", return_value=mp):
            assert hc._is_process_alive(info) is True

    def test_is_process_alive_no_such(self):
        import psutil
        pm = MagicMock()
        hc = HealthChecker(pm)
        cfg = ProcessConfig(name="t", command="echo")
        info = ProcessInfo(config=cfg, pid=1234)
        with patch(
            "deployment.process_manager.psutil.Process",
            side_effect=psutil.NoSuchProcess(1234),
        ):
            assert hc._is_process_alive(info) is False

    def test_run_health_check_command_ok(self):
        pm = MagicMock()
        hc = HealthChecker(pm)
        cfg = ProcessConfig(name="t", command="echo", health_check_command="true")
        info = ProcessInfo(config=cfg)
        mr = MagicMock(returncode=0)
        with patch("deployment.process_manager.subprocess.run", return_value=mr):
            hc._run_health_check_command(info)

    def test_run_health_check_command_fail(self):
        pm = MagicMock()
        hc = HealthChecker(pm)
        cfg = ProcessConfig(name="t", command="echo", health_check_command="false")
        info = ProcessInfo(config=cfg)
        mr = MagicMock(returncode=1, stderr="err")
        with patch("deployment.process_manager.subprocess.run", return_value=mr):
            hc._run_health_check_command(info)

    def test_run_health_check_command_timeout(self):
        import subprocess
        pm = MagicMock()
        hc = HealthChecker(pm)
        cfg = ProcessConfig(name="t", command="echo", health_check_command="sleep 99")
        info = ProcessInfo(config=cfg)
        with patch(
            "deployment.process_manager.subprocess.run",
            side_effect=subprocess.TimeoutExpired("cmd", 10),
        ):
            hc._run_health_check_command(info)

    def test_run_health_check_command_error(self):
        pm = MagicMock()
        hc = HealthChecker(pm)
        cfg = ProcessConfig(name="t", command="echo", health_check_command="bad")
        info = ProcessInfo(config=cfg)
        with patch(
            "deployment.process_manager.subprocess.run",
            side_effect=Exception("err"),
        ):
            hc._run_health_check_command(info)

    def test_check_resource_limits_no_pid(self):
        pm = MagicMock()
        hc = HealthChecker(pm)
        cfg = ProcessConfig(name="t", command="echo")
        info = ProcessInfo(config=cfg, pid=None)
        hc._check_resource_limits(info)

    def test_check_resource_limits_ok(self):
        pm = MagicMock()
        hc = HealthChecker(pm)
        cfg = ProcessConfig(name="t", command="echo", memory_limit=512, cpu_limit=80)
        info = ProcessInfo(config=cfg, pid=1234)
        mp = MagicMock()
        mp.cpu_percent.return_value = 10.0
        mp.memory_info.return_value = MagicMock(rss=100 * 1024 * 1024)
        mp.memory_percent.return_value = 5.0
        with patch("deployment.process_manager.psutil.Process", return_value=mp):
            hc._check_resource_limits(info)
        assert info.cpu_percent == 10.0

    def test_check_resource_limits_exceeded(self):
        pm = MagicMock()
        hc = HealthChecker(pm)
        cfg = ProcessConfig(name="t", command="echo", memory_limit=50, cpu_limit=10)
        info = ProcessInfo(config=cfg, pid=1234)
        mp = MagicMock()
        mp.cpu_percent.return_value = 90.0
        mp.memory_info.return_value = MagicMock(rss=100 * 1024 * 1024)
        mp.memory_percent.return_value = 50.0
        with patch("deployment.process_manager.psutil.Process", return_value=mp):
            hc._check_resource_limits(info)

    def test_check_resource_limits_no_such(self):
        import psutil
        pm = MagicMock()
        hc = HealthChecker(pm)
        cfg = ProcessConfig(name="t", command="echo")
        info = ProcessInfo(config=cfg, pid=1234)
        with patch(
            "deployment.process_manager.psutil.Process",
            side_effect=psutil.NoSuchProcess(1234),
        ):
            hc._check_resource_limits(info)

    def test_check_process_health_dead(self):
        pm = MagicMock()
        hc = HealthChecker(pm)
        cfg = ProcessConfig(name="t", command="echo")
        info = ProcessInfo(config=cfg, pid=1234)
        with patch.object(hc, "_is_process_alive", return_value=False):
            hc._check_process_health(info)
            pm._handle_process_death.assert_called_once()

    def test_check_process_health_alive_with_hc(self):
        pm = MagicMock()
        hc = HealthChecker(pm)
        cfg = ProcessConfig(name="t", command="echo", health_check_command="true")
        info = ProcessInfo(config=cfg, pid=1234)
        with patch.object(hc, "_is_process_alive", return_value=True):
            with patch.object(hc, "_run_health_check_command"):
                with patch.object(hc, "_check_resource_limits"):
                    hc._check_process_health(info)

    def test_check_process_health_exception(self):
        pm = MagicMock()
        hc = HealthChecker(pm)
        cfg = ProcessConfig(name="t", command="echo")
        info = ProcessInfo(config=cfg, pid=1234)
        with patch.object(hc, "_is_process_alive", side_effect=Exception("err")):
            hc._check_process_health(info)


class TestProcessManager:
    def test_init(self, tmp_path):
        mgr = ProcessManager(str(tmp_path / "pm"))
        assert mgr.log_dir.exists()
        mgr.shutdown_all()

    def test_add_process(self, tmp_path):
        mgr = ProcessManager(str(tmp_path / "pm"))
        cfg = ProcessConfig(name="svc", command="echo hi")
        info = mgr.add_process(cfg)
        assert info.config.name == "svc"
        mgr.shutdown_all()

    def test_add_process_duplicate(self, tmp_path):
        mgr = ProcessManager(str(tmp_path / "pm"))
        cfg = ProcessConfig(name="svc", command="echo hi")
        mgr.add_process(cfg)
        with pytest.raises(ValueError, match="already exists"):
            mgr.add_process(cfg)
        mgr.shutdown_all()

    def test_remove_process(self, tmp_path):
        mgr = ProcessManager(str(tmp_path / "pm"))
        cfg = ProcessConfig(name="svc", command="echo hi")
        mgr.add_process(cfg)
        assert mgr.remove_process("svc") is True
        assert mgr.remove_process("svc") is False
        mgr.shutdown_all()

    def test_start_process_not_found(self, tmp_path):
        mgr = ProcessManager(str(tmp_path / "pm"))
        assert mgr.start_process("bad") is False
        mgr.shutdown_all()

    def test_start_process_already_running(self, tmp_path):
        mgr = ProcessManager(str(tmp_path / "pm"))
        cfg = ProcessConfig(name="svc", command="echo hi")
        mgr.add_process(cfg)
        mgr._processes["svc"].state = ProcessState.RUNNING
        assert mgr.start_process("svc") is True
        mgr.shutdown_all()

    def test_start_process_success(self, tmp_path):
        mgr = ProcessManager(str(tmp_path / "pm"))
        cfg = ProcessConfig(name="svc", command="echo hi")
        mgr.add_process(cfg)
        mp = MagicMock()
        mp.pid = 12345
        with patch("deployment.process_manager.subprocess.Popen", return_value=mp):
            result = mgr.start_process("svc")
            assert result is True
            assert mgr._processes["svc"].pid == 12345
        mgr.shutdown_all()

    def test_start_process_failure(self, tmp_path):
        mgr = ProcessManager(str(tmp_path / "pm"))
        cfg = ProcessConfig(name="svc", command="echo hi")
        mgr.add_process(cfg)
        with patch(
            "deployment.process_manager.subprocess.Popen",
            side_effect=Exception("fail"),
        ):
            result = mgr.start_process("svc")
            assert result is False
            assert mgr._processes["svc"].state == ProcessState.FAILED
        mgr.shutdown_all()

    def test_start_with_callbacks(self, tmp_path):
        mgr = ProcessManager(str(tmp_path / "pm"))
        start_cb = MagicMock()
        mgr.on_process_start = start_cb
        cfg = ProcessConfig(name="svc", command="echo hi")
        mgr.add_process(cfg)
        mp = MagicMock()
        mp.pid = 100
        with patch("deployment.process_manager.subprocess.Popen", return_value=mp):
            mgr.start_process("svc")
            start_cb.assert_called_once()
        mgr.shutdown_all()

    def test_start_fail_callback(self, tmp_path):
        mgr = ProcessManager(str(tmp_path / "pm"))
        fail_cb = MagicMock()
        mgr.on_process_fail = fail_cb
        cfg = ProcessConfig(name="svc", command="echo hi")
        mgr.add_process(cfg)
        with patch(
            "deployment.process_manager.subprocess.Popen",
            side_effect=Exception("fail"),
        ):
            mgr.start_process("svc")
            fail_cb.assert_called_once()
        mgr.shutdown_all()

    def test_stop_process_not_found(self, tmp_path):
        mgr = ProcessManager(str(tmp_path / "pm"))
        assert mgr.stop_process("bad") is False
        mgr.shutdown_all()

    def test_stop_process_already_stopped(self, tmp_path):
        mgr = ProcessManager(str(tmp_path / "pm"))
        cfg = ProcessConfig(name="svc", command="echo hi")
        mgr.add_process(cfg)
        assert mgr.stop_process("svc") is True
        mgr.shutdown_all()

    def test_stop_process_no_pid(self, tmp_path):
        mgr = ProcessManager(str(tmp_path / "pm"))
        cfg = ProcessConfig(name="svc", command="echo hi")
        mgr.add_process(cfg)
        mgr._processes["svc"].state = ProcessState.RUNNING
        mgr._processes["svc"].pid = None
        assert mgr.stop_process("svc") is False
        mgr.shutdown_all()

    def test_stop_process_success(self, tmp_path):
        import psutil
        mgr = ProcessManager(str(tmp_path / "pm"))
        cfg = ProcessConfig(name="svc", command="echo hi")
        mgr.add_process(cfg)
        mgr._processes["svc"].state = ProcessState.RUNNING
        mgr._processes["svc"].pid = 12345
        mp = MagicMock()
        mp.is_running.return_value = False
        with patch("deployment.process_manager.os.kill"):
            with patch("deployment.process_manager.psutil.Process", return_value=mp):
                result = mgr.stop_process("svc")
                assert result is True
        mgr.shutdown_all()

    def test_stop_process_force_kill(self, tmp_path):
        mgr = ProcessManager(str(tmp_path / "pm"))
        cfg = ProcessConfig(name="svc", command="echo hi")
        mgr.add_process(cfg)
        mgr._processes["svc"].state = ProcessState.RUNNING
        mgr._processes["svc"].pid = 12345
        mp = MagicMock()
        mp.is_running.return_value = True
        with patch("deployment.process_manager.os.kill"):
            with patch("deployment.process_manager.psutil.Process", return_value=mp):
                with patch("deployment.process_manager.time.sleep"):
                    result = mgr.stop_process("svc", timeout=0.01)
                    assert result is True
        mgr.shutdown_all()

    def test_stop_process_error(self, tmp_path):
        mgr = ProcessManager(str(tmp_path / "pm"))
        cfg = ProcessConfig(name="svc", command="echo hi")
        mgr.add_process(cfg)
        mgr._processes["svc"].state = ProcessState.RUNNING
        mgr._processes["svc"].pid = 12345
        with patch(
            "deployment.process_manager.os.kill", side_effect=Exception("fail")
        ):
            result = mgr.stop_process("svc")
            assert result is False
        mgr.shutdown_all()

    def test_restart_process(self, tmp_path):
        mgr = ProcessManager(str(tmp_path / "pm"))
        cfg = ProcessConfig(name="svc", command="echo hi")
        mgr.add_process(cfg)
        with patch.object(mgr, "stop_process"):
            with patch.object(mgr, "start_process", return_value=True):
                with patch("deployment.process_manager.time.sleep"):
                    result = mgr.restart_process("svc")
                    assert result is True
        mgr.shutdown_all()

    def test_get_process_info(self, tmp_path):
        mgr = ProcessManager(str(tmp_path / "pm"))
        assert mgr.get_process_info("bad") is None
        cfg = ProcessConfig(name="svc", command="echo hi")
        mgr.add_process(cfg)
        assert mgr.get_process_info("svc") is not None
        mgr.shutdown_all()

    def test_list_processes(self, tmp_path):
        mgr = ProcessManager(str(tmp_path / "pm"))
        cfg = ProcessConfig(name="svc", command="echo hi")
        mgr.add_process(cfg)
        procs = mgr.list_processes()
        assert "svc" in procs
        mgr.shutdown_all()

    def test_get_status(self, tmp_path):
        mgr = ProcessManager(str(tmp_path / "pm"))
        cfg = ProcessConfig(name="svc", command="echo hi")
        mgr.add_process(cfg)
        status = mgr.get_status()
        assert status["total_processes"] == 1
        assert status["running_processes"] == 0
        mgr.shutdown_all()

    def test_start_all(self, tmp_path):
        mgr = ProcessManager(str(tmp_path / "pm"))
        cfg1 = ProcessConfig(name="s1", command="echo 1")
        cfg2 = ProcessConfig(name="s2", command="echo 2")
        mgr.add_process(cfg1)
        mgr.add_process(cfg2)
        with patch.object(mgr, "start_process", return_value=True):
            results = mgr.start_all()
            assert all(results.values())
        mgr.shutdown_all()

    def test_stop_all(self, tmp_path):
        mgr = ProcessManager(str(tmp_path / "pm"))
        cfg = ProcessConfig(name="s1", command="echo 1")
        mgr.add_process(cfg)
        with patch.object(mgr, "stop_process", return_value=True):
            results = mgr.stop_all()
            assert all(results.values())
        mgr.shutdown_all()

    def test_save_load_config(self, tmp_path):
        mgr = ProcessManager(str(tmp_path / "pm"))
        cfg = ProcessConfig(name="svc", command=["echo", "hi"], memory_limit=256)
        mgr.add_process(cfg)
        config_file = str(tmp_path / "config.json")
        mgr.save_config(config_file)
        mgr2 = ProcessManager(str(tmp_path / "pm2"))
        mgr2.load_config(config_file)
        assert "svc" in mgr2._processes
        assert mgr2._processes["svc"].config.memory_limit == 256
        mgr.shutdown_all()
        mgr2.shutdown_all()

    def test_handle_death_restart_always(self, tmp_path):
        mgr = ProcessManager(str(tmp_path / "pm"))
        cfg = ProcessConfig(
            name="svc", command="echo hi", restart_policy=RestartPolicy.ALWAYS,
        )
        mgr.add_process(cfg)
        info = mgr._processes["svc"]
        info.last_exit_code = 0
        info.restart_count = 0
        stop_cb = MagicMock()
        restart_cb = MagicMock()
        mgr.on_process_stop = stop_cb
        mgr.on_process_restart = restart_cb
        with patch("deployment.process_manager.threading.Thread"):
            mgr._handle_process_death(info)
        stop_cb.assert_called_once()
        restart_cb.assert_called_once()
        assert info.state == ProcessState.RESTARTING
        mgr.shutdown_all()

    def test_handle_death_on_failure_zero(self, tmp_path):
        mgr = ProcessManager(str(tmp_path / "pm"))
        cfg = ProcessConfig(
            name="svc", command="echo hi", restart_policy=RestartPolicy.ON_FAILURE,
        )
        mgr.add_process(cfg)
        info = mgr._processes["svc"]
        info.last_exit_code = 0
        with patch("deployment.process_manager.threading.Thread"):
            mgr._handle_process_death(info)
        assert info.state != ProcessState.RESTARTING
        mgr.shutdown_all()

    def test_handle_death_on_failure_nonzero(self, tmp_path):
        mgr = ProcessManager(str(tmp_path / "pm"))
        cfg = ProcessConfig(
            name="svc", command="echo hi", restart_policy=RestartPolicy.ON_FAILURE,
        )
        mgr.add_process(cfg)
        info = mgr._processes["svc"]
        info.last_exit_code = 1
        with patch("deployment.process_manager.threading.Thread"):
            mgr._handle_process_death(info)
        assert info.state == ProcessState.RESTARTING
        mgr.shutdown_all()

    def test_handle_death_never(self, tmp_path):
        mgr = ProcessManager(str(tmp_path / "pm"))
        cfg = ProcessConfig(
            name="svc", command="echo hi", restart_policy=RestartPolicy.NEVER,
        )
        mgr.add_process(cfg)
        info = mgr._processes["svc"]
        info.last_exit_code = 1
        mgr._handle_process_death(info)
        assert info.state != ProcessState.RESTARTING
        mgr.shutdown_all()

    def test_handle_death_max_restarts(self, tmp_path):
        mgr = ProcessManager(str(tmp_path / "pm"))
        cfg = ProcessConfig(
            name="svc", command="echo hi",
            restart_policy=RestartPolicy.ALWAYS, max_restarts=2,
        )
        mgr.add_process(cfg)
        info = mgr._processes["svc"]
        info.restart_count = 3
        mgr._handle_process_death(info)
        assert info.state == ProcessState.FAILED
        mgr.shutdown_all()

    def test_handle_death_unless_stopped(self, tmp_path):
        mgr = ProcessManager(str(tmp_path / "pm"))
        cfg = ProcessConfig(
            name="svc", command="echo hi",
            restart_policy=RestartPolicy.UNLESS_STOPPED,
        )
        mgr.add_process(cfg)
        info = mgr._processes["svc"]
        info.state = ProcessState.FAILED
        with patch("deployment.process_manager.threading.Thread"):
            mgr._handle_process_death(info)
        assert info.state == ProcessState.RESTARTING
        mgr.shutdown_all()

    def test_start_stop_health_checking(self, tmp_path):
        mgr = ProcessManager(str(tmp_path / "pm"))
        mgr.start_health_checking()
        time.sleep(0.1)
        mgr.stop_health_checking()
        mgr.shutdown_all()


class TestGetProcessManager:
    def test_global_instance(self, tmp_path):
        import deployment.process_manager as mod
        original = getattr(mod, "_process_manager", None)
        mod._process_manager = None
        try:
            mgr = get_process_manager(str(tmp_path / "pm"))
            assert mgr is not None
            mgr.shutdown_all()
        finally:
            mod._process_manager = original
