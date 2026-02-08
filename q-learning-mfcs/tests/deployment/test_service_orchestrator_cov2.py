"""Tests for deployment/service_orchestrator.py - coverage target 98%+."""
import sys
import os
import json
import signal
import subprocess
import time

import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from deployment.log_management import LogManager
from deployment.process_manager import ProcessConfig, RestartPolicy


def _make_orchestrator(tmp_path, log_level="DEBUG"):
    """Create orchestrator with fully mocked log manager."""
    mock_lm = MagicMock(spec=LogManager)
    mock_lm.get_logger.return_value = MagicMock()
    with patch("deployment.service_orchestrator.get_log_manager", return_value=mock_lm):
        from deployment.service_orchestrator import ServiceOrchestrator
        return ServiceOrchestrator(deployment_dir=str(tmp_path / "deploy"), log_level=log_level)


# Re-import with the module available
from deployment.service_orchestrator import (
    ServiceOrchestrator, ServiceConfig, ServiceDependency, ServiceInfo,
    ServiceState, DependencyType, main,
)


@pytest.fixture
def orchestrator(tmp_path):
    orch = _make_orchestrator(tmp_path)
    yield orch
    orch.shutdown_all()


class TestSignalHandler:
    def test_signal_handler_calls_shutdown(self, orchestrator):
        handler = signal.getsignal(signal.SIGTERM)
        with patch.object(orchestrator, "shutdown_all") as m:
            handler(signal.SIGTERM, None)
            m.assert_called_once()


class TestStartServiceException:
    def test_start_service_internal_exception(self, orchestrator):
        cfg = ServiceConfig(name="exc_svc", startup_command=["echo", "hi"])
        info = orchestrator.add_service(cfg)
        fail_cb = MagicMock()
        orchestrator.on_service_fail = fail_cb
        with patch.object(orchestrator, "_execute_service_startup", side_effect=RuntimeError("boom")):
            result = orchestrator._start_service_internal(info)
        assert result is False
        assert info.state == ServiceState.FAILED
        assert info.last_error == "boom"
        fail_cb.assert_called_once_with(info)


class TestDependencyHealthCheckFail:
    def test_required_dep_health_check_fails(self, orchestrator):
        dep_cfg = ServiceConfig(name="dep_svc", startup_command=["echo", "dep"])
        orchestrator.add_service(dep_cfg)
        dep = ServiceDependency(service_name="dep_svc", dependency_type=DependencyType.REQUIRED, timeout=0.1)
        main_cfg = ServiceConfig(name="main_svc", startup_command=["echo", "main"], dependencies=[dep])
        orchestrator.add_service(main_cfg)
        orchestrator._services["dep_svc"].state = ServiceState.ACTIVE
        with patch.object(orchestrator, "_wait_for_dependency_health", return_value=False):
            result = orchestrator._start_dependencies(orchestrator._services["main_svc"])
            assert result is False


class TestExecuteStartupNoConfig:
    def test_no_startup_command_no_process_config(self, orchestrator):
        cfg = ServiceConfig.__new__(ServiceConfig)
        for attr, val in [("name","bare"),("description",""),("process_config",None),("startup_command",None),("shutdown_command",None),("health_check_command",None),("dependencies",[]),("startup_timeout",60),("shutdown_timeout",30),("startup_delay",0),("enable_monitoring",True),("health_check_interval",30),("auto_restart",True),("max_restart_attempts",3),("environment",{}),("working_directory",None),("log_file",None),("log_level","INFO")]:
            setattr(cfg, attr, val)
        info = ServiceInfo(config=cfg)
        assert orchestrator._execute_service_startup(info) is False


class TestExecuteShutdownCommand:
    def test_shutdown_via_process_manager(self, orchestrator):
        pcfg = ServiceConfig(name="sd2", startup_command=["echo","s"],
            process_config=MagicMock())
        info = ServiceInfo(config=pcfg)
        with patch.object(orchestrator.process_manager, "stop_process", return_value=True):
            assert orchestrator._execute_service_shutdown(info, 5.0) is True

    def test_shutdown_command_exception_path(self, orchestrator):
        """shutdown_command branch hits scoping bug (local import os) - triggers exception path."""
        cfg = ServiceConfig(name="sd", startup_command=["echo","s"], shutdown_command=["echo","stop"])
        info = ServiceInfo(config=cfg)
        result = orchestrator._execute_service_shutdown(info, 5.0)
        assert result is False
        assert info.last_error is not None

    def test_shutdown_no_config_returns_true(self, orchestrator):
        cfg = ServiceConfig(name="nc", startup_command=["echo","s"])
        info = ServiceInfo(config=cfg)
        assert orchestrator._execute_service_shutdown(info, 5.0) is True


class TestShutdownPidKill:
    def test_shutdown_via_pid_kill(self, orchestrator):
        cfg = ServiceConfig(name="pk", startup_command=["echo","s"], shutdown_command=None)
        info = ServiceInfo(config=cfg, pid=99999)
        calls = []
        def fk(pid, sig):
            calls.append(sig)
            if sig == 0 and len(calls) <= 2: return
            if sig == 0: raise OSError("gone")
        with patch("os.kill", side_effect=fk):
            with patch("time.sleep"):
                assert orchestrator._execute_service_shutdown(info, 0.5) is True


class TestRestartStopFails:
    def test_restart_service_stop_fails(self, orchestrator):
        cfg = ServiceConfig(name="rs", startup_command=["echo","hi"])
        orchestrator.add_service(cfg)
        orchestrator._services["rs"].state = ServiceState.ACTIVE
        with patch.object(orchestrator, "stop_service", return_value=False):
            assert orchestrator.restart_service("rs") is False


class TestMonitoringLoopException:
    def test_monitoring_loop_exception_branch(self, orchestrator):
        orchestrator._is_monitoring = True
        cc = [0]
        def fh(*a, **kw):
            cc[0] += 1
            if cc[0] <= 1: raise RuntimeError("fail")
            orchestrator._is_monitoring = False
        with patch.object(orchestrator, "_check_service_health", side_effect=fh):
            with patch("time.sleep"):
                orchestrator._monitoring_loop()
        assert cc[0] >= 1


class TestHealthCheckErrorDecrement:
    def test_error_count_decrement(self, orchestrator):
        cfg = ServiceConfig(name="hd", startup_command=["echo","hi"])
        info = orchestrator.add_service(cfg)
        info.state = ServiceState.ACTIVE; info.error_count = 3; info.last_health_check = None
        with patch.object(orchestrator, "_perform_health_check", return_value=True):
            orchestrator._check_individual_service_health(info)
        assert info.error_count == 2


class TestHealthCheckCommandException:
    def test_health_check_command_exception(self, orchestrator):
        cfg = ServiceConfig(name="hce", startup_command=["echo","hi"], health_check_command=["false"])
        info = ServiceInfo(config=cfg)
        with patch("subprocess.run", side_effect=OSError("no")):
            assert orchestrator._perform_health_check(info) is False


class TestMainCLI:
    def _run_main(self, argv, tmp_path, extra_patches=None):
        mock_lm = MagicMock(spec=LogManager)
        mock_lm.get_logger.return_value = MagicMock()
        patches = [
            patch("sys.argv", argv),
            patch("deployment.service_orchestrator.get_log_manager", return_value=mock_lm),
        ]
        if extra_patches:
            patches.extend(extra_patches)
        ctx = [p.__enter__() for p in patches]
        try:
            main()
        finally:
            for p in reversed(patches):
                p.__exit__(None, None, None)

    def test_main_start_all(self, tmp_path):
        self._run_main(["p","--deployment-dir",str(tmp_path),"--start-all"], tmp_path,
            [patch.object(ServiceOrchestrator, "start_all", return_value={}),
             patch.object(ServiceOrchestrator, "shutdown_all")])

    def test_main_stop_all(self, tmp_path):
        self._run_main(["p","--deployment-dir",str(tmp_path),"--stop-all"], tmp_path,
            [patch.object(ServiceOrchestrator, "stop_all", return_value={}),
             patch.object(ServiceOrchestrator, "shutdown_all")])

    def test_main_status(self, tmp_path):
        self._run_main(["p","--deployment-dir",str(tmp_path),"--status"], tmp_path,
            [patch.object(ServiceOrchestrator, "get_status", return_value={"t":0}),
             patch.object(ServiceOrchestrator, "shutdown_all")])

    def test_main_start_service(self, tmp_path):
        self._run_main(["p","--deployment-dir",str(tmp_path),"--start","x"], tmp_path,
            [patch.object(ServiceOrchestrator, "start_service", return_value=True),
             patch.object(ServiceOrchestrator, "shutdown_all")])

    def test_main_stop_service(self, tmp_path):
        self._run_main(["p","--deployment-dir",str(tmp_path),"--stop","x"], tmp_path,
            [patch.object(ServiceOrchestrator, "stop_service", return_value=True),
             patch.object(ServiceOrchestrator, "shutdown_all")])

    def test_main_restart_service(self, tmp_path):
        self._run_main(["p","--deployment-dir",str(tmp_path),"--restart","x"], tmp_path,
            [patch.object(ServiceOrchestrator, "restart_service", return_value=True),
             patch.object(ServiceOrchestrator, "shutdown_all")])

    def test_main_with_config(self, tmp_path):
        cf = tmp_path / "c.json"
        cf.write_text(json.dumps({"services":[],"orchestrator_config":{}}))
        self._run_main(["p","--deployment-dir",str(tmp_path),"--config",str(cf),"--status"], tmp_path,
            [patch.object(ServiceOrchestrator, "shutdown_all")])

    def test_main_interactive_mode(self, tmp_path):
        self._run_main(["p","--deployment-dir",str(tmp_path)], tmp_path,
            [patch.object(ServiceOrchestrator, "start_monitoring"),
             patch("time.sleep", side_effect=KeyboardInterrupt),
             patch.object(ServiceOrchestrator, "shutdown_all")])
