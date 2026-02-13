"""Extra coverage tests for monitoring/start_monitoring.py - targeting 99%+.

Covers remaining uncovered paths:
- create_ssl_config_interactive with staging=False (domain != localhost, answer no to LE)
- main() daemon mode with shutdown_requested loop (non-KeyboardInterrupt exit)
- main() with no_ssl=False (loads ssl_config)
- MonitoringOrchestrator._initialize_services with no ssl_config
- print_status with uptime_seconds=None but running=True
- print_status with restart_count=0
- health check loop where restart fails
"""
import importlib.util
import os
import signal
import subprocess
import sys
import threading
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
_src = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.insert(0, _src)

# Mock ssl_config before importing
mock_ssl_config_cls = MagicMock()
mock_ssl_config_cls.https_port_api = 8443
mock_ssl_config_cls.https_port_frontend = 8501
mock_ssl_config_cls.wss_port_streaming = 8444
mock_ssl_config_cls.domain = "localhost"
mock_ssl_config_cls.cert_file = "/tmp/test.crt"
mock_ssl_config_cls.key_file = "/tmp/test.key"

mock_ssl_module = MagicMock()
mock_ssl_module.SSLConfig = MagicMock(return_value=mock_ssl_config_cls)
mock_ssl_module.load_ssl_config = MagicMock(return_value=mock_ssl_config_cls)
mock_ssl_module.save_ssl_config = MagicMock(return_value=True)
mock_ssl_module.initialize_ssl_infrastructure = MagicMock(
    return_value=(True, mock_ssl_config_cls)
)
mock_ssl_module.test_ssl_connection = MagicMock(return_value=True)

sys.modules["monitoring.ssl_config"] = mock_ssl_module

_spec = importlib.util.spec_from_file_location(
    "monitoring.start_monitoring",
    os.path.join(_src, "monitoring", "start_monitoring.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["monitoring.start_monitoring"] = _mod
_spec.loader.exec_module(_mod)

MonitoringService = _mod.MonitoringService
MonitoringOrchestrator = _mod.MonitoringOrchestrator
create_ssl_config_interactive = _mod.create_ssl_config_interactive
main = _mod.main
class TestCreateSSLConfigInteractiveEdgeCases:
    @patch("builtins.input", side_effect=["example.com", "admin@test.com", "no"])
    def test_domain_not_localhost_no_letsencrypt(self, mock_input):
        """Cover: domain != localhost, user says no to LE."""
        result = create_ssl_config_interactive()
        assert result is not None

    @patch("builtins.input", side_effect=["example.com", "", "yes", "no"])
    def test_letsencrypt_no_staging(self, mock_input):
        """Cover: LE=yes, staging=no."""
        result = create_ssl_config_interactive()
        assert result is not None

    @patch("builtins.input", side_effect=["", ""])
    def test_all_defaults(self, mock_input):
        """Cover: all defaults (localhost domain, default email)."""
        result = create_ssl_config_interactive()
        assert result is not None

class TestMonitoringOrchestratorInitializeServicesNoSSL:
    @patch("monitoring.start_monitoring.signal.signal")
    def test_services_initialized_without_ssl(self, mock_signal):
        """Cover: _initialize_services with ssl_config=None."""
        mock_ssl_module.load_ssl_config.return_value = None
        orch = MonitoringOrchestrator(ssl_config=None)
        assert len(orch.services) == 3
        for svc in orch.services:
            assert svc.ssl_required is False
        mock_ssl_module.load_ssl_config.return_value = mock_ssl_config_cls

class TestPrintStatusEdgeCases:
    @patch("monitoring.start_monitoring.signal.signal")
    def test_print_status_running_no_uptime(self, mock_signal):
        """Cover: running=True but uptime_seconds=None (start_time not set)."""
        orch = MonitoringOrchestrator(ssl_config=mock_ssl_config_cls)
        for svc in orch.services:
            svc.get_status = MagicMock(
                return_value={
                    "name": "test",
                    "running": True,
                    "port": 8000,
                    "pid": 123,
                    "uptime_seconds": None,
                    "restart_count": 0,
                    "ssl_required": False,
                }
            )
        orch.print_status()

    @patch("monitoring.start_monitoring.signal.signal")
    def test_print_status_not_running_with_ssl(self, mock_signal):
        """Cover: running=False with ssl_required=True."""
        orch = MonitoringOrchestrator(ssl_config=mock_ssl_config_cls)
        for svc in orch.services:
            svc.get_status = MagicMock(
                return_value={
                    "name": "test",
                    "running": False,
                    "port": 8443,
                    "pid": None,
                    "uptime_seconds": None,
                    "restart_count": 0,
                    "ssl_required": True,
                }
            )
        orch.print_status()

class TestHealthCheckLoopEdgeCases:
    @patch("monitoring.start_monitoring.signal.signal")
    @patch("monitoring.start_monitoring.time.sleep")
    def test_health_check_loop_restart_failure(self, mock_sleep, mock_signal):
        """Cover: service fails health check >= max_failures, restart fails."""
        orch = MonitoringOrchestrator(ssl_config=mock_ssl_config_cls)
        fail_count = [0]

        for svc in orch.services:
            svc.check_health = MagicMock(return_value=False)
            svc.restart = MagicMock(return_value=False)

        call_count = [0]

        def stop_after(*a):
            call_count[0] += 1
            if call_count[0] >= 5:
                orch.shutdown_requested = True

        mock_sleep.side_effect = stop_after
        orch._health_check_loop()

    @patch("monitoring.start_monitoring.signal.signal")
    @patch("monitoring.start_monitoring.time.sleep")
    def test_health_check_loop_mixed_health(self, mock_sleep, mock_signal):
        """Cover: some services healthy, others not."""
        orch = MonitoringOrchestrator(ssl_config=mock_ssl_config_cls)
        orch.services[0].check_health = MagicMock(return_value=True)
        orch.services[1].check_health = MagicMock(return_value=False)
        orch.services[1].restart = MagicMock(return_value=True)
        orch.services[2].check_health = MagicMock(return_value=True)

        call_count = [0]

        def stop_after(*a):
            call_count[0] += 1
            if call_count[0] >= 5:
                orch.shutdown_requested = True

        mock_sleep.side_effect = stop_after
        orch._health_check_loop()

class TestMainEdgeCases:
    @patch("monitoring.start_monitoring.signal.signal")
    @patch("monitoring.start_monitoring.argparse.ArgumentParser.parse_args")
    @patch("monitoring.start_monitoring.time.sleep")
    def test_daemon_mode_shutdown_requested(self, mock_sleep, mock_args, mock_signal):
        """Cover: daemon mode exits via shutdown_requested (not KeyboardInterrupt)."""
        mock_args.return_value = MagicMock(
            config_ssl=False, init_ssl=False, test_ssl=False,
            status=False, stop=False, no_ssl=True, daemon=True,
        )
        call_count = [0]

        def stop_after(*a):
            call_count[0] += 1

        mock_sleep.side_effect = stop_after

        with patch.object(
            MonitoringOrchestrator, "start_all_services", return_value=True
        ) as mock_start:
            # Make shutdown_requested True after a few loops
            orig_init = MonitoringOrchestrator.__init__

            def patched_init(self, *a, **kw):
                orig_init(self, *a, **kw)
                self.shutdown_requested = True

            with patch.object(MonitoringOrchestrator, "__init__", patched_init):
                main()

    @patch("monitoring.start_monitoring.signal.signal")
    @patch("monitoring.start_monitoring.argparse.ArgumentParser.parse_args")
    def test_main_with_ssl_config(self, mock_args, mock_signal):
        """Cover: no_ssl=False, loads ssl_config."""
        mock_args.return_value = MagicMock(
            config_ssl=False, init_ssl=False, test_ssl=False,
            status=True, stop=False, no_ssl=False, daemon=False,
        )
        mock_ssl_module.load_ssl_config.return_value = mock_ssl_config_cls
        with patch.object(MonitoringOrchestrator, "print_status"):
            main()

    @patch("monitoring.start_monitoring.signal.signal")
    @patch("monitoring.start_monitoring.argparse.ArgumentParser.parse_args")
    def test_test_ssl_all_pass(self, mock_args, mock_signal):
        """Cover: test_ssl with all ports passing."""
        mock_args.return_value = MagicMock(
            config_ssl=False, init_ssl=False, test_ssl=True,
            status=False, stop=False, no_ssl=False, daemon=False,
        )
        mock_ssl_module.load_ssl_config.return_value = mock_ssl_config_cls
        mock_ssl_module.test_ssl_connection.return_value = True
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    @patch("monitoring.start_monitoring.signal.signal")
    @patch("monitoring.start_monitoring.argparse.ArgumentParser.parse_args")
    def test_test_ssl_some_fail(self, mock_args, mock_signal):
        """Cover: test_ssl with some ports failing."""
        mock_args.return_value = MagicMock(
            config_ssl=False, init_ssl=False, test_ssl=True,
            status=False, stop=False, no_ssl=False, daemon=False,
        )
        mock_ssl_module.load_ssl_config.return_value = mock_ssl_config_cls
        mock_ssl_module.test_ssl_connection.side_effect = [True, True, False]
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        mock_ssl_module.test_ssl_connection.side_effect = None
        mock_ssl_module.test_ssl_connection.return_value = True

class TestMonitoringServiceCheckHealthUrlSuccess:
    def test_check_health_url_200(self):
        """Cover: check_health with check_url returning 200."""
        svc = MonitoringService(
            "t", ["echo"], 8000, check_url="http://localhost:8000/health"
        )
        svc.is_running = True
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        svc.process = mock_proc
        mock_requests = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_requests.get.return_value = mock_resp
        with patch.dict(sys.modules, {"requests": mock_requests}):
            result = svc.check_health()
        assert result is True

    def test_check_health_url_500(self):
        """Cover: check_health with check_url returning non-200."""
        svc = MonitoringService(
            "t", ["echo"], 8000, check_url="http://localhost:8000/health"
        )
        svc.is_running = True
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        svc.process = mock_proc
        mock_requests = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_requests.get.return_value = mock_resp
        with patch.dict(sys.modules, {"requests": mock_requests}):
            result = svc.check_health()
        assert result is False

class TestOrchestratorCheckSSLInfraEdgeCases:
    @patch("monitoring.start_monitoring.signal.signal")
    @patch("monitoring.start_monitoring.Path.exists", return_value=False)
    def test_check_ssl_init_success_updates_config(self, mock_exists, mock_signal):
        """Cover: SSL init succeeds and updates ssl_config."""
        new_config = MagicMock()
        mock_ssl_module.initialize_ssl_infrastructure.return_value = (True, new_config)
        orch = MonitoringOrchestrator(ssl_config=mock_ssl_config_cls)
        result = orch.check_ssl_infrastructure()
        assert result is True
        assert orch.ssl_config == new_config
        mock_ssl_module.initialize_ssl_infrastructure.return_value = (
            True, mock_ssl_config_cls,
        )

class TestRunInteractiveModeEdgeCases:
    @patch("monitoring.start_monitoring.signal.signal")
    @patch("monitoring.start_monitoring.time.sleep")
    def test_run_interactive_shutdown_in_inner_loop(self, mock_sleep, mock_signal):
        """Cover: shutdown_requested becomes True in the inner for loop."""
        orch = MonitoringOrchestrator(ssl_config=mock_ssl_config_cls)
        orch.print_status = MagicMock()
        orch.stop_all_services = MagicMock()
        call_count = [0]

        def stop_mid(*a):
            call_count[0] += 1
            if call_count[0] >= 1:
                orch.shutdown_requested = True

        mock_sleep.side_effect = stop_mid
        orch.run_interactive_mode()
        orch.stop_all_services.assert_called_once()