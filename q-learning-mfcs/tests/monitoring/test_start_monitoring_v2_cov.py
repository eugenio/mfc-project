"""Tests for monitoring/start_monitoring.py - coverage target 98%+."""
import importlib.util
import sys
import os
import signal
import subprocess
import threading
from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock

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

# Pre-register mock so import inside start_monitoring succeeds
sys.modules["monitoring.ssl_config"] = mock_ssl_module

# Load start_monitoring via importlib to avoid package conflicts
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


class TestMonitoringService:
    def test_init(self):
        svc = MonitoringService("test", ["echo", "hi"], 8000)
        assert svc.name == "test"
        assert svc.port == 8000
        assert svc.is_running is False
        assert svc.restart_count == 0
        assert svc.process is None
        assert svc.ssl_required is False
        assert svc.start_time is None

    def test_init_with_ssl(self):
        svc = MonitoringService(
            "test", ["echo"], 443, check_url="https://localhost", ssl_required=True
        )
        assert svc.ssl_required is True
        assert svc.check_url == "https://localhost"

    @patch("monitoring.start_monitoring.subprocess.Popen")
    def test_start_success(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.pid = 1234
        mock_popen.return_value = mock_proc
        svc = MonitoringService("test", ["echo", "hi"], 8000)
        result = svc.start()
        assert result is True
        assert svc.is_running is True
        assert svc.start_time is not None

    def test_start_already_running(self):
        svc = MonitoringService("test", ["echo"], 8000)
        svc.is_running = True
        result = svc.start()
        assert result is True

    @patch("monitoring.start_monitoring.subprocess.Popen", side_effect=OSError("fail"))
    def test_start_failure(self, mock_popen):
        svc = MonitoringService("test", ["bad_cmd"], 8000)
        result = svc.start()
        assert result is False

    def test_check_health_not_running(self):
        svc = MonitoringService("test", ["echo"], 8000)
        assert svc.check_health() is False

    def test_check_health_process_stopped(self):
        svc = MonitoringService("test", ["echo"], 8000)
        svc.is_running = True
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1
        svc.process = mock_proc
        assert svc.check_health() is False
        assert svc.is_running is False

    def test_check_health_running_no_url(self):
        svc = MonitoringService("test", ["echo"], 8000)
        svc.is_running = True
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        svc.process = mock_proc
        assert svc.check_health() is True

    @patch("requests.get")
    def test_check_health_with_url_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp
        svc = MonitoringService("test", ["echo"], 8000, check_url="http://localhost")
        svc.is_running = True
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        svc.process = mock_proc
        assert svc.check_health() is True

    @patch("requests.get", side_effect=Exception("conn refused"))
    def test_check_health_with_url_failure(self, mock_get):
        svc = MonitoringService("test", ["echo"], 8000, check_url="http://localhost")
        svc.is_running = True
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        svc.process = mock_proc
        assert svc.check_health() is False

    def test_stop_not_running(self):
        svc = MonitoringService("test", ["echo"], 8000)
        assert svc.stop() is True

    def test_stop_graceful(self):
        svc = MonitoringService("test", ["echo"], 8000)
        svc.is_running = True
        mock_proc = MagicMock()
        svc.process = mock_proc
        result = svc.stop()
        assert result is True
        assert svc.is_running is False
        assert svc.process is None
        mock_proc.terminate.assert_called_once()

    def test_stop_force_kill(self):
        svc = MonitoringService("test", ["echo"], 8000)
        svc.is_running = True
        mock_proc = MagicMock()
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 10), None]
        svc.process = mock_proc
        result = svc.stop()
        assert result is True
        mock_proc.kill.assert_called_once()

    def test_stop_exception(self):
        svc = MonitoringService("test", ["echo"], 8000)
        svc.is_running = True
        mock_proc = MagicMock()
        mock_proc.terminate.side_effect = OSError("fail")
        svc.process = mock_proc
        result = svc.stop()
        assert result is False

    @patch.object(MonitoringService, "stop", return_value=True)
    @patch.object(MonitoringService, "start", return_value=True)
    @patch("monitoring.start_monitoring.time.sleep")
    def test_restart(self, mock_sleep, mock_start, mock_stop):
        svc = MonitoringService("test", ["echo"], 8000)
        result = svc.restart()
        assert result is True
        assert svc.restart_count == 1

    @patch.object(MonitoringService, "stop", return_value=False)
    def test_restart_stop_fails(self, mock_stop):
        svc = MonitoringService("test", ["echo"], 8000)
        result = svc.restart()
        assert result is False

    def test_get_status_not_running(self):
        svc = MonitoringService("test", ["echo"], 8000, ssl_required=True)
        status = svc.get_status()
        assert status["name"] == "test"
        assert status["running"] is False
        assert status["pid"] is None
        assert status["uptime_seconds"] is None
        assert status["ssl_required"] is True

    def test_get_status_running(self):
        svc = MonitoringService("test", ["echo"], 8000)
        svc.is_running = True
        svc.start_time = datetime.now()
        svc.restart_count = 2
        mock_proc = MagicMock()
        mock_proc.pid = 999
        svc.process = mock_proc
        status = svc.get_status()
        assert status["running"] is True
        assert status["pid"] == 999
        assert status["uptime_seconds"] is not None
        assert status["restart_count"] == 2


class TestMonitoringOrchestrator:
    @patch("monitoring.start_monitoring.signal.signal")
    def test_init_with_ssl(self, mock_signal):
        orch = MonitoringOrchestrator(ssl_config=mock_ssl_config_cls)
        assert len(orch.services) == 3
        assert orch.ssl_config is not None

    @patch("monitoring.start_monitoring.signal.signal")
    def test_init_no_ssl(self, mock_signal):
        mock_ssl_module.load_ssl_config.return_value = None
        orch = MonitoringOrchestrator(ssl_config=None)
        mock_ssl_module.load_ssl_config.return_value = mock_ssl_config_cls

    @patch("monitoring.start_monitoring.signal.signal")
    def test_signal_handler(self, mock_signal):
        orch = MonitoringOrchestrator(ssl_config=mock_ssl_config_cls)
        orch._signal_handler(signal.SIGINT, None)
        assert orch.shutdown_requested is True

    @patch("monitoring.start_monitoring.signal.signal")
    def test_check_ssl_no_config(self, mock_signal):
        orch = MonitoringOrchestrator(ssl_config=mock_ssl_config_cls)
        orch.ssl_config = None
        result = orch.check_ssl_infrastructure()
        assert result is True

    @patch("monitoring.start_monitoring.Path.exists", return_value=True)
    @patch("monitoring.start_monitoring.signal.signal")
    def test_check_ssl_certs_exist(self, mock_signal, mock_exists):
        orch = MonitoringOrchestrator(ssl_config=mock_ssl_config_cls)
        result = orch.check_ssl_infrastructure()
        assert result is True

    @patch("monitoring.start_monitoring.Path.exists", return_value=False)
    @patch("monitoring.start_monitoring.signal.signal")
    def test_check_ssl_init_success(self, mock_signal, mock_exists):
        mock_ssl_module.initialize_ssl_infrastructure.return_value = (
            True,
            mock_ssl_config_cls,
        )
        orch = MonitoringOrchestrator(ssl_config=mock_ssl_config_cls)
        result = orch.check_ssl_infrastructure()
        assert result is True

    @patch("monitoring.start_monitoring.Path.exists", return_value=False)
    @patch("monitoring.start_monitoring.signal.signal")
    def test_check_ssl_init_failure(self, mock_signal, mock_exists):
        mock_ssl_module.initialize_ssl_infrastructure.return_value = (False, None)
        orch = MonitoringOrchestrator(ssl_config=mock_ssl_config_cls)
        result = orch.check_ssl_infrastructure()
        assert result is True
        assert orch.ssl_config is None
        mock_ssl_module.initialize_ssl_infrastructure.return_value = (
            True,
            mock_ssl_config_cls,
        )

    @patch("monitoring.start_monitoring.signal.signal")
    @patch("monitoring.start_monitoring.time.sleep")
    def test_start_all_services(self, mock_sleep, mock_signal):
        orch = MonitoringOrchestrator(ssl_config=mock_ssl_config_cls)
        for svc in orch.services:
            svc.start = MagicMock(return_value=True)
            svc.check_health = MagicMock(return_value=True)
        orch.check_ssl_infrastructure = MagicMock(return_value=True)
        orch.start_health_monitoring = MagicMock()
        result = orch.start_all_services()
        assert result is True

    @patch("monitoring.start_monitoring.signal.signal")
    @patch("monitoring.start_monitoring.time.sleep")
    def test_start_all_partial_failure(self, mock_sleep, mock_signal):
        orch = MonitoringOrchestrator(ssl_config=mock_ssl_config_cls)
        orch.check_ssl_infrastructure = MagicMock(return_value=True)
        orch.services[0].start = MagicMock(return_value=True)
        orch.services[1].start = MagicMock(return_value=False)
        orch.services[2].start = MagicMock(return_value=True)
        result = orch.start_all_services()
        assert result is False

    @patch("monitoring.start_monitoring.signal.signal")
    def test_start_all_ssl_check_fails(self, mock_signal):
        orch = MonitoringOrchestrator(ssl_config=mock_ssl_config_cls)
        orch.check_ssl_infrastructure = MagicMock(return_value=False)
        result = orch.start_all_services()
        assert result is False

    @patch("monitoring.start_monitoring.signal.signal")
    def test_stop_all_services(self, mock_signal):
        orch = MonitoringOrchestrator(ssl_config=mock_ssl_config_cls)
        for svc in orch.services:
            svc.stop = MagicMock(return_value=True)
        orch.stop_all_services()
        for svc in orch.services:
            svc.stop.assert_called_once()

    @patch("monitoring.start_monitoring.signal.signal")
    def test_stop_all_with_health_thread(self, mock_signal):
        orch = MonitoringOrchestrator(ssl_config=mock_ssl_config_cls)
        mock_thread = MagicMock()
        orch.health_check_thread = mock_thread
        for svc in orch.services:
            svc.stop = MagicMock(return_value=True)
        orch.stop_all_services()
        assert orch.shutdown_requested is True
        mock_thread.join.assert_called_once()

    @patch("monitoring.start_monitoring.signal.signal")
    def test_start_health_monitoring(self, mock_signal):
        orch = MonitoringOrchestrator(ssl_config=mock_ssl_config_cls)
        with patch("monitoring.start_monitoring.threading.Thread") as mock_thr:
            mock_t = MagicMock()
            mock_thr.return_value = mock_t
            orch.start_health_monitoring()
            mock_t.start.assert_called_once()

    @patch("monitoring.start_monitoring.signal.signal")
    @patch("monitoring.start_monitoring.time.sleep")
    def test_health_check_loop_healthy(self, mock_sleep, mock_signal):
        orch = MonitoringOrchestrator(ssl_config=mock_ssl_config_cls)
        for svc in orch.services:
            svc.check_health = MagicMock(return_value=True)
        call_count = [0]

        def stop_after(*args):
            call_count[0] += 1
            if call_count[0] >= 2:
                orch.shutdown_requested = True

        mock_sleep.side_effect = stop_after
        orch._health_check_loop()

    @patch("monitoring.start_monitoring.signal.signal")
    @patch("monitoring.start_monitoring.time.sleep")
    def test_health_check_loop_failures_restart(self, mock_sleep, mock_signal):
        orch = MonitoringOrchestrator(ssl_config=mock_ssl_config_cls)
        for svc in orch.services:
            svc.check_health = MagicMock(return_value=False)
            svc.restart = MagicMock(return_value=True)
        call_count = [0]

        def stop_after(*args):
            call_count[0] += 1
            if call_count[0] >= 4:
                orch.shutdown_requested = True

        mock_sleep.side_effect = stop_after
        orch._health_check_loop()

    @patch("monitoring.start_monitoring.signal.signal")
    @patch("monitoring.start_monitoring.time.sleep")
    def test_health_check_loop_restart_fails(self, mock_sleep, mock_signal):
        orch = MonitoringOrchestrator(ssl_config=mock_ssl_config_cls)
        for svc in orch.services:
            svc.check_health = MagicMock(return_value=False)
            svc.restart = MagicMock(return_value=False)
        call_count = [0]

        def stop_after(*args):
            call_count[0] += 1
            if call_count[0] >= 4:
                orch.shutdown_requested = True

        mock_sleep.side_effect = stop_after
        orch._health_check_loop()

    @patch("monitoring.start_monitoring.signal.signal")
    def test_print_configuration_with_ssl(self, mock_signal):
        orch = MonitoringOrchestrator(ssl_config=mock_ssl_config_cls)
        orch.print_configuration()

    @patch("monitoring.start_monitoring.signal.signal")
    def test_print_configuration_no_ssl(self, mock_signal):
        orch = MonitoringOrchestrator(ssl_config=mock_ssl_config_cls)
        orch.ssl_config = None
        orch.print_configuration()

    @patch("monitoring.start_monitoring.signal.signal")
    def test_print_status(self, mock_signal):
        orch = MonitoringOrchestrator(ssl_config=mock_ssl_config_cls)
        for svc in orch.services:
            svc.get_status = MagicMock(
                return_value={
                    "name": "t",
                    "running": True,
                    "port": 8000,
                    "pid": 123,
                    "uptime_seconds": 3700.0,
                    "restart_count": 1,
                    "ssl_required": True,
                }
            )
        orch.print_status()

    @patch("monitoring.start_monitoring.signal.signal")
    def test_print_status_not_running(self, mock_signal):
        orch = MonitoringOrchestrator(ssl_config=mock_ssl_config_cls)
        for svc in orch.services:
            svc.get_status = MagicMock(
                return_value={
                    "name": "t",
                    "running": False,
                    "port": 8000,
                    "pid": None,
                    "uptime_seconds": None,
                    "restart_count": 0,
                    "ssl_required": False,
                }
            )
        orch.print_status()

    @patch("monitoring.start_monitoring.signal.signal")
    @patch("monitoring.start_monitoring.time.sleep")
    def test_run_interactive_mode(self, mock_sleep, mock_signal):
        orch = MonitoringOrchestrator(ssl_config=mock_ssl_config_cls)
        orch.print_status = MagicMock()
        orch.stop_all_services = MagicMock()
        call_count = [0]

        def stop_quickly(*a):
            call_count[0] += 1
            if call_count[0] >= 2:
                orch.shutdown_requested = True

        mock_sleep.side_effect = stop_quickly
        orch.run_interactive_mode()
        orch.stop_all_services.assert_called_once()

    @patch("monitoring.start_monitoring.signal.signal")
    @patch("monitoring.start_monitoring.time.sleep", side_effect=KeyboardInterrupt)
    def test_run_interactive_keyboard_interrupt(self, mock_sleep, mock_signal):
        orch = MonitoringOrchestrator(ssl_config=mock_ssl_config_cls)
        orch.print_status = MagicMock()
        orch.stop_all_services = MagicMock()
        orch.run_interactive_mode()
        orch.stop_all_services.assert_called_once()


class TestCreateSSLConfigInteractive:
    @patch("builtins.input", side_effect=["", ""])
    def test_defaults(self, mock_input):
        create_ssl_config_interactive()
        mock_ssl_module.SSLConfig.assert_called()

    @patch("builtins.input", side_effect=["example.com", "a@b.com", "y", "y"])
    def test_letsencrypt_staging(self, mock_input):
        create_ssl_config_interactive()

    @patch("builtins.input", side_effect=["example.com", "a@b.com", "n"])
    def test_no_letsencrypt(self, mock_input):
        create_ssl_config_interactive()


class TestMain:
    @patch("monitoring.start_monitoring.signal.signal")
    @patch("monitoring.start_monitoring.argparse.ArgumentParser.parse_args")
    def test_config_ssl(self, mock_args, mock_signal):
        mock_args.return_value = MagicMock(
            config_ssl=True, init_ssl=False, test_ssl=False,
            status=False, stop=False, no_ssl=False, daemon=False,
        )
        with patch(
            "monitoring.start_monitoring.create_ssl_config_interactive"
        ) as mc:
            mc.return_value = MagicMock()
            mock_ssl_module.save_ssl_config.return_value = True
            main()

    @patch("monitoring.start_monitoring.signal.signal")
    @patch("monitoring.start_monitoring.argparse.ArgumentParser.parse_args")
    def test_config_ssl_save_fails(self, mock_args, mock_signal):
        mock_args.return_value = MagicMock(
            config_ssl=True, init_ssl=False, test_ssl=False,
            status=False, stop=False, no_ssl=False, daemon=False,
        )
        with patch(
            "monitoring.start_monitoring.create_ssl_config_interactive"
        ) as mc:
            mc.return_value = MagicMock()
            mock_ssl_module.save_ssl_config.return_value = False
            main()
            mock_ssl_module.save_ssl_config.return_value = True

    @patch("monitoring.start_monitoring.signal.signal")
    @patch("monitoring.start_monitoring.argparse.ArgumentParser.parse_args")
    def test_init_ssl_success(self, mock_args, mock_signal):
        mock_args.return_value = MagicMock(
            config_ssl=False, init_ssl=True, test_ssl=False,
            status=False, stop=False, no_ssl=False, daemon=False,
        )
        mock_ssl_module.initialize_ssl_infrastructure.return_value = (
            True, mock_ssl_config_cls,
        )
        main()

    @patch("monitoring.start_monitoring.signal.signal")
    @patch("monitoring.start_monitoring.argparse.ArgumentParser.parse_args")
    def test_init_ssl_failure(self, mock_args, mock_signal):
        mock_args.return_value = MagicMock(
            config_ssl=False, init_ssl=True, test_ssl=False,
            status=False, stop=False, no_ssl=False, daemon=False,
        )
        mock_ssl_module.initialize_ssl_infrastructure.return_value = (False, None)
        with pytest.raises(SystemExit):
            main()
        mock_ssl_module.initialize_ssl_infrastructure.return_value = (
            True, mock_ssl_config_cls,
        )

    @patch("monitoring.start_monitoring.signal.signal")
    @patch("monitoring.start_monitoring.argparse.ArgumentParser.parse_args")
    def test_test_ssl_success(self, mock_args, mock_signal):
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
    def test_test_ssl_no_config(self, mock_args, mock_signal):
        mock_args.return_value = MagicMock(
            config_ssl=False, init_ssl=False, test_ssl=True,
            status=False, stop=False, no_ssl=False, daemon=False,
        )
        mock_ssl_module.load_ssl_config.return_value = None
        with pytest.raises(SystemExit):
            main()
        mock_ssl_module.load_ssl_config.return_value = mock_ssl_config_cls

    @patch("monitoring.start_monitoring.signal.signal")
    @patch("monitoring.start_monitoring.argparse.ArgumentParser.parse_args")
    def test_test_ssl_partial_failure(self, mock_args, mock_signal):
        mock_args.return_value = MagicMock(
            config_ssl=False, init_ssl=False, test_ssl=True,
            status=False, stop=False, no_ssl=False, daemon=False,
        )
        mock_ssl_module.load_ssl_config.return_value = mock_ssl_config_cls
        mock_ssl_module.test_ssl_connection.side_effect = [True, False, True]
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        mock_ssl_module.test_ssl_connection.side_effect = None
        mock_ssl_module.test_ssl_connection.return_value = True

    @patch("monitoring.start_monitoring.signal.signal")
    @patch("monitoring.start_monitoring.argparse.ArgumentParser.parse_args")
    def test_status_mode(self, mock_args, mock_signal):
        mock_args.return_value = MagicMock(
            config_ssl=False, init_ssl=False, test_ssl=False,
            status=True, stop=False, no_ssl=True, daemon=False,
        )
        with patch.object(MonitoringOrchestrator, "print_status"):
            main()

    @patch("monitoring.start_monitoring.signal.signal")
    @patch("monitoring.start_monitoring.argparse.ArgumentParser.parse_args")
    def test_stop_mode(self, mock_args, mock_signal):
        mock_args.return_value = MagicMock(
            config_ssl=False, init_ssl=False, test_ssl=False,
            status=False, stop=True, no_ssl=True, daemon=False,
        )
        with patch.object(MonitoringOrchestrator, "stop_all_services"):
            main()

    @patch("monitoring.start_monitoring.signal.signal")
    @patch("monitoring.start_monitoring.argparse.ArgumentParser.parse_args")
    @patch("monitoring.start_monitoring.time.sleep")
    def test_daemon_mode(self, mock_sleep, mock_args, mock_signal):
        mock_args.return_value = MagicMock(
            config_ssl=False, init_ssl=False, test_ssl=False,
            status=False, stop=False, no_ssl=True, daemon=True,
        )
        call_count = [0]

        def stop_after(*a):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise KeyboardInterrupt

        mock_sleep.side_effect = stop_after
        with patch.object(
            MonitoringOrchestrator, "start_all_services", return_value=True
        ):
            main()

    @patch("monitoring.start_monitoring.signal.signal")
    @patch("monitoring.start_monitoring.argparse.ArgumentParser.parse_args")
    def test_interactive_mode(self, mock_args, mock_signal):
        mock_args.return_value = MagicMock(
            config_ssl=False, init_ssl=False, test_ssl=False,
            status=False, stop=False, no_ssl=True, daemon=False,
        )
        with patch.object(
            MonitoringOrchestrator, "start_all_services", return_value=True
        ):
            with patch.object(MonitoringOrchestrator, "run_interactive_mode"):
                main()

    @patch("monitoring.start_monitoring.signal.signal")
    @patch("monitoring.start_monitoring.argparse.ArgumentParser.parse_args")
    def test_start_fails(self, mock_args, mock_signal):
        mock_args.return_value = MagicMock(
            config_ssl=False, init_ssl=False, test_ssl=False,
            status=False, stop=False, no_ssl=True, daemon=False,
        )
        with patch.object(
            MonitoringOrchestrator, "start_all_services", return_value=False
        ):
            with pytest.raises(SystemExit):
                main()
