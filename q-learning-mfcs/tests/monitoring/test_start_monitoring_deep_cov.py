import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import signal
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Mock ssl_config before importing
mock_ssl_module = MagicMock()
mock_ssl_module.SSLConfig = MagicMock
mock_ssl_module.load_ssl_config = MagicMock(return_value=None)
mock_ssl_module.save_ssl_config = MagicMock(return_value=True)
mock_ssl_module.initialize_ssl_infrastructure = MagicMock(return_value=(True, MagicMock()))
mock_ssl_module.test_ssl_connection = MagicMock(return_value=True)
sys.modules["monitoring.ssl_config"] = mock_ssl_module

from monitoring.start_monitoring import MonitoringService, MonitoringOrchestrator


class TestMonitoringService:
    def test_init(self):
        svc = MonitoringService("test", ["echo", "hi"], 8000)
        assert svc.name == "test"
        assert svc.port == 8000
        assert svc.is_running is False

    def test_init_ssl(self):
        svc = MonitoringService("t", ["echo"], 8000, check_url="https://x", ssl_required=True)
        assert svc.ssl_required is True

    def test_start_success(self):
        svc = MonitoringService("t", [sys.executable, "-c", "import time; time.sleep(60)"], 8000)
        assert svc.start() is True
        assert svc.is_running is True
        svc.stop()

    def test_start_already_running(self):
        svc = MonitoringService("t", [sys.executable, "-c", "import time; time.sleep(60)"], 8000)
        svc.start()
        assert svc.start() is True
        svc.stop()

    def test_start_failure(self):
        svc = MonitoringService("t", ["/nonexistent_binary_xyz"], 8000)
        assert svc.start() is False

    def test_check_health_not_running(self):
        svc = MonitoringService("t", ["echo"], 8000)
        assert svc.check_health() is False

    def test_check_health_stopped(self):
        svc = MonitoringService("t", [sys.executable, "-c", "pass"], 8000)
        svc.start()
        time.sleep(0.5)
        assert svc.check_health() is False

    def test_check_health_running(self):
        svc = MonitoringService("t", [sys.executable, "-c", "import time; time.sleep(60)"], 8000)
        svc.start()
        time.sleep(0.2)
        assert svc.check_health() is True
        svc.stop()

    def test_check_health_url_fail(self):
        svc = MonitoringService("t", [sys.executable, "-c", "import time; time.sleep(60)"], 8000, check_url="http://localhost:99999")
        svc.start()
        time.sleep(0.2)
        assert svc.check_health() is False
        svc.stop()

    def test_stop_not_running(self):
        svc = MonitoringService("t", ["echo"], 8000)
        assert svc.stop() is True

    def test_stop_running(self):
        svc = MonitoringService("t", [sys.executable, "-c", "import time; time.sleep(60)"], 8000)
        svc.start()
        assert svc.stop() is True
        assert svc.is_running is False

    def test_stop_force_kill(self):
        svc = MonitoringService("t", ["echo"], 8000)
        mock_proc = MagicMock()
        mock_proc.wait = MagicMock(side_effect=[subprocess.TimeoutExpired("c", 10), None])
        svc.process = mock_proc
        svc.is_running = True
        assert svc.stop() is True
        mock_proc.kill.assert_called_once()

    def test_stop_exception(self):
        svc = MonitoringService("t", ["echo"], 8000)
        svc.is_running = True
        svc.process = MagicMock()
        svc.process.terminate = MagicMock(side_effect=Exception("fail"))
        assert svc.stop() is False

    def test_restart(self):
        svc = MonitoringService("t", ["echo"], 8000)
        with patch.object(svc, "stop", return_value=True):
            with patch.object(svc, "start", return_value=True):
                with patch("monitoring.start_monitoring.time.sleep"):
                    assert svc.restart() is True
                    assert svc.restart_count == 1

    def test_restart_stop_fails(self):
        svc = MonitoringService("t", ["echo"], 8000)
        with patch.object(svc, "stop", return_value=False):
            with patch("monitoring.start_monitoring.time.sleep"):
                assert svc.restart() is False

    def test_get_status_not_running(self):
        svc = MonitoringService("t", ["echo"], 8000)
        s = svc.get_status()
        assert s["running"] is False
        assert s["pid"] is None

    def test_get_status_running(self):
        svc = MonitoringService("t", [sys.executable, "-c", "import time; time.sleep(60)"], 8000, ssl_required=True)
        svc.start()
        time.sleep(0.1)
        s = svc.get_status()
        assert s["running"] is True
        assert s["pid"] is not None
        assert s["uptime_seconds"] is not None
        svc.stop()


class TestMonitoringOrchestrator:
    def test_init_no_ssl(self):
        orch = MonitoringOrchestrator(ssl_config=None)
        assert len(orch.services) == 3

    def test_init_with_ssl(self):
        ssl = MagicMock()
        ssl.https_port_api = 8443
        ssl.https_port_frontend = 8501
        ssl.wss_port_streaming = 8444
        ssl.domain = "localhost"
        ssl.cert_file = "/tmp/cert.pem"
        ssl.key_file = "/tmp/key.pem"
        orch = MonitoringOrchestrator(ssl_config=ssl)
        assert len(orch.services) == 3

    def test_signal_handler(self):
        orch = MonitoringOrchestrator(ssl_config=None)
        orch._signal_handler(signal.SIGINT, None)
        assert orch.shutdown_requested is True

    def test_check_ssl_no_config(self):
        mock_ssl_module.load_ssl_config.return_value = None
        orch = MonitoringOrchestrator(ssl_config=None)
        assert orch.check_ssl_infrastructure() is True

    def test_check_ssl_certs_exist(self, tmp_path):
        cert = tmp_path / "cert.pem"
        key = tmp_path / "key.pem"
        cert.write_text("c")
        key.write_text("k")
        ssl = MagicMock()
        ssl.cert_file = str(cert)
        ssl.key_file = str(key)
        ssl.https_port_api = 8443
        ssl.https_port_frontend = 8501
        ssl.wss_port_streaming = 8444
        ssl.domain = "localhost"
        orch = MonitoringOrchestrator(ssl_config=ssl)
        assert orch.check_ssl_infrastructure() is True

    def test_check_ssl_init_fail(self, tmp_path):
        ssl = MagicMock()
        ssl.cert_file = str(tmp_path / "no.pem")
        ssl.key_file = str(tmp_path / "no.key")
        ssl.https_port_api = 8443
        ssl.https_port_frontend = 8501
        ssl.wss_port_streaming = 8444
        ssl.domain = "localhost"
        mock_ssl_module.initialize_ssl_infrastructure.return_value = (False, None)
        orch = MonitoringOrchestrator(ssl_config=ssl)
        assert orch.check_ssl_infrastructure() is True

    def test_start_all(self):
        orch = MonitoringOrchestrator(ssl_config=None)
        for svc in orch.services:
            svc.start = MagicMock(return_value=True)
        with patch.object(orch, "check_ssl_infrastructure", return_value=True):
            with patch.object(orch, "start_health_monitoring"):
                with patch("time.sleep"):
                    assert orch.start_all_services() is True

    def test_start_all_partial(self):
        mock_ssl_module.load_ssl_config.return_value = None
        orch = MonitoringOrchestrator(ssl_config=None)
        orch.services[0].start = MagicMock(return_value=True)
        orch.services[1].start = MagicMock(return_value=False)
        orch.services[2].start = MagicMock(return_value=True)
        with patch.object(orch, "check_ssl_infrastructure", return_value=True):
            with patch("monitoring.start_monitoring.time.sleep"):
                assert orch.start_all_services() is False

    def test_stop_all(self):
        orch = MonitoringOrchestrator(ssl_config=None)
        for s in orch.services:
            s.stop = MagicMock(return_value=True)
        orch.stop_all_services()

    def test_stop_with_health_thread(self):
        orch = MonitoringOrchestrator(ssl_config=None)
        orch.health_check_thread = MagicMock()
        for s in orch.services:
            s.stop = MagicMock(return_value=True)
        orch.stop_all_services()

    def test_print_config_no_ssl(self):
        orch = MonitoringOrchestrator(ssl_config=None)
        orch.print_configuration()

    def test_print_config_ssl(self):
        ssl = MagicMock()
        ssl.https_port_api = 8443
        ssl.https_port_frontend = 8501
        ssl.wss_port_streaming = 8444
        ssl.domain = "localhost"
        ssl.cert_file = "/tmp/cert.pem"
        ssl.key_file = "/tmp/key.pem"
        orch = MonitoringOrchestrator(ssl_config=ssl)
        orch.print_configuration()

    def test_print_status(self):
        orch = MonitoringOrchestrator(ssl_config=None)
        orch.print_status()

    def test_print_status_running(self):
        orch = MonitoringOrchestrator(ssl_config=None)
        for s in orch.services:
            s.is_running = True
            s.start_time = datetime.now()
            s.process = MagicMock(pid=123)
            s.restart_count = 1
        orch.print_status()

    def test_health_check_loop(self):
        orch = MonitoringOrchestrator(ssl_config=None)
        for s in orch.services:
            s.check_health = MagicMock(return_value=True)
        orch.health_check_interval = 0
        def stop():
            time.sleep(0.05)
            orch.shutdown_requested = True
        t = threading.Thread(target=stop)
        t.start()
        orch._health_check_loop()
        t.join()

    def test_health_check_loop_restart(self):
        orch = MonitoringOrchestrator(ssl_config=None)
        cnt = [0]
        def fail():
            cnt[0] += 1
            if cnt[0] >= 4:
                orch.shutdown_requested = True
            return False
        for s in orch.services:
            s.check_health = fail
            s.restart = MagicMock(return_value=True)
        orch.health_check_interval = 0
        orch._health_check_loop()
