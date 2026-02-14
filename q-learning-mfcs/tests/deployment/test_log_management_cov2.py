"""Tests for deployment/log_management.py - coverage target 98%+."""
import sys
import os
import json
import logging
import time
import signal

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from deployment.log_management import (
    LogManager, LogLevel, LogRotationConfig, RotatingCompressedFileHandler, main,
)


@pytest.fixture
def log_manager(tmp_path):
    mgr = LogManager(log_dir=str(tmp_path / "logs"))
    yield mgr
    mgr.shutdown()


@pytest.mark.coverage_extra
class TestCompressOldLogs:
    def test_compress_old_logs(self, tmp_path):
        lf = tmp_path / "test.log"
        lf.write_text("initial")
        h = RotatingCompressedFileHandler(str(lf), maxBytes=100, backupCount=3, compress_after_days=0)
        old = tmp_path / "test.1"
        old.write_text("old content")
        h._compress_old_logs()
        assert (tmp_path / "test.1.gz").exists() or not old.exists()
        h.close()


@pytest.mark.coverage_extra
class TestSignalHandler:
    def test_signal_handler(self, log_manager):
        handler = signal.getsignal(signal.SIGTERM)
        with patch.object(log_manager, "shutdown"):
            handler(signal.SIGTERM, None)


@pytest.mark.coverage_extra
class TestMinuteCounter:
    def test_add_to_minute_counter(self, log_manager):
        cm = int(time.time() // 60)
        counter = [{"minute": cm, "count": 1}]
        log_manager._add_to_minute_counter(counter, cm)

    def test_add_to_minute_counter_new(self, log_manager):
        counter = []
        log_manager._add_to_minute_counter(counter, int(time.time() // 60))


@pytest.mark.coverage_extra
class TestMonitoringLoopException:
    def test_monitoring_loop_exception(self, log_manager):
        log_manager.monitoring_enabled = True
        cc = [0]
        def fail():
            cc[0] += 1
            if cc[0] == 1: raise RuntimeError("fail")
            log_manager.monitoring_enabled = False
        with patch.object(log_manager, "_check_error_rates", side_effect=fail):
            with patch("time.sleep"):
                log_manager._monitoring_loop()
        assert cc[0] >= 1


@pytest.mark.coverage_extra
class TestExportLogsParseError:
    def test_export_logs_parse_error(self, log_manager):
        ld = Path(log_manager.log_dir)
        (ld / "bad.log").write_text("not valid\n")
        logs = log_manager.export_logs()
        assert isinstance(logs, list)


@pytest.mark.coverage_extra
class TestParseLogFileTimestampError:
    def test_bad_timestamp_skipped(self, log_manager):
        ld = Path(log_manager.log_dir)
        lf = ld / "ts.log"
        lf.write_text("XXXX-XX-XX XX:XX:XX,XXX - mylog - INFO - [1:2] - msg\n")
        assert len(log_manager._parse_log_file(lf, None, None, None, None)) == 0


@pytest.mark.coverage_extra
class TestParseLogFileFilters:
    def test_end_time_filter(self, log_manager):
        ld = Path(log_manager.log_dir)
        lf = ld / "f.log"
        lf.write_text("2025-01-01 12:00:00,000 - tl - INFO - [1:2] - msg\n")
        assert len(log_manager._parse_log_file(lf, None, datetime(2024,1,1), None, None)) == 0
        assert len(log_manager._parse_log_file(lf, None, None, LogLevel.ERROR, None)) == 0
        assert len(log_manager._parse_log_file(lf, None, None, None, "nonexistent")) == 0


@pytest.mark.coverage_extra
class TestShutdownHandlerException:
    def test_shutdown_handler_exception(self, log_manager):
        mh = MagicMock(); mh.flush.side_effect = RuntimeError("fail")
        log_manager.handlers["bad"] = mh
        log_manager.shutdown()


@pytest.mark.coverage_extra
class TestMainCLI:
    def _mock_main(self, argv, extra_patches=None):
        mock_mgr = MagicMock(spec=LogManager)
        mock_mgr.get_statistics.return_value = {"loggers": {}, "handlers": {}}
        mock_mgr.export_logs.return_value = []
        mock_mgr.create_log_archive.return_value = Path("/tmp/a.tar.gz")
        patches = [
            patch("sys.argv", argv),
            patch("deployment.log_management.get_log_manager", return_value=mock_mgr),
        ]
        if extra_patches:
            patches.extend(extra_patches)
        ctx = [p.__enter__() for p in patches]
        try:
            main()
        finally:
            for p in reversed(patches):
                p.__exit__(None, None, None)
        return mock_mgr

    def test_main_stats(self):
        self._mock_main(["p", "--stats"])

    def test_main_export(self, tmp_path):
        ef = str(tmp_path / "e.json")
        self._mock_main(["p", "--export", ef, "--start-time", "2025-01-01 00:00:00",
            "--end-time", "2025-12-31 23:59:59", "--level", "ERROR", "--logger", "t"])

    def test_main_archive(self):
        self._mock_main(["p", "--archive"])

    def test_main_cleanup(self):
        self._mock_main(["p", "--cleanup"])

    def test_main_monitor(self):
        self._mock_main(["p", "--monitor"],
            [patch("time.sleep", side_effect=KeyboardInterrupt)])

    def test_main_keyboard_interrupt(self):
        mock_mgr = MagicMock(spec=LogManager)
        mock_mgr.get_statistics.side_effect = KeyboardInterrupt
        with patch("sys.argv", ["p", "--stats"]):
            with patch("deployment.log_management.get_log_manager", return_value=mock_mgr):
                main()
