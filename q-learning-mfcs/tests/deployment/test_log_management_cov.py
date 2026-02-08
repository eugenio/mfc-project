"""Tests for deployment/log_management.py - coverage target 98%+."""
import sys
import os
import json
import logging
import time
import gzip
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from deployment.log_management import (
    LogLevel,
    LogRotationConfig,
    LogMonitoringConfig,
    LogFormatter,
    RotatingCompressedFileHandler,
    LogManager,
    get_log_manager,
    setup_service_logging,
)


class TestLogLevel:
    def test_all_levels(self):
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"


class TestLogRotationConfig:
    def test_defaults(self):
        cfg = LogRotationConfig()
        assert cfg.max_size_mb == 50
        assert cfg.max_files == 5
        assert cfg.compress_after_days == 1
        assert cfg.delete_after_days == 30
        assert cfg.rotation_interval == "daily"

    def test_custom(self):
        cfg = LogRotationConfig(max_size_mb=100, max_files=10)
        assert cfg.max_size_mb == 100
        assert cfg.max_files == 10


class TestLogMonitoringConfig:
    def test_defaults(self):
        cfg = LogMonitoringConfig()
        assert cfg.enable_monitoring is True
        assert cfg.error_threshold == 10
        assert cfg.warning_threshold == 50
        assert cfg.monitor_interval == 60
        assert cfg.alert_callback is None

    def test_custom(self):
        cb = lambda msg: None
        cfg = LogMonitoringConfig(alert_callback=cb)
        assert cfg.alert_callback is cb


class TestLogFormatter:
    def test_basic_format(self):
        fmt = LogFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None,
        )
        result = fmt.format(record)
        assert "hello" in result
        assert "test" in result

    def test_engine_type_in_name(self):
        fmt = LogFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="msg", args=(), exc_info=None,
        )
        record.engine_type = "Q-learning"
        result = fmt.format(record)
        assert "Q-learning" in result

    def test_service_id_in_name(self):
        fmt = LogFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="msg", args=(), exc_info=None,
        )
        record.service_id = "svc-1"
        result = fmt.format(record)
        assert "svc-1" in result

    def test_console_handler_colors(self):
        fmt = LogFormatter()
        for level_name in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            record = logging.LogRecord(
                name="test", level=getattr(logging, level_name),
                pathname="", lineno=0, msg="msg", args=(), exc_info=None,
            )
            record.console_handler = True
            result = fmt.format(record)
            assert "msg" in result

    def test_console_handler_unknown_level(self):
        fmt = LogFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="", lineno=0, msg="msg", args=(), exc_info=None,
        )
        record.levelname = "CUSTOM"
        record.console_handler = True
        result = fmt.format(record)
        assert "msg" in result


class TestRotatingCompressedFileHandler:
    def test_init(self, tmp_path):
        log_file = tmp_path / "test.log"
        handler = RotatingCompressedFileHandler(
            filename=str(log_file),
            maxBytes=1024,
            backupCount=3,
            compress_after_days=1,
        )
        assert handler.compress_after_days == 1
        handler.close()

    def test_compress_old_logs(self, tmp_path):
        log_file = tmp_path / "test.log"
        log_file.write_text("data")

        handler = RotatingCompressedFileHandler(
            filename=str(log_file),
            maxBytes=1024,
            backupCount=3,
            compress_after_days=0,
        )

        old_log = tmp_path / "test.1"
        old_log.write_text("old data")
        old_time = (datetime.now() - timedelta(days=2)).timestamp()
        os.utime(str(old_log), (old_time, old_time))

        handler._compress_old_logs()
        assert (tmp_path / "test.1.gz").exists()
        assert not old_log.exists()
        handler.close()

    def test_compress_skips_gz(self, tmp_path):
        log_file = tmp_path / "test.log"
        log_file.write_text("data")

        handler = RotatingCompressedFileHandler(
            filename=str(log_file),
            maxBytes=1024,
            backupCount=3,
            compress_after_days=0,
        )
        gz_file = tmp_path / "test.1.gz"
        gz_file.write_text("compressed")

        handler._compress_old_logs()
        assert gz_file.exists()
        handler.close()

    def test_compress_file(self, tmp_path):
        log_file = tmp_path / "test.log"
        log_file.write_text("data")

        handler = RotatingCompressedFileHandler(
            filename=str(log_file), maxBytes=1024, backupCount=3,
        )

        target = tmp_path / "tocompress.log"
        target.write_text("compress me")
        handler._compress_file(target)

        assert (tmp_path / "tocompress.log.gz").exists()
        assert not target.exists()
        handler.close()

    def test_compress_file_error(self, tmp_path, capsys):
        log_file = tmp_path / "test.log"
        log_file.write_text("data")

        handler = RotatingCompressedFileHandler(
            filename=str(log_file), maxBytes=1024, backupCount=3,
        )
        handler._compress_file(Path("/nonexistent/file.log"))
        captured = capsys.readouterr()
        assert "Failed to compress" in captured.err
        handler.close()

    def test_compress_old_logs_error(self, tmp_path, capsys):
        log_file = tmp_path / "test.log"
        log_file.write_text("data")

        handler = RotatingCompressedFileHandler(
            filename=str(log_file), maxBytes=1024, backupCount=3,
        )
        with patch("pathlib.Path.parent", new_callable=lambda: property(
            lambda self: (_ for _ in ()).throw(OSError("fail"))
        )):
            handler._compress_old_logs()
        captured = capsys.readouterr()
        assert "Log compression error" in captured.err
        handler.close()


class TestLogManager:
    def test_init(self, tmp_path):
        mgr = LogManager(str(tmp_path / "logs"))
        assert mgr.log_dir.exists()
        assert mgr.monitoring_enabled is False
        mgr.shutdown()

    def test_create_logger(self, tmp_path):
        mgr = LogManager(str(tmp_path / "logs"))
        logger = mgr.create_logger("test_svc")
        assert isinstance(logger, logging.Logger)
        assert "test_svc" in mgr.loggers
        mgr.shutdown()

    def test_create_logger_cached(self, tmp_path):
        mgr = LogManager(str(tmp_path / "logs"))
        l1 = mgr.create_logger("test_svc")
        l2 = mgr.create_logger("test_svc")
        assert l1 is l2
        mgr.shutdown()

    def test_create_logger_file_only(self, tmp_path):
        mgr = LogManager(str(tmp_path / "logs"))
        logger = mgr.create_logger("f", log_to_console=False)
        assert "f_file" in mgr.handlers
        assert "f_console" not in mgr.handlers
        mgr.shutdown()

    def test_create_logger_console_only(self, tmp_path):
        mgr = LogManager(str(tmp_path / "logs"))
        logger = mgr.create_logger("c", log_to_file=False)
        assert "c_file" not in mgr.handlers
        assert "c_console" in mgr.handlers
        mgr.shutdown()

    def test_create_logger_custom_rotation(self, tmp_path):
        mgr = LogManager(str(tmp_path / "logs"))
        cfg = LogRotationConfig(max_size_mb=10, max_files=2)
        logger = mgr.create_logger("custom", rotation_config=cfg)
        assert logger is not None
        mgr.shutdown()

    def test_get_logger(self, tmp_path):
        mgr = LogManager(str(tmp_path / "logs"))
        assert mgr.get_logger("missing") is None
        mgr.create_logger("present")
        assert mgr.get_logger("present") is not None
        mgr.shutdown()

    def test_set_log_level(self, tmp_path):
        mgr = LogManager(str(tmp_path / "logs"))
        mgr.create_logger("svc", level=LogLevel.INFO)
        mgr.set_log_level("svc", LogLevel.DEBUG)
        assert mgr.loggers["svc"].level == logging.DEBUG
        mgr.set_log_level("nonexistent", LogLevel.ERROR)
        mgr.shutdown()

    def test_stats_filter(self, tmp_path):
        mgr = LogManager(str(tmp_path / "logs"))
        logger = mgr.create_logger("svc", log_to_file=False)
        with patch.object(logging.StreamHandler, "emit"):
            logger.info("test message")
        assert mgr.log_stats["total_messages"] >= 1
        mgr.shutdown()

    def test_stats_filter_error_tracking(self, tmp_path):
        mgr = LogManager(str(tmp_path / "logs"))
        logger = mgr.create_logger("svc", log_to_file=False)
        with patch.object(logging.StreamHandler, "emit"):
            logger.error("error msg")
            logger.warning("warn msg")
        assert mgr.log_stats["by_level"]["ERROR"] >= 1
        assert mgr.log_stats["by_level"]["WARNING"] >= 1
        mgr.shutdown()

    def test_add_to_minute_counter(self, tmp_path):
        mgr = LogManager(str(tmp_path / "logs"))
        counter = []
        current = int(time.time() // 60)
        mgr._add_to_minute_counter(counter, current)
        mgr._add_to_minute_counter(counter, current)
        mgr.shutdown()

    def test_start_stop_monitoring(self, tmp_path):
        mgr = LogManager(str(tmp_path / "logs"))
        mgr.monitoring_config.monitor_interval = 1
        mgr.start_monitoring()
        assert mgr.monitoring_enabled is True
        mgr.start_monitoring()  # idempotent
        time.sleep(0.1)
        mgr.stop_monitoring()
        assert mgr.monitoring_enabled is False
        mgr.shutdown()

    def test_check_error_rates_triggers_alert(self, tmp_path):
        mgr = LogManager(str(tmp_path / "logs"))
        current_minute = int(time.time() // 60)
        mgr.log_stats["errors_per_minute"] = [
            {"minute": current_minute - 1, "count": 20}
        ]
        mgr.monitoring_config.error_threshold = 5
        with patch.object(mgr, "_trigger_alert") as mock_alert:
            mgr._check_error_rates()
            mock_alert.assert_called()
        mgr.shutdown()

    def test_check_warning_rates_triggers_alert(self, tmp_path):
        mgr = LogManager(str(tmp_path / "logs"))
        current_minute = int(time.time() // 60)
        mgr.log_stats["warnings_per_minute"] = [
            {"minute": current_minute - 1, "count": 100}
        ]
        mgr.monitoring_config.warning_threshold = 50
        with patch.object(mgr, "_trigger_alert") as mock_alert:
            mgr._check_error_rates()
            mock_alert.assert_called()
        mgr.shutdown()

    def test_cleanup_old_logs(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        mgr = LogManager(str(log_dir))
        mgr.rotation_config.delete_after_days = 0

        old_log = log_dir / "old.log"
        old_log.write_text("old data")
        old_time = (datetime.now() - timedelta(days=2)).timestamp()
        os.utime(str(old_log), (old_time, old_time))

        mgr._cleanup_old_logs()
        assert not old_log.exists()
        mgr.shutdown()

    def test_cleanup_old_logs_error(self, tmp_path, capsys):
        mgr = LogManager(str(tmp_path / "logs"))
        with patch.object(Path, "rglob", side_effect=OSError("fail")):
            mgr._cleanup_old_logs()
        captured = capsys.readouterr()
        assert "Log cleanup error" in captured.err
        mgr.shutdown()

    def test_trigger_alert_with_callback(self, tmp_path):
        cb = MagicMock()
        mgr = LogManager(str(tmp_path / "logs"))
        mgr.monitoring_config.alert_callback = cb
        mgr._trigger_alert("test alert")
        cb.assert_called_once_with("test alert")
        mgr.shutdown()

    def test_trigger_alert_callback_error(self, tmp_path, capsys):
        cb = MagicMock(side_effect=Exception("cb fail"))
        mgr = LogManager(str(tmp_path / "logs"))
        mgr.monitoring_config.alert_callback = cb
        mgr._trigger_alert("test alert")
        captured = capsys.readouterr()
        assert "Alert callback error" in captured.err
        mgr.shutdown()

    def test_get_statistics(self, tmp_path):
        mgr = LogManager(str(tmp_path / "logs"))
        mgr.create_logger("svc", log_to_file=False)
        stats = mgr.get_statistics()
        assert "active_loggers" in stats
        assert "svc" in stats["active_loggers"]
        assert "log_dir" in stats
        mgr.shutdown()

    def test_export_logs(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        mgr = LogManager(str(log_dir))

        log_file = log_dir / "svc.log"
        log_file.write_text(
            "2025-01-01 12:00:00,000 - svc - INFO - [1234:5678] - test msg\n"
        )
        logs = mgr.export_logs()
        assert len(logs) == 1
        assert logs[0]["message"] == "test msg"
        mgr.shutdown()

    def test_export_logs_with_filters(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        mgr = LogManager(str(log_dir))

        log_file = log_dir / "svc.log"
        log_file.write_text(
            "2025-01-01 12:00:00,000 - svc - INFO - [1234:5678] - msg1\n"
            "2025-01-01 13:00:00,000 - svc - ERROR - [1234:5678] - msg2\n"
        )
        logs = mgr.export_logs(level_filter=LogLevel.ERROR)
        assert len(logs) == 1
        assert logs[0]["level"] == "ERROR"
        mgr.shutdown()

    def test_export_logs_time_filter(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        mgr = LogManager(str(log_dir))

        log_file = log_dir / "svc.log"
        log_file.write_text(
            "2025-01-01 12:00:00,000 - svc - INFO - [1:1] - early\n"
            "2025-06-01 12:00:00,000 - svc - INFO - [1:1] - late\n"
        )
        logs = mgr.export_logs(
            start_time=datetime(2025, 3, 1),
            end_time=datetime(2025, 7, 1),
        )
        assert len(logs) == 1
        assert logs[0]["message"] == "late"
        mgr.shutdown()

    def test_export_logs_logger_filter(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        mgr = LogManager(str(log_dir))

        log_file = log_dir / "svc.log"
        log_file.write_text(
            "2025-01-01 12:00:00,000 - svc.a - INFO - [1:1] - m1\n"
            "2025-01-01 12:00:00,000 - svc.b - INFO - [1:1] - m2\n"
        )
        logs = mgr.export_logs(logger_filter="svc.a")
        assert len(logs) == 1
        mgr.shutdown()

    def test_export_logs_bad_format(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        mgr = LogManager(str(log_dir))

        log_file = log_dir / "svc.log"
        log_file.write_text("not a valid log line\n")
        logs = mgr.export_logs()
        assert len(logs) == 0
        mgr.shutdown()

    def test_export_logs_bad_timestamp(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        mgr = LogManager(str(log_dir))

        log_file = log_dir / "svc.log"
        log_file.write_text(
            "XXXX-01-01 12:00:00,000 - svc - INFO - [1:1] - msg\n"
        )
        logs = mgr.export_logs()
        assert len(logs) == 0
        mgr.shutdown()

    def test_create_log_archive(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        mgr = LogManager(str(log_dir))

        (log_dir / "test.log").write_text("data")
        archive = mgr.create_log_archive("test_archive")
        assert archive.exists()
        assert archive.name == "test_archive.tar.gz"
        archive.unlink()
        mgr.shutdown()

    def test_create_log_archive_default_name(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        mgr = LogManager(str(log_dir))

        archive = mgr.create_log_archive()
        assert archive.exists()
        assert "service_logs_" in archive.name
        archive.unlink()
        mgr.shutdown()

    def test_shutdown(self, tmp_path):
        mgr = LogManager(str(tmp_path / "logs"))
        mgr.create_logger("svc", log_to_file=False)
        mgr.shutdown()
        assert len(mgr.loggers) == 0
        assert len(mgr.handlers) == 0


class TestGetLogManager:
    def test_global_instance(self, tmp_path):
        import deployment.log_management as mod
        original = getattr(mod, "_log_manager", None)
        mod._log_manager = None
        try:
            mgr = get_log_manager()
            assert mgr is not None
            mgr.shutdown()
        finally:
            mod._log_manager = original


class TestSetupServiceLogging:
    def test_development(self, tmp_path):
        import deployment.log_management as mod
        original = getattr(mod, "_log_manager", None)
        mod._log_manager = LogManager(str(tmp_path / "logs"))
        try:
            logger = setup_service_logging("dev_svc", "development")
            assert logger is not None
        finally:
            mod._log_manager.shutdown()
            mod._log_manager = original

    def test_production(self, tmp_path):
        import deployment.log_management as mod
        original = getattr(mod, "_log_manager", None)
        mod._log_manager = LogManager(str(tmp_path / "logs"))
        try:
            logger = setup_service_logging("prod_svc", "production")
            assert logger is not None
        finally:
            mod._log_manager.shutdown()
            mod._log_manager = original

    def test_other_env(self, tmp_path):
        import deployment.log_management as mod
        original = getattr(mod, "_log_manager", None)
        mod._log_manager = LogManager(str(tmp_path / "logs"))
        try:
            logger = setup_service_logging("stg_svc", "staging")
            assert logger is not None
        finally:
            mod._log_manager.shutdown()
            mod._log_manager = original
