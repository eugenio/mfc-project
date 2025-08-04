"""
Log Management System
Agent Zeta - Deployment and Process Management

Handles log rotation, compression, retention, and monitoring
"""
import argparse
import gzip
import json
import logging
import logging.handlers
import re
import shutil
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path


class LogLevel(Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class LogRotationConfig:
    """Log rotation configuration"""
    max_size_mb: int = 50          # Maximum log file size in MB
    max_files: int = 5             # Maximum number of rotated files to keep
    compress_after_days: int = 1   # Compress logs older than this many days
    delete_after_days: int = 30    # Delete logs older than this many days
    rotation_interval: str = "daily"  # daily, weekly, monthly

@dataclass
class LogMonitoringConfig:
    """Log monitoring configuration"""
    enable_monitoring: bool = True
    error_threshold: int = 10      # Alert after this many errors per minute
    warning_threshold: int = 50    # Alert after this many warnings per minute
    monitor_interval: int = 60     # Monitoring interval in seconds
    alert_callback: Callable | None = None
class LogFormatter(logging.Formatter):
    """Custom log formatter for services"""

    def __init__(self):
        super().__init__()
        self.format_string = "%(asctime)s - %(name)s - %(levelname)s - [%(process)d:%(thread)d] - %(message)s"

    def format(self, record):
        # Add service-specific context
        if hasattr(record, 'engine_type'):
            record.name = f"{record.name}[{record.engine_type}]"

        if hasattr(record, 'service_id'):
            record.name = f"{record.name}({record.service_id})"

        # Color formatting for console output
        if hasattr(record, 'console_handler'):
            colors = {
                'DEBUG': '\033[36m',    # Cyan
                'INFO': '\033[32m',     # Green
                'WARNING': '\033[33m',  # Yellow
                'ERROR': '\033[31m',    # Red
                'CRITICAL': '\033[35m', # Magenta
            }
            reset_color = '\033[0m'

            color = colors.get(record.levelname, '')
            record.levelname = f"{color}{record.levelname}{reset_color}"

        formatter = logging.Formatter(self.format_string)
        return formatter.format(record)
class RotatingCompressedFileHandler(logging.handlers.RotatingFileHandler):
    """Rotating file handler with compression support"""

    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0,
                 encoding=None, delay=False, compress_after_days=1):
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)
        self.compress_after_days = compress_after_days

    def doRollover(self):
        """Override to add compression"""
        super().doRollover()

        # Compress old log files
        self._compress_old_logs()

    def _compress_old_logs(self):
        """Compress log files older than specified days"""
        try:
            log_dir = Path(self.baseFilename).parent
            log_name = Path(self.baseFilename).stem

            cutoff_time = datetime.now() - timedelta(days=self.compress_after_days)

            for log_file in log_dir.glob(f"{log_name}.*"):
                if log_file.suffix == '.gz':
                    continue  # Already compressed

                # Check file age
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_time < cutoff_time:
                    self._compress_file(log_file)

        except Exception as e:
            # Don't let compression errors break logging
            print(f"Log compression error: {e}", file=sys.stderr)

    def _compress_file(self, file_path: Path):
        """Compress a single log file"""
        compressed_path = file_path.with_suffix(file_path.suffix + '.gz')

        try:
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Remove original file
            file_path.unlink()

        except Exception as e:
            print(f"Failed to compress {file_path}: {e}", file=sys.stderr)
class LogManager:
    """Centralized log management for services"""

    def __init__(self, log_dir: str = "/tmp/service-logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.loggers: dict[str, logging.Logger] = {}
        self.handlers: dict[str, logging.Handler] = {}
        self.monitoring_enabled = False
        self.monitor_thread: threading.Thread | None = None
        self.shutdown_event = threading.Event()

        # Default configurations
        self.rotation_config = LogRotationConfig()
        self.monitoring_config = LogMonitoringConfig()

        # Log statistics
        self.log_stats = {
            'total_messages': 0,
            'by_level': {level.value: 0 for level in LogLevel},
            'by_logger': {},
            'errors_per_minute': [],
            'warnings_per_minute': []
        }

        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.shutdown()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    def create_logger(self, name: str,
                     level: LogLevel = LogLevel.INFO,
                     log_to_file: bool = True,
                     log_to_console: bool = True,
                     rotation_config: LogRotationConfig | None = None) -> logging.Logger:
        """Create a configured logger"""

        if name in self.loggers:
            return self.loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.value))

        # Clear any existing handlers
        logger.handlers = []

        formatter = LogFormatter()

        # File handler with rotation
        if log_to_file:
            config = rotation_config or self.rotation_config
            log_file = self.log_dir / f"{name}.log"

            file_handler = RotatingCompressedFileHandler(
                filename=str(log_file),
                maxBytes=config.max_size_mb * 1024 * 1024,
                backupCount=config.max_files,
                compress_after_days=config.compress_after_days
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            self.handlers[f"{name}_file"] = file_handler

        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            # Mark for color formatting
            def emit_with_color(record):
                record.console_handler = True
                return logging.StreamHandler.emit(console_handler, record)
            console_handler.emit = emit_with_color

            logger.addHandler(console_handler)
            self.handlers[f"{name}_console"] = console_handler

        # Add custom filter for statistics
        logger.addFilter(self._create_stats_filter())

        self.loggers[name] = logger
        self.log_stats['by_logger'][name] = {
            'total': 0,
            'by_level': {level.value: 0 for level in LogLevel}
        }

        return logger

    def _create_stats_filter(self) -> logging.Filter:
        """Create filter to collect log statistics"""
        def stats_filter(record):
            self.log_stats['total_messages'] += 1
            self.log_stats['by_level'][record.levelname] += 1

            logger_name = record.name
            if logger_name in self.log_stats['by_logger']:
                self.log_stats['by_logger'][logger_name]['total'] += 1
                self.log_stats['by_logger'][logger_name]['by_level'][record.levelname] += 1

            # Track errors and warnings for monitoring
            current_minute = int(time.time() // 60)

            if record.levelname == 'ERROR':
                self._add_to_minute_counter(self.log_stats['errors_per_minute'], current_minute)
            elif record.levelname == 'WARNING':
                self._add_to_minute_counter(self.log_stats['warnings_per_minute'], current_minute)

            return True

        filter_obj = logging.Filter()
        filter_obj.filter = stats_filter
        return filter_obj

    def _add_to_minute_counter(self, counter: list, minute: int):
        """Add count to minute-based counter"""
        # Keep only last 60 minutes
        counter = [entry for entry in counter if entry['minute'] >= minute - 60]

        # Find or create entry for current minute
        for entry in counter:
            if entry['minute'] == minute:
                entry['count'] += 1
                return

        # Create new entry
        counter.append({'minute': minute, 'count': 1})

    def get_logger(self, name: str) -> logging.Logger | None:
        """Get existing logger by name"""
        return self.loggers.get(name)

    def set_log_level(self, logger_name: str, level: LogLevel):
        """Change log level for a logger"""
        if logger_name in self.loggers:
            self.loggers[logger_name].setLevel(getattr(logging, level.value))

    def start_monitoring(self):
        """Start log monitoring"""
        if self.monitoring_enabled:
            return

        self.monitoring_enabled = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop log monitoring"""
        self.monitoring_enabled = False
        self.shutdown_event.set()

        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def _monitoring_loop(self):
        """Log monitoring loop"""
        while self.monitoring_enabled and not self.shutdown_event.is_set():
            try:
                self._check_error_rates()
                self._cleanup_old_logs()

                # Wait for next check
                for _ in range(self.monitoring_config.monitor_interval):
                    if self.shutdown_event.is_set():
                        break
                    time.sleep(1)

            except Exception as e:
                print(f"Log monitoring error: {e}", file=sys.stderr)
                time.sleep(10)

    def _check_error_rates(self):
        """Check error and warning rates"""
        current_minute = int(time.time() // 60)

        # Count errors in last minute
        error_count = sum(
            entry['count'] for entry in self.log_stats['errors_per_minute']
            if entry['minute'] == current_minute - 1
        )

        warning_count = sum(
            entry['count'] for entry in self.log_stats['warnings_per_minute']
            if entry['minute'] == current_minute - 1
        )

        # Check thresholds
        if error_count >= self.monitoring_config.error_threshold:
            self._trigger_alert(f"High error rate: {error_count} errors/minute")

        if warning_count >= self.monitoring_config.warning_threshold:
            self._trigger_alert(f"High warning rate: {warning_count} warnings/minute")

    def _cleanup_old_logs(self):
        """Clean up old log files"""
        try:
            cutoff_time = datetime.now() - timedelta(days=self.rotation_config.delete_after_days)

            for log_file in self.log_dir.rglob("*.log*"):
                if log_file.is_file():
                    file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_time < cutoff_time:
                        log_file.unlink()

        except Exception as e:
            print(f"Log cleanup error: {e}", file=sys.stderr)

    def _trigger_alert(self, message: str):
        """Trigger monitoring alert"""
        alert_logger = self.get_logger("log_monitor") or self.create_logger("log_monitor")
        alert_logger.warning(f"LOG ALERT: {message}")

        if self.monitoring_config.alert_callback:
            try:
                self.monitoring_config.alert_callback(message)
            except Exception as e:
                print(f"Alert callback error: {e}", file=sys.stderr)

    def get_statistics(self) -> dict:
        """Get log statistics"""
        return {
            **self.log_stats,
            'active_loggers': list(self.loggers.keys()),
            'log_dir': str(self.log_dir),
            'monitoring_enabled': self.monitoring_enabled
        }

    def export_logs(self, start_time: datetime | None = None,
                   end_time: datetime | None = None,
                   level_filter: LogLevel | None = None,
                   logger_filter: str | None = None) -> list[dict]:
        """Export log entries matching criteria"""
        logs = []

        for log_file in self.log_dir.rglob("*.log"):
            try:
                logs.extend(self._parse_log_file(log_file, start_time, end_time, level_filter, logger_filter))
            except Exception as e:
                print(f"Error parsing {log_file}: {e}", file=sys.stderr)

        return sorted(logs, key=lambda x: x['timestamp'])

    def _parse_log_file(self, log_file: Path, start_time: datetime | None,
                       end_time: datetime | None, level_filter: LogLevel | None,
                       logger_filter: str | None) -> list[dict]:
        """Parse log file and extract matching entries"""
        logs = []

        # Log format: timestamp - logger - level - [process:thread] - message
        log_pattern = re.compile(
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (.+?) - (\w+) - \[(\d+):(\d+)\] - (.+)'
        )

        with open(log_file, encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = log_pattern.match(line.strip())
                if not match:
                    continue

                timestamp_str, logger_name, level, process_id, thread_id, message = match.groups()

                # Parse timestamp
                try:
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                except ValueError:
                    continue

                # Apply filters
                if start_time and timestamp < start_time:
                    continue
                if end_time and timestamp > end_time:
                    continue
                if level_filter and level != level_filter.value:
                    continue
                if logger_filter and logger_filter not in logger_name:
                    continue

                logs.append({
                    'timestamp': timestamp,
                    'logger': logger_name,
                    'level': level,
                    'process_id': int(process_id),
                    'thread_id': int(thread_id),
                    'message': message,
                    'source_file': str(log_file)
                })

        return logs

    def create_log_archive(self, archive_name: str | None = None) -> Path:
        """Create compressed archive of all logs"""
        if not archive_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_name = f"service_logs_{timestamp}"

        archive_path = Path(f"/tmp/{archive_name}.tar.gz")

        import tarfile
        with tarfile.open(archive_path, 'w:gz') as tar:
            tar.add(self.log_dir, arcname=archive_name)

        return archive_path

    def shutdown(self):
        """Graceful shutdown of log manager"""
        self.stop_monitoring()

        # Flush all handlers
        for handler in self.handlers.values():
            try:
                handler.flush()
                handler.close()
            except Exception:
                pass

        # Clear loggers
        self.loggers.clear()
        self.handlers.clear()

# Global log manager instance
def get_log_manager() -> LogManager:
    """Get global log manager instance"""
    global _log_manager
    if _log_manager is None:
        _log_manager = LogManager()
    return _log_manager
def setup_service_logging(service_name: str = "service",
                     environment: str = "production") -> logging.Logger:
    """Setup logging for service"""
    log_manager = get_log_manager()

    # Configure based on environment
    if environment == "development":
        level = LogLevel.DEBUG
        log_to_console = True
    elif environment == "production":
        level = LogLevel.INFO
        log_to_console = False
    else:
        level = LogLevel.INFO
        log_to_console = True

    # Create main service logger
    logger = log_manager.create_logger(
        name=service_name,
        level=level,
        log_to_console=log_to_console
    )

    # Create component loggers
    component_loggers = [
        "handler",
        "engine",
        "service-manager",
        "health-monitor",
        "deployment"
    ]

    for component in component_loggers:
        log_manager.create_logger(
            name=f"{service_name}.{component}",
            level=level,
            log_to_console=log_to_console
        )

    # Start monitoring in production
    if environment == "production":
        log_manager.start_monitoring()

    return logger
def main():
    """Main entry point for log management utilities"""
    parser = argparse.ArgumentParser(description="Service Log Management")
    parser.add_argument("--stats", action="store_true", help="Show log statistics")
    parser.add_argument("--export", help="Export logs to JSON file")
    parser.add_argument("--archive", action="store_true", help="Create log archive")
    parser.add_argument("--cleanup", action="store_true", help="Clean up old logs")
    parser.add_argument("--monitor", action="store_true", help="Start log monitoring")
    parser.add_argument("--start-time", help="Start time for export (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--end-time", help="End time for export (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Filter by log level")
    parser.add_argument("--logger", help="Filter by logger name")

    args = parser.parse_args()

    log_manager = get_log_manager()

    try:
        if args.stats:
            stats = log_manager.get_statistics()
            print(json.dumps(stats, indent=2, default=str))

        if args.export:
            start_time = None
            end_time = None

            if args.start_time:
                start_time = datetime.strptime(args.start_time, '%Y-%m-%d %H:%M:%S')
            if args.end_time:
                end_time = datetime.strptime(args.end_time, '%Y-%m-%d %H:%M:%S')

            level_filter = LogLevel(args.level) if args.level else None

            logs = log_manager.export_logs(start_time, end_time, level_filter, args.logger)

            with open(args.export, 'w') as f:
                json.dump(logs, f, indent=2, default=str)

            print(f"Exported {len(logs)} log entries to {args.export}")

        if args.archive:
            archive_path = log_manager.create_log_archive()
            print(f"Log archive created: {archive_path}")

        if args.cleanup:
            log_manager._cleanup_old_logs()
            print("Log cleanup completed")

        if args.monitor:
            log_manager.start_monitoring()
            print("Log monitoring started. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            finally:
                log_manager.stop_monitoring()

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        log_manager.shutdown()
if __name__ == "__main__":
    main()
