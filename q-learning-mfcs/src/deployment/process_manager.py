"""Process Manager for MFC Deployment System.

This module provides process management capabilities for the MFC deployment system,
including process lifecycle management, monitoring, and resource allocation.
"""
import json
import logging
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import psutil


class ProcessState(Enum):
    """Process states."""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    RESTARTING = "restarting"
    UNKNOWN = "unknown"


class RestartPolicy(Enum):
    """Process restart policies."""
    NEVER = "never"
    ON_FAILURE = "on_failure"
    ALWAYS = "always"
    UNLESS_STOPPED = "unless_stopped"

@dataclass
class ProcessConfig:
    """Process configuration."""
    name: str
    command: str | list[str]
    working_dir: str | None = None
    environment: dict[str, str] = field(default_factory=dict)
    restart_policy: RestartPolicy = RestartPolicy.ON_FAILURE
    max_restarts: int = 5
    restart_delay: float = 1.0
    health_check_interval: float = 30.0
    health_check_timeout: float = 10.0
    health_check_command: str | None = None
    memory_limit: int | None = None  # MB
    cpu_limit: float | None = None  # Percentage
    log_file: str | None = None
    user: str | None = None
    group: str | None = None

    def __post_init__(self):
        """Post-initialization validation."""
        if isinstance(self.command, str):
            self.command = self.command.split()

        if not self.command:
            raise ValueError("Command cannot be empty")

@dataclass
class ProcessInfo:
    """Process runtime information."""
    config: ProcessConfig
    pid: int | None = None
    state: ProcessState = ProcessState.STOPPED
    start_time: datetime | None = None
    restart_count: int = 0
    last_exit_code: int | None = None
    last_error: str | None = None
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_mb: float = 0.0

    @property
    def uptime(self) -> timedelta | None:
        """Get process uptime."""
        if self.start_time and self.state == ProcessState.RUNNING:
            return datetime.now() - self.start_time
        return None

class HealthChecker:
    """Health check manager for processes."""

    def __init__(self, process_manager: 'ProcessManager'):
        self.process_manager = process_manager
        self.running = False
        self.thread: threading.Thread | None = None
        self.logger = logging.getLogger(__name__)

    def start(self):
        """Start health checking."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.thread.start()
        self.logger.info("Health checker started")

    def stop(self):
        """Stop health checking."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        self.logger.info("Health checker stopped")

    def _health_check_loop(self):
        """Main health check loop."""
        while self.running:
            try:
                for process_name, process_info in self.process_manager._processes.items():
                    if process_info.state == ProcessState.RUNNING:
                        self._check_process_health(process_info)

                time.sleep(5)  # Check every 5 seconds

            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                time.sleep(10)

    def _check_process_health(self, process_info: ProcessInfo):
        """Check health of a single process."""
        try:
            # Basic process existence check
            if not self._is_process_alive(process_info):
                self.logger.warning(f"Process {process_info.config.name} is not alive")
                self.process_manager._handle_process_death(process_info)
                return

            # Custom health check if configured
            if process_info.config.health_check_command:
                self._run_health_check_command(process_info)

            # Resource usage check
            self._check_resource_limits(process_info)

        except Exception as e:
            self.logger.error(f"Health check failed for {process_info.config.name}: {e}")

    def _is_process_alive(self, process_info: ProcessInfo) -> bool:
        """Check if process is alive."""
        if not process_info.pid:
            return False

        try:
            process = psutil.Process(process_info.pid)
            return process.is_running()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    def _run_health_check_command(self, process_info: ProcessInfo):
        """Run custom health check command."""
        try:
            result = subprocess.run(
                process_info.config.health_check_command.split(),
                timeout=process_info.config.health_check_timeout,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                self.logger.warning(
                    f"Health check failed for {process_info.config.name}: {result.stderr}"
                )
                # Could trigger restart based on policy

        except subprocess.TimeoutExpired:
            self.logger.warning(f"Health check timeout for {process_info.config.name}")
        except Exception as e:
            self.logger.error(f"Health check command error for {process_info.config.name}: {e}")

    def _check_resource_limits(self, process_info: ProcessInfo):
        """Check if process exceeds resource limits."""
        if not process_info.pid:
            return

        try:
            process = psutil.Process(process_info.pid)

            # Update resource usage stats
            process_info.cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            process_info.memory_mb = memory_info.rss / 1024 / 1024
            process_info.memory_percent = process.memory_percent()

            # Check memory limit
            if (process_info.config.memory_limit and
                process_info.memory_mb > process_info.config.memory_limit):
                self.logger.warning(
                    f"Process {process_info.config.name} exceeds memory limit: "
                    f"{process_info.memory_mb:.1f}MB > {process_info.config.memory_limit}MB"
                )
                # Could kill and restart process

            # Check CPU limit
            if (process_info.config.cpu_limit and
                process_info.cpu_percent > process_info.config.cpu_limit):
                self.logger.warning(
                    f"Process {process_info.config.name} exceeds CPU limit: "
                    f"{process_info.cpu_percent:.1f}% > {process_info.config.cpu_limit}%"
                )

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            self.logger.error(f"Failed to get resource usage for {process_info.config.name}: {e}")

class ProcessManager:
    """Manages multiple processes with lifecycle control and monitoring."""

    def __init__(self, log_dir: str = "/tmp/process-manager-logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)

        self._processes: dict[str, ProcessInfo] = {}
        self._lock = threading.RLock()
        self.health_checker = HealthChecker(self)
        self.logger = logging.getLogger(__name__)

        # Event callbacks
        self.on_process_start: Callable[[ProcessInfo], None] | None = None
        self.on_process_stop: Callable[[ProcessInfo], None] | None = None
        self.on_process_restart: Callable[[ProcessInfo], None] | None = None
        self.on_process_fail: Callable[[ProcessInfo], None] | None = None

        # Signal handlers
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down...")
            self.shutdown_all()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    def add_process(self, config: ProcessConfig) -> ProcessInfo:
        """Add a process configuration."""
        with self._lock:
            if config.name in self._processes:
                raise ValueError(f"Process {config.name} already exists")

            process_info = ProcessInfo(config=config)
            self._processes[config.name] = process_info

            self.logger.info(f"Added process configuration: {config.name}")
            return process_info

    def remove_process(self, name: str) -> bool:
        """Remove a process configuration."""
        with self._lock:
            if name not in self._processes:
                return False

            # Stop process if running
            self.stop_process(name)

            del self._processes[name]
            self.logger.info(f"Removed process configuration: {name}")
            return True

    def start_process(self, name: str) -> bool:
        """Start a process."""
        with self._lock:
            if name not in self._processes:
                self.logger.error(f"Process {name} not found")
                return False

            process_info = self._processes[name]

            if process_info.state in [ProcessState.STARTING, ProcessState.RUNNING]:
                self.logger.warning(f"Process {name} is already running")
                return True

            return self._start_process_internal(process_info)

    def _start_process_internal(self, process_info: ProcessInfo) -> bool:
        """Internal process start implementation."""
        config = process_info.config

        try:
            process_info.state = ProcessState.STARTING
            self.logger.info(f"Starting process: {config.name}")

            # Prepare environment
            env = os.environ.copy()
            env.update(config.environment)

            # Prepare log file
            log_file = None
            if config.log_file:
                log_file = open(config.log_file, 'a')

            # Start the process
            process = subprocess.Popen(
                config.command,
                cwd=config.working_dir,
                env=env,
                stdout=log_file or subprocess.PIPE,
                stderr=log_file or subprocess.PIPE,
                start_new_session=True
            )

            process_info.pid = process.pid
            process_info.state = ProcessState.RUNNING
            process_info.start_time = datetime.now()
            process_info.last_exit_code = None
            process_info.last_error = None

            self.logger.info(f"Process {config.name} started with PID {process.pid}")

            # Start monitoring thread
            monitor_thread = threading.Thread(
                target=self._monitor_process,
                args=(process_info, process),
                daemon=True
            )
            monitor_thread.start()

            # Trigger callback
            if self.on_process_start:
                self.on_process_start(process_info)

            return True

        except Exception as e:
            process_info.state = ProcessState.FAILED
            process_info.last_error = str(e)
            self.logger.error(f"Failed to start process {config.name}: {e}")

            if self.on_process_fail:
                self.on_process_fail(process_info)

            return False

    def _monitor_process(self, process_info: ProcessInfo, process: subprocess.Popen):
        """Monitor a process and handle its completion."""
        try:
            exit_code = process.wait()

            with self._lock:
                process_info.last_exit_code = exit_code

                if exit_code == 0:
                    process_info.state = ProcessState.STOPPED
                    self.logger.info(f"Process {process_info.config.name} exited normally")
                else:
                    process_info.state = ProcessState.FAILED
                    self.logger.warning(f"Process {process_info.config.name} exited with code {exit_code}")

                # Handle restart policy
                self._handle_process_death(process_info)

        except Exception as e:
            self.logger.error(f"Process monitoring error for {process_info.config.name}: {e}")
            process_info.state = ProcessState.FAILED
            process_info.last_error = str(e)

    def _handle_process_death(self, process_info: ProcessInfo):
        """Handle process death and restart if needed."""
        config = process_info.config

        # Trigger stop callback
        if self.on_process_stop:
            self.on_process_stop(process_info)

        # Check restart policy
        should_restart = False

        if config.restart_policy == RestartPolicy.ALWAYS:
            should_restart = True
        elif config.restart_policy == RestartPolicy.ON_FAILURE:
            should_restart = process_info.last_exit_code != 0
        elif config.restart_policy == RestartPolicy.UNLESS_STOPPED:
            should_restart = process_info.state != ProcessState.STOPPED

        # Check restart limits
        if should_restart and process_info.restart_count >= config.max_restarts:
            self.logger.error(
                f"Process {config.name} exceeded max restarts ({config.max_restarts})"
            )
            should_restart = False
            process_info.state = ProcessState.FAILED

        if should_restart:
            self.logger.info(f"Restarting process {config.name} in {config.restart_delay}s")
            process_info.state = ProcessState.RESTARTING
            process_info.restart_count += 1

            # Trigger restart callback
            if self.on_process_restart:
                self.on_process_restart(process_info)

            # Schedule restart
            restart_thread = threading.Thread(
                target=self._delayed_restart,
                args=(process_info,),
                daemon=True
            )
            restart_thread.start()

    def _delayed_restart(self, process_info: ProcessInfo):
        """Restart process after delay."""
        time.sleep(process_info.config.restart_delay)
        with self._lock:
            if process_info.state == ProcessState.RESTARTING:
                self._start_process_internal(process_info)

    def stop_process(self, name: str, timeout: float = 30.0) -> bool:
        """Stop a process gracefully."""
        with self._lock:
            if name not in self._processes:
                return False

            process_info = self._processes[name]

            if process_info.state not in [ProcessState.RUNNING, ProcessState.STARTING]:
                return True

            return self._stop_process_internal(process_info, timeout)

    def _stop_process_internal(self, process_info: ProcessInfo, timeout: float) -> bool:
        """Internal process stop implementation."""
        if not process_info.pid:
            return False

        try:
            process_info.state = ProcessState.STOPPING
            self.logger.info(f"Stopping process: {process_info.config.name}")

            # Try graceful shutdown first
            os.kill(process_info.pid, signal.SIGTERM)

            # Wait for graceful shutdown
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    process = psutil.Process(process_info.pid)
                    if not process.is_running():
                        break
                except psutil.NoSuchProcess:
                    break

                time.sleep(0.1)

            # Force kill if still running
            try:
                process = psutil.Process(process_info.pid)
                if process.is_running():
                    self.logger.warning(f"Force killing process {process_info.config.name}")
                    os.kill(process_info.pid, signal.SIGKILL)
                    time.sleep(1)
            except psutil.NoSuchProcess:
                pass

            process_info.state = ProcessState.STOPPED
            process_info.pid = None

            self.logger.info(f"Process {process_info.config.name} stopped")
            return True

        except Exception as e:
            self.logger.error(f"Failed to stop process {process_info.config.name}: {e}")
            process_info.last_error = str(e)
            return False

    def restart_process(self, name: str) -> bool:
        """Restart a process."""
        self.stop_process(name)
        time.sleep(1)  # Brief delay
        return self.start_process(name)

    def get_process_info(self, name: str) -> ProcessInfo | None:
        """Get process information."""
        with self._lock:
            return self._processes.get(name)

    def list_processes(self) -> dict[str, ProcessInfo]:
        """List all processes."""
        with self._lock:
            return self._processes.copy()

    def get_status(self) -> dict[str, Any]:
        """Get overall status."""
        with self._lock:
            total = len(self._processes)
            running = sum(1 for p in self._processes.values() if p.state == ProcessState.RUNNING)
            failed = sum(1 for p in self._processes.values() if p.state == ProcessState.FAILED)

            return {
                'total_processes': total,
                'running_processes': running,
                'failed_processes': failed,
                'processes': {
                    name: {
                        'state': info.state.value,
                        'pid': info.pid,
                        'uptime': str(info.uptime) if info.uptime else None,
                        'restart_count': info.restart_count,
                        'cpu_percent': info.cpu_percent,
                        'memory_mb': info.memory_mb,
                        'last_exit_code': info.last_exit_code,
                        'last_error': info.last_error
                    }
                    for name, info in self._processes.items()
                }
            }

    def start_health_checking(self):
        """Start health checking."""
        self.health_checker.start()

    def stop_health_checking(self):
        """Stop health checking."""
        self.health_checker.stop()

    def start_all(self) -> dict[str, bool]:
        """Start all processes."""
        results = {}
        with self._lock:
            for name in self._processes:
                results[name] = self.start_process(name)
        return results

    def stop_all(self, timeout: float = 30.0) -> dict[str, bool]:
        """Stop all processes."""
        results = {}
        with self._lock:
            for name in self._processes:
                results[name] = self.stop_process(name, timeout)
        return results

    def shutdown_all(self):
        """Shutdown all processes and the manager."""
        self.logger.info("Shutting down process manager...")

        self.stop_health_checking()
        self.stop_all()

        self.logger.info("Process manager shutdown complete")

    def save_config(self, config_file: str):
        """Save process configurations to file."""
        config_data = {
            'processes': [
                {
                    'name': info.config.name,
                    'command': info.config.command,
                    'working_dir': info.config.working_dir,
                    'environment': info.config.environment,
                    'restart_policy': info.config.restart_policy.value,
                    'max_restarts': info.config.max_restarts,
                    'restart_delay': info.config.restart_delay,
                    'health_check_interval': info.config.health_check_interval,
                    'health_check_timeout': info.config.health_check_timeout,
                    'health_check_command': info.config.health_check_command,
                    'memory_limit': info.config.memory_limit,
                    'cpu_limit': info.config.cpu_limit,
                    'log_file': info.config.log_file,
                    'user': info.config.user,
                    'group': info.config.group
                }
                for info in self._processes.values()
            ]
        }

        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)

        self.logger.info(f"Configuration saved to {config_file}")

    def load_config(self, config_file: str):
        """Load process configurations from file."""
        with open(config_file) as f:
            config_data = json.load(f)

        for proc_config in config_data.get('processes', []):
            config = ProcessConfig(
                name=proc_config['name'],
                command=proc_config['command'],
                working_dir=proc_config.get('working_dir'),
                environment=proc_config.get('environment', {}),
                restart_policy=RestartPolicy(proc_config.get('restart_policy', 'on_failure')),
                max_restarts=proc_config.get('max_restarts', 5),
                restart_delay=proc_config.get('restart_delay', 1.0),
                health_check_interval=proc_config.get('health_check_interval', 30.0),
                health_check_timeout=proc_config.get('health_check_timeout', 10.0),
                health_check_command=proc_config.get('health_check_command'),
                memory_limit=proc_config.get('memory_limit'),
                cpu_limit=proc_config.get('cpu_limit'),
                log_file=proc_config.get('log_file'),
                user=proc_config.get('user'),
                group=proc_config.get('group')
            )

            self.add_process(config)

        self.logger.info(f"Configuration loaded from {config_file}")


# Global process manager instance
def get_process_manager(log_dir: str = "/tmp/process-manager-logs") -> ProcessManager:
    """Get global process manager instance."""
    global _process_manager
    if _process_manager is None:
        _process_manager = ProcessManager(log_dir=log_dir)
    return _process_manager

def main():
    """Main entry point for process manager CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="MFC Process Manager")
    parser.add_argument("--config", help="Configuration file")
    parser.add_argument("--start", help="Start process by name")
    parser.add_argument("--stop", help="Stop process by name")
    parser.add_argument("--restart", help="Restart process by name")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--start-all", action="store_true", help="Start all processes")
    parser.add_argument("--stop-all", action="store_true", help="Stop all processes")
    parser.add_argument("--monitor", action="store_true", help="Start monitoring mode")

    args = parser.parse_args()

    manager = get_process_manager()

    try:
        if args.config:
            manager.load_config(args.config)

        if args.start:
            success = manager.start_process(args.start)
            print(f"Start {'succeeded' if success else 'failed'} for process {args.start}")

        if args.stop:
            success = manager.stop_process(args.stop)
            print(f"Stop {'succeeded' if success else 'failed'} for process {args.stop}")

        if args.restart:
            success = manager.restart_process(args.restart)
            print(f"Restart {'succeeded' if success else 'failed'} for process {args.restart}")

        if args.start_all:
            results = manager.start_all()
            for name, success in results.items():
                print(f"Start {'succeeded' if success else 'failed'} for process {name}")

        if args.stop_all:
            results = manager.stop_all()
            for name, success in results.items():
                print(f"Stop {'succeeded' if success else 'failed'} for process {name}")

        if args.status:
            status = manager.get_status()
            print(json.dumps(status, indent=2))

        if args.monitor:
            print("Starting monitoring mode (Ctrl+C to exit)...")
            manager.start_health_checking()
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nExiting monitoring mode...")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    finally:
        manager.shutdown_all()

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
