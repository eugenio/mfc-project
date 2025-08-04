#!/usr/bin/env python3
"""
Process Manager - Multi-platform process management
Agent Zeta - Deployment and Process Management

Supports systemd, supervisor, and manual process management
"""

import argparse
import json
import logging
import platform
import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path


class ProcessManagerType(Enum):
    """Supported process managers"""
    SYSTEMD = "systemd"
    SUPERVISOR = "supervisor"
    MANUAL = "manual"
    AUTO = "auto"

class ProcessManager:
    """Cross-platform process management"""

    def __init__(self, manager_type: ProcessManagerType = ProcessManagerType.AUTO):
        self.manager_type = self._detect_manager_type() if manager_type == ProcessManagerType.AUTO else manager_type
        self.logger = self._setup_logging()
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.deployment_dir = Path(__file__).parent

    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger("process_manager")

    def _detect_manager_type(self) -> ProcessManagerType:
        """Auto-detect the best process manager for the system"""
        system = platform.system().lower()

        # Check for systemd
        if system == "linux":
            # Check if systemd is available
            if self._command_exists("systemctl"):
                try:
                    result = subprocess.run(
                        ["systemctl", "is-system-running"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode in [0, 1]:  # 0=running, 1=degraded but functional
                        self.logger.info("Detected systemd")
                        return ProcessManagerType.SYSTEMD
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                    pass

            # Check for supervisor
            if self._command_exists("supervisorctl"):
                self.logger.info("Detected supervisor")
                return ProcessManagerType.SUPERVISOR

        # Fallback to manual
        self.logger.info("Using manual process management")
        return ProcessManagerType.MANUAL

    def _command_exists(self, command: str) -> bool:
        """Check if a command exists"""
        return shutil.which(command) is not None

    def install_service(self, service_name: str = "tts-service") -> bool:
        """Install the TTS service"""
        self.logger.info(f"Installing {service_name} using {self.manager_type.value}")

        if self.manager_type == ProcessManagerType.SYSTEMD:
            return self._install_systemd_service(service_name)
        elif self.manager_type == ProcessManagerType.SUPERVISOR:
            return self._install_supervisor_service(service_name)
        else:
            self.logger.info("Manual mode - no installation needed")
            return True

    def uninstall_service(self, service_name: str = "tts-service") -> bool:
        """Uninstall the TTS service"""
        self.logger.info(f"Uninstalling {service_name} using {self.manager_type.value}")

        if self.manager_type == ProcessManagerType.SYSTEMD:
            return self._uninstall_systemd_service(service_name)
        elif self.manager_type == ProcessManagerType.SUPERVISOR:
            return self._uninstall_supervisor_service(service_name)
        else:
            self.logger.info("Manual mode - no uninstallation needed")
            return True

    def start_service(self, service_name: str = "tts-service") -> bool:
        """Start the TTS service"""
        self.logger.info(f"Starting {service_name} using {self.manager_type.value}")

        if self.manager_type == ProcessManagerType.SYSTEMD:
            return self._systemd_start(service_name)
        elif self.manager_type == ProcessManagerType.SUPERVISOR:
            return self._supervisor_start(service_name)
        else:
            return self._manual_start()

    def stop_service(self, service_name: str = "tts-service") -> bool:
        """Stop the TTS service"""
        self.logger.info(f"Stopping {service_name} using {self.manager_type.value}")

        if self.manager_type == ProcessManagerType.SYSTEMD:
            return self._systemd_stop(service_name)
        elif self.manager_type == ProcessManagerType.SUPERVISOR:
            return self._supervisor_stop(service_name)
        else:
            return self._manual_stop()

    def restart_service(self, service_name: str = "tts-service") -> bool:
        """Restart the TTS service"""
        self.logger.info(f"Restarting {service_name} using {self.manager_type.value}")

        if self.manager_type == ProcessManagerType.SYSTEMD:
            return self._systemd_restart(service_name)
        elif self.manager_type == ProcessManagerType.SUPERVISOR:
            return self._supervisor_restart(service_name)
        else:
            return self._manual_restart()

    def get_status(self, service_name: str = "tts-service") -> dict:
        """Get service status"""
        if self.manager_type == ProcessManagerType.SYSTEMD:
            return self._systemd_status(service_name)
        elif self.manager_type == ProcessManagerType.SUPERVISOR:
            return self._supervisor_status(service_name)
        else:
            return self._manual_status()

    def enable_service(self, service_name: str = "tts-service") -> bool:
        """Enable service to start on boot"""
        if self.manager_type == ProcessManagerType.SYSTEMD:
            return self._systemd_enable(service_name)
        elif self.manager_type == ProcessManagerType.SUPERVISOR:
            # Supervisor services are enabled by default if autostart=true
            return True
        else:
            self.logger.warning("Manual mode - cannot enable automatic startup")
            return False

    def disable_service(self, service_name: str = "tts-service") -> bool:
        """Disable service from starting on boot"""
        if self.manager_type == ProcessManagerType.SYSTEMD:
            return self._systemd_disable(service_name)
        elif self.manager_type == ProcessManagerType.SUPERVISOR:
            self.logger.warning("Supervisor - modify config file to disable autostart")
            return True
        else:
            return True

    # Systemd implementation
    def _install_systemd_service(self, service_name: str) -> bool:
        """Install systemd service"""
        try:
            service_file = self.deployment_dir / "systemd" / f"{service_name}.service"
            target_file = Path(f"/etc/systemd/system/{service_name}.service")

            if not service_file.exists():
                self.logger.error(f"Service file not found: {service_file}")
                return False

            # Copy service file
            shutil.copy2(service_file, target_file)

            # Reload systemd
            subprocess.run(["systemctl", "daemon-reload"], check=True)

            self.logger.info(f"Systemd service {service_name} installed")
            return True

        except Exception as e:
            self.logger.error(f"Failed to install systemd service: {e}")
            return False

    def _uninstall_systemd_service(self, service_name: str) -> bool:
        """Uninstall systemd service"""
        try:
            # Stop and disable first
            self._systemd_stop(service_name)
            self._systemd_disable(service_name)

            # Remove service file
            target_file = Path(f"/etc/systemd/system/{service_name}.service")
            if target_file.exists():
                target_file.unlink()

            # Reload systemd
            subprocess.run(["systemctl", "daemon-reload"], check=True)

            self.logger.info(f"Systemd service {service_name} uninstalled")
            return True

        except Exception as e:
            self.logger.error(f"Failed to uninstall systemd service: {e}")
            return False

    def _systemd_start(self, service_name: str) -> bool:
        """Start systemd service"""
        try:
            subprocess.run(["systemctl", "start", service_name], check=True)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to start systemd service: {e}")
            return False

    def _systemd_stop(self, service_name: str) -> bool:
        """Stop systemd service"""
        try:
            subprocess.run(["systemctl", "stop", service_name], check=True)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to stop systemd service: {e}")
            return False

    def _systemd_restart(self, service_name: str) -> bool:
        """Restart systemd service"""
        try:
            subprocess.run(["systemctl", "restart", service_name], check=True)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to restart systemd service: {e}")
            return False

    def _systemd_enable(self, service_name: str) -> bool:
        """Enable systemd service"""
        try:
            subprocess.run(["systemctl", "enable", service_name], check=True)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to enable systemd service: {e}")
            return False

    def _systemd_disable(self, service_name: str) -> bool:
        """Disable systemd service"""
        try:
            subprocess.run(["systemctl", "disable", service_name], check=True)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to disable systemd service: {e}")
            return False

    def _systemd_status(self, service_name: str) -> dict:
        """Get systemd service status"""
        try:
            result = subprocess.run(
                ["systemctl", "show", service_name, "--property=ActiveState,SubState,MainPID,ExecMainStartTimestamp"],
                capture_output=True,
                text=True
            )

            status = {}
            for line in result.stdout.strip().split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    status[key] = value

            return {
                "manager": "systemd",
                "service": service_name,
                "active": status.get("ActiveState") == "active",
                "state": status.get("SubState"),
                "pid": status.get("MainPID"),
                "start_time": status.get("ExecMainStartTimestamp")
            }

        except Exception as e:
            self.logger.error(f"Failed to get systemd status: {e}")
            return {"manager": "systemd", "error": str(e)}

    # Supervisor implementation
    def _install_supervisor_service(self, service_name: str) -> bool:
        """Install supervisor service"""
        try:
            config_file = self.deployment_dir / "supervisor" / f"{service_name}.conf"

            # Find supervisor config directory
            supervisor_dirs = [
                "/etc/supervisor/conf.d",
                "/etc/supervisord/conf.d",
                "/usr/local/etc/supervisor/conf.d"
            ]

            target_dir = None
            for dir_path in supervisor_dirs:
                if Path(dir_path).exists():
                    target_dir = Path(dir_path)
                    break

            if not target_dir:
                self.logger.error("Could not find supervisor configuration directory")
                return False

            target_file = target_dir / f"{service_name}.conf"
            shutil.copy2(config_file, target_file)

            # Reload supervisor
            subprocess.run(["supervisorctl", "reread"], check=True)
            subprocess.run(["supervisorctl", "update"], check=True)

            self.logger.info(f"Supervisor service {service_name} installed")
            return True

        except Exception as e:
            self.logger.error(f"Failed to install supervisor service: {e}")
            return False

    def _uninstall_supervisor_service(self, service_name: str) -> bool:
        """Uninstall supervisor service"""
        try:
            # Stop service first
            self._supervisor_stop(service_name)

            # Find and remove config file
            supervisor_dirs = [
                "/etc/supervisor/conf.d",
                "/etc/supervisord/conf.d",
                "/usr/local/etc/supervisor/conf.d"
            ]

            for dir_path in supervisor_dirs:
                config_file = Path(dir_path) / f"{service_name}.conf"
                if config_file.exists():
                    config_file.unlink()
                    break

            # Update supervisor
            subprocess.run(["supervisorctl", "reread"], check=True)
            subprocess.run(["supervisorctl", "update"], check=True)

            self.logger.info(f"Supervisor service {service_name} uninstalled")
            return True

        except Exception as e:
            self.logger.error(f"Failed to uninstall supervisor service: {e}")
            return False

    def _supervisor_start(self, service_name: str) -> bool:
        """Start supervisor service"""
        try:
            subprocess.run(["supervisorctl", "start", f"{service_name}:*"], check=True)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to start supervisor service: {e}")
            return False

    def _supervisor_stop(self, service_name: str) -> bool:
        """Stop supervisor service"""
        try:
            subprocess.run(["supervisorctl", "stop", f"{service_name}:*"], check=True)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to stop supervisor service: {e}")
            return False

    def _supervisor_restart(self, service_name: str) -> bool:
        """Restart supervisor service"""
        try:
            subprocess.run(["supervisorctl", "restart", f"{service_name}:*"], check=True)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to restart supervisor service: {e}")
            return False

    def _supervisor_status(self, service_name: str) -> dict:
        """Get supervisor service status"""
        try:
            result = subprocess.run(
                ["supervisorctl", "status", f"{service_name}:*"],
                capture_output=True,
                text=True
            )

            return {
                "manager": "supervisor",
                "service": service_name,
                "output": result.stdout,
                "active": "RUNNING" in result.stdout
            }

        except Exception as e:
            self.logger.error(f"Failed to get supervisor status: {e}")
            return {"manager": "supervisor", "error": str(e)}

    # Manual implementation
    def _manual_start(self) -> bool:
        """Start service manually"""
        try:
            script_path = self.deployment_dir / "start_tts_service.sh"
            subprocess.run([str(script_path), "--daemon"], check=True)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to start service manually: {e}")
            return False

    def _manual_stop(self) -> bool:
        """Stop service manually"""
        try:
            script_path = self.deployment_dir / "stop_tts_service.sh"
            subprocess.run([str(script_path)], check=True)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to stop service manually: {e}")
            return False

    def _manual_restart(self) -> bool:
        """Restart service manually"""
        return self._manual_stop() and self._manual_start()

    def _manual_status(self) -> dict:
        """Get manual service status"""
        try:
            manager_script = self.deployment_dir / "tts_service_manager.py"
            result = subprocess.run(
                [sys.executable, str(manager_script), "--status"],
                capture_output=True,
                text=True,
                cwd=self.project_root / "q-learning-mfcs" / "src"
            )

            return {
                "manager": "manual",
                "output": result.stdout,
                "active": "🟢" in result.stdout
            }

        except Exception as e:
            self.logger.error(f"Failed to get manual status: {e}")
            return {"manager": "manual", "error": str(e)}

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="TTS Process Manager")
    parser.add_argument("--manager", choices=["systemd", "supervisor", "manual", "auto"],
                       default="auto", help="Process manager to use")
    parser.add_argument("--service-name", default="tts-service", help="Service name")
    parser.add_argument("--install", action="store_true", help="Install service")
    parser.add_argument("--uninstall", action="store_true", help="Uninstall service")
    parser.add_argument("--start", action="store_true", help="Start service")
    parser.add_argument("--stop", action="store_true", help="Stop service")
    parser.add_argument("--restart", action="store_true", help="Restart service")
    parser.add_argument("--enable", action="store_true", help="Enable service")
    parser.add_argument("--disable", action="store_true", help="Disable service")
    parser.add_argument("--status", action="store_true", help="Show service status")

    args = parser.parse_args()

    # Create process manager
    manager_type = ProcessManagerType(args.manager)
    pm = ProcessManager(manager_type)

    success = True

    if args.install:
        success = pm.install_service(args.service_name)

    if args.uninstall:
        success = success and pm.uninstall_service(args.service_name)

    if args.start:
        success = success and pm.start_service(args.service_name)

    if args.stop:
        success = success and pm.stop_service(args.service_name)

    if args.restart:
        success = success and pm.restart_service(args.service_name)

    if args.enable:
        success = success and pm.enable_service(args.service_name)

    if args.disable:
        success = success and pm.disable_service(args.service_name)

    if args.status:
        status = pm.get_status(args.service_name)
        print(json.dumps(status, indent=2))

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
