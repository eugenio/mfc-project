#!/usr/bin/env python3
"""
MFC Monitoring System Startup Script with HTTPS Support
Orchestrates the secure startup of all monitoring components.
"""

import sys
import time
import signal
import logging
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import argparse

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from ssl_config import (
        SSLConfig, load_ssl_config, initialize_ssl_infrastructure,
        test_ssl_connection, save_ssl_config
    )
except ImportError as e:
    print(f"❌ Failed to import SSL configuration: {e}")
    print("Please ensure ssl_config.py is in the correct location")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/mfc-monitoring.log')
    ]
)
logger = logging.getLogger(__name__)

class MonitoringService:
    """Represents a monitoring service process"""

    def __init__(self, name: str, command: List[str], port: int,
                 check_url: Optional[str] = None, ssl_required: bool = False):
        self.name = name
        self.command = command
        self.port = port
        self.check_url = check_url
        self.ssl_required = ssl_required
        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.restart_count = 0

    def start(self) -> bool:
        """Start the service"""
        if self.is_running:
            logger.warning(f"{self.name} is already running")
            return True

        try:
            logger.info(f"Starting {self.name}...")
            logger.debug(f"Command: {' '.join(self.command)}")

            # Start process
            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            self.is_running = True
            self.start_time = datetime.now()

            logger.info(f"✅ {self.name} started (PID: {self.process.pid})")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to start {self.name}: {e}")
            return False

    def check_health(self) -> bool:
        """Check if service is healthy"""
        if not self.is_running or not self.process:
            return False

        # Check if process is still running
        if self.process.poll() is not None:
            self.is_running = False
            logger.error(f"{self.name} process has stopped (exit code: {self.process.returncode})")
            return False

        # Check URL if provided
        if self.check_url:
            try:
                import requests
                response = requests.get(self.check_url, timeout=5, verify=False)
                return response.status_code == 200
            except Exception:
                return False

        return True

    def stop(self) -> bool:
        """Stop the service"""
        if not self.is_running or not self.process:
            return True

        try:
            logger.info(f"Stopping {self.name}...")

            # Send SIGTERM first
            self.process.terminate()

            # Wait for graceful shutdown
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if necessary
                logger.warning(f"Force killing {self.name}...")
                self.process.kill()
                self.process.wait()

            self.is_running = False
            self.process = None

            logger.info(f"✅ {self.name} stopped")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to stop {self.name}: {e}")
            return False

    def restart(self) -> bool:
        """Restart the service"""
        logger.info(f"Restarting {self.name}...")
        self.restart_count += 1

        if self.stop():
            time.sleep(2)  # Wait before restart
            return self.start()

        return False

    def get_status(self) -> Dict:
        """Get service status information"""
        uptime = None
        if self.start_time and self.is_running:
            uptime = (datetime.now() - self.start_time).total_seconds()

        return {
            "name": self.name,
            "running": self.is_running,
            "port": self.port,
            "pid": self.process.pid if self.process else None,
            "uptime_seconds": uptime,
            "restart_count": self.restart_count,
            "ssl_required": self.ssl_required
        }

class MonitoringOrchestrator:
    """Orchestrates all monitoring services with HTTPS support"""

    ssl_config: Optional[SSLConfig]

    def __init__(self, ssl_config: Optional[SSLConfig] = None):
        self.ssl_config = ssl_config or load_ssl_config()
        self.services: List[MonitoringService] = []
        self.shutdown_requested = False
        self.health_check_interval = 30  # seconds
        self.health_check_thread: Optional[threading.Thread] = None

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self._initialize_services()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_requested = True

    def _initialize_services(self):
        """Initialize all monitoring services"""
        current_dir = Path(__file__).parent

        # FastAPI Dashboard API (HTTPS)
        api_command = [
            sys.executable, str(current_dir / "dashboard_api.py"),
            "--host", "0.0.0.0"
        ]
        if self.ssl_config:
            api_command.extend(["--port", str(self.ssl_config.https_port_api)])

        api_service = MonitoringService(
            name="Dashboard API",
            command=api_command,
            port=self.ssl_config.https_port_api if self.ssl_config else 8000,
            check_url=f"{'https' if self.ssl_config else 'http'}://{self.ssl_config.domain if self.ssl_config else 'localhost'}:{self.ssl_config.https_port_api if self.ssl_config else 8000}/health",
            ssl_required=bool(self.ssl_config)
        )
        self.services.append(api_service)

        # Streamlit Frontend (HTTPS)
        frontend_command = [
            sys.executable, str(current_dir / "dashboard_frontend.py"),
            "run_https"
        ]
        if self.ssl_config:
            frontend_command.append(str(self.ssl_config.https_port_frontend))

        frontend_service = MonitoringService(
            name="Dashboard Frontend",
            command=frontend_command,
            port=self.ssl_config.https_port_frontend if self.ssl_config else 8501,
            ssl_required=bool(self.ssl_config)
        )
        self.services.append(frontend_service)

        # WebSocket Streamer (WSS)
        ws_command = [
            sys.executable, str(current_dir / "realtime_streamer.py"),
            "--host", "0.0.0.0"
        ]
        if self.ssl_config:
            ws_command.extend(["--port", str(self.ssl_config.wss_port_streaming)])

        ws_service = MonitoringService(
            name="WebSocket Streamer",
            command=ws_command,
            port=self.ssl_config.wss_port_streaming if self.ssl_config else 8001,
            ssl_required=bool(self.ssl_config)
        )
        self.services.append(ws_service)

    def check_ssl_infrastructure(self) -> bool:
        """Check SSL infrastructure before starting services"""
        if not self.ssl_config:
            logger.warning("No SSL configuration found - running in HTTP mode")
            return True

        logger.info("Checking SSL infrastructure...")

        # Check certificate files
        cert_path = Path(self.ssl_config.cert_file)
        key_path = Path(self.ssl_config.key_file)

        if not cert_path.exists() or not key_path.exists():
            logger.warning("SSL certificates not found, attempting to initialize...")
            success, updated_config = initialize_ssl_infrastructure(self.ssl_config)

            if success:
                self.ssl_config = updated_config
                logger.info("✅ SSL infrastructure initialized")
                return True
            else:
                logger.error("❌ Failed to initialize SSL infrastructure")
                logger.info("Falling back to HTTP mode")
                self.ssl_config = None
                self._initialize_services()  # Reinitialize without SSL
                return True

        logger.info("✅ SSL certificates found")
        return True

    def start_all_services(self) -> bool:
        """Start all monitoring services"""
        logger.info("=== Starting MFC Monitoring System ===")

        # Check SSL infrastructure first
        if not self.check_ssl_infrastructure():
            return False

        # Display configuration
        self.print_configuration()

        success_count = 0

        for service in self.services:
            if service.start():
                success_count += 1
                # Give service time to start
                time.sleep(2)
            else:
                logger.error(f"Failed to start {service.name}")

        if success_count == len(self.services):
            logger.info("✅ All services started successfully")
            self.start_health_monitoring()
            return True
        else:
            logger.error(f"❌ Only {success_count}/{len(self.services)} services started")
            return False

    def stop_all_services(self):
        """Stop all monitoring services"""
        logger.info("=== Stopping MFC Monitoring System ===")

        # Stop health monitoring
        if self.health_check_thread:
            self.shutdown_requested = True
            self.health_check_thread.join(timeout=5)

        # Stop services in reverse order
        for service in reversed(self.services):
            service.stop()

        logger.info("✅ All services stopped")

    def start_health_monitoring(self):
        """Start health monitoring thread"""
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self.health_check_thread.start()
        logger.info("Health monitoring started")

    def _health_check_loop(self):
        """Continuous health checking loop"""
        consecutive_failures = {service.name: 0 for service in self.services}
        max_failures = 3

        while not self.shutdown_requested:
            for service in self.services:
                if not service.check_health():
                    consecutive_failures[service.name] += 1
                    logger.warning(f"{service.name} health check failed ({consecutive_failures[service.name]}/{max_failures})")

                    if consecutive_failures[service.name] >= max_failures:
                        logger.error(f"{service.name} has failed {max_failures} consecutive health checks, restarting...")
                        if service.restart():
                            consecutive_failures[service.name] = 0
                        else:
                            logger.error(f"Failed to restart {service.name}")
                else:
                    consecutive_failures[service.name] = 0

            # Wait before next health check
            for _ in range(self.health_check_interval):
                if self.shutdown_requested:
                    break
                time.sleep(1)

    def print_configuration(self):
        """Print current configuration"""
        logger.info("=== Configuration ===")

        if self.ssl_config:
            logger.info("SSL Mode: HTTPS/WSS")
            logger.info(f"Domain: {self.ssl_config.domain}")
            logger.info(f"Certificate: {self.ssl_config.cert_file}")
            logger.info("Ports:")
            logger.info(f"  - Dashboard API: {self.ssl_config.https_port_api} (HTTPS)")
            logger.info(f"  - Dashboard Frontend: {self.ssl_config.https_port_frontend} (HTTPS)")
            logger.info(f"  - WebSocket Streamer: {self.ssl_config.wss_port_streaming} (WSS)")
        else:
            logger.info("SSL Mode: HTTP/WS (Development)")
            logger.info("Ports:")
            logger.info("  - Dashboard API: 8000 (HTTP)")
            logger.info("  - Dashboard Frontend: 8501 (HTTP)")
            logger.info("  - WebSocket Streamer: 8001 (WS)")

        logger.info("===================")

    def print_status(self):
        """Print status of all services"""
        print("\n=== MFC Monitoring System Status ===")

        for service in self.services:
            status = service.get_status()
            status_icon = "🟢" if status["running"] else "🔴"
            ssl_icon = "🔒" if status["ssl_required"] else "🔓"

            print(f"{status_icon} {ssl_icon} {status['name']}")
            print(f"   Port: {status['port']}")

            if status["running"]:
                print(f"   PID: {status['pid']}")
                if status["uptime_seconds"]:
                    uptime_hours = status["uptime_seconds"] / 3600
                    print(f"   Uptime: {uptime_hours:.2f} hours")
                if status["restart_count"] > 0:
                    print(f"   Restarts: {status['restart_count']}")
            else:
                print("   Status: Stopped")

            print()

        print("=====================================\n")

    def run_interactive_mode(self):
        """Run in interactive mode with status monitoring"""
        try:
            while not self.shutdown_requested:
                # Print status every 30 seconds
                self.print_status()

                # Wait with interrupt checking
                for _ in range(30):
                    if self.shutdown_requested:
                        break
                    time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Interactive mode interrupted")
        finally:
            self.stop_all_services()

def create_ssl_config_interactive() -> SSLConfig:
    """Create SSL configuration interactively"""
    print("=== SSL Configuration Setup ===")

    domain = input("Domain name (default: localhost): ").strip() or "localhost"
    email = input("Email for Let's Encrypt (default: admin@example.com): ").strip() or "admin@example.com"

    use_letsencrypt = domain != "localhost"
    if domain != "localhost":
        use_le = input("Use Let's Encrypt for certificates? (y/N): ").strip().lower()
        use_letsencrypt = use_le in ['y', 'yes']

    staging = False
    if use_letsencrypt:
        staging_input = input("Use Let's Encrypt staging (for testing)? (y/N): ").strip().lower()
        staging = staging_input in ['y', 'yes']

    config = SSLConfig(
        domain=domain,
        email=email,
        use_letsencrypt=use_letsencrypt,
        staging=staging
    )

    print("\nConfiguration created:")
    print(f"  Domain: {config.domain}")
    print(f"  Let's Encrypt: {config.use_letsencrypt}")
    print(f"  Staging: {config.staging}")
    print(f"  API Port: {config.https_port_api}")
    print(f"  Frontend Port: {config.https_port_frontend}")
    print(f"  WebSocket Port: {config.wss_port_streaming}")

    return config

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="MFC Monitoring System with HTTPS Support")
    parser.add_argument("--init-ssl", action="store_true", help="Initialize SSL infrastructure")
    parser.add_argument("--config-ssl", action="store_true", help="Configure SSL interactively")
    parser.add_argument("--test-ssl", action="store_true", help="Test SSL connections")
    parser.add_argument("--status", action="store_true", help="Show service status and exit")
    parser.add_argument("--stop", action="store_true", help="Stop all services")
    parser.add_argument("--no-ssl", action="store_true", help="Disable SSL (HTTP mode)")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon (non-interactive)")

    args = parser.parse_args()

    if args.config_ssl:
        config = create_ssl_config_interactive()
        if save_ssl_config(config):
            print("✅ SSL configuration saved")
        else:
            print("❌ Failed to save SSL configuration")
        return

    if args.init_ssl:
        logger.info("Initializing SSL infrastructure...")
        success, config = initialize_ssl_infrastructure()
        if success:
            print("✅ SSL infrastructure initialized successfully")
            print(f"Domain: {config.domain}")
            print(f"Certificate: {config.cert_file}")
            print(f"Key: {config.key_file}")
        else:
            print("❌ SSL initialization failed")
            sys.exit(1)
        return

    if args.test_ssl:
        ssl_config = load_ssl_config()
        if not ssl_config:
            print("❌ No SSL configuration found")
            sys.exit(1)

        ports = [ssl_config.https_port_api, ssl_config.https_port_frontend, ssl_config.wss_port_streaming]
        all_passed = True

        for service, port in zip(["API", "Frontend", "WebSocket"], ports):
            if test_ssl_connection(ssl_config.domain, port, timeout=5):
                print(f"✅ {service} SSL test passed (port {port})")
            else:
                print(f"❌ {service} SSL test failed (port {port})")
                all_passed = False

        sys.exit(0 if all_passed else 1)

    # Load SSL configuration (unless disabled)
    ssl_config = None if args.no_ssl else load_ssl_config()

    # Create orchestrator
    orchestrator = MonitoringOrchestrator(ssl_config)

    if args.status:
        orchestrator.print_status()
        return

    if args.stop:
        orchestrator.stop_all_services()
        return

    # Start all services
    if orchestrator.start_all_services():
        if args.daemon:
            logger.info("Running in daemon mode. Use SIGTERM to stop.")
            try:
                while not orchestrator.shutdown_requested:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        else:
            logger.info("Running in interactive mode. Press Ctrl+C to stop.")
            orchestrator.run_interactive_mode()
    else:
        logger.error("Failed to start monitoring system")
        sys.exit(1)

if __name__ == "__main__":
    main()
