#!/usr/bin/env python3
"""
Service Orchestrator - TTS Service Dependencies and Coordinated Startup
Agent Zeta - Deployment and Process Management

Handles service dependencies, coordinated startup/shutdown, and integration with main project
"""

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from deployment.log_management import LogManager, setup_tts_logging
    from deployment.process_manager import ProcessManager
    from deployment.service_discovery import ServiceRegistry, TTSServiceDiscovery
    from deployment.tts_service_manager import TTSServiceManager
    from notifications.base import NotificationConfig, NotificationLevel
    from notifications.tts_handler import TTSMode, TTSNotificationHandler
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    sys.exit(1)

class ServiceDependencyType(Enum):
    """Types of service dependencies"""
    REQUIRED = "required"        # Must be available before starting
    OPTIONAL = "optional"        # Nice to have, but not blocking
    WEAK = "weak"               # No startup dependency, but may be used at runtime

class OrchestrationPhase(Enum):
    """Orchestration phases"""
    INITIALIZATION = "initialization"
    PRE_START = "pre_start"
    STARTUP = "startup"
    POST_START = "post_start"
    RUNNING = "running"
    PRE_STOP = "pre_stop"
    SHUTDOWN = "shutdown"
    POST_STOP = "post_stop"
    CLEANUP = "cleanup"

@dataclass
class ServiceDependency:
    """Service dependency definition"""
    service_name: str
    dependency_type: ServiceDependencyType
    timeout: int = 30  # seconds
    health_check_url: str | None = None
    startup_delay: int = 0  # seconds to wait after dependency starts

@dataclass
class OrchestrationStep:
    """Single orchestration step"""
    phase: OrchestrationPhase
    name: str
    description: str
    action: Callable
    timeout: int = 60
    required: bool = True
    dependencies: list[str] = field(default_factory=list)

class ServiceOrchestrator:
    """Orchestrates TTS service lifecycle with dependency management"""

    def __init__(self, config_file: str | None = None):
        self.config_file = config_file or "orchestration_config.json"
        self.logger = self._setup_logging()

        # Core components
        self.tts_manager: TTSServiceManager | None = None
        self.service_registry: ServiceRegistry | None = None
        self.service_discovery: TTSServiceDiscovery | None = None
        self.process_manager: ProcessManager | None = None
        self.log_manager: LogManager | None = None

        # Orchestration state
        self.dependencies: list[ServiceDependency] = []
        self.orchestration_steps: list[OrchestrationStep] = []
        self.current_phase = OrchestrationPhase.INITIALIZATION
        self.running_services: set[str] = set()
        self.failed_services: set[str] = set()
        self.shutdown_requested = False

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.health_monitor_thread: threading.Thread | None = None
        self.shutdown_event = threading.Event()

        # Load configuration
        self._load_configuration()
        self._setup_orchestration_steps()
        self._setup_signal_handlers()

    def _setup_logging(self) -> logging.Logger:
        """Setup orchestrator logging"""
        return setup_tts_logging("tts-orchestrator", environment="production")

    def _load_configuration(self):
        """Load orchestration configuration"""
        config_path = Path(self.config_file)

        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)

                # Load dependencies
                for dep_config in config.get('dependencies', []):
                    dependency = ServiceDependency(
                        service_name=dep_config['service_name'],
                        dependency_type=ServiceDependencyType(dep_config['dependency_type']),
                        timeout=dep_config.get('timeout', 30),
                        health_check_url=dep_config.get('health_check_url'),
                        startup_delay=dep_config.get('startup_delay', 0)
                    )
                    self.dependencies.append(dependency)

                self.logger.info(f"Loaded {len(self.dependencies)} service dependencies")

            except Exception as e:
                self.logger.error(f"Failed to load configuration: {e}")
                self._create_default_configuration()
        else:
            self._create_default_configuration()

    def _create_default_configuration(self):
        """Create default orchestration configuration"""
        # Default dependencies for TTS service
        default_dependencies = [
            ServiceDependency(
                service_name="monitoring-api",
                dependency_type=ServiceDependencyType.OPTIONAL,
                timeout=15,
                health_check_url="http://localhost:8000/health"
            ),
            ServiceDependency(
                service_name="logging-service",
                dependency_type=ServiceDependencyType.WEAK,
                timeout=10
            )
        ]

        self.dependencies = default_dependencies
        self._save_configuration()

    def _save_configuration(self):
        """Save current configuration"""
        config = {
            "dependencies": [
                {
                    "service_name": dep.service_name,
                    "dependency_type": dep.dependency_type.value,
                    "timeout": dep.timeout,
                    "health_check_url": dep.health_check_url,
                    "startup_delay": dep.startup_delay
                }
                for dep in self.dependencies
            ]
        }

        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)

            self.logger.info("Configuration saved")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")

    def _setup_orchestration_steps(self):
        """Setup orchestration steps"""
        self.orchestration_steps = [
            # Initialization phase
            OrchestrationStep(
                phase=OrchestrationPhase.INITIALIZATION,
                name="initialize_components",
                description="Initialize core components",
                action=self._initialize_components,
                timeout=30
            ),

            # Pre-start phase
            OrchestrationStep(
                phase=OrchestrationPhase.PRE_START,
                name="check_dependencies",
                description="Check service dependencies",
                action=self._check_dependencies,
                timeout=60,
                dependencies=["initialize_components"]
            ),

            OrchestrationStep(
                phase=OrchestrationPhase.PRE_START,
                name="prepare_environment",
                description="Prepare TTS service environment",
                action=self._prepare_environment,
                timeout=30,
                dependencies=["check_dependencies"]
            ),

            # Startup phase
            OrchestrationStep(
                phase=OrchestrationPhase.STARTUP,
                name="start_tts_service",
                description="Start TTS service",
                action=self._start_tts_service,
                timeout=120,
                dependencies=["prepare_environment"]
            ),

            OrchestrationStep(
                phase=OrchestrationPhase.STARTUP,
                name="register_service",
                description="Register TTS service for discovery",
                action=self._register_service,
                timeout=15,
                dependencies=["start_tts_service"]
            ),

            # Post-start phase
            OrchestrationStep(
                phase=OrchestrationPhase.POST_START,
                name="validate_service",
                description="Validate TTS service health",
                action=self._validate_service,
                timeout=30,
                dependencies=["register_service"]
            ),

            OrchestrationStep(
                phase=OrchestrationPhase.POST_START,
                name="start_monitoring",
                description="Start health monitoring",
                action=self._start_monitoring,
                timeout=15,
                dependencies=["validate_service"]
            ),

            # Shutdown steps (reverse order)
            OrchestrationStep(
                phase=OrchestrationPhase.PRE_STOP,
                name="stop_monitoring",
                description="Stop health monitoring",
                action=self._stop_monitoring,
                timeout=15
            ),

            OrchestrationStep(
                phase=OrchestrationPhase.SHUTDOWN,
                name="deregister_service",
                description="Deregister TTS service",
                action=self._deregister_service,
                timeout=15
            ),

            OrchestrationStep(
                phase=OrchestrationPhase.SHUTDOWN,
                name="stop_tts_service",
                description="Stop TTS service",
                action=self._stop_tts_service,
                timeout=60
            ),

            OrchestrationStep(
                phase=OrchestrationPhase.CLEANUP,
                name="cleanup_resources",
                description="Clean up resources",
                action=self._cleanup_resources,
                timeout=30
            )
        ]

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown_requested = True
            self.shutdown_event.set()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    def start_orchestration(self) -> bool:
        """Start complete service orchestration"""
        self.logger.info("=== Starting TTS Service Orchestration ===")

        try:
            # Execute startup phases
            startup_phases = [
                OrchestrationPhase.INITIALIZATION,
                OrchestrationPhase.PRE_START,
                OrchestrationPhase.STARTUP,
                OrchestrationPhase.POST_START
            ]

            for phase in startup_phases:
                if not self._execute_phase(phase):
                    self.logger.error(f"Failed during {phase.value} phase")
                    return False

            self.current_phase = OrchestrationPhase.RUNNING
            self.logger.info("✅ TTS Service orchestration completed successfully")

            # Send success notification
            self._send_notification(
                "TTS Service Started",
                "TTS service orchestration completed successfully",
                NotificationLevel.SUCCESS
            )

            return True

        except Exception as e:
            self.logger.error(f"Orchestration failed: {e}")
            self._send_notification(
                "TTS Service Failed",
                f"TTS service orchestration failed: {e}",
                NotificationLevel.ERROR
            )
            return False

    def stop_orchestration(self) -> bool:
        """Stop service orchestration"""
        self.logger.info("=== Stopping TTS Service Orchestration ===")

        try:
            # Execute shutdown phases
            shutdown_phases = [
                OrchestrationPhase.PRE_STOP,
                OrchestrationPhase.SHUTDOWN,
                OrchestrationPhase.CLEANUP
            ]

            for phase in shutdown_phases:
                self._execute_phase(phase)  # Don't fail on shutdown errors

            self.current_phase = OrchestrationPhase.INITIALIZATION
            self.logger.info("✅ TTS Service orchestration stopped")

            return True

        except Exception as e:
            self.logger.error(f"Shutdown failed: {e}")
            return False

    def _execute_phase(self, phase: OrchestrationPhase) -> bool:
        """Execute all steps in a phase"""
        self.logger.info(f"Executing phase: {phase.value}")
        self.current_phase = phase

        # Get steps for this phase
        phase_steps = [step for step in self.orchestration_steps if step.phase == phase]

        if not phase_steps:
            self.logger.debug(f"No steps for phase {phase.value}")
            return True

        # Execute steps with dependency resolution
        completed_steps = set()
        failed_steps = set()

        while len(completed_steps) < len(phase_steps):
            progress_made = False

            for step in phase_steps:
                if step.name in completed_steps or step.name in failed_steps:
                    continue

                # Check if dependencies are met
                dependencies_met = all(
                    dep in completed_steps for dep in step.dependencies
                )

                if not dependencies_met:
                    continue

                # Execute step
                self.logger.info(f"Executing step: {step.name} - {step.description}")

                try:
                    success = self._execute_step(step)

                    if success:
                        completed_steps.add(step.name)
                        self.logger.info(f"✅ Step completed: {step.name}")
                    else:
                        if step.required:
                            failed_steps.add(step.name)
                            self.logger.error(f"❌ Required step failed: {step.name}")
                            return False
                        else:
                            self.logger.warning(f"⚠️ Optional step failed: {step.name}")
                            completed_steps.add(step.name)  # Mark as completed to continue

                    progress_made = True

                except Exception as e:
                    self.logger.error(f"❌ Step error: {step.name} - {e}")
                    if step.required:
                        failed_steps.add(step.name)
                        return False
                    else:
                        completed_steps.add(step.name)
                        progress_made = True

            if not progress_made:
                self.logger.error("No progress made in phase execution")
                return False

        self.logger.info(f"✅ Phase completed: {phase.value}")
        return True

    def _execute_step(self, step: OrchestrationStep) -> bool:
        """Execute a single orchestration step"""
        try:
            # Execute with timeout
            future = self.executor.submit(step.action)
            result = future.result(timeout=step.timeout)
            return result if result is not None else True

        except Exception as e:
            self.logger.error(f"Step execution failed: {step.name} - {e}")
            return False

    # Orchestration step implementations
    def _initialize_components(self) -> bool:
        """Initialize core components"""
        try:
            # Initialize log manager
            self.log_manager = LogManager()

            # Initialize service registry
            self.service_registry = ServiceRegistry()

            # Initialize service discovery
            self.service_discovery = TTSServiceDiscovery(self.service_registry)

            # Initialize process manager
            self.process_manager = ProcessManager()

            # Initialize TTS service manager
            self.tts_manager = TTSServiceManager()

            self.logger.info("Core components initialized")
            return True

        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            return False

    def _check_dependencies(self) -> bool:
        """Check service dependencies"""
        self.logger.info("Checking service dependencies...")

        for dependency in self.dependencies:
            self.logger.info(f"Checking dependency: {dependency.service_name} ({dependency.dependency_type.value})")

            if dependency.dependency_type == ServiceDependencyType.WEAK:
                # Skip weak dependencies
                continue

            available = self._check_dependency_availability(dependency)

            if not available:
                if dependency.dependency_type == ServiceDependencyType.REQUIRED:
                    self.logger.error(f"Required dependency not available: {dependency.service_name}")
                    return False
                else:
                    self.logger.warning(f"Optional dependency not available: {dependency.service_name}")

        self.logger.info("Dependency check completed")
        return True

    def _check_dependency_availability(self, dependency: ServiceDependency) -> bool:
        """Check if a specific dependency is available"""
        try:
            if dependency.health_check_url:
                # HTTP health check
                import requests
                response = requests.get(dependency.health_check_url, timeout=dependency.timeout)
                return response.status_code == 200
            else:
                # Service discovery check
                services = self.service_registry.find_services(service_name=dependency.service_name)
                return len(services) > 0

        except Exception as e:
            self.logger.warning(f"Dependency check failed for {dependency.service_name}: {e}")
            return False

    def _prepare_environment(self) -> bool:
        """Prepare TTS service environment"""
        try:
            # Ensure required directories exist
            directories = [
                "/tmp/tts-service-logs",
                "/tmp/tts-service-pids"
            ]

            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)

            # Set environment variables
            os.environ["TTS_ORCHESTRATION_MODE"] = "true"
            os.environ["TTS_SERVICE_STARTUP_TIME"] = datetime.now().isoformat()

            self.logger.info("Environment prepared")
            return True

        except Exception as e:
            self.logger.error(f"Environment preparation failed: {e}")
            return False

    def _start_tts_service(self) -> bool:
        """Start TTS service"""
        try:
            if not self.tts_manager:
                return False

            success = self.tts_manager.start_all_services()

            if success:
                # Wait for dependencies to stabilize
                for dependency in self.dependencies:
                    if dependency.startup_delay > 0:
                        self.logger.info(f"Waiting {dependency.startup_delay}s for {dependency.service_name} to stabilize")
                        time.sleep(dependency.startup_delay)

            return success

        except Exception as e:
            self.logger.error(f"TTS service start failed: {e}")
            return False

    def _register_service(self) -> bool:
        """Register TTS service for discovery"""
        try:
            if not self.service_discovery:
                return False

            service_id = self.service_discovery.register_tts_service(
                service_name="tts-service-orchestrated",
                capabilities=["pyttsx3", "hybrid", "orchestrated"]
            )

            if service_id:
                self.logger.info(f"Service registered with ID: {service_id}")
                return True
            else:
                return False

        except Exception as e:
            self.logger.error(f"Service registration failed: {e}")
            return False

    def _validate_service(self) -> bool:
        """Validate TTS service health"""
        try:
            # Test TTS functionality
            handler = TTSNotificationHandler(TTSMode.TTS_ONLY)
            config = NotificationConfig(
                title="Orchestration Test",
                message="TTS service orchestration validation",
                level=NotificationLevel.DEBUG
            )

            return handler.send_notification(config)

        except Exception as e:
            self.logger.error(f"Service validation failed: {e}")
            return False

    def _start_monitoring(self) -> bool:
        """Start health monitoring"""
        try:
            self.health_monitor_thread = threading.Thread(
                target=self._health_monitor_loop,
                daemon=True
            )
            self.health_monitor_thread.start()

            self.logger.info("Health monitoring started")
            return True

        except Exception as e:
            self.logger.error(f"Monitoring start failed: {e}")
            return False

    def _stop_monitoring(self) -> bool:
        """Stop health monitoring"""
        try:
            self.shutdown_event.set()

            if self.health_monitor_thread:
                self.health_monitor_thread.join(timeout=10)

            self.logger.info("Health monitoring stopped")
            return True

        except Exception as e:
            self.logger.error(f"Monitoring stop failed: {e}")
            return False

    def _deregister_service(self) -> bool:
        """Deregister TTS service"""
        try:
            if self.service_discovery:
                result = self.service_discovery.deregister_tts_service()
                self.logger.info("Service deregistered")
                return result
            return True

        except Exception as e:
            self.logger.error(f"Service deregistration failed: {e}")
            return False

    def _stop_tts_service(self) -> bool:
        """Stop TTS service"""
        try:
            if self.tts_manager:
                self.tts_manager.stop_all_services()
                self.logger.info("TTS service stopped")

            return True

        except Exception as e:
            self.logger.error(f"TTS service stop failed: {e}")
            return False

    def _cleanup_resources(self) -> bool:
        """Clean up resources"""
        try:
            # Shutdown executor
            self.executor.shutdown(wait=True)

            # Clean up components
            if self.log_manager:
                self.log_manager.shutdown()

            # Clean up environment variables
            for var in ["TTS_ORCHESTRATION_MODE", "TTS_SERVICE_STARTUP_TIME"]:
                os.environ.pop(var, None)

            self.logger.info("Resources cleaned up")
            return True

        except Exception as e:
            self.logger.error(f"Resource cleanup failed: {e}")
            return False

    def _health_monitor_loop(self):
        """Health monitoring loop"""
        self.logger.info("Health monitoring loop started")

        while not self.shutdown_event.is_set():
            try:
                # Check TTS service health
                if self.tts_manager:
                    status = self.tts_manager.get_status_all()

                    healthy_services = 0
                    total_services = len(status)

                    for service_name, service_status in status.items():
                        if service_status.get("status") == "running":
                            healthy_services += 1
                        elif service_status.get("status") == "failed":
                            self.logger.warning(f"Service unhealthy: {service_name}")

                    # Log health status
                    if healthy_services == total_services:
                        self.logger.debug(f"All {total_services} services healthy")
                    else:
                        self.logger.warning(f"Health status: {healthy_services}/{total_services} services healthy")

                # Wait for next check
                for _ in range(60):  # 60 second intervals
                    if self.shutdown_event.is_set():
                        break
                    time.sleep(1)

            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                time.sleep(30)

        self.logger.info("Health monitoring loop stopped")

    def _send_notification(self, title: str, message: str, level: NotificationLevel):
        """Send orchestration notification"""
        try:
            handler = TTSNotificationHandler(TTSMode.TTS_WITH_FALLBACK)
            config = NotificationConfig(title, message, level)
            handler.send_notification(config)
        except Exception as e:
            self.logger.warning(f"Failed to send notification: {e}")

    def get_orchestration_status(self) -> dict:
        """Get current orchestration status"""
        status = {
            "current_phase": self.current_phase.value,
            "running_services": list(self.running_services),
            "failed_services": list(self.failed_services),
            "dependencies": [
                {
                    "service_name": dep.service_name,
                    "type": dep.dependency_type.value,
                    "available": self._check_dependency_availability(dep)
                }
                for dep in self.dependencies
            ]
        }

        if self.tts_manager:
            status["tts_services"] = self.tts_manager.get_status_all()

        return status

    def run_orchestration(self):
        """Run complete orchestration lifecycle"""
        try:
            # Start orchestration
            if not self.start_orchestration():
                self.logger.error("Failed to start orchestration")
                return False

            # Run until shutdown requested
            self.logger.info("Orchestration running. Press Ctrl+C to stop.")

            while not self.shutdown_requested:
                time.sleep(1)

        except KeyboardInterrupt:
            self.logger.info("Shutdown requested via keyboard interrupt")
        finally:
            # Stop orchestration
            self.stop_orchestration()

        return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="TTS Service Orchestrator")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--start", action="store_true", help="Start orchestration")
    parser.add_argument("--stop", action="store_true", help="Stop orchestration")
    parser.add_argument("--status", action="store_true", help="Show orchestration status")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")

    args = parser.parse_args()

    # Create orchestrator
    orchestrator = ServiceOrchestrator(args.config)

    try:
        if args.start or args.daemon:
            if args.daemon:
                success = orchestrator.run_orchestration()
            else:
                success = orchestrator.start_orchestration()
            sys.exit(0 if success else 1)

        elif args.stop:
            success = orchestrator.stop_orchestration()
            sys.exit(0 if success else 1)

        elif args.status:
            status = orchestrator.get_orchestration_status()
            print(json.dumps(status, indent=2))

        else:
            # Interactive mode
            orchestrator.run_orchestration()

    except KeyboardInterrupt:
        print("Shutting down...")
        orchestrator.stop_orchestration()

if __name__ == "__main__":
    main()
