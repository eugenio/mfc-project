"""
Service Orchestrator Module

Provides comprehensive service orchestration and deployment coordination for the
Q-Learning MFC project. This module integrates process management and logging
systems to provide unified deployment control.

The ServiceOrchestrator coordinates multiple services, manages dependencies,
handles graceful startup/shutdown, and provides monitoring capabilities.

Created: 2025-08-04
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Callable, Any, Union
from dataclasses import dataclass, field
import json
import signal
import threading
from collections import defaultdict
import os
import sys

# Try to import process_manager and log_management with fallback
    from .process_manager import ProcessManager, ProcessConfig, ProcessState, RestartPolicy
    from .log_management import LogManager, get_log_manager
    import importlib.util
    
    # Load process_manager
    pm_path = Path(__file__).parent / "process_manager.py"
    spec = importlib.util.spec_from_file_location("process_manager", pm_path)
    process_manager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(process_manager)
    ProcessManager = process_manager.ProcessManager
    ProcessConfig = process_manager.ProcessConfig
    ProcessState = process_manager.ProcessState
    RestartPolicy = process_manager.RestartPolicy
    
    # Load log_management
    lm_path = Path(__file__).parent / "log_management.py"
    spec = importlib.util.spec_from_file_location("log_management", lm_path)
    log_management = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(log_management)
    LogManager = log_management.LogManager
    get_log_manager = log_management.get_log_manager
class ServiceState(Enum):
    """Service states for orchestration."""
    INACTIVE = "inactive"
    STARTING = "starting"
    ACTIVE = "active"
    STOPPING = "stopping"
    FAILED = "failed"
    MAINTENANCE = "maintenance"


class DependencyType(Enum):
    """Types of service dependencies."""
    REQUIRED = "required"      # Service cannot start without this dependency
    OPTIONAL = "optional"      # Service can start but prefers this dependency
    WEAK = "weak"             # Service should start after this dependency

@dataclass
class ServiceDependency:
    """Represents a dependency relationship between services."""
    service_name: str
    dependency_type: DependencyType = DependencyType.REQUIRED
    timeout: float = 30.0
    health_check: Optional[Callable[[], bool]] = None

@dataclass
class ServiceConfig:
    """Configuration for a managed service."""
    name: str
    description: str = ""
    
    # Process configuration
    process_config: Optional[ProcessConfig] = None
    
    # Service lifecycle
    startup_command: Optional[List[str]] = None
    shutdown_command: Optional[List[str]] = None
    health_check_command: Optional[List[str]] = None
    
    # Dependencies
    dependencies: List[ServiceDependency] = field(default_factory=list)
    
    # Timing
    startup_timeout: float = 60.0
    shutdown_timeout: float = 30.0
    startup_delay: float = 0.0
    
    # Monitoring
    enable_monitoring: bool = True
    health_check_interval: float = 30.0
    
    # Auto-management
    auto_restart: bool = True
    max_restart_attempts: int = 3
    
    # Environment
    environment: Dict[str, str] = field(default_factory=dict)
    working_directory: Optional[str] = None
    
    # Logging
    log_file: Optional[str] = None
    log_level: str = "INFO"

@dataclass
class ServiceInfo:
    """Runtime information about a service."""
    config: ServiceConfig
    state: ServiceState = ServiceState.INACTIVE
    start_time: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    restart_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    pid: Optional[int] = None
    
    @property
    def uptime(self) -> Optional[timedelta]:
        """Get service uptime."""
        if self.start_time and self.state == ServiceState.ACTIVE:
            return datetime.now() - self.start_time
        return None
    
    @property
    def is_healthy(self) -> bool:
        """Check if service is in a healthy state."""
        return self.state == ServiceState.ACTIVE and self.error_count < 5

class ServiceOrchestrator:
    """
    Orchestrates multiple services with dependency management and monitoring.
    
    This class provides comprehensive service management including:
    - Dependency-aware startup sequencing
    - Health monitoring and auto-recovery
    - Graceful shutdown coordination
    - Resource monitoring and alerting
    - Configuration management
    """
    
    def __init__(self, 
                 deployment_dir: str = "/tmp/mfc-deployment",
                 log_level: str = "INFO"):
        """
        Initialize the service orchestrator.
        
        Args:
            deployment_dir: Directory for deployment files and logs
            log_level: Logging level for the orchestrator
        """
        self.deployment_dir = Path(deployment_dir)
        self.deployment_dir.mkdir(exist_ok=True, parents=True)
        
        # Core components
        self.log_manager = get_log_manager(str(self.deployment_dir / "logs"))
        self.process_manager = ProcessManager(str(self.deployment_dir / "processes"))
        
        # Service management
        self._services: Dict[str, ServiceInfo] = {}
        self._dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self._reverse_deps: Dict[str, Set[str]] = defaultdict(set)
        
        # State management
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        self._monitoring_thread: Optional[threading.Thread] = None
        self._is_monitoring = False
        
        # Configuration
        self.startup_parallelism = 3  # Max parallel service starts
        self.global_timeout = 300.0   # Global operation timeout
        
        # Callbacks
        self.on_service_start: Optional[Callable[[ServiceInfo], None]] = None
        self.on_service_stop: Optional[Callable[[ServiceInfo], None]] = None
        self.on_service_fail: Optional[Callable[[ServiceInfo], None]] = None
        self.on_dependency_failure: Optional[Callable[[str, str], None]] = None
        
        # Setup logging
        self.logger = self.log_manager.get_logger(
            "service_orchestrator", 
            level=getattr(logging, log_level.upper())
        )
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        self.logger.info(f"ServiceOrchestrator initialized in {deployment_dir}")
    
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown")
            self.shutdown_all()
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def add_service(self, config: ServiceConfig) -> ServiceInfo:
        """
        Add a service to the orchestrator.
        
        Args:
            config: Service configuration
            
        Returns:
            ServiceInfo: Created service information
            
        Raises:
            ValueError: If service already exists or has invalid configuration
        """
        with self._lock:
            if config.name in self._services:
                raise ValueError(f"Service {config.name} already exists")
            
            # Validate configuration
            self._validate_service_config(config)
            
            # Create service info
            service_info = ServiceInfo(config=config)
            self._services[config.name] = service_info
            
            # Build dependency graph
            self._update_dependency_graph(config)
            
            # Create process config if needed
            if config.process_config:
                self.process_manager.add_process(config.process_config)
            
            self.logger.info(f"Added service: {config.name}")
            return service_info
    
    def _validate_service_config(self, config: ServiceConfig) -> None:
        """Validate service configuration."""
        if not config.name:
            raise ValueError("Service name cannot be empty")
        
        if not config.startup_command and not config.process_config:
            raise ValueError("Service must have either startup_command or process_config")
        
        # Validate dependencies exist (will exist when they're added)
        for dep in config.dependencies:
            if dep.service_name == config.name:
                raise ValueError("Service cannot depend on itself")
    
    def _update_dependency_graph(self, config: ServiceConfig) -> None:
        """Update the dependency graph for a service."""
        service_name = config.name
        
        # Clear existing dependencies for this service
        self._dependency_graph[service_name] = set()
        
        # Add new dependencies
        for dep in config.dependencies:
            self._dependency_graph[service_name].add(dep.service_name)
            self._reverse_deps[dep.service_name].add(service_name)
        
        # Check for circular dependencies
        if self._has_circular_dependency():
            # Restore state and raise error
            for dep in config.dependencies:
                self._dependency_graph[service_name].discard(dep.service_name)
                self._reverse_deps[dep.service_name].discard(service_name)
            raise ValueError(f"Adding service {service_name} would create circular dependency")
    
    def _has_circular_dependency(self) -> bool:
        """Check if the dependency graph has circular dependencies."""
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self._dependency_graph.get(node, set()):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for service in self._services:
            if service not in visited:
                if has_cycle(service):
                    return True
        return False
    
    def remove_service(self, service_name: str) -> bool:
        """
        Remove a service from the orchestrator.
        
        Args:
            service_name: Name of service to remove
            
        Returns:
            bool: True if service was removed, False if not found
        """
        with self._lock:
            if service_name not in self._services:
                return False
            
            service_info = self._services[service_name]
            
            # Stop service if running
            if service_info.state in [ServiceState.ACTIVE, ServiceState.STARTING]:
                self.stop_service(service_name)
            
            # Remove from process manager
            if service_info.config.process_config:
                self.process_manager.remove_process(service_name)
            
            # Clean up dependency graph
            del self._services[service_name]
            del self._dependency_graph[service_name]
            
            # Remove reverse dependencies
            for deps in self._reverse_deps.values():
                deps.discard(service_name)
            del self._reverse_deps[service_name]
            
            self.logger.info(f"Removed service: {service_name}")
            return True
    
    def start_service(self, service_name: str) -> bool:
        """
        Start a specific service and its dependencies.
        
        Args:
            service_name: Name of service to start
            
        Returns:
            bool: True if service started successfully
        """
        with self._lock:
            if service_name not in self._services:
                self.logger.error(f"Service {service_name} not found")
                return False
            
            service_info = self._services[service_name]
            
            if service_info.state == ServiceState.ACTIVE:
                self.logger.info(f"Service {service_name} already active")
                return True
            
            return self._start_service_internal(service_info)
    
    def _start_service_internal(self, service_info: ServiceInfo) -> bool:
        """Internal service start implementation."""
        service_name = service_info.config.name
        
        try:
            # Set starting state
            service_info.state = ServiceState.STARTING
            service_info.last_error = None
            
            self.logger.info(f"Starting service: {service_name}")
            
            # Start dependencies first
            if not self._start_dependencies(service_info):
                service_info.state = ServiceState.FAILED
                service_info.last_error = "Dependency start failed"
                return False
            
            # Wait for startup delay
            if service_info.config.startup_delay > 0:
                time.sleep(service_info.config.startup_delay)
            
            # Start the service
            success = self._execute_service_startup(service_info)
            
            if success:
                service_info.state = ServiceState.ACTIVE
                service_info.start_time = datetime.now()
                service_info.restart_count = 0
                
                if self.on_service_start:
                    self.on_service_start(service_info)
                
                self.logger.info(f"Service {service_name} started successfully")
                return True
            else:
                service_info.state = ServiceState.FAILED
                service_info.error_count += 1
                
                if self.on_service_fail:
                    self.on_service_fail(service_info)
                
                return False
                
        except Exception as e:
            service_info.state = ServiceState.FAILED
            service_info.last_error = str(e)
            service_info.error_count += 1
            
            self.logger.error(f"Failed to start service {service_name}: {e}")
            
            if self.on_service_fail:
                self.on_service_fail(service_info)
            
            return False
    
    def _start_dependencies(self, service_info: ServiceInfo) -> bool:
        """Start all dependencies for a service."""
        for dep in service_info.config.dependencies:
            if dep.dependency_type == DependencyType.OPTIONAL:
                continue  # Skip optional dependencies
            
            if dep.service_name not in self._services:
                if dep.dependency_type == DependencyType.REQUIRED:
                    self.logger.error(f"Required dependency {dep.service_name} not found")
                    return False
                continue
            
            dep_info = self._services[dep.service_name]
            
            if dep_info.state != ServiceState.ACTIVE:
                if not self._start_service_internal(dep_info):
                    if dep.dependency_type == DependencyType.REQUIRED:
                        self.logger.error(f"Failed to start required dependency {dep.service_name}")
                        if self.on_dependency_failure:
                            self.on_dependency_failure(service_info.config.name, dep.service_name)
                        return False
            
            # Wait for dependency to be healthy
            if not self._wait_for_dependency_health(dep):
                if dep.dependency_type == DependencyType.REQUIRED:
                    return False
        
        return True
    
    def _wait_for_dependency_health(self, dependency: ServiceDependency) -> bool:
        """Wait for a dependency to become healthy."""
        start_time = time.time()
        
        while time.time() - start_time < dependency.timeout:
            if dependency.service_name in self._services:
                dep_info = self._services[dependency.service_name]
                
                if dep_info.state == ServiceState.ACTIVE:
                    if dependency.health_check:
                        if dependency.health_check():
                            return True
                    else:
                        return True
            
            time.sleep(1.0)
        
        return False
    
    def _execute_service_startup(self, service_info: ServiceInfo) -> bool:
        """Execute the actual service startup."""
        config = service_info.config
        
        # Use process manager if process config is available
        if config.process_config:
            success = self.process_manager.start_process(config.name)
            if success:
                proc_info = self.process_manager.get_process_info(config.name)
                if proc_info:
                    service_info.pid = proc_info.pid
            return success
        
        # Use startup command
        elif config.startup_command:
            try:
                import subprocess
                
                env = os.environ.copy()
                env.update(config.environment)
                
                process = subprocess.Popen(
                    config.startup_command,
                    cwd=config.working_directory,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                service_info.pid = process.pid
                
                # Wait for startup timeout
                try:
                    process.wait(timeout=config.startup_timeout)
                    return process.returncode == 0
                except subprocess.TimeoutExpired:
                    process.kill()
                    service_info.last_error = "Startup timeout"
                    return False
                    
            except Exception as e:
                service_info.last_error = str(e)
                return False
        
        return False
    
    def stop_service(self, service_name: str, timeout: Optional[float] = None) -> bool:
        """
        Stop a specific service and its dependents.
        
        Args:
            service_name: Name of service to stop
            timeout: Shutdown timeout override
            
        Returns:
            bool: True if service stopped successfully
        """
        with self._lock:
            if service_name not in self._services:
                return False
            
            service_info = self._services[service_name]
            
            if service_info.state not in [ServiceState.ACTIVE, ServiceState.STARTING]:
                return True
            
            return self._stop_service_internal(service_info, timeout)
    
    def _stop_service_internal(self, service_info: ServiceInfo, timeout: Optional[float] = None) -> bool:
        """Internal service stop implementation."""
        service_name = service_info.config.name
        config = service_info.config
        
        if timeout is None:
            timeout = config.shutdown_timeout
        
        try:
            service_info.state = ServiceState.STOPPING
            
            self.logger.info(f"Stopping service: {service_name}")
            
            # Stop dependents first
            self._stop_dependents(service_info)
            
            # Execute shutdown
            success = self._execute_service_shutdown(service_info, timeout)
            
            if success:
                service_info.state = ServiceState.INACTIVE
                service_info.pid = None
                service_info.start_time = None
                
                if self.on_service_stop:
                    self.on_service_stop(service_info)
                
                self.logger.info(f"Service {service_name} stopped successfully")
                return True
            else:
                service_info.state = ServiceState.FAILED
                service_info.error_count += 1
                return False
                
        except Exception as e:
            service_info.state = ServiceState.FAILED
            service_info.last_error = str(e)
            self.logger.error(f"Failed to stop service {service_name}: {e}")
            return False
    
    def _stop_dependents(self, service_info: ServiceInfo) -> None:
        """Stop all services that depend on this service."""
        service_name = service_info.config.name
        
        for dependent_name in self._reverse_deps.get(service_name, set()):
            if dependent_name in self._services:
                dependent_info = self._services[dependent_name]
                if dependent_info.state == ServiceState.ACTIVE:
                    self._stop_service_internal(dependent_info)
    
    def _execute_service_shutdown(self, service_info: ServiceInfo, timeout: float) -> bool:
        """Execute the actual service shutdown."""
        config = service_info.config
        
        # Use process manager if available
        if config.process_config:
            return self.process_manager.stop_process(config.name, timeout)
        
        # Use shutdown command
        elif config.shutdown_command:
            try:
                import subprocess
                
                env = os.environ.copy()
                env.update(config.environment)
                
                result = subprocess.run(
                    config.shutdown_command,
                    cwd=config.working_directory,
                    env=env,
                    timeout=timeout,
                    capture_output=True,
                    text=True
                )
                
                return result.returncode == 0
                
            except subprocess.TimeoutExpired:
                service_info.last_error = "Shutdown timeout"
                return False
            except Exception as e:
                service_info.last_error = str(e)
                return False
        
        # Fallback to process termination
        elif service_info.pid:
            try:
                import os
                import signal
                
                os.kill(service_info.pid, signal.SIGTERM)
                
                # Wait for graceful shutdown
                start_time = time.time()
                while time.time() - start_time < timeout:
                    try:
                        os.kill(service_info.pid, 0)  # Check if process exists
                        time.sleep(0.1)
                    except OSError:
                        return True  # Process is gone
                
                # Force kill
                os.kill(service_info.pid, signal.SIGKILL)
                return True
                
            except Exception as e:
                service_info.last_error = str(e)
                return False
        
        return True
    
    def restart_service(self, service_name: str) -> bool:
        """
        Restart a specific service.
        
        Args:
            service_name: Name of service to restart
            
        Returns:
            bool: True if service restarted successfully
        """
        with self._lock:
            if service_name not in self._services:
                return False
            
            service_info = self._services[service_name]
            
            # Stop the service
            if not self.stop_service(service_name):
                return False
            
            # Increment restart count
            service_info.restart_count += 1
            
            # Start the service
            return self.start_service(service_name)
    
    def start_all(self) -> Dict[str, bool]:
        """
        Start all services in dependency order.
        
        Returns:
            Dict[str, bool]: Results for each service
        """
        results = {}
        
        # Get startup order based on dependencies
        startup_order = self._get_startup_order()
        
        for service_name in startup_order:
            results[service_name] = self.start_service(service_name)
        
        return results
    
    def stop_all(self) -> Dict[str, bool]:
        """
        Stop all services in reverse dependency order.
        
        Returns:
            Dict[str, bool]: Results for each service
        """
        results = {}
        
        # Get shutdown order (reverse of startup)
        shutdown_order = list(reversed(self._get_startup_order()))
        
        for service_name in shutdown_order:
            results[service_name] = self.stop_service(service_name)
        
        return results
    
    def _get_startup_order(self) -> List[str]:
        """Get the order in which services should be started based on dependencies."""
        # Topological sort
        in_degree = {service: 0 for service in self._services}
        
        # Calculate in-degrees
        for service in self._services:
            for dep_service in self._dependency_graph.get(service, set()):
                if dep_service in in_degree:
                    in_degree[service] += 1
        
        # Find services with no dependencies
        queue = [service for service, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            service = queue.pop(0)
            result.append(service)
            
            # Reduce in-degree for dependents
            for dependent in self._reverse_deps.get(service, set()):
                if dependent in in_degree:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)
        
        return result
    
    def get_service_info(self, service_name: str) -> Optional[ServiceInfo]:
        """Get information about a specific service."""
        return self._services.get(service_name)
    
    def list_services(self) -> Dict[str, ServiceInfo]:
        """Get information about all services."""
        return self._services.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall orchestrator status."""
        with self._lock:
            status_counts = defaultdict(int)
            service_statuses = {}
            
            for name, info in self._services.items():
                status_counts[info.state.value] += 1
                service_statuses[name] = {
                    "state": info.state.value,
                    "uptime": str(info.uptime) if info.uptime else None,
                    "restart_count": info.restart_count,
                    "error_count": info.error_count,
                    "is_healthy": info.is_healthy,
                    "last_error": info.last_error,
                    "pid": info.pid
                }
            
            return {
                "total_services": len(self._services),
                "status_counts": dict(status_counts),
                "services": service_statuses,
                "is_monitoring": self._is_monitoring,
                "deployment_dir": str(self.deployment_dir)
            }
    
    def start_monitoring(self) -> None:
        """Start the monitoring thread."""
        if self._is_monitoring:
            return
        
        self._is_monitoring = True
        self._shutdown_event.clear()
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        self.logger.info("Started service monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring thread."""
        if not self._is_monitoring:
            return
        
        self._is_monitoring = False
        self._shutdown_event.set()
        
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Stopped service monitoring")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._is_monitoring and not self._shutdown_event.is_set():
            try:
                self._check_service_health()
                self._shutdown_event.wait(10.0)  # Check every 10 seconds
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)
    
    def _check_service_health(self) -> None:
        """Check health of all active services."""
        with self._lock:
            for service_info in self._services.values():
                if service_info.state == ServiceState.ACTIVE:
                    self._check_individual_service_health(service_info)
    
    def _check_individual_service_health(self, service_info: ServiceInfo) -> None:
        """Check health of an individual service."""
        config = service_info.config
        
        if not config.enable_monitoring:
            return
        
        # Check if it's time for a health check
        now = datetime.now()
        if (service_info.last_health_check and 
            (now - service_info.last_health_check).total_seconds() < config.health_check_interval):
            return
        
        service_info.last_health_check = now
        
        # Perform health check
        is_healthy = self._perform_health_check(service_info)
        
        if not is_healthy:
            service_info.error_count += 1
            self.logger.warning(f"Health check failed for service {config.name}")
            
            # Auto-restart if configured
            if config.auto_restart and service_info.restart_count < config.max_restart_attempts:
                self.logger.info(f"Auto-restarting service {config.name}")
                self.restart_service(config.name)
        else:
            # Reset error count on successful health check
            if service_info.error_count > 0:
                service_info.error_count = max(0, service_info.error_count - 1)
    
    def _perform_health_check(self, service_info: ServiceInfo) -> bool:
        """Perform a health check on a service."""
        config = service_info.config
        
        # Check if process is still running
        if config.process_config:
            proc_info = self.process_manager.get_process_info(config.name)
            if not proc_info or proc_info.state != ProcessState.RUNNING:
                return False
        
        # Run custom health check command
        if config.health_check_command:
            try:
                import subprocess
                
                result = subprocess.run(
                    config.health_check_command,
                    timeout=config.health_check_interval / 2,
                    capture_output=True,
                    text=True
                )
                
                return result.returncode == 0
                
            except Exception:
                return False
        
        return True
    
    def shutdown_all(self) -> None:
        """Shutdown all services and the orchestrator."""
        self.logger.info("Initiating orchestrator shutdown")
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Stop all services
        self.stop_all()
        
        # Shutdown process manager
        self.process_manager.shutdown_all()
        
        self.logger.info("Orchestrator shutdown complete")
    
    def save_configuration(self, config_file: str) -> None:
        """
        Save the current service configuration to a file.
        
        Args:
            config_file: Path to configuration file
        """
        config_data = {
            "services": [],
            "orchestrator_config": {
                "deployment_dir": str(self.deployment_dir),
                "startup_parallelism": self.startup_parallelism,
                "global_timeout": self.global_timeout
            }
        }
        
        for service_info in self._services.values():
            config = service_info.config
            service_data = {
                "name": config.name,
                "description": config.description,
                "startup_command": config.startup_command,
                "shutdown_command": config.shutdown_command,
                "health_check_command": config.health_check_command,
                "dependencies": [
                    {
                        "service_name": dep.service_name,
                        "dependency_type": dep.dependency_type.value,
                        "timeout": dep.timeout
                    }
                    for dep in config.dependencies
                ],
                "startup_timeout": config.startup_timeout,
                "shutdown_timeout": config.shutdown_timeout,
                "startup_delay": config.startup_delay,
                "enable_monitoring": config.enable_monitoring,
                "health_check_interval": config.health_check_interval,
                "auto_restart": config.auto_restart,
                "max_restart_attempts": config.max_restart_attempts,
                "environment": config.environment,
                "working_directory": config.working_directory,
                "log_file": config.log_file,
                "log_level": config.log_level
            }
            
            # Add process config if present
            if config.process_config:
                proc_config = config.process_config
                service_data["process_config"] = {
                    "command": proc_config.command,
                    "working_dir": proc_config.working_dir,
                    "environment": proc_config.environment,
                    "restart_policy": proc_config.restart_policy.value,
                    "max_restarts": proc_config.max_restarts,
                    "restart_delay": proc_config.restart_delay,
                    "memory_limit": proc_config.memory_limit,
                    "cpu_limit": proc_config.cpu_limit
                }
            
            config_data["services"].append(service_data)
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        self.logger.info(f"Configuration saved to {config_file}")
    
    def load_configuration(self, config_file: str) -> None:
        """
        Load service configuration from a file.
        
        Args:
            config_file: Path to configuration file
        """
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        # Load orchestrator config
        if "orchestrator_config" in config_data:
            orc_config = config_data["orchestrator_config"]
            self.startup_parallelism = orc_config.get("startup_parallelism", self.startup_parallelism)
            self.global_timeout = orc_config.get("global_timeout", self.global_timeout)
        
        # Load services
        for service_data in config_data.get("services", []):
            # Create dependencies
            dependencies = []
            for dep_data in service_data.get("dependencies", []):
                dep = ServiceDependency(
                    service_name=dep_data["service_name"],
                    dependency_type=DependencyType(dep_data["dependency_type"]),
                    timeout=dep_data.get("timeout", 30.0)
                )
                dependencies.append(dep)
            
            # Create process config if present
            process_config = None
            if "process_config" in service_data:
                proc_data = service_data["process_config"]
                process_config = ProcessConfig(
                    name=service_data["name"],
                    command=proc_data["command"],
                    working_dir=proc_data.get("working_dir"),
                    environment=proc_data.get("environment", {}),
                    restart_policy=RestartPolicy(proc_data.get("restart_policy", "on_failure")),
                    max_restarts=proc_data.get("max_restarts", 5),
                    restart_delay=proc_data.get("restart_delay", 1.0),
                    memory_limit=proc_data.get("memory_limit"),
                    cpu_limit=proc_data.get("cpu_limit")
                )
            
            # Create service config
            config = ServiceConfig(
                name=service_data["name"],
                description=service_data.get("description", ""),
                process_config=process_config,
                startup_command=service_data.get("startup_command"),
                shutdown_command=service_data.get("shutdown_command"),
                health_check_command=service_data.get("health_check_command"),
                dependencies=dependencies,
                startup_timeout=service_data.get("startup_timeout", 60.0),
                shutdown_timeout=service_data.get("shutdown_timeout", 30.0),
                startup_delay=service_data.get("startup_delay", 0.0),
                enable_monitoring=service_data.get("enable_monitoring", True),
                health_check_interval=service_data.get("health_check_interval", 30.0),
                auto_restart=service_data.get("auto_restart", True),
                max_restart_attempts=service_data.get("max_restart_attempts", 3),
                environment=service_data.get("environment", {}),
                working_directory=service_data.get("working_directory"),
                log_file=service_data.get("log_file"),
                log_level=service_data.get("log_level", "INFO")
            )
            
            self.add_service(config)
        
        self.logger.info(f"Configuration loaded from {config_file}")


# Global instance management
def get_service_orchestrator(deployment_dir: str = "/tmp/mfc-deployment") -> ServiceOrchestrator:
    """
    Get the global service orchestrator instance.
    
    Args:
        deployment_dir: Deployment directory for first-time initialization
        
    Returns:
        ServiceOrchestrator: Global instance
    """
    global _service_orchestrator
    
    if _service_orchestrator is None:
        _service_orchestrator = ServiceOrchestrator(deployment_dir=deployment_dir)
    
    return _service_orchestrator

def main():
    """Command-line interface for the service orchestrator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MFC Service Orchestrator")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--deployment-dir", default="/tmp/mfc-deployment", 
                       help="Deployment directory")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    # Actions
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument("--start-all", action="store_true", 
                             help="Start all services")
    action_group.add_argument("--stop-all", action="store_true", 
                             help="Stop all services")
    action_group.add_argument("--status", action="store_true", 
                             help="Show status")
    action_group.add_argument("--start", help="Start specific service")
    action_group.add_argument("--stop", help="Stop specific service")
    action_group.add_argument("--restart", help="Restart specific service")
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = ServiceOrchestrator(
        deployment_dir=args.deployment_dir,
        log_level=args.log_level
    )
    
    # Load configuration if provided
    if args.config:
        orchestrator.load_configuration(args.config)
    
    try:
        # Execute actions
        if args.start_all:
            results = orchestrator.start_all()
            print(f"Start results: {results}")
        
        elif args.stop_all:
            results = orchestrator.stop_all()
            print(f"Stop results: {results}")
        
        elif args.status:
            status = orchestrator.get_status()
            print(json.dumps(status, indent=2))
        
        elif args.start:
            result = orchestrator.start_service(args.start)
            print(f"Start {args.start}: {result}")
        
        elif args.stop:
            result = orchestrator.stop_service(args.stop)
            print(f"Stop {args.stop}: {result}")
        
        elif args.restart:
            result = orchestrator.restart_service(args.restart)
            print(f"Restart {args.restart}: {result}")
        
        else:
            # Interactive mode
            orchestrator.start_monitoring()
            print("Service orchestrator running. Press Ctrl+C to exit.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
    
    finally:
        orchestrator.shutdown_all()

if __name__ == "__main__":
    main()