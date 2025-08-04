#!/usr/bin/env python3
"""
Service Discovery - TTS Service Registration and Discovery
Agent Zeta - Deployment and Process Management

Handles service registration, discovery, and health monitoring
"""

import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path


class ServiceState(Enum):
    """Service states"""
    REGISTERED = "registered"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEREGISTERED = "deregistered"

@dataclass
class ServiceEndpoint:
    """Service endpoint information"""
    type: str  # management, api, websocket, etc.
    url: str
    port: int
    protocol: str = "http"
    health_check_path: str | None = None

@dataclass
class ServiceInfo:
    """Complete service information"""
    service_id: str
    service_name: str
    service_type: str
    version: str
    host: str
    endpoints: list[ServiceEndpoint]
    metadata: dict = None
    state: ServiceState = ServiceState.REGISTERED
    last_heartbeat: datetime | None = None
    registration_time: datetime | None = None
    capabilities: list[str] = None
    dependencies: list[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.capabilities is None:
            self.capabilities = []
        if self.dependencies is None:
            self.dependencies = []
        if self.registration_time is None:
            self.registration_time = datetime.now()

class ServiceRegistry:
    """In-memory service registry with persistence"""

    def __init__(self, registry_file: str | None = None):
        self.services: dict[str, ServiceInfo] = {}
        self.registry_file = registry_file or "/tmp/tts-service-registry.json"
        self.logger = logging.getLogger("service_registry")
        self.lock = threading.RLock()
        self.subscribers: dict[str, list[Callable]] = {}

        # Load existing registry
        self.load_registry()

    def register_service(self, service_info: ServiceInfo) -> bool:
        """Register a service"""
        with self.lock:
            try:
                # Generate unique ID if not provided
                if not service_info.service_id:
                    service_info.service_id = self._generate_service_id(service_info)

                service_info.registration_time = datetime.now()
                service_info.last_heartbeat = datetime.now()
                service_info.state = ServiceState.REGISTERED

                self.services[service_info.service_id] = service_info
                self.save_registry()

                self.logger.info(f"Registered service: {service_info.service_name} ({service_info.service_id})")
                self._notify_subscribers("service_registered", service_info)

                return True

            except Exception as e:
                self.logger.error(f"Failed to register service {service_info.service_name}: {e}")
                return False

    def deregister_service(self, service_id: str) -> bool:
        """Deregister a service"""
        with self.lock:
            try:
                if service_id in self.services:
                    service_info = self.services[service_id]
                    service_info.state = ServiceState.DEREGISTERED

                    del self.services[service_id]
                    self.save_registry()

                    self.logger.info(f"Deregistered service: {service_info.service_name} ({service_id})")
                    self._notify_subscribers("service_deregistered", service_info)

                    return True
                else:
                    self.logger.warning(f"Service not found for deregistration: {service_id}")
                    return False

            except Exception as e:
                self.logger.error(f"Failed to deregister service {service_id}: {e}")
                return False

    def update_heartbeat(self, service_id: str) -> bool:
        """Update service heartbeat"""
        with self.lock:
            if service_id in self.services:
                self.services[service_id].last_heartbeat = datetime.now()
                if self.services[service_id].state == ServiceState.UNHEALTHY:
                    self.services[service_id].state = ServiceState.HEALTHY
                    self._notify_subscribers("service_healthy", self.services[service_id])
                return True
            return False

    def mark_unhealthy(self, service_id: str) -> bool:
        """Mark service as unhealthy"""
        with self.lock:
            if service_id in self.services:
                if self.services[service_id].state != ServiceState.UNHEALTHY:
                    self.services[service_id].state = ServiceState.UNHEALTHY
                    self.logger.warning(f"Service marked unhealthy: {service_id}")
                    self._notify_subscribers("service_unhealthy", self.services[service_id])
                return True
            return False

    def get_service(self, service_id: str) -> ServiceInfo | None:
        """Get service by ID"""
        with self.lock:
            return self.services.get(service_id)

    def find_services(self, service_name: str | None = None,
                     service_type: str | None = None,
                     state: ServiceState | None = None) -> list[ServiceInfo]:
        """Find services by criteria"""
        with self.lock:
            results = []
            for service in self.services.values():
                if service_name and service.service_name != service_name:
                    continue
                if service_type and service.service_type != service_type:
                    continue
                if state and service.state != state:
                    continue
                results.append(service)
            return results

    def get_all_services(self) -> dict[str, ServiceInfo]:
        """Get all registered services"""
        with self.lock:
            return self.services.copy()

    def cleanup_stale_services(self, max_age: timedelta = timedelta(minutes=5)) -> int:
        """Clean up services that haven't sent heartbeats"""
        with self.lock:
            stale_services = []
            current_time = datetime.now()

            for service_id, service in self.services.items():
                if service.last_heartbeat:
                    age = current_time - service.last_heartbeat
                    if age > max_age:
                        stale_services.append(service_id)

            # Remove stale services
            for service_id in stale_services:
                service = self.services[service_id]
                self.logger.warning(f"Removing stale service: {service.service_name} ({service_id})")
                del self.services[service_id]
                self._notify_subscribers("service_stale", service)

            if stale_services:
                self.save_registry()

            return len(stale_services)

    def save_registry(self):
        """Save registry to disk"""
        try:
            registry_data = {
                "timestamp": datetime.now().isoformat(),
                "services": {}
            }

            for service_id, service in self.services.items():
                service_dict = asdict(service)
                # Convert datetime objects to ISO format
                if service_dict.get("last_heartbeat"):
                    service_dict["last_heartbeat"] = service.last_heartbeat.isoformat()
                if service_dict.get("registration_time"):
                    service_dict["registration_time"] = service.registration_time.isoformat()
                service_dict["state"] = service.state.value

                registry_data["services"][service_id] = service_dict

            with open(self.registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save registry: {e}")

    def load_registry(self):
        """Load registry from disk"""
        try:
            if Path(self.registry_file).exists():
                with open(self.registry_file) as f:
                    registry_data = json.load(f)

                for service_id, service_dict in registry_data.get("services", {}).items():
                    # Convert ISO format back to datetime
                    if service_dict.get("last_heartbeat"):
                        service_dict["last_heartbeat"] = datetime.fromisoformat(service_dict["last_heartbeat"])
                    if service_dict.get("registration_time"):
                        service_dict["registration_time"] = datetime.fromisoformat(service_dict["registration_time"])

                    # Convert state string to enum
                    service_dict["state"] = ServiceState(service_dict["state"])

                    # Convert endpoints
                    endpoints = []
                    for ep_dict in service_dict.get("endpoints", []):
                        endpoints.append(ServiceEndpoint(**ep_dict))
                    service_dict["endpoints"] = endpoints

                    service_info = ServiceInfo(**service_dict)
                    self.services[service_id] = service_info

                self.logger.info(f"Loaded {len(self.services)} services from registry")

        except Exception as e:
            self.logger.error(f"Failed to load registry: {e}")

    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to registry events"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)

    def _notify_subscribers(self, event_type: str, service_info: ServiceInfo):
        """Notify event subscribers"""
        for callback in self.subscribers.get(event_type, []):
            try:
                callback(service_info)
            except Exception as e:
                self.logger.error(f"Error in subscriber callback: {e}")

    def _generate_service_id(self, service_info: ServiceInfo) -> str:
        """Generate unique service ID"""
        unique_string = f"{service_info.service_name}:{service_info.host}:{service_info.endpoints[0].port}:{time.time()}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:12]

class TTSServiceDiscovery:
    """TTS service discovery client"""

    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self.logger = logging.getLogger("tts_service_discovery")
        self.heartbeat_thread: threading.Thread | None = None
        self.service_id: str | None = None
        self.shutdown_event = threading.Event()

    def register_tts_service(self,
                           service_name: str = "tts-service",
                           port: int = 9000,
                           host: str = "localhost",
                           capabilities: list[str] = None) -> str | None:
        """Register TTS service"""
        try:
            endpoints = [
                ServiceEndpoint(
                    type="management",
                    url=f"http://{host}:{port}",
                    port=port,
                    protocol="http",
                    health_check_path="/health"
                )
            ]

            if capabilities is None:
                capabilities = ["pyttsx3", "hybrid-engine"]

            service_info = ServiceInfo(
                service_id="",  # Will be generated
                service_name=service_name,
                service_type="text-to-speech",
                version="1.0.0",
                host=host,
                endpoints=endpoints,
                capabilities=capabilities,
                metadata={
                    "startup_time": datetime.now().isoformat(),
                    "environment": "production"
                }
            )

            if self.registry.register_service(service_info):
                self.service_id = service_info.service_id
                self._start_heartbeat()
                return self.service_id
            else:
                return None

        except Exception as e:
            self.logger.error(f"Failed to register TTS service: {e}")
            return None

    def deregister_tts_service(self) -> bool:
        """Deregister TTS service"""
        if self.service_id:
            self.shutdown_event.set()
            if self.heartbeat_thread:
                self.heartbeat_thread.join(timeout=5)
            return self.registry.deregister_service(self.service_id)
        return True

    def find_tts_services(self, healthy_only: bool = True) -> list[ServiceInfo]:
        """Find available TTS services"""
        services = self.registry.find_services(
            service_type="text-to-speech",
            state=ServiceState.HEALTHY if healthy_only else None
        )
        return services

    def get_tts_service_endpoint(self, service_id: str, endpoint_type: str = "management") -> ServiceEndpoint | None:
        """Get specific endpoint for a TTS service"""
        service = self.registry.get_service(service_id)
        if service:
            for endpoint in service.endpoints:
                if endpoint.type == endpoint_type:
                    return endpoint
        return None

    def _start_heartbeat(self):
        """Start heartbeat thread"""
        if self.service_id:
            self.heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                daemon=True
            )
            self.heartbeat_thread.start()

    def _heartbeat_loop(self):
        """Heartbeat loop"""
        while not self.shutdown_event.is_set():
            try:
                if self.service_id:
                    self.registry.update_heartbeat(self.service_id)

                # Wait for next heartbeat
                for _ in range(30):  # 30 second intervals
                    if self.shutdown_event.is_set():
                        break
                    time.sleep(1)

            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                time.sleep(5)

class ServiceHealthMonitor:
    """Monitors service health"""

    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self.logger = logging.getLogger("service_health_monitor")
        self.monitoring = False
        self.monitor_thread: threading.Thread | None = None
        self.shutdown_event = threading.Event()

    def start_monitoring(self, check_interval: int = 60):
        """Start health monitoring"""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(check_interval,),
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info("Health monitoring started")

    def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring = False
        self.shutdown_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        self.logger.info("Health monitoring stopped")

    def _monitor_loop(self, check_interval: int):
        """Health monitoring loop"""
        while self.monitoring and not self.shutdown_event.is_set():
            try:
                # Check for stale services
                stale_count = self.registry.cleanup_stale_services()
                if stale_count > 0:
                    self.logger.info(f"Cleaned up {stale_count} stale services")

                # Perform health checks on active services
                services = self.registry.find_services(state=ServiceState.REGISTERED)
                for service in services:
                    self._check_service_health(service)

                # Wait for next check
                for _ in range(check_interval):
                    if self.shutdown_event.is_set():
                        break
                    time.sleep(1)

            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                time.sleep(10)

    def _check_service_health(self, service: ServiceInfo):
        """Check health of a specific service"""
        try:
            # Find health check endpoint
            health_endpoint = None
            for endpoint in service.endpoints:
                if endpoint.health_check_path:
                    health_endpoint = endpoint
                    break

            if not health_endpoint:
                # No health check endpoint, just update heartbeat if recent
                if service.last_heartbeat:
                    age = datetime.now() - service.last_heartbeat
                    if age < timedelta(minutes=2):
                        self.registry.update_heartbeat(service.service_id)
                return

            # Perform HTTP health check
            import requests

            health_url = f"{health_endpoint.url}{health_endpoint.health_check_path}"
            response = requests.get(health_url, timeout=5)

            if response.status_code == 200:
                self.registry.update_heartbeat(service.service_id)
            else:
                self.registry.mark_unhealthy(service.service_id)

        except Exception as e:
            self.logger.warning(f"Health check failed for {service.service_name}: {e}")
            self.registry.mark_unhealthy(service.service_id)

def main():
    """Main entry point for testing"""
    import argparse

    parser = argparse.ArgumentParser(description="TTS Service Discovery")
    parser.add_argument("--register", action="store_true", help="Register a test service")
    parser.add_argument("--list", action="store_true", help="List all services")
    parser.add_argument("--monitor", action="store_true", help="Start health monitoring")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup stale services")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create registry
    registry = ServiceRegistry()
    discovery = TTSServiceDiscovery(registry)
    monitor = ServiceHealthMonitor(registry)

    try:
        if args.register:
            service_id = discovery.register_tts_service(
                service_name="test-tts-service",
                capabilities=["pyttsx3", "coqui-tts", "hybrid"]
            )
            print(f"Registered service: {service_id}")

        if args.list:
            services = registry.get_all_services()
            print(f"Found {len(services)} services:")
            for service_id, service in services.items():
                print(f"  {service.service_name} ({service_id}) - {service.state.value}")
                for endpoint in service.endpoints:
                    print(f"    {endpoint.type}: {endpoint.url}")

        if args.cleanup:
            cleaned = registry.cleanup_stale_services()
            print(f"Cleaned up {cleaned} stale services")

        if args.monitor:
            monitor.start_monitoring()
            print("Health monitoring started. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            finally:
                monitor.stop_monitoring()
                discovery.deregister_tts_service()

    except KeyboardInterrupt:
        print("Shutting down...")
        discovery.deregister_tts_service()
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()
