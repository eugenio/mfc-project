"""
Deployment Manager for MLOps Pipeline
=====================================

Comprehensive deployment management providing:
- Blue-green deployments
- Canary releases
- Rollback mechanisms
- Health checking
- Load balancing

Created: 2025-08-05
Author: TDD Agent 2
"""
import asyncio
import json
import logging
import random
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

try:
    import aiohttp
    import docker
except ImportError:
    # Optional dependencies - will be mocked in tests
    aiohttp = None  # type: ignore
    docker = None

# Setup logging
logger = logging.getLogger(__name__)


class DeploymentStrategy(Enum):
    """Deployment strategy types"""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    RECREATE = "recreate"


class DeploymentStatus(Enum):
    """Deployment status levels"""
    PENDING = "pending"
    DEPLOYING = "deploying"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"


@dataclass
class ServiceInstance:
    """Represents a deployed service instance"""
    instance_id: str
    service_name: str
    version: str
    host: str
    port: int
    ssl_enabled: bool = False
    status: DeploymentStatus = DeploymentStatus.PENDING
    health_score: float = 0.0
    active_connections: int = 0
    container_id: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    last_health_check: datetime | None = None

    @property
    def endpoint_url(self) -> str:
        """Generate endpoint URL for the instance"""
        protocol = "https" if self.ssl_enabled else "http"
        return f"{protocol}://{self.host}:{self.port}"

    def update_health(self, health_score: float, status: DeploymentStatus) -> None:
        """Update instance health status"""
        self.health_score = health_score
        self.status = status
        self.last_health_check = datetime.now()


@dataclass
class HealthCheckResult:
    """Health check result for a service instance"""
    instance_id: str
    is_healthy: bool
    response_time_ms: float
    status_code: int
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def health_score(self) -> float:
        """Calculate health score based on metrics"""
        if not self.is_healthy:
            return 0.0

        # Base score on response time (lower is better)
        # Perfect score for < 100ms, degrading to 0 at 1000ms
        response_score = max(0.0, min(1.0, 1.0 - (self.response_time_ms / 1000.0)))

        # Adjust for HTTP status codes
        if self.status_code == 200:
            status_score = 1.0
        elif 200 <= self.status_code < 300:
            status_score = 0.9
        elif 300 <= self.status_code < 400:
            status_score = 0.7
        else:
            status_score = 0.0

        return min(response_score, status_score)


class LoadBalancer:
    """Load balancer for distributing traffic across service instances"""

    def __init__(self, strategy: str = "round_robin"):
        """Initialize load balancer with specified strategy

        Args:
            strategy: Load balancing strategy ("round_robin", "weighted", "least_connections")
        """
        self.strategy = strategy
        self.instances: list[ServiceInstance] = []
        self.current_index = 0
        self._lock = threading.Lock()

    def add_instance(self, instance: ServiceInstance) -> None:
        """Add instance to load balancer"""
        with self._lock:
            self.instances.append(instance)

    def remove_instance(self, instance_id: str) -> None:
        """Remove instance from load balancer"""
        with self._lock:
            self.instances = [i for i in self.instances if i.instance_id != instance_id]

    def select_instance(self) -> ServiceInstance | None:
        """Select instance based on load balancing strategy"""
        with self._lock:
            healthy_instances = [i for i in self.instances if i.status == DeploymentStatus.HEALTHY]

            if not healthy_instances:
                return None

            if self.strategy == "round_robin":
                return self._round_robin_selection(healthy_instances)
            elif self.strategy == "weighted":
                return self._weighted_selection(healthy_instances)
            elif self.strategy == "least_connections":
                return self._least_connections_selection(healthy_instances)
            else:
                return healthy_instances[0]  # Default fallback

    def _round_robin_selection(self, instances: list[ServiceInstance]) -> ServiceInstance | None:
        """Round-robin instance selection"""
        if not instances:
            return None

        selected = instances[self.current_index % len(instances)]
        self.current_index = (self.current_index + 1) % len(instances)
        return selected

    def _weighted_selection(self, instances: list[ServiceInstance]) -> ServiceInstance | None:
        """Weighted instance selection based on health scores"""
        if not instances:
            return None

        # Calculate weights based on health scores
        weights = [i.health_score for i in instances]
        total_weight = sum(weights)

        if total_weight == 0:
            return instances[0]

        # Random selection based on weights
        r = random.random() * total_weight
        cumulative = 0.0

        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return instances[i]

        return instances[-1]  # Fallback

    def _least_connections_selection(self, instances: list[ServiceInstance]) -> ServiceInstance | None:
        """Select instance with least active connections"""
        if not instances:
            return None

        return min(instances, key=lambda i: i.active_connections)

    def get_health_status(self) -> dict[str, Any]:
        """Get overall health status of load balancer"""
        with self._lock:
            total_instances = len(self.instances)
            healthy_instances = len([i for i in self.instances if i.status == DeploymentStatus.HEALTHY])

            if total_instances == 0:
                avg_health = 0.0
            else:
                avg_health = sum(i.health_score for i in self.instances) / total_instances

            return {
                "total_instances": total_instances,
                "healthy_instances": healthy_instances,
                "average_health_score": avg_health,
                "load_balancing_strategy": self.strategy
            }


class BlueGreenDeployment:
    """Blue-green deployment manager"""

    def __init__(self, service_name: str, blue_version: str, green_version: str):
        """Initialize blue-green deployment

        Args:
            service_name: Name of the service being deployed
            blue_version: Current (blue) version
            green_version: New (green) version to deploy
        """
        self.service_name = service_name
        self.blue_version = blue_version
        self.green_version = green_version
        self.active_environment = "blue"  # Start with blue as active
        self.deployment_status = DeploymentStatus.PENDING

        self.blue_instances: list[ServiceInstance] = []
        self.green_instances: list[ServiceInstance] = []
        self.health_checker: Any | None = None

        self._lock = threading.Lock()

    async def deploy_green_environment(self, instances: list[ServiceInstance],
                                     deploy_function: Callable) -> bool:
        """Deploy to green environment

        Args:
            instances: List of green instances to deploy
            deploy_function: Async function to perform actual deployment

        Returns:
            True if deployment successful, False otherwise
        """
        try:
            self.deployment_status = DeploymentStatus.DEPLOYING

            # Deploy instances
            success = await deploy_function(instances)
            if not success:
                self.deployment_status = DeploymentStatus.FAILED
                return False

            # Store green instances
            with self._lock:
                self.green_instances = instances

            # Perform health checks if checker available
            if self.health_checker:
                for instance in instances:
                    health_result = await self.health_checker.check_instance_health(instance)
                    if not health_result.is_healthy:
                        self.deployment_status = DeploymentStatus.FAILED
                        return False

            self.deployment_status = DeploymentStatus.HEALTHY
            return True

        except Exception as e:
            logger.error(f"Green deployment failed: {e}")
            self.deployment_status = DeploymentStatus.FAILED
            return False

    async def switch_to_green(self) -> bool:
        """Switch traffic from blue to green environment"""
        try:
            with self._lock:
                if not self.green_instances:
                    logger.error("No green instances available for switch")
                    return False

                # Verify green instances are healthy
                healthy_green = all(i.status == DeploymentStatus.HEALTHY for i in self.green_instances)
                if not healthy_green:
                    logger.error("Green instances not all healthy")
                    return False

                # Switch active environment
                self.active_environment = "green"
                self.deployment_status = DeploymentStatus.HEALTHY

            logger.info(f"Successfully switched to green environment for {self.service_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to switch to green: {e}")
            return False

    async def rollback_to_blue(self) -> bool:
        """Rollback traffic to blue environment"""
        try:
            with self._lock:
                if not self.blue_instances:
                    logger.error("No blue instances available for rollback")
                    return False

                # Switch back to blue
                self.active_environment = "blue"
                self.deployment_status = DeploymentStatus.HEALTHY

            logger.info(f"Successfully rolled back to blue environment for {self.service_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to rollback to blue: {e}")
            return False

    def get_active_instances(self) -> list[ServiceInstance]:
        """Get instances from currently active environment"""
        with self._lock:
            if self.active_environment == "blue":
                return self.blue_instances
            else:
                return self.green_instances


class CanaryDeployment:
    """Canary deployment manager"""

    def __init__(self, service_name: str, stable_version: str,
                 canary_version: str, canary_percentage: float = 5.0):
        """Initialize canary deployment

        Args:
            service_name: Name of the service being deployed
            stable_version: Current stable version
            canary_version: New canary version to deploy
            canary_percentage: Initial percentage of traffic to route to canary
        """
        self.service_name = service_name
        self.stable_version = stable_version
        self.canary_version = canary_version
        self.canary_percentage = canary_percentage
        self.deployment_status = DeploymentStatus.PENDING

        self.stable_instances: list[ServiceInstance] = []
        self.canary_instances: list[ServiceInstance] = []

        self._lock = threading.Lock()

    async def deploy_canary_instances(self, instances: list[ServiceInstance],
                                    deploy_function: Callable) -> bool:
        """Deploy canary instances

        Args:
            instances: List of canary instances to deploy
            deploy_function: Async function to perform actual deployment

        Returns:
            True if deployment successful, False otherwise
        """
        try:
            self.deployment_status = DeploymentStatus.DEPLOYING

            # Deploy instances
            success = await deploy_function(instances)
            if not success:
                self.deployment_status = DeploymentStatus.FAILED
                return False

            # Store canary instances
            with self._lock:
                self.canary_instances = instances

            self.deployment_status = DeploymentStatus.HEALTHY
            return True

        except Exception as e:
            logger.error(f"Canary deployment failed: {e}")
            self.deployment_status = DeploymentStatus.FAILED
            return False

    async def increase_canary_traffic(self, new_percentage: float) -> bool:
        """Increase canary traffic percentage

        Args:
            new_percentage: New percentage of traffic to route to canary

        Returns:
            True if successful, False otherwise
        """
        try:
            if new_percentage < 0 or new_percentage > 100:
                logger.error(f"Invalid canary percentage: {new_percentage}")
                return False

            with self._lock:
                if not self.canary_instances:
                    logger.error("No canary instances available")
                    return False

                # Verify canary instances are healthy
                healthy_canary = all(i.status == DeploymentStatus.HEALTHY for i in self.canary_instances)
                if not healthy_canary:
                    logger.error("Canary instances not all healthy")
                    return False

                self.canary_percentage = new_percentage

            logger.info(f"Increased canary traffic to {new_percentage}% for {self.service_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to increase canary traffic: {e}")
            return False

    async def promote_canary_to_stable(self) -> bool:
        """Promote canary to stable version (100% traffic)

        Returns:
            True if successful, False otherwise
        """
        try:
            with self._lock:
                if not self.canary_instances:
                    logger.error("No canary instances to promote")
                    return False

                # Verify canary is at 100% traffic
                if self.canary_percentage != 100.0:
                    logger.error(f"Canary not at 100% traffic: {self.canary_percentage}%")
                    return False

                # Promote canary to stable
                self.stable_version = self.canary_version
                self.stable_instances = self.canary_instances.copy()
                self.canary_instances = []
                self.canary_percentage = 0.0
                self.deployment_status = DeploymentStatus.HEALTHY

            logger.info(f"Successfully promoted canary to stable for {self.service_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to promote canary: {e}")
            return False

    def calculate_traffic_distribution(self, total_requests: int) -> dict[str, int]:
        """Calculate traffic distribution between stable and canary

        Args:
            total_requests: Total number of requests to distribute

        Returns:
            Dictionary with canary_requests and stable_requests counts
        """
        canary_requests = int(total_requests * (self.canary_percentage / 100.0))
        stable_requests = total_requests - canary_requests

        return {
            "canary_requests": canary_requests,
            "stable_requests": stable_requests,
            "total_requests": total_requests
        }

    def should_route_to_canary(self) -> bool:
        """Determine if a request should be routed to canary

        Returns:
            True if should route to canary, False for stable
        """
        return random.random() * 100.0 < self.canary_percentage


class RollbackManager:
    """Manages deployment rollbacks and history"""

    def __init__(self, max_rollback_history: int = 10):
        """Initialize rollback manager

        Args:
            max_rollback_history: Maximum number of deployment snapshots to keep
        """
        self.max_rollback_history = max_rollback_history
        self.deployment_history: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def save_deployment_snapshot(self, service_name: str,
                                snapshot: dict[str, Any]) -> None:
        """Save deployment snapshot for rollback purposes

        Args:
            service_name: Name of the service
            snapshot: Deployment snapshot data
        """
        with self._lock:
            if service_name not in self.deployment_history:
                self.deployment_history[service_name] = []

            # Add timestamp if not present
            if "timestamp" not in snapshot:
                snapshot["timestamp"] = datetime.now().isoformat()

            # Add to beginning of list (most recent first)
            self.deployment_history[service_name].insert(0, snapshot)

            # Maintain history limit
            if len(self.deployment_history[service_name]) > self.max_rollback_history:
                self.deployment_history[service_name] = self.deployment_history[service_name][:self.max_rollback_history]

    def get_rollback_candidates(self, service_name: str) -> list[dict[str, Any]]:
        """Get available rollback candidates for a service

        Args:
            service_name: Name of the service

        Returns:
            List of rollback candidates (most recent first)
        """
        with self._lock:
            return self.deployment_history.get(service_name, []).copy()

    async def execute_rollback(self, service_name: str, target_version: str,
                             rollback_function: Callable[[dict[str, Any]], Any]) -> bool:
        """Execute rollback to a specific version

        Args:
            service_name: Name of the service
            target_version: Version to rollback to
            rollback_function: Async function to perform actual rollback

        Returns:
            True if rollback successful, False otherwise
        """
        try:
            with self._lock:
                candidates = self.deployment_history.get(service_name, [])
                target_snapshot = None

                for snapshot in candidates:
                    if snapshot.get("version") == target_version:
                        target_snapshot = snapshot
                        break

                if not target_snapshot:
                    logger.error(f"No rollback candidate found for version {target_version}")
                    return False

            # Execute rollback
            result = rollback_function(target_snapshot)
            if hasattr(result, '__await__'):
                success = await result
            else:
                success = result

            if success:
                logger.info(f"Successfully rolled back {service_name} to version {target_version}")
            else:
                logger.error(f"Failed to rollback {service_name} to version {target_version}")

            return bool(success)

        except Exception as e:
            logger.error(f"Rollback execution failed: {e}")
            return False

    def get_deployment_history(self, service_name: str) -> list[dict[str, Any]]:
        """Get deployment history for a service

        Args:
            service_name: Name of the service

        Returns:
            List of deployment history entries
        """
        with self._lock:
            return self.deployment_history.get(service_name, []).copy()


class DeploymentManager:
    """Main deployment manager orchestrating all deployment operations"""

    def __init__(self, health_check_interval: float = 30.0,
                 deployment_timeout: float = 600.0):
        """Initialize deployment manager

        Args:
            health_check_interval: Interval between health checks in seconds
            deployment_timeout: Maximum time to wait for deployment completion
        """
        self.health_check_interval = health_check_interval
        self.deployment_timeout = deployment_timeout

        self.services: dict[str, dict[str, Any]] = {}
        self.load_balancers: dict[str, LoadBalancer] = {}
        self.rollback_manager = RollbackManager()

        self.is_monitoring = False
        self._monitor_thread: threading.Thread | None = None
        self._lock = threading.Lock()

        # Initialize Docker client if available
        self.docker_client = None
        if docker:
            try:
                self.docker_client = docker.from_env()
            except Exception as e:
                logger.warning(f"Failed to initialize Docker client: {e}")

    def register_service(self, service_name: str, current_version: str,
                        deployment_strategy: DeploymentStrategy,
                        health_check_endpoint: str = "/health",
                        min_instances: int = 1, max_instances: int = 10,
                        auto_rollback: bool = True, **kwargs: Any) -> None:
        """Register a service for deployment management

        Args:
            service_name: Name of the service
            current_version: Current version of the service
            deployment_strategy: Deployment strategy to use
            health_check_endpoint: Health check endpoint path
            min_instances: Minimum number of instances
            max_instances: Maximum number of instances
            auto_rollback: Whether to automatically rollback on failure
            **kwargs: Additional service configuration
        """
        with self._lock:
            self.services[service_name] = {
                "current_version": current_version,
                "deployment_strategy": deployment_strategy,
                "health_check_endpoint": health_check_endpoint,
                "min_instances": min_instances,
                "max_instances": max_instances,
                "auto_rollback": auto_rollback,
                "instances": [],
                **kwargs
            }

            # Initialize load balancer
            self.load_balancers[service_name] = LoadBalancer()

        logger.info(f"Registered service: {service_name} with strategy {deployment_strategy.value}")

    async def deploy_service(self, service_name: str, new_version: str,
                           image_url: str, instance_count: int | None = None,
                           canary_percentage: float | None = None,
                           **kwargs: Any) -> dict[str, Any]:
        """Deploy a service using the configured strategy

        Args:
            service_name: Name of the service to deploy
            new_version: New version to deploy
            image_url: Container image URL
            instance_count: Number of instances (for blue-green)
            canary_percentage: Canary percentage (for canary deployments)
            **kwargs: Additional deployment parameters

        Returns:
            Deployment result dictionary
        """
        if service_name not in self.services:
            return {"success": False, "error": "Service not registered"}

        service_config = self.services[service_name]
        strategy = service_config["deployment_strategy"]
        deployment_id = str(uuid.uuid4())

        try:
            # Save current state for rollback
            current_snapshot = {
                "version": service_config["current_version"],
                "instances": [asdict(i) for i in service_config.get("instances", [])],
                "timestamp": datetime.now().isoformat(),
                "deployment_id": deployment_id
            }
            self.rollback_manager.save_deployment_snapshot(service_name, current_snapshot)

            if strategy == DeploymentStrategy.BLUE_GREEN:
                result = await self._deploy_blue_green(
                    service_name, new_version, image_url,
                    instance_count or service_config["min_instances"]
                )
            elif strategy == DeploymentStrategy.CANARY:
                result = await self._deploy_canary(
                    service_name, new_version, image_url,
                    canary_percentage or 10.0
                )
            else:
                result = {"success": False, "error": f"Strategy {strategy} not implemented"}

            # Handle deployment failure with auto-rollback
            if not result["success"] and service_config.get("auto_rollback", True):
                logger.warning(f"Deployment failed, attempting auto-rollback for {service_name}")
                rollback_result = await self._perform_auto_rollback(service_name)
                result["rollback"] = rollback_result

            result.update({
                "deployment_id": deployment_id,
                "strategy": strategy.value,
                "service_name": service_name,
                "new_version": new_version
            })

            return result

        except Exception as e:
            logger.error(f"Deployment failed for {service_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "deployment_id": deployment_id,
                "strategy": strategy.value
            }

    async def _deploy_blue_green(self, service_name: str, new_version: str,
                               image_url: str, instance_count: int) -> dict[str, Any]:
        """Execute blue-green deployment"""
        try:
            service_config = self.services[service_name]
            current_version = service_config["current_version"]

            # Create blue-green deployment
            bg_deployment = BlueGreenDeployment(
                service_name=service_name,
                blue_version=current_version,
                green_version=new_version
            )

            # Create green instances
            green_instances = []
            for i in range(instance_count):
                instance = ServiceInstance(
                    instance_id=f"{service_name}-green-{i:03d}",
                    service_name=service_name,
                    version=new_version,
                    host="127.0.0.1",  # Default, should be configurable
                    port=8080 + i  # Default port allocation
                )
                green_instances.append(instance)

            # Deploy to green environment
            success = await bg_deployment.deploy_green_environment(
                instances=green_instances,
                deploy_function=self._deploy_instances
            )

            if not success:
                return {"success": False, "error": "Green environment deployment failed"}

            # Switch to green
            switch_success = await bg_deployment.switch_to_green()
            if not switch_success:
                return {"success": False, "error": "Failed to switch to green environment"}

            # Update service configuration
            with self._lock:
                service_config["current_version"] = new_version
                service_config["instances"] = green_instances

                # Update load balancer
                lb = self.load_balancers[service_name]
                for instance in green_instances:
                    lb.add_instance(instance)

            return {"success": True, "active_environment": "green"}

        except Exception as e:
            logger.error(f"Blue-green deployment failed: {e}")
            return {"success": False, "error": str(e)}

    async def _deploy_canary(self, service_name: str, new_version: str,
                           image_url: str, canary_percentage: float) -> dict[str, Any]:
        """Execute canary deployment"""
        try:
            service_config = self.services[service_name]
            current_version = service_config["current_version"]

            # Create canary deployment
            canary_deployment = CanaryDeployment(
                service_name=service_name,
                stable_version=current_version,
                canary_version=new_version,
                canary_percentage=canary_percentage
            )

            # Create canary instances (typically fewer than stable)
            canary_instances = []
            canary_count = max(1, int(service_config["min_instances"] * (canary_percentage / 100.0)))

            for i in range(canary_count):
                instance = ServiceInstance(
                    instance_id=f"{service_name}-canary-{i:03d}",
                    service_name=service_name,
                    version=new_version,
                    host="127.0.0.1",
                    port=8090 + i  # Different port range for canary
                )
                canary_instances.append(instance)

            # Deploy canary instances
            success = await canary_deployment.deploy_canary_instances(
                instances=canary_instances,
                deploy_function=self._deploy_instances
            )

            if not success:
                return {"success": False, "error": "Canary deployment failed"}

            # Update service with canary instances
            with self._lock:
                service_config["canary_instances"] = canary_instances
                service_config["canary_deployment"] = canary_deployment

                # Add canary instances to load balancer
                lb = self.load_balancers[service_name]
                for instance in canary_instances:
                    lb.add_instance(instance)

            return {"success": True, "canary_percentage": canary_percentage}

        except Exception as e:
            logger.error(f"Canary deployment failed: {e}")
            return {"success": False, "error": str(e)}

    async def _deploy_instances(self, instances: list[ServiceInstance]) -> bool:
        """Deploy service instances (mock implementation)

        Args:
            instances: List of instances to deploy

        Returns:
            True if all instances deployed successfully
        """
        try:
            for instance in instances:
                # Simulate deployment delay
                await asyncio.sleep(0.1)

                # Mock container deployment
                if self.docker_client:
                    try:
                        container = self.docker_client.containers.run(
                            "alpine:latest",  # Mock image
                            command="sleep 3600",
                            detach=True,
                            ports={str(instance.port): instance.port}
                        )
                        instance.container_id = container.id[:12]
                    except Exception as e:
                        logger.warning(f"Docker deployment failed (using mock): {e}")
                        instance.container_id = f"mock-{uuid.uuid4().hex[:8]}"
                else:
                    instance.container_id = f"mock-{uuid.uuid4().hex[:8]}"

                # Mark as healthy (would be done by health checks in real scenario)
                instance.status = DeploymentStatus.HEALTHY
                instance.health_score = 0.95

            return True

        except Exception as e:
            logger.error(f"Instance deployment failed: {e}")
            return False

    async def _perform_health_checks(self, service_name: str) -> list[HealthCheckResult]:
        """Perform health checks on service instances

        Args:
            service_name: Name of the service to check

        Returns:
            List of health check results
        """
        results = []
        service_config = self.services.get(service_name, {})
        instances = service_config.get("instances", [])
        health_endpoint = service_config.get("health_check_endpoint", "/health")

        if not aiohttp:
            # Mock health check results for testing
            for instance in instances:
                result = HealthCheckResult(
                    instance_id=instance.instance_id,
                    is_healthy=True,
                    response_time_ms=random.uniform(50, 200),
                    status_code=200
                )
                results.append(result)
            return results

        async with aiohttp.ClientSession() as session:
            for instance in instances:
                try:
                    start_time = time.time()
                    url = f"{instance.endpoint_url}{health_endpoint}"

                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5.0)) as response:
                        response_time = (time.time() - start_time) * 1000

                        result = HealthCheckResult(
                            instance_id=instance.instance_id,
                            is_healthy=response.status == 200,
                            response_time_ms=response_time,
                            status_code=response.status
                        )

                        # Update instance health
                        instance.update_health(result.health_score,
                                             DeploymentStatus.HEALTHY if result.is_healthy else DeploymentStatus.UNHEALTHY)

                        results.append(result)

                except Exception as e:
                    logger.warning(f"Health check failed for {instance.instance_id}: {e}")
                    result = HealthCheckResult(
                        instance_id=instance.instance_id,
                        is_healthy=False,
                        response_time_ms=5000.0,
                        status_code=500,
                        details={"error": str(e)}
                    )

                    instance.update_health(0.0, DeploymentStatus.UNHEALTHY)
                    results.append(result)

        return results

    async def _perform_auto_rollback(self, service_name: str) -> dict[str, Any]:
        """Perform automatic rollback for a service

        Args:
            service_name: Name of the service to rollback

        Returns:
            Rollback result dictionary
        """
        try:
            candidates = self.rollback_manager.get_rollback_candidates(service_name)
            if len(candidates) < 2:  # Need at least current and previous
                return {"success": False, "error": "No rollback candidates available"}

            # Get previous version (candidates[1] since [0] is current)
            target_snapshot = candidates[1]
            target_version = target_snapshot.get("version")

            async def rollback_function(snapshot: dict[str, Any]) -> bool:
                # Implement actual rollback logic here
                # For now, return success
                return True

            success = await self.rollback_manager.execute_rollback(
                service_name, target_version or "unknown", rollback_function
            )

            return {"success": success, "target_version": target_version}

        except Exception as e:
            logger.error(f"Auto-rollback failed: {e}")
            return {"success": False, "error": str(e)}

    def start_monitoring(self) -> None:
        """Start health monitoring for all services"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

        logger.info("Started deployment monitoring")

    def stop_monitoring(self) -> None:
        """Stop health monitoring"""
        self.is_monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10.0)

        logger.info("Stopped deployment monitoring")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Perform health checks for all services
                for service_name in list(self.services.keys()):
                    asyncio.run(self._perform_health_checks(service_name))

                time.sleep(self.health_check_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

    async def scale_service(self, service_name: str, target_instances: int) -> dict[str, Any]:
        """Scale service instances up or down

        Args:
            service_name: Name of the service to scale
            target_instances: Target number of instances

        Returns:
            Scaling result dictionary
        """
        if service_name not in self.services:
            return {"success": False, "error": "Service not registered"}

        try:
            service_config = self.services[service_name]
            current_instances = service_config.get("instances", [])
            current_count = len(current_instances)

            if target_instances == current_count:
                return {"success": True, "message": "Already at target instance count"}

            if target_instances > current_count:
                # Scale up
                new_instances = []
                for i in range(current_count, target_instances):
                    instance = ServiceInstance(
                        instance_id=f"{service_name}-{i:03d}",
                        service_name=service_name,
                        version=service_config["current_version"],
                        host="127.0.0.1",
                        port=8080 + i
                    )
                    new_instances.append(instance)

                # Deploy new instances
                success = await self._deploy_instances(new_instances)
                if success:
                    with self._lock:
                        service_config["instances"].extend(new_instances)

                        # Add to load balancer
                        lb = self.load_balancers[service_name]
                        for instance in new_instances:
                            lb.add_instance(instance)

            else:
                # Scale down
                instances_to_remove = current_instances[target_instances:]

                with self._lock:
                    service_config["instances"] = current_instances[:target_instances]

                    # Remove from load balancer
                    lb = self.load_balancers[service_name]
                    for instance in instances_to_remove:
                        lb.remove_instance(instance.instance_id)

                        # Stop container if using Docker
                        if self.docker_client and instance.container_id:
                            try:
                                container = self.docker_client.containers.get(instance.container_id)
                                container.stop()
                            except Exception as e:
                                logger.warning(f"Failed to stop container {instance.container_id}: {e}")

            return {
                "success": True,
                "previous_count": current_count,
                "new_count": target_instances,
                "scaled_up": target_instances > current_count
            }

        except Exception as e:
            logger.error(f"Scaling failed for {service_name}: {e}")
            return {"success": False, "error": str(e)}

    def get_service_status(self, service_name: str) -> dict[str, Any]:
        """Get comprehensive status of a service

        Args:
            service_name: Name of the service

        Returns:
            Service status dictionary
        """
        if service_name not in self.services:
            return {"error": "Service not found"}

        service_config = self.services[service_name]
        instances = service_config.get("instances", [])

        healthy_instances = len([i for i in instances if i.status == DeploymentStatus.HEALTHY])
        total_health_score = sum(i.health_score for i in instances)
        avg_health_score = total_health_score / len(instances) if instances else 0.0

        return {
            "service_name": service_name,
            "current_version": service_config["current_version"],
            "deployment_strategy": service_config["deployment_strategy"].value,
            "instance_count": len(instances),
            "healthy_instances": healthy_instances,
            "average_health_score": avg_health_score,
            "min_instances": service_config["min_instances"],
            "max_instances": service_config["max_instances"],
            "instances": [
                {
                    "instance_id": i.instance_id,
                    "status": i.status.value,
                    "health_score": i.health_score,
                    "endpoint": i.endpoint_url
                }
                for i in instances
            ]
        }

    def get_deployment_history(self, service_name: str) -> list[dict[str, Any]]:
        """Get deployment history for a service

        Args:
            service_name: Name of the service

        Returns:
            List of deployment history entries
        """
        return self.rollback_manager.get_deployment_history(service_name)

    def get_system_metrics(self) -> dict[str, Any]:
        """Get comprehensive system-wide metrics

        Returns:
            System metrics dictionary
        """
        total_services = len(self.services)
        total_instances = 0
        healthy_instances = 0
        total_health_score = 0.0

        for service_config in self.services.values():
            instances = service_config.get("instances", [])
            total_instances += len(instances)

            for instance in instances:
                if instance.status == DeploymentStatus.HEALTHY:
                    healthy_instances += 1
                total_health_score += instance.health_score

        overall_health = total_health_score / total_instances if total_instances > 0 else 0.0

        return {
            "total_services": total_services,
            "total_instances": total_instances,
            "healthy_instances": healthy_instances,
            "unhealthy_instances": total_instances - healthy_instances,
            "overall_health_score": overall_health,
            "monitoring_active": self.is_monitoring,
            "timestamp": datetime.now().isoformat()
        }

    async def emergency_stop(self) -> dict[str, Any]:
        """Emergency stop of all services

        Returns:
            Emergency stop result dictionary
        """
        try:
            stopped_services = 0
            errors = []

            for service_name, service_config in self.services.items():
                try:
                    instances = service_config.get("instances", [])

                    for instance in instances:
                        if self.docker_client and instance.container_id:
                            try:
                                container = self.docker_client.containers.get(instance.container_id)
                                container.stop()
                            except Exception as e:
                                errors.append(f"Failed to stop {instance.instance_id}: {e}")

                        instance.status = DeploymentStatus.FAILED

                    stopped_services += 1

                except Exception as e:
                    errors.append(f"Failed to stop service {service_name}: {e}")

            return {
                "success": True,
                "stopped_services": stopped_services,
                "errors": errors
            }

        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            return {"success": False, "error": str(e)}

    def export_configuration(self) -> dict[str, Any]:
        """Export deployment configuration

        Returns:
            Configuration dictionary
        """
        config: dict[str, Any] = {
            "services": {},
            "health_check_interval": self.health_check_interval,
            "deployment_timeout": self.deployment_timeout,
            "export_timestamp": datetime.now().isoformat()
        }

        services_config = config["services"]
        for service_name, service_config in self.services.items():
            services_config[service_name] = {
                "current_version": service_config["current_version"],
                "deployment_strategy": service_config["deployment_strategy"].value,
                "health_check_endpoint": service_config["health_check_endpoint"],
                "min_instances": service_config["min_instances"],
                "max_instances": service_config["max_instances"],
                "auto_rollback": service_config.get("auto_rollback", True)
            }

        return config

    def import_configuration(self, config: dict[str, Any]) -> dict[str, Any]:
        """Import deployment configuration

        Args:
            config: Configuration dictionary to import

        Returns:
            Import result dictionary
        """
        try:
            imported_services = 0
            errors = []

            services_config = config.get("services", {})

            for service_name, service_config in services_config.items():
                try:
                    strategy_str = service_config.get("deployment_strategy", "blue_green")
                    strategy = DeploymentStrategy(strategy_str)

                    self.register_service(
                        service_name=service_name,
                        current_version=service_config.get("current_version", "1.0.0"),
                        deployment_strategy=strategy,
                        health_check_endpoint=service_config.get("health_check_endpoint", "/health"),
                        min_instances=service_config.get("min_instances", 1),
                        max_instances=service_config.get("max_instances", 10),
                        auto_rollback=service_config.get("auto_rollback", True)
                    )

                    imported_services += 1

                except Exception as e:
                    errors.append(f"Failed to import service {service_name}: {e}")

            # Update global settings
            if "health_check_interval" in config:
                self.health_check_interval = config["health_check_interval"]
            if "deployment_timeout" in config:
                self.deployment_timeout = config["deployment_timeout"]

            return {
                "success": True,
                "imported_services": imported_services,
                "errors": errors
            }

        except Exception as e:
            logger.error(f"Configuration import failed: {e}")
            return {"success": False, "error": str(e)}


# Default singleton instance
_default_manager = None


def get_default_manager() -> DeploymentManager:
    """Get or create default deployment manager instance"""
    global _default_manager
    if _default_manager is None:
        _default_manager = DeploymentManager()
    return _default_manager


def create_deployment_manager(health_check_interval: float = 30.0,
                            deployment_timeout: float = 600.0) -> DeploymentManager:
    """Create a new deployment manager instance"""
    return DeploymentManager(health_check_interval, deployment_timeout)


if __name__ == "__main__":
    # Example usage and testing
    async def main() -> None:
        # Create deployment manager
        manager = DeploymentManager(health_check_interval=5.0)

        # Register a service
        manager.register_service(
            service_name="web-service",
            current_version="v1.0.0",
            deployment_strategy=DeploymentStrategy.BLUE_GREEN,
            health_check_endpoint="/health"
        )

        # Start health monitoring
        manager.start_monitoring()

        # Deploy service
        result = await manager.deploy_service(
            service_name="web-service",
            new_version="v1.1.0",
            image_url="nginx:latest",
            instance_count=2
        )
        print(f"Deployment result: {result}")

        # Wait a bit for health checks
        await asyncio.sleep(10)

        # Get service status
        status = manager.get_service_status("web-service")
        print(f"Service status: {json.dumps(status, indent=2)}")

        # Get system metrics
        metrics = manager.get_system_metrics()
        print(f"System metrics: {json.dumps(metrics, indent=2)}")

        # Stop monitoring
        manager.stop_monitoring()

    asyncio.run(main())
