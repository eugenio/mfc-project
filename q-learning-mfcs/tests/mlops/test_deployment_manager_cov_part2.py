"""Tests for deployment_manager module - Part 2: Manager and utilities."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import asyncio
from unittest.mock import patch

import pytest

from mlops.deployment_manager import (
    DeploymentManager,
    DeploymentStatus,
    DeploymentStrategy,
    RollbackManager,
    ServiceInstance,
    create_deployment_manager,
    get_default_manager,
)


# --- RollbackManager Tests ---

class TestRollbackManager:
    def test_save_and_get_snapshot(self):
        rm = RollbackManager()
        rm.save_deployment_snapshot("svc", {"version": "v1"})
        candidates = rm.get_rollback_candidates("svc")
        assert len(candidates) == 1
        assert candidates[0]["version"] == "v1"
        assert "timestamp" in candidates[0]

    def test_save_snapshot_with_timestamp(self):
        rm = RollbackManager()
        rm.save_deployment_snapshot("svc", {"version": "v1", "timestamp": "2024-01-01"})
        candidates = rm.get_rollback_candidates("svc")
        assert candidates[0]["timestamp"] == "2024-01-01"

    def test_history_limit(self):
        rm = RollbackManager(max_rollback_history=3)
        for i in range(5):
            rm.save_deployment_snapshot("svc", {"version": f"v{i}"})
        candidates = rm.get_rollback_candidates("svc")
        assert len(candidates) == 3

    def test_get_candidates_nonexistent(self):
        rm = RollbackManager()
        assert rm.get_rollback_candidates("none") == []

    def test_execute_rollback_success(self):
        rm = RollbackManager()
        rm.save_deployment_snapshot("svc", {"version": "v1"})

        def rollback_fn(snapshot):
            return True

        result = asyncio.get_event_loop().run_until_complete(
            rm.execute_rollback("svc", "v1", rollback_fn)
        )
        assert result is True

    def test_execute_rollback_async_fn(self):
        rm = RollbackManager()
        rm.save_deployment_snapshot("svc", {"version": "v1"})

        async def rollback_fn(snapshot):
            return True

        result = asyncio.get_event_loop().run_until_complete(
            rm.execute_rollback("svc", "v1", rollback_fn)
        )
        assert result is True

    def test_execute_rollback_not_found(self):
        rm = RollbackManager()
        rm.save_deployment_snapshot("svc", {"version": "v1"})

        result = asyncio.get_event_loop().run_until_complete(
            rm.execute_rollback("svc", "v999", lambda s: True)
        )
        assert result is False

    def test_execute_rollback_failure(self):
        rm = RollbackManager()
        rm.save_deployment_snapshot("svc", {"version": "v1"})

        def rollback_fn(snapshot):
            return False

        result = asyncio.get_event_loop().run_until_complete(
            rm.execute_rollback("svc", "v1", rollback_fn)
        )
        assert result is False

    def test_execute_rollback_exception(self):
        rm = RollbackManager()
        rm.save_deployment_snapshot("svc", {"version": "v1"})

        def rollback_fn(snapshot):
            raise RuntimeError("fail")

        result = asyncio.get_event_loop().run_until_complete(
            rm.execute_rollback("svc", "v1", rollback_fn)
        )
        assert result is False

    def test_get_deployment_history(self):
        rm = RollbackManager()
        rm.save_deployment_snapshot("svc", {"version": "v1"})
        history = rm.get_deployment_history("svc")
        assert len(history) == 1
        assert rm.get_deployment_history("none") == []


# --- DeploymentManager Tests ---

class TestDeploymentManager:
    def test_init_no_docker(self):
        with patch("mlops.deployment_manager.docker", None):
            dm = DeploymentManager()
            assert dm.docker_client is None

    def test_register_service(self):
        dm = DeploymentManager()
        dm.docker_client = None
        dm.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN)
        assert "svc" in dm.services
        assert "svc" in dm.load_balancers

    def test_deploy_unregistered_service(self):
        dm = DeploymentManager()
        dm.docker_client = None
        result = asyncio.get_event_loop().run_until_complete(
            dm.deploy_service("none", "v2", "img:latest")
        )
        assert result["success"] is False

    def test_deploy_blue_green(self):
        dm = DeploymentManager()
        dm.docker_client = None
        dm.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN,
                            min_instances=1)
        result = asyncio.get_event_loop().run_until_complete(
            dm.deploy_service("svc", "v2", "img:latest", instance_count=1)
        )
        assert result["success"] is True
        assert result["strategy"] == "blue_green"

    def test_deploy_canary(self):
        dm = DeploymentManager()
        dm.docker_client = None
        dm.register_service("svc", "v1", DeploymentStrategy.CANARY,
                            min_instances=2)
        result = asyncio.get_event_loop().run_until_complete(
            dm.deploy_service("svc", "v2", "img:latest", canary_percentage=10.0)
        )
        assert result["success"] is True

    def test_deploy_unsupported_strategy(self):
        dm = DeploymentManager()
        dm.docker_client = None
        dm.register_service("svc", "v1", DeploymentStrategy.ROLLING)
        result = asyncio.get_event_loop().run_until_complete(
            dm.deploy_service("svc", "v2", "img:latest")
        )
        assert result["success"] is False

    def test_scale_service_up(self):
        dm = DeploymentManager()
        dm.docker_client = None
        dm.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN)
        dm.services["svc"]["instances"] = []
        result = asyncio.get_event_loop().run_until_complete(
            dm.scale_service("svc", 2)
        )
        assert result["success"] is True
        assert result["scaled_up"] is True

    def test_scale_service_down(self):
        dm = DeploymentManager()
        dm.docker_client = None
        dm.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN)
        inst1 = ServiceInstance("svc-000", "svc", "v1", "localhost", 8080,
                                status=DeploymentStatus.HEALTHY)
        inst2 = ServiceInstance("svc-001", "svc", "v1", "localhost", 8081,
                                status=DeploymentStatus.HEALTHY)
        dm.services["svc"]["instances"] = [inst1, inst2]
        dm.load_balancers["svc"].add_instance(inst1)
        dm.load_balancers["svc"].add_instance(inst2)
        result = asyncio.get_event_loop().run_until_complete(
            dm.scale_service("svc", 1)
        )
        assert result["success"] is True
        assert result["scaled_up"] is False

    def test_scale_service_same_count(self):
        dm = DeploymentManager()
        dm.docker_client = None
        dm.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN)
        dm.services["svc"]["instances"] = [
            ServiceInstance("svc-000", "svc", "v1", "localhost", 8080)
        ]
        result = asyncio.get_event_loop().run_until_complete(
            dm.scale_service("svc", 1)
        )
        assert result["success"] is True

    def test_scale_service_not_registered(self):
        dm = DeploymentManager()
        dm.docker_client = None
        result = asyncio.get_event_loop().run_until_complete(
            dm.scale_service("none", 2)
        )
        assert result["success"] is False

    def test_get_service_status(self):
        dm = DeploymentManager()
        dm.docker_client = None
        dm.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN)
        inst = ServiceInstance("id1", "svc", "v1", "localhost", 8080,
                               status=DeploymentStatus.HEALTHY, health_score=0.9)
        dm.services["svc"]["instances"] = [inst]
        status = dm.get_service_status("svc")
        assert status["service_name"] == "svc"
        assert status["instance_count"] == 1

    def test_get_service_status_not_found(self):
        dm = DeploymentManager()
        dm.docker_client = None
        status = dm.get_service_status("none")
        assert "error" in status

    def test_get_system_metrics_empty(self):
        dm = DeploymentManager()
        dm.docker_client = None
        metrics = dm.get_system_metrics()
        assert metrics["total_services"] == 0
        assert metrics["overall_health_score"] == 0.0

    def test_get_system_metrics_with_services(self):
        dm = DeploymentManager()
        dm.docker_client = None
        dm.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN)
        inst = ServiceInstance("id1", "svc", "v1", "localhost", 8080,
                               status=DeploymentStatus.HEALTHY, health_score=1.0)
        dm.services["svc"]["instances"] = [inst]
        metrics = dm.get_system_metrics()
        assert metrics["total_services"] == 1
        assert metrics["healthy_instances"] == 1

    def test_emergency_stop(self):
        dm = DeploymentManager()
        dm.docker_client = None
        dm.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN)
        inst = ServiceInstance("id1", "svc", "v1", "localhost", 8080)
        dm.services["svc"]["instances"] = [inst]
        result = asyncio.get_event_loop().run_until_complete(dm.emergency_stop())
        assert result["success"] is True
        assert inst.status == DeploymentStatus.FAILED

    def test_export_and_import_configuration(self):
        dm = DeploymentManager(health_check_interval=15.0)
        dm.docker_client = None
        dm.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN,
                            auto_rollback=True)
        config = dm.export_configuration()
        assert "svc" in config["services"]

        dm2 = DeploymentManager()
        dm2.docker_client = None
        result = dm2.import_configuration(config)
        assert result["success"] is True
        assert result["imported_services"] == 1
        assert dm2.health_check_interval == 15.0

    def test_import_configuration_error(self):
        dm = DeploymentManager()
        dm.docker_client = None
        config = {"services": {"bad": {"deployment_strategy": "invalid_strategy"}}}
        result = dm.import_configuration(config)
        assert len(result["errors"]) > 0

    def test_start_stop_monitoring(self):
        dm = DeploymentManager(health_check_interval=0.1)
        dm.docker_client = None
        dm.start_monitoring()
        assert dm.is_monitoring is True
        # Starting again should be no-op
        dm.start_monitoring()
        dm.stop_monitoring()
        assert dm.is_monitoring is False

    def test_get_deployment_history(self):
        dm = DeploymentManager()
        dm.docker_client = None
        history = dm.get_deployment_history("svc")
        assert history == []

    def test_health_checks_no_aiohttp(self):
        dm = DeploymentManager()
        dm.docker_client = None
        dm.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN)
        inst = ServiceInstance("id1", "svc", "v1", "localhost", 8080)
        dm.services["svc"]["instances"] = [inst]
        with patch("mlops.deployment_manager.aiohttp", None):
            results = asyncio.get_event_loop().run_until_complete(
                dm._perform_health_checks("svc")
            )
        assert len(results) == 1
        assert results[0].is_healthy is True

    def test_auto_rollback_insufficient_candidates(self):
        dm = DeploymentManager()
        dm.docker_client = None
        result = asyncio.get_event_loop().run_until_complete(
            dm._perform_auto_rollback("svc")
        )
        assert result["success"] is False

    def test_deploy_instances_no_docker(self):
        dm = DeploymentManager()
        dm.docker_client = None
        inst = ServiceInstance("id1", "svc", "v1", "localhost", 8080)
        result = asyncio.get_event_loop().run_until_complete(
            dm._deploy_instances([inst])
        )
        assert result is True
        assert inst.status == DeploymentStatus.HEALTHY
        assert inst.container_id is not None


# --- Module-level function tests ---

class TestModuleFunctions:
    def test_get_default_manager(self):
        import mlops.deployment_manager as mod
        mod._default_manager = None
        mgr = get_default_manager()
        assert isinstance(mgr, DeploymentManager)
        mgr2 = get_default_manager()
        assert mgr is mgr2
        mod._default_manager = None

    def test_create_deployment_manager(self):
        dm = create_deployment_manager(10.0, 300.0)
        assert dm.health_check_interval == 10.0
        assert dm.deployment_timeout == 300.0


# --- Enum Tests ---

class TestEnums:
    def test_deployment_strategy_values(self):
        assert DeploymentStrategy.BLUE_GREEN.value == "blue_green"
        assert DeploymentStrategy.CANARY.value == "canary"
        assert DeploymentStrategy.ROLLING.value == "rolling"
        assert DeploymentStrategy.RECREATE.value == "recreate"

    def test_deployment_status_values(self):
        assert DeploymentStatus.PENDING.value == "pending"
        assert DeploymentStatus.DEPLOYING.value == "deploying"
        assert DeploymentStatus.HEALTHY.value == "healthy"
        assert DeploymentStatus.DEGRADED.value == "degraded"
        assert DeploymentStatus.UNHEALTHY.value == "unhealthy"
        assert DeploymentStatus.FAILED.value == "failed"
        assert DeploymentStatus.ROLLING_BACK.value == "rolling_back"
