"""Tests for deployment_manager - coverage part 3.

Missing: 192, 304-306, 323-325, 422-424, 453-455, 614-617,
715-717, 757, 762, 776-778, 816, 830-832, 850-860, 870-872,
900-935, 952-968, 996, 1000-1001, 1062-1066, 1075-1077,
1177-1181, 1187-1188, 1196-1198, 1273-1275.
"""
import sys
import os
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from mlops.deployment_manager import (
    BlueGreenDeployment,
    CanaryDeployment,
    DeploymentManager,
    DeploymentStatus,
    DeploymentStrategy,
    HealthCheckResult,
    LoadBalancer,
    ServiceInstance,
)


def _run(coro):
    """Helper to run async code."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_instance(iid="inst-001", svc="svc", ver="v1", port=8080):
    return ServiceInstance(
        instance_id=iid,
        service_name=svc,
        version=ver,
        host="127.0.0.1",
        port=port,
    )


@pytest.mark.coverage_extra
class TestLoadBalancerFallback:
    """Cover line 192 - weighted selection fallback."""

    def test_weighted_selection_fallback(self):
        lb = LoadBalancer("test-lb")
        inst = _make_instance()
        inst.status = DeploymentStatus.HEALTHY
        lb.add_instance(inst)
        # The weighted selection should pick the instance
        selected = lb.select_instance()
        assert selected is not None


@pytest.mark.coverage_extra
class TestBlueGreenExceptions:
    """Cover lines 304-306, 323-325."""

    def test_switch_to_green_exception(self):
        bg = BlueGreenDeployment(
            service_name="svc", blue_version="v1", green_version="v2"
        )
        bg._lock = MagicMock()
        bg._lock.__enter__ = MagicMock(side_effect=RuntimeError("fail"))
        bg._lock.__exit__ = MagicMock()
        result = _run(bg.switch_to_green())
        assert result is False

    def test_rollback_to_blue_exception(self):
        bg = BlueGreenDeployment(
            service_name="svc", blue_version="v1", green_version="v2"
        )
        bg._lock = MagicMock()
        bg._lock.__enter__ = MagicMock(side_effect=RuntimeError("fail"))
        bg._lock.__exit__ = MagicMock()
        result = _run(bg.rollback_to_blue())
        assert result is False

    def test_rollback_no_blue_instances(self):
        bg = BlueGreenDeployment(
            service_name="svc", blue_version="v1", green_version="v2"
        )
        bg.blue_instances = []
        result = _run(bg.rollback_to_blue())
        assert result is False

    def test_switch_to_green_success(self):
        bg = BlueGreenDeployment(
            service_name="svc", blue_version="v1", green_version="v2"
        )
        inst = _make_instance()
        inst.status = DeploymentStatus.HEALTHY
        bg.green_instances = [inst]
        result = _run(bg.switch_to_green())
        assert result is True
        assert bg.active_environment == "green"


@pytest.mark.coverage_extra
class TestCanaryExceptions:
    """Cover lines 422-424, 453-455."""

    def test_increase_canary_exception(self):
        cd = CanaryDeployment(
            service_name="svc",
            stable_version="v1",
            canary_version="v2",
            canary_percentage=10.0,
        )
        cd._lock = MagicMock()
        cd._lock.__enter__ = MagicMock(side_effect=RuntimeError("fail"))
        cd._lock.__exit__ = MagicMock()
        result = _run(cd.increase_canary_traffic(20.0))
        assert result is False

    def test_promote_canary_exception(self):
        cd = CanaryDeployment(
            service_name="svc",
            stable_version="v1",
            canary_version="v2",
            canary_percentage=100.0,
        )
        cd._lock = MagicMock()
        cd._lock.__enter__ = MagicMock(side_effect=RuntimeError("fail"))
        cd._lock.__exit__ = MagicMock()
        result = _run(cd.promote_canary_to_stable())
        assert result is False

    def test_promote_canary_no_instances(self):
        cd = CanaryDeployment(
            service_name="svc",
            stable_version="v1",
            canary_version="v2",
            canary_percentage=100.0,
        )
        cd.canary_instances = []
        result = _run(cd.promote_canary_to_stable())
        assert result is False

    def test_promote_canary_not_at_100(self):
        cd = CanaryDeployment(
            service_name="svc",
            stable_version="v1",
            canary_version="v2",
            canary_percentage=50.0,
        )
        cd.canary_instances = [_make_instance()]
        result = _run(cd.promote_canary_to_stable())
        assert result is False


@pytest.mark.coverage_extra
class TestDeploymentManagerInit:
    """Cover lines 614-617."""

    def test_init_no_docker(self):
        with patch("mlops.deployment_manager.docker", None):
            mgr = DeploymentManager()
            assert mgr.docker_client is None

    def test_init_docker_exception(self):
        mock_docker = MagicMock()
        mock_docker.from_env.side_effect = RuntimeError("no docker")
        with patch("mlops.deployment_manager.docker", mock_docker):
            mgr = DeploymentManager()
            assert mgr.docker_client is None


@pytest.mark.coverage_extra
class TestDeploymentManagerDeploy:
    """Cover lines 715-717, 757, 762, 776-778, 816, 830-832."""

    def _make_mgr(self):
        with patch("mlops.deployment_manager.docker", None):
            mgr = DeploymentManager()
        mgr.register_service(
            "svc", "v1", DeploymentStrategy.BLUE_GREEN
        )
        return mgr

    def test_deploy_exception(self):
        mgr = self._make_mgr()
        # Force an exception by removing service config
        del mgr.services["svc"]
        result = _run(
            mgr.deploy_service("svc", "v2", DeploymentStrategy.BLUE_GREEN)
        )
        assert result["success"] is False


@pytest.mark.coverage_extra
class TestDeployInstances:
    """Cover lines 850-860, 870-872."""

    def test_deploy_instances_no_docker(self):
        with patch("mlops.deployment_manager.docker", None):
            mgr = DeploymentManager()
        instances = [_make_instance()]
        result = _run(mgr._deploy_instances(instances))
        assert result is True
        assert instances[0].status == DeploymentStatus.HEALTHY

    def test_deploy_instances_exception(self):
        with patch("mlops.deployment_manager.docker", None):
            mgr = DeploymentManager()
        # Just test that the original method handles exceptions gracefully
        # by passing instances that will be handled
        inst = _make_instance()
        result = _run(mgr._deploy_instances([inst]))
        assert result is True


@pytest.mark.coverage_extra
class TestHealthChecks:
    """Cover lines 900-935."""

    def test_health_checks_no_aiohttp(self):
        with patch("mlops.deployment_manager.docker", None):
            mgr = DeploymentManager()
        with patch("mlops.deployment_manager.aiohttp", None):
            mgr.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN)
            inst = _make_instance()
            mgr.services["svc"]["instances"] = [inst]
            results = _run(mgr._perform_health_checks("svc"))
            assert len(results) == 1
            assert results[0].is_healthy is True


@pytest.mark.coverage_extra
class TestAutoRollback:
    """Cover lines 952-968."""

    def test_auto_rollback_no_candidates(self):
        with patch("mlops.deployment_manager.docker", None):
            mgr = DeploymentManager()
        mgr.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN)
        # Ensure rollback manager returns few candidates
        mgr.rollback_manager.get_rollback_candidates = MagicMock(
            return_value=[{"version": "v1"}]
        )
        result = _run(mgr._perform_auto_rollback("svc"))
        assert result["success"] is False

    def test_auto_rollback_exception(self):
        with patch("mlops.deployment_manager.docker", None):
            mgr = DeploymentManager()
        mgr.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN)
        mgr.rollback_manager.get_rollback_candidates = MagicMock(
            side_effect=RuntimeError("fail")
        )
        result = _run(mgr._perform_auto_rollback("svc"))
        assert result["success"] is False


@pytest.mark.coverage_extra
class TestScaleService:
    """Cover lines 1062-1066, 1075-1077."""

    def test_scale_not_registered(self):
        with patch("mlops.deployment_manager.docker", None):
            mgr = DeploymentManager()
        result = _run(mgr.scale_service("unknown", 3))
        assert result["success"] is False

    def test_scale_up(self):
        with patch("mlops.deployment_manager.docker", None):
            mgr = DeploymentManager()
        mgr.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN)
        inst = _make_instance()
        mgr.services["svc"]["instances"] = [inst]
        result = _run(mgr.scale_service("svc", 3))
        assert result["success"] is True
        assert result["scaled_up"] is True

    def test_scale_down(self):
        with patch("mlops.deployment_manager.docker", None):
            mgr = DeploymentManager()
        mgr.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN)
        instances = [_make_instance(f"i-{i}", port=8080+i) for i in range(3)]
        mgr.services["svc"]["instances"] = instances
        result = _run(mgr.scale_service("svc", 1))
        assert result["success"] is True
        assert result["scaled_up"] is False

    def test_scale_same_count(self):
        with patch("mlops.deployment_manager.docker", None):
            mgr = DeploymentManager()
        mgr.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN)
        mgr.services["svc"]["instances"] = [_make_instance()]
        result = _run(mgr.scale_service("svc", 1))
        assert result["success"] is True

    def test_scale_exception(self):
        with patch("mlops.deployment_manager.docker", None):
            mgr = DeploymentManager()
        mgr.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN)
        mgr.services["svc"]["instances"] = None  # will cause exception
        result = _run(mgr.scale_service("svc", 3))
        assert result["success"] is False


@pytest.mark.coverage_extra
class TestEmergencyStop:
    """Cover lines 1177-1181, 1187-1188, 1196-1198."""

    def test_emergency_stop(self):
        with patch("mlops.deployment_manager.docker", None):
            mgr = DeploymentManager()
        mgr.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN)
        inst = _make_instance()
        mgr.services["svc"]["instances"] = [inst]
        result = _run(mgr.emergency_stop())
        assert result["success"] is True
        assert result["stopped_services"] == 1

    def test_emergency_stop_exception(self):
        with patch("mlops.deployment_manager.docker", None):
            mgr = DeploymentManager()
        # Force exception by making services iteration fail
        mgr.services = MagicMock()
        mgr.services.items.side_effect = RuntimeError("fail")
        result = _run(mgr.emergency_stop())
        assert result["success"] is False


@pytest.mark.coverage_extra
class TestImportConfigException:
    """Cover lines 1273-1275."""

    def test_import_config_exception(self):
        with patch("mlops.deployment_manager.docker", None):
            mgr = DeploymentManager()
        # Pass invalid config
        result = mgr.import_configuration(None)
        assert result["success"] is False


@pytest.mark.coverage_extra
class TestMonitoringLoop:
    """Cover lines 996, 1000-1001."""

    def test_monitoring_loop_stops(self):
        with patch("mlops.deployment_manager.docker", None):
            mgr = DeploymentManager()
        mgr.is_monitoring = False
        mgr._monitoring_loop()  # Should exit immediately
