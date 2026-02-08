"""Tests for deployment_manager module - Part 1: Core classes."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from mlops.deployment_manager import (
    BlueGreenDeployment,
    CanaryDeployment,
    DeploymentStatus,
    DeploymentStrategy,
    HealthCheckResult,
    LoadBalancer,
    ServiceInstance,
)


class TestServiceInstance:
    def test_create_instance(self):
        inst = ServiceInstance(
            instance_id="inst-001",
            service_name="svc",
            version="v1",
            host="127.0.0.1",
            port=8080,
        )
        assert inst.instance_id == "inst-001"
        assert inst.status == DeploymentStatus.PENDING
        assert inst.health_score == 0.0
        assert inst.active_connections == 0
        assert inst.container_id is None
        assert inst.last_health_check is None

    def test_endpoint_url_http(self):
        inst = ServiceInstance("id", "svc", "v1", "localhost", 8080)
        assert inst.endpoint_url == "http://localhost:8080"

    def test_endpoint_url_https(self):
        inst = ServiceInstance("id", "svc", "v1", "localhost", 443, ssl_enabled=True)
        assert inst.endpoint_url == "https://localhost:443"

    def test_update_health(self):
        inst = ServiceInstance("id", "svc", "v1", "localhost", 8080)
        inst.update_health(0.95, DeploymentStatus.HEALTHY)
        assert inst.health_score == 0.95
        assert inst.status == DeploymentStatus.HEALTHY
        assert inst.last_health_check is not None


class TestHealthCheckResult:
    def test_health_score_unhealthy(self):
        r = HealthCheckResult("id", is_healthy=False, response_time_ms=100, status_code=500)
        assert r.health_score == 0.0

    def test_health_score_200_fast(self):
        r = HealthCheckResult("id", is_healthy=True, response_time_ms=50, status_code=200)
        assert r.health_score == pytest.approx(0.95, abs=0.01)

    def test_health_score_200_slow(self):
        r = HealthCheckResult("id", is_healthy=True, response_time_ms=900, status_code=200)
        assert r.health_score == pytest.approx(0.1, abs=0.01)

    def test_health_score_2xx_non_200(self):
        r = HealthCheckResult("id", is_healthy=True, response_time_ms=50, status_code=201)
        assert r.health_score == pytest.approx(0.9, abs=0.05)

    def test_health_score_3xx(self):
        r = HealthCheckResult("id", is_healthy=True, response_time_ms=50, status_code=301)
        assert r.health_score == pytest.approx(0.7, abs=0.05)

    def test_health_score_4xx(self):
        r = HealthCheckResult("id", is_healthy=True, response_time_ms=50, status_code=404)
        assert r.health_score == 0.0

    def test_health_score_over_1000ms(self):
        r = HealthCheckResult("id", is_healthy=True, response_time_ms=1500, status_code=200)
        assert r.health_score == 0.0


class TestLoadBalancer:
    def test_add_and_remove_instance(self):
        lb = LoadBalancer()
        inst = ServiceInstance("id1", "svc", "v1", "localhost", 8080)
        lb.add_instance(inst)
        assert len(lb.instances) == 1
        lb.remove_instance("id1")
        assert len(lb.instances) == 0

    def test_select_instance_no_healthy(self):
        lb = LoadBalancer()
        inst = ServiceInstance("id1", "svc", "v1", "localhost", 8080)
        lb.add_instance(inst)
        assert lb.select_instance() is None

    def test_round_robin_selection(self):
        lb = LoadBalancer(strategy="round_robin")
        i1 = ServiceInstance("id1", "svc", "v1", "localhost", 8080,
                             status=DeploymentStatus.HEALTHY)
        i2 = ServiceInstance("id2", "svc", "v1", "localhost", 8081,
                             status=DeploymentStatus.HEALTHY)
        lb.add_instance(i1)
        lb.add_instance(i2)
        first = lb.select_instance()
        second = lb.select_instance()
        assert first.instance_id != second.instance_id

    def test_weighted_selection(self):
        lb = LoadBalancer(strategy="weighted")
        i1 = ServiceInstance("id1", "svc", "v1", "localhost", 8080,
                             status=DeploymentStatus.HEALTHY, health_score=1.0)
        lb.add_instance(i1)
        selected = lb.select_instance()
        assert selected is not None

    def test_weighted_selection_zero_weight(self):
        lb = LoadBalancer(strategy="weighted")
        i1 = ServiceInstance("id1", "svc", "v1", "localhost", 8080,
                             status=DeploymentStatus.HEALTHY, health_score=0.0)
        lb.add_instance(i1)
        selected = lb.select_instance()
        assert selected is not None

    def test_least_connections_selection(self):
        lb = LoadBalancer(strategy="least_connections")
        i1 = ServiceInstance("id1", "svc", "v1", "localhost", 8080,
                             status=DeploymentStatus.HEALTHY, active_connections=10)
        i2 = ServiceInstance("id2", "svc", "v1", "localhost", 8081,
                             status=DeploymentStatus.HEALTHY, active_connections=2)
        lb.add_instance(i1)
        lb.add_instance(i2)
        selected = lb.select_instance()
        assert selected.instance_id == "id2"

    def test_unknown_strategy_fallback(self):
        lb = LoadBalancer(strategy="unknown")
        i1 = ServiceInstance("id1", "svc", "v1", "localhost", 8080,
                             status=DeploymentStatus.HEALTHY)
        lb.add_instance(i1)
        assert lb.select_instance() is not None

    def test_round_robin_empty(self):
        lb = LoadBalancer()
        assert lb._round_robin_selection([]) is None

    def test_weighted_empty(self):
        lb = LoadBalancer()
        assert lb._weighted_selection([]) is None

    def test_least_connections_empty(self):
        lb = LoadBalancer()
        assert lb._least_connections_selection([]) is None

    def test_get_health_status_empty(self):
        lb = LoadBalancer()
        status = lb.get_health_status()
        assert status["total_instances"] == 0
        assert status["average_health_score"] == 0.0

    def test_get_health_status_with_instances(self):
        lb = LoadBalancer()
        i1 = ServiceInstance("id1", "svc", "v1", "localhost", 8080,
                             status=DeploymentStatus.HEALTHY, health_score=0.8)
        lb.add_instance(i1)
        status = lb.get_health_status()
        assert status["total_instances"] == 1
        assert status["healthy_instances"] == 1
        assert status["average_health_score"] == 0.8


class TestBlueGreenDeployment:
    def test_init(self):
        bg = BlueGreenDeployment("svc", "v1", "v2")
        assert bg.active_environment == "blue"
        assert bg.deployment_status == DeploymentStatus.PENDING

    def test_deploy_green_success(self):
        bg = BlueGreenDeployment("svc", "v1", "v2")
        instances = [ServiceInstance("id1", "svc", "v2", "localhost", 8080)]

        async def deploy(insts):
            return True

        result = asyncio.get_event_loop().run_until_complete(
            bg.deploy_green_environment(instances, deploy)
        )
        assert result is True
        assert bg.deployment_status == DeploymentStatus.HEALTHY

    def test_deploy_green_failure(self):
        bg = BlueGreenDeployment("svc", "v1", "v2")

        async def deploy(insts):
            return False

        result = asyncio.get_event_loop().run_until_complete(
            bg.deploy_green_environment([], deploy)
        )
        assert result is False
        assert bg.deployment_status == DeploymentStatus.FAILED

    def test_deploy_green_exception(self):
        bg = BlueGreenDeployment("svc", "v1", "v2")

        async def deploy(insts):
            raise RuntimeError("boom")

        result = asyncio.get_event_loop().run_until_complete(
            bg.deploy_green_environment([], deploy)
        )
        assert result is False

    def test_deploy_green_with_health_checker_unhealthy(self):
        bg = BlueGreenDeployment("svc", "v1", "v2")
        instances = [ServiceInstance("id1", "svc", "v2", "localhost", 8080)]

        health_checker = MagicMock()
        health_result = MagicMock()
        health_result.is_healthy = False
        health_checker.check_instance_health = AsyncMock(return_value=health_result)
        bg.health_checker = health_checker

        async def deploy(insts):
            return True

        result = asyncio.get_event_loop().run_until_complete(
            bg.deploy_green_environment(instances, deploy)
        )
        assert result is False

    def test_switch_to_green_success(self):
        bg = BlueGreenDeployment("svc", "v1", "v2")
        inst = ServiceInstance("id1", "svc", "v2", "localhost", 8080,
                               status=DeploymentStatus.HEALTHY)
        bg.green_instances = [inst]
        result = asyncio.get_event_loop().run_until_complete(bg.switch_to_green())
        assert result is True
        assert bg.active_environment == "green"

    def test_switch_to_green_no_instances(self):
        bg = BlueGreenDeployment("svc", "v1", "v2")
        result = asyncio.get_event_loop().run_until_complete(bg.switch_to_green())
        assert result is False

    def test_switch_to_green_unhealthy(self):
        bg = BlueGreenDeployment("svc", "v1", "v2")
        inst = ServiceInstance("id1", "svc", "v2", "localhost", 8080,
                               status=DeploymentStatus.PENDING)
        bg.green_instances = [inst]
        result = asyncio.get_event_loop().run_until_complete(bg.switch_to_green())
        assert result is False

    def test_rollback_to_blue_success(self):
        bg = BlueGreenDeployment("svc", "v1", "v2")
        bg.blue_instances = [ServiceInstance("id1", "svc", "v1", "localhost", 8080)]
        bg.active_environment = "green"
        result = asyncio.get_event_loop().run_until_complete(bg.rollback_to_blue())
        assert result is True
        assert bg.active_environment == "blue"

    def test_rollback_to_blue_no_instances(self):
        bg = BlueGreenDeployment("svc", "v1", "v2")
        result = asyncio.get_event_loop().run_until_complete(bg.rollback_to_blue())
        assert result is False

    def test_get_active_instances_blue(self):
        bg = BlueGreenDeployment("svc", "v1", "v2")
        blue_inst = ServiceInstance("b1", "svc", "v1", "localhost", 8080)
        bg.blue_instances = [blue_inst]
        assert bg.get_active_instances() == [blue_inst]

    def test_get_active_instances_green(self):
        bg = BlueGreenDeployment("svc", "v1", "v2")
        green_inst = ServiceInstance("g1", "svc", "v2", "localhost", 8080)
        bg.green_instances = [green_inst]
        bg.active_environment = "green"
        assert bg.get_active_instances() == [green_inst]


class TestCanaryDeployment:
    def test_deploy_canary_success(self):
        cd = CanaryDeployment("svc", "v1", "v2", 5.0)
        instances = [ServiceInstance("id1", "svc", "v2", "localhost", 8090)]

        async def deploy(insts):
            return True

        result = asyncio.get_event_loop().run_until_complete(
            cd.deploy_canary_instances(instances, deploy)
        )
        assert result is True
        assert cd.deployment_status == DeploymentStatus.HEALTHY

    def test_deploy_canary_failure(self):
        cd = CanaryDeployment("svc", "v1", "v2")

        async def deploy(insts):
            return False

        result = asyncio.get_event_loop().run_until_complete(
            cd.deploy_canary_instances([], deploy)
        )
        assert result is False

    def test_deploy_canary_exception(self):
        cd = CanaryDeployment("svc", "v1", "v2")

        async def deploy(insts):
            raise RuntimeError("err")

        result = asyncio.get_event_loop().run_until_complete(
            cd.deploy_canary_instances([], deploy)
        )
        assert result is False

    def test_increase_traffic_success(self):
        cd = CanaryDeployment("svc", "v1", "v2", 5.0)
        inst = ServiceInstance("id1", "svc", "v2", "localhost", 8090,
                               status=DeploymentStatus.HEALTHY)
        cd.canary_instances = [inst]
        result = asyncio.get_event_loop().run_until_complete(
            cd.increase_canary_traffic(25.0)
        )
        assert result is True
        assert cd.canary_percentage == 25.0

    def test_increase_traffic_invalid_percentage(self):
        cd = CanaryDeployment("svc", "v1", "v2")
        result = asyncio.get_event_loop().run_until_complete(
            cd.increase_canary_traffic(-1)
        )
        assert result is False
        result = asyncio.get_event_loop().run_until_complete(
            cd.increase_canary_traffic(101)
        )
        assert result is False

    def test_increase_traffic_no_instances(self):
        cd = CanaryDeployment("svc", "v1", "v2")
        result = asyncio.get_event_loop().run_until_complete(
            cd.increase_canary_traffic(50)
        )
        assert result is False

    def test_increase_traffic_unhealthy_instances(self):
        cd = CanaryDeployment("svc", "v1", "v2")
        inst = ServiceInstance("id1", "svc", "v2", "localhost", 8090,
                               status=DeploymentStatus.PENDING)
        cd.canary_instances = [inst]
        result = asyncio.get_event_loop().run_until_complete(
            cd.increase_canary_traffic(50)
        )
        assert result is False

    def test_promote_canary_success(self):
        cd = CanaryDeployment("svc", "v1", "v2", 100.0)
        inst = ServiceInstance("id1", "svc", "v2", "localhost", 8090)
        cd.canary_instances = [inst]
        result = asyncio.get_event_loop().run_until_complete(
            cd.promote_canary_to_stable()
        )
        assert result is True
        assert cd.stable_version == "v2"
        assert cd.canary_percentage == 0.0

    def test_promote_canary_not_at_100(self):
        cd = CanaryDeployment("svc", "v1", "v2", 50.0)
        cd.canary_instances = [ServiceInstance("id1", "svc", "v2", "localhost", 8090)]
        result = asyncio.get_event_loop().run_until_complete(
            cd.promote_canary_to_stable()
        )
        assert result is False

    def test_promote_canary_no_instances(self):
        cd = CanaryDeployment("svc", "v1", "v2", 100.0)
        result = asyncio.get_event_loop().run_until_complete(
            cd.promote_canary_to_stable()
        )
        assert result is False

    def test_calculate_traffic_distribution(self):
        cd = CanaryDeployment("svc", "v1", "v2", 10.0)
        dist = cd.calculate_traffic_distribution(100)
        assert dist["canary_requests"] == 10
        assert dist["stable_requests"] == 90
        assert dist["total_requests"] == 100

    def test_should_route_to_canary(self):
        cd = CanaryDeployment("svc", "v1", "v2", 100.0)
        assert cd.should_route_to_canary() is True
        cd2 = CanaryDeployment("svc", "v1", "v2", 0.0)
        assert cd2.should_route_to_canary() is False
