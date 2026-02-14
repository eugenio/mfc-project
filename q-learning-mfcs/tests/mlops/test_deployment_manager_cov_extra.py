"""Extra coverage tests for deployment_manager.py.

Targets remaining uncovered lines: 192, 715-717, 757, 762,
776-778, 816, 830-832, 850-860, 870-872, 900-935, 952-964,
996, 1000-1001, 1062-1066, 1177-1181, 1187-1188.
"""
import asyncio
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from mlops.deployment_manager import (
    DeploymentManager,
    DeploymentStatus,
    DeploymentStrategy,
    LoadBalancer,
    ServiceInstance,
)


def _run(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())


def _inst(iid="i-001", svc="svc", ver="v1", port=8080, **kw):
    return ServiceInstance(
        instance_id=iid, service_name=svc, version=ver,
        host="127.0.0.1", port=port, **kw,
    )


def _mgr():
    with patch("mlops.deployment_manager.docker", None):
        return DeploymentManager()


class _ACM:
    """Async context manager helper."""
    def __init__(self, val):
        self._val = val

    async def __aenter__(self):
        return self._val

    async def __aexit__(self, *a):
        pass


@pytest.mark.coverage_extra
class TestWeightedFallback:
    def test_fallback_line(self):
        lb = LoadBalancer(strategy="weighted")
        i1 = _inst("a", health_score=0.5, status=DeploymentStatus.HEALTHY)
        i2 = _inst("b", port=8081, health_score=0.5, status=DeploymentStatus.HEALTHY)
        lb.add_instance(i1)
        lb.add_instance(i2)
        with patch("mlops.deployment_manager.random") as mr:
            mr.random.return_value = 1.0 + 1e-10
            sel = lb.select_instance()
        assert sel is not None
        assert sel.instance_id == "b"


@pytest.mark.coverage_extra
class TestDeployServiceException:
    def test_exception_during_deploy(self):
        mgr = _mgr()
        mgr.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN)
        mgr.services["svc"]["instances"] = "bad"
        r = _run(mgr.deploy_service("svc", "v2", "img"))
        assert r["success"] is False
        assert "deployment_id" in r


@pytest.mark.coverage_extra
class TestBlueGreenDetails:
    def test_switch_fails(self):
        mgr = _mgr()
        mgr.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN, min_instances=1)
        orig = mgr._deploy_instances

        async def deploy_unhealthy(instances):
            await orig(instances)
            for i in instances:
                i.status = DeploymentStatus.PENDING
            return True

        mgr._deploy_instances = deploy_unhealthy
        r = _run(mgr._deploy_blue_green("svc", "v2", "img", 1))
        assert r["success"] is False

    def test_full_success(self):
        mgr = _mgr()
        mgr.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN, min_instances=1)
        r = _run(mgr._deploy_blue_green("svc", "v2", "img", 2))
        assert r["success"] is True
        assert r["active_environment"] == "green"
        assert mgr.services["svc"]["current_version"] == "v2"


@pytest.mark.coverage_extra
class TestBlueGreenException:
    def test_exception(self):
        mgr = _mgr()
        mgr.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN)

        async def fail(instances):
            raise RuntimeError("boom")

        mgr._deploy_instances = fail
        r = _run(mgr._deploy_blue_green("svc", "v2", "img", 1))
        assert r["success"] is False

    def test_exception_in_service_update(self):
        """Hit lines 776-778: exception after deploy+switch succeed,
        during service config update."""
        mgr = _mgr()
        mgr.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN, min_instances=1)
        # Remove load_balancers entry so the lb lookup at line 770 raises KeyError
        del mgr.load_balancers["svc"]
        r = _run(mgr._deploy_blue_green("svc", "v2", "img", 1))
        assert r["success"] is False


@pytest.mark.coverage_extra
class TestCanaryFailure:
    def test_deploy_fail(self):
        mgr = _mgr()
        mgr.register_service("svc", "v1", DeploymentStrategy.CANARY, min_instances=2)

        async def fail(instances):
            return False

        mgr._deploy_instances = fail
        r = _run(mgr._deploy_canary("svc", "v2", "img", 10.0))
        assert r["success"] is False

    def test_canary_success(self):
        mgr = _mgr()
        mgr.register_service("svc", "v1", DeploymentStrategy.CANARY, min_instances=2)
        r = _run(mgr._deploy_canary("svc", "v2", "img", 10.0))
        assert r["success"] is True


@pytest.mark.coverage_extra
class TestCanaryException:
    def test_exception(self):
        mgr = _mgr()
        mgr.register_service("svc", "v1", DeploymentStrategy.CANARY)

        async def fail(instances):
            raise RuntimeError("boom")

        mgr._deploy_instances = fail
        r = _run(mgr._deploy_canary("svc", "v2", "img", 5.0))
        assert r["success"] is False

    def test_exception_in_service_update(self):
        """Hit lines 830-832: exception after canary deploy succeeds,
        during service config update (lb lookup fails)."""
        mgr = _mgr()
        mgr.register_service("svc", "v1", DeploymentStrategy.CANARY, min_instances=2)
        del mgr.load_balancers["svc"]
        r = _run(mgr._deploy_canary("svc", "v2", "img", 10.0))
        assert r["success"] is False


@pytest.mark.coverage_extra
class TestDeployDocker:
    def test_docker_success(self):
        mgr = _mgr()
        mc = MagicMock()
        mc.id = "abcdef123456deadbeef"
        md = MagicMock()
        md.containers.run.return_value = mc
        mgr.docker_client = md
        inst = _inst()
        r = _run(mgr._deploy_instances([inst]))
        assert r is True
        assert inst.container_id == "abcdef123456"

    def test_docker_fallback(self):
        mgr = _mgr()
        md = MagicMock()
        md.containers.run.side_effect = RuntimeError("err")
        mgr.docker_client = md
        inst = _inst()
        r = _run(mgr._deploy_instances([inst]))
        assert r is True
        assert inst.container_id.startswith("mock-")


@pytest.mark.coverage_extra
class TestDeployInstancesException:
    def test_general_exception(self):
        mgr = _mgr()
        r = _run(mgr._deploy_instances(None))
        assert r is False


@pytest.mark.coverage_extra
class TestHealthAiohttp:
    def _mock_aiohttp(self, status=200, side_effect=None):
        resp = MagicMock()
        resp.status = status
        sess = MagicMock()
        if side_effect:
            sess.get.side_effect = side_effect
        else:
            sess.get.return_value = _ACM(resp)
        m = MagicMock()
        m.ClientSession.return_value = _ACM(sess)
        m.ClientTimeout = MagicMock()
        return m

    def test_success(self):
        mgr = _mgr()
        mgr.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN)
        inst = _inst()
        inst.status = DeploymentStatus.HEALTHY
        mgr.services["svc"]["instances"] = [inst]
        with patch("mlops.deployment_manager.aiohttp", self._mock_aiohttp(200)):
            results = _run(mgr._perform_health_checks("svc"))
        assert len(results) == 1
        assert results[0].is_healthy is True

    def test_failure(self):
        mgr = _mgr()
        mgr.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN)
        mgr.services["svc"]["instances"] = [_inst()]
        with patch("mlops.deployment_manager.aiohttp",
                   self._mock_aiohttp(side_effect=ConnectionError("x"))):
            results = _run(mgr._perform_health_checks("svc"))
        assert len(results) == 1
        assert results[0].is_healthy is False
        assert results[0].status_code == 500

    def test_non_200(self):
        mgr = _mgr()
        mgr.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN)
        mgr.services["svc"]["instances"] = [_inst()]
        with patch("mlops.deployment_manager.aiohttp", self._mock_aiohttp(503)):
            results = _run(mgr._perform_health_checks("svc"))
        assert len(results) == 1
        assert results[0].is_healthy is False


@pytest.mark.coverage_extra
class TestAutoRollbackSuccess:
    def test_with_candidates(self):
        mgr = _mgr()
        mgr.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN)
        mgr.rollback_manager.save_deployment_snapshot("svc", {"version": "v0", "instances": []})
        mgr.rollback_manager.save_deployment_snapshot("svc", {"version": "v1", "instances": []})
        r = _run(mgr._perform_auto_rollback("svc"))
        assert r["success"] is True
        assert r["target_version"] == "v0"

    def test_exception(self):
        mgr = _mgr()
        mgr.rollback_manager.get_rollback_candidates = MagicMock(side_effect=RuntimeError("x"))
        r = _run(mgr._perform_auto_rollback("svc"))
        assert r["success"] is False


@pytest.mark.coverage_extra
class TestMonitoringLoopCov:
    def test_runs_one_iteration(self):
        mgr = _mgr()
        mgr.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN)
        mgr.services["svc"]["instances"] = [_inst()]
        mgr.health_check_interval = 0.01
        mgr.is_monitoring = True

        def stop(secs):
            mgr.is_monitoring = False
            time.sleep(0.001)

        with patch("mlops.deployment_manager.time.sleep", side_effect=stop):
            with patch("mlops.deployment_manager.aiohttp", None):
                mgr._monitoring_loop()

    def test_handles_exception(self):
        mgr = _mgr()
        mgr.health_check_interval = 0.01
        mgr.is_monitoring = True

        call_count = 0
        original_keys = mgr.services.keys

        def keys_then_stop():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                mgr.is_monitoring = False
            raise RuntimeError("x")

        mgr.services = MagicMock()
        mgr.services.keys.side_effect = keys_then_stop

        mgr._monitoring_loop()
        assert call_count >= 2


@pytest.mark.coverage_extra
class TestScaleDownDocker:
    def test_stops_containers(self):
        mgr = _mgr()
        mgr.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN)
        i1 = _inst("i1", port=8080)
        i1.container_id = "c1"
        i1.status = DeploymentStatus.HEALTHY
        i2 = _inst("i2", port=8081)
        i2.container_id = "c2"
        i2.status = DeploymentStatus.HEALTHY
        mgr.services["svc"]["instances"] = [i1, i2]
        mgr.load_balancers["svc"].add_instance(i1)
        mgr.load_balancers["svc"].add_instance(i2)
        md = MagicMock()
        mc = MagicMock()
        md.containers.get.return_value = mc
        mgr.docker_client = md
        r = _run(mgr.scale_service("svc", 1))
        assert r["success"] is True
        mc.stop.assert_called()

    def test_docker_stop_fails(self):
        mgr = _mgr()
        mgr.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN)
        i1 = _inst("i1", port=8080)
        i1.container_id = "c1"
        i2 = _inst("i2", port=8081)
        i2.container_id = "c2"
        mgr.services["svc"]["instances"] = [i1, i2]
        mgr.load_balancers["svc"].add_instance(i1)
        mgr.load_balancers["svc"].add_instance(i2)
        md = MagicMock()
        md.containers.get.side_effect = RuntimeError("x")
        mgr.docker_client = md
        r = _run(mgr.scale_service("svc", 1))
        assert r["success"] is True


@pytest.mark.coverage_extra
class TestEmergencyStopDocker:
    def test_with_docker(self):
        mgr = _mgr()
        mgr.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN)
        inst = _inst()
        inst.container_id = "cid"
        mgr.services["svc"]["instances"] = [inst]
        mc = MagicMock()
        md = MagicMock()
        md.containers.get.return_value = mc
        mgr.docker_client = md
        r = _run(mgr.emergency_stop())
        assert r["success"] is True
        mc.stop.assert_called_once()

    def test_docker_stop_fails(self):
        mgr = _mgr()
        mgr.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN)
        inst = _inst()
        inst.container_id = "cid"
        mgr.services["svc"]["instances"] = [inst]
        md = MagicMock()
        md.containers.get.side_effect = RuntimeError("x")
        mgr.docker_client = md
        r = _run(mgr.emergency_stop())
        assert r["success"] is True
        assert len(r["errors"]) > 0


@pytest.mark.coverage_extra
class TestEmergencyStopServiceEx:
    def test_service_exception(self):
        mgr = _mgr()
        mgr.register_service("svc", "v1", DeploymentStrategy.BLUE_GREEN)
        orig = mgr.services["svc"]
        bad = MagicMock(wraps=orig)
        bad.get = MagicMock(side_effect=RuntimeError("bad"))
        mgr.services["svc"] = bad
        r = _run(mgr.emergency_stop())
        assert r["success"] is True
        assert len(r["errors"]) > 0


@pytest.mark.coverage_extra
class TestDeployAutoRollback:
    def test_triggers_auto_rollback(self):
        mgr = _mgr()
        mgr.register_service("svc", "v1", DeploymentStrategy.ROLLING, auto_rollback=True)
        mgr.rollback_manager.save_deployment_snapshot("svc", {"version": "v0", "instances": []})
        r = _run(mgr.deploy_service("svc", "v2", "img"))
        assert r["success"] is False
        assert "rollback" in r

    def test_no_auto_rollback(self):
        mgr = _mgr()
        mgr.register_service("svc", "v1", DeploymentStrategy.ROLLING, auto_rollback=False)
        r = _run(mgr.deploy_service("svc", "v2", "img"))
        assert r["success"] is False
        assert "rollback" not in r
