"""Extra coverage tests for monitoring/observability_manager.py - targeting 99%+.

Covers missing lines 458-470: _evaluate_alert_conditions when the
performance_metrics import succeeds and when a generic exception occurs.
"""
import importlib.util
import os
import sys
import types
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

# Snapshot sys.modules before mocking
_original_modules = dict(sys.modules)

_src = os.path.join(os.path.dirname(__file__), "..", "..", "src")
_spec = importlib.util.spec_from_file_location(
    "monitoring.observability_manager",
    os.path.join(_src, "monitoring", "observability_manager.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["monitoring.observability_manager"] = _mod
_spec.loader.exec_module(_mod)

ObservabilityManager = _mod.ObservabilityManager
AlertCondition = _mod.AlertCondition
AlertSeverity = _mod.AlertSeverity

# --- Restore sys.modules to prevent mock leakage ---
for _mock_key in list(sys.modules):
    if _mock_key not in _original_modules:
        if isinstance(sys.modules[_mock_key], (MagicMock, types.ModuleType)):
            del sys.modules[_mock_key]
    elif isinstance(sys.modules[_mock_key], MagicMock):
        sys.modules[_mock_key] = _original_modules[_mock_key]


@pytest.mark.coverage_extra
class TestEvaluateAlertConditionsImportSuccess:
    """Cover lines 458-464: successful import of performance_metrics."""

    def test_evaluate_with_successful_import(self):
        """Patch the relative import so it succeeds and exercises the loop."""
        mgr = ObservabilityManager()

        # Create a mock collector
        mock_collector = MagicMock()
        mock_collector.get_metric_values.return_value = []

        # Create a mock module that provides get_default_collector
        mock_perf_module = types.ModuleType("performance.performance_metrics")
        mock_perf_module.get_default_collector = MagicMock(
            return_value=mock_collector,
        )

        # We need to make the relative import succeed.
        # The code does: from ..performance.performance_metrics import get_default_collector
        # That means it needs to find the module at the right relative path.
        # We can patch __import__ to intercept this.

        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def patched_import(name, *args, **kwargs):
            # The relative import resolves to something like
            # "performance.performance_metrics" or with the parent package
            if "performance_metrics" in name or (
                args and len(args) >= 3 and args[2] and
                any("get_default_collector" in str(x) for x in args[2] if isinstance(x, str))
            ):
                return mock_perf_module
            return original_import(name, *args, **kwargs)

        # Alternative approach: directly call _evaluate_alert_conditions
        # with the import mocked at module level
        # The code does:
        #   from ..performance.performance_metrics import get_default_collector
        # This is a relative import. Let's mock it by injecting into sys.modules.

        # First figure out what the full module path would be
        # The module is monitoring.observability_manager
        # ..performance.performance_metrics -> performance.performance_metrics
        # But that won't work directly. Let's just test via direct method patching.

        # Simpler approach: replace _evaluate_alert_conditions temporarily
        # to exercise the actual code path with a mock import

        # Simplest approach: patch builtins.__import__
        import builtins

        real_import = builtins.__import__

        def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level > 0 and "performance_metrics" in str(fromlist or []):
                # This is the relative import we want to intercept
                m = types.ModuleType("performance.performance_metrics")
                m.get_default_collector = MagicMock(return_value=mock_collector)
                return m
            if "performance" in name and "metrics" in name:
                m = types.ModuleType(name)
                m.get_default_collector = MagicMock(return_value=mock_collector)
                return m
            return real_import(name, globals, locals, fromlist, level)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            # Add a disabled condition to test that branch too
            mgr.alert_conditions["disabled_cond"] = AlertCondition(
                name="disabled_cond",
                metric_name="test.metric",
                threshold=50.0,
                comparison="gt",
                duration_seconds=10,
                severity=AlertSeverity.WARNING,
                enabled=False,
            )
            mgr._evaluate_alert_conditions()

        # Verify _check_condition was called for enabled conditions
        # (the default ones are all enabled)
        assert mock_collector.get_metric_values.called

    def test_evaluate_with_generic_exception(self):
        """Cover lines 469-470: generic (non-ImportError) exception path."""
        mgr = ObservabilityManager()

        import builtins

        real_import = builtins.__import__

        def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level > 0 and "performance_metrics" in str(fromlist or []):
                m = types.ModuleType("performance.performance_metrics")
                m.get_default_collector = MagicMock(
                    side_effect=RuntimeError("collector unavailable"),
                )
                return m
            if "performance" in name and "metrics" in name:
                m = types.ModuleType(name)
                m.get_default_collector = MagicMock(
                    side_effect=RuntimeError("collector unavailable"),
                )
                return m
            return real_import(name, globals, locals, fromlist, level)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            # Should not raise - catches Exception
            mgr._evaluate_alert_conditions()
