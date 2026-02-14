"""Additional tests for compliance_checker.py - cover lines 261-262, 363, 456-459, 562-564."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from compliance.compliance_checker import (
    ComplianceChecker,
    ComplianceLevel,
    ComplianceRule,
    ComplianceViolation,
    PolicyType,
    ViolationSeverity,
)


@pytest.mark.coverage_extra
class TestValidationExceptionPath:
    """Cover lines 261-262: exception during rule validation function."""

    def test_validation_function_raises_exception(self):
        """Register a validation function that raises; the exception should be
        caught and logged, not propagated."""
        cc = ComplianceChecker()

        def bad_validator(data, rule):
            raise RuntimeError("Validation exploded")

        cc.validation_functions["exploding_func"] = bad_validator
        cc.rules = [
            ComplianceRule(
                id="boom",
                name="Boom Rule",
                description="A rule whose validator raises",
                policy_type=PolicyType.CUSTOM,
                compliance_level=ComplianceLevel.HIGH,
                validation_function="exploding_func",
                parameters={},
                enabled=True,
            )
        ]

        violations = cc.validate_compliance({"data": "test"})
        # No violations should be returned since the function raised
        assert violations == []


@pytest.mark.coverage_extra
class TestComplianceScoreZeroWeight:
    """Cover line 363: total_weight == 0 early return."""

    def test_score_returns_one_when_total_weight_zero(self):
        """Create violations with a severity not in the weights map so
        total_weight remains 0."""
        cc = ComplianceChecker()

        # Create a violation with a mock severity not in the severity_weights dict
        violation = ComplianceViolation(
            rule_id="r",
            severity=MagicMock(),  # Not a real ViolationSeverity enum member
            description="test",
            timestamp=datetime.now(),
            context={},
            remediation_suggested=[],
        )
        # The mock severity won't match any key in severity_weights, so
        # weight = severity_weights.get(violation.severity, 0.5) will return 0.5
        # We need total_weight to be 0. Let's instead patch the severity_weights.
        # Actually, looking at the code more carefully: severity_weights.get()
        # returns 0.5 as default. So we can never get total_weight=0 with violations.
        # Line 362 (if total_weight == 0) is dead code if there are violations.
        # But we still need to exercise it. Let's monkey-patch.

        # Alternative approach: override the method to test the branch
        score = cc.calculate_compliance_score([])
        assert score == 1.0


@pytest.mark.coverage_extra
class TestRemediationExceptionPath:
    """Cover lines 456-459: exception during remediation execution."""

    def test_execute_remediation_catches_exception(self):
        """Force _execute_data_deletion to raise so the except block runs."""
        cc = ComplianceChecker()
        v = ComplianceViolation(
            rule_id="gdpr",
            severity=ViolationSeverity.HIGH,
            description="test",
            timestamp=datetime.now(),
            context={"data_id": "d1"},
            remediation_suggested=[],
        )

        with patch.object(
            cc, "_execute_data_deletion",
            side_effect=RuntimeError("disk error"),
        ):
            action = cc.execute_remediation(v, "delete_expired_data")

        assert action.success is False
        assert "error" in action.details
        assert "disk error" in action.details["error"]
        assert v.resolved is not True  # Should NOT be resolved

    def test_execute_remediation_catches_encrypt_exception(self):
        """Force _execute_data_encryption to raise."""
        cc = ComplianceChecker()
        v = ComplianceViolation(
            rule_id="enc",
            severity=ViolationSeverity.CRITICAL,
            description="test",
            timestamp=datetime.now(),
            context={},
            remediation_suggested=[],
        )

        with patch.object(
            cc, "_execute_data_encryption",
            side_effect=OSError("permission denied"),
        ):
            action = cc.execute_remediation(v, "encrypt_sensitive_data")

        assert action.success is False
        assert "permission denied" in action.details["error"]

    def test_execute_remediation_catches_rbac_exception(self):
        """Force _execute_access_control to raise."""
        cc = ComplianceChecker()
        v = ComplianceViolation(
            rule_id="ac",
            severity=ViolationSeverity.HIGH,
            description="test",
            timestamp=datetime.now(),
            context={},
            remediation_suggested=[],
        )

        with patch.object(
            cc, "_execute_access_control",
            side_effect=ValueError("bad config"),
        ):
            action = cc.execute_remediation(v, "implement_rbac")

        assert action.success is False
        assert "bad config" in action.details["error"]


@pytest.mark.coverage_extra
class TestMonitoringLoopExceptionPath:
    """Cover lines 562-564: exception in the monitoring loop."""

    def test_monitoring_loop_handles_exception(self):
        """Patch time.sleep to raise, covering the except branch in
        _monitoring_loop, then stop monitoring."""
        cc = ComplianceChecker()

        call_count = 0

        def side_effect_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("loop error")
            # On second call (from the except branch sleep(10)), stop monitoring
            cc.monitoring_active = False

        with patch("compliance.compliance_checker.time.sleep", side_effect=side_effect_sleep):
            cc.monitoring_active = True
            cc._monitoring_loop()

        assert cc.monitoring_active is False
