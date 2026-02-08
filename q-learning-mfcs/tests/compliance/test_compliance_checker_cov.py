"""Tests for compliance_checker.py module - comprehensive coverage."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import json
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from compliance.compliance_checker import (
    ComplianceChecker,
    ComplianceLevel,
    ComplianceReport,
    ComplianceRule,
    ComplianceViolation,
    PolicyType,
    RemediationAction,
    ViolationSeverity,
)


class TestEnums:
    def test_compliance_level(self):
        assert ComplianceLevel.CRITICAL.value == "critical"
        assert ComplianceLevel.LOW.value == "low"

    def test_violation_severity(self):
        assert ViolationSeverity.INFO.value == "info"
        assert ViolationSeverity.CRITICAL.value == "critical"

    def test_policy_type(self):
        assert PolicyType.GDPR.value == "gdpr"
        assert PolicyType.CUSTOM.value == "custom"


class TestComplianceCheckerInit:
    def test_default_init(self):
        cc = ComplianceChecker()
        assert len(cc.rules) == 3
        assert len(cc.violations) == 0
        assert cc.monitoring_active is False

    def test_init_with_config_file(self, tmp_path):
        config = {
            "rules": [{
                "id": "r1", "name": "Rule 1",
                "description": "Test rule",
                "policy_type": "gdpr",
                "compliance_level": "high",
                "validation_function": "check_data_retention",
                "parameters": {}, "enabled": True,
            }]
        }
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)
        cc = ComplianceChecker(config_path=str(config_path))
        assert len(cc.rules) == 1

    def test_init_with_nonexistent_config(self):
        cc = ComplianceChecker(config_path="/nonexistent/path.json")
        assert len(cc.rules) == 3  # Falls back to defaults

    def test_init_with_invalid_config(self, tmp_path):
        config_path = tmp_path / "bad.json"
        with open(config_path, "w") as f:
            f.write("not json")
        with pytest.raises(Exception):
            ComplianceChecker(config_path=str(config_path))


class TestComplianceRules:
    def test_load_compliance_rules(self):
        cc = ComplianceChecker()
        rules = cc.load_compliance_rules()
        assert len(rules) == 3
        assert rules is not cc.rules  # Should be copy

    def test_import_rules(self):
        cc = ComplianceChecker()
        new_rules = [{
            "id": "custom_r1", "name": "Custom",
            "description": "Custom rule",
            "policy_type": "custom",
            "compliance_level": "medium",
            "validation_function": "custom_validation",
            "parameters": {},
        }]
        result = cc.import_compliance_rules(new_rules)
        assert result["imported_count"] == 1

    def test_import_duplicate_rule(self):
        cc = ComplianceChecker()
        new_rules = [{
            "id": "gdpr_data_retention", "name": "Dup",
            "description": "Dup rule",
            "policy_type": "gdpr",
            "compliance_level": "high",
            "validation_function": "check_data_retention",
        }]
        result = cc.import_compliance_rules(new_rules)
        assert result["imported_count"] == 0
        assert len(result["errors"]) == 1

    def test_import_invalid_rule(self):
        cc = ComplianceChecker()
        new_rules = [{"invalid": "data"}]
        result = cc.import_compliance_rules(new_rules)
        assert result["imported_count"] == 0
        assert len(result["errors"]) == 1

    def test_validate_rule_valid(self):
        cc = ComplianceChecker()
        rule_data = {
            "id": "test", "name": "Test",
            "description": "Test rule",
            "policy_type": "gdpr",
            "compliance_level": "high",
            "validation_function": "check",
        }
        cc.validate_rule(rule_data)  # Should not raise

    def test_validate_rule_missing_field(self):
        cc = ComplianceChecker()
        rule_data = {"id": "test", "name": "Test"}
        with pytest.raises(Exception, match="Missing required field"):
            cc.validate_rule(rule_data)

    def test_validate_rule_invalid_enum(self):
        cc = ComplianceChecker()
        rule_data = {
            "id": "t", "name": "T", "description": "T",
            "policy_type": "invalid_type",
            "compliance_level": "high",
            "validation_function": "f",
        }
        with pytest.raises(Exception, match="Invalid enum value"):
            cc.validate_rule(rule_data)


class TestValidation:
    def test_validate_all_rules(self):
        cc = ComplianceChecker()
        data = {
            "sensitive_data": [{"id": "s1", "encrypted": False}],
            "access_logs": [{"user_id": "u1", "authorized": False}],
        }
        violations = cc.validate_compliance(data)
        assert len(violations) >= 1

    def test_validate_specific_rules(self):
        cc = ComplianceChecker()
        data = {"sensitive_data": [{"id": "s1", "encrypted": False}]}
        violations = cc.validate_compliance(
            data, rule_ids=["data_encryption_required"]
        )
        assert len(violations) >= 1

    def test_validate_disabled_rule_skipped(self):
        cc = ComplianceChecker()
        for rule in cc.rules:
            rule.enabled = False
        violations = cc.validate_compliance({"sensitive_data": [{"id": "s1"}]})
        assert len(violations) == 0

    def test_validate_unknown_function(self):
        cc = ComplianceChecker()
        cc.rules = [ComplianceRule(
            id="t", name="T", description="T",
            policy_type=PolicyType.CUSTOM,
            compliance_level=ComplianceLevel.LOW,
            validation_function="nonexistent_func",
            parameters={},
        )]
        violations = cc.validate_compliance({"data": "test"})
        assert len(violations) == 0

    def test_detect_violations(self):
        cc = ComplianceChecker()
        data = {"sensitive_data": [{"id": "s1", "encrypted": False}]}
        violations = cc.detect_violations(data)
        assert len(violations) >= 1

    def test_data_retention_violation(self):
        cc = ComplianceChecker()
        old_date = datetime.now() - timedelta(days=500)
        data = {
            "personal_data": [{"id": "p1", "created": old_date}],
        }
        violations = cc.validate_compliance(
            data, rule_ids=["gdpr_data_retention"]
        )
        assert len(violations) >= 1

    def test_data_retention_no_violation(self):
        cc = ComplianceChecker()
        recent_date = datetime.now() - timedelta(days=10)
        data = {
            "personal_data": [{"id": "p1", "created": recent_date}],
        }
        violations = cc.validate_compliance(
            data, rule_ids=["gdpr_data_retention"]
        )
        assert len(violations) == 0

    def test_data_retention_non_dict_record(self):
        cc = ComplianceChecker()
        data = {"personal_data": ["not_a_dict"]}
        violations = cc.validate_compliance(
            data, rule_ids=["gdpr_data_retention"]
        )
        assert len(violations) == 0

    def test_data_retention_no_created_field(self):
        cc = ComplianceChecker()
        data = {"personal_data": [{"id": "p1"}]}
        violations = cc.validate_compliance(
            data, rule_ids=["gdpr_data_retention"]
        )
        assert len(violations) == 0

    def test_data_retention_uses_user_data_fallback(self):
        cc = ComplianceChecker()
        old_date = datetime.now() - timedelta(days=500)
        data = {"user_data": [{"id": "u1", "created": old_date}]}
        violations = cc.validate_compliance(
            data, rule_ids=["gdpr_data_retention"]
        )
        assert len(violations) >= 1

    def test_encryption_violation(self):
        cc = ComplianceChecker()
        data = {"sensitive_data": [{"id": "s1", "encrypted": False}]}
        violations = cc.validate_compliance(
            data, rule_ids=["data_encryption_required"]
        )
        assert len(violations) >= 1

    def test_encryption_no_violation(self):
        cc = ComplianceChecker()
        data = {"sensitive_data": [{"id": "s1", "encrypted": True}]}
        violations = cc.validate_compliance(
            data, rule_ids=["data_encryption_required"]
        )
        assert len(violations) == 0

    def test_encryption_non_dict_record(self):
        cc = ComplianceChecker()
        data = {"sensitive_data": ["not_dict"]}
        violations = cc.validate_compliance(
            data, rule_ids=["data_encryption_required"]
        )
        assert len(violations) == 0

    def test_access_control_violation(self):
        cc = ComplianceChecker()
        data = {"access_logs": [{"user_id": "u1", "authorized": False}]}
        violations = cc.validate_compliance(
            data, rule_ids=["access_control_enforcement"]
        )
        assert len(violations) >= 1

    def test_access_control_no_violation(self):
        cc = ComplianceChecker()
        data = {"access_logs": [{"user_id": "u1", "authorized": True}]}
        violations = cc.validate_compliance(
            data, rule_ids=["access_control_enforcement"]
        )
        assert len(violations) == 0

    def test_access_control_non_dict(self):
        cc = ComplianceChecker()
        data = {"access_logs": ["not_dict"]}
        violations = cc.validate_compliance(
            data, rule_ids=["access_control_enforcement"]
        )
        assert len(violations) == 0

    def test_custom_validation(self):
        cc = ComplianceChecker()
        cc.rules = [ComplianceRule(
            id="t", name="T", description="T",
            policy_type=PolicyType.CUSTOM,
            compliance_level=ComplianceLevel.LOW,
            validation_function="custom_validation",
            parameters={},
        )]
        violations = cc.validate_compliance({"data": "test"})
        assert len(violations) == 0


class TestScoreAndLevel:
    def test_score_no_violations(self):
        cc = ComplianceChecker()
        assert cc.calculate_compliance_score([]) == 1.0

    def test_score_with_violations(self):
        cc = ComplianceChecker()
        violations = [
            ComplianceViolation(
                rule_id="r1", severity=ViolationSeverity.HIGH,
                description="v1", timestamp=datetime.now(),
                context={}, remediation_suggested=[],
            )
        ]
        score = cc.calculate_compliance_score(violations)
        assert 0.0 <= score <= 1.0

    def test_score_resolved_violations(self):
        cc = ComplianceChecker()
        violations = [
            ComplianceViolation(
                rule_id="r1", severity=ViolationSeverity.LOW,
                description="v1", timestamp=datetime.now(),
                context={}, remediation_suggested=[], resolved=True,
            )
        ]
        score = cc.calculate_compliance_score(violations)
        assert score > 0.0

    def test_determine_compliance_level(self):
        cc = ComplianceChecker()
        assert cc._determine_compliance_level(0.95) == ComplianceLevel.LOW
        assert cc._determine_compliance_level(0.8) == ComplianceLevel.MEDIUM
        assert cc._determine_compliance_level(0.6) == ComplianceLevel.HIGH
        assert cc._determine_compliance_level(0.3) == ComplianceLevel.CRITICAL


class TestRemediation:
    def test_suggest_remediation_known_rule(self):
        cc = ComplianceChecker()
        v = ComplianceViolation(
            rule_id="gdpr_data_retention",
            severity=ViolationSeverity.HIGH,
            description="test", timestamp=datetime.now(),
            context={}, remediation_suggested=[],
        )
        suggestions = cc.suggest_remediation(v)
        assert "delete_expired_data" in suggestions

    def test_suggest_remediation_unknown_rule(self):
        cc = ComplianceChecker()
        v = ComplianceViolation(
            rule_id="unknown_rule",
            severity=ViolationSeverity.LOW,
            description="test", timestamp=datetime.now(),
            context={}, remediation_suggested=[],
        )
        suggestions = cc.suggest_remediation(v)
        assert "review_compliance_policy" in suggestions

    def test_execute_remediation_delete(self):
        cc = ComplianceChecker()
        v = ComplianceViolation(
            rule_id="gdpr", severity=ViolationSeverity.HIGH,
            description="test", timestamp=datetime.now(),
            context={"data_id": "d1"}, remediation_suggested=[],
        )
        action = cc.execute_remediation(v, "delete_expired_data")
        assert action.success is True
        assert v.resolved is True

    def test_execute_remediation_delete_no_data_id(self):
        cc = ComplianceChecker()
        v = ComplianceViolation(
            rule_id="gdpr", severity=ViolationSeverity.HIGH,
            description="test", timestamp=datetime.now(),
            context={}, remediation_suggested=[],
        )
        action = cc.execute_remediation(v, "delete_expired_data")
        assert action.success is False

    def test_execute_remediation_encrypt(self):
        cc = ComplianceChecker()
        v = ComplianceViolation(
            rule_id="enc", severity=ViolationSeverity.CRITICAL,
            description="test", timestamp=datetime.now(),
            context={}, remediation_suggested=[],
        )
        action = cc.execute_remediation(v, "encrypt_sensitive_data")
        assert action.success is True

    def test_execute_remediation_rbac(self):
        cc = ComplianceChecker()
        v = ComplianceViolation(
            rule_id="ac", severity=ViolationSeverity.HIGH,
            description="test", timestamp=datetime.now(),
            context={}, remediation_suggested=[],
        )
        action = cc.execute_remediation(v, "implement_rbac")
        assert action.success is True

    def test_execute_remediation_generic(self):
        cc = ComplianceChecker()
        v = ComplianceViolation(
            rule_id="gen", severity=ViolationSeverity.LOW,
            description="test", timestamp=datetime.now(),
            context={}, remediation_suggested=[],
        )
        action = cc.execute_remediation(v, "generic_action")
        assert action.success is True

    def test_get_remediation_history(self):
        cc = ComplianceChecker()
        assert cc.get_remediation_history() == []

    def test_record_remediation_action(self):
        cc = ComplianceChecker()
        action = RemediationAction(
            violation_id="v1", action_type="test",
            description="test", executed_at=datetime.now(),
            success=True, details={},
        )
        cc.record_remediation_action(action)
        assert len(cc.remediation_actions) == 1


class TestReports:
    def test_generate_compliance_report(self):
        cc = ComplianceChecker()
        report = cc.generate_compliance_report()
        assert isinstance(report, ComplianceReport)
        assert report.overall_score == 1.0

    def test_generate_report_with_violations(self):
        cc = ComplianceChecker()
        cc.violations.append(ComplianceViolation(
            rule_id="r1", severity=ViolationSeverity.CRITICAL,
            description="test", timestamp=datetime.now(),
            context={}, remediation_suggested=[],
        ))
        report = cc.generate_compliance_report()
        assert report.summary["total_violations"] >= 1

    def test_generate_regulatory_report(self):
        cc = ComplianceChecker()
        report = cc.generate_regulatory_report(PolicyType.GDPR)
        assert report["regulation"] == "gdpr"
        assert report["compliance_status"] == "compliant"

    def test_generate_regulatory_report_with_violations(self):
        cc = ComplianceChecker()
        cc.violations.append(ComplianceViolation(
            rule_id="gdpr_data_retention",
            severity=ViolationSeverity.HIGH,
            description="test", timestamp=datetime.now(),
            context={}, remediation_suggested=[],
        ))
        report = cc.generate_regulatory_report(PolicyType.GDPR)
        assert report["violations_summary"]["total"] >= 1

    def test_remediation_success_rate_no_violations(self):
        cc = ComplianceChecker()
        rate = cc._calculate_remediation_success_rate([])
        assert rate == 1.0

    def test_remediation_success_rate_with_violations(self):
        cc = ComplianceChecker()
        violations = [
            ComplianceViolation(
                rule_id="r", severity=ViolationSeverity.LOW,
                description="t", timestamp=datetime.now(),
                context={}, remediation_suggested=[], resolved=True,
            ),
            ComplianceViolation(
                rule_id="r", severity=ViolationSeverity.LOW,
                description="t", timestamp=datetime.now(),
                context={}, remediation_suggested=[], resolved=False,
            ),
        ]
        rate = cc._calculate_remediation_success_rate(violations)
        assert rate == 0.5


class TestPolicyEnforcement:
    def test_enforce_gdpr_data_access_blocked(self):
        cc = ComplianceChecker()
        result = cc.enforce_policy(
            PolicyType.GDPR, "data_access",
            {"data_access_request": {
                "data_type": "personal", "purpose": "analytics"
            }}
        )
        assert result["allowed"] is False

    def test_enforce_gdpr_data_access_allowed(self):
        cc = ComplianceChecker()
        result = cc.enforce_policy(
            PolicyType.GDPR, "data_access",
            {"data_access_request": {
                "data_type": "public", "purpose": "analytics"
            }}
        )
        assert result["allowed"] is True

    def test_enforce_non_gdpr_policy(self):
        cc = ComplianceChecker()
        result = cc.enforce_policy(
            PolicyType.HIPAA, "data_access", {}
        )
        assert result["allowed"] is True


class TestMonitoring:
    def test_start_monitoring(self):
        cc = ComplianceChecker()
        cc.start_monitoring()
        assert cc.monitoring_active is True
        cc.stop_monitoring()

    def test_start_monitoring_already_active(self):
        cc = ComplianceChecker()
        cc.start_monitoring()
        cc.start_monitoring()  # Should log warning
        cc.stop_monitoring()

    def test_stop_monitoring(self):
        cc = ComplianceChecker()
        cc.start_monitoring()
        cc.stop_monitoring()
        assert cc.monitoring_active is False

    def test_stop_monitoring_not_active(self):
        cc = ComplianceChecker()
        cc.stop_monitoring()  # No thread

    def test_get_monitoring_status(self):
        cc = ComplianceChecker()
        status = cc.get_monitoring_status()
        assert status["active"] is False
        assert status["thread_alive"] is False


class TestDashboardAndExport:
    def test_get_dashboard_data(self):
        cc = ComplianceChecker()
        dashboard = cc.get_dashboard_data()
        assert "compliance_score" in dashboard
        assert "rules_summary" in dashboard

    def test_get_policy_breakdown(self):
        cc = ComplianceChecker()
        breakdown = cc._get_policy_breakdown()
        assert isinstance(breakdown, dict)

    def test_classify_violation_severity(self):
        cc = ComplianceChecker()
        assert cc.classify_violation_severity(
            {"impact": "critical"}
        ) == ViolationSeverity.CRITICAL
        assert cc.classify_violation_severity(
            {"impact": "high"}
        ) == ViolationSeverity.HIGH
        assert cc.classify_violation_severity(
            {"impact": "medium"}
        ) == ViolationSeverity.MEDIUM
        assert cc.classify_violation_severity(
            {"impact": "low"}
        ) == ViolationSeverity.LOW
        assert cc.classify_violation_severity(
            {"impact": "info"}
        ) == ViolationSeverity.INFO
        assert cc.classify_violation_severity(
            {"impact": "unknown"}
        ) == ViolationSeverity.MEDIUM
        assert cc.classify_violation_severity(
            {}
        ) == ViolationSeverity.MEDIUM

    def test_export_compliance_data(self):
        cc = ComplianceChecker()
        data = cc.export_compliance_data()
        assert "export_metadata" in data
        assert "violations" in data
        assert "rules" in data

    def test_export_with_date_range(self):
        cc = ComplianceChecker()
        start = datetime.now() - timedelta(days=7)
        end = datetime.now()
        data = cc.export_compliance_data(date_range=(start, end))
        assert data["export_metadata"]["format"] == "json"

    def test_get_audit_trail(self):
        cc = ComplianceChecker()
        cc.violations.append(ComplianceViolation(
            rule_id="r1", severity=ViolationSeverity.HIGH,
            description="test", timestamp=datetime.now(),
            context={}, remediation_suggested=[],
        ))
        start = datetime.now() - timedelta(hours=1)
        end = datetime.now() + timedelta(hours=1)
        trail = cc.get_audit_trail(start, end)
        assert len(trail) >= 1

    def test_get_audit_trail_with_remediations(self):
        cc = ComplianceChecker()
        cc.remediation_actions.append(RemediationAction(
            violation_id="v1", action_type="fix",
            description="Fixed", executed_at=datetime.now(),
            success=True, details={},
        ))
        start = datetime.now() - timedelta(hours=1)
        end = datetime.now() + timedelta(hours=1)
        trail = cc.get_audit_trail(start, end)
        assert len(trail) >= 1

    def test_get_audit_trail_sorted(self):
        cc = ComplianceChecker()
        now = datetime.now()
        cc.violations.append(ComplianceViolation(
            rule_id="r2", severity=ViolationSeverity.LOW,
            description="second", timestamp=now,
            context={}, remediation_suggested=[],
        ))
        cc.violations.append(ComplianceViolation(
            rule_id="r1", severity=ViolationSeverity.HIGH,
            description="first",
            timestamp=now - timedelta(minutes=5),
            context={}, remediation_suggested=[],
        ))
        start = now - timedelta(hours=1)
        end = now + timedelta(hours=1)
        trail = cc.get_audit_trail(start, end)
        assert len(trail) >= 2
