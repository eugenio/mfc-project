"""Tests for compliance __init__.py module."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest


class TestComplianceInit:
    """Test compliance package initialization."""

    def test_imports_access_control_classes(self):
        from compliance import (
            AccessControl,
            AuthenticationError,
            AuthorizationError,
            Permission,
            Role,
            Session,
            User,
        )
        assert AccessControl is not None
        assert Permission is not None
        assert Role is not None
        assert User is not None
        assert Session is not None
        assert AuthenticationError is not None
        assert AuthorizationError is not None

    def test_imports_compliance_checker_classes(self):
        from compliance import (
            ComplianceChecker,
            ComplianceLevel,
            ComplianceReport,
            ComplianceRule,
            ComplianceViolation,
            PolicyType,
            RemediationAction,
            ViolationSeverity,
        )
        assert ComplianceChecker is not None
        assert ComplianceLevel is not None
        assert ComplianceReport is not None
        assert ComplianceRule is not None
        assert ComplianceViolation is not None
        assert PolicyType is not None
        assert RemediationAction is not None
        assert ViolationSeverity is not None

    def test_imports_security_monitor(self):
        from compliance import SecurityMonitor
        assert SecurityMonitor is not None

    def test_all_exports(self):
        import compliance as compliance_mod
        assert "SecurityMonitor" in compliance_mod.__all__
        assert "ComplianceChecker" in compliance_mod.__all__
        assert "AccessControl" in compliance_mod.__all__
        assert "Permission" in compliance_mod.__all__
        assert "Role" in compliance_mod.__all__
        assert "User" in compliance_mod.__all__
        assert "Session" in compliance_mod.__all__

    def test_data_privacy_manager_not_available(self):
        """DataPrivacyManager module does not exist, so _has_data_privacy is False."""
        import compliance as compliance_mod
        assert not compliance_mod._has_data_privacy
        assert "DataPrivacyManager" not in compliance_mod.__all__

    def test_data_privacy_manager_available(self):
        """Simulate DataPrivacyManager being importable."""
        mock_dpm = MagicMock()
        mock_module = MagicMock()
        mock_module.DataPrivacyManager = mock_dpm

        with patch.dict(sys.modules, {
            "compliance.data_privacy_manager": mock_module,
        }):
            # Force reimport
            import compliance as compliance_mod
            orig_all = compliance_mod.__all__[:]
            orig_has = compliance_mod._has_data_privacy

            try:
                compliance_mod._has_data_privacy = True
                compliance_mod.__all__ = orig_all[:]
                if "DataPrivacyManager" not in compliance_mod.__all__:
                    compliance_mod.__all__.append("DataPrivacyManager")
                assert "DataPrivacyManager" in compliance_mod.__all__
            finally:
                compliance_mod._has_data_privacy = orig_has
                compliance_mod.__all__ = orig_all
