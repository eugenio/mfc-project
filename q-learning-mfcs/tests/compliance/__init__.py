"""Compliance and Regulatory Testing Suite.

This module provides comprehensive testing for regulatory compliance including:
- GDPR/CCPA data privacy compliance
- Audit trail generation and integrity
- Access control and authorization (RBAC/ABAC)
- Data retention and deletion policies
- Compliance reporting and violation detection
- Security middleware compliance
- Differential privacy mechanisms

All tests are designed to ensure the MFC Q-learning system meets
regulatory requirements across multiple jurisdictions.
"""

__version__ = "1.0.0"
__author__ = "TDD Agent 64 - Compliance Testing Specialist"

# Test suite components - import only if available
__all__ = []

_imports = [
    ("test_data_privacy_compliance", "DataPrivacyComplianceTests"),
    ("test_audit_trail_systems", "AuditTrailSystemsTests"),
    ("test_access_control_compliance", "AccessControlComplianceTests"),
    ("test_security_middleware_compliance", "SecurityMiddlewareComplianceTests"),
    ("test_differential_privacy_compliance", "DifferentialPrivacyComplianceTests"),
    ("test_compliance_reporting", "ComplianceReportingTests"),
    ("test_data_retention_policies", "DataRetentionPolicyTests"),
]

for _mod, _cls in _imports:
    try:
        exec(f"from .{_mod} import *")
        __all__ += [_cls]
    except (ImportError, ModuleNotFoundError):
        pass
