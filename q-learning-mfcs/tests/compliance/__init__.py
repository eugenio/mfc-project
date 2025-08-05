"""
Compliance and Regulatory Testing Suite

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

# Test suite components
from .test_data_privacy_compliance import *
from .test_audit_trail_systems import *
from .test_access_control_compliance import *
from .test_security_middleware_compliance import *
from .test_differential_privacy_compliance import *
from .test_compliance_reporting import *
from .test_data_retention_policies import *

__all__ = [
    "DataPrivacyComplianceTests",
    "AuditTrailSystemsTests", 
    "AccessControlComplianceTests",
    "SecurityMiddlewareComplianceTests",
    "DifferentialPrivacyComplianceTests",
    "ComplianceReportingTests",
    "DataRetentionPolicyTests"
]