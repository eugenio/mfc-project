"""
Compliance and security monitoring module for MFC systems.

This module provides comprehensive security monitoring, threat detection,
compliance management, and data privacy capabilities for microbial fuel cell systems.

Features include:
- Security monitoring and threat detection
- Access control and authentication
- Compliance checking and reporting
- PII detection and masking
- Data anonymization
- GDPR compliance (Right to be Forgotten)
- CCPA compliance
- Consent management
"""

from .access_control import (
    AccessControl,
    AuthenticationError,
    AuthorizationError,
    Permission,
    Role,
    Session,
    User,
)
from .compliance_checker import (
    ComplianceChecker,
    ComplianceLevel,
    ComplianceReport,
    ComplianceRule,
    ComplianceViolation,
    PolicyType,
    RemediationAction,
    ViolationSeverity,
)
from .security_monitor import SecurityMonitor

# Import DataPrivacyManager if available
try:
    from .data_privacy_manager import DataPrivacyManager
    _has_data_privacy = True
except ImportError:
    _has_data_privacy = False

__all__ = [
    "SecurityMonitor",
    "ComplianceChecker",
    "ComplianceRule",
    "ComplianceViolation",
    "ComplianceReport",
    "RemediationAction",
    "ComplianceLevel",
    "ViolationSeverity",
    "PolicyType",
    "AccessControl",
    "Permission",
    "Role",
    "User",
    "Session",
    "AuthenticationError",
    "AuthorizationError",
]

if _has_data_privacy:
    __all__.append("DataPrivacyManager")
