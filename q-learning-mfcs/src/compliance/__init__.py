"""
Compliance module for data privacy and regulatory compliance.

This module provides comprehensive data privacy management including:
- PII detection and masking
- Data anonymization
- GDPR compliance (Right to be Forgotten)
- CCPA compliance
- Consent management

Author: TDD Agent 4
"""

from .data_privacy_manager import DataPrivacyManager

__all__ = ["DataPrivacyManager"]