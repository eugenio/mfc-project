#!/usr/bin/env python3
"""Simple test for DataPrivacyManager."""

from src.compliance.data_privacy_manager import DataPrivacyManager, PIIType


def test_basic_functionality():
    """Test basic DataPrivacyManager functionality."""
    config = {'encryption_key': 'test_key_32_characters_long_here!!'}
    manager = DataPrivacyManager(config)

    # Test PII detection
    text = 'Contact support@example.com for help'
    findings = manager.detect_pii(text)

    print(f'Found {len(findings)} PII items')
    for f in findings:
        print(f'  - {f.pii_type}: {f.value}')

    assert len(findings) == 1
    assert findings[0].pii_type == PIIType.EMAIL
    assert findings[0].value == 'support@example.com'

    print('✅ PII detection working!')

    # Test masking
    masked = manager.mask_pii(text, [PIIType.EMAIL])
    print(f'Masked text: {masked}')
    assert 'support@example.com' not in masked
    assert '[EMAIL_MASKED]' in masked

    print('✅ PII masking working!')

    # Test data storage
    user_data = {'email': 'test@example.com', 'name': 'Test User'}
    manager.store_user_data('test_user', user_data)
    retrieved = manager.get_user_data('test_user')

    assert retrieved is not None
    assert retrieved['email'] == user_data['email']

    print('✅ Data storage/retrieval working!')

    print('🎉 All basic tests passed!')

if __name__ == '__main__':
    test_basic_functionality()
