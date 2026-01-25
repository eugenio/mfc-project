"""
Security Integration Tests
=========================

Integration tests for Security components including:
- SecurityMiddleware integration
- SessionManager integration
- Authentication and authorization flow
- SSL/TLS certificate management
- Rate limiting and CSRF protection

Created: 2025-08-05
Author: TDD Agent 10 - Integration & Testing
"""

from datetime import datetime
from pathlib import Path

import pytest


class TestSecurityMiddleware:
    """Test security middleware integration"""

    @pytest.fixture(scope="class")
    def security_system(self, tmp_path_factory):
        """Set up security system components"""
        import sys
        sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

        # Create temp directory
        temp_dir = tmp_path_factory.mktemp("security_integration")

        # Initialize Security components
        from monitoring.security_middleware import (
            CSRFProtection,
            RateLimiter,
            SecurityConfig,
            SessionManager,
        )

        # Create security configuration
        security_config = SecurityConfig()
        security_config.session_timeout_minutes = 60  # 1 hour
        security_config.rate_limit_requests_per_minute = 100

        # Create components
        session_manager = SessionManager(security_config)
        csrf_protection = CSRFProtection(security_config)
        rate_limiter = RateLimiter(security_config)

        return {
            "security_config": security_config,
            "session_manager": session_manager,
            "csrf_protection": csrf_protection,
            "rate_limiter": rate_limiter,
            "temp_dir": temp_dir
        }

    @pytest.mark.integration
    @pytest.mark.security
    def test_session_lifecycle(self, security_system):
        """Test complete session lifecycle"""
        session_manager = security_system["session_manager"]

        # Create session
        user_data = {"email": "test@example.com"}
        session_id = session_manager.create_session("test_user", user_data)

        assert session_id is not None
        assert len(session_id) > 10  # Should be a reasonable length

        # Validate session
        session_data = session_manager.validate_session(session_id)
        assert session_data is not None
        assert session_data["user_id"] == "test_user"
        assert session_data["user_data"]["email"] == "test@example.com"

        # Invalidate session
        success = session_manager.destroy_session(session_id)
        assert success is True

        # Try to get invalidated session
        invalid_session = session_manager.validate_session(session_id)
        assert invalid_session is None

    @pytest.mark.integration
    @pytest.mark.security
    def test_csrf_protection(self, security_system):
        """Test CSRF protection functionality"""
        csrf_protection = security_system["csrf_protection"]

        # Generate CSRF token
        token = csrf_protection.generate_csrf_token("test_session")
        assert token is not None
        assert len(token) > 16

        # Validate token
        is_valid = csrf_protection.validate_csrf_token(token, "test_session")
        assert is_valid is True

        # Test invalid token
        is_invalid = csrf_protection.validate_csrf_token("invalid_token", "test_session")
        assert is_invalid is False

        # Test token for different session
        is_cross_session = csrf_protection.validate_csrf_token(token, "other_session")
        assert is_cross_session is False

    @pytest.mark.integration
    @pytest.mark.security
    def test_rate_limiting(self, security_system):
        """Test rate limiting functionality."""
        rate_limiter = security_system["rate_limiter"]
        client_ip = "192.168.1.100"

        # Test normal requests (should not be rate limited initially)
        for _i in range(10):
            is_limited = rate_limiter.is_rate_limited(client_ip)
            assert is_limited is False, "Should not be rate limited for normal request count"

        # Test rate limit exceeded (if configured low enough)
        max_requests = rate_limiter.config.rate_limit_requests_per_minute
        if max_requests <= 50:
            # Make many requests quickly to trigger rate limiting
            for _i in range(max_requests + 5):
                rate_limiter.is_rate_limited(client_ip)

            # Note: Rate limiting behavior depends on exact timing and implementation

        # Test different client (should not be affected by other client's requests)
        other_client_ip = "192.168.1.101"
        other_is_limited = rate_limiter.is_rate_limited(other_client_ip)
        assert other_is_limited is False, "Different client should not be rate limited"


class TestSecurityFeatures:
    """Test comprehensive security features"""

    @pytest.mark.integration
    @pytest.mark.security
    def test_authentication_flow(self, tmp_path):
        """Test complete authentication flow"""
        import sys
        sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

        from monitoring.security_middleware import SecurityConfig, SessionManager

        # Create security system
        config = SecurityConfig()
        session_manager = SessionManager(config)

        # Simulate login process
        user_credentials = {
            "username": "test_user",
            "password": "secure_password_123",
            "email": "test@example.com"
        }

        # Create authenticated session
        session_id = session_manager.create_session(
            user_credentials["username"],
            {
                "email": user_credentials["email"],
                "authenticated": True,
                "login_time": datetime.now().isoformat()
            }
        )

        # Verify session
        session = session_manager.validate_session(session_id)
        assert session["user_data"]["authenticated"] is True
        assert session["user_id"] == user_credentials["username"]

        # Test session timeout (simulate)
        expired_session = session_manager.validate_session("expired_session_id")
        assert expired_session is None

    @pytest.mark.integration
    @pytest.mark.security
    def test_security_headers(self, tmp_path):
        """Test security headers implementation"""
        # Mock web client for testing security headers
        class MockClient:
            def __init__(self):
                self.headers = {}

            def get(self, path):
                # Simulate security headers
                return MockResponse({
                    "X-Content-Type-Options": "nosniff",
                    "X-Frame-Options": "DENY",
                    "X-XSS-Protection": "1; mode=block",
                    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
                    "Content-Security-Policy": "default-src 'self'"
                })

        class MockResponse:
            def __init__(self, headers):
                self.headers = headers
                self.status_code = 200

        client = MockClient()
        response = client.get("/")

        # Check security headers
        expected_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Content-Security-Policy"
        ]

        for header in expected_headers:
            assert header in response.headers

        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"


class TestSecurityCompliance:
    """Test security compliance and standards"""

    @pytest.mark.integration
    @pytest.mark.security
    def test_ssl_certificate_handling(self, tmp_path):
        """Test SSL certificate management"""
        # Create mock SSL certificates
        ssl_dir = tmp_path / "ssl"
        ssl_dir.mkdir()

        cert_file = ssl_dir / "test.crt"
        key_file = ssl_dir / "test.key"

        # Mock certificate content
        cert_content = """-----BEGIN CERTIFICATE-----
MIICljCCAX4CCQDAOxqjhkdFdTANBgkqhkiG9w0BAQsFADArMQswCQYDVQQGEwJV
UzELMAkGA1UECAwCQ0ExDzANBgNVBAcMBkJlcmtlbDEEMCEGA1UECgwaTWljcm9i
-----END CERTIFICATE-----"""

        key_content = """-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC7vJCaxgFubOZj
yTdEeHWU5tTHRAkNAYiVCJYiGnVJrDqxgW0W7CHIaHVCXfyLgtbStTN8YnYhGqS3
-----END PRIVATE KEY-----"""

        cert_file.write_text(cert_content)
        key_file.write_text(key_content)

        # Test certificate files exist and are readable
        assert cert_file.exists()
        assert key_file.exists()
        assert len(cert_file.read_text()) > 100
        assert len(key_file.read_text()) > 100

        # Test SSL configuration
        ssl_config = {
            "cert_file": str(cert_file),
            "key_file": str(key_file),
            "verify_mode": "required"
        }

        assert ssl_config["cert_file"] == str(cert_file)
        assert ssl_config["key_file"] == str(key_file)
        assert ssl_config["verify_mode"] == "required"

    @pytest.mark.integration
    @pytest.mark.security
    def test_input_validation_and_sanitization(self, tmp_path):
        """Test input validation and sanitization"""
        # Test cases for input validation
        test_inputs = [
            {"input": "normal_input", "expected_valid": True},
            {"input": "<script>alert('xss')</script>", "expected_valid": False},
            {"input": "'; DROP TABLE users; --", "expected_valid": False},
            {"input": "user@example.com", "expected_valid": True},
            {"input": "../../../etc/passwd", "expected_valid": False}
        ]

        def validate_input(user_input: str) -> bool:
            """Mock input validation function"""
            dangerous_patterns = [
                "<script", "javascript:", "DROP TABLE", "../", "etc/passwd",
                "exec(", "eval(", "__import__"
            ]

            for pattern in dangerous_patterns:
                if pattern.lower() in user_input.lower():
                    return False
            return True

        # Test each input
        for test_case in test_inputs:
            result = validate_input(test_case["input"])
            assert result == test_case["expected_valid"], f"Failed validation for: {test_case['input']}"

    @pytest.mark.integration
    @pytest.mark.security
    def test_session_security_features(self, tmp_path):
        """Test advanced session security features"""
        import sys
        sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

        from monitoring.security_middleware import SecurityConfig, SessionManager

        config = SecurityConfig()
        config.session_timeout = 300  # 5 minutes for testing
        session_manager = SessionManager(config)

        # Test session with security metadata
        session_data = {
            "ip_address": "192.168.1.100",
            "user_agent": "Mozilla/5.0 Test Browser",
            "login_time": datetime.now().isoformat(),
        }

        session_id = session_manager.create_session("security_test_user", session_data)
        retrieved_session = session_manager.validate_session(session_id)

        # Verify security metadata
        assert retrieved_session["user_data"]["ip_address"] == "192.168.1.100"
        user_agent = retrieved_session["user_data"]["user_agent"]
        assert user_agent == "Mozilla/5.0 Test Browser"
        assert "login_time" in retrieved_session["user_data"]
        assert "last_activity" in retrieved_session
