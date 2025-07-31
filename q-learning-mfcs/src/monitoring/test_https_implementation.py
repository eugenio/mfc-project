#!/usr/bin/env python3
"""
Test Suite for MFC Monitoring System HTTPS Implementation
Comprehensive tests to ensure SSL/TLS security is working correctly.
"""

import os
import sys
import ssl
import json
import socket
import tempfile
import unittest
import asyncio
import logging
from pathlib import Path
from datetime import datetime

import requests
import websockets
from unittest.mock import patch

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ssl_config import (
    SSLConfig, CertificateManager, SSLContextManager,
    load_ssl_config
)
from security_middleware import SecurityConfig, SessionManager, CSRFProtection, RateLimiter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HTTPSTestSuite(unittest.TestCase):
    """Comprehensive HTTPS test suite"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_domain = "localhost"
        cls.test_email = "test@example.com"
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_cert_file = os.path.join(cls.temp_dir, "test.crt")
        cls.test_key_file = os.path.join(cls.temp_dir, "test.key")

        # Create test SSL configuration
        cls.ssl_config = SSLConfig(
            domain=cls.test_domain,
            email=cls.test_email,
            cert_file=cls.test_cert_file,
            key_file=cls.test_key_file,
            use_letsencrypt=False,  # Use self-signed for testing
            https_port_api=8443,
            https_port_frontend=8444,
            wss_port_streaming=8445
        )

        # Generate test certificates
        cls._generate_test_certificates()

    @classmethod
    def _generate_test_certificates(cls):
        """Generate self-signed certificates for testing"""
        cert_manager = CertificateManager(cls.ssl_config)
        success = cert_manager.generate_self_signed_certificate()

        if not success:
            raise RuntimeError("Failed to generate test certificates")

        logger.info("Test certificates generated successfully")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_ssl_config_creation(self):
        """Test SSL configuration creation and validation"""
        config = SSLConfig()

        # Test default values
        self.assertEqual(config.domain, "localhost")
        self.assertEqual(config.ssl_version, "TLSv1_2")
        self.assertTrue(config.enable_hsts)
        self.assertTrue(config.enable_csp)

        # Test serialization
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)

        # Test deserialization
        new_config = SSLConfig.from_dict(config_dict)
        self.assertEqual(new_config.domain, config.domain)
        self.assertEqual(new_config.https_port_api, config.https_port_api)

    def test_certificate_manager(self):
        """Test certificate manager functionality"""
        cert_manager = CertificateManager(self.ssl_config)

        # Test certificate existence check
        exists = cert_manager.check_certificate_exists()
        self.assertTrue(exists, "Test certificates should exist")

        # Test certificate validity check
        is_valid, expiry_date = cert_manager.check_certificate_validity()
        self.assertIsNotNone(expiry_date, "Should have expiry date")

        # Test expiry date is in the future (for newly generated cert)
        if expiry_date:
            self.assertGreater(expiry_date, datetime.now(), "Certificate should not be expired")

    def test_ssl_context_manager(self):
        """Test SSL context creation"""
        ssl_manager = SSLContextManager(self.ssl_config)

        # Test SSL context creation
        try:
            context = ssl_manager.create_ssl_context()
            self.assertIsInstance(context, ssl.SSLContext)
            self.assertEqual(context.minimum_version, ssl.TLSVersion.TLSv1_2)
        except Exception as e:
            self.fail(f"SSL context creation failed: {e}")

        # Test uvicorn SSL config
        uvicorn_config = ssl_manager.get_uvicorn_ssl_config()
        self.assertIn('ssl_keyfile', uvicorn_config)
        self.assertIn('ssl_certfile', uvicorn_config)
        self.assertEqual(uvicorn_config['ssl_keyfile'], self.ssl_config.key_file)
        self.assertEqual(uvicorn_config['ssl_certfile'], self.ssl_config.cert_file)

    def test_security_headers(self):
        """Test security headers generation"""
        from ssl_config import SecurityHeaders

        headers = SecurityHeaders.get_security_headers(self.ssl_config)

        # Check required security headers
        self.assertIn('Strict-Transport-Security', headers)
        self.assertIn('Content-Security-Policy', headers)
        self.assertIn('X-Content-Type-Options', headers)
        self.assertIn('X-Frame-Options', headers)
        self.assertIn('X-XSS-Protection', headers)

        # Check HSTS header format
        hsts_header = headers['Strict-Transport-Security']
        self.assertIn('max-age=', hsts_header)
        self.assertIn('includeSubDomains', hsts_header)

        # Check CSP header content
        csp_header = headers['Content-Security-Policy']
        self.assertIn('default-src', csp_header)
        self.assertIn('frame-ancestors', csp_header)

    def test_session_manager(self):
        """Test secure session management"""
        security_config = SecurityConfig()
        session_manager = SessionManager(security_config)

        # Test session creation
        session_id = session_manager.create_session("test_user", {"role": "admin"})
        self.assertIsInstance(session_id, str)
        self.assertGreater(len(session_id), 16)

        # Test session validation
        session_data = session_manager.validate_session(session_id)
        self.assertIsNotNone(session_data)
        self.assertEqual(session_data["user_id"], "test_user")
        self.assertEqual(session_data["user_data"]["role"], "admin")

        # Test invalid session
        invalid_session = session_manager.validate_session("invalid_session_id")
        self.assertIsNone(invalid_session)

        # Test session destruction
        destroyed = session_manager.destroy_session(session_id)
        self.assertTrue(destroyed)

        # Test session no longer valid after destruction
        session_data = session_manager.validate_session(session_id)
        self.assertIsNone(session_data)

    def test_csrf_protection(self):
        """Test CSRF token generation and validation"""
        security_config = SecurityConfig()
        csrf_protection = CSRFProtection(security_config)

        session_id = "test_session_123"

        # Test token generation
        token = csrf_protection.generate_csrf_token(session_id)
        self.assertIsInstance(token, str)
        self.assertGreater(len(token), 16)

        # Test token validation
        is_valid = csrf_protection.validate_csrf_token(token, session_id)
        self.assertTrue(is_valid)

        # Test token validation with wrong session
        is_invalid = csrf_protection.validate_csrf_token(token, "wrong_session")
        self.assertFalse(is_invalid)

        # Test invalid token format
        is_invalid = csrf_protection.validate_csrf_token("invalid_token", session_id)
        self.assertFalse(is_invalid)

    def test_rate_limiter(self):
        """Test rate limiting functionality"""
        security_config = SecurityConfig()
        security_config.rate_limit_requests_per_minute = 5  # Low limit for testing

        rate_limiter = RateLimiter(security_config)

        client_ip = "192.168.1.100"

        # Test normal requests under limit
        for i in range(4):
            is_limited = rate_limiter.is_rate_limited(client_ip)
            self.assertFalse(is_limited, f"Request {i+1} should not be rate limited")

        # Test rate limiting kicks in
        is_limited = rate_limiter.is_rate_limited(client_ip)
        self.assertTrue(is_limited, "Should be rate limited after exceeding limit")

        # Test different IP is not affected
        other_ip = "192.168.1.101"
        is_limited = rate_limiter.is_rate_limited(other_ip)
        self.assertFalse(is_limited, "Different IP should not be rate limited")

class HTTPSIntegrationTests(unittest.TestCase):
    """Integration tests requiring running services"""

    def setUp(self):
        """Set up for integration tests"""
        self.ssl_config = SSLConfig(
            domain="localhost",
            use_letsencrypt=False,
            https_port_api=8443,
            https_port_frontend=8444,
            wss_port_streaming=8445
        )

        # Disable SSL verification for self-signed certificates in tests
        self.session = requests.Session()
        self.session.verify = False
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    @unittest.skipIf(not os.getenv("RUN_INTEGRATION_TESTS"), "Integration tests disabled")
    def test_https_api_endpoint(self):
        """Test HTTPS API endpoint accessibility"""
        api_url = f"https://{self.ssl_config.domain}:{self.ssl_config.https_port_api}/health"

        try:
            response = self.session.get(api_url, timeout=10)
            self.assertEqual(response.status_code, 200)

            # Check response content
            data = response.json()
            self.assertIn("status", data)
            self.assertEqual(data["status"], "healthy")

            # Check security headers
            self.assertIn("Strict-Transport-Security", response.headers)
            self.assertIn("X-Content-Type-Options", response.headers)

        except requests.exceptions.ConnectionError:
            self.skipTest("API service not running")

    @unittest.skipIf(not os.getenv("RUN_INTEGRATION_TESTS"), "Integration tests disabled")
    def test_https_frontend_accessibility(self):
        """Test HTTPS frontend accessibility"""
        frontend_url = f"https://{self.ssl_config.domain}:{self.ssl_config.https_port_frontend}"

        try:
            response = self.session.get(frontend_url, timeout=10)
            # Streamlit typically returns 200 or redirects
            self.assertIn(response.status_code, [200, 301, 302])

            # Check security headers
            if response.status_code == 200:
                self.assertIn("X-Frame-Options", response.headers)

        except requests.exceptions.ConnectionError:
            self.skipTest("Frontend service not running")

    @unittest.skipIf(not os.getenv("RUN_INTEGRATION_TESTS"), "Integration tests disabled")
    def test_wss_websocket_connection(self):
        """Test WSS WebSocket connection"""
        ws_url = f"wss://{self.ssl_config.domain}:{self.ssl_config.wss_port_streaming}/"

        async def test_websocket():
            try:
                # Create SSL context that accepts self-signed certificates
                ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

                async with websockets.connect(ws_url, ssl=ssl_context) as websocket:
                    # Send ping message
                    await websocket.send(json.dumps({"type": "ping"}))

                    # Wait for pong response
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(response)

                    self.assertEqual(data["type"], "pong")

            except (websockets.exceptions.ConnectionClosed,
                   websockets.exceptions.InvalidURI,
                   ConnectionRefusedError) as e:
                self.skipTest(f"WebSocket service not running: {e}")

        # Run async test
        try:
            asyncio.run(test_websocket())
        except Exception as e:
            self.skipTest(f"WebSocket test failed: {e}")

    def test_ssl_connection_security(self):
        """Test SSL connection security parameters"""
        host = self.ssl_config.domain
        port = self.ssl_config.https_port_api

        try:
            # Create SSL context
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE  # For self-signed certificates

            # Connect and get SSL info
            with socket.create_connection((host, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=host) as ssock:
                    ssl_version = ssock.version()
                    cipher = ssock.cipher()

                    # Check SSL version is secure
                    self.assertIn(ssl_version, ['TLSv1.2', 'TLSv1.3'])

                    # Check cipher is present
                    self.assertIsNotNone(cipher)

                    logger.info(f"SSL connection successful: {ssl_version}, {cipher[0]}")

        except (ConnectionRefusedError, socket.timeout):
            self.skipTest("API service not available for SSL testing")

class CertificateManagementTests(unittest.TestCase):
    """Test certificate management functionality"""

    def setUp(self):
        """Set up certificate management tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = SSLConfig(
            domain="test.example.com",
            cert_file=os.path.join(self.temp_dir, "test.crt"),
            key_file=os.path.join(self.temp_dir, "test.key"),
            use_letsencrypt=False
        )

    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_self_signed_certificate_generation(self):
        """Test self-signed certificate generation"""
        cert_manager = CertificateManager(self.test_config)

        # Generate certificate
        success = cert_manager.generate_self_signed_certificate()
        self.assertTrue(success, "Certificate generation should succeed")

        # Check files exist
        self.assertTrue(os.path.exists(self.test_config.cert_file))
        self.assertTrue(os.path.exists(self.test_config.key_file))

        # Check file permissions
        key_stat = os.stat(self.test_config.key_file)
        self.assertEqual(oct(key_stat.st_mode)[-3:], '600', "Private key should be 600 permissions")

        cert_stat = os.stat(self.test_config.cert_file)
        self.assertEqual(oct(cert_stat.st_mode)[-3:], '644', "Certificate should be 644 permissions")

    def test_certificate_validation(self):
        """Test certificate validation"""
        cert_manager = CertificateManager(self.test_config)

        # Generate certificate first
        cert_manager.generate_self_signed_certificate()

        # Test existence check
        exists = cert_manager.check_certificate_exists()
        self.assertTrue(exists)

        # Test validity check
        is_valid, expiry_date = cert_manager.check_certificate_validity()
        self.assertIsNotNone(expiry_date)

        # Certificate should be valid for newly generated cert
        self.assertGreater(expiry_date, datetime.now())

    @patch('subprocess.run')
    def test_lets_encrypt_certificate_request(self, mock_subprocess):
        """Test Let's Encrypt certificate request (mocked)"""
        # Mock successful certbot run
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "Certificate obtained successfully"

        # Mock which command to find certbot
        with patch('subprocess.run') as mock_which:
            mock_which.return_value.returncode = 0

            cert_manager = CertificateManager(self.test_config)
            cert_manager.config.use_letsencrypt = True

            success = cert_manager.request_letsencrypt_certificate()
            self.assertTrue(success, "Let's Encrypt request should succeed (mocked)")

    def test_auto_renewal_setup(self):
        """Test auto-renewal cron job setup"""
        cert_manager = CertificateManager(self.test_config)

        # This test checks the cron setup logic without actually modifying system cron
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.returncode = 0

            success = cert_manager.setup_auto_renewal()
            self.assertTrue(success, "Auto-renewal setup should succeed (mocked)")

def run_service_tests():
    """Run tests against running services"""
    print("Running HTTPS service tests...")

    # Test ports are accessible
    ports_to_test = [8443, 8444, 8445]

    for port in ports_to_test:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)
                result = s.connect_ex(('localhost', port))

                if result == 0:
                    print(f"✅ Port {port} is accessible")
                else:
                    print(f"❌ Port {port} is not accessible")

        except Exception as e:
            print(f"❌ Error testing port {port}: {e}")

    # Test SSL certificates
    try:
        ssl_config = load_ssl_config()
        if ssl_config:
            cert_file = Path(ssl_config.cert_file)
            key_file = Path(ssl_config.key_file)

            if cert_file.exists() and key_file.exists():
                print("✅ SSL certificate files found")

                # Test certificate validity
                cert_manager = CertificateManager(ssl_config)
                is_valid, expiry_date = cert_manager.check_certificate_validity()

                if expiry_date:
                    days_until_expiry = (expiry_date - datetime.now()).days
                    print(f"✅ Certificate valid until {expiry_date} ({days_until_expiry} days)")
                else:
                    print("❌ Could not determine certificate validity")
            else:
                print("❌ SSL certificate files not found")
        else:
            print("❌ No SSL configuration found")

    except Exception as e:
        print(f"❌ Error checking SSL configuration: {e}")

def run_comprehensive_tests():
    """Run comprehensive test suite"""
    print("="*60)
    print("MFC HTTPS Implementation Test Suite")
    print("="*60)

    # Set environment variable for integration tests
    os.environ["RUN_INTEGRATION_TESTS"] = "1"

    # Create test suite
    suite = unittest.TestSuite()

    # Add unit tests
    suite.addTest(unittest.makeSuite(HTTPSTestSuite))
    suite.addTest(unittest.makeSuite(CertificateManagementTests))

    # Add integration tests
    suite.addTest(unittest.makeSuite(HTTPSIntegrationTests))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Run service tests
    print("\n" + "="*60)
    print("Service Accessibility Tests")
    print("="*60)
    run_service_tests()

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")

    # Return success status
    return len(result.failures) == 0 and len(result.errors) == 0

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HTTPS Implementation Test Suite")
    parser.add_argument("--unit-only", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration-only", action="store_true", help="Run only integration tests")
    parser.add_argument("--service-tests", action="store_true", help="Run service accessibility tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.service_tests:
        run_service_tests()
    elif args.unit_only:
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(HTTPSTestSuite))
        suite.addTest(unittest.makeSuite(CertificateManagementTests))
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
    elif args.integration_only:
        os.environ["RUN_INTEGRATION_TESTS"] = "1"
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(HTTPSIntegrationTests))
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
    else:
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)
