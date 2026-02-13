"""
Integration Tests for MFC Q-Learning System
==========================================

This package contains integration tests that validate the interaction between
different components of the MFC system, including:

- MLOps components (monitoring, observability, deployment)
- Security components (middleware, authentication, SSL)
- Core MFC system components
- Q-Learning optimization system
- Real-time monitoring and alerting

Integration tests focus on:
1. Component interaction and data flow
2. End-to-end workflows
3. System reliability and performance
4. Security and compliance requirements
5. Deployment and operational scenarios

Created: 2025-08-05
Author: TDD Agent 10 - Integration & Testing
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

# Configure logging for integration tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_SSL_DIR = Path(__file__).parent / "ssl_certs"

# Ensure test directories exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_SSL_DIR.mkdir(exist_ok=True)

class IntegrationTestConfig:
    """Configuration for integration tests"""
    
    # Test environment settings
    TEST_PORT_API = 18000
    TEST_PORT_FRONTEND = 18501
    TEST_SSL_PORT_API = 18443
    TEST_SSL_PORT_FRONTEND = 18444
    
    # Test timeouts
    STARTUP_TIMEOUT = 30.0
    REQUEST_TIMEOUT = 10.0
    SHUTDOWN_TIMEOUT = 15.0
    
    # Test data settings
    MAX_TEST_PROCESSES = 5
    MAX_TEST_ALERTS = 10
    TEST_SESSION_TIMEOUT = 300  # 5 minutes
    
    # Mock settings
    MOCK_EMAIL = True
    MOCK_TTS = True
    MOCK_EXTERNAL_APIS = True
    
    @classmethod
    def get_test_ssl_config(cls) -> dict[str, Any]:
        """Get SSL configuration for testing"""
        return {
            "enabled": True,
            "cert_file": str(TEST_SSL_DIR / "test_cert.pem"),
            "key_file": str(TEST_SSL_DIR / "test_key.pem"),
            "ca_file": str(TEST_SSL_DIR / "test_ca.pem"),
            "verify_mode": "optional",
            "https_port_api": cls.TEST_SSL_PORT_API,
            "https_port_frontend": cls.TEST_SSL_PORT_FRONTEND
        }
    
    @classmethod
    def get_test_environment(cls) -> dict[str, str]:
        """Get environment variables for testing"""
        return {
            "MFC_ENV": "test",
            "MFC_LOG_LEVEL": "INFO",
            "MFC_TEST_MODE": "true",
            "DISABLE_AUDIO": "true",
            "MOCK_EMAIL": "true" if cls.MOCK_EMAIL else "false",
            "MOCK_TTS": "true" if cls.MOCK_TTS else "false",
            "MFC_API_TOKEN": "test-api-token-12345",
            "MFC_SESSION_SECRET": "test-session-secret-key",
            "MFC_CSRF_SECRET": "test-csrf-secret-key",
            "MFC_JWT_SECRET": "test-jwt-secret-key"
        }


def pytest_configure(config):
    """Configure pytest for integration tests"""
    # Set test environment variables
    for key, value in IntegrationTestConfig.get_test_environment().items():
        os.environ[key] = value
    
    # Register custom markers
    config.addinivalue_line(
        "markers", 
        "integration: marks tests as integration tests (may be slow)"
    )
    config.addinivalue_line(
        "markers",
        "mlops: marks tests as MLOps integration tests"
    )
    config.addinivalue_line(
        "markers",
        "security: marks tests as security integration tests"
    )
    config.addinivalue_line(
        "markers",
        "system: marks tests as full system integration tests"
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration"""
    return IntegrationTestConfig()


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test data"""
    with tempfile.TemporaryDirectory(prefix="mfc_integration_test_") as temp:
        yield Path(temp)


@pytest.fixture(scope="session")
def mock_ssl_certs(temp_dir):
    """Create mock SSL certificates for testing"""
    ssl_dir = temp_dir / "ssl"
    ssl_dir.mkdir(exist_ok=True)
    
    # Create mock certificate files
    cert_content = """-----BEGIN CERTIFICATE-----
MIICljCCAX4CCQDAOxqjhkdFdTANBgkqhkiG9w0BAQsFADArMQswCQYDVQQGEwJV
UzELMAkGA1UECAwCQ0ExDzANBgNVBAcMBkJlcmtlbDEEMCEGA1UECgwaTWljcm9i
aWFsIEZ1ZWwgQ2VsbCBUZXN0MQswCQYDVQQDDAJ0ZXN0MB4XDTI1MDEwMTAwMDAw
MFoXDTI2MDEwMTAwMDAwMFowKzELMAkGA1UEBhMCVVMxCzAJBgNVBAgMAkNBMQ8w
DQYDVQQHDAZCZXJrZWwwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQC7
vJCaxgFubOZjyTdEeHWU5tTHRAkNAYiVCJYiGnVJrDqxgW0W7CHIaHVCXfyLgtbS
-----END CERTIFICATE-----"""
    
    key_content = """-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC7vJCaxgFubOZj
yTdEeHWU5tTHRAkNAYiVCJYiGnVJrDqxgW0W7CHIaHVCXfyLgtbStTN8YnYhGqS3
uHaAVgcFsA7QE8vQfBKRxYSPNvxQmCmZ9UqQKfOv2v3n2yPfYLz8Q8fRLYpL7K2F
-----END PRIVATE KEY-----"""
    
    ca_content = """-----BEGIN CERTIFICATE-----
MIICljCCAX4CCQDAOxqjhkdFdTANBgkqhkiG9w0BAQsFADArMQswCQYDVQQGEwJV
UzELMAkGA1UECAwCQ0ExDzANBgNVBAcMBkJlcmtlbDEEMCEGA1UECgwaTWljcm9i
aWFsIEZ1ZWwgQ2VsbCBUZXN0MQswCQYDVQQDDAJDQTAeFw0yNTAxMDEwMDAwMDBa
Fw0yNjAxMDEwMDAwMDBaMCsxCzAJBgNVBAYTAlVTMQswCQYDVQQIDAJDQTEPMA0G
-----END CERTIFICATE-----"""
    
    # Write certificate files
    (ssl_dir / "test_cert.pem").write_text(cert_content)
    (ssl_dir / "test_key.pem").write_text(key_content)  
    (ssl_dir / "test_ca.pem").write_text(ca_content)
    
    return ssl_dir


@pytest.fixture(scope="function")
def clean_environment():
    """Ensure clean test environment between tests"""
    # Store original environment
    original_env = os.environ.copy()
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


class IntegrationTestHelper:
    """Helper utilities for integration tests"""
    
    @staticmethod
    def wait_for_port(host: str, port: int, timeout: float = 10.0) -> bool:
        """Wait for a port to become available"""
        import socket
        import time
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                result = sock.connect_ex((host, port))
                sock.close()
                
                if result == 0:
                    return True
                    
            except Exception:
                pass
            
            time.sleep(0.1)
        
        return False
    
    @staticmethod
    def wait_for_http_service(url: str, timeout: float = 30.0) -> bool:
        """Wait for HTTP service to become available"""
        import requests
        import time
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=2.0, verify=False)
                if response.status_code < 500:
                    return True
            except Exception:
                pass
            
            time.sleep(1.0)
        
        return False
    
    @staticmethod
    def generate_test_data(data_type: str, count: int = 10) -> list[dict[str, Any]]:
        """Generate test data for various types"""
        import random
        import uuid
        from datetime import datetime, timedelta
        
        if data_type == "metrics":
            return [
                {
                    "name": f"test_metric_{i}",
                    "value": random.uniform(0, 100),
                    "service": "test_service",
                    "timestamp": datetime.now() - timedelta(minutes=i),
                    "labels": {"test": "true", "instance": str(i)}
                }
                for i in range(count)
            ]
        
        elif data_type == "alerts":
            severities = ["info", "warning", "error", "critical"]
            return [
                {
                    "alert_id": str(uuid.uuid4()),
                    "name": f"test_alert_{i}",
                    "severity": random.choice(severities),
                    "message": f"Test alert message {i}",
                    "service": "test_service",
                    "triggered_at": datetime.now() - timedelta(minutes=i*5)
                }
                for i in range(count)
            ]
        
        elif data_type == "processes":
            states = ["running", "stopped", "failed", "starting"]
            return [
                {
                    "name": f"test_process_{i}",
                    "pid": random.randint(1000, 9999),
                    "state": random.choice(states),
                    "cpu_percent": random.uniform(0, 100),
                    "memory_mb": random.uniform(50, 500),
                    "start_time": datetime.now() - timedelta(hours=i)
                }
                for i in range(count)
            ]
        
        else:
            return [{"id": i, "data": f"test_data_{i}"} for i in range(count)]


# Export commonly used items
__all__ = [
    "IntegrationTestConfig",
    "IntegrationTestHelper", 
    "TEST_DATA_DIR",
    "TEST_SSL_DIR"
]