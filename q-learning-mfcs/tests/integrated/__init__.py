"""
Integrated MFC Model Tests Package

This package contains comprehensive test suites for integrated MFC modeling
including multi-physics coupling, system integration, and end-to-end validation.
"""

import pytest
import asyncio
import threading
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import sys
import os

# Add the source directory to the path for imports
project_root = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(project_root))

# Test utilities for integration testing
class IntegrationTestHelper:
    """Helper class for integration testing with shared utilities."""
    
    @staticmethod
    def create_temp_deployment_dir():
        """Create a temporary deployment directory for testing."""
        return tempfile.mkdtemp(prefix="mfc_test_deployment_")
    
    @staticmethod
    def cleanup_temp_dir(temp_dir):
        """Clean up temporary directory."""
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @staticmethod
    def create_mock_service_config(name, **kwargs):
        """Create a mock service configuration."""
        from deployment.process_manager import ProcessConfig, RestartPolicy
        
        default_config = {
            'name': name,
            'command': ['/bin/echo', f'service_{name}'],
            'working_dir': '/tmp',
            'env': {},
            'restart_policy': RestartPolicy.ALWAYS,
            'startup_timeout': 10.0,
            'shutdown_timeout': 5.0,
            'health_check_interval': 2.0,
            'max_restart_attempts': 3,
            'restart_delay': 1.0,
            'memory_limit_mb': None,
            'cpu_limit_percent': None
        }
        default_config.update(kwargs)
        return ProcessConfig(**default_config)
    
    @staticmethod
    def wait_for_condition(condition_func, timeout=10.0, interval=0.1):
        """Wait for a condition to become true."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition_func():
                return True
            time.sleep(interval)
        return False

__all__ = ['IntegrationTestHelper']