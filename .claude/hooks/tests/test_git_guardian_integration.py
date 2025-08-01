#!/usr/bin/env python3
"""
Test suite for git-commit-guardian integration in hooks.

Tests the integration of git-commit-guardian with:
- pre_tool_use.py chunked edit operations
- enhanced_file_chunking.py file creation
- Fallback mechanisms
- Security and quality validation

Created: 2025-08-01
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import modules to test
from utils.git_guardian import (
    GitGuardianClient, 
    request_guardian_commit, 
    fallback_to_direct_commit,
    start_git_guardian_if_needed
)


class TestGitGuardianClient(unittest.TestCase):
    """Test GitGuardianClient functionality."""
    
    def setUp(self):
        self.client = GitGuardianClient()
    
    def test_initial_state(self):
        """Test initial client state."""
        self.assertFalse(self.client.guardian_active)
        self.assertEqual(self.client.request_timeout, 30)
    
    def test_is_guardian_active(self):
        """Test guardian active check."""
        # Should return False initially
        self.assertFalse(self.client.is_guardian_active())
    
    @patch('sys.stderr')
    def test_start_guardian(self, mock_stderr):
        """Test starting the guardian."""
        result = self.client.start_guardian()
        self.assertTrue(result)
        self.assertTrue(self.client.guardian_active)
    
    @patch('sys.stderr')
    def test_request_commit_without_guardian(self, mock_stderr):
        """Test commit request when guardian is not active."""
        with patch.object(self.client, 'start_guardian', return_value=True):
            success, message = self.client.request_commit(
                files=['test.py'],
                commit_message='Test commit',
                auto_generated=True
            )
            self.assertTrue(success)
            self.assertIn("Commit completed successfully", message)
    
    @patch('sys.stderr')
    def test_scan_files(self, mock_stderr):
        """Test file scanning functionality."""
        self.client.guardian_active = True
        success, results = self.client.scan_files(['test.py'])
        
        self.assertTrue(success)
        self.assertEqual(results['scanned_files'], ['test.py'])
        self.assertEqual(results['secrets_found'], [])
        self.assertEqual(results['quality_issues'], [])
        self.assertEqual(results['overall_status'], 'PASSED')
    
    @patch('sys.stderr')
    def test_check_branch_strategy(self, mock_stderr):
        """Test branch strategy checking."""
        self.client.guardian_active = True
        
        # Test with test files
        success, branch = self.client.check_branch_strategy(['test_file.py'], 'feature')
        self.assertTrue(success)
        self.assertEqual(branch, 'testing')
        
        # Test with doc files
        success, branch = self.client.check_branch_strategy(['README.md'], 'docs')
        self.assertTrue(success)
        self.assertEqual(branch, 'docs')
        
        # Test with config files
        success, branch = self.client.check_branch_strategy(['config.json'], 'config')
        self.assertTrue(success)
        self.assertEqual(branch, 'config')


class TestGitGuardianIntegrationFunctions(unittest.TestCase):
    """Test integration helper functions."""
    
    @patch('utils.git_guardian.GitGuardianClient')
    @patch('sys.stderr')
    def test_request_guardian_commit_success(self, mock_stderr, mock_client_class):
        """Test successful commit request through guardian."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock successful scan and commit
        mock_client.scan_files.return_value = (True, {
            'secrets_found': [],
            'quality_issues': [],
            'overall_status': 'PASSED'
        })
        mock_client.check_branch_strategy.return_value = (True, 'current')
        mock_client.request_commit.return_value = (True, 'Commit successful')
        
        result = request_guardian_commit(
            files=['test.py'],
            commit_message='Test commit',
            change_type='edit'
        )
        
        self.assertTrue(result)
        mock_client.scan_files.assert_called_once_with(['test.py'])
        mock_client.request_commit.assert_called_once()
    
    @patch('utils.git_guardian.GitGuardianClient')
    @patch('sys.stderr')
    def test_request_guardian_commit_with_security_issues(self, mock_stderr, mock_client_class):
        """Test commit blocked due to security issues."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock scan with secrets found
        mock_client.scan_files.return_value = (True, {
            'secrets_found': ['API_KEY found in test.py'],
            'quality_issues': [],
            'overall_status': 'FAILED'
        })
        
        result = request_guardian_commit(
            files=['test.py'],
            commit_message='Test commit'
        )
        
        self.assertFalse(result)
        mock_client.scan_files.assert_called_once()
        mock_client.request_commit.assert_not_called()
    
    @patch('subprocess.run')
    @patch('sys.stderr')
    def test_fallback_to_direct_commit_success(self, mock_stderr, mock_run):
        """Test successful fallback to direct commit."""
        mock_run.return_value = Mock(returncode=0, stdout='', stderr='')
        
        result = fallback_to_direct_commit(['test.py'], 'Test commit')
        
        self.assertTrue(result)
        # Check git add was called
        self.assertEqual(mock_run.call_args_list[0][0][0], ['git', 'add', 'test.py'])
        # Check git commit was called
        self.assertEqual(mock_run.call_args_list[1][0][0][:3], ['git', 'commit', '-m'])
    
    @patch('subprocess.run')
    @patch('sys.stderr')
    def test_fallback_to_direct_commit_failure(self, mock_stderr, mock_run):
        """Test failed fallback to direct commit."""
        # Since check=True is used, it will raise CalledProcessError
        from subprocess import CalledProcessError
        mock_run.side_effect = [
            Mock(returncode=0),  # git add succeeds
            CalledProcessError(1, 'git commit', stderr='Commit failed')  # git commit fails
        ]
        
        result = fallback_to_direct_commit(['test.py'], 'Test commit')
        
        self.assertFalse(result)


class TestPreToolUseIntegration(unittest.TestCase):
    """Test pre_tool_use.py integration with git-commit-guardian."""
    
    def setUp(self):
        # Import pre_tool_use module
        import pre_tool_use
        self.pre_tool_use = pre_tool_use
    
    @patch('pre_tool_use.request_guardian_commit')
    @patch('pre_tool_use.fallback_to_direct_commit')
    @patch('subprocess.run')
    @patch('sys.stderr')
    def test_chunked_edit_with_guardian_success(self, mock_stderr, mock_run, mock_fallback, mock_guardian):
        """Test chunked edit operation using git-commit-guardian."""
        # Mock successful guardian commit
        mock_guardian.return_value = True
        mock_run.return_value = Mock(returncode=0)  # git add success
        
        # Create test config
        config = {
            'commit_message_prefix': 'Auto-commit: ',
            'max_lines_added': 50,
            'max_lines_removed': 50
        }
        
        # Create temporary test file
        test_file = '/tmp/test_chunked_edit.py'
        old_content = "line1\nline2\nline3\n" * 10
        new_content = "new1\nnew2\nnew3\n" * 15
        
        with open(test_file, 'w') as f:
            f.write(old_content)
        
        try:
            # Perform chunked edit
            self.pre_tool_use.perform_chunked_edit(
                test_file, old_content, new_content, config
            )
            
            # Guardian should have been called
            self.assertTrue(mock_guardian.called)
            # Fallback should not have been called
            self.assertFalse(mock_fallback.called)
            
        finally:
            # Cleanup
            if os.path.exists(test_file):
                os.remove(test_file)
    
    @patch('pre_tool_use.request_guardian_commit')
    @patch('pre_tool_use.fallback_to_direct_commit')
    @patch('subprocess.run')
    @patch('sys.stderr')
    def test_chunked_edit_with_guardian_failure_and_fallback(self, mock_stderr, mock_run, mock_fallback, mock_guardian):
        """Test chunked edit with guardian failure triggering fallback."""
        # Mock guardian failure and successful fallback
        mock_guardian.return_value = False
        mock_fallback.return_value = True
        mock_run.return_value = Mock(returncode=0)  # git add success
        
        config = {
            'commit_message_prefix': 'Auto-commit: ',
            'max_lines_added': 50,
            'max_lines_removed': 50
        }
        
        test_file = '/tmp/test_chunked_edit_fallback.py'
        old_content = "line1\nline2\nline3\n"
        new_content = "new1\nnew2\nnew3\n"
        
        with open(test_file, 'w') as f:
            f.write(old_content)
        
        try:
            self.pre_tool_use.perform_chunked_edit(
                test_file, old_content, new_content, config
            )
            
            # Both guardian and fallback should have been called
            self.assertTrue(mock_guardian.called)
            self.assertTrue(mock_fallback.called)
            
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)


class TestEnhancedFileChunkingIntegration(unittest.TestCase):
    """Test enhanced_file_chunking.py integration with git-commit-guardian."""
    
    def setUp(self):
        # Import enhanced_file_chunking module
        import enhanced_file_chunking
        self.chunking = enhanced_file_chunking
    
    @patch('enhanced_file_chunking.request_guardian_commit')
    @patch('enhanced_file_chunking.fallback_to_direct_commit')
    @patch('subprocess.run')
    @patch('sys.stderr')
    def test_create_file_with_chunks_guardian_success(self, mock_stderr, mock_run, mock_fallback, mock_guardian):
        """Test file creation with chunks using git-commit-guardian."""
        # Mock successful guardian commit
        mock_guardian.return_value = True
        mock_run.return_value = Mock(returncode=0)  # git add success
        
        test_file = '/tmp/test_chunked_file.py'
        test_content = """
def function1():
    pass

def function2():
    pass

class TestClass:
    def method(self):
        pass
""" * 10  # Make it large enough to chunk
        
        try:
            # Test file creation with chunks
            config = {
                'max_lines_per_chunk': 50,
                'commit_message_prefix': 'Auto-commit: '
            }
            result = self.chunking.perform_chunked_file_creation(
                test_file, 
                test_content,
                config
            )
            
            self.assertTrue(result)
            # Guardian should have been called for each chunk
            self.assertTrue(mock_guardian.called)
            
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)


class TestGitGuardianLoggging(unittest.TestCase):
    """Test git-commit-guardian logging functionality."""
    
    @patch('pathlib.Path.mkdir')
    @patch('builtins.open', create=True)
    @patch('json.load')
    @patch('json.dump')
    def test_log_guardian_request(self, mock_dump, mock_load, mock_open, mock_mkdir):
        """Test logging of guardian requests."""
        client = GitGuardianClient()
        
        # Mock existing log file
        mock_load.return_value = []
        
        request_data = {
            "operation": "test-operation",
            "files": ["test.py"],
            "timestamp": "2025-08-01T10:00:00"
        }
        
        # This should not raise any exceptions
        client._log_guardian_request(request_data)
        
        # Check that json.dump was called
        self.assertTrue(mock_dump.called)


class TestIntegrationEndToEnd(unittest.TestCase):
    """End-to-end integration tests."""
    
    @patch('sys.stderr')
    def test_start_git_guardian_if_needed(self, mock_stderr):
        """Test starting guardian if needed."""
        with patch('utils.git_guardian.GitGuardianClient.is_guardian_active', return_value=False):
            with patch('utils.git_guardian.GitGuardianClient.start_guardian', return_value=True):
                result = start_git_guardian_if_needed()
                self.assertTrue(result)
    
    @patch('sys.stderr')
    def test_guardian_already_active(self, mock_stderr):
        """Test when guardian is already active."""
        with patch('utils.git_guardian.GitGuardianClient.is_guardian_active', return_value=True):
            result = start_git_guardian_if_needed()
            self.assertTrue(result)


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)