#!/usr/bin/env python3
"""
Test suite for hook-specific git-commit-guardian behavior.

Tests specific behaviors like:
- File creation threshold auto-commits
- Edit threshold chunking with guardian
- Security checks and blocks

Created: 2025-08-01
"""

import unittest
from unittest.mock import Mock, patch
import json
import sys
from pathlib import Path

# Add hooks directory to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / '.claude' / 'hooks'))

try:
    import pre_tool_use
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    pre_tool_use = None

import pytest


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Hook modules not available")
class TestFileCreationThresholdWithGuardian(unittest.TestCase):
    """Test file creation threshold behavior with git-commit-guardian."""
    
    @patch('pre_tool_use.get_session_file_count')
    @patch('pre_tool_use.log_file_creation')
    @patch('pre_tool_use.request_guardian_commit')
    @patch('subprocess.run')
    @patch('sys.stderr')
    def test_file_creation_triggers_guardian_on_threshold(self, mock_stderr, mock_run, 
                                                         mock_guardian, mock_log, mock_count):
        """Test that file creation threshold triggers git-commit-guardian."""
        # Setup mocks
        mock_count.return_value = 5  # Already at max files
        mock_guardian.return_value = True
        mock_run.return_value = Mock(returncode=0, stdout='M test.py\n')  # git status shows changes
        
        # Test input
        tool_input = {
            'file_path': '/tmp/new_test_file.py',
            'content': 'print("test")\n' * 150  # Large file
        }
        
        # Test file creation threshold check
        result = pre_tool_use.check_file_creation_thresholds('Write', tool_input)
        
        self.assertTrue(result)
        # When threshold is exceeded, the function returns True
        # log_file_creation is not called in this case because it returns early
    
    @patch('pre_tool_use.get_session_file_count')
    @patch('pre_tool_use.log_file_creation')
    @patch('pre_tool_use.request_guardian_commit')
    @patch('sys.stderr')
    def test_file_creation_no_trigger_below_threshold(self, mock_stderr, mock_guardian, 
                                                      mock_log, mock_count):
        """Test that small files don't trigger auto-commit."""
        mock_count.return_value = 2  # Below threshold
        
        tool_input = {
            'file_path': '/tmp/small_file.py',
            'content': 'print("small")\n'  # Small file
        }
        
        result = pre_tool_use.check_file_creation_thresholds('Write', tool_input)
        
        self.assertFalse(result)
        # Guardian should not have been called
        mock_guardian.assert_not_called()
        # But file creation should still be logged
        mock_log.assert_called_once_with('/tmp/small_file.py')


class TestEditThresholdWithGuardian(unittest.TestCase):
    """Test edit threshold behavior with git-commit-guardian."""
    
    @patch('pre_tool_use.perform_chunked_edit')
    @patch('pre_tool_use.estimate_edit_changes')
    @patch('sys.stderr')
    def test_large_edit_triggers_chunked_with_guardian(self, mock_stderr, mock_estimate, mock_chunked):
        """Test that large edits trigger chunked commits with guardian."""
        # Mock large edit detection
        mock_estimate.return_value = (100, 50, 'test.py')  # 100 lines added, 50 removed
        mock_chunked.return_value = True
        
        tool_input = {
            'file_path': 'test.py',
            'old_string': 'old' * 100,
            'new_string': 'new' * 200
        }
        
        result = pre_tool_use.check_edit_thresholds('Edit', tool_input)
        
        # When threshold is exceeded and chunking succeeds, it returns True (blocks original edit)
        self.assertTrue(result)
        # Chunked edit should have been called
        mock_chunked.assert_called_once()


class TestSecurityBlocks(unittest.TestCase):
    """Test security blocking behaviors."""
    
    def test_env_file_access_blocked(self):
        """Test that .env file access is blocked."""
        # Test Read access
        self.assertTrue(pre_tool_use.is_env_file_access('Read', {'file_path': '.env'}))
        self.assertTrue(pre_tool_use.is_env_file_access('Read', {'file_path': '/path/to/.env'}))
        self.assertFalse(pre_tool_use.is_env_file_access('Read', {'file_path': '.env.sample'}))
        
        # Test Bash commands
        self.assertTrue(pre_tool_use.is_env_file_access('Bash', {'command': 'cat .env'}))
        self.assertTrue(pre_tool_use.is_env_file_access('Bash', {'command': 'echo SECRET > .env'}))
        self.assertFalse(pre_tool_use.is_env_file_access('Bash', {'command': 'cat .env.sample'}))
    
    def test_dangerous_rm_detection(self):
        """Test dangerous rm command detection."""
        # Dangerous commands
        self.assertTrue(pre_tool_use.is_dangerous_rm_command('rm -rf /'))
        self.assertTrue(pre_tool_use.is_dangerous_rm_command('rm -fr ~'))
        self.assertTrue(pre_tool_use.is_dangerous_rm_command('rm --recursive --force /'))
        self.assertTrue(pre_tool_use.is_dangerous_rm_command('rm -r /*'))
        
        # Safe commands
        self.assertFalse(pre_tool_use.is_dangerous_rm_command('rm file.txt'))
        self.assertFalse(pre_tool_use.is_dangerous_rm_command('rm -f old_file.py'))
        self.assertFalse(pre_tool_use.is_dangerous_rm_command('rmdir empty_dir'))


class TestCodeAnalysis(unittest.TestCase):
    """Test code analysis for commit messages."""
    
    def test_analyze_python_code(self):
        """Test Python code analysis."""
        python_code = '''
def test_function():
    """Test docstring"""
    return True

class TestClass:
    def method(self):
        pass

import os
import sys

CONSTANT_VALUE = 42
'''
        
        analysis = pre_tool_use.analyze_code_content(python_code, 'test.py')
        
        self.assertIn('test_function', analysis['functions'])
        self.assertIn('TestClass', analysis['classes'])
        self.assertEqual(len(analysis['imports']), 2)
        self.assertIn('CONSTANT_VALUE', analysis['constants'])
        self.assertIn('Test docstring', analysis['docstrings'])
    
    def test_analyze_javascript_code(self):
        """Test JavaScript code analysis."""
        js_code = '''
import React from 'react';
export { default } from './Component';

function handleClick() {
    console.log('clicked');
}

const arrowFunc = () => {
    return true;
};

class MyComponent {
    render() {
        return null;
    }
}
'''
        
        analysis = pre_tool_use.analyze_code_content(js_code, 'test.js')
        
        self.assertIn('handleClick', analysis['functions'])
        self.assertIn('arrowFunc', analysis['functions'])
        self.assertIn('MyComponent', analysis['classes'])
        self.assertEqual(len(analysis['imports']), 2)
    
    def test_meaningful_commit_message_generation(self):
        """Test generation of meaningful commit messages."""
        # Test create operation
        msg = pre_tool_use.generate_meaningful_commit_message(
            'create', 'new_module.py', '', 'def main():\n    pass\n', 'feat: '
        )
        self.assertIn('new_module.py', msg)
        self.assertIn('2 lines', msg)
        self.assertIn('functions: main', msg)
        
        # Test edit operation
        msg = pre_tool_use.generate_meaningful_commit_message(
            'edit', 'module.py', 
            'def old():\n    pass\n', 
            'def new():\n    pass\ndef added():\n    pass\n',
            'refactor: '
        )
        self.assertIn('module.py', msg)
        self.assertIn('added', msg)


class TestMockHookExecution(unittest.TestCase):
    """Test mock hook execution with guardian integration."""
    
    @patch('sys.stdin')
    @patch('sys.stdout')
    @patch('sys.stderr')
    @patch('pre_tool_use.check_file_creation_thresholds')
    @patch('pre_tool_use.check_edit_thresholds')
    def test_pre_tool_hook_execution(self, mock_edit_check, mock_file_check, 
                                    mock_stderr, mock_stdout, mock_stdin):
        """Test pre_tool_use hook execution flow."""
        # Mock hook input
        hook_input = {
            'tool_name': 'Write',
            'tool_input': {
                'file_path': 'test.py',
                'content': 'print("test")'
            },
            'session_id': 'test-session'
        }
        
        mock_stdin.read.return_value = json.dumps(hook_input)
        mock_file_check.return_value = False
        mock_edit_check.return_value = None
        
        # Import and run main
        with patch('json.load', return_value=hook_input):
            with patch('sys.exit') as mock_exit:
                pre_tool_use.main()
                
                # Should exit with 0 (success)
                mock_exit.assert_called_with(0)
                # File creation check should have been called
                mock_file_check.assert_called_once_with('Write', hook_input['tool_input'])


if __name__ == '__main__':
    unittest.main(verbosity=2)