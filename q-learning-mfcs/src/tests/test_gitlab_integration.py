#!/usr/bin/env python3
"""
Test suite for GitLab integration functionality

Tests the GitLab API integration and automatic issue creation system.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

try:
    from utils.gitlab_issue_manager import GitLabIssueManager
    from utils.gitlab_auto_issue import AutoIssueDetector, analyze_user_input
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_AVAILABLE = False

class TestGitLabIntegration(unittest.TestCase):
    """Test GitLab API integration functionality."""
    
    def setUp(self):
        """Set up test environment."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("GitLab integration modules not available")
        
        # Mock GitLab connection to avoid real API calls
        self.mock_gitlab = Mock()
        self.mock_project = Mock()
        self.mock_gitlab.projects.get.return_value = self.mock_project
        
    @patch('utils.gitlab_issue_manager.gitlab')
    @patch.dict(os.environ, {'GITLAB_TOKEN': 'test_token', 'GITLAB_PROJECT_ID': '123'})
    def test_connection_setup(self, mock_gitlab_module):
        """Test GitLab connection setup."""
        mock_gitlab_module.Gitlab.return_value = self.mock_gitlab
        
        GitLabIssueManager()
        
        mock_gitlab_module.Gitlab.assert_called_once()
        self.mock_gitlab.projects.get.assert_called_once_with('123')
    
    def test_bug_issue_creation(self):
        """Test bug issue creation formatting."""
        manager = GitLabIssueManager()
        manager.project = self.mock_project
        
        # Mock issue creation
        mock_issue = Mock()
        mock_issue.iid = 42
        mock_issue.web_url = "https://gitlab.com/project/issues/42"
        self.mock_project.issues.create.return_value = mock_issue
        
        issue_id = manager.create_bug_issue(
            title="Test bug",
            description="This is a test bug",
            steps_to_reproduce="1. Do something\n2. See error",
            expected_behavior="Should work correctly"
        )
        
        self.assertEqual(issue_id, 42)
        self.mock_project.issues.create.assert_called_once()
        
        # Check call arguments
        call_args = self.mock_project.issues.create.call_args[0][0]
        self.assertTrue(call_args['title'].startswith('🐛'))
        self.assertIn('bug', call_args['labels'])
        self.assertIn('This is a test bug', call_args['description'])
    
    def test_enhancement_issue_creation(self):
        """Test enhancement issue creation formatting."""
        manager = GitLabIssueManager()
        manager.project = self.mock_project
        
        # Mock issue creation
        mock_issue = Mock()
        mock_issue.iid = 43
        mock_issue.web_url = "https://gitlab.com/project/issues/43"
        self.mock_project.issues.create.return_value = mock_issue
        
        todo_list = ["Task 1", "Task 2", "Task 3"]
        
        issue_id = manager.create_enhancement_issue(
            title="Test enhancement",
            description="This is a test enhancement",
            todo_list=todo_list,
            priority="high"
        )
        
        self.assertEqual(issue_id, 43)
        self.mock_project.issues.create.assert_called_once()
        
        # Check call arguments
        call_args = self.mock_project.issues.create.call_args[0][0]
        self.assertTrue(call_args['title'].startswith('✨'))
        self.assertIn('enhancement', call_args['labels'])
        self.assertIn('priority::high', call_args['labels'])
        self.assertIn('Task 1', call_args['description'])
    
    def test_issue_update(self):
        """Test issue updating functionality."""
        manager = GitLabIssueManager()
        manager.project = self.mock_project
        
        # Mock issue retrieval and update
        mock_issue = Mock()
        mock_notes = Mock()
        mock_issue.notes = mock_notes
        self.mock_project.issues.get.return_value = mock_issue
        
        result = manager.update_issue(42, "Test comment", close=True)
        
        self.assertTrue(result)
        self.mock_project.issues.get.assert_called_once_with(42)
        mock_notes.create.assert_called_once_with({'body': 'Test comment'})
        self.assertEqual(mock_issue.state_event, 'close')
        mock_issue.save.assert_called_once()

class TestAutoIssueDetection(unittest.TestCase):
    """Test automatic issue detection and classification."""
    
    def setUp(self):
        """Set up test environment."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("GitLab integration modules not available")
        
        self.detector = AutoIssueDetector()
    
    def test_bug_detection(self):
        """Test bug detection from user descriptions."""
        bug_descriptions = [
            "There's a bug in the GUI that crashes the application",
            "The simulation is not working correctly and throws an error",
            "I found a problem with the cathode model implementation"
        ]
        
        for desc in bug_descriptions:
            with self.subTest(description=desc):
                analysis = self.detector.analyze_description(desc)
                self.assertEqual(analysis['type'], 'bug')
                self.assertGreater(analysis['confidence'], 0.5)
    
    def test_enhancement_detection(self):
        """Test enhancement detection from user descriptions."""
        enhancement_descriptions = [
            "I would like to add a new feature for cathode modeling",
            "Could we implement better visualization for the results?",
            "It would be great to have export functionality for simulation data"
        ]
        
        for desc in enhancement_descriptions:
            with self.subTest(description=desc):
                analysis = self.detector.analyze_description(desc)
                self.assertEqual(analysis['type'], 'enhancement')
                self.assertGreater(analysis['confidence'], 0.5)
    
    def test_priority_detection(self):
        """Test priority level detection."""
        test_cases = [
            ("This is an urgent bug that needs immediate attention", "high"),
            ("Minor issue that can be fixed when possible", "low"),
            ("Regular bug report without priority indicators", "medium")
        ]
        
        for desc, expected_priority in test_cases:
            with self.subTest(description=desc, priority=expected_priority):
                analysis = self.detector.analyze_description(desc)
                self.assertEqual(analysis['priority'], expected_priority)
    
    def test_title_extraction(self):
        """Test automatic title extraction from descriptions."""
        test_cases = [
            ("GUI crashes when loading large datasets. This happens every time...", 
             "GUI crashes when loading large datasets"),
            ("Add cathode models\nWe need to implement platinum and biological cathode models",
             "Add cathode models"),
            ("Very long description that goes on and on and should be truncated because it's way too long for a title and exceeds the reasonable limit",
             "Very long description that goes on and on and should be truncated because...")
        ]
        
        for desc, expected_title in test_cases:
            with self.subTest(description=desc):
                analysis = self.detector.analyze_description(desc)
                self.assertEqual(analysis['suggested_title'], expected_title)
    
    def test_todo_extraction(self):
        """Test todo list extraction from enhancement descriptions."""
        desc_with_todos = """
        Add new cathode models to the simulation.
        
        Tasks:
        1. Research platinum cathode parameters
        2. Implement biological cathode model
        3. Add validation tests
        """
        
        analysis = self.detector.analyze_description(desc_with_todos)
        # Note: This tests the analysis, actual todo extraction happens in _create_enhancement_from_analysis
        self.assertEqual(analysis['type'], 'enhancement')
    
    @patch('utils.gitlab_issue_manager.create_bug_issue')
    def test_auto_bug_creation(self, mock_create_bug):
        """Test automatic bug issue creation."""
        mock_create_bug.return_value = 42
        
        bug_desc = "The GUI is broken and crashes constantly"
        
        issue_id = self.detector.create_issue_from_description(bug_desc)
        
        self.assertEqual(issue_id, 42)
        mock_create_bug.assert_called_once()
        
        # Check that the call includes the expected arguments
        call_args = mock_create_bug.call_args
        self.assertIn('broken', call_args.kwargs['description'])
    
    @patch('utils.gitlab_issue_manager.create_enhancement_issue')
    def test_auto_enhancement_creation(self, mock_create_enhancement):
        """Test automatic enhancement issue creation."""
        mock_create_enhancement.return_value = 43
        
        enhancement_desc = "I want to add a new feature for better data visualization"
        
        issue_id = self.detector.create_issue_from_description(enhancement_desc)
        
        self.assertEqual(issue_id, 43)
        mock_create_enhancement.assert_called_once()
        
        # Check that the call includes the expected arguments
        call_args = mock_create_enhancement.call_args
        self.assertIn('visualization', call_args.kwargs['description'])

class TestIntegrationWorkflow(unittest.TestCase):
    """Test end-to-end integration workflows."""
    
    def setUp(self):
        """Set up test environment."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("GitLab integration modules not available")
    
    def test_analyze_user_input_function(self):
        """Test the convenience function for analyzing user input."""
        test_input = "There's a critical bug in the simulation"
        
        result = analyze_user_input(test_input)
        
        self.assertIsInstance(result, dict)
        self.assertIn('type', result)
        self.assertIn('confidence', result)
        self.assertIn('priority', result)
        self.assertEqual(result['type'], 'bug')
    
    def test_real_world_scenarios(self):
        """Test with real-world user input scenarios."""
        scenarios = [
            {
                'input': "the gui it's still not updated with real time graphs and the gpu is still active after the simulation ended",
                'expected_type': 'bug',
                'expected_priority': 'medium'
            },
            {
                'input': "now i want you to start thinking how to add a model for the cathode, research on line literature for platinum base and biological based cathode models",
                'expected_type': 'enhancement',
                'expected_priority': 'medium'
            }
        ]
        
        for scenario in scenarios:
            with self.subTest(input=scenario['input'][:50]):
                analysis = analyze_user_input(scenario['input'])
                self.assertEqual(analysis['type'], scenario['expected_type'])
                self.assertEqual(analysis['priority'], scenario['expected_priority'])

def run_tests():
    """Run all GitLab integration tests."""
    print("🧪 Running GitLab Integration Tests")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestGitLabIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestAutoIssueDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationWorkflow))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📋 Test Summary")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n{'✅ All tests passed!' if success else '❌ Some tests failed.'}")
    
    return success

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)