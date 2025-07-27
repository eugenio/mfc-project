#!/usr/bin/env python3
"""
GitLab Issue Management for MFC Q-Learning Project.
Creates, updates, and manages issues based on test results and bug reports.
"""

import os
import sys
import json
import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

try:
    import gitlab
    GITLAB_AVAILABLE = True
except ImportError:
    GITLAB_AVAILABLE = False
    print("Warning: python-gitlab not available. Install with: pip install python-gitlab")


class IssueSeverity(Enum):
    """Issue severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IssueUrgency(Enum):
    """Issue urgency levels."""
    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IssueType(Enum):
    """Issue types."""
    BUG = "bug"
    ENHANCEMENT = "enhancement"
    PERFORMANCE = "performance"
    SECURITY = "security"
    DOCUMENTATION = "documentation"
    TEST = "test"


@dataclass
class IssueData:
    """Issue data structure."""
    title: str
    description: str
    severity: IssueSeverity
    urgency: IssueUrgency
    issue_type: IssueType
    labels: List[str]
    component: Optional[str] = None
    test_case: Optional[str] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None


class GitLabIssueManager:
    """Manages GitLab issues for the MFC project."""
    
    def __init__(self, project_id: Optional[str] = None, token: Optional[str] = None):
        """Initialize GitLab issue manager."""
        if not GITLAB_AVAILABLE:
            raise ImportError("python-gitlab is required. Install with: pip install python-gitlab")
        
        # Get configuration from environment
        self.gitlab_url = os.getenv('GITLAB_URL', 'https://gitlab.com')
        self.project_id = project_id or os.getenv('GITLAB_PROJECT_ID')
        self.token = token or os.getenv('GITLAB_TOKEN')
        
        if not self.token:
            raise ValueError("GitLab token is required. Set GITLAB_TOKEN environment variable.")
        
        if not self.project_id:
            raise ValueError("GitLab project ID is required. Set GITLAB_PROJECT_ID environment variable.")
        
        # Initialize GitLab connection
        self.gl = gitlab.Gitlab(self.gitlab_url, private_token=self.token)
        try:
            self.project = self.gl.projects.get(self.project_id)
        except gitlab.exceptions.GitlabGetError as e:
            raise ValueError(f"Cannot access GitLab project {self.project_id}: {e}")
    
    def create_issue(self, issue_data: IssueData) -> Dict:
        """Create a new GitLab issue."""
        try:
            # Prepare issue content
            labels = self._prepare_labels(issue_data)
            
            # Create issue
            issue = self.project.issues.create({
                'title': issue_data.title,
                'description': self._format_description(issue_data),
                'labels': labels,
                'assignee_ids': [],  # Can be set if needed
                'milestone_id': None,  # Can be set if needed
            })
            
            print(f"✅ Created issue #{issue.iid}: {issue_data.title}")
            
            return {
                'id': issue.id,
                'iid': issue.iid,
                'title': issue.title,
                'web_url': issue.web_url,
                'state': issue.state,
                'labels': issue.labels
            }
            
        except gitlab.exceptions.GitlabCreateError as e:
            print(f"❌ Failed to create issue: {e}")
            raise
    
    def update_issue(self, issue_iid: int, updates: Dict) -> Dict:
        """Update an existing GitLab issue."""
        try:
            issue = self.project.issues.get(issue_iid)
            
            # Update issue
            for key, value in updates.items():
                setattr(issue, key, value)
            
            issue.save()
            
            print(f"✅ Updated issue #{issue.iid}: {issue.title}")
            
            return {
                'id': issue.id,
                'iid': issue.iid,
                'title': issue.title,
                'web_url': issue.web_url,
                'state': issue.state,
                'labels': issue.labels
            }
            
        except gitlab.exceptions.GitlabGetError as e:
            print(f"❌ Issue #{issue_iid} not found: {e}")
            raise
        except gitlab.exceptions.GitlabUpdateError as e:
            print(f"❌ Failed to update issue #{issue_iid}: {e}")
            raise
    
    def close_issue(self, issue_iid: int, comment: Optional[str] = None) -> Dict:
        """Close a GitLab issue."""
        try:
            issue = self.project.issues.get(issue_iid)
            
            # Add closing comment if provided
            if comment:
                issue.notes.create({'body': comment})
            
            # Close issue
            issue.state_event = 'close'
            issue.save()
            
            print(f"✅ Closed issue #{issue.iid}: {issue.title}")
            
            return {
                'id': issue.id,
                'iid': issue.iid,
                'title': issue.title,
                'web_url': issue.web_url,
                'state': issue.state
            }
            
        except gitlab.exceptions.GitlabGetError as e:
            print(f"❌ Issue #{issue_iid} not found: {e}")
            raise
    
    def search_issues(self, search_terms: List[str], state: str = 'opened') -> List[Dict]:
        """Search for existing issues."""
        try:
            issues = self.project.issues.list(
                state=state,
                search=' '.join(search_terms),
                all=True
            )
            
            results = []
            for issue in issues:
                results.append({
                    'id': issue.id,
                    'iid': issue.iid,
                    'title': issue.title,
                    'description': issue.description,
                    'web_url': issue.web_url,
                    'state': issue.state,
                    'labels': issue.labels,
                    'created_at': issue.created_at,
                    'updated_at': issue.updated_at
                })
            
            return results
            
        except gitlab.exceptions.GitlabListError as e:
            print(f"❌ Failed to search issues: {e}")
            return []
    

    
    def _prepare_labels(self, issue_data: IssueData) -> List[str]:
        """Prepare labels for the issue."""
        labels = []
        
        # Add type label
        labels.append(issue_data.issue_type.value)
        
        # Add severity label
        labels.append(f"severity::{issue_data.severity.value}")
        
        # Add urgency label
        labels.append(f"urgency::{issue_data.urgency.value}")
        
        # Add component label if provided
        if issue_data.component:
            labels.append(f"component::{issue_data.component}")
        
        # Add test label if this is from a test case
        if issue_data.test_case:
            labels.append("test-failure")
        
        # Add custom labels
        labels.extend(issue_data.labels)
        
        return labels
    
    def _format_description(self, issue_data: IssueData) -> str:
        """Format issue description."""
        description = issue_data.description
        
        # Add metadata section
        metadata = []
        metadata.append("## Issue Metadata")
        metadata.append(f"- **Type**: {issue_data.issue_type.value}")
        metadata.append(f"- **Severity**: {issue_data.severity.value}")
        metadata.append(f"- **Urgency**: {issue_data.urgency.value}")
        
        if issue_data.component:
            metadata.append(f"- **Component**: {issue_data.component}")
        
        if issue_data.test_case:
            metadata.append(f"- **Test Case**: {issue_data.test_case}")
        
        metadata.append(f"- **Created**: {datetime.datetime.now().isoformat()}")
        
        # Add error information if available
        if issue_data.error_message:
            metadata.append("")
            metadata.append("## Error Details")
            metadata.append("```")
            metadata.append(issue_data.error_message)
            metadata.append("```")
        
        if issue_data.stack_trace:
            metadata.append("")
            metadata.append("## Stack Trace")
            metadata.append("```")
            metadata.append(issue_data.stack_trace)
            metadata.append("```")
        
        # Add footer
        metadata.append("")
        metadata.append("---")
        metadata.append("*This issue was automatically created by the MFC Q-Learning test suite.*")
        
        return description + "\n\n" + "\n".join(metadata)


class TestResultProcessor:
    """Processes test results and creates appropriate issues."""
    
    def __init__(self, issue_manager: GitLabIssueManager):
        """Initialize test result processor."""
        self.issue_manager = issue_manager
    
    def process_test_failures(self, test_results: Dict) -> List[Dict]:
        """Process test failures and create issues."""
        created_issues = []
        
        failures = test_results.get('failures', [])
        errors = test_results.get('errors', [])
        
        # Process failures
        for failure in failures:
            issue_data = self._create_issue_from_failure(failure)
            
            # Check if similar issue already exists
            existing_issue = self.issue_manager.get_issue_by_title(issue_data.title)
            
            if existing_issue:
                print(f"⚠️ Similar issue already exists: #{existing_issue['iid']}")
                # Could update existing issue with new information
                continue
            
            # Create new issue
            created_issue = self.issue_manager.create_issue(issue_data)
            created_issues.append(created_issue)
        
        # Process errors
        for error in errors:
            issue_data = self._create_issue_from_error(error)
            
            # Check if similar issue already exists
            existing_issue = self.issue_manager.get_issue_by_title(issue_data.title)
            
            if existing_issue:
                print(f"⚠️ Similar issue already exists: #{existing_issue['iid']}")
                continue
            
            # Create new issue
            created_issue = self.issue_manager.create_issue(issue_data)
            created_issues.append(created_issue)
        
        return created_issues
    
    def _create_issue_from_failure(self, failure: Tuple[str, str]) -> IssueData:
        """Create issue data from test failure."""
        test_name, traceback = failure
        
        # Extract useful information from test name
        component = self._extract_component_from_test_name(test_name)
        
        # Determine severity based on test type
        severity = self._determine_severity_from_test_name(test_name)
        urgency = self._determine_urgency_from_severity(severity)
        
        # Create title
        title = f"Test Failure: {test_name}"
        
        # Create description
        description = f"""
## Test Failure Report

The test `{test_name}` is failing consistently.

### Expected Behavior
The test should pass without errors.

### Actual Behavior
The test is failing with assertion errors.

### Impact
This test failure indicates a potential issue with the {component or 'system'} functionality.

### Steps to Reproduce
1. Run the test suite: `python tests/run_tests.py -c {test_name.split('.')[-1] if '.' in test_name else 'all'}`
2. Observe the failure

### Additional Context
This issue was automatically detected during automated testing.
"""
        
        return IssueData(
            title=title,
            description=description,
            severity=severity,
            urgency=urgency,
            issue_type=IssueType.BUG,
            labels=["automated", "test-failure"],
            component=component,
            test_case=test_name,
            error_message=self._extract_error_message(traceback),
            stack_trace=traceback
        )
    
    def _create_issue_from_error(self, error: Tuple[str, str]) -> IssueData:
        """Create issue data from test error."""
        test_name, traceback = error
        
        # Extract useful information
        component = self._extract_component_from_test_name(test_name)
        
        # Errors are typically more severe than failures
        severity = IssueSeverity.HIGH
        urgency = IssueUrgency.HIGH
        
        # Create title
        title = f"Test Error: {test_name}"
        
        # Create description
        description = f"""
## Test Error Report

The test `{test_name}` is encountering runtime errors.

### Expected Behavior
The test should execute without runtime errors.

### Actual Behavior
The test is failing with runtime exceptions.

### Impact
This error indicates a serious issue with the {component or 'system'} that prevents proper testing.

### Steps to Reproduce
1. Run the test suite: `python tests/run_tests.py -c {test_name.split('.')[-1] if '.' in test_name else 'all'}`
2. Observe the error

### Additional Context
This issue was automatically detected during automated testing and requires immediate attention.
"""
        
        return IssueData(
            title=title,
            description=description,
            severity=severity,
            urgency=urgency,
            issue_type=IssueType.BUG,
            labels=["automated", "test-error", "runtime-error"],
            component=component,
            test_case=test_name,
            error_message=self._extract_error_message(traceback),
            stack_trace=traceback
        )
    
    def _extract_component_from_test_name(self, test_name: str) -> Optional[str]:
        """Extract component name from test name."""
        if 'gpu' in test_name.lower():
            return 'gpu-acceleration'
        elif 'biofilm' in test_name.lower():
            return 'biofilm-model'
        elif 'metabolic' in test_name.lower():
            return 'metabolic-model'
        elif 'sensor' in test_name.lower():
            return 'sensor-fusion'
        elif 'qlearning' in test_name.lower() or 'q_learning' in test_name.lower():
            return 'q-learning'
        elif 'mfc' in test_name.lower():
            return 'mfc-stack'
        elif 'config' in test_name.lower():
            return 'configuration'
        elif 'path' in test_name.lower():
            return 'path-management'
        else:
            return None
    
    def _determine_severity_from_test_name(self, test_name: str) -> IssueSeverity:
        """Determine issue severity from test name."""
        test_name_lower = test_name.lower()
        
        if any(keyword in test_name_lower for keyword in ['critical', 'security', 'safety']):
            return IssueSeverity.CRITICAL
        elif any(keyword in test_name_lower for keyword in ['performance', 'stress', 'memory']):
            return IssueSeverity.HIGH
        elif any(keyword in test_name_lower for keyword in ['integration', 'core', 'main']):
            return IssueSeverity.MEDIUM
        else:
            return IssueSeverity.LOW
    
    def _determine_urgency_from_severity(self, severity: IssueSeverity) -> IssueUrgency:
        """Determine urgency from severity."""
        if severity == IssueSeverity.CRITICAL:
            return IssueUrgency.URGENT
        elif severity == IssueSeverity.HIGH:
            return IssueUrgency.HIGH
        elif severity == IssueSeverity.MEDIUM:
            return IssueUrgency.MEDIUM
        else:
            return IssueUrgency.LOW
    
    def _extract_error_message(self, traceback: str) -> Optional[str]:
        """Extract error message from traceback."""
        lines = traceback.strip().split('\n')
        
        # Look for the last line that contains an error
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith('File ') and not line.startswith('  '):
                return line
        
        return None


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='GitLab Issue Manager for MFC Q-Learning Project')
    parser.add_argument('--test-results', type=str, help='JSON file with test results')
    parser.add_argument('--create-issue', action='store_true', help='Create a new issue interactively')
    parser.add_argument('--list-issues', action='store_true', help='List open issues')
    parser.add_argument('--close-issue', type=int, help='Close issue by IID')
    
    args = parser.parse_args()
    
    if not GITLAB_AVAILABLE:
        print("❌ python-gitlab is not available. Install with: pip install python-gitlab")
        sys.exit(1)
    
    try:
        issue_manager = GitLabIssueManager()
        
        if args.test_results:
            # Process test results
            with open(args.test_results, 'r') as f:
                test_results = json.load(f)
            
            processor = TestResultProcessor(issue_manager)
            created_issues = processor.process_test_failures(test_results)
            
            print(f"\n📊 Summary: Created {len(created_issues)} issues")
            for issue in created_issues:
                print(f"  #{issue['iid']}: {issue['title']}")
        
        elif args.list_issues:
            # List open issues
            issues = issue_manager.search_issues([''], state='opened')
            
            print(f"\n📋 Open Issues ({len(issues)}):")
            for issue in sorted(issues, key=lambda x: x['iid']):
                labels_str = ', '.join(issue['labels'][:3])  # Show first 3 labels
                print(f"  #{issue['iid']}: {issue['title'][:60]}... [{labels_str}]")
        
        elif args.close_issue:
            # Close specific issue
            issue_manager.close_issue(args.close_issue, "Issue resolved via automated testing.")
        
        elif args.create_issue:
            # Interactive issue creation
            print("Creating new issue interactively...")
            title = input("Issue title: ")
            description = input("Issue description: ")
            
            issue_data = IssueData(
                title=title,
                description=description,
                severity=IssueSeverity.MEDIUM,
                urgency=IssueUrgency.MEDIUM,
                issue_type=IssueType.BUG,
                labels=["manual"]
            )
            
            created_issue = issue_manager.create_issue(issue_data)
            print(f"Created issue: {created_issue['web_url']}")
        
        else:
            parser.print_help()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()