#!/usr/bin/env python3
"""GitLab Issue Management for MFC Q-Learning Project.

Consolidated module for creating, updating, and managing GitLab issues.
Supports test result processing, bug reports, and feature requests.

Usage:
    export GITLAB_TOKEN=your_token_here
    export GITLAB_PROJECT_ID=project_id_here

    # CLI usage
    python -m utils.gitlab_issue_manager --list-issues
    python -m utils.gitlab_issue_manager --create-issue
    python -m utils.gitlab_issue_manager --close-issue 42

    # Programmatic usage
    from utils.gitlab_issue_manager import (
        GitLabIssueManager,
        IssueData,
        IssueSeverity,
        IssueUrgency,
        IssueType,
        create_bug_issue,
        create_enhancement_issue,
    )
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from dataclasses import dataclass
from enum import Enum

try:
    import gitlab

    GITLAB_AVAILABLE = True
except ImportError:
    GITLAB_AVAILABLE = False


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
    labels: list[str] | None = None
    component: str | None = None
    test_case: str | None = None
    error_message: str | None = None
    stack_trace: str | None = None

    def __post_init__(self):
        if self.labels is None:
            self.labels = []


class GitLabIssueManager:
    """Manages GitLab issues for the MFC project.

    Provides full CRUD operations for GitLab issues with support for
    automated test result processing and structured issue creation.
    """

    def __init__(self, project_id: str | None = None, token: str | None = None) -> None:
        """Initialize GitLab issue manager.

        Args:
            project_id: GitLab project ID. If not provided, uses GITLAB_PROJECT_ID env var.
            token: GitLab API token. If not provided, uses GITLAB_TOKEN env var.

        Raises:
            ImportError: If python-gitlab is not installed.
            ValueError: If token or project_id is not provided.
        """
        if not GITLAB_AVAILABLE:
            msg = "python-gitlab is required. Install with: pip install python-gitlab"
            raise ImportError(msg)

        # Get configuration from environment
        self.gitlab_url = os.getenv("GITLAB_URL", "https://gitlab.com")
        self.project_id = project_id or os.getenv("GITLAB_PROJECT_ID")
        self.token = token or os.getenv("GITLAB_TOKEN")

        if not self.token:
            msg = "GitLab token is required. Set GITLAB_TOKEN environment variable."
            raise ValueError(msg)

        if not self.project_id:
            msg = "GitLab project ID is required. Set GITLAB_PROJECT_ID environment variable."
            raise ValueError(msg)

        # Initialize GitLab connection
        self.gl = gitlab.Gitlab(self.gitlab_url, private_token=self.token)
        try:
            self.project = self.gl.projects.get(self.project_id)
        except gitlab.exceptions.GitlabGetError as e:
            msg = f"Cannot access GitLab project {self.project_id}: {e}"
            raise ValueError(msg) from e

    def create_issue(self, issue_data: IssueData) -> dict:
        """Create a new GitLab issue.

        Args:
            issue_data: Structured issue data.

        Returns:
            Dictionary with created issue details (id, iid, title, web_url, state, labels).
        """
        try:
            # Prepare issue content
            labels = self._prepare_labels(issue_data)

            # Create issue
            issue = self.project.issues.create(
                {
                    "title": issue_data.title,
                    "description": self._format_description(issue_data),
                    "labels": labels,
                    "assignee_ids": [],
                    "milestone_id": None,
                },
            )

            return {
                "id": issue.id,
                "iid": issue.iid,
                "title": issue.title,
                "web_url": issue.web_url,
                "state": issue.state,
                "labels": issue.labels,
            }

        except gitlab.exceptions.GitlabCreateError:
            raise

    def create_bug_issue(
        self,
        title: str,
        description: str,
        steps_to_reproduce: str | None = None,
        expected_behavior: str | None = None,
        environment: str | None = None,
        severity: IssueSeverity = IssueSeverity.MEDIUM,
        urgency: IssueUrgency = IssueUrgency.MEDIUM,
    ) -> dict:
        """Create a bug issue with structured formatting.

        Args:
            title: Bug title (will be prefixed with bug emoji).
            description: Bug description.
            steps_to_reproduce: Steps to reproduce the bug.
            expected_behavior: Expected behavior description.
            environment: Environment information.
            severity: Bug severity level.
            urgency: Bug urgency level.

        Returns:
            Dictionary with created issue details.
        """
        full_desc = f"""## Bug Description
{description}

**Reported:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

        if steps_to_reproduce:
            full_desc += f"""
## Steps to Reproduce
{steps_to_reproduce}
"""

        if expected_behavior:
            full_desc += f"""
## Expected Behavior
{expected_behavior}
"""

        if environment:
            full_desc += f"""
## Environment
{environment}
"""

        issue_data = IssueData(
            title=f"Bug: {title}",
            description=full_desc,
            severity=severity,
            urgency=urgency,
            issue_type=IssueType.BUG,
            labels=["bug"],
        )

        return self.create_issue(issue_data)

    def create_enhancement_issue(
        self,
        title: str,
        description: str,
        todo_list: list[str] | None = None,
        priority: str = "medium",
    ) -> dict:
        """Create an enhancement/feature request issue.

        Args:
            title: Enhancement title (will be prefixed with sparkle emoji).
            description: Enhancement description.
            todo_list: List of implementation tasks.
            priority: Priority level (high, medium, low).

        Returns:
            Dictionary with created issue details.
        """
        # Map priority string to enums
        severity_map = {
            "high": IssueSeverity.HIGH,
            "medium": IssueSeverity.MEDIUM,
            "low": IssueSeverity.LOW,
        }
        urgency_map = {
            "high": IssueUrgency.HIGH,
            "medium": IssueUrgency.MEDIUM,
            "low": IssueUrgency.LOW,
        }

        severity = severity_map.get(priority, IssueSeverity.MEDIUM)
        urgency = urgency_map.get(priority, IssueUrgency.MEDIUM)

        full_desc = f"""## Enhancement Description
{description}

**Requested:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Priority:** {priority.title()}
"""

        if todo_list:
            full_desc += "\n## Implementation Tasks\n\n"
            for task in todo_list:
                full_desc += f"- [ ] {task}\n"

        labels = ["enhancement"]
        if priority == "high":
            labels.append("priority::high")
        elif priority == "low":
            labels.append("priority::low")

        issue_data = IssueData(
            title=f"Enhancement: {title}",
            description=full_desc,
            severity=severity,
            urgency=urgency,
            issue_type=IssueType.ENHANCEMENT,
            labels=labels,
        )

        return self.create_issue(issue_data)

    def update_issue(self, issue_iid: int, updates: dict) -> dict:
        """Update an existing GitLab issue.

        Args:
            issue_iid: Issue internal ID (IID).
            updates: Dictionary of fields to update.

        Returns:
            Dictionary with updated issue details.
        """
        try:
            issue = self.project.issues.get(issue_iid)

            # Update issue
            for key, value in updates.items():
                setattr(issue, key, value)

            issue.save()

            return {
                "id": issue.id,
                "iid": issue.iid,
                "title": issue.title,
                "web_url": issue.web_url,
                "state": issue.state,
                "labels": issue.labels,
            }

        except gitlab.exceptions.GitlabGetError:
            raise
        except gitlab.exceptions.GitlabUpdateError:
            raise

    def add_comment(self, issue_iid: int, comment: str) -> bool:
        """Add a comment to an issue.

        Args:
            issue_iid: Issue internal ID (IID).
            comment: Comment text.

        Returns:
            True if comment was added successfully.
        """
        try:
            issue = self.project.issues.get(issue_iid)
            issue.notes.create({"body": comment})
            return True
        except Exception:
            return False

    def close_issue(self, issue_iid: int, comment: str | None = None) -> dict:
        """Close a GitLab issue.

        Args:
            issue_iid: Issue internal ID (IID).
            comment: Optional closing comment.

        Returns:
            Dictionary with closed issue details.
        """
        try:
            issue = self.project.issues.get(issue_iid)

            # Add closing comment if provided
            if comment:
                issue.notes.create({"body": comment})

            # Close issue
            issue.state_event = "close"
            issue.save()

            return {
                "id": issue.id,
                "iid": issue.iid,
                "title": issue.title,
                "web_url": issue.web_url,
                "state": issue.state,
            }

        except gitlab.exceptions.GitlabGetError:
            raise

    def search_issues(
        self,
        search_terms: list[str],
        state: str = "opened",
    ) -> list[dict]:
        """Search for existing issues.

        Args:
            search_terms: List of search terms.
            state: Issue state filter ('opened', 'closed', 'all').

        Returns:
            List of matching issue dictionaries.
        """
        try:
            issues = self.project.issues.list(
                state=state,
                search=" ".join(search_terms),
                all=True,
            )

            results = []
            for issue in issues:
                results.append(
                    {
                        "id": issue.id,
                        "iid": issue.iid,
                        "title": issue.title,
                        "description": issue.description,
                        "web_url": issue.web_url,
                        "state": issue.state,
                        "labels": issue.labels,
                        "created_at": issue.created_at,
                        "updated_at": issue.updated_at,
                    },
                )

            return results

        except gitlab.exceptions.GitlabListError:
            return []

    def get_issue_by_title(self, title: str) -> dict | None:
        """Get issue by exact title match.

        Args:
            title: Exact title to search for.

        Returns:
            Issue dictionary if found, None otherwise.
        """
        issues = self.search_issues([title])

        for issue in issues:
            if issue["title"] == title:
                return issue

        return None

    def get_issue_details(self, issue_iid: int) -> dict | None:
        """Get full details of a specific issue.

        Args:
            issue_iid: Issue internal ID (IID).

        Returns:
            Dictionary with full issue details including comments, or None if not found.
        """
        try:
            issue = self.project.issues.get(issue_iid)

            # Get all notes/comments for the issue
            notes = issue.notes.list(all=True)
            comments = []
            for note in notes:
                comments.append(
                    {
                        "id": note.id,
                        "author": (
                            note.author.get("name", "Unknown")
                            if note.author
                            else "System"
                        ),
                        "created_at": note.created_at,
                        "updated_at": note.updated_at,
                        "body": note.body,
                        "system": note.system,
                    },
                )

            # Get assignees if any
            assignees = []
            if hasattr(issue, "assignees") and issue.assignees:
                for assignee in issue.assignees:
                    assignees.append(
                        {
                            "id": assignee.get("id"),
                            "name": assignee.get("name"),
                            "username": assignee.get("username"),
                        },
                    )
            elif hasattr(issue, "assignee") and issue.assignee:
                assignees.append(
                    {
                        "id": issue.assignee.get("id"),
                        "name": issue.assignee.get("name"),
                        "username": issue.assignee.get("username"),
                    },
                )

            return {
                "id": issue.id,
                "iid": issue.iid,
                "title": issue.title,
                "description": issue.description,
                "state": issue.state,
                "labels": issue.labels,
                "assignees": assignees,
                "author": {
                    "id": issue.author.get("id") if issue.author else None,
                    "name": issue.author.get("name") if issue.author else "Unknown",
                    "username": (
                        issue.author.get("username") if issue.author else "unknown"
                    ),
                },
                "created_at": issue.created_at,
                "updated_at": issue.updated_at,
                "closed_at": getattr(issue, "closed_at", None),
                "web_url": issue.web_url,
                "milestone": (
                    {
                        "id": issue.milestone.get("id"),
                        "title": issue.milestone.get("title"),
                        "description": issue.milestone.get("description"),
                    }
                    if issue.milestone
                    else None
                ),
                "comments": comments,
                "user_notes_count": issue.user_notes_count,
                "upvotes": getattr(issue, "upvotes", 0),
                "downvotes": getattr(issue, "downvotes", 0),
                "merge_requests_count": getattr(issue, "merge_requests_count", 0),
                "has_tasks": getattr(issue, "has_tasks", False),
                "task_status": getattr(issue, "task_status", None),
                "confidential": getattr(issue, "confidential", False),
                "discussion_locked": getattr(issue, "discussion_locked", False),
                "issue_type": getattr(issue, "issue_type", "issue"),
                "severity": getattr(issue, "severity", None),
                "priority": getattr(issue, "priority", None),
            }

        except gitlab.exceptions.GitlabGetError:
            return None
        except Exception:
            return None

    def _prepare_labels(self, issue_data: IssueData) -> list[str]:
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
        if issue_data.labels:
            labels.extend(issue_data.labels)

        return labels

    def _format_description(self, issue_data: IssueData) -> str:
        """Format issue description with metadata."""
        description = issue_data.description

        # Add metadata section
        metadata = []
        metadata.append("\n## Issue Metadata")
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
        metadata.append(
            "*This issue was automatically created by the MFC Q-Learning test suite.*",
        )

        return description + "\n".join(metadata)


class TestResultProcessor:
    """Processes test results and creates appropriate issues."""

    def __init__(self, issue_manager: GitLabIssueManager) -> None:
        """Initialize test result processor.

        Args:
            issue_manager: GitLabIssueManager instance.
        """
        self.issue_manager = issue_manager

    def process_test_failures(self, test_results: dict) -> list[dict]:
        """Process test failures and create issues.

        Args:
            test_results: Dictionary with 'failures' and 'errors' lists.

        Returns:
            List of created issue dictionaries.
        """
        created_issues = []

        failures = test_results.get("failures", [])
        errors = test_results.get("errors", [])

        # Process failures
        for failure in failures:
            issue_data = self._create_issue_from_failure(failure)

            # Check if similar issue already exists
            existing_issue = self.issue_manager.get_issue_by_title(issue_data.title)

            if existing_issue:
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
                continue

            # Create new issue
            created_issue = self.issue_manager.create_issue(issue_data)
            created_issues.append(created_issue)

        return created_issues

    def _create_issue_from_failure(self, failure: tuple[str, str]) -> IssueData:
        """Create issue data from test failure."""
        test_name, traceback = failure

        component = self._extract_component_from_test_name(test_name)
        severity = self._determine_severity_from_test_name(test_name)
        urgency = self._determine_urgency_from_severity(severity)

        title = f"Test Failure: {test_name}"

        description = f"""## Test Failure Report

The test `{test_name}` is failing consistently.

### Expected Behavior
The test should pass without errors.

### Actual Behavior
The test is failing with assertion errors.

### Impact
This test failure indicates a potential issue with the {component or "system"} functionality.

### Steps to Reproduce
1. Run the test suite
2. Observe the failure
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
            stack_trace=traceback,
        )

    def _create_issue_from_error(self, error: tuple[str, str]) -> IssueData:
        """Create issue data from test error."""
        test_name, traceback = error

        component = self._extract_component_from_test_name(test_name)
        severity = IssueSeverity.HIGH
        urgency = IssueUrgency.HIGH

        title = f"Test Error: {test_name}"

        description = f"""## Test Error Report

The test `{test_name}` is encountering runtime errors.

### Expected Behavior
The test should execute without runtime errors.

### Actual Behavior
The test is failing with runtime exceptions.

### Impact
This error indicates a serious issue with the {component or "system"} that prevents proper testing.
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
            stack_trace=traceback,
        )

    def _extract_component_from_test_name(self, test_name: str) -> str | None:
        """Extract component name from test name."""
        test_name_lower = test_name.lower()

        component_keywords = {
            "gpu": "gpu-acceleration",
            "biofilm": "biofilm-model",
            "metabolic": "metabolic-model",
            "sensor": "sensor-fusion",
            "qlearning": "q-learning",
            "q_learning": "q-learning",
            "mfc": "mfc-stack",
            "config": "configuration",
            "path": "path-management",
        }

        for keyword, component in component_keywords.items():
            if keyword in test_name_lower:
                return component

        return None

    def _determine_severity_from_test_name(self, test_name: str) -> IssueSeverity:
        """Determine issue severity from test name."""
        test_name_lower = test_name.lower()

        if any(
            keyword in test_name_lower for keyword in ["critical", "security", "safety"]
        ):
            return IssueSeverity.CRITICAL
        if any(
            keyword in test_name_lower
            for keyword in ["performance", "stress", "memory"]
        ):
            return IssueSeverity.HIGH
        if any(
            keyword in test_name_lower for keyword in ["integration", "core", "main"]
        ):
            return IssueSeverity.MEDIUM
        return IssueSeverity.LOW

    def _determine_urgency_from_severity(self, severity: IssueSeverity) -> IssueUrgency:
        """Determine urgency from severity."""
        urgency_map = {
            IssueSeverity.CRITICAL: IssueUrgency.URGENT,
            IssueSeverity.HIGH: IssueUrgency.HIGH,
            IssueSeverity.MEDIUM: IssueUrgency.MEDIUM,
            IssueSeverity.LOW: IssueUrgency.LOW,
        }
        return urgency_map[severity]

    def _extract_error_message(self, traceback: str) -> str | None:
        """Extract error message from traceback."""
        lines = traceback.strip().split("\n")

        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith("File ") and not line.startswith("  "):
                return line

        return None


# Convenience functions for simple issue creation
def create_bug_issue(
    title: str,
    description: str,
    steps_to_reproduce: str | None = None,
    expected_behavior: str | None = None,
    environment: str | None = None,
) -> int | None:
    """Create a bug issue.

    Args:
        title: Bug title.
        description: Bug description.
        steps_to_reproduce: Steps to reproduce.
        expected_behavior: Expected behavior.
        environment: Environment info.

    Returns:
        Issue IID if created successfully, None otherwise.
    """
    try:
        manager = GitLabIssueManager()
        result = manager.create_bug_issue(
            title=title,
            description=description,
            steps_to_reproduce=steps_to_reproduce,
            expected_behavior=expected_behavior,
            environment=environment,
        )
        return result.get("iid")
    except Exception:
        return None


def create_enhancement_issue(
    title: str,
    description: str,
    todo_list: list[str] | None = None,
    priority: str = "medium",
) -> int | None:
    """Create an enhancement issue.

    Args:
        title: Enhancement title.
        description: Enhancement description.
        todo_list: Implementation tasks.
        priority: Priority level.

    Returns:
        Issue IID if created successfully, None otherwise.
    """
    try:
        manager = GitLabIssueManager()
        result = manager.create_enhancement_issue(
            title=title,
            description=description,
            todo_list=todo_list,
            priority=priority,
        )
        return result.get("iid")
    except Exception:
        return None


def update_issue(issue_iid: int, comment: str, close: bool = False) -> bool:
    """Update an issue with a comment.

    Args:
        issue_iid: Issue internal ID.
        comment: Comment to add.
        close: Whether to close the issue.

    Returns:
        True if successful, False otherwise.
    """
    try:
        manager = GitLabIssueManager()
        manager.add_comment(issue_iid, comment)
        if close:
            manager.close_issue(issue_iid)
        return True
    except Exception:
        return False


def main() -> None:
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="GitLab Issue Manager for MFC Q-Learning Project",
    )
    parser.add_argument("--test-results", type=str, help="JSON file with test results")
    parser.add_argument(
        "--create-issue",
        action="store_true",
        help="Create a new issue interactively",
    )
    parser.add_argument("--list-issues", action="store_true", help="List open issues")
    parser.add_argument("--close-issue", type=int, help="Close issue by IID")
    parser.add_argument(
        "--get-issue",
        type=int,
        help="Get full details of issue by IID",
    )

    args = parser.parse_args()

    if not GITLAB_AVAILABLE:
        print("GitLab library not available. Install with: pip install python-gitlab")
        sys.exit(1)

    try:
        issue_manager = GitLabIssueManager()

        if args.test_results:
            with open(args.test_results) as f:
                test_results = json.load(f)

            processor = TestResultProcessor(issue_manager)
            created_issues = processor.process_test_failures(test_results)

            print(f"Created {len(created_issues)} issues from test results")
            for issue in created_issues:
                print(f"  #{issue['iid']}: {issue['title']}")

        elif args.list_issues:
            issues = issue_manager.search_issues([""], state="opened")

            print(f"Found {len(issues)} open issues:")
            for issue in sorted(issues, key=lambda x: x["iid"]):
                labels_str = ", ".join(issue["labels"][:3])
                print(f"  #{issue['iid']}: {issue['title']} [{labels_str}]")

        elif args.close_issue:
            result = issue_manager.close_issue(
                args.close_issue,
                "Issue resolved via CLI.",
            )
            print(f"Closed issue #{result['iid']}: {result['title']}")

        elif args.get_issue:
            details = issue_manager.get_issue_details(args.get_issue)
            if details:
                print(f"Issue #{details['iid']}: {details['title']}")
                print(f"State: {details['state']}")
                print(f"Labels: {', '.join(details['labels'])}")
                print(f"URL: {details['web_url']}")
                print(f"\nDescription:\n{details['description'][:500]}...")
            else:
                print(f"Issue #{args.get_issue} not found")

        elif args.create_issue:
            print("Interactive issue creation")
            title = input("Issue title: ").strip()
            if not title:
                print("Title is required")
                sys.exit(1)

            print("Enter description (Ctrl+D to finish):")
            description_lines = []
            try:
                while True:
                    line = input()
                    description_lines.append(line)
            except EOFError:
                pass

            description = "\n".join(description_lines)

            issue_type = input("Type (bug/enhancement) [bug]: ").strip() or "bug"
            severity = input("Severity (critical/high/medium/low) [medium]: ").strip() or "medium"

            issue_data = IssueData(
                title=title,
                description=description,
                severity=IssueSeverity(severity),
                urgency=IssueUrgency(severity),
                issue_type=IssueType.BUG if issue_type == "bug" else IssueType.ENHANCEMENT,
            )

            result = issue_manager.create_issue(issue_data)
            print(f"Created issue #{result['iid']}: {result['web_url']}")

        else:
            parser.print_help()

    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
