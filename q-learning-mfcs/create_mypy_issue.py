"""
Create GitLab issue for mypy type checking errors
"""
import sys
import os

from gitlab_issue_manager import GitLabIssueManager, IssueData, IssueType, IssueSeverity, IssueUrgency
