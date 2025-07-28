"""
Create GitLab issues for test failures found in comprehensive test run.
"""
import sys
import os
from gitlab_issue_manager import GitLabIssueManager, IssueData, IssueType, IssueSeverity, IssueUrgency
