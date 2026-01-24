#!/usr/bin/env python3
"""GitLab Issue Manager - Backward Compatibility Module.

This module re-exports from the consolidated gitlab_issue_manager in src/utils/.
New code should import directly from utils.gitlab_issue_manager.

Example:
    # New way (preferred)
    from utils.gitlab_issue_manager import GitLabIssueManager

    # Old way (still works for backward compatibility)
    from gitlab_issue_manager import GitLabIssueManager
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Re-export everything from the consolidated module
from utils.gitlab_issue_manager import (  # noqa: E402, F401
    GITLAB_AVAILABLE,
    GitLabIssueManager,
    IssueData,
    IssueSeverity,
    IssueType,
    IssueUrgency,
    TestResultProcessor,
    create_bug_issue,
    create_enhancement_issue,
    main,
    update_issue,
)

__all__ = [
    "GITLAB_AVAILABLE",
    "GitLabIssueManager",
    "IssueData",
    "IssueSeverity",
    "IssueType",
    "IssueUrgency",
    "TestResultProcessor",
    "create_bug_issue",
    "create_enhancement_issue",
    "main",
    "update_issue",
]

if __name__ == "__main__":
    main()
