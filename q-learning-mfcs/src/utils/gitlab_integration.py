#!/usr/bin/env python3
"""GitLab API integration for automatic issue creation and management.

Usage:
    export GITLAB_TOKEN=your_token_here
    export GITLAB_PROJECT_ID=project_id_here

    python -c "from utils.gitlab_integration import create_enhancement_issue;
               create_enhancement_issue('Add cathode models', 'Implement platinum and biological cathode models')"
"""

from __future__ import annotations

import os
from datetime import datetime

try:
    import gitlab

    GITLAB_AVAILABLE = True
except ImportError:
    GITLAB_AVAILABLE = False

# Load configuration from .env file
try:
    from .config_loader import setup_gitlab_config

    setup_gitlab_config()
except ImportError:
    pass


class GitLabIssueManager:
    """Manages GitLab issues for bug reports and feature requests."""

    def __init__(self) -> None:
        self.gl = None
        self.project = None
        self._setup_connection()

    def _setup_connection(self) -> None:
        """Setup GitLab connection."""
        if not GITLAB_AVAILABLE:
            return

        token = os.getenv("GITLAB_TOKEN")
        project_id = os.getenv("GITLAB_PROJECT_ID")
        gitlab_url = os.getenv("GITLAB_URL", "https://gitlab.com")

        if not token:
            return
        if not project_id:
            return

        try:
            self.gl = gitlab.Gitlab(gitlab_url, private_token=token)
            self.project = self.gl.projects.get(project_id)
        except Exception:
            pass

    def create_bug_issue(
        self,
        title: str,
        description: str,
        steps_to_reproduce: str | None = None,
        expected_behavior: str | None = None,
        environment: str | None = None,
    ) -> int | None:
        """Create a bug issue."""
        if not self.project:
            return None

        issue_desc = f"""## 🐛 Bug Description
{description}

**Reported:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Reporter:** Claude Code Assistant

"""

        if steps_to_reproduce:
            issue_desc += f"""## 🔄 Steps to Reproduce
{steps_to_reproduce}

"""

        if expected_behavior:
            issue_desc += f"""## ✅ Expected Behavior
{expected_behavior}

"""

        if environment:
            issue_desc += f"""## 🖥️ Environment
{environment}

"""

        try:
            issue = self.project.issues.create(
                {"title": f"🐛 {title}", "description": issue_desc, "labels": ["bug"]},
            )
            return issue.iid
        except Exception:
            return None

    def create_enhancement_issue(
        self,
        title: str,
        description: str,
        todo_list: list[str] | None = None,
        priority: str = "medium",
    ) -> int | None:
        """Create an enhancement issue."""
        if not self.project:
            return None

        issue_desc = f"""## ✨ Enhancement Description
{description}

**Requested:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Requester:** Claude Code Assistant
**Priority:** {priority.title()}

"""

        if todo_list:
            issue_desc += """## 📝 Implementation Tasks

"""
            for task in todo_list:
                issue_desc += f"- [ ] {task}\n"

        labels = ["enhancement"]
        if priority == "high":
            labels.append("priority::high")
        elif priority == "low":
            labels.append("priority::low")

        try:
            issue = self.project.issues.create(
                {"title": f"✨ {title}", "description": issue_desc, "labels": labels},
            )
            return issue.iid
        except Exception:
            return None

    def update_issue(self, issue_iid: int, comment: str, close: bool = False) -> bool:
        """Update an issue with a comment and optionally close it."""
        if not self.project:
            return False

        try:
            issue = self.project.issues.get(issue_iid)

            # Add comment
            issue.notes.create({"body": comment})

            # Close if requested
            if close:
                issue.state_event = "close"
                issue.save()

            return True
        except Exception:
            return False


# Global instance
gitlab_manager = GitLabIssueManager()


def create_bug_issue(title: str, description: str, **kwargs) -> int | None:
    """Convenience function to create a bug issue."""
    return gitlab_manager.create_bug_issue(title, description, **kwargs)


def create_enhancement_issue(title: str, description: str, **kwargs) -> int | None:
    """Convenience function to create an enhancement issue."""
    return gitlab_manager.create_enhancement_issue(title, description, **kwargs)


def update_issue(issue_iid: int, comment: str, close: bool = False) -> bool:
    """Convenience function to update an issue."""
    return gitlab_manager.update_issue(issue_iid, comment, close)


if __name__ == "__main__":
    if gitlab_manager.project:
        pass
    else:
        pass
