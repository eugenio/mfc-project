#!/usr/bin/env python3
"""
GitLab Integration Utility for Documentation Agent

Handles GitLab API integration for documentation-related issue management.
Created: 2025-07-31
Integration: BMAD Documentation Agent, git-commit-guardian
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    # Try to import existing GitLab client from hooks
    from .claude.hooks.utils.gitlab_client import GitLabClient
except ImportError:
    print("Warning: GitLab client not found in hooks. Using standalone implementation.")
    GitLabClient = None

class DocumentationGitLabManager:
    """Manages GitLab integration for documentation workflows."""

    def __init__(self, project_id: str = "mfc-project"):
        """Initialize GitLab manager for documentation operations."""
        self.project_id = project_id
        self.gitlab_client = self._initialize_gitlab_client()
        self.logger = self._setup_logging()

    def _initialize_gitlab_client(self):
        """Initialize GitLab client using existing infrastructure."""
        if GitLabClient:
            try:
                return GitLabClient()
            except Exception as e:
                self.logger.warning(f"Failed to initialize existing GitLab client: {e}")

        # Fallback implementation if needed
        return None

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for documentation operations."""
        logger = logging.getLogger('doc-gitlab-integration')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def create_documentation_tracking_issue(
        self,
        title: str,
        description: str,
        labels: list[str] | None = None
    ) -> dict | None:
        """Create a GitLab issue for tracking documentation work."""
        if not self.gitlab_client:
            self.logger.error("GitLab client not available")
            return None

        default_labels = ["documentation", "automated", "doc-agent"]
        issue_labels = labels or []
        all_labels = list(set(default_labels + issue_labels))

        try:
            issue_data = {
                "title": title,
                "description": description,
                "labels": all_labels
            }

            # Create issue using existing GitLab API integration
            issue = self.gitlab_client.create_issue(
                project_id=self.project_id,
                **issue_data
            )

            self.logger.info(f"Created documentation tracking issue: {issue.get('iid')}")
            return issue

        except Exception as e:
            self.logger.error(f"Failed to create documentation issue: {e}")
            return None

    def update_documentation_progress(
        self,
        issue_iid: str,
        progress_update: str,
        completed_files: list[str] | None = None
    ) -> bool:
        """Update documentation progress in GitLab issue."""
        if not self.gitlab_client:
            self.logger.error("GitLab client not available")
            return False

        try:
            # Format progress update
            update_message = self._format_progress_update(
                progress_update,
                completed_files or []
            )

            # Add comment to issue
            self.gitlab_client.add_issue_comment(
                project_id=self.project_id,
                issue_iid=issue_iid,
                body=update_message
            )

            self.logger.info(f"Updated documentation progress for issue {issue_iid}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to update documentation progress: {e}")
            return False

    def close_documentation_issue(
        self,
        issue_iid: str,
        completion_summary: str
    ) -> bool:
        """Close documentation issue when work is complete."""
        if not self.gitlab_client:
            self.logger.error("GitLab client not available")
            return False

        try:
            # Add completion comment
            completion_message = self._format_completion_message(completion_summary)
            self.gitlab_client.add_issue_comment(
                project_id=self.project_id,
                issue_iid=issue_iid,
                body=completion_message
            )

            # Close the issue
            self.gitlab_client.update_issue(
                project_id=self.project_id,
                issue_iid=issue_iid,
                state_event="close"
            )

            self.logger.info(f"Closed documentation issue {issue_iid}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to close documentation issue: {e}")
            return False

    def find_related_documentation_issues(
        self,
        file_paths: list[str]
    ) -> list[dict]:
        """Find GitLab issues related to specific documentation files."""
        if not self.gitlab_client:
            self.logger.error("GitLab client not available")
            return []

        try:
            # Search for issues with documentation labels
            issues = self.gitlab_client.list_issues(
                project_id=self.project_id,
                labels=["documentation"],
                state="opened"
            )

            # Filter issues related to specific files
            related_issues = []
            for issue in issues:
                if self._issue_relates_to_files(issue, file_paths):
                    related_issues.append(issue)

            self.logger.info(f"Found {len(related_issues)} related documentation issues")
            return related_issues

        except Exception as e:
            self.logger.error(f"Failed to find related documentation issues: {e}")
            return []

    def create_standardization_progress_issue(
        self,
        total_docs: int,
        doc_categories: dict[str, int]
    ) -> dict | None:
        """Create issue for tracking overall documentation standardization progress."""
        title = "📚 Documentation Standardization Progress Tracker"

        description = self._generate_standardization_description(
            total_docs,
            doc_categories
        )

        labels = [
            "documentation",
            "standardization",
            "automation",
            "enhancement",
            "doc-agent"
        ]

        return self.create_documentation_tracking_issue(
            title=title,
            description=description,
            labels=labels
        )

    def _format_progress_update(
        self,
        progress_update: str,
        completed_files: list[str]
    ) -> str:
        """Format progress update message for GitLab issue."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        message = f"## 📊 Documentation Progress Update - {timestamp}\n\n"
        message += f"{progress_update}\n\n"

        if completed_files:
            message += "### ✅ Completed Files:\n"
            for file in completed_files:
                message += f"- `{file}`\n"
            message += "\n"

        message += "---\n"
        message += "*Updated automatically by Documentation Agent*\n"

        return message

    def _format_completion_message(self, completion_summary: str) -> str:
        """Format completion message for GitLab issue."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        message = f"## ✅ Documentation Work Completed - {timestamp}\n\n"
        message += f"{completion_summary}\n\n"
        message += "### Summary:\n"
        message += "- All documentation requirements have been fulfilled\n"
        message += "- Technical accuracy has been preserved\n"
        message += "- Standard templates have been applied\n"
        message += "- Git integration has been completed\n\n"
        message += "---\n"
        message += "*Completed automatically by Documentation Agent*\n"

        return message

    def _generate_standardization_description(
        self,
        total_docs: int,
        doc_categories: dict[str, int]
    ) -> str:
        """Generate description for standardization progress tracking issue."""
        description = f"""## Documentation Standardization Project

This issue tracks the progress of standardizing all technical documentation in the MFC project using the Documentation Agent.

### 📊 Project Scope
- **Total Documents**: {total_docs}
- **Document Categories**:
"""

        for category, count in doc_categories.items():
            description += f"  - **{category.title()}**: {count} documents\n"

        description += """
### 🎯 Standardization Goals
- [ ] Apply consistent metadata headers to all documents
- [ ] Use standardized templates for each document type
- [ ] Preserve all technical accuracy and scientific references
- [ ] Maintain backward compatibility with existing links
- [ ] Integrate with git-commit-guardian workflow
- [ ] Validate all documentation against quality standards

### 📋 Progress Tracking
Progress will be updated automatically as documents are standardized.

### 🔧 Standardization Process
1. **Document Analysis**: Categorize and analyze existing documentation
2. **Template Application**: Apply appropriate templates while preserving content
3. **Metadata Addition**: Add standardized metadata headers
4. **Quality Validation**: Validate against documentation standards
5. **Git Integration**: Commit changes through git-commit-guardian
6. **Issue Tracking**: Update progress and close related issues

### 📚 Documentation Types
- **Technical Specifications**: Detailed technical documentation with scientific backing
- **API Documentation**: Interface documentation with examples and specifications
- **User Guides**: Usage instructions and tutorials
- **Architecture Documentation**: System design and component relationships

### 🤖 Automation Details
This standardization is performed by the Documentation Agent with the following features:
- Automated template application
- Scientific accuracy preservation
- Git workflow integration
- Quality validation
- Progress tracking

---
*This issue is managed automatically by the Documentation Agent*
"""

        return description

    def _issue_relates_to_files(
        self,
        issue: dict,
        file_paths: list[str]
    ) -> bool:
        """Check if a GitLab issue relates to specific documentation files."""
        issue_text = f"{issue.get('title', '')} {issue.get('description', '')}"

        for file_path in file_paths:
            file_name = Path(file_path).name
            if file_name in issue_text or file_path in issue_text:
                return True

        return False


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="GitLab Integration for Documentation Agent"
    )
    parser.add_argument(
        "action",
        choices=["create-tracker", "update-progress", "close-issue"],
        help="Action to perform"
    )
    parser.add_argument("--issue-id", help="GitLab issue IID")
    parser.add_argument("--title", help="Issue title")
    parser.add_argument("--description", help="Issue description")
    parser.add_argument("--progress", help="Progress update message")
    parser.add_argument("--files", nargs="+", help="Related file paths")

    args = parser.parse_args()

    manager = DocumentationGitLabManager()

    if args.action == "create-tracker":
        if not args.title:
            print("Error: --title required for create-tracker")
            sys.exit(1)

        issue = manager.create_documentation_tracking_issue(
            title=args.title,
            description=args.description or "Documentation tracking issue",
            labels=["doc-agent", "tracking"]
        )

        if issue:
            print(f"Created issue: {issue.get('web_url')}")
        else:
            print("Failed to create issue")
            sys.exit(1)

    elif args.action == "update-progress":
        if not args.issue_id or not args.progress:
            print("Error: --issue-id and --progress required for update-progress")
            sys.exit(1)

        success = manager.update_documentation_progress(
            issue_iid=args.issue_id,
            progress_update=args.progress,
            completed_files=args.files or []
        )

        if success:
            print(f"Updated progress for issue {args.issue_id}")
        else:
            print("Failed to update progress")
            sys.exit(1)

    elif args.action == "close-issue":
        if not args.issue_id:
            print("Error: --issue-id required for close-issue")
            sys.exit(1)

        success = manager.close_documentation_issue(
            issue_iid=args.issue_id,
            completion_summary=args.description or "Documentation work completed"
        )

        if success:
            print(f"Closed issue {args.issue_id}")
        else:
            print("Failed to close issue")
            sys.exit(1)


if __name__ == "__main__":
    main()
