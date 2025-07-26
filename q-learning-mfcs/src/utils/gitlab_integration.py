#!/usr/bin/env python3
"""
GitLab API integration for automatic issue creation and management

Usage:
    export GITLAB_TOKEN=your_token_here
    export GITLAB_PROJECT_ID=project_id_here
    
    python -c "from utils.gitlab_integration import create_enhancement_issue; 
               create_enhancement_issue('Add cathode models', 'Implement platinum and biological cathode models')"
"""

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, List

try:
    import gitlab
    GITLAB_AVAILABLE = True
except ImportError:
    GITLAB_AVAILABLE = False
    print("⚠️  python-gitlab not available. Install with: pixi add python-gitlab")

class GitLabIssueManager:
    """Manages GitLab issues for bug reports and feature requests."""
    
    def __init__(self):
        self.gl = None
        self.project = None
        self._setup_connection()
    
    def _setup_connection(self):
        """Setup GitLab connection."""
        if not GITLAB_AVAILABLE:
            return
            
        token = os.getenv('GITLAB_TOKEN')
        project_id = os.getenv('GITLAB_PROJECT_ID')
        gitlab_url = os.getenv('GITLAB_URL', 'https://gitlab.com')
        
        if not token:
            print("⚠️  GITLAB_TOKEN environment variable not set")
            return
        if not project_id:
            print("⚠️  GITLAB_PROJECT_ID environment variable not set")
            return
            
        try:
            self.gl = gitlab.Gitlab(gitlab_url, private_token=token)
            self.project = self.gl.projects.get(project_id)
            print(f"✅ Connected to GitLab project: {self.project.name}")
        except Exception as e:
            print(f"❌ GitLab connection failed: {e}")
    
    def create_bug_issue(self, title: str, description: str, 
                        steps_to_reproduce: Optional[str] = None,
                        expected_behavior: Optional[str] = None,
                        environment: Optional[str] = None) -> Optional[int]:
        """Create a bug issue."""
        if not self.project:
            print("❌ GitLab not configured")
            return None
        
        issue_desc = f"""## 🐛 Bug Description
{description}

**Reported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
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
            issue = self.project.issues.create({
                'title': f"🐛 {title}",
                'description': issue_desc,
                'labels': ['bug']
            })
            print(f"✅ Created bug issue #{issue.iid}: {issue.web_url}")
            return issue.iid
        except Exception as e:
            print(f"❌ Failed to create bug issue: {e}")
            return None
    
    def create_enhancement_issue(self, title: str, description: str,
                               todo_list: Optional[List[str]] = None,
                               priority: str = 'medium') -> Optional[int]:
        """Create an enhancement issue."""
        if not self.project:
            print("❌ GitLab not configured")
            return None
        
        issue_desc = f"""## ✨ Enhancement Description
{description}

**Requested:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Requester:** Claude Code Assistant
**Priority:** {priority.title()}

"""
        
        if todo_list:
            issue_desc += """## 📝 Implementation Tasks

"""
            for task in todo_list:
                issue_desc += f"- [ ] {task}\n"
        
        labels = ['enhancement']
        if priority == 'high':
            labels.append('priority::high')
        elif priority == 'low':
            labels.append('priority::low')
        
        try:
            issue = self.project.issues.create({
                'title': f"✨ {title}",
                'description': issue_desc,
                'labels': labels
            })
            print(f"✅ Created enhancement issue #{issue.iid}: {issue.web_url}")
            return issue.iid
        except Exception as e:
            print(f"❌ Failed to create enhancement issue: {e}")
            return None
    
    def update_issue(self, issue_iid: int, comment: str, close: bool = False) -> bool:
        """Update an issue with a comment and optionally close it."""
        if not self.project:
            return False
        
        try:
            issue = self.project.issues.get(issue_iid)
            
            # Add comment
            issue.notes.create({'body': comment})
            
            # Close if requested
            if close:
                issue.state_event = 'close'
                issue.save()
            
            print(f"✅ Updated issue #{issue_iid}")
            return True
        except Exception as e:
            print(f"❌ Failed to update issue: {e}")
            return False

# Global instance
gitlab_manager = GitLabIssueManager()

def create_bug_issue(title: str, description: str, **kwargs) -> Optional[int]:
    """Convenience function to create a bug issue."""
    return gitlab_manager.create_bug_issue(title, description, **kwargs)

def create_enhancement_issue(title: str, description: str, **kwargs) -> Optional[int]:
    """Convenience function to create an enhancement issue."""
    return gitlab_manager.create_enhancement_issue(title, description, **kwargs)

def update_issue(issue_iid: int, comment: str, close: bool = False) -> bool:
    """Convenience function to update an issue."""
    return gitlab_manager.update_issue(issue_iid, comment, close)

if __name__ == "__main__":
    print("🔧 GitLab Integration Test")
    if gitlab_manager.project:
        print(f"✅ Connected to: {gitlab_manager.project.name}")
    else:
        print("❌ Not connected to GitLab")
        print("Set GITLAB_TOKEN and GITLAB_PROJECT_ID environment variables")