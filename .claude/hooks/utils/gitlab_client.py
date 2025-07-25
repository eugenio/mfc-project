#!/usr/bin/env python3
"""
GitLab API client utility for Claude Code hooks.

This module provides functionality to interact with GitLab repositories,
including creating issues, merge requests, and managing repository data.
"""

import os
import json
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    import gitlab
    from gitlab.exceptions import GitlabError, GitlabAuthenticationError
    GITLAB_AVAILABLE = True
except ImportError:
    GITLAB_AVAILABLE = False
    print("WARNING: python-gitlab not installed. GitLab integration disabled.", file=sys.stderr)

def load_gitlab_config() -> Dict[str, Any]:
    """
    Load GitLab configuration from settings.json or environment variables.
    
    Returns:
        dict: GitLab configuration with URL, token, and project details
    """
    config = {
        "enabled": False,
        "url": None,
        "token": None,
        "project_id": None,
        "default_branch": "main"
    }
    
    # Try to load from settings.json
    try:
        settings_path = Path(__file__).parent.parent.parent / "settings.json"
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                settings = json.load(f)
                gitlab_config = settings.get("gitlab", {})
                config.update(gitlab_config)
    except Exception as e:
        print(f"Warning: Could not load GitLab config from settings: {e}", file=sys.stderr)
    
    # Override with environment variables if available
    env_token = os.getenv("GITLAB_TOKEN") or os.getenv("GITLAB_PRIVATE_TOKEN")
    if env_token:
        config["token"] = env_token
        config["enabled"] = True
    
    env_url = os.getenv("GITLAB_URL")
    if env_url:
        config["url"] = env_url
    
    env_project = os.getenv("GITLAB_PROJECT_ID")
    if env_project:
        config["project_id"] = env_project
    
    return config

def get_gitlab_client() -> Optional[gitlab.Gitlab]:
    """
    Get authenticated GitLab client instance.
    
    Returns:
        gitlab.Gitlab: Authenticated GitLab client or None if not available
    """
    if not GITLAB_AVAILABLE:
        return None
    
    config = load_gitlab_config()
    
    if not config.get("enabled") or not config.get("token") or not config.get("url"):
        print("GitLab integration not properly configured", file=sys.stderr)
        return None
    
    try:
        gl = gitlab.Gitlab(
            url=config["url"],
            private_token=config["token"]
        )
        # Test authentication
        gl.auth()
        return gl
    except GitlabAuthenticationError:
        print("GitLab authentication failed. Check your token.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Failed to connect to GitLab: {e}", file=sys.stderr)
        return None

def get_current_project():
    """
    Get the current GitLab project based on configuration.
    
    Returns:
        gitlab.Project: GitLab project object or None
    """
    gl = get_gitlab_client()
    if not gl:
        return None
    
    config = load_gitlab_config()
    project_id = config.get("project_id")
    
    if not project_id:
        # Try to determine project from git remote
        try:
            result = subprocess.run(
                ['git', 'remote', 'get-url', 'origin'],
                capture_output=True, text=True, check=True
            )
            remote_url = result.stdout.strip()
            
            # Extract project path from GitLab URL
            # Handle both SSH and HTTPS URLs
            if ":" in remote_url and "@" in remote_url:
                # SSH format: git@gitlab.com:user/project.git
                project_path = remote_url.split(":")[-1].replace(".git", "")
            elif remote_url.startswith("http"):
                # HTTPS format: https://gitlab.com/user/project.git
                project_path = "/".join(remote_url.split("/")[-2:]).replace(".git", "")
            else:
                print(f"Cannot parse GitLab URL: {remote_url}", file=sys.stderr)
                return None
            
            project = gl.projects.get(project_path)
            return project
            
        except subprocess.CalledProcessError:
            print("Cannot determine GitLab project from git remote", file=sys.stderr)
            return None
        except GitlabError as e:
            print(f"GitLab project not found: {e}", file=sys.stderr)
            return None
    
    try:
        project = gl.projects.get(project_id)
        return project
    except GitlabError as e:
        print(f"GitLab project not accessible: {e}", file=sys.stderr)
        return None

def create_issue(title: str, description: str, labels: List[str] = None, 
                assignee: str = None) -> Optional[Dict[str, Any]]:
    """
    Create a GitLab issue.
    
    Args:
        title: Issue title
        description: Issue description
        labels: List of label names
        assignee: Username to assign the issue to
        
    Returns:
        dict: Issue data or None if failed
    """
    project = get_current_project()
    if not project:
        return None
    
    try:
        issue_data = {
            'title': title,
            'description': description
        }
        
        if labels:
            issue_data['labels'] = labels
        
        if assignee:
            # Find user by username
            gl = get_gitlab_client()
            users = gl.users.list(username=assignee)
            if users:
                issue_data['assignee_id'] = users[0].id
        
        issue = project.issues.create(issue_data)
        
        print(f"Created GitLab issue #{issue.iid}: {issue.web_url}", file=sys.stderr)
        
        return {
            'id': issue.id,
            'iid': issue.iid,
            'title': issue.title,
            'web_url': issue.web_url,
            'state': issue.state
        }
        
    except GitlabError as e:
        print(f"Failed to create GitLab issue: {e}", file=sys.stderr)
        return None

def get_project_issues(state: str = "all", labels: List[str] = None, 
                      assignee: str = None, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get all issues from the GitLab project.
    
    Args:
        state: Issue state filter ('opened', 'closed', 'all')
        labels: Filter by labels
        assignee: Filter by assignee username
        limit: Maximum number of issues to return
        
    Returns:
        list: List of issue dictionaries
    """
    project = get_current_project()
    if not project:
        return []
    
    try:
        issues_params = {
            'state': state,
            'per_page': min(limit, 100),
            'all': True if limit > 100 else False
        }
        
        if labels:
            issues_params['labels'] = ','.join(labels)
        
        if assignee:
            # Find user by username
            gl = get_gitlab_client()
            users = gl.users.list(username=assignee)
            if users:
                issues_params['assignee_id'] = users[0].id
        
        issues = project.issues.list(**issues_params)
        
        # Convert to dictionary format
        result = []
        for issue in issues[:limit]:
            issue_dict = {
                'id': issue.id,
                'iid': issue.iid,
                'title': issue.title,
                'description': issue.description,
                'state': issue.state,
                'created_at': issue.created_at,
                'updated_at': issue.updated_at,
                'web_url': issue.web_url,
                'labels': issue.labels,
                'author': {
                    'name': issue.author.get('name', ''),
                    'username': issue.author.get('username', '')
                } if hasattr(issue, 'author') and issue.author else {}
            }
            
            # Add assignee if present
            if hasattr(issue, 'assignee') and issue.assignee:
                issue_dict['assignee'] = {
                    'name': issue.assignee.get('name', ''),
                    'username': issue.assignee.get('username', '')
                }
            
            result.append(issue_dict)
        
        print(f"Retrieved {len(result)} issues from GitLab project", file=sys.stderr)
        return result
        
    except GitlabError as e:
        print(f"Failed to get GitLab issues: {e}", file=sys.stderr)
        return []

def create_merge_request(source_branch: str, target_branch: str = None,
                        title: str = None, description: str = None,
                        remove_source_branch: bool = True) -> Optional[Dict[str, Any]]:
    """
    Create a GitLab merge request.
    
    Args:
        source_branch: Source branch name
        target_branch: Target branch name (defaults to project default)
        title: MR title
        description: MR description
        remove_source_branch: Whether to remove source branch when merged
        
    Returns:
        dict: Merge request data or None if failed
    """
    project = get_current_project()
    if not project:
        return None
    
    try:
        config = load_gitlab_config()
        if not target_branch:
            target_branch = config.get("default_branch", "main")
        
        if not title:
            title = f"Merge {source_branch} into {target_branch}"
        
        if not description:
            description = f"Automatically created merge request for branch {source_branch}"
        
        mr_data = {
            'source_branch': source_branch,
            'target_branch': target_branch,
            'title': title,
            'description': description,
            'remove_source_branch': remove_source_branch
        }
        
        mr = project.mergerequests.create(mr_data)
        
        print(f"Created GitLab MR !{mr.iid}: {mr.web_url}", file=sys.stderr)
        
        return {
            'id': mr.id,
            'iid': mr.iid,
            'title': mr.title,
            'web_url': mr.web_url,
            'state': mr.state,
            'source_branch': mr.source_branch,
            'target_branch': mr.target_branch
        }
        
    except GitlabError as e:
        print(f"Failed to create GitLab merge request: {e}", file=sys.stderr)
        return None

def add_commit_comment(commit_sha: str, comment: str, 
                      file_path: str = None, line: int = None) -> bool:
    """
    Add a comment to a specific commit.
    
    Args:
        commit_sha: Commit SHA to comment on
        comment: Comment text
        file_path: File path for inline comments
        line: Line number for inline comments
        
    Returns:
        bool: True if successful, False otherwise
    """
    project = get_current_project()
    if not project:
        return False
    
    try:
        commit = project.commits.get(commit_sha)
        
        comment_data = {'note': comment}
        
        if file_path and line:
            comment_data.update({
                'path': file_path,
                'line': line,
                'line_type': 'new'
            })
        
        commit.discussions.create({'body': comment})
        
        print(f"Added comment to commit {commit_sha[:8]}", file=sys.stderr)
        return True
        
    except GitlabError as e:
        print(f"Failed to add commit comment: {e}", file=sys.stderr)
        return False

def get_project_info() -> Optional[Dict[str, Any]]:
    """
    Get current project information.
    
    Returns:
        dict: Project information or None if not available
    """
    project = get_current_project()
    if not project:
        return None
    
    try:
        return {
            'id': project.id,
            'name': project.name,
            'path': project.path,
            'web_url': project.web_url,
            'default_branch': project.default_branch,
            'description': project.description,
            'visibility': project.visibility
        }
    except GitlabError as e:
        print(f"Failed to get project info: {e}", file=sys.stderr)
        return None

def get_current_branch() -> Optional[str]:
    """
    Get the current git branch name.
    
    Returns:
        str: Branch name or None if not in git repository
    """
    try:
        result = subprocess.run(
            ['git', 'branch', '--show-current'],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None

def create_hook_failure_issue(hook_name: str, error_message: str, 
                             file_path: str = None) -> Optional[Dict[str, Any]]:
    """
    Create an issue for a hook failure.
    
    Args:
        hook_name: Name of the failed hook
        error_message: Error message details
        file_path: File that caused the failure (if applicable)
        
    Returns:
        dict: Issue data or None if failed
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    branch = get_current_branch() or "unknown"
    
    title = f"Hook Failure: {hook_name} on {branch}"
    
    description = f"""
## Hook Failure Report

**Hook**: {hook_name}
**Branch**: {branch}
**Timestamp**: {timestamp}
**File**: {file_path or "N/A"}

### Error Details
```
{error_message}
```

### Environment
- Git Branch: {branch}
- Timestamp: {timestamp}

This issue was automatically created by the Claude Code hooks system.
"""
    
    return create_issue(
        title=title,
        description=description,
        labels=["hook-failure", "automation", "bug"]
    )

def create_large_commit_mr(commit_count: int, branch: str = None) -> Optional[Dict[str, Any]]:
    """
    Create a merge request when many commits accumulate on a branch.
    
    Args:
        commit_count: Number of commits on the branch
        branch: Branch name (defaults to current branch)
        
    Returns:
        dict: Merge request data or None if failed
    """
    if not branch:
        branch = get_current_branch()
    
    if not branch or branch == "main":
        return None
    
    title = f"Auto-MR: {commit_count} commits on {branch}"
    description = f"""
## Automatic Merge Request

This merge request was automatically created because {commit_count} commits have been made on the `{branch}` branch.

### Summary
- **Source Branch**: {branch}
- **Commit Count**: {commit_count}
- **Created**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Please review the changes and merge when ready.

---
*This MR was created automatically by the Claude Code hooks system.*
"""
    
    return create_merge_request(
        source_branch=branch,
        title=title,
        description=description
    )

def test_gitlab_connection() -> bool:
    """
    Test GitLab API connection and configuration.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    if not GITLAB_AVAILABLE:
        print("❌ python-gitlab library not available", file=sys.stderr)
        return False
    
    config = load_gitlab_config()
    print(f"🔧 GitLab Config: enabled={config.get('enabled')}, url={config.get('url')}", file=sys.stderr)
    
    gl = get_gitlab_client()
    if not gl:
        print("❌ GitLab client connection failed", file=sys.stderr)
        return False
    
    print("✅ GitLab client connected", file=sys.stderr)
    
    project = get_current_project()
    if not project:
        print("❌ GitLab project not accessible", file=sys.stderr)
        return False
    
    print(f"✅ GitLab project: {project.name} ({project.web_url})", file=sys.stderr)
    
    branch = get_current_branch()
    print(f"📂 Current branch: {branch}", file=sys.stderr)
    
    return True