#!/usr/bin/env -S pixi run python
# /// script
# requires-python = ">=3.8"
# ///

import json
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from utils.constants import ensure_session_log_dir
from utils.gitlab_client import (
    load_gitlab_config, 
    create_hook_failure_issue, 
    create_large_commit_mr,
    get_current_branch,
    test_gitlab_connection
)

def count_recent_commits(branch: str = None, hours: int = 24) -> int:
    """
    Count commits made in the last N hours on a branch.
    
    Args:
        branch: Branch name (defaults to current branch)
        hours: Hours to look back
        
    Returns:
        int: Number of commits in the timeframe
    """
    if not branch:
        branch = get_current_branch()
    
    if not branch:
        return 0
    
    try:
        # Get commits from last N hours
        since_time = f"--since={hours} hours ago"
        result = subprocess.run([
            'git', 'rev-list', '--count', since_time, branch
        ], capture_output=True, text=True, check=True)
        
        return int(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0

def handle_gitlab_integrations(input_data: dict):
    """
    Handle GitLab integrations based on hook data and configuration.
    
    Args:
        input_data: Hook input data from Claude Code
    """
    config = load_gitlab_config()
    
    if not config.get("enabled"):
        return
    
    features = config.get("features", {})
    tool_name = input_data.get("tool_name", "")
    
    # Handle hook failures
    if features.get("auto_issue_on_hook_failure") and "error" in input_data:
        handle_hook_failure(input_data)
    
    # Handle multiple commits triggering MR creation
    if features.get("auto_mr_on_multiple_commits"):
        handle_multiple_commits(features.get("commit_threshold_for_mr", 5))
    
    # Handle successful large edits
    if tool_name in ["Edit", "MultiEdit", "Write"]:
        handle_successful_edit(input_data)

def handle_hook_failure(input_data: dict):
    """
    Create GitLab issue for hook failures.
    
    Args:
        input_data: Hook data containing error information
    """
    try:
        error_info = input_data.get("error", "Unknown error")
        tool_name = input_data.get("tool_name", "Unknown tool")
        file_path = input_data.get("tool_input", {}).get("file_path")
        
        issue = create_hook_failure_issue(
            hook_name=f"post_tool_use ({tool_name})",
            error_message=str(error_info),
            file_path=file_path
        )
        
        if issue:
            print(f"Created GitLab issue for hook failure: {issue['web_url']}", file=sys.stderr)
    
    except Exception as e:
        print(f"Failed to create hook failure issue: {e}", file=sys.stderr)

def handle_multiple_commits(threshold: int):
    """
    Create MR if multiple commits accumulate on a branch.
    
    Args:
        threshold: Number of commits that trigger MR creation
    """
    try:
        branch = get_current_branch()
        
        if not branch or branch == "main":
            return
        
        # Count recent commits (last 24 hours)
        recent_commits = count_recent_commits(branch, 24)
        
        if recent_commits >= threshold:
            # Check if MR already exists for this branch
            if not mr_exists_for_branch(branch):
                mr = create_large_commit_mr(recent_commits, branch)
                
                if mr:
                    print(f"Created auto-MR for {recent_commits} commits: {mr['web_url']}", file=sys.stderr)
    
    except Exception as e:
        print(f"Failed to handle multiple commits: {e}", file=sys.stderr)

def mr_exists_for_branch(branch: str) -> bool:
    """
    Check if a merge request already exists for the given branch.
    
    Args:
        branch: Source branch name
        
    Returns:
        bool: True if MR exists, False otherwise
    """
    try:
        from utils.gitlab_client import get_current_project
        
        project = get_current_project()
        if not project:
            return False
        
        # Get open merge requests for this branch
        mrs = project.mergerequests.list(
            source_branch=branch,
            state='opened'
        )
        
        return len(mrs) > 0
    
    except Exception:
        return False

def handle_successful_edit(input_data: dict):
    """
    Handle successful edit operations for GitLab integration.
    
    Args:
        input_data: Tool input data
    """
    try:
        tool_name = input_data.get("tool_name")
        
        # Log successful operations (could be extended for GitLab commit comments)
        if tool_name in ["Edit", "MultiEdit"]:
            file_path = input_data.get("tool_input", {}).get("file_path")
            if file_path:
                print(f"GitLab: Successful {tool_name} operation on {file_path}", file=sys.stderr)
        
        elif tool_name == "Write":
            file_path = input_data.get("tool_input", {}).get("file_path")
            if file_path:
                print(f"GitLab: Successful file creation: {file_path}", file=sys.stderr)
    
    except Exception as e:
        print(f"Failed to handle successful edit: {e}", file=sys.stderr)

def main():
    try:
        # Read JSON input from stdin
        input_data = json.load(sys.stdin)

        # Extract session_id
        session_id = input_data.get("session_id", "unknown")

        # Ensure session log directory exists
        log_dir = ensure_session_log_dir(session_id)
        log_path = log_dir / "post_tool_use.json"

        # Read existing log data or initialize empty list
        if log_path.exists():
            with open(log_path, "r") as f:
                try:
                    log_data = json.load(f)
                except (json.JSONDecodeError, ValueError):
                    log_data = []
        else:
            log_data = []

        # Process GitLab integrations if enabled
        try:
            handle_gitlab_integrations(input_data)
        except Exception as e:
            print(f"GitLab integration error: {e}", file=sys.stderr)

        # Append new data
        log_data.append(input_data)

        # Write back to file with formatting
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        sys.exit(0)

    except json.JSONDecodeError:
        # Handle JSON decode errors gracefully
        sys.exit(0)
    except Exception:
        # Exit cleanly on any other error
        sys.exit(0)


if __name__ == "__main__":
    main()
