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
# GitLab integration removed
def get_current_branch():
    """Get current git branch name."""
    try:
        result = subprocess.run(['git', 'branch', '--show-current'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"

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
    """GitLab integrations disabled - no action taken."""
    pass

def handle_hook_failure(input_data: dict):
    """GitLab issue creation disabled - logging only."""
    error_info = input_data.get("error", "Unknown error")
    tool_name = input_data.get("tool_name", "Unknown tool")
    print(f"Hook failure detected for {tool_name}: {error_info}", file=sys.stderr)

def handle_multiple_commits(threshold: int):
    """GitLab MR creation disabled - logging only."""
    branch = get_current_branch()
    if branch and branch != "main":
        recent_commits = count_recent_commits(branch, 24)
        if recent_commits >= threshold:
            print(f"Multiple commits detected ({recent_commits}) on branch {branch}", file=sys.stderr)

def mr_exists_for_branch(branch: str) -> bool:
    """GitLab MR checking disabled - always returns False."""
    return False

def handle_successful_edit(input_data: dict):
    """GitLab integration disabled - logging only."""
    tool_name = input_data.get("tool_name")
    if tool_name in ["Edit", "MultiEdit"]:
        file_path = input_data.get("tool_input", {}).get("file_path")
        if file_path:
            print(f"Successful {tool_name} operation on {file_path}", file=sys.stderr)
    elif tool_name == "Write":
        file_path = input_data.get("tool_input", {}).get("file_path")
        if file_path:
            print(f"Successful file creation: {file_path}", file=sys.stderr)

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
