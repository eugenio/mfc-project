#!/usr/bin/env -S pixi run python
# /// script
# requires-python = ">=3.8"
# ///

"""
PostToolUse Hook - Fixed Version
Handles non-JSON serializable responses by preprocessing input before cchooks validation
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.constants import ensure_session_log_dir


def safe_serialize(obj: Any) -> Any:
    """
    Recursively convert objects to JSON-serializable format

    Args:
        obj: Object to serialize

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: safe_serialize(value) for key, value in obj.items()}
    elif isinstance(obj, list | tuple):
        return [safe_serialize(item) for item in obj]
    elif isinstance(obj, set):
        return [safe_serialize(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        # Convert objects with __dict__ to dictionary
        return safe_serialize(obj.__dict__)
    else:
        # Try to serialize as-is, convert to string if it fails
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)

def get_current_branch() -> str:
    """Get current git branch name."""
    try:
        result = subprocess.run(['git', 'branch', '--show-current'],
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, Exception):
        return "unknown"

def count_recent_commits(branch: str | None = None, hours: int = 24) -> int:
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
    except (subprocess.CalledProcessError, ValueError, Exception):
        return 0

def handle_gitlab_integrations(input_data: dict[str, Any]) -> None:
    """GitLab integrations disabled - no action taken."""
    pass

def handle_auto_commit_if_needed(input_data: dict[str, Any]) -> None:
    """Handle auto-commit for operations that exceed thresholds."""
    try:
        # Check if this was a file creation or modification that should trigger auto-commit
        tool_name = input_data.get('tool_name')
        if tool_name in ['Write', 'Edit', 'MultiEdit']:
            # For now, just log the operation
            tool_input = input_data.get('tool_input', {})
            file_path = tool_input.get('file_path')
            if file_path:
                print(f"File operation detected: {tool_name} on {file_path}", file=sys.stderr)

    except Exception as e:
        print(f"Auto-commit error: {e}", file=sys.stderr)

def main() -> int:
    """
    Main entry point for the PostToolUse hook

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        # Read input from stdin
        input_text = sys.stdin.read()

        # Parse JSON input
        try:
            raw_input_data = json.loads(input_text)
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}", file=sys.stderr)
            return 1

        # Safely serialize the input data to handle non-JSON serializable objects
        input_data = safe_serialize(raw_input_data)

        # Extract session_id
        session_id = input_data.get("session_id", "unknown")

        # Ensure session log directory exists
        log_dir = ensure_session_log_dir(session_id)
        log_path = log_dir / "post_tool_use.json"

        # Read existing log data or initialize empty list
        log_data = []
        if log_path.exists():
            try:
                with open(log_path) as f:
                    log_data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                log_data = []

        # Process GitLab integrations if enabled
        try:
            handle_gitlab_integrations(input_data)
        except Exception as e:
            print(f"GitLab integration error: {e}", file=sys.stderr)

        # Handle auto-commit for large changes
        try:
            handle_auto_commit_if_needed(input_data)
        except Exception as e:
            print(f"Auto-commit integration error: {e}", file=sys.stderr)

        # Create log entry with additional metadata
        log_entry = {
            'tool_name': input_data.get('tool_name'),
            'tool_input': input_data.get('tool_input'),
            'tool_response': input_data.get('tool_response'),
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'serialized_fix_applied': True  # Mark that this version handled serialization
        }

        # Append new data
        log_data.append(log_entry)

        # Write back to file with formatting
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        # Success
        return 0

    except Exception as e:
        print(f"❌ Unexpected error in PostToolUse hook: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
