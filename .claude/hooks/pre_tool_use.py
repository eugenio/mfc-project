#!/usr/bin/env -S pixi run python
# /// script
# requires-python = ">=3.8"
# ///

import json
import sys
import re
import os
import subprocess
from pathlib import Path
from utils.constants import ensure_session_log_dir

def is_dangerous_rm_command(command):
    """
    Comprehensive detection of dangerous rm commands.
    Matches various forms of rm -rf and similar destructive patterns.
    """
    # Normalize command by removing extra spaces and converting to lowercase
    normalized = ' '.join(command.lower().split())
    
    # Pattern 1: Standard rm -rf variations
    patterns = [
        r'\brm\s+.*-[a-z]*r[a-z]*f',  # rm -rf, rm -fr, rm -Rf, etc.
        r'\brm\s+.*-[a-z]*f[a-z]*r',  # rm -fr variations
        r'\brm\s+--recursive\s+--force',  # rm --recursive --force
        r'\brm\s+--force\s+--recursive',  # rm --force --recursive
        r'\brm\s+-r\s+.*-f',  # rm -r ... -f
        r'\brm\s+-f\s+.*-r',  # rm -f ... -r
    ]
    
    # Check for dangerous patterns
    for pattern in patterns:
        if re.search(pattern, normalized):
            return True
    
    # Pattern 2: Check for rm with recursive flag targeting dangerous paths
    dangerous_paths = [
        r'/',           # Root directory
        r'/\*',         # Root with wildcard
        r'~',           # Home directory
        r'~/',          # Home directory path
        r'\$HOME',      # Home environment variable
        r'\.\.',        # Parent directory references
        r'\*',          # Wildcards in general rm -rf context
        r'\.',          # Current directory
        r'\.\s*$',      # Current directory at end of command
    ]
    
    if re.search(r'\brm\s+.*-[a-z]*r', normalized):  # If rm has recursive flag
        for path in dangerous_paths:
            if re.search(path, normalized):
                return True
    
    return False

def is_env_file_access(tool_name, tool_input):
    """
    Check if any tool is trying to access .env files containing sensitive data.
    """
    if tool_name in ['Read', 'Edit', 'MultiEdit', 'Write', 'Bash']:
        # Check file paths for file-based tools
        if tool_name in ['Read', 'Edit', 'MultiEdit', 'Write']:
            file_path = tool_input.get('file_path', '')
            if '.env' in file_path and not file_path.endswith('.env.sample'):
                return True
        
        # Check bash commands for .env file access
        elif tool_name == 'Bash':
            command = tool_input.get('command', '')
            # Pattern to detect .env file access (but allow .env.sample)
            env_patterns = [
                r'\b\.env\b(?!\.sample)',  # .env but not .env.sample
                r'cat\s+.*\.env\b(?!\.sample)',  # cat .env
                r'echo\s+.*>\s*\.env\b(?!\.sample)',  # echo > .env
                r'touch\s+.*\.env\b(?!\.sample)',  # touch .env
                r'cp\s+.*\.env\b(?!\.sample)',  # cp .env
                r'mv\s+.*\.env\b(?!\.sample)',  # mv .env
            ]
            
            for pattern in env_patterns:
                if re.search(pattern, command):
                    return True
    
    return False

def load_edit_thresholds():
    """
    Load edit threshold configuration from settings.json.
    
    Returns:
        dict: Edit threshold configuration with defaults
    """
    default_config = {
        "max_lines_added": 50,
        "max_lines_removed": 50,
        "max_total_changes": 100,
        "enabled": True,
        "auto_commit": True,
        "commit_message_prefix": "Auto-commit: Large file changes detected - "
    }
    
    try:
        settings_path = Path(__file__).parent.parent / "settings.json"
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                settings = json.load(f)
                return settings.get("edit_thresholds", default_config)
    except Exception:
        pass
    
    return default_config

def count_file_lines(file_path):
    """
    Count the number of lines in a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        int: Number of lines in the file, 0 if file doesn't exist
    """
    try:
        if not os.path.exists(file_path):
            return 0
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return len(f.readlines())
    except Exception:
        return 0

def estimate_edit_changes(tool_name, tool_input):
    """
    Estimate the number of lines that will be added/removed for edit operations.
    
    Args:
        tool_name: Name of the tool being used
        tool_input: Tool input parameters
        
    Returns:
        tuple: (lines_added, lines_removed, file_path) or None if not an edit operation
    """
    if tool_name == 'Edit':
        file_path = tool_input.get('file_path', '')
        old_string = tool_input.get('old_string', '')
        new_string = tool_input.get('new_string', '')
        
        if not file_path:
            return None
            
        # Count lines in old and new strings
        old_lines = len(old_string.splitlines()) if old_string else 0
        new_lines = len(new_string.splitlines()) if new_string else 0
        
        lines_removed = old_lines
        lines_added = new_lines
        
        return (lines_added, lines_removed, file_path)
    
    elif tool_name == 'MultiEdit':
        file_path = tool_input.get('file_path', '')
        edits = tool_input.get('edits', [])
        
        if not file_path or not edits:
            return None
            
        total_added = 0
        total_removed = 0
        
        for edit in edits:
            old_string = edit.get('old_string', '')
            new_string = edit.get('new_string', '')
            
            old_lines = len(old_string.splitlines()) if old_string else 0
            new_lines = len(new_string.splitlines()) if new_string else 0
            
            total_removed += old_lines
            total_added += new_lines
        
        return (total_added, total_removed, file_path)
    
    elif tool_name == 'Write':
        file_path = tool_input.get('file_path', '')
        content = tool_input.get('content', '')
        
        if not file_path:
            return None
            
        # For Write operations, count existing lines vs new lines
        existing_lines = count_file_lines(file_path)
        new_lines = len(content.splitlines()) if content else 0
        
        if existing_lines == 0:
            # New file creation
            return (new_lines, 0, file_path)
        else:
            # File replacement
            return (new_lines, existing_lines, file_path)
    
    return None

def perform_auto_commit(file_path, config, changes_summary):
    """
    Perform automatic git commit for large file changes.
    
    Args:
        file_path: Path to the file being modified
        config: Edit threshold configuration
        changes_summary: Summary of changes (lines_added, lines_removed)
    """
    try:
        # Change to the repository root directory
        repo_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], 
                                          cwd=os.path.dirname(file_path), 
                                          stderr=subprocess.DEVNULL).decode().strip()
        
        # Add the specific file
        subprocess.run(['git', 'add', file_path], 
                      cwd=repo_root, 
                      check=True, 
                      capture_output=True)
        
        # Create commit message
        lines_added, lines_removed = changes_summary
        commit_msg = (f"{config['commit_message_prefix']}"
                     f"+{lines_added}/-{lines_removed} lines in {os.path.basename(file_path)}")
        
        # Commit the changes
        subprocess.run(['git', 'commit', '-m', commit_msg], 
                      cwd=repo_root, 
                      check=True, 
                      capture_output=True)
        
        print(f"AUTO-COMMIT: Committed changes to {file_path} (+{lines_added}/-{lines_removed} lines)", 
              file=sys.stderr)
        return True
        
    except subprocess.CalledProcessError:
        print(f"WARNING: Auto-commit failed for {file_path}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"WARNING: Auto-commit error for {file_path}: {e}", file=sys.stderr)
        return False

def check_edit_thresholds(tool_name, tool_input):
    """
    Check if edit operation exceeds configured thresholds.
    
    Args:
        tool_name: Name of the tool being used
        tool_input: Tool input parameters
        
    Returns:
        bool: True if operation should be blocked, False otherwise
    """
    config = load_edit_thresholds()
    
    # Skip if threshold checking is disabled
    if not config.get('enabled', True):
        return False
    
    # Estimate changes for edit operations
    changes = estimate_edit_changes(tool_name, tool_input)
    if not changes:
        return False  # Not an edit operation
    
    lines_added, lines_removed, file_path = changes
    total_changes = lines_added + lines_removed
    
    # Check thresholds
    max_added = config.get('max_lines_added', 50)
    max_removed = config.get('max_lines_removed', 50)
    max_total = config.get('max_total_changes', 100)
    
    threshold_exceeded = (
        lines_added > max_added or 
        lines_removed > max_removed or 
        total_changes > max_total
    )
    
    if threshold_exceeded:
        print("LARGE EDIT DETECTED:", file=sys.stderr)
        print(f"  File: {file_path}", file=sys.stderr)
        print(f"  Lines to add: {lines_added} (max: {max_added})", file=sys.stderr)
        print(f"  Lines to remove: {lines_removed} (max: {max_removed})", file=sys.stderr)
        print(f"  Total changes: {total_changes} (max: {max_total})", file=sys.stderr)
        
        if config.get('auto_commit', True):
            print("AUTO-COMMIT: Will commit existing changes before proceeding", file=sys.stderr)
            
            # Try to commit any existing staged changes first
            try:
                result = subprocess.run(['git', 'status', '--porcelain'], 
                                      capture_output=True, text=True, check=True)
                if result.stdout.strip():
                    subprocess.run(['git', 'add', '.'], check=True, capture_output=True)
                    subprocess.run(['git', 'commit', '-m', 
                                  f"{config['commit_message_prefix']}staging before large edit"], 
                                  check=True, capture_output=True)
                    print("AUTO-COMMIT: Committed existing changes", file=sys.stderr)
            except subprocess.CalledProcessError:
                pass  # No changes to commit or commit failed
        
        # Allow the operation to continue but log the large edit
        print("PROCEEDING: Large edit operation will continue", file=sys.stderr)
        return False  # Don't block, just warn and commit
    
    return False

def main():
    try:
        # Read JSON input from stdin
        input_data = json.load(sys.stdin)
        
        tool_name = input_data.get('tool_name', '')
        tool_input = input_data.get('tool_input', {})
        
        # Check edit thresholds for large file changes
        if check_edit_thresholds(tool_name, tool_input):
            print("BLOCKED: Edit operation exceeds configured thresholds", file=sys.stderr)
            sys.exit(2)  # Exit code 2 blocks tool call and shows error to Claude
        
        # Check for .env file access (blocks access to sensitive environment files)
        if is_env_file_access(tool_name, tool_input):
            print("BLOCKED: Access to .env files containing sensitive data is prohibited", file=sys.stderr)
            print("Use .env.sample for template files instead", file=sys.stderr)
            sys.exit(2)  # Exit code 2 blocks tool call and shows error to Claude
        
        # Check for dangerous rm -rf commands
        if tool_name == 'Bash':
            command = tool_input.get('command', '')
            
            # Block rm -rf commands with comprehensive pattern matching
            if is_dangerous_rm_command(command):
                print("BLOCKED: Dangerous rm command detected and prevented", file=sys.stderr)
                sys.exit(2)  # Exit code 2 blocks tool call and shows error to Claude
        
        # Extract session_id
        session_id = input_data.get('session_id', 'unknown')
        
        # Ensure session log directory exists
        log_dir = ensure_session_log_dir(session_id)
        log_path = log_dir / 'pre_tool_use.json'
        
        # Read existing log data or initialize empty list
        if log_path.exists():
            with open(log_path, 'r') as f:
                try:
                    log_data = json.load(f)
                except (json.JSONDecodeError, ValueError):
                    log_data = []
        else:
            log_data = []
        
        # Append new data
        log_data.append(input_data)
        
        # Write back to file with formatting
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        sys.exit(0)
        
    except json.JSONDecodeError:
        # Gracefully handle JSON decode errors
        sys.exit(0)
    except Exception:
        # Handle any other errors gracefully
        sys.exit(0)

if __name__ == '__main__':
    main()