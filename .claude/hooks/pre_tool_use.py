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

def load_file_creation_thresholds():
    """
    Load file creation threshold configuration from settings.json.
    
    Returns:
        dict: File creation threshold configuration with defaults
    """
    default_config = {
        "max_new_file_lines": 100,
        "max_files_per_session": 5,
        "enabled": True,
        "auto_commit": True,
        "commit_message_prefix": "Auto-commit: New file created - ",
        "exclude_patterns": ["*.tmp", "*.log", "*.cache", ".DS_Store", "Thumbs.db"],
        "include_patterns": ["*.py", "*.md", "*.yaml", "*.yml", "*.json", "*.toml", "*.txt"]
    }
    
    try:
        settings_path = Path(__file__).parent.parent / "settings.json"
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                settings = json.load(f)
                return settings.get("file_creation_thresholds", default_config)
    except Exception:
        pass
    
    return default_config

def should_track_file(file_path, config):
    """
    Check if a file should be tracked based on include/exclude patterns.
    
    Args:
        file_path: Path to the file
        config: File creation threshold configuration
        
    Returns:
        bool: True if file should be tracked
    """
    import fnmatch
    
    file_name = os.path.basename(file_path)
    
    # Check exclude patterns first
    for pattern in config.get('exclude_patterns', []):
        if fnmatch.fnmatch(file_name, pattern):
            return False
    
    # Check include patterns
    include_patterns = config.get('include_patterns', [])
    if not include_patterns:
        return True  # If no include patterns, include all (except excluded)
    
    for pattern in include_patterns:
        if fnmatch.fnmatch(file_name, pattern):
            return True
    
    return False

def get_session_file_count():
    """
    Get the number of files created in the current session.
    
    Returns:
        int: Number of files created in current session
    """
    try:
        session_log_dir = ensure_session_log_dir("current")
        file_creation_log = session_log_dir / "file_creations.log"
        
        if not file_creation_log.exists():
            return 0
        
        with open(file_creation_log, 'r') as f:
            return len(f.readlines())
    except Exception:
        return 0

def log_file_creation(file_path):
    """
    Log a file creation event.
    
    Args:
        file_path: Path to the created file
    """
    try:
        session_log_dir = ensure_session_log_dir("current")
        file_creation_log = session_log_dir / "file_creations.log"
        
        with open(file_creation_log, 'a') as f:
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            f.write(f"{timestamp}: {file_path}\n")
    except Exception:
        pass

def check_file_creation_thresholds(tool_name, tool_input):
    """
    Check if file creation operation should trigger auto-commit.
    
    Args:
        tool_name: Name of the tool being used
        tool_input: Tool input parameters
        
    Returns:
        bool: True if auto-commit should be triggered, False otherwise
    """
    # Debug logging
    print(f"DEBUG: check_file_creation_thresholds called with tool_name={tool_name}", file=sys.stderr)
    
    config = load_file_creation_thresholds()
    
    # Skip if threshold checking is disabled
    if not config.get('enabled', True):
        print("DEBUG: File creation threshold checking is disabled", file=sys.stderr)
        return False
    
    # Only check Write tool for new file creation
    if tool_name != 'Write':
        print(f"DEBUG: Skipping - tool is {tool_name}, not Write", file=sys.stderr)
        return False
    
    file_path = tool_input.get('file_path', '')
    content = tool_input.get('content', '')
    
    print(f"DEBUG: Write tool detected - file_path={file_path}", file=sys.stderr)
    
    if not file_path:
        return False
    
    # Check if this is a new file creation
    if os.path.exists(file_path):
        print(f"DEBUG: File already exists, not a new creation: {file_path}", file=sys.stderr)
        return False  # File already exists, not a new creation
    
    # Check if file should be tracked
    if not should_track_file(file_path, config):
        return False
    
    # Check file size threshold
    new_lines = len(content.splitlines()) if content else 0
    max_lines = config.get('max_new_file_lines', 100)
    
    # Check session file count threshold
    current_file_count = get_session_file_count()
    max_files = config.get('max_files_per_session', 5)
    
    threshold_exceeded = (
        new_lines > max_lines or 
        current_file_count >= max_files
    )
    
    if threshold_exceeded:
        print("NEW FILE CREATION THRESHOLD EXCEEDED:", file=sys.stderr)
        print(f"  File: {file_path}", file=sys.stderr)
        print(f"  Lines in new file: {new_lines} (max: {max_lines})", file=sys.stderr)
        print(f"  Files created this session: {current_file_count} (max: {max_files})", file=sys.stderr)
        
        if config.get('auto_commit', True):
            print("AUTO-COMMIT: Will commit new file creation", file=sys.stderr)
            
            # Log the file creation
            log_file_creation(file_path)
            
            # Try to commit any existing staged changes first
            try:
                result = subprocess.run(['git', 'status', '--porcelain'], 
                                      capture_output=True, text=True, cwd=os.getcwd())
                
                if result.stdout.strip():
                    # Stage and commit the new file when it's created
                    commit_message = f"{config.get('commit_message_prefix', 'Auto-commit: New file created - ')}{os.path.basename(file_path)} ({new_lines} lines)"
                    
                    print(f"AUTO-COMMIT: Staging changes for commit", file=sys.stderr)
                    subprocess.run(['git', 'add', '.'], cwd=os.getcwd(), capture_output=True)
                    
                    print(f"AUTO-COMMIT: Creating commit: {commit_message}", file=sys.stderr)
                    subprocess.run(['git', 'commit', '-m', commit_message], 
                                 cwd=os.getcwd(), capture_output=True)
                    
                    print("AUTO-COMMIT: Commit completed successfully", file=sys.stderr)
                    
            except Exception as e:
                print(f"AUTO-COMMIT: Error during commit: {e}", file=sys.stderr)
        
        return True
    
    # If not exceeding thresholds, still log the file creation
    log_file_creation(file_path)
    return False

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
    print("DEBUG: pre_tool_use.py hook started", file=sys.stderr)
    try:
        # Read JSON input from stdin
        input_data = json.load(sys.stdin)
        
        tool_name = input_data.get('tool_name', '')
        tool_input = input_data.get('tool_input', {})
        
        print(f"DEBUG: Received tool_name={tool_name}", file=sys.stderr)
        
        # Check edit thresholds for large file changes
        if check_edit_thresholds(tool_name, tool_input):
            print("BLOCKED: Edit operation exceeds configured thresholds", file=sys.stderr)
            sys.exit(2)  # Exit code 2 blocks tool call and shows error to Claude
        
        # Check file creation thresholds for new file creation
        if check_file_creation_thresholds(tool_name, tool_input):
            print("FILE CREATION: Threshold exceeded, auto-commit triggered", file=sys.stderr)
            # Note: We don't block file creation, just trigger auto-commit
        
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