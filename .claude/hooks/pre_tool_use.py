#!/usr/bin/env -S pixi run python
# /// script
# requires-python = ">=3.8"
# ///

import json
import re
from datetime import datetime
from pathlib import Path

from cchooks import PreToolUseContext, create_context
from utils.constants import ensure_session_log_dir

# Create context
c = create_context()
assert isinstance(c, PreToolUseContext)

def load_edit_thresholds():
    """Load edit threshold configuration from settings.json."""
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
            with open(settings_path) as f:
                settings = json.load(f)
                return settings.get('edit_thresholds', default_config)
    except Exception:
        pass

    return default_config

def is_dangerous_rm_command(command):
    """Check if a command is a dangerous rm -rf command."""
    dangerous_patterns = [
        r'\brm\s+(-rf|-fr|-r\s+-f|-f\s+-r)\s+/',
        r'\brm\s+(-rf|-fr|-r\s+-f|-f\s+-r)\s+~',
        r'\brm\s+(-rf|-fr|-r\s+-f|-f\s+-r)\s+\*',
        r'\brm\s+(-rf|-fr|-r\s+-f|-f\s+-r)\s+\.',
        r'\brm\s+(-rf|-fr|-r\s+-f|-f\s+-r)\s+\$',
        r'\brm\s+.*\s+/\s*$',
        r'\brm\s+.*\s+/\*',
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, command, re.IGNORECASE):
            return True

    return False

def check_edit_thresholds():
    """Check if edit operation exceeds configured thresholds."""
    config = load_edit_thresholds()

    if not config.get('enabled', True):
        return False

    if c.tool_name in ['Edit', 'MultiEdit']:
        if c.tool_name == 'Edit':
            old_string = c.tool_input.get('old_string', '')
            new_string = c.tool_input.get('new_string', '')
            lines_removed = len(old_string.splitlines())
            lines_added = len(new_string.splitlines())
            total_changes = lines_removed + lines_added
        else:  # MultiEdit
            total_removed = 0
            total_added = 0
            for edit in c.tool_input.get('edits', []):
                old_string = edit.get('old_string', '')
                new_string = edit.get('new_string', '')
                total_removed += len(old_string.splitlines())
                total_added += len(new_string.splitlines())
            lines_removed = total_removed
            lines_added = total_added
            total_changes = total_removed + total_added

        # Log threshold information
        print(f"DEBUG: Checking thresholds - lines_added={lines_added}, lines_removed={lines_removed}, total={total_changes}")
        print(f"DEBUG: Thresholds - max_added={config['max_lines_added']}, max_removed={config['max_lines_removed']}, max_total={config['max_total_changes']}")

        # Check thresholds (handle -1 as unlimited)
        max_added = config['max_lines_added']
        max_removed = config['max_lines_removed']
        max_total = config['max_total_changes']

        exceeds_added = max_added > 0 and lines_added > max_added
        exceeds_removed = max_removed > 0 and lines_removed > max_removed
        exceeds_total = max_total > 0 and total_changes > max_total

        if exceeds_added or exceeds_removed or exceeds_total:
            print(f"DEBUG: Threshold exceeded: {exceeds_added} or {exceeds_removed} or {exceeds_total}")

            # Check for git-guardian integration
            if config.get('git_guardian_integration', True):
                print("🛡️  Git-commit guardian integration enabled for chunked edit operation")

            return True

    return False

def load_file_creation_thresholds():
    """Load file creation threshold configuration from settings.json."""
    default_config = {
        "max_new_file_lines": -1,  # -1 means unlimited
        "max_files_per_session": 10,
        "enabled": True,
        "auto_commit": True,
        "commit_message_prefix": "Auto-commit: New file created - ",
        "exclude_patterns": ["*.tmp", "*.log", "*.cache", ".DS_Store", "Thumbs.db"],
        "include_patterns": ["*.py", "*.md", "*.yaml", "*.yml", "*.json", "*.toml", "*.txt"],
        "chunking_threshold": 50,
        "git_guardian_integration": True
    }

    try:
        settings_path = Path(__file__).parent.parent / "settings.json"
        if settings_path.exists():
            with open(settings_path) as f:
                settings = json.load(f)
                return settings.get('file_creation_thresholds', default_config)
    except Exception:
        pass

    return default_config

def check_file_creation_thresholds():
    """Check if file creation exceeds thresholds and handle chunking."""
    if c.tool_name == 'Write':
        content = c.tool_input.get('content', '')
        file_path = c.tool_input.get('file_path', '')

        if Path(file_path).exists():
            return False

        lines = len(content.splitlines())
        size = len(content.encode('utf-8'))
        config = load_file_creation_thresholds()

        # Check if file creation should be chunked
        chunking_threshold = config.get('chunking_threshold', 50)

        if lines > chunking_threshold:
            print(f"🔄 Large file creation detected: {file_path}")
            print(f"   - Lines: {lines}")
            print(f"   - Size: {size} bytes")
            print(f"   - Chunking threshold: {chunking_threshold}")

            # Enable chunking for large files
            if config.get('git_guardian_integration', True):
                print("🛡️  Git-commit guardian integration enabled for chunked file creation")

            return True

    return False

# Main hook logic
print("DEBUG: pre_tool_use.py hook started")

# Log to debug file
try:
    with open('/tmp/hook_debug.log', 'a') as f:
        f.write(f"Hook called at {datetime.now()}\n")
except Exception:
    pass

print(f"DEBUG: Received tool_name={c.tool_name}")
print(f"DEBUG: Tool input keys: {list(c.tool_input.keys())}")

# Check edit thresholds
if check_edit_thresholds():
    c.output.exit_deny("Edit operation exceeds configured thresholds")

# Check for chunked file creation
try:
    from enhanced_file_chunking import check_chunked_file_creation
    if check_chunked_file_creation(c.tool_name, c.tool_input):
        c.output.exit_deny("Large file created in chunks, blocking original Write operation")
except ImportError:
    print("DEBUG: Enhanced file chunking not available, using standard approach")

# Check file creation thresholds
if check_file_creation_thresholds():
    print("FILE CREATION: Threshold exceeded, auto-commit triggered")

# Block .env file access
if c.tool_name in ['Write', 'Edit', 'Read'] and '.env' in c.tool_input.get('file_path', ''):
    c.output.exit_deny("Access to .env files containing sensitive data is prohibited. Use .env.sample for template files instead")

# Block dangerous rm commands
if c.tool_name == 'Bash':
    command = c.tool_input.get('command', '')
    if is_dangerous_rm_command(command):
        c.output.exit_deny("Dangerous rm command detected and prevented")

# Session logging
session_id = c.session_id or 'unknown'
log_dir = ensure_session_log_dir(session_id)
log_path = log_dir / 'pre_tool_use.json'

# Read existing log data
log_data = []
if log_path.exists():
    try:
        with open(log_path) as f:
            log_data = json.load(f)
    except (json.JSONDecodeError, ValueError):
        log_data = []

# Append new data
log_data.append({
    'tool_name': c.tool_name,
    'tool_input': c.tool_input,
    'session_id': session_id,
    'timestamp': datetime.now().isoformat()
})

# Write back to file
with open(log_path, 'w') as f:
    json.dump(log_data, f, indent=2)

# Exit success if nothing blocked
c.output.exit_success()
