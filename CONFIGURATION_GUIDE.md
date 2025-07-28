# Configuration System Guide
## Overview

The MFC project uses a sophisticated hierarchical configuration system that supports multiple profiles, environment variable substitution, validation, and runtime overrides. This guide covers all aspects of configuration management.
## Configuration Structure

### Main Configuration File
**Location**: `.claude/settings.json`

```json
{
  "hooks": {
    "PreToolUse": [...],
    "PostToolUse": [...],
    "Notification": [...],
    "Stop": [...],
    "SubagentStop": [...],
    "PreCompact": [...],
    "UserPromptSubmit": [...]
  },
  "gitlab": {
    "enabled": true,
    "url": "https://gitlab.com",
    "project_id": null,
    "default_branch": "main",
    "features": {
      "auto_issue_on_test_failure": true,
      "auto_issue_on_build_failure": true,
      "auto_issue_on_performance_regression": true,
      "auto_issue_on_documentation_gap": true,
      "auto_issue_on_hook_failure": true,
      "auto_mr_on_multiple_commits": false,
      "commit_threshold_for_mr": 5
    }
  },
  "chunked_file_creation": {
    "enabled": true,
    "max_new_file_lines": 50,
    "max_lines_per_chunk": 25,
    "min_segments_for_chunking": 3,
    "commit_message_prefix": "Auto-commit: ",
    "supported_extensions": [".py", ".js", ".ts", ".jsx", ".tsx", ".mojo", ".🔥", ".md", ".markdown"]
  }
}
```
## Hook Configuration

### Pre-Tool Use Hooks
Executes before any tool operation:
```json
"PreToolUse": [
  {
    "matcher": "",
    "hooks": [
      {
        "type": "command",
        "command": "pixi run /home/uge/mfc-project/.claude/hooks/pre_tool_use.py"
      },
      {
        "type": "command",
        "command": "pixi run /home/uge/mfc-project/.claude/hooks/send_event.py --source-app mfc-project --event-type PreToolUse --summarize"
      }
    ]
  }
]
```

### Hook Types
- **PreToolUse**: Before tool execution
- **PostToolUse**: After tool execution
- **Notification**: System notifications
- **Stop**: Session termination
- **SubagentStop**: Subagent termination
- **PreCompact**: Before compacting conversation
- **UserPromptSubmit**: User input processing
## GitLab Integration

### Configuration Options
```json
"gitlab": {
  "enabled": true,
  "url": "https://gitlab.com",
  "project_id": null,
  "default_branch": "main",
  "features": {
    "auto_issue_on_test_failure": true,
    "auto_issue_on_build_failure": true,
    "auto_issue_on_performance_regression": true,
    "auto_issue_on_documentation_gap": true,
    "auto_issue_on_hook_failure": true,
    "auto_mr_on_multiple_commits": false,
    "commit_threshold_for_mr": 5
  }
}
```

### Feature Flags
- **auto_issue_on_test_failure**: Create issues for test failures
- **auto_issue_on_build_failure**: Create issues for build failures
- **auto_issue_on_performance_regression**: Track performance issues
- **auto_issue_on_documentation_gap**: Flag missing documentation
- **auto_issue_on_hook_failure**: Report hook execution failures
- **auto_mr_on_multiple_commits**: Auto-create merge requests
- **commit_threshold_for_mr**: Number of commits before MR
