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
