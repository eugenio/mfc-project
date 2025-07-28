# Hooks and Automation System
## Overview

The MFC project features a sophisticated automation system built around Claude Code hooks. This system provides intelligent file handling, automated issue tracking, event logging, and development workflow enhancements.
## Hook System Architecture

### Hook Types and Execution Order

```
User Action
    │
    ├─► UserPromptSubmit Hook
    │       ├─► Log user input
    │       └─► Send event
    │
    ├─► PreToolUse Hook
    │       ├─► Validate operation
    │       ├─► Check thresholds
    │       ├─► Chunk large files
    │       └─► Send event
    │
    ├─► Tool Execution
    │
    ├─► PostToolUse Hook
    │       ├─► Log results
    │       ├─► Check for issues
    │       └─► Send event
    │
    └─► Notification Hook (if needed)
            ├─► Create notifications
            └─► Send event
```
## Core Hook Implementations

### 1. Pre-Tool Use Hook (`pre_tool_use.py`)

**Purpose**: Monitors and modifies tool operations before execution

**Key Features**:
- File operation threshold checking
- Intelligent file chunking for large files
- Dangerous operation prevention
- GitLab issue creation for failures

**Threshold Configuration**:
```python
THRESHOLDS = {
    'max_lines_added': 50,
    'max_lines_removed': 50,
    'max_total_changes': 100
}
```

**Protected Operations**:
- Prevents accidental file deletions
- Blocks dangerous system commands
- Validates file paths

### 2. Enhanced File Chunking (`enhanced_file_chunking.py`)

**Purpose**: Intelligently chunks large files into logical commits

**Language Support**:
- **Python**: Classes, functions, imports
- **JavaScript/TypeScript**: Classes, functions, modules
- **Mojo**: Structs, functions
- **Markdown**: Sections, subsections, code blocks

**Chunking Algorithm**:
```python
def create_logical_chunks(segments, max_lines_per_chunk):
    """Create chunks based on logical boundaries"""
    chunks = []
    current_chunk = []
    current_lines = 0
    
    for segment in sorted(segments, key=lambda x: x['priority']):
        if current_lines + segment['lines'] > max_lines_per_chunk:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_lines = 0
        
        current_chunk.append(segment)
        current_lines += segment['lines']
    
    return chunks
```

### 3. GitLab Issue Manager (`gitlab_issue_manager.py`)

**Purpose**: Automated issue creation and management

**Capabilities**:
- Create issues for various failure types
- Update existing issues
- Add labels and milestones
- Close resolved issues

**Issue Templates**:
```python
ISSUE_TEMPLATES = {
    'test_failure': {
        'title': 'Test Failure: {test_name}',
        'labels': ['bug', 'automated', 'test-failure'],
        'template': '''
## Test Failure Report
**Test**: {test_name}
**Error**: {error_message}
**File**: {file_path}:{line_number}
        '''
    },
    'performance_regression': {
        'title': 'Performance Regression: {metric}',
        'labels': ['performance', 'automated', 'regression'],
        'template': '''
## Performance Regression Detected
**Metric**: {metric}
**Previous**: {previous_value}
**Current**: {current_value}
**Degradation**: {percentage}%
        '''
    }
}
```

### 4. Event Logging (`send_event.py`)

**Purpose**: Centralized event logging and analytics

**Event Types**:
- PreToolUse
- PostToolUse
- UserPromptSubmit
- Notification
- Stop
- SubagentStop
- PreCompact

**Event Structure**:
```json
{
    "timestamp": "2024-01-27T10:30:00Z",
    "event_type": "PreToolUse",
    "source_app": "mfc-project",
    "tool_name": "Write",
    "details": {
        "file_path": "/path/to/file.py",
        "lines_added": 150,
        "chunked": true,
        "chunks_created": 6
    }
}
```
## Automation Workflows

### Large File Creation Workflow

1. **Detection**: File creation >50 lines triggers chunking
2. **Analysis**: Parse file structure to identify logical segments
3. **Chunking**: Create logical chunks respecting boundaries
4. **Commits**: Generate individual commits with descriptive messages
5. **Blocking**: Block original Write operation

**Example Commit Sequence**:
```bash
Auto-commit: chunk 1/6 - module.py (15 lines) - 4 imports | module docstring
Auto-commit: chunk 2/6 - module.py (25 lines) - class DataProcessor
Auto-commit: chunk 3/6 - module.py (20 lines) - class FileHandler
Auto-commit: chunk 4/6 - module.py (30 lines) - functions: process_data, validate_input
Auto-commit: chunk 5/6 - module.py (25 lines) - functions: export_results, cleanup
Auto-commit: chunk 6/6 - module.py (10 lines) - main entry point
```

### Test Failure Automation

1. **Detection**: Test runner reports failure
2. **Analysis**: Parse error message and stack trace
3. **Issue Creation**: Create GitLab issue with details
4. **Notification**: Send event and notify user
5. **Tracking**: Monitor issue until resolved

### Performance Monitoring

1. **Baseline**: Establish performance baselines
2. **Monitoring**: Track metrics during execution
3. **Comparison**: Compare against baselines
4. **Alert**: Create issue if regression detected
5. **Resolution**: Track fix and update baseline
## Hook Configuration

### Settings Structure
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "pixi run /home/uge/mfc-project/.claude/hooks/pre_tool_use.py"
          }
        ]
      }
    ]
  }
}
```

### Environment Variables
```bash
# Hook behavior
export CLAUDE_HOOK_DEBUG=true
export CLAUDE_HOOK_TIMEOUT=30

# GitLab integration
export GITLAB_TOKEN=your-token
export GITLAB_PROJECT_ID=12345

# Event logging
export EVENT_LOG_LEVEL=DEBUG
export EVENT_LOG_FILE=/path/to/events.log
```
## Custom Hook Development

### Hook Template
```python
#!/usr/bin/env python3
"""Custom hook implementation."""
import json
import sys
import os

def main():
    """Main hook entry point."""
    # Read input from stdin
    input_data = json.loads(sys.stdin.read())
    
    # Extract hook context
    tool_name = input_data.get('tool_name')
    tool_input = input_data.get('tool_input', {})
    
    # Implement hook logic
    if should_modify(tool_name, tool_input):
        # Modify operation
        modified_input = modify_input(tool_input)
        
        # Return modified input
        print(json.dumps({
            'action': 'modify',
            'modified_input': modified_input
        }))
    elif should_block(tool_name, tool_input):
        # Block operation
        print(json.dumps({
            'action': 'block',
            'message': 'Operation blocked by hook'
        }))
        sys.exit(1)
    else:
        # Allow operation
        sys.exit(0)

if __name__ == '__main__':
    main()
```

### Hook Registration
```json
{
  "hooks": {
    "CustomHook": [
      {
        "matcher": "*.py",
        "hooks": [
          {
            "type": "command",
            "command": "python /path/to/custom_hook.py"
          }
        ]
      }
    ]
  }
}
```
## Best Practices

### 1. Hook Performance
- Keep hooks lightweight and fast
- Use async operations where possible
- Cache expensive computations
- Set appropriate timeouts

### 2. Error Handling
```python
try:
    # Hook logic
    result = process_input(input_data)
except Exception as e:
    # Log error but don't block operation
    log_error(f"Hook error: {e}")
    sys.exit(0)  # Allow operation to continue
```

### 3. Logging
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if os.getenv('CLAUDE_HOOK_DEBUG') else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
```

### 4. Testing Hooks
```bash
# Test hook manually
echo '{"tool_name": "Write", "tool_input": {"file_path": "test.py", "content": "..."}}' | python pre_tool_use.py

# Run hook tests
pytest .claude/hooks/tests/
```
