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