# Automatic File Chunking System

## Overview

The enhanced file chunking system automatically breaks large file creation into smaller, logical commits. When Claude Code creates a file with many lines of code, the system analyzes the code structure and creates incremental commits that follow logical boundaries (imports, classes, functions).

## Features

### 🧠 **Intelligent Code Analysis**
- **Structure Recognition**: Analyzes code to identify imports, classes, functions, constants
- **Language Support**: Python, JavaScript/TypeScript, Mojo, Java, C/C++
- **Logical Segmentation**: Breaks code along natural boundaries rather than arbitrary line counts

### 📦 **Smart Chunking Strategy**
- **Priority-Based Ordering**: Imports first, then classes, then functions
- **Size-Aware Chunking**: Respects maximum lines per chunk while maintaining logical coherence
- **Context Preservation**: Ensures each chunk is meaningful and self-contained

### 🚀 **Automatic Integration**
- **Pre-Tool-Use Hook**: Intercepts large Write operations before execution
- **Transparent Process**: User sees normal file creation, system handles chunking automatically
- **Fallback Support**: Falls back to standard creation if chunking fails

## How It Works

### 1. **Detection Phase**
```python
# Triggers when Write tool creates files meeting criteria:
- File size > 50 lines (configurable)
- File type in supported list (.py, .js, .ts, etc.)
- Code has >= 3 logical segments (imports, classes, functions)
```

### 2. **Analysis Phase**
```python
# Code structure analysis identifies:
- Module docstrings and documentation
- Import statements and dependencies  
- Constants and module-level variables
- Class definitions with methods
- Function definitions and implementations
```

### 3. **Chunking Phase**
```python
# Creates logical chunks based on:
Priority 1: Module documentation
Priority 2: Imports and constants
Priority 3: Classes (with complete methods)
Priority 4: Functions and utilities
```

### 4. **Commit Phase**
```python
# Incremental file building:
1. Create file with chunk 1 → commit
2. Append chunk 2 → commit
3. Continue until complete
4. Block original Write operation
```

## Configuration

### Settings in `.claude/settings.json`:

```json
{
  "chunked_file_creation": {
    "enabled": true,
    "max_new_file_lines": 50,
    "max_lines_per_chunk": 25,
    "min_segments_for_chunking": 3,
    "commit_message_prefix": "Auto-commit: ",
    "supported_extensions": [".py", ".js", ".ts", ".jsx", ".tsx", ".mojo", ".🔥", ".java", ".cpp", ".c", ".h"]
  }
}
```

### Configuration Options:

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | `true` | Enable/disable chunked file creation |
| `max_new_file_lines` | `50` | Minimum lines to trigger chunking |
| `max_lines_per_chunk` | `25` | Maximum lines per individual chunk |
| `min_segments_for_chunking` | `3` | Minimum code segments needed |
| `commit_message_prefix` | `"Auto-commit: "` | Prefix for commit messages |
| `supported_extensions` | `[".py", ...]` | File types that support chunking |

## Example Workflow

### Input: Large Python File (82 lines)
```python
#!/usr/bin/env python3
"""Large module with multiple components."""

import os
import sys
from pathlib import Path

MAX_ITEMS = 1000
DEFAULT_TIMEOUT = 30

class DataProcessor:
    """Process data with algorithms."""
    
    def __init__(self, config):
        self.config = config
    
    def process_data(self):
        return "processed"

class DatabaseManager:
    """Manage database connections."""
    
    def connect(self):
        return "connected"

def utility_function():
    """Utility function."""
    return "utility"

def main():
    """Main entry point."""
    processor = DataProcessor({})
    # ... implementation
```

### Output: 4 Logical Chunks

**Chunk 1/4: Foundation (13 lines)**
```
Auto-commit: chunk 1/4 - large_module.py (13 lines) - 3 imports | 2 constants | module documentation
```
- Module docstring
- Import statements  
- Constants

**Chunk 2/4: Data Processing (20 lines)**
```
Auto-commit: chunk 2/4 - large_module.py (20 lines) - class DataProcessor
```
- Complete DataProcessor class with all methods

**Chunk 3/4: Database Management (20 lines)**
```
Auto-commit: chunk 3/4 - large_module.py (20 lines) - class DatabaseManager  
```
- Complete DatabaseManager class with all methods

**Chunk 4/4: Utilities & Main (29 lines)**
```
Auto-commit: chunk 4/4 - large_module.py (29 lines) - functions: utility_function, main
```
- Utility functions
- Main entry point
- Module-level execution code

## Language-Specific Analysis

### Python (`.py`)
- **Imports**: `import`, `from ... import`
- **Classes**: `class ClassName:`
- **Functions**: `def function_name():`
- **Constants**: `UPPER_CASE = value`
- **Docstrings**: `"""triple quotes"""`

### JavaScript/TypeScript (`.js`, `.ts`, `.jsx`, `.tsx`)
- **Imports**: `import`, `export`, `require()`
- **Classes**: `class ClassName`
- **Functions**: `function name()`, `const name = () =>`
- **Constants**: `const UPPER_CASE`

### Mojo (`.mojo`, `.🔥`)
- **Imports**: `from ... import`
- **Structs**: `struct StructName:`
- **Functions**: `fn function_name():`

### C/C++ (`.c`, `.cpp`, `.h`)
- **Includes**: `#include`
- **Functions**: Function definitions
- **Classes**: `class ClassName` (C++)
- **Structs**: `struct name`

## Commit Message Format

### Template
```
Auto-commit: chunk {N}/{total} - {filename} ({lines} lines) - {description}
```

### Examples
```bash
Auto-commit: chunk 1/3 - utils.py (15 lines) - 4 imports | 2 constants | module documentation
Auto-commit: chunk 2/3 - utils.py (22 lines) - class ConfigManager  
Auto-commit: chunk 3/3 - utils.py (18 lines) - functions: parse_config, validate_settings
```

## Benefits

### 🎯 **Better Code Review**
- **Logical Progression**: Each commit represents a complete, reviewable unit
- **Easier Debugging**: Issues can be traced to specific functional chunks
- **Clear History**: Git history shows incremental development rather than large dumps

### 📈 **Improved Development Workflow**
- **Reduced Merge Conflicts**: Smaller, focused commits reduce conflict likelihood
- **Atomic Changes**: Each commit represents a complete logical unit
- **Rollback Precision**: Can revert specific functionality without affecting others

### 🔍 **Enhanced Traceability**
- **Feature Tracking**: Easy to see when specific classes/functions were added
- **Change Attribution**: Clear ownership and context for each code segment
- **Documentation**: Commit messages provide automatic documentation of development progress

## Integration with Existing Systems

### Pre-Tool-Use Hook Integration
```python
# In pre_tool_use.py
from enhanced_file_chunking import check_chunked_file_creation

if check_chunked_file_creation(tool_name, tool_input):
    # File was created in chunks, block original Write operation
    sys.exit(2)
```

### Fallback Mechanism
- If chunking fails → Falls back to standard file creation
- If file type unsupported → Uses standard approach
- If chunking disabled → Uses existing threshold system

### Compatibility
- **Works with existing**: File creation thresholds, auto-commit system
- **Respects settings**: GitLab integration, commit message formats
- **Preserves functionality**: All existing pre-tool-use protections remain active

## Troubleshooting

### Common Issues

1. **Chunking not triggered**
   - Check file size meets `max_new_file_lines` threshold
   - Verify file extension in `supported_extensions`
   - Ensure code has enough segments (`min_segments_for_chunking`)

2. **Import errors**
   - Verify `enhanced_file_chunking.py` is in hooks directory
   - Check Python path includes hooks directory
   - Ensure all dependencies are available

3. **Commit failures**
   - Check git repository is initialized
   - Verify git user configuration
   - Ensure working directory is clean

### Debug Information

Enable debug output by checking stderr during file creation:
```bash
🔨 Starting chunked file creation for filename.py
📊 Analysis: 82 lines, 8 segments  
📦 Created 4 logical chunks
🔧 Processing chunk 1/4: description
✅ Committed chunk 1/4: description
```

### Configuration Testing

Test chunking decision without creating files:
```python
from enhanced_file_chunking import should_use_chunked_creation, load_chunking_config

config = load_chunking_config()
should_chunk = should_use_chunked_creation('test.py', content, config)
print(f"Would use chunking: {should_chunk}")
```

## Performance Considerations

### Analysis Speed
- **Fast parsing**: Uses line-by-line analysis without AST parsing
- **Efficient chunking**: O(n) complexity for most operations
- **Minimal overhead**: Only processes files that meet size thresholds

### Git Performance
- **Small commits**: Each chunk is typically 15-25 lines
- **Sequential commits**: Commits created in logical order
- **No history pollution**: Failed chunking doesn't leave partial commits

### Memory Usage
- **Streaming approach**: Processes content line by line
- **No full file loading**: Works with content as provided by Claude Code
- **Minimal buffering**: Only stores current chunk content in memory

## Advanced Usage

### Custom Commit Messages
Modify `commit_message_prefix` in settings to customize:
```json
{
  "chunked_file_creation": {
    "commit_message_prefix": "feat: add component - "
  }
}
```

### Selective Language Support
Restrict to specific file types:
```json
{
  "chunked_file_creation": {
    "supported_extensions": [".py"]  // Python only
  }
}
```

### Aggressive Chunking
For very small chunks:
```json
{
  "chunked_file_creation": {
    "max_new_file_lines": 25,      // Lower threshold
    "max_lines_per_chunk": 10,     // Smaller chunks
    "min_segments_for_chunking": 2 // Less strict
  }
}
```

This automatic chunking system significantly improves code organization and git history management while requiring zero manual intervention from the developer.