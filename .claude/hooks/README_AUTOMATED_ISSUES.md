# Automated GitLab Issue Creation

## Overview

The post-tool-use hook now automatically detects various types of issues and creates GitLab issues when problems are detected. This system extends the existing GitLab integration with enhanced pattern recognition and classification.

## Features

### 1. Test Failure Detection
- **Pattern**: Detects pytest, unittest, ruff, mypy failures
- **Action**: Creates bug issues with failure details
- **Labels**: `bug`, `automated`, `test-failure`, priority levels
- **Smart Deduplication**: Updates existing issues rather than creating duplicates

### 2. Build Failure Detection (Enhanced)
- **Pattern**: Detects compilation errors, linker errors, build tool failures
- **Triggers**: `make`, `cmake`, `gcc`, `pip install`, `pixi install`, etc.
- **Action**: Creates critical/high priority bug issues
- **Labels**: `bug`, `automated`, `build-failure`, severity-based priority

### 3. Performance Regression Detection (Enhanced)
- **Pattern**: Detects timeouts, memory issues, GPU errors, convergence failures
- **Triggers**: Commands containing `benchmark`, `profile`, `simulation`, `gpu`
- **Action**: Creates performance issues with regression analysis
- **Labels**: `performance`, `automated`, severity-based priority

### 4. Documentation Gap Detection (Enhanced)  
- **Pattern**: Detects missing docstrings, README files in new modules
- **Triggers**: File creation/modification operations
- **Action**: Creates low-priority enhancement issues
- **Labels**: `documentation`, `enhancement`, `automated`, `priority::low`

## Configuration

In `.claude/settings.json`:

```json
{
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
}
```

## Environment Variables

Required:
- `GITLAB_TOKEN`: GitLab API token with project access
- `GITLAB_PROJECT_ID`: GitLab project ID (optional, auto-detected from git remote)

Optional:
- `GITLAB_URL`: GitLab instance URL (defaults to https://gitlab.com)

## Issue Classification

### Severity Levels
- **Critical**: Fatal errors, security issues, compilation failures
- **High**: Build errors, performance regressions, test errors  
- **Medium**: Test failures, integration issues
- **Low**: Documentation gaps, minor issues

### Priority Assignment
- **Urgent**: Critical severity issues
- **High**: High severity or multiple failures
- **Medium**: Medium severity or moderate impact
- **Low**: Low severity or documentation issues

## File Structure

```
.claude/hooks/
├── post_tool_use.py              # Main hook with GitLab integration
├── enhanced_issue_detection.py   # Enhanced detection patterns
├── utils/
│   ├── gitlab_client.py          # GitLab API client
│   └── constants.py              # Configuration utilities
└── README_AUTOMATED_ISSUES.md    # This documentation
```

## Implementation Details

### Enhanced Issue Detection (`enhanced_issue_detection.py`)

1. **Build Failure Analysis**
   - Analyzes command output for compilation errors
   - Extracts error types and severity levels
   - Creates detailed issue descriptions with context

2. **Performance Regression Analysis**
   - Detects timeout, memory, and numerical instability issues
   - Analyzes GPU-related errors (CUDA, ROCm)
   - Provides performance-specific issue templates

3. **Documentation Gap Analysis**
   - Scans new Python files for missing docstrings
   - Checks directories for missing README files
   - Suggests documentation improvements

### Integration Points

The enhanced detection integrates with the existing post-tool-use hook through:

```python
# Import enhanced functions with fallback
try:
    from .enhanced_issue_detection import handle_build_failures as enhanced_build_failures
    enhanced_build_failures(input_data)
except ImportError:
    handle_build_failures(input_data)  # Fallback to basic implementation
```

## Testing

Test the automation system:

```bash
# Test build failure detection
echo '{"tool_name": "Bash", "tool_input": {"command": "make"}, "tool_result": {"output": "error: compilation failed"}}' | python enhanced_issue_detection.py

# Test performance detection  
echo '{"tool_name": "Bash", "tool_input": {"command": "benchmark"}, "tool_result": {"output": "timeout occurred"}}' | python enhanced_issue_detection.py

# Test documentation detection
echo '{"tool_name": "Write", "tool_input": {"file_path": "src/new.py", "content": "def func(): pass"}}' | python enhanced_issue_detection.py
```

## Issue Templates

### Build Failure Issue
- Title: `🔨 Build failure detected: [build_type] build failed with [count] errors`
- Labels: `bug`, `build-failure`, `automated`, `priority::[level]`
- Includes: Command, build type, failure analysis, next steps

### Performance Issue  
- Title: `⚡ Performance regression detected: [type] issues found`
- Labels: `performance`, `automated`, `priority::[level]`
- Includes: Performance metrics, regression analysis, optimization steps

### Documentation Issue
- Title: `📚 Documentation gaps detected: [count] missing documentation items`
- Labels: `documentation`, `enhancement`, `automated`, `priority::low`
- Includes: Missing items, documentation standards, improvement suggestions

## Benefits

1. **Automatic Bug Detection**: Catches issues immediately when they occur
2. **Consistent Issue Tracking**: Standardized issue format and labeling
3. **Priority Classification**: Automatic severity and urgency assignment
4. **Comprehensive Coverage**: Detects build, test, performance, and documentation issues
5. **Smart Deduplication**: Avoids creating duplicate issues for known problems
6. **Detailed Context**: Provides actionable information for debugging

## Troubleshooting

1. **Issues not being created**: Check GitLab token and project permissions
2. **Import errors**: Verify `enhanced_issue_detection.py` is in correct location
3. **Configuration issues**: Validate JSON syntax in `settings.json`
4. **API errors**: Check GitLab connectivity and project access

This automated system significantly enhances the development workflow by providing immediate feedback on issues and maintaining a comprehensive issue tracking system.