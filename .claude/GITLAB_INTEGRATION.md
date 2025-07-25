# GitLab API Integration for Claude Code Hooks

## Overview

The GitLab integration enhances Claude Code hooks with automatic GitLab repository management capabilities. It provides seamless integration between your development workflow and GitLab features like issues, merge requests, and project management.

## Features

### 🚨 Automatic Issue Creation
- **Hook Failures**: Creates GitLab issues when hooks encounter errors
- **Detailed Reports**: Includes error messages, file paths, timestamps, and environment info
- **Automatic Labeling**: Tags issues with `hook-failure`, `automation`, and `bug` labels

### 🔀 Smart Merge Request Management
- **Multi-Commit Detection**: Automatically creates MRs when multiple commits accumulate on feature branches
- **Threshold Configuration**: Configurable commit count threshold (default: 5 commits in 24 hours)
- **Branch Protection**: Only creates MRs for non-main branches
- **Duplicate Prevention**: Checks for existing MRs before creating new ones

### 📊 Project Integration
- **Repository Information**: Access to project details, branches, and metadata
- **Commit Analysis**: Integration with commit counting and branch management
- **Status Updates**: Optional commit commenting and status reporting

## Installation

### 1. Install Dependencies

The python-gitlab library is automatically installed via pixi:

```bash
pixi install
```

### 2. Configuration

Choose one of two configuration methods:

#### Method A: Environment Variables (Recommended for Security)

```bash
export GITLAB_TOKEN="glpat-xxxxxxxxxxxxxxxxxxxx"
export GITLAB_URL="https://gitlab-runner.tail301d0a.ts.net"
export GITLAB_PROJECT_ID="123"  # Optional: can be auto-detected
```

#### Method B: Settings File Configuration

Edit `.claude/settings.json`:

```json
{
  "gitlab": {
    "enabled": true,
    "url": "https://gitlab-runner.tail301d0a.ts.net",
    "token": "glpat-xxxxxxxxxxxxxxxxxxxx",
    "project_id": "123",
    "default_branch": "main",
    "features": {
      "auto_issue_on_hook_failure": true,
      "auto_mr_on_multiple_commits": true,
      "commit_comments": false,
      "commit_threshold_for_mr": 5
    }
  }
}
```

### 3. GitLab Token Setup

1. Go to your GitLab instance
2. Navigate to **User Settings > Access Tokens**
3. Create a new token with these scopes:
   - `api` (full API access)
   - `read_user` (read user information)
   - `read_repository` (read repository data)
   - `write_repository` (write repository data)

## Configuration Options

### Core Settings

| Setting | Description | Default | Required |
|---------|-------------|---------|----------|
| `enabled` | Enable/disable GitLab integration | `false` | Yes |
| `url` | GitLab instance URL | - | Yes |
| `token` | GitLab access token | - | Yes |
| `project_id` | Project ID (auto-detected if not set) | - | No |
| `default_branch` | Default target branch for MRs | `"main"` | No |

### Feature Toggles

| Feature | Description | Default |
|---------|-------------|---------|
| `auto_issue_on_hook_failure` | Create issues for hook failures | `true` |
| `auto_mr_on_multiple_commits` | Create MRs for multiple commits | `true` |
| `commit_comments` | Add comments to commits | `false` |
| `commit_threshold_for_mr` | Commit count threshold for auto-MR | `5` |

## Usage Examples

### Testing the Integration

```bash
# Run the test script
cd .claude/hooks
python test_gitlab_integration.py
```

### Manual API Usage

```python
from utils.gitlab_client import (
    create_issue,
    create_merge_request,
    get_project_info,
    test_gitlab_connection
)

# Test connection
if test_gitlab_connection():
    print("GitLab connected successfully!")

# Create an issue
issue = create_issue(
    title="Bug Report",
    description="Description of the issue",
    labels=["bug", "urgent"]
)

# Create a merge request
mr = create_merge_request(
    source_branch="feature-branch",
    target_branch="main",
    title="Add new feature",
    description="This MR adds exciting new functionality"
)

# Get project information
project_info = get_project_info()
print(f"Project: {project_info['name']}")
```

## Automatic Workflows

### Hook Failure Workflow

1. **Hook Encounters Error**: Pre or post-tool hook fails
2. **Issue Creation**: GitLab issue created automatically with:
   - Descriptive title including hook name and branch
   - Detailed error message and stack trace
   - File path and timestamp information
   - Automatic labels for categorization
3. **Notification**: Issue URL logged to stderr for visibility

### Multi-Commit Workflow

1. **Commit Detection**: Post-tool hook counts recent commits (24-hour window)
2. **Threshold Check**: Compares count against configured threshold
3. **Branch Validation**: Ensures current branch is not main/default
4. **Duplicate Check**: Verifies no existing MR for the branch
5. **MR Creation**: Creates merge request with:
   - Descriptive title indicating commit count
   - Detailed description with timestamps
   - Automatic source/target branch configuration

## Integration Points

### Pre-Tool Hook (`pre_tool_use.py`)
- **Error Handling**: Creates GitLab issues for hook failures
- **Configuration Loading**: Loads GitLab settings and validates connectivity

### Post-Tool Hook (`post_tool_use.py`)
- **Success Tracking**: Logs successful operations
- **Commit Monitoring**: Tracks commit counts and triggers MR creation
- **Workflow Automation**: Orchestrates GitLab automation features

## Error Handling

### Graceful Degradation
- **Missing Dependencies**: Warns if python-gitlab not installed, continues without GitLab features
- **Authentication Failures**: Logs authentication errors, doesn't block hook execution
- **Network Issues**: Handles connection timeouts and API errors gracefully
- **Configuration Errors**: Validates configuration, provides helpful error messages

### Debug Information
```bash
# Enable verbose logging
export GITLAB_DEBUG=1

# Check configuration
python -c "from utils.gitlab_client import load_gitlab_config; print(load_gitlab_config())"
```

## Security Considerations

### Token Management
- **Environment Variables**: Store tokens in environment variables, not config files
- **Scope Limitation**: Use minimal required token scopes
- **Rotation**: Regularly rotate access tokens
- **Access Control**: Limit token access to necessary team members

### Repository Security
- **Branch Protection**: Configure branch protection rules in GitLab
- **MR Requirements**: Set up MR approval requirements
- **Hook Validation**: Ensure hooks don't expose sensitive information

## Troubleshooting

### Common Issues

#### "GitLab integration not properly configured"
**Solution**: Verify token, URL, and project configuration
```bash
# Check environment variables
echo $GITLAB_TOKEN
echo $GITLAB_URL

# Test connection
python test_gitlab_integration.py
```

#### "GitLab authentication failed"
**Solutions**:
- Verify token is valid and not expired
- Check token scopes include required permissions
- Ensure GitLab URL is accessible

#### "GitLab project not accessible"
**Solutions**:
- Verify project ID is correct
- Check token has access to the specific project
- Ensure project exists and is not archived

#### "python-gitlab library not available"
**Solution**: Install dependencies
```bash
pixi install
# or
pip install python-gitlab
```

### Debug Commands

```bash
# Test GitLab connection
python -c "from utils.gitlab_client import test_gitlab_connection; test_gitlab_connection()"

# Check current project
python -c "from utils.gitlab_client import get_project_info; print(get_project_info())"

# Verify current branch
python -c "from utils.gitlab_client import get_current_branch; print(get_current_branch())"
```

## API Reference

### Core Functions

#### `get_gitlab_client() -> gitlab.Gitlab`
Returns authenticated GitLab client instance.

#### `get_current_project() -> gitlab.Project`
Gets the current GitLab project based on configuration.

#### `create_issue(title, description, labels=None, assignee=None) -> dict`
Creates a GitLab issue with specified parameters.

#### `create_merge_request(source_branch, target_branch=None, title=None, description=None) -> dict`
Creates a merge request between branches.

#### `test_gitlab_connection() -> bool`
Tests GitLab API connection and configuration.

### Utility Functions

#### `get_current_branch() -> str`
Gets the current git branch name.

#### `load_gitlab_config() -> dict`
Loads GitLab configuration from settings and environment.

#### `create_hook_failure_issue(hook_name, error_message, file_path=None) -> dict`
Creates a standardized issue for hook failures.

## Best Practices

### Development Workflow
1. **Feature Branches**: Work on feature branches, let auto-MR handle integration
2. **Descriptive Commits**: Write clear commit messages for better MR descriptions
3. **Regular Integration**: Monitor auto-created MRs and issues for workflow health

### Configuration Management
1. **Environment-Specific**: Use different tokens/projects for development/production
2. **Feature Toggles**: Disable features not needed in specific environments
3. **Threshold Tuning**: Adjust commit thresholds based on team size and velocity

### Monitoring
1. **Issue Tracking**: Monitor auto-created issues for hook health
2. **MR Management**: Review auto-created MRs promptly
3. **Token Rotation**: Set up alerts for token expiration

## Examples

### Complete Setup Example

```bash
# 1. Set environment variables
export GITLAB_TOKEN="glpat-xxxxxxxxxxxxxxxxxxxx"
export GITLAB_URL="https://gitlab-runner.tail301d0a.ts.net"

# 2. Test connection
cd .claude/hooks
python test_gitlab_integration.py

# 3. Enable in settings
# Edit .claude/settings.json:
# "gitlab": { "enabled": true, ... }

# 4. Test hook integration
echo '{"tool_name": "Test", "error": "Test error"}' | pixi run post-tool-hook
```

### Custom Integration Example

```python
# Custom workflow integration
from utils.gitlab_client import create_issue, get_current_branch

def handle_custom_event():
    branch = get_current_branch()
    
    if branch.startswith('hotfix/'):
        issue = create_issue(
            title=f"Hotfix Alert: {branch}",
            description="Hotfix branch detected, requires immediate review",
            labels=["hotfix", "urgent", "review-required"]
        )
        print(f"Created hotfix tracking issue: {issue['web_url']}")
```

## Conclusion

The GitLab integration transforms Claude Code hooks from simple automation tools into comprehensive development workflow enhancers. By automatically managing issues and merge requests, it reduces manual overhead while maintaining visibility into code changes and hook operations.

For additional support or feature requests, create an issue in your GitLab repository using the integrated tools!