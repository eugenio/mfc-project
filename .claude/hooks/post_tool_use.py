#!/usr/bin/env -S pixi run python
# /// script
# requires-python = ">=3.8"
# ///

import json
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from utils.constants import ensure_session_log_dir
from utils.gitlab_client import (
    load_gitlab_config, 
    create_hook_failure_issue, 
    create_large_commit_mr,
    get_current_branch,
    test_gitlab_connection
)

def count_recent_commits(branch: str = None, hours: int = 24) -> int:
    """
    Count commits made in the last N hours on a branch.
    
    Args:
        branch: Branch name (defaults to current branch)
        hours: Hours to look back
        
    Returns:
        int: Number of commits in the timeframe
    """
    if not branch:
        branch = get_current_branch()
    
    if not branch:
        return 0
    
    try:
        # Get commits from last N hours
        since_time = f"--since={hours} hours ago"
        result = subprocess.run([
            'git', 'rev-list', '--count', since_time, branch
        ], capture_output=True, text=True, check=True)
        
        return int(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0

def handle_gitlab_integrations(input_data: dict):
    """
    Handle GitLab integrations based on hook data and configuration.
    
    Args:
        input_data: Hook input data from Claude Code
    """
    config = load_gitlab_config()
    
    if not config.get("enabled"):
        return
    
    features = config.get("features", {})
    tool_name = input_data.get("tool_name", "")
    
    # Handle hook failures
    if features.get("auto_issue_on_hook_failure") and "error" in input_data:
        handle_hook_failure(input_data)
    
    # Handle multiple commits triggering MR creation
    if features.get("auto_mr_on_multiple_commits"):
        handle_multiple_commits(features.get("commit_threshold_for_mr", 5))
    
    # Handle successful large edits
    if tool_name in ["Edit", "MultiEdit", "Write"]:
        handle_successful_edit(input_data)
    
    # Handle test failures and create automatic bug issues
    if features.get("auto_issue_on_test_failure", True) and tool_name == "Bash":
        handle_test_failures(input_data)

def handle_test_failures(input_data: dict):
    """
    Analyze bash command output for test failures and create GitLab issues.
    
    Args:
        input_data: Hook data containing command and output information
    """
    try:
        tool_input = input_data.get("tool_input", {})
        tool_result = input_data.get("tool_result", {})
        
        command = tool_input.get("command", "")
        output = tool_result.get("output", "")
        
        # Skip if no output or not a test command
        if not output or not _is_test_command(command):
            return
            
        # Analyze output for test failures
        test_analysis = _analyze_test_output(command, output)
        
        if test_analysis["has_failures"]:
            _create_test_failure_issues(test_analysis, command)
            
    except Exception as e:
        print(f"Failed to handle test failures: {e}", file=sys.stderr)

def _is_test_command(command: str) -> bool:
    """Check if command is a test-related command."""
    test_indicators = [
        "pytest", "python -m pytest", "unittest", "test_", 
        "tests/", "/test", "ruff", "mypy", "python tests/",
        "python -m unittest", "coverage run"
    ]
    return any(indicator in command.lower() for indicator in test_indicators)

def _analyze_test_output(command: str, output: str) -> dict:
    """
    Analyze test output to identify failures and extract details.
    
    Returns:
        dict: Analysis results with failure information
    """
    analysis = {
        "has_failures": False,
        "failure_count": 0,
        "failures": [],
        "test_type": _detect_test_type(command, output)
    }
    
    # Analyze different types of test output
    if "pytest" in command.lower() or "FAILED" in output:
        _analyze_pytest_output(output, analysis)
    elif "unittest" in command.lower() or "AssertionError" in output:
        _analyze_unittest_output(output, analysis)
    elif "ruff" in command.lower():
        _analyze_ruff_output(output, analysis)
    elif "mypy" in command.lower():
        _analyze_mypy_output(output, analysis)
    else:
        _analyze_generic_test_output(output, analysis)
    
    return analysis

def _detect_test_type(command: str, output: str) -> str:
    """Detect the type of test being run."""
    if "pytest" in command.lower():
        return "pytest"
    elif "unittest" in command.lower():
        return "unittest"
    elif "ruff" in command.lower():
        return "linting"
    elif "mypy" in command.lower():
        return "type_checking"
    else:
        return "generic"

def _analyze_pytest_output(output: str, analysis: dict):
    """Analyze pytest output for failures."""
    lines = output.split('\n')
    
    for line in lines:
        if "FAILED" in line:
            analysis["has_failures"] = True
            analysis["failure_count"] += 1
            analysis["failures"].append({
                "type": "test_failure",
                "description": line.strip(),
                "severity": "medium"
            })
        elif "ERROR" in line:
            analysis["has_failures"] = True
            analysis["failure_count"] += 1
            analysis["failures"].append({
                "type": "test_error", 
                "description": line.strip(),
                "severity": "high"
            })

def _analyze_unittest_output(output: str, analysis: dict):
    """Analyze unittest output for failures."""
    if "FAILED" in output or "AssertionError" in output:
        analysis["has_failures"] = True
        
        # Count failures
        failure_count = output.count("FAIL:")
        error_count = output.count("ERROR:")
        analysis["failure_count"] = failure_count + error_count
        
        # Extract failure details
        lines = output.split('\n')
        for line in lines:
            if "FAIL:" in line or "ERROR:" in line:
                analysis["failures"].append({
                    "type": "unittest_failure",
                    "description": line.strip(),
                    "severity": "medium"
                })

def _analyze_ruff_output(output: str, analysis: dict):
    """Analyze ruff linting output for issues."""
    if "error:" in output.lower() or "warning:" in output.lower():
        analysis["has_failures"] = True
        
        # Count errors vs warnings
        error_count = output.lower().count("error:")
        warning_count = output.lower().count("warning:")
        analysis["failure_count"] = error_count + warning_count
        
        if error_count > 0:
            analysis["failures"].append({
                "type": "linting_error",
                "description": f"Ruff found {error_count} errors and {warning_count} warnings",
                "severity": "high" if error_count > 0 else "low"
            })

def _analyze_mypy_output(output: str, analysis: dict):
    """Analyze mypy type checking output for issues."""
    if "error:" in output.lower():
        analysis["has_failures"] = True
        
        error_count = output.lower().count("error:")
        analysis["failure_count"] = error_count
        
        analysis["failures"].append({
            "type": "type_checking_error",
            "description": f"MyPy found {error_count} type errors",
            "severity": "medium"
        })

def _analyze_generic_test_output(output: str, analysis: dict):
    """Analyze generic test output for common failure patterns."""
    failure_patterns = [
        "AssertionError", "not greater than", "not equal", 
        "not less than", "FAIL", "ERROR", "Exception:",
        "Traceback"
    ]
    
    for pattern in failure_patterns:
        if pattern in output:
            analysis["has_failures"] = True
            analysis["failure_count"] += 1
            analysis["failures"].append({
                "type": "generic_failure",
                "description": f"Detected failure pattern: {pattern}",
                "severity": "medium"
            })
            break

def _create_test_failure_issues(analysis: dict, command: str):
    """Create GitLab issues for test failures."""
    try:
        # Import GitLab functions
        from utils.gitlab_client import create_issue
        
        test_type = analysis["test_type"]
        failure_count = analysis["failure_count"]
        failures = analysis["failures"]
        
        # Create issue title
        if test_type == "pytest":
            title = f"🐛 Pytest failures detected: {failure_count} tests failing"
        elif test_type == "unittest":
            title = f"🐛 Unit test failures detected: {failure_count} tests failing"
        elif test_type == "linting":
            title = f"🔧 Code linting issues detected: {failure_count} issues found"
        elif test_type == "type_checking":
            title = f"🔍 Type checking errors detected: {failure_count} errors found"
        else:
            title = f"🐛 Test failures detected: {failure_count} issues found"
        
        # Build detailed description
        description = f"""Automatic issue creation from failed test execution.

**Command executed:** `{command}`

**Test type:** {test_type}
**Failure count:** {failure_count}

**Failures detected:**
"""
        
        for i, failure in enumerate(failures[:5], 1):  # Limit to first 5 failures
            description += f"\n{i}. **{failure['type']}** (severity: {failure['severity']})\n   {failure['description']}\n"
        
        if len(failures) > 5:
            description += f"\n... and {len(failures) - 5} more failures"
        
        description += f"""

**Environment:**
- Test execution time: {datetime.now().isoformat()}
- Hook: post_tool_use (automatic detection)
- Source: Claude Code automatic test failure detection

**Next steps:**
1. Review failing tests and error messages
2. Fix underlying issues causing test failures
3. Re-run tests to verify fixes
4. Close this issue when all tests pass

**Automated Issue Creation**
This issue was automatically created by the Claude Code post-tool-use hook when test failures were detected."""
        
        # Determine appropriate labels
        labels = ["bug", "automated"]
        if test_type in ["linting", "type_checking"]:
            labels.append("code-quality")
        else:
            labels.append("test-failure")
            
        # Add priority label based on severity
        high_severity_count = sum(1 for f in failures if f.get("severity") == "high")
        if high_severity_count > 0:
            labels.append("priority::high")
        elif failure_count > 5:
            labels.append("priority::medium")
        else:
            labels.append("priority::low")
        
        # Create the issue
        issue = create_issue(
            title=title,
            description=description,
            labels=labels
        )
        
        if issue:
            print(f"✅ Created GitLab issue for test failures: {issue['web_url']}", file=sys.stderr)
            
    except Exception as e:
        print(f"❌ Failed to create test failure issue: {e}", file=sys.stderr)

def handle_hook_failure(input_data: dict):
    """
    Create GitLab issue for hook failures.
    
    Args:
        input_data: Hook data containing error information
    """
    try:
        error_info = input_data.get("error", "Unknown error")
        tool_name = input_data.get("tool_name", "Unknown tool")
        file_path = input_data.get("tool_input", {}).get("file_path")
        
        issue = create_hook_failure_issue(
            hook_name=f"post_tool_use ({tool_name})",
            error_message=str(error_info),
            file_path=file_path
        )
        
        if issue:
            print(f"Created GitLab issue for hook failure: {issue['web_url']}", file=sys.stderr)
    
    except Exception as e:
        print(f"Failed to create hook failure issue: {e}", file=sys.stderr)

def handle_multiple_commits(threshold: int):
    """
    Create MR if multiple commits accumulate on a branch.
    
    Args:
        threshold: Number of commits that trigger MR creation
    """
    try:
        branch = get_current_branch()
        
        if not branch or branch == "main":
            return
        
        # Count recent commits (last 24 hours)
        recent_commits = count_recent_commits(branch, 24)
        
        if recent_commits >= threshold:
            # Check if MR already exists for this branch
            if not mr_exists_for_branch(branch):
                mr = create_large_commit_mr(recent_commits, branch)
                
                if mr:
                    print(f"Created auto-MR for {recent_commits} commits: {mr['web_url']}", file=sys.stderr)
    
    except Exception as e:
        print(f"Failed to handle multiple commits: {e}", file=sys.stderr)

def mr_exists_for_branch(branch: str) -> bool:
    """
    Check if a merge request already exists for the given branch.
    
    Args:
        branch: Source branch name
        
    Returns:
        bool: True if MR exists, False otherwise
    """
    try:
        from utils.gitlab_client import get_current_project
        
        project = get_current_project()
        if not project:
            return False
        
        # Get open merge requests for this branch
        mrs = project.mergerequests.list(
            source_branch=branch,
            state='opened'
        )
        
        return len(mrs) > 0
    
    except Exception:
        return False

def handle_successful_edit(input_data: dict):
    """
    Handle successful edit operations for GitLab integration.
    
    Args:
        input_data: Tool input data
    """
    try:
        tool_name = input_data.get("tool_name")
        
        # Log successful operations (could be extended for GitLab commit comments)
        if tool_name in ["Edit", "MultiEdit"]:
            file_path = input_data.get("tool_input", {}).get("file_path")
            if file_path:
                print(f"GitLab: Successful {tool_name} operation on {file_path}", file=sys.stderr)
        
        elif tool_name == "Write":
            file_path = input_data.get("tool_input", {}).get("file_path")
            if file_path:
                print(f"GitLab: Successful file creation: {file_path}", file=sys.stderr)
    
    except Exception as e:
        print(f"Failed to handle successful edit: {e}", file=sys.stderr)

def main():
    try:
        # Read JSON input from stdin
        input_data = json.load(sys.stdin)

        # Extract session_id
        session_id = input_data.get("session_id", "unknown")

        # Ensure session log directory exists
        log_dir = ensure_session_log_dir(session_id)
        log_path = log_dir / "post_tool_use.json"

        # Read existing log data or initialize empty list
        if log_path.exists():
            with open(log_path, "r") as f:
                try:
                    log_data = json.load(f)
                except (json.JSONDecodeError, ValueError):
                    log_data = []
        else:
            log_data = []

        # Process GitLab integrations if enabled
        try:
            handle_gitlab_integrations(input_data)
        except Exception as e:
            print(f"GitLab integration error: {e}", file=sys.stderr)

        # Append new data
        log_data.append(input_data)

        # Write back to file with formatting
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        sys.exit(0)

    except json.JSONDecodeError:
        # Handle JSON decode errors gracefully
        sys.exit(0)
    except Exception:
        # Exit cleanly on any other error
        sys.exit(0)


if __name__ == "__main__":
    main()
