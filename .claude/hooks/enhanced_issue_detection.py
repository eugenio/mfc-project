#!/usr/bin/env python3
"""
Enhanced issue detection functions for post_tool_use hook.

This module provides additional automatic issue detection capabilities:
- Build failure detection
- Performance regression detection  
- Documentation gap detection

These functions extend the existing post_tool_use hook for GitLab issue #9.
"""

# Add the utils directory to the path for GitLab client imports
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))


def handle_build_failures(input_data: dict[str, Any]):
    """
    Analyze bash command output for build failures and create GitLab issues.
    
    Args:
        input_data: Hook data containing command and output information
    """
    try:
        tool_input = input_data.get("tool_input", {})
        tool_result = input_data.get("tool_result", {})

        command = tool_input.get("command", "")
        output = tool_result.get("output", "")

        # Skip if no output or not a build command
        if not output or not _is_build_command(command):
            return

        # Analyze output for build failures
        build_analysis = _analyze_build_output(command, output)

        if build_analysis["has_failures"]:
            _create_build_failure_issues(build_analysis, command)

    except Exception as e:
        print(f"Failed to handle build failures: {e}", file=sys.stderr)


def handle_performance_analysis(input_data: dict[str, Any]):
    """
    Analyze command output for performance regressions and create GitLab issues.
    
    Args:
        input_data: Hook data containing command and output information
    """
    try:
        tool_input = input_data.get("tool_input", {})
        tool_result = input_data.get("tool_result", {})

        command = tool_input.get("command", "")
        output = tool_result.get("output", "")

        # Skip if no output or not a performance-related command
        if not output or not _is_performance_command(command):
            return

        # Analyze output for performance issues
        perf_analysis = _analyze_performance_output(command, output)

        if perf_analysis["has_regressions"]:
            _create_performance_issues(perf_analysis, command)

    except Exception as e:
        print(f"Failed to handle performance analysis: {e}", file=sys.stderr)


def handle_documentation_gaps(input_data: dict[str, Any]):
    """
    Analyze file operations for documentation gaps and create GitLab issues.
    
    Args:
        input_data: Hook data containing tool information
    """
    try:
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})

        # Only analyze file creation/modification operations
        if tool_name not in ["Write", "Edit", "MultiEdit"]:
            return

        # Analyze for documentation gaps
        doc_analysis = _analyze_documentation_gaps(tool_name, tool_input)

        if doc_analysis["has_gaps"]:
            _create_documentation_issues(doc_analysis, tool_name, tool_input)

    except Exception as e:
        print(f"Failed to handle documentation gaps: {e}", file=sys.stderr)


def _is_build_command(command: str) -> bool:
    """Check if command is a build-related command."""
    build_indicators = [
        "make", "cmake", "gcc", "g++", "clang", "cargo build",
        "npm run build", "yarn build", "mvn compile", "gradle build",
        "pip install", "python setup.py", "poetry build", "pyproject",
        "pixi install", "mojo build", "ninja", "bazel build"
    ]
    return any(indicator in command.lower() for indicator in build_indicators)


def _is_performance_command(command: str) -> bool:
    """Check if command is performance-related."""
    perf_indicators = [
        "benchmark", "profile", "perf", "time ", "timeout",
        "memory", "cpu", "performance", "speed", "optimization",
        "stress", "load", "simulation", "gpu", "cuda", "rocm"
    ]
    return any(indicator in command.lower() for indicator in perf_indicators)


def _analyze_build_output(command: str, output: str) -> dict[str, Any]:
    """
    Analyze build output to identify failures and extract details.
    
    Returns:
        dict: Analysis results with failure information
    """
    analysis = {
        "has_failures": False,
        "failure_count": 0,
        "failures": [],
        "build_type": _detect_build_type(command, output)
    }

    # Common build failure patterns
    failure_patterns = [
        ("error:", "compilation_error", "high"),
        ("fatal error:", "fatal_error", "critical"),
        ("undefined reference", "linker_error", "high"),
        ("permission denied", "permission_error", "medium"),
        ("no such file", "file_not_found", "medium"),
        ("failed to", "generic_failure", "medium"),
        ("build failed", "build_failure", "high"),
        ("compilation terminated", "compilation_failure", "high"),
        ("make: ***", "make_error", "high")
    ]

    for pattern, error_type, severity in failure_patterns:
        if pattern in output.lower():
            analysis["has_failures"] = True
            analysis["failure_count"] += output.lower().count(pattern)
            analysis["failures"].append({
                "type": error_type,
                "description": f"Build failure detected: {pattern}",
                "severity": severity,
                "pattern": pattern
            })

    return analysis


def _analyze_performance_output(command: str, output: str) -> dict[str, Any]:
    """
    Analyze performance output to identify regressions.
    
    Returns:
        dict: Analysis results with performance information
    """
    analysis = {
        "has_regressions": False,
        "regression_count": 0,
        "regressions": [],
        "performance_type": _detect_performance_type(command, output)
    }

    # Performance regression patterns
    regression_patterns = [
        ("timeout", "timeout_regression", "high"),
        ("out of memory", "memory_regression", "critical"),
        ("memory leak", "memory_leak", "high"),
        ("slow", "performance_degradation", "medium"),
        ("failed to converge", "convergence_failure", "medium"),
        ("nan", "numerical_instability", "high"),
        ("inf", "numerical_overflow", "high"),
        ("cuda out of memory", "gpu_memory_error", "high"),
        ("rocm error", "rocm_error", "high")
    ]

    for pattern, regression_type, severity in regression_patterns:
        if pattern in output.lower():
            analysis["has_regressions"] = True
            analysis["regression_count"] += 1
            analysis["regressions"].append({
                "type": regression_type,
                "description": f"Performance issue detected: {pattern}",
                "severity": severity,
                "pattern": pattern
            })

    return analysis


def _analyze_documentation_gaps(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """
    Analyze file operations for documentation gaps.
    
    Returns:
        dict: Analysis results with documentation gap information
    """
    analysis = {
        "has_gaps": False,
        "gap_count": 0,
        "gaps": []
    }

    file_path = tool_input.get("file_path", "")

    if not file_path:
        return analysis

    # Check for code files without documentation
    code_extensions = ['.py', '.js', '.ts', '.cpp', '.c', '.h', '.java', '.go', '.rs', '.mojo']

    if any(file_path.endswith(ext) for ext in code_extensions):
        # Check if it's a new file creation
        if tool_name == "Write":
            content = tool_input.get("content", "")

            # Check for missing docstrings in Python files
            if file_path.endswith('.py') and _check_python_documentation_gaps(content):
                analysis["has_gaps"] = True
                analysis["gap_count"] += 1
                analysis["gaps"].append({
                    "type": "missing_docstring",
                    "description": f"New Python file {file_path} missing docstrings",
                    "severity": "low",
                    "file_path": file_path
                })

            # Check for missing README in new directories
            if 'src/' in file_path or 'lib/' in file_path:
                dir_path = str(Path(file_path).parent)
                if not _has_readme_in_directory(dir_path):
                    analysis["has_gaps"] = True
                    analysis["gap_count"] += 1
                    analysis["gaps"].append({
                        "type": "missing_readme",
                        "description": f"Directory {dir_path} missing README documentation",
                        "severity": "low",
                        "directory": dir_path
                    })

    return analysis


def _check_python_documentation_gaps(content: str) -> bool:
    """Check if Python code is missing docstrings."""
    lines = content.split('\n')

    # Look for function or class definitions without docstrings
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (stripped.startswith('def ') or stripped.startswith('class ')) and not stripped.startswith('def _'):
            # Check if next non-empty line is a docstring
            for j in range(i + 1, min(i + 5, len(lines))):
                next_line = lines[j].strip()
                if next_line:
                    if not (next_line.startswith('"""') or next_line.startswith("'''")):
                        return True
                    break

    return False


def _has_readme_in_directory(dir_path: str) -> bool:
    """Check if directory has README file."""
    readme_files = ['README.md', 'README.rst', 'README.txt', 'README']
    try:
        dir_obj = Path(dir_path)
        if dir_obj.exists():
            for readme in readme_files:
                if (dir_obj / readme).exists():
                    return True
    except Exception:
        pass
    return False


def _detect_build_type(command: str, output: str) -> str:
    """Detect the type of build being performed."""
    if "make" in command.lower():
        return "make"
    elif "cmake" in command.lower():
        return "cmake"
    elif "cargo" in command.lower():
        return "rust"
    elif "npm" in command.lower() or "yarn" in command.lower():
        return "javascript"
    elif "pip" in command.lower() or "python setup.py" in command.lower():
        return "python"
    elif "mojo" in command.lower():
        return "mojo"
    else:
        return "generic"


def _detect_performance_type(command: str, output: str) -> str:
    """Detect the type of performance test being run."""
    if "benchmark" in command.lower():
        return "benchmark"
    elif "profile" in command.lower():
        return "profiling"
    elif "gpu" in command.lower() or "cuda" in command.lower() or "rocm" in command.lower():
        return "gpu_performance"
    elif "simulation" in command.lower():
        return "simulation"
    else:
        return "generic"


def _create_build_failure_issues(analysis: dict, command: str):
    """Create GitLab issues for build failures."""
    try:
        from utils.gitlab_client import create_issue

        build_type = analysis["build_type"]
        failure_count = analysis["failure_count"]
        failures = analysis["failures"]

        # Create issue title
        title = f"🔨 Build failure detected: {build_type} build failed with {failure_count} errors"

        # Build detailed description
        description = f"""Automatic issue creation from failed build execution.

**Command executed:** `{command}`

**Build type:** {build_type}
**Failure count:** {failure_count}

**Build failures detected:**
"""

        for i, failure in enumerate(failures[:5], 1):
            description += f"\n{i}. **{failure['type']}** (severity: {failure['severity']})\n   {failure['description']}\n"

        if len(failures) > 5:
            description += f"\n... and {len(failures) - 5} more failures"

        description += f"""

**Environment:**
- Build execution time: {datetime.now().isoformat()}
- Hook: post_tool_use (automatic detection)
- Source: Claude Code automatic build failure detection

**Next steps:**
1. Review build errors and error messages
2. Fix compilation/build issues
3. Re-run build to verify fixes
4. Close this issue when build succeeds

**Automated Issue Creation**
This issue was automatically created by the Claude Code post-tool-use hook when build failures were detected."""

        # Determine appropriate labels
        labels = ["bug", "build-failure", "automated"]

        # Add priority label based on severity
        critical_count = sum(1 for f in failures if f.get("severity") == "critical")
        high_severity_count = sum(1 for f in failures if f.get("severity") == "high")

        if critical_count > 0:
            labels.append("priority::critical")
        elif high_severity_count > 0:
            labels.append("priority::high")
        else:
            labels.append("priority::medium")

        # Create issue
        issue = create_issue(
            title=title,
            description=description,
            labels=labels
        )

        if issue:
            print(f"✅ Created GitLab issue for build failures: {issue['web_url']}", file=sys.stderr)

    except Exception as e:
        print(f"❌ Failed to create build failure issue: {e}", file=sys.stderr)


def _create_performance_issues(analysis: dict, command: str):
    """Create GitLab issues for performance regressions."""
    try:
        from utils.gitlab_client import create_issue

        perf_type = analysis["performance_type"]
        regression_count = analysis["regression_count"]
        regressions = analysis["regressions"]

        # Create issue title
        title = f"⚡ Performance regression detected: {perf_type} issues found"

        # Build detailed description
        description = f"""Automatic issue creation from performance regression detection.

**Command executed:** `{command}`

**Performance type:** {perf_type}
**Regression count:** {regression_count}

**Performance issues detected:**
"""

        for i, regression in enumerate(regressions[:5], 1):
            description += f"\n{i}. **{regression['type']}** (severity: {regression['severity']})\n   {regression['description']}\n"

        if len(regressions) > 5:
            description += f"\n... and {len(regressions) - 5} more issues"

        description += f"""

**Environment:**
- Performance test time: {datetime.now().isoformat()}
- Hook: post_tool_use (automatic detection)
- Source: Claude Code automatic performance regression detection

**Next steps:**
1. Review performance metrics and error messages
2. Analyze performance bottlenecks
3. Optimize code for better performance
4. Re-run performance tests to verify improvements
5. Close this issue when performance is acceptable

**Automated Issue Creation**
This issue was automatically created by the Claude Code post-tool-use hook when performance regressions were detected."""

        # Determine appropriate labels
        labels = ["performance", "automated"]

        # Add priority and type labels based on severity
        critical_count = sum(1 for r in regressions if r.get("severity") == "critical")
        high_severity_count = sum(1 for r in regressions if r.get("severity") == "high")

        if critical_count > 0:
            labels.extend(["priority::critical", "bug"])
        elif high_severity_count > 0:
            labels.extend(["priority::high", "bug"])
        else:
            labels.extend(["priority::medium", "enhancement"])

        # Create issue
        issue = create_issue(
            title=title,
            description=description,
            labels=labels
        )

        if issue:
            print(f"✅ Created GitLab issue for performance regression: {issue['web_url']}", file=sys.stderr)

    except Exception as e:
        print(f"❌ Failed to create performance issue: {e}", file=sys.stderr)


def _create_documentation_issues(analysis: dict, tool_name: str, tool_input: dict):
    """Create GitLab issues for documentation gaps."""
    try:
        from utils.gitlab_client import create_issue

        gap_count = analysis["gap_count"]
        gaps = analysis["gaps"]

        # Create issue title
        title = f"📚 Documentation gaps detected: {gap_count} missing documentation items"

        # Build detailed description
        description = f"""Automatic issue creation from documentation gap detection.

**Tool operation:** {tool_name}
**Gap count:** {gap_count}

**Documentation gaps detected:**
"""

        for i, gap in enumerate(gaps[:5], 1):
            description += f"\n{i}. **{gap['type']}** (severity: {gap['severity']})\n   {gap['description']}\n"

        if len(gaps) > 5:
            description += f"\n... and {len(gaps) - 5} more gaps"

        description += f"""

**Environment:**
- Detection time: {datetime.now().isoformat()}
- Hook: post_tool_use (automatic detection)
- Source: Claude Code automatic documentation gap detection

**Next steps:**
1. Review missing documentation items
2. Add appropriate docstrings, README files, or comments
3. Follow project documentation standards
4. Close this issue when documentation is complete

**Automated Issue Creation**
This issue was automatically created by the Claude Code post-tool-use hook when documentation gaps were detected."""

        # Determine appropriate labels
        labels = ["documentation", "enhancement", "automated", "priority::low"]

        # Create issue
        issue = create_issue(
            title=title,
            description=description,
            labels=labels
        )

        if issue:
            print(f"✅ Created GitLab issue for documentation gaps: {issue['web_url']}", file=sys.stderr)

    except Exception as e:
        print(f"❌ Failed to create documentation issue: {e}", file=sys.stderr)
