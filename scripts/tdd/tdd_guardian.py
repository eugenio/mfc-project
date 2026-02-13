"""TDD Guardian - Enforces TDD compliance."""

import subprocess
import sys
from pathlib import Path


def check_test_coverage():
    """Ensure test coverage is above 95%."""
    result = subprocess.run(
        ["pytest", "--cov", "--cov-fail-under=95", "--quiet"],
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def check_tests_pass():
    """Ensure all tests pass."""
    result = subprocess.run(["pytest"], check=False, capture_output=True)
    return result.returncode == 0


def check_tdd_structure() -> bool:
    """Verify TDD project structure."""
    required_dirs = ["tests", "src"]
    return all(Path(dir_name).exists() for dir_name in required_dirs)


def main() -> int:
    """Run TDD compliance checks."""
    checks = [
        ("TDD project structure", check_tdd_structure),
        ("All tests pass", check_tests_pass),
        ("Test coverage ≥95%", check_test_coverage),
    ]

    all_passed = True
    for _check_name, check_func in checks:
        if check_func():
            pass
        else:
            all_passed = False

    if all_passed:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
