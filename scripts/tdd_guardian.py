#!/usr/bin/env python3
"""
TDD Guardian - Enforces TDD compliance
"""
import subprocess
import sys
from pathlib import Path

def check_test_coverage():
    """Ensure test coverage is above 95%"""
    result = subprocess.run(
        ["pytest", "--cov", "--cov-fail-under=95", "--quiet"],
        capture_output=True
    )
    return result.returncode == 0

def check_tests_pass():
    """Ensure all tests pass"""
    result = subprocess.run(["pytest"], capture_output=True)
    return result.returncode == 0

def check_tdd_structure():
    """Verify TDD project structure"""
    required_dirs = ["tests", "src"]
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            print(f"❌ Missing required directory: {dir_name}")
            return False
    return True

def main():
    """Run TDD compliance checks"""
    print("🔍 Running TDD Guardian checks...")
    
    checks = [
        ("TDD project structure", check_tdd_structure),
        ("All tests pass", check_tests_pass), 
        ("Test coverage ≥95%", check_test_coverage),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        if check_func():
            print(f"✅ {check_name}")
        else:
            print(f"❌ {check_name}")
            all_passed = False
    
    if all_passed:
        print("🎉 TDD compliance verified!")
        return 0
    else:
        print("🚨 TDD compliance violations detected!")
        return 1

if __name__ == "__main__":
    sys.exit(main())