#!/usr/bin/env python3
"""
Integrated Test Runner for MFC Project
Combines hooks, enhanced git-guardian, and main project tests

Created: 2025-08-01
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / ".claude" / "hooks"))
sys.path.insert(0, str(project_root / "q-learning-mfcs" / "src"))


class IntegratedTestRunner:
    """Comprehensive test runner for all project components."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.hooks_dir = self.project_root / ".claude" / "hooks"
        self.mfc_tests_dir = self.project_root / "q-learning-mfcs" / "tests"
        self.results = {}
        
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> tuple[int, str, str]:
        """Run a command and return exit code, stdout, stderr."""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", "Command timed out after 10 minutes"
        except FileNotFoundError:
            return 1, "", f"Command not found: {' '.join(cmd)}"
    
    def run_hooks_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run Claude Code hooks tests."""
        print("🔧 Running Claude Code hooks tests...")
        
        cmd = ["python", "-m", "pytest", "tests/", "-v" if verbose else "-q"]
        if not verbose:
            cmd.extend(["--tb=no"])
            
        exit_code, stdout, stderr = self.run_command(cmd, self.hooks_dir)
        
        result = {
            "name": "hooks",
            "exit_code": exit_code,
            "passed": exit_code == 0,
            "stdout": stdout,
            "stderr": stderr,
            "summary": self._extract_pytest_summary(stdout)
        }
        
        if result["passed"]:
            print("  ✅ Hooks tests passed")
        else:
            print("  ❌ Hooks tests failed")
            
        return result
    
    def run_security_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run enhanced security guardian tests."""
        print("🛡️  Running enhanced security guardian tests...")
        
        cmd = [
            "python", "-m", "pytest", 
            "tests/test_enhanced_security_guardian.py",
            "tests/test_git_guardian_integration.py",
            "-v" if verbose else "-q"
        ]
        if not verbose:
            cmd.extend(["--tb=no"])
            
        exit_code, stdout, stderr = self.run_command(cmd, self.hooks_dir)
        
        result = {
            "name": "security",
            "exit_code": exit_code,
            "passed": exit_code == 0,
            "stdout": stdout,
            "stderr": stderr,
            "summary": self._extract_pytest_summary(stdout)
        }
        
        if result["passed"]:
            print("  ✅ Security tests passed")
        else:
            print("  ❌ Security tests failed")
            
        return result
    
    def run_mfc_tests(self, verbose: bool = False, fast: bool = False) -> Dict[str, Any]:
        """Run MFC project tests."""
        print("⚡ Running MFC project tests...")
        
        cmd = ["python", "-m", "pytest", "."]
        if fast:
            cmd.extend(["-k", "not integration and not selenium and not browser"])
        if verbose:
            cmd.append("-v")
        else:
            cmd.extend(["-q", "--tb=no"])
            
        exit_code, stdout, stderr = self.run_command(cmd, self.mfc_tests_dir)
        
        result = {
            "name": "mfc",
            "exit_code": exit_code,
            "passed": exit_code == 0,
            "stdout": stdout,
            "stderr": stderr,
            "summary": self._extract_pytest_summary(stdout)
        }
        
        if result["passed"]:
            print("  ✅ MFC tests passed")
        else:
            print("  ❌ MFC tests failed")
            
        return result
    
    def run_integration_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run cross-component integration tests."""
        print("🔗 Running integration tests...")
        
        # Use main pytest config to run all tests together
        cmd = ["python", "-m", "pytest", "-m", "integration"]
        if verbose:
            cmd.append("-v")
        else:
            cmd.extend(["-q", "--tb=no"])
            
        exit_code, stdout, stderr = self.run_command(cmd)
        
        result = {
            "name": "integration",
            "exit_code": exit_code,
            "passed": exit_code == 0,
            "stdout": stdout,
            "stderr": stderr,
            "summary": self._extract_pytest_summary(stdout)
        }
        
        if result["passed"]:
            print("  ✅ Integration tests passed")
        else:
            print("  ❌ Integration tests failed")
            
        return result
    
    def run_coverage_analysis(self, verbose: bool = False) -> Dict[str, Any]:
        """Run coverage analysis on all components."""
        print("📊 Running coverage analysis...")
        
        cmd = [
            "python", "-m", "pytest", 
            "--cov=.claude/hooks",
            "--cov=q-learning-mfcs/src",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml",
            "-q" if not verbose else "-v"
        ]
        
        exit_code, stdout, stderr = self.run_command(cmd)
        
        # Extract coverage percentage
        coverage_pct = self._extract_coverage_percentage(stdout)
        
        result = {
            "name": "coverage",
            "exit_code": exit_code,
            "passed": exit_code == 0,
            "coverage_percentage": coverage_pct,
            "stdout": stdout,
            "stderr": stderr,
            "summary": f"Coverage: {coverage_pct}%" if coverage_pct else "Coverage analysis completed"
        }
        
        if result["passed"]:
            print(f"  ✅ Coverage analysis completed ({coverage_pct}%)")
        else:
            print("  ❌ Coverage analysis failed")
            
        return result
    
    def run_lint_checks(self, verbose: bool = False) -> Dict[str, Any]:
        """Run linting checks on all components."""
        print("🧹 Running linting checks...")
        
        # Run ruff on both hooks and MFC code
        cmd = ["ruff", "check", ".", "--format=text"]
        exit_code, stdout, stderr = self.run_command(cmd)
        
        result = {
            "name": "linting",
            "exit_code": exit_code,
            "passed": exit_code == 0,
            "stdout": stdout,
            "stderr": stderr,
            "summary": "No linting issues found" if exit_code == 0 else "Linting issues detected"
        }
        
        if result["passed"]:
            print("  ✅ Linting checks passed")
        else:
            print("  ⚠️  Linting issues found (check output)")
            
        return result
    
    def run_type_checks(self, verbose: bool = False) -> Dict[str, Any]:
        """Run type checking with mypy."""
        print("🔍 Running type checks...")
        
        cmd = ["mypy", ".claude/hooks", "q-learning-mfcs/src", "--ignore-missing-imports"]
        exit_code, stdout, stderr = self.run_command(cmd)
        
        result = {
            "name": "typing",
            "exit_code": exit_code,
            "passed": exit_code == 0,
            "stdout": stdout,
            "stderr": stderr,
            "summary": "Type checking passed" if exit_code == 0 else "Type issues detected"
        }
        
        if result["passed"]:
            print("  ✅ Type checks passed")
        else:
            print("  ⚠️  Type issues found (check output)")
            
        return result
    
    def _extract_pytest_summary(self, output: str) -> str:
        """Extract pytest summary from output."""
        lines = output.split('\n')
        for line in reversed(lines):
            if 'passed' in line or 'failed' in line or 'error' in line:
                if any(keyword in line for keyword in ['passed', 'failed', 'error', 'warnings']):
                    return line.strip()
        return "No summary available"
    
    def _extract_coverage_percentage(self, output: str) -> Optional[str]:
        """Extract coverage percentage from pytest-cov output."""
        lines = output.split('\n')
        for line in lines:
            if 'TOTAL' in line and '%' in line:
                parts = line.split()
                for part in parts:
                    if '%' in part:
                        return part
        return None
    
    def generate_report(self, results: List[Dict[str, Any]], format: str = "text") -> str:
        """Generate test report in specified format."""
        if format == "json":
            return json.dumps({
                "timestamp": datetime.now().isoformat(),
                "results": results,
                "summary": {
                    "total_suites": len(results),
                    "passed_suites": len([r for r in results if r["passed"]]),
                    "failed_suites": len([r for r in results if not r["passed"]])
                }
            }, indent=2)
        
        # Text format
        report_lines = [
            "=" * 80,
            "MFC PROJECT INTEGRATED TEST REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        
        passed = 0
        failed = 0
        
        for result in results:
            status = "✅ PASSED" if result["passed"] else "❌ FAILED"
            report_lines.extend([
                f"{result['name'].upper()} TESTS: {status}",
                f"  Summary: {result['summary']}",
                ""
            ])
            
            if result["passed"]:
                passed += 1
            else:
                failed += 1
        
        report_lines.extend([
            "=" * 80,
            "OVERALL SUMMARY",
            "=" * 80,
            f"Total test suites: {len(results)}",
            f"Passed: {passed}",
            f"Failed: {failed}",
            f"Success rate: {(passed/len(results)*100):.1f}%" if results else "0%",
            ""
        ])
        
        if failed == 0:
            report_lines.append("🎉 ALL TESTS PASSED! 🎉")
        else:
            report_lines.append("⚠️  SOME TESTS FAILED - CHECK DETAILS ABOVE")
        
        return "\n".join(report_lines)
    
    def run_all(self, args: argparse.Namespace) -> bool:
        """Run all test suites based on arguments."""
        results = []
        
        if args.hooks or args.all:
            results.append(self.run_hooks_tests(args.verbose))
        
        if args.security or args.all:
            results.append(self.run_security_tests(args.verbose))
        
        if args.mfc or args.all:
            results.append(self.run_mfc_tests(args.verbose, args.fast))
        
        if args.integration or args.all:
            results.append(self.run_integration_tests(args.verbose))
        
        if args.coverage or args.all:
            results.append(self.run_coverage_analysis(args.verbose))
        
        if args.lint or args.all:
            results.append(self.run_lint_checks(args.verbose))
        
        if args.typing or args.all:
            results.append(self.run_type_checks(args.verbose))
        
        # Generate and save report
        report = self.generate_report(results, args.format)
        
        if args.format == "json":
            with open("test_report.json", "w") as f:
                f.write(report)
            print(f"\n📄 JSON report saved to: test_report.json")
        else:
            print(f"\n{report}")
            with open("test_report.txt", "w") as f:
                f.write(report)
            print(f"\n📄 Text report saved to: test_report.txt")
        
        # Return overall success
        return all(r["passed"] for r in results)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Integrated test runner for MFC project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_integrated_tests.py --all                    # Run all tests
  python run_integrated_tests.py --hooks --security      # Run hooks and security tests
  python run_integrated_tests.py --mfc --fast            # Run MFC tests quickly
  python run_integrated_tests.py --coverage --verbose    # Run with coverage and verbose output
  python run_integrated_tests.py --all --format json     # Generate JSON report
        """
    )
    
    # Test selection
    parser.add_argument("--all", action="store_true", help="Run all test suites")
    parser.add_argument("--hooks", action="store_true", help="Run Claude Code hooks tests")
    parser.add_argument("--security", action="store_true", help="Run security guardian tests")
    parser.add_argument("--mfc", action="store_true", help="Run MFC project tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--coverage", action="store_true", help="Run coverage analysis")
    parser.add_argument("--lint", action="store_true", help="Run linting checks")
    parser.add_argument("--typing", action="store_true", help="Run type checks")
    
    # Options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Report format")
    
    args = parser.parse_args()
    
    # Default to all tests if no specific selection
    if not any([args.hooks, args.security, args.mfc, args.integration, 
               args.coverage, args.lint, args.typing]):
        args.all = True
    
    print("🚀 Starting MFC Project Integrated Test Suite")
    print("=" * 80)
    
    runner = IntegratedTestRunner()
    success = runner.run_all(args)
    
    print("=" * 80)
    if success:
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()