#!/usr/bin/env python3
"""
Test Result Collector for MFC Q-Learning Project.
Collects, analyzes, and processes test results for issue creation.
"""

import json
import unittest
import sys
import os
import traceback
import time
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

# Add source path
tests_dir = Path(__file__).parent
src_dir = tests_dir.parent / 'src'
sys.path.insert(0, str(tests_dir))
sys.path.insert(0, str(src_dir))


@dataclass
class TestResult:
    """Individual test result."""
    test_name: str
    test_class: str
    test_method: str
    status: str  # 'pass', 'fail', 'error', 'skip'
    duration: float
    error_message: str = ""
    stack_trace: str = ""
    skip_reason: str = ""


@dataclass
class TestSuiteResult:
    """Complete test suite result."""
    timestamp: str
    total_tests: int
    passed: int
    failed: int
    errors: int
    skipped: int
    duration: float
    test_results: List[TestResult]
    system_info: Dict[str, Any]


class CustomTestResult(unittest.TestResult):
    """Custom test result collector."""
    
    def __init__(self):
        super().__init__()
        self.test_results = []
        self.start_time = time.time()
        self.test_start_times = {}
    
    def startTest(self, test):
        """Called when a test starts."""
        super().startTest(test)
        self.test_start_times[test] = time.time()
    
    def addSuccess(self, test):
        """Called when a test passes."""
        super().addSuccess(test)
        duration = time.time() - self.test_start_times.get(test, time.time())
        
        test_result = TestResult(
            test_name=str(test),
            test_class=test.__class__.__name__,
            test_method=test._testMethodName,
            status='pass',
            duration=duration
        )
        self.test_results.append(test_result)
    
    def addError(self, test, err):
        """Called when a test has an error."""
        super().addError(test, err)
        duration = time.time() - self.test_start_times.get(test, time.time())
        
        exc_type, exc_value, exc_traceback = err
        error_message = str(exc_value)
        stack_trace = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        
        test_result = TestResult(
            test_name=str(test),
            test_class=test.__class__.__name__,
            test_method=test._testMethodName,
            status='error',
            duration=duration,
            error_message=error_message,
            stack_trace=stack_trace
        )
        self.test_results.append(test_result)
    
    def addFailure(self, test, err):
        """Called when a test fails."""
        super().addFailure(test, err)
        duration = time.time() - self.test_start_times.get(test, time.time())
        
        exc_type, exc_value, exc_traceback = err
        error_message = str(exc_value)
        stack_trace = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        
        test_result = TestResult(
            test_name=str(test),
            test_class=test.__class__.__name__,
            test_method=test._testMethodName,
            status='fail',
            duration=duration,
            error_message=error_message,
            stack_trace=stack_trace
        )
        self.test_results.append(test_result)
    
    def addSkip(self, test, reason):
        """Called when a test is skipped."""
        super().addSkip(test, reason)
        duration = time.time() - self.test_start_times.get(test, time.time())
        
        test_result = TestResult(
            test_name=str(test),
            test_class=test.__class__.__name__,
            test_method=test._testMethodName,
            status='skip',
            duration=duration,
            skip_reason=reason
        )
        self.test_results.append(test_result)


class TestRunner:
    """Enhanced test runner with result collection."""
    
    def __init__(self):
        """Initialize test runner."""
        self.result_collector = CustomTestResult()
    
    def run_test_suite(self, test_suite) -> TestSuiteResult:
        """Run test suite and collect results."""
        start_time = time.time()
        
        # Run tests
        runner = unittest.TextTestRunner(
            stream=sys.stderr,  # Send output to stderr to keep stdout clean
            verbosity=1
        )
        
        # Replace the result with our custom collector
        runner.resultclass = CustomTestResult
        
        # Collect system information
        system_info = self._collect_system_info()
        
        # Run tests
        unittest_result = runner.run(test_suite)
        
        # Calculate duration
        total_duration = time.time() - start_time
        
        # Create comprehensive result
        suite_result = TestSuiteResult(
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            total_tests=unittest_result.testsRun,
            passed=unittest_result.testsRun - len(unittest_result.failures) - len(unittest_result.errors) - len(getattr(unittest_result, 'skipped', [])),
            failed=len(unittest_result.failures),
            errors=len(unittest_result.errors),
            skipped=len(getattr(unittest_result, 'skipped', [])),
            duration=total_duration,
            test_results=unittest_result.test_results if hasattr(unittest_result, 'test_results') else [],
            system_info=system_info
        )
        
        return suite_result
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information."""
        import platform
        try:
            import psutil
            memory_info = {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'used': psutil.virtual_memory().used
            }
        except ImportError:
            memory_info = {'error': 'psutil not available'}
        
        try:
            import gpu_acceleration
            gpu_info = {
                'gpu_available': True,
                'backend': getattr(gpu_acceleration.GPUAccelerator(), 'backend', 'unknown')
            }
        except ImportError:
            gpu_info = {'gpu_available': False}
        
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'architecture': platform.architecture(),
            'processor': platform.processor(),
            'memory': memory_info,
            'gpu': gpu_info,
            'working_directory': os.getcwd(),
            'python_path': sys.path[:3]  # First 3 entries
        }


class ResultAnalyzer:
    """Analyzes test results and categorizes issues."""
    
    def __init__(self):
        """Initialize result analyzer."""
        pass
    
    def analyze_results(self, suite_result: TestSuiteResult) -> Dict[str, Any]:
        """Analyze test results and categorize issues."""
        analysis = {
            'summary': self._create_summary(suite_result),
            'failure_analysis': self._analyze_failures(suite_result),
            'error_analysis': self._analyze_errors(suite_result),
            'performance_analysis': self._analyze_performance(suite_result),
            'component_analysis': self._analyze_by_component(suite_result),
            'recommendations': self._generate_recommendations(suite_result)
        }
        
        return analysis
    
    def _create_summary(self, suite_result: TestSuiteResult) -> Dict[str, Any]:
        """Create test summary."""
        success_rate = (suite_result.passed / suite_result.total_tests * 100) if suite_result.total_tests > 0 else 0
        
        return {
            'total_tests': suite_result.total_tests,
            'passed': suite_result.passed,
            'failed': suite_result.failed,
            'errors': suite_result.errors,
            'skipped': suite_result.skipped,
            'success_rate': round(success_rate, 2),
            'duration': round(suite_result.duration, 2),
            'timestamp': suite_result.timestamp
        }
    
    def _analyze_failures(self, suite_result: TestSuiteResult) -> List[Dict[str, Any]]:
        """Analyze test failures."""
        failures = [tr for tr in suite_result.test_results if tr.status == 'fail']
        
        failure_analysis = []
        for failure in failures:
            analysis = {
                'test_name': failure.test_name,
                'test_class': failure.test_class,
                'test_method': failure.test_method,
                'error_message': failure.error_message,
                'component': self._identify_component(failure.test_name),
                'severity': self._assess_severity(failure),
                'likely_cause': self._identify_likely_cause(failure),
                'duration': failure.duration
            }
            failure_analysis.append(analysis)
        
        return failure_analysis
    
    def _analyze_errors(self, suite_result: TestSuiteResult) -> List[Dict[str, Any]]:
        """Analyze test errors."""
        errors = [tr for tr in suite_result.test_results if tr.status == 'error']
        
        error_analysis = []
        for error in errors:
            analysis = {
                'test_name': error.test_name,
                'test_class': error.test_class,
                'test_method': error.test_method,
                'error_message': error.error_message,
                'component': self._identify_component(error.test_name),
                'severity': 'high',  # Errors are typically more severe
                'error_type': self._classify_error_type(error),
                'duration': error.duration
            }
            error_analysis.append(analysis)
        
        return error_analysis
    
    def _analyze_performance(self, suite_result: TestSuiteResult) -> Dict[str, Any]:
        """Analyze performance aspects."""
        durations = [tr.duration for tr in suite_result.test_results]
        
        if not durations:
            return {'no_data': True}
        
        slow_tests = [tr for tr in suite_result.test_results if tr.duration > 5.0]
        
        return {
            'total_duration': round(suite_result.duration, 2),
            'average_test_duration': round(sum(durations) / len(durations), 3),
            'slowest_test_duration': round(max(durations), 2),
            'fastest_test_duration': round(min(durations), 3),
            'slow_tests_count': len(slow_tests),
            'slow_tests': [{'name': tr.test_name, 'duration': tr.duration} for tr in slow_tests[:5]]
        }
    
    def _analyze_by_component(self, suite_result: TestSuiteResult) -> Dict[str, Any]:
        """Analyze results by component."""
        components = {}
        
        for test_result in suite_result.test_results:
            component = self._identify_component(test_result.test_name)
            
            if component not in components:
                components[component] = {
                    'total': 0,
                    'passed': 0,
                    'failed': 0,
                    'errors': 0,
                    'skipped': 0
                }
            
            components[component]['total'] += 1
            
            if test_result.status == 'pass':
                components[component]['passed'] += 1
            elif test_result.status == 'fail':
                components[component]['failed'] += 1
            elif test_result.status == 'error':
                components[component]['errors'] += 1
            elif test_result.status == 'skip':
                components[component]['skipped'] += 1
        
        # Calculate success rates
        for component, stats in components.items():
            if stats['total'] > 0:
                stats['success_rate'] = round((stats['passed'] / stats['total']) * 100, 2)
            else:
                stats['success_rate'] = 0
        
        return components
    
    def _generate_recommendations(self, suite_result: TestSuiteResult) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Check success rate
        success_rate = (suite_result.passed / suite_result.total_tests * 100) if suite_result.total_tests > 0 else 0
        
        if success_rate < 80:
            recommendations.append(f"Low success rate ({success_rate:.1f}%). Focus on fixing critical failures first.")
        
        # Check for errors
        if suite_result.errors > 0:
            recommendations.append(f"{suite_result.errors} tests have runtime errors. These should be prioritized as they indicate serious issues.")
        
        # Check for performance issues
        slow_tests = [tr for tr in suite_result.test_results if tr.duration > 10.0]
        if slow_tests:
            recommendations.append(f"{len(slow_tests)} tests are running slowly (>10s). Consider optimization or moving to performance test category.")
        
        # Check for skipped tests
        if suite_result.skipped > suite_result.total_tests * 0.3:
            recommendations.append(f"High number of skipped tests ({suite_result.skipped}). Review dependencies and test prerequisites.")
        
        # Component-specific recommendations
        component_analysis = self._analyze_by_component(suite_result)
        for component, stats in component_analysis.items():
            if stats['success_rate'] < 50 and stats['total'] > 1:
                recommendations.append(f"Component '{component}' has low success rate ({stats['success_rate']:.1f}%). Needs focused attention.")
        
        return recommendations
    
    def _identify_component(self, test_name: str) -> str:
        """Identify component from test name."""
        test_name_lower = test_name.lower()
        
        if 'gpu' in test_name_lower:
            return 'gpu-acceleration'
        elif 'biofilm' in test_name_lower:
            return 'biofilm-model'
        elif 'metabolic' in test_name_lower:
            return 'metabolic-model'
        elif 'sensor' in test_name_lower or 'eis' in test_name_lower or 'qcm' in test_name_lower:
            return 'sensor-fusion'
        elif 'qlearning' in test_name_lower or 'q_learning' in test_name_lower:
            return 'q-learning'
        elif 'mfc' in test_name_lower and 'stack' in test_name_lower:
            return 'mfc-stack'
        elif 'config' in test_name_lower:
            return 'configuration'
        elif 'path' in test_name_lower:
            return 'path-management'
        elif 'performance' in test_name_lower or 'stress' in test_name_lower:
            return 'performance'
        elif 'biological' in test_name_lower or 'physical' in test_name_lower:
            return 'constraints'
        else:
            return 'core'
    
    def _assess_severity(self, test_result: TestResult) -> str:
        """Assess severity of test failure."""
        test_name_lower = test_result.test_name.lower()
        error_message_lower = test_result.error_message.lower()
        
        # Critical indicators
        if any(keyword in test_name_lower for keyword in ['critical', 'security', 'safety']):
            return 'critical'
        
        if any(keyword in error_message_lower for keyword in ['segmentation fault', 'memory error', 'stack overflow']):
            return 'critical'
        
        # High severity indicators
        if any(keyword in test_name_lower for keyword in ['integration', 'core', 'main']):
            return 'high'
        
        if any(keyword in error_message_lower for keyword in ['import error', 'module not found', 'connection']):
            return 'high'
        
        # Medium severity indicators
        if any(keyword in test_name_lower for keyword in ['performance', 'optimization']):
            return 'medium'
        
        # Default to low
        return 'low'
    
    def _identify_likely_cause(self, test_result: TestResult) -> str:
        """Identify likely cause of failure."""
        error_message_lower = test_result.error_message.lower()
        
        if 'import' in error_message_lower or 'module' in error_message_lower:
            return 'missing_dependency'
        elif 'connection' in error_message_lower or 'timeout' in error_message_lower:
            return 'network_issue'
        elif 'memory' in error_message_lower:
            return 'memory_issue'
        elif 'permission' in error_message_lower or 'access' in error_message_lower:
            return 'permission_issue'
        elif 'assertion' in error_message_lower:
            return 'logic_error'
        elif 'gpu' in error_message_lower or 'cuda' in error_message_lower or 'rocm' in error_message_lower:
            return 'gpu_configuration'
        else:
            return 'unknown'
    
    def _classify_error_type(self, test_result: TestResult) -> str:
        """Classify type of error."""
        error_message_lower = test_result.error_message.lower()
        
        if 'importerror' in error_message_lower or 'modulenotfounderror' in error_message_lower:
            return 'import_error'
        elif 'attributeerror' in error_message_lower:
            return 'attribute_error'
        elif 'typeerror' in error_message_lower:
            return 'type_error'
        elif 'valueerror' in error_message_lower:
            return 'value_error'
        elif 'keyerror' in error_message_lower:
            return 'key_error'
        elif 'indexerror' in error_message_lower:
            return 'index_error'
        elif 'memoryerror' in error_message_lower:
            return 'memory_error'
        elif 'timeouterror' in error_message_lower:
            return 'timeout_error'
        else:
            return 'runtime_error'


def save_results(suite_result: TestSuiteResult, analysis: Dict[str, Any], output_file: str = None):
    """Save test results and analysis to file."""
    if output_file is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_file = f"test_results_{timestamp}.json"
    
    output_data = {
        'test_suite_result': asdict(suite_result),
        'analysis': analysis
    }
    
    # Convert TestResult objects to dictionaries
    output_data['test_suite_result']['test_results'] = [
        asdict(tr) for tr in suite_result.test_results
    ]
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"✅ Test results saved to: {output_file}")
    return output_file


def main():
    """Main function for standalone usage."""
    import argparse
    from run_tests import create_test_suite
    
    parser = argparse.ArgumentParser(description='Test Result Collector for MFC Q-Learning Project')
    parser.add_argument('--test-category', '-t',
                       choices=['all', 'core', 'enhanced', 'edge_cases', 'biological', 'performance'],
                       default='core',
                       help='Test category to run')
    parser.add_argument('--output', '-o', type=str, help='Output file for results')
    parser.add_argument('--create-issues', action='store_true', help='Create GitLab issues for failures')
    
    args = parser.parse_args()
    
    print("🧪 Running MFC Q-Learning Test Suite with Result Collection")
    print("=" * 60)
    
    # Create test suite
    suite = create_test_suite(args.test_category)
    
    # Run tests with result collection
    runner = TestRunner()
    suite_result = runner.run_test_suite(suite)
    
    # Analyze results
    analyzer = ResultAnalyzer()
    analysis = analyzer.analyze_results(suite_result)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    summary = analysis['summary']
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Errors: {summary['errors']}")
    print(f"Skipped: {summary['skipped']}")
    print(f"Success Rate: {summary['success_rate']}%")
    print(f"Duration: {summary['duration']}s")
    
    # Print recommendations
    if analysis['recommendations']:
        print("\n📋 RECOMMENDATIONS:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"{i}. {rec}")
    
    # Save results
    save_results(suite_result, analysis, args.output)
    
    # Create GitLab issues if requested
    if args.create_issues and (suite_result.failed > 0 or suite_result.errors > 0):
        try:
            from gitlab_issue_manager import GitLabIssueManager, TestResultProcessor
            
            issue_manager = GitLabIssueManager()
            processor = TestResultProcessor(issue_manager)
            
            # Convert results to expected format
            test_results_for_gitlab = {
                'failures': [(tr.test_name, tr.stack_trace) for tr in suite_result.test_results if tr.status == 'fail'],
                'errors': [(tr.test_name, tr.stack_trace) for tr in suite_result.test_results if tr.status == 'error']
            }
            
            created_issues = processor.process_test_failures(test_results_for_gitlab)
            
            print(f"\n🎫 Created {len(created_issues)} GitLab issues for test failures")
            for issue in created_issues:
                print(f"  #{issue['iid']}: {issue['title']}")
        
        except Exception as e:
            print(f"⚠️ Could not create GitLab issues: {e}")
    
    # Exit with appropriate code
    sys.exit(0 if suite_result.failed == 0 and suite_result.errors == 0 else 1)


if __name__ == '__main__':
    main()