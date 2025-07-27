#!/usr/bin/env python3
"""
Test runner for the MFC Q-Learning project path output tests.
"""

import unittest
import sys
import os
from pathlib import Path

# Add the tests directory and src directory to the path
tests_dir = Path(__file__).parent
src_dir = tests_dir.parent / 'src'
sys.path.insert(0, str(tests_dir))
sys.path.insert(0, str(src_dir))

# Import test modules
from test_path_config import TestPathConfig
from test_file_outputs import TestFileOutputIntegration, TestSpecificFileImports
from test_actual_executions import TestActualFileExecutions, TestFileOutputPatterns
from test_gpu_capability import TestGPUCapability
from test_gpu_acceleration import TestGPUAcceleration

# Import enhanced test modules
try:
    from test_comprehensive_edge_cases import (
        TestBoundaryConditions, TestConfigurationValidation, 
        TestMemoryAndPerformance, TestErrorRecovery, TestDataIntegrity,
        TestConcurrencyAndThreadSafety, TestRegressionCases, TestCompatibilityMatrix
    )
    enhanced_tests_available = True
except ImportError as e:
    print(f"Enhanced edge case tests not available: {e}")
    enhanced_tests_available = False

try:
    from test_biological_constraints import (
        TestBiologicalConstraints, TestPhysicalConstraints,
        TestChemicalConstraints, TestThermodynamicConstraints
    )
    biological_tests_available = True
except ImportError as e:
    print(f"Biological constraint tests not available: {e}")
    biological_tests_available = False

try:
    from test_performance_stress import (
        TestPerformanceBenchmarks, TestStressTests, 
        TestScalabilityTests, TestResourceUtilization
    )
    performance_tests_available = True
except ImportError as e:
    print(f"Performance stress tests not available: {e}")
    performance_tests_available = False


def create_test_suite(test_category='all'):
    """Create a comprehensive test suite."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Core tests - always included
    if test_category in ['all', 'core', 'path_config']:
        suite.addTest(loader.loadTestsFromTestCase(TestPathConfig))
    
    if test_category in ['all', 'core', 'file_outputs']:
        suite.addTest(loader.loadTestsFromTestCase(TestFileOutputIntegration))
        suite.addTest(loader.loadTestsFromTestCase(TestSpecificFileImports))
    
    if test_category in ['all', 'core', 'executions']:
        suite.addTest(loader.loadTestsFromTestCase(TestActualFileExecutions))
        suite.addTest(loader.loadTestsFromTestCase(TestFileOutputPatterns))
    
    if test_category in ['all', 'core', 'gpu_capability']:
        suite.addTest(loader.loadTestsFromTestCase(TestGPUCapability))
    
    if test_category in ['all', 'core', 'gpu_acceleration']:
        suite.addTest(loader.loadTestsFromTestCase(TestGPUAcceleration))
    
    # Enhanced tests
    if enhanced_tests_available and test_category in ['all', 'enhanced', 'edge_cases']:
        suite.addTest(loader.loadTestsFromTestCase(TestBoundaryConditions))
        suite.addTest(loader.loadTestsFromTestCase(TestConfigurationValidation))
        suite.addTest(loader.loadTestsFromTestCase(TestMemoryAndPerformance))
        suite.addTest(loader.loadTestsFromTestCase(TestErrorRecovery))
        suite.addTest(loader.loadTestsFromTestCase(TestDataIntegrity))
        suite.addTest(loader.loadTestsFromTestCase(TestConcurrencyAndThreadSafety))
        suite.addTest(loader.loadTestsFromTestCase(TestRegressionCases))
        suite.addTest(loader.loadTestsFromTestCase(TestCompatibilityMatrix))
    
    # Biological constraint tests
    if biological_tests_available and test_category in ['all', 'enhanced', 'biological']:
        suite.addTest(loader.loadTestsFromTestCase(TestBiologicalConstraints))
        suite.addTest(loader.loadTestsFromTestCase(TestPhysicalConstraints))
        suite.addTest(loader.loadTestsFromTestCase(TestChemicalConstraints))
        suite.addTest(loader.loadTestsFromTestCase(TestThermodynamicConstraints))
    
    # Performance and stress tests
    if performance_tests_available and test_category in ['all', 'enhanced', 'performance']:
        suite.addTest(loader.loadTestsFromTestCase(TestPerformanceBenchmarks))
        suite.addTest(loader.loadTestsFromTestCase(TestStressTests))
        suite.addTest(loader.loadTestsFromTestCase(TestScalabilityTests))
        suite.addTest(loader.loadTestsFromTestCase(TestResourceUtilization))
    
    return suite


def run_tests(verbosity=2, test_category='all'):
    """Run tests with specified verbosity and category."""
    print("🧪 MFC Q-Learning Project - Comprehensive Test Suite")
    print("=" * 60)
    print(f"Running tests from: {Path(__file__).parent}")
    print(f"Source directory: {src_dir}")
    print(f"Test category: {test_category}")
    print("=" * 60)
    
    # Print availability status
    if test_category in ['all', 'enhanced']:
        print("📊 Enhanced Test Modules:")
        print(f"  Edge Cases: {'✅' if enhanced_tests_available else '❌'}")
        print(f"  Biological: {'✅' if biological_tests_available else '❌'}")
        print(f"  Performance: {'✅' if performance_tests_available else '❌'}")
        print("-" * 60)
    
    # Create and run test suite
    suite = create_test_suite(test_category)
    runner = unittest.TextTestRunner(verbosity=verbosity, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    successful = total_tests - failures - errors - skipped
    
    print(f"Tests run: {total_tests}")
    print(f"Successful: {successful}")
    print(f"Failures: {failures}")  
    print(f"Errors: {errors}")
    print(f"Skipped: {skipped}")
    
    if failures == 0 and errors == 0:
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print("❌ Some tests failed. See details above.")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
        
        return False


def run_specific_test_class(test_class_name, verbosity=2):
    """Run a specific test class."""
    test_classes = {
        'path_config': TestPathConfig,
        'file_outputs': TestFileOutputIntegration,
        'imports': TestSpecificFileImports,
        'executions': TestActualFileExecutions,
        'patterns': TestFileOutputPatterns,
        'gpu_capability': TestGPUCapability,
        'gpu_acceleration': TestGPUAcceleration
    }
    
    # Add enhanced test classes if available
    if enhanced_tests_available:
        test_classes.update({
            'boundary_conditions': TestBoundaryConditions,
            'config_validation': TestConfigurationValidation,
            'memory_performance': TestMemoryAndPerformance,
            'error_recovery': TestErrorRecovery,
            'data_integrity': TestDataIntegrity,
            'concurrency': TestConcurrencyAndThreadSafety,
            'regression': TestRegressionCases,
            'compatibility': TestCompatibilityMatrix,
        })
    
    if biological_tests_available:
        test_classes.update({
            'biological_constraints': TestBiologicalConstraints,
            'physical_constraints': TestPhysicalConstraints,
            'chemical_constraints': TestChemicalConstraints,
            'thermodynamic_constraints': TestThermodynamicConstraints,
        })
    
    if performance_tests_available:
        test_classes.update({
            'performance_benchmarks': TestPerformanceBenchmarks,
            'stress_tests': TestStressTests,
            'scalability_tests': TestScalabilityTests,
            'resource_utilization': TestResourceUtilization,
        })
    
    if test_class_name not in test_classes:
        print(f"Unknown test class: {test_class_name}")
        print(f"Available: {', '.join(test_classes.keys())}")
        return False
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(test_classes[test_class_name])
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run MFC Q-Learning path output tests')
    parser.add_argument('--test-class', '-c', 
                       help='Run specific test class')
    parser.add_argument('--test-category', '-t',
                       choices=['all', 'core', 'enhanced', 'edge_cases', 'biological', 'performance'],
                       default='all',
                       help='Test category to run')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', 
                       help='Quiet output')
    parser.add_argument('--list-tests', '-l', action='store_true',
                       help='List available test classes')
    
    args = parser.parse_args()
    
    # List available tests
    if args.list_tests:
        print("Available test classes:")
        print("\nCore tests:")
        core_tests = ['path_config', 'file_outputs', 'imports', 'executions', 'patterns', 'gpu_capability', 'gpu_acceleration']
        for test in core_tests:
            print(f"  {test}")
        
        if enhanced_tests_available:
            print("\nEnhanced tests:")
            enhanced_test_names = ['boundary_conditions', 'config_validation', 'memory_performance', 'error_recovery', 'data_integrity', 'concurrency', 'regression', 'compatibility']
            for test in enhanced_test_names:
                print(f"  {test}")
        
        if biological_tests_available:
            print("\nBiological tests:")
            bio_test_names = ['biological_constraints', 'physical_constraints', 'chemical_constraints', 'thermodynamic_constraints']
            for test in bio_test_names:
                print(f"  {test}")
        
        if performance_tests_available:
            print("\nPerformance tests:")
            perf_test_names = ['performance_benchmarks', 'stress_tests', 'scalability_tests', 'resource_utilization']
            for test in perf_test_names:
                print(f"  {test}")
        
        sys.exit(0)
    
    # Set verbosity
    if args.quiet:
        verbosity = 0
    elif args.verbose:
        verbosity = 2
    else:
        verbosity = 1
    
    # Run tests
    if args.test_class:
        success = run_specific_test_class(args.test_class, verbosity)
    else:
        success = run_tests(verbosity, args.test_category)
    
    sys.exit(0 if success else 1)