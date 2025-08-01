#!/usr/bin/env python3
"""
Test runner for the MFC Q-Learning project path output tests.
"""

import unittest
import sys
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
from test_advanced_electrode_model import (
    TestCellGeometry, TestFluidDynamicsProperties, TestMassTransportProperties,
    TestBiofilmDynamics, TestAdvancedElectrodeModel, TestIntegrationTests
)


def create_test_suite():
    """Create a comprehensive test suite."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add path configuration tests
    suite.addTest(loader.loadTestsFromTestCase(TestPathConfig))

    # Add file output integration tests
    suite.addTest(loader.loadTestsFromTestCase(TestFileOutputIntegration))
    suite.addTest(loader.loadTestsFromTestCase(TestSpecificFileImports))

    # Add actual execution tests
    suite.addTest(loader.loadTestsFromTestCase(TestActualFileExecutions))
    suite.addTest(loader.loadTestsFromTestCase(TestFileOutputPatterns))

    # Add GPU capability tests
    suite.addTest(loader.loadTestsFromTestCase(TestGPUCapability))
    suite.addTest(loader.loadTestsFromTestCase(TestGPUAcceleration))

    # Add advanced electrode model tests (Phase 2)
    suite.addTest(loader.loadTestsFromTestCase(TestCellGeometry))
    suite.addTest(loader.loadTestsFromTestCase(TestFluidDynamicsProperties))
    suite.addTest(loader.loadTestsFromTestCase(TestMassTransportProperties))
    suite.addTest(loader.loadTestsFromTestCase(TestBiofilmDynamics))
    suite.addTest(loader.loadTestsFromTestCase(TestAdvancedElectrodeModel))
    suite.addTest(loader.loadTestsFromTestCase(TestIntegrationTests))

    return suite


def run_tests(verbosity=2):
    """Run all tests with specified verbosity."""
    print("🧪 MFC Q-Learning Project - Comprehensive Test Suite")
    print("=" * 60)
    print(f"Running tests from: {Path(__file__).parent}")
    print(f"Source directory: {src_dir}")
    print("=" * 60)

    # Create and run test suite
    suite = create_test_suite()
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
        'gpu_acceleration': TestGPUAcceleration,
        'cell_geometry': TestCellGeometry,
        'fluid_dynamics': TestFluidDynamicsProperties,
        'mass_transport': TestMassTransportProperties,
        'biofilm_dynamics': TestBiofilmDynamics,
        'advanced_electrode': TestAdvancedElectrodeModel,
        'integration_tests': TestIntegrationTests
    }

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
                       help='Run specific test class (path_config, file_outputs, imports, executions, patterns, gpu_capability, gpu_acceleration, cell_geometry, fluid_dynamics, mass_transport, biofilm_dynamics, advanced_electrode, integration_tests)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet output')

    args = parser.parse_args()

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
        success = run_tests(verbosity)

    sys.exit(0 if success else 1)
