#!/usr/bin/env python3
"""
GUI Test Runner for MFC Streamlit Application

This script runs all GUI tests and provides a summary report.
Run from the tests directory or the project root.
"""

import sys
import os
from pathlib import Path

def run_gui_tests():
    """Run all GUI tests and provide summary"""
    
    print("🧪 MFC GUI Test Suite Runner")
    print("=" * 50)
    
    # Get the tests directory
    tests_dir = Path(__file__).parent
    gui_tests_dir = tests_dir / "gui"
    
    if not gui_tests_dir.exists():
        print("❌ GUI tests directory not found!")
        return 1
    
    # Add src to path
    src_dir = tests_dir.parent / "src"
    sys.path.insert(0, str(src_dir))
    
    results = {}
    
    # Test 1: Simple GUI Test (HTTP-based, no browser required)
    print("\n🚀 Running Simple GUI Tests...")
    print("-" * 30)
    
    try:
        # Change to GUI tests directory
        original_cwd = os.getcwd()
        os.chdir(gui_tests_dir)
        
        # Import and run simple test
        from test_gui_simple import SimpleGUITester
        
        tester = SimpleGUITester(port=8506)  # Use different port to avoid conflicts
        test_results = tester.run_all_tests()
        
        results['simple_gui'] = {
            'passed': all(test_results.values()) if test_results else False,
            'details': test_results
        }
        
        os.chdir(original_cwd)
        
    except Exception as e:
        print(f"❌ Error running simple GUI tests: {e}")
        results['simple_gui'] = {'passed': False, 'details': {'error': str(e)}}
    
    # Test 2: Browser-based tests (requires Chrome/Selenium)
    print("\n🌐 Running Browser GUI Tests...")
    print("-" * 30)
    
    try:
        os.chdir(gui_tests_dir)
        
        # Check if Selenium is available
        try:
            import selenium
            from test_gui_browser import StreamlitGUITester
            
            browser_tester = StreamlitGUITester(port=8507)
            browser_results = browser_tester.run_all_tests()
            
            results['browser_gui'] = {
                'passed': all(browser_results.values()) if browser_results else False,
                'details': browser_results
            }
            
        except ImportError:
            print("⚠️ Selenium not available - skipping browser tests")
            print("💡 Install with: pixi add --pypi selenium webdriver-manager")
            results['browser_gui'] = {'passed': None, 'details': {'skipped': 'Selenium not available'}}
        
        os.chdir(original_cwd)
        
    except Exception as e:
        print(f"❌ Error running browser GUI tests: {e}")
        results['browser_gui'] = {'passed': False, 'details': {'error': str(e)}}
    
    # Test 3: Debug and specific functionality tests
    print("\n🐛 Running Specific Functionality Tests...")
    print("-" * 30)
    
    specific_results = {}
    
    test_files = [
        ('test_data_loading_fix.py', 'Data Loading Fix'),
        ('test_gui_autorefresh.py', 'GUI Autorefresh'),
        ('test_debug_simulation.py', 'Debug Simulation')
    ]
    
    for test_file, test_name in test_files:
        try:
            os.chdir(gui_tests_dir)
            
            # Run the test file as a subprocess to avoid conflicts
            import subprocess
            result = subprocess.run([sys.executable, test_file], 
                                  capture_output=True, text=True, timeout=60)
            
            test_passed = result.returncode == 0
            specific_results[test_name.lower().replace(' ', '_')] = test_passed
            
            if test_passed:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                if result.stderr:
                    print(f"   Error: {result.stderr[:200]}...")
            
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_name}: TIMEOUT")
            specific_results[test_name.lower().replace(' ', '_')] = False
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            specific_results[test_name.lower().replace(' ', '_')] = False
    
    results['specific_tests'] = {
        'passed': all(specific_results.values()),
        'details': specific_results
    }
    
    os.chdir(original_cwd)
    
    # Generate summary report
    print("\n📊 Test Results Summary")
    print("=" * 50)
    
    total_passed = 0
    total_tests = 0
    
    for category, result in results.items():
        if result['passed'] is True:
            status = "✅ PASSED"
            total_passed += 1
        elif result['passed'] is False:
            status = "❌ FAILED"
        else:
            status = "⚠️ SKIPPED"
        
        total_tests += 1
        print(f"{category.replace('_', ' ').title()}: {status}")
        
        # Show details for failed tests
        if result['passed'] is False and 'details' in result:
            for detail_name, detail_result in result['details'].items():
                if not detail_result and detail_result is not None:
                    print(f"   ❌ {detail_name}")
    
    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    print(f"\nOverall Success Rate: {total_passed}/{total_tests} ({success_rate:.1f}%)")
    
    # Determine exit code
    if total_passed == total_tests:
        print("🎉 All tests passed!")
        return 0
    elif total_passed > 0:
        print("⚠️ Some tests passed, some failed")
        return 1
    else:
        print("❌ All tests failed")
        return 2

if __name__ == "__main__":
    exit(run_gui_tests())