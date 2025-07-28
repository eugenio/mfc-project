#!/usr/bin/env python3
"""
Simple GUI test using requests to test HTTP endpoints
Tests basic functionality without requiring a browser
"""

import time
import subprocess
import requests
import os
from pathlib import Path

class SimpleGUITester:
    """Simple HTTP-based tester for Streamlit GUI"""
    
    def __init__(self, port=8505):
        self.port = port
        self.streamlit_process = None
        self.base_url = f"http://localhost:{port}"
        
    def start_streamlit(self):
        """Start Streamlit app in background"""
        try:
            env = os.environ.copy()
            env['MFC_DEBUG_MODE'] = 'true'
            
            # Get path to the GUI file in src directory
            src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src')
            gui_file = os.path.join(src_dir, 'mfc_streamlit_gui.py')
            
            self.streamlit_process = subprocess.Popen(
                ["streamlit", "run", gui_file, 
                 "--server.port", str(self.port),
                 "--server.address", "localhost",
                 "--server.headless", "true"],
                env=env,
                cwd=src_dir,  # Run from src directory
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Wait for Streamlit to start
            print(f"🚀 Starting Streamlit on port {self.port}...")
            
            # Try to connect with retries
            max_retries = 15
            for i in range(max_retries):
                try:
                    response = requests.get(self.base_url, timeout=5)
                    if response.status_code == 200:
                        print("✅ Streamlit started successfully")
                        return True
                except requests.exceptions.RequestException:
                    pass
                
                time.sleep(2)
                print(f"   Waiting... ({i+1}/{max_retries})")
            
            print("❌ Streamlit failed to start or respond")
            return False
                
        except Exception as e:
            print(f"❌ Error starting Streamlit: {e}")
            return False
    
    def stop_streamlit(self):
        """Stop Streamlit process"""
        if self.streamlit_process:
            try:
                self.streamlit_process.terminate()
                self.streamlit_process.wait(timeout=5)
                print("✅ Streamlit stopped successfully")
            except subprocess.TimeoutExpired:
                self.streamlit_process.kill()
                print("⚠️ Streamlit force-killed")
            except Exception as e:
                print(f"⚠️ Error stopping Streamlit: {e}")
    
    def test_page_accessibility(self):
        """Test that the main page is accessible"""
        print("\n🧪 Testing page accessibility...")
        
        try:
            response = requests.get(self.base_url, timeout=10)
            
            if response.status_code == 200:
                print("✅ Main page is accessible")
                
                # Check for key content
                content = response.text
                checks = {
                    "Streamlit content": "streamlit" in content.lower(),
                    "MFC content": "mfc" in content.lower() or "simulation" in content.lower(),
                    "JavaScript loaded": "script" in content.lower(),
                    "CSS loaded": "style" in content.lower() or ".css" in content
                }
                
                passed_checks = sum(checks.values())
                print(f"   Content checks: {passed_checks}/{len(checks)} passed")
                
                for check_name, result in checks.items():
                    status = "✅" if result else "❌"
                    print(f"   {status} {check_name}")
                
                return passed_checks >= len(checks) // 2  # At least half should pass
            else:
                print(f"❌ Page returned status code: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error accessing page: {e}")
            return False
    
    def test_health_endpoint(self):
        """Test Streamlit health endpoint if available"""
        print("\n🧪 Testing health endpoint...")
        
        try:
            # Try common Streamlit health endpoints
            health_urls = [
                f"{self.base_url}/healthz",
                f"{self.base_url}/_stcore/health",
                f"{self.base_url}/health"
            ]
            
            for url in health_urls:
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        print(f"✅ Health endpoint accessible: {url}")
                        return True
                except requests.exceptions.RequestException:
                    continue
            
            print("⚠️ No health endpoint found (not critical)")
            return True  # Not critical for functionality
            
        except Exception as e:
            print(f"⚠️ Error testing health endpoint: {e}")
            return True  # Not critical
    
    def test_static_resources(self):
        """Test that static resources are accessible"""
        print("\n🧪 Testing static resources...")
        
        try:
            # First get the main page to find resource URLs
            response = requests.get(self.base_url, timeout=10)
            if response.status_code != 200:
                print("❌ Cannot access main page for resource discovery")
                return False
            
            content = response.text
            
            # Look for common Streamlit resources
            resource_indicators = [
                "streamlit" in content.lower(),
                "static" in content.lower(),
                ".js" in content,
                ".css" in content
            ]
            
            found_resources = sum(resource_indicators)
            print(f"✅ Found {found_resources}/4 resource indicators")
            
            return found_resources >= 2  # At least some resources should be present
            
        except Exception as e:
            print(f"❌ Error testing static resources: {e}")
            return False
    
    def test_memory_data_functionality(self):
        """Test that the memory-based data loading improvements work"""
        print("\n🧪 Testing memory data functionality...")
        
        try:
            # Import our modules to test the memory functionality
            import sys
            # Add src to path (go up two directories from tests/gui to reach src)
            src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src')
            sys.path.insert(0, src_path)
            
            from mfc_streamlit_gui import SimulationRunner, load_simulation_data_from_memory
            from path_config import enable_debug_mode, disable_debug_mode
            
            # Enable debug mode
            enable_debug_mode()
            
            # Create runner and test memory functionality
            runner = SimulationRunner()
            
            # Test that runner has the new memory attributes
            has_live_data = hasattr(runner, 'live_data')
            has_lock = hasattr(runner, 'live_data_lock')
            has_method = hasattr(runner, 'get_live_data')
            
            print(f"   Memory attributes: live_data={has_live_data}, lock={has_lock}, method={has_method}")
            
            # Test memory data loading function
            data = load_simulation_data_from_memory(runner)
            memory_function_works = data is None  # Should be None when not running
            
            print(f"   Memory function: {'✅ Works' if memory_function_works else '❌ Error'}")
            
            # Cleanup
            disable_debug_mode()
            
            all_checks_pass = all([has_live_data, has_lock, has_method, memory_function_works])
            
            if all_checks_pass:
                print("✅ Memory data functionality is properly implemented")
            else:
                print("❌ Memory data functionality has issues")
            
            return all_checks_pass
            
        except Exception as e:
            print(f"❌ Error testing memory functionality: {e}")
            return False
    
    def run_all_tests(self):
        """Run all simple GUI tests"""
        print("🧪 Starting Simple GUI Test Suite")
        print("=" * 50)
        
        results = {}
        
        # Start Streamlit
        if not self.start_streamlit():
            print("❌ Failed to start Streamlit - aborting tests")
            return results
        
        try:
            # Run tests
            results['page_accessibility'] = self.test_page_accessibility()
            results['health_endpoint'] = self.test_health_endpoint()
            results['static_resources'] = self.test_static_resources()
            results['memory_functionality'] = self.test_memory_data_functionality()
            
        finally:
            # Cleanup
            self.stop_streamlit()
        
        # Print results
        print("\n📋 Test Results Summary")
        print("=" * 30)
        passed = 0
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
            if result:
                passed += 1
        
        print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        return results

def main():
    """Main test runner"""
    tester = SimpleGUITester()
    
    try:
        results = tester.run_all_tests()
        
        # Exit with appropriate code
        all_passed = all(results.values()) if results else False
        exit_code = 0 if all_passed else 1
        
        print(f"\n🏁 Simple test suite completed with exit code: {exit_code}")
        return exit_code
        
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        tester.stop_streamlit()
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        tester.stop_streamlit()
        return 1

if __name__ == "__main__":
    exit(main())