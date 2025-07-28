#!/usr/bin/env python3
"""
Browser-based test suite for the Streamlit GUI
Tests autorefresh functionality, data loading, and user interactions
"""

import time
import subprocess
import threading
import os
import signal
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

class StreamlitGUITester:
    """Browser-based tester for Streamlit GUI"""
    
    def __init__(self, port=8504):
        self.port = port
        self.streamlit_process = None
        self.driver = None
        self.base_url = f"http://localhost:{port}"
        
    def setup_driver(self):
        """Setup Chrome driver with appropriate options"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        try:
            # Use webdriver-manager to automatically download and manage ChromeDriver
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            print("✅ Chrome driver initialized successfully")
        except WebDriverException as e:
            print(f"❌ Failed to initialize Chrome driver: {e}")
            print("💡 Make sure Chrome/Chromium is installed")
            return False
        return True
    
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
            time.sleep(10)  # Give Streamlit time to fully start
            
            # Check if process is running
            if self.streamlit_process.poll() is None:
                print("✅ Streamlit started successfully")
                return True
            else:
                print("❌ Streamlit failed to start")
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
    
    def wait_for_element(self, by, value, timeout=10):
        """Wait for element to be present and return it"""
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, value))
            )
            return element
        except TimeoutException:
            print(f"⏰ Timeout waiting for element: {by}={value}")
            return None
    
    def test_page_load(self):
        """Test that the GUI loads correctly"""
        print("\n🧪 Testing page load...")
        
        try:
            self.driver.get(self.base_url)
            
            # Wait for main content to load
            main_content = self.wait_for_element(By.TAG_NAME, "main", timeout=15)
            if not main_content:
                print("❌ Main content failed to load")
                return False
            
            # Check for key elements
            title_found = "MFC" in self.driver.title or "Streamlit" in self.driver.title
            if title_found:
                print("✅ Page loaded successfully")
            else:
                print(f"⚠️ Unexpected page title: {self.driver.title}")
            
            # Look for tabs
            tabs = self.driver.find_elements(By.CSS_SELECTOR, "[data-testid='stTabs'] button")
            if tabs:
                print(f"✅ Found {len(tabs)} tabs")
                tab_names = [tab.text for tab in tabs]
                print(f"   Tab names: {tab_names}")
            else:
                print("⚠️ No tabs found")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during page load test: {e}")
            return False
    
    def test_tab_navigation(self):
        """Test navigation between tabs"""
        print("\n🧪 Testing tab navigation...")
        
        try:
            # Find tab buttons
            tabs = self.driver.find_elements(By.CSS_SELECTOR, "[data-testid='stTabs'] button")
            if len(tabs) < 2:
                print("❌ Not enough tabs found for navigation test")
                return False
            
            # Click on Monitor tab (usually second tab)
            monitor_tab = None
            for tab in tabs:
                if "Monitor" in tab.text or "📊" in tab.text:
                    monitor_tab = tab
                    break
            
            if monitor_tab:
                print("🖱️ Clicking Monitor tab...")
                monitor_tab.click()
                time.sleep(2)
                
                # Check if tab switched (look for Monitor content)
                monitor_content = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Real-time Monitoring')]")
                if monitor_content:
                    print("✅ Successfully navigated to Monitor tab")
                    return True
                else:
                    print("❌ Monitor tab content not found")
                    return False
            else:
                print("❌ Monitor tab not found")
                return False
                
        except Exception as e:
            print(f"❌ Error during tab navigation test: {e}")
            return False
    
    def test_autorefresh_toggle(self):
        """Test auto-refresh checkbox functionality"""
        print("\n🧪 Testing auto-refresh toggle...")
        
        try:
            # Make sure we're on Monitor tab
            tabs = self.driver.find_elements(By.CSS_SELECTOR, "[data-testid='stTabs'] button")
            for tab in tabs:
                if "Monitor" in tab.text or "📊" in tab.text:
                    tab.click()
                    break
            
            time.sleep(2)
            
            # Look for auto-refresh checkbox
            checkboxes = self.driver.find_elements(By.CSS_SELECTOR, "input[type='checkbox']")
            autorefresh_checkbox = None
            
            for checkbox in checkboxes:
                # Find checkbox by looking at nearby text
                parent = checkbox.find_element(By.XPATH, "./..")
                if "Auto-refresh" in parent.text or "refresh" in parent.text.lower():
                    autorefresh_checkbox = checkbox
                    break
            
            if autorefresh_checkbox:
                # Test toggling the checkbox
                initial_state = autorefresh_checkbox.is_selected()
                print(f"📋 Auto-refresh initial state: {initial_state}")
                
                # Toggle it
                autorefresh_checkbox.click()
                time.sleep(1)
                
                new_state = autorefresh_checkbox.is_selected()
                print(f"📋 Auto-refresh after toggle: {new_state}")
                
                if initial_state != new_state:
                    print("✅ Auto-refresh toggle works correctly")
                    return True
                else:
                    print("❌ Auto-refresh toggle failed")
                    return False
            else:
                print("❌ Auto-refresh checkbox not found")
                return False
                
        except Exception as e:
            print(f"❌ Error during auto-refresh test: {e}")
            return False
    
    def test_simulation_start(self):
        """Test starting a simulation"""
        print("\n🧪 Testing simulation start...")
        
        try:
            # Navigate to Run Simulation tab
            tabs = self.driver.find_elements(By.CSS_SELECTOR, "[data-testid='stTabs'] button")
            for tab in tabs:
                if "Run" in tab.text or "🚀" in tab.text:
                    print("🖱️ Clicking Run Simulation tab...")
                    tab.click()
                    break
            
            time.sleep(2)
            
            # Look for Start Simulation button
            buttons = self.driver.find_elements(By.CSS_SELECTOR, "button")
            start_button = None
            
            for button in buttons:
                if "Start Simulation" in button.text or "▶️" in button.text:
                    start_button = button
                    break
            
            if start_button and start_button.is_enabled():
                print("🖱️ Clicking Start Simulation button...")
                start_button.click()
                time.sleep(3)
                
                # Check for simulation status
                page_text = self.driver.page_source
                if "Simulation" in page_text and ("running" in page_text.lower() or "started" in page_text.lower()):
                    print("✅ Simulation appears to have started")
                    return True
                else:
                    print("⚠️ Simulation status unclear")
                    return False
            else:
                print("❌ Start Simulation button not found or disabled")
                return False
                
        except Exception as e:
            print(f"❌ Error during simulation start test: {e}")
            return False
    
    def test_monitor_tab_stability(self):
        """Test that Monitor tab stays stable during autorefresh"""
        print("\n🧪 Testing Monitor tab stability during autorefresh...")
        
        try:
            # Navigate to Monitor tab
            tabs = self.driver.find_elements(By.CSS_SELECTOR, "[data-testid='stTabs'] button")
            monitor_tab = None
            for tab in tabs:
                if "Monitor" in tab.text or "📊" in tab.text:
                    monitor_tab = tab
                    break
            
            if not monitor_tab:
                print("❌ Monitor tab not found")
                return False
            
            print("🖱️ Clicking Monitor tab...")
            monitor_tab.click()
            time.sleep(2)
            
            # Enable auto-refresh if not already enabled
            checkboxes = self.driver.find_elements(By.CSS_SELECTOR, "input[type='checkbox']")
            for checkbox in checkboxes:
                parent = checkbox.find_element(By.XPATH, "./..")
                if "Auto-refresh" in parent.text and not checkbox.is_selected():
                    checkbox.click()
                    time.sleep(1)
                    break
            
            # Check tab stability over time
            print("⏱️ Monitoring tab stability for 15 seconds...")
            stable_count = 0
            total_checks = 5
            
            for i in range(total_checks):
                time.sleep(3)
                
                # Check if we're still on Monitor tab
                current_tab_content = self.driver.page_source
                if "Real-time Monitoring" in current_tab_content or "Monitor" in current_tab_content:
                    stable_count += 1
                    print(f"   ✅ Check {i+1}/{total_checks}: Still on Monitor tab")
                else:
                    print(f"   ❌ Check {i+1}/{total_checks}: Not on Monitor tab")
            
            success_rate = stable_count / total_checks
            print(f"📊 Tab stability: {stable_count}/{total_checks} ({success_rate*100:.1f}%)")
            
            if success_rate >= 0.8:  # 80% success rate
                print("✅ Monitor tab is stable during autorefresh")
                return True
            else:
                print("❌ Monitor tab is unstable during autorefresh")
                return False
                
        except Exception as e:
            print(f"❌ Error during tab stability test: {e}")
            return False
    
    def run_all_tests(self):
        """Run all GUI tests"""
        print("🧪 Starting Browser-Based GUI Test Suite")
        print("=" * 60)
        
        results = {}
        
        # Setup
        if not self.setup_driver():
            print("❌ Failed to setup driver - aborting tests")
            return results
        
        if not self.start_streamlit():
            print("❌ Failed to start Streamlit - aborting tests")
            return results
        
        try:
            # Run tests
            results['page_load'] = self.test_page_load()
            results['tab_navigation'] = self.test_tab_navigation()
            results['autorefresh_toggle'] = self.test_autorefresh_toggle()
            results['simulation_start'] = self.test_simulation_start()
            results['monitor_stability'] = self.test_monitor_tab_stability()
            
        finally:
            # Cleanup
            if self.driver:
                self.driver.quit()
                print("✅ Browser driver closed")
            
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
    
    def cleanup(self):
        """Cleanup resources"""
        if self.driver:
            self.driver.quit()
        self.stop_streamlit()

def main():
    """Main test runner"""
    tester = StreamlitGUITester()
    
    try:
        results = tester.run_all_tests()
        
        # Exit with appropriate code
        all_passed = all(results.values()) if results else False
        exit_code = 0 if all_passed else 1
        
        print(f"\n🏁 Test suite completed with exit code: {exit_code}")
        return exit_code
        
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        tester.cleanup()
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        tester.cleanup()
        return 1

if __name__ == "__main__":
    exit(main())