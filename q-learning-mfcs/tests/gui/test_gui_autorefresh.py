#!/usr/bin/env python3
"""Test script to verify GUI autorefresh functionality
"""

import os
import sys
import time

# Add src to path (go up two directories from tests/gui to reach src)
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
sys.path.insert(0, src_path)

# Import required modules
import pytest

try:
    from config.qlearning_config import DEFAULT_QLEARNING_CONFIG
    from mfc_streamlit_gui import SimulationRunner
except ImportError as e:
    pytest.skip(
        f"Unable to import mfc_streamlit_gui: {e}",
        allow_module_level=True,
    )

def test_gui_autorefresh():
    """Test that GUI autorefresh properly tracks simulation status"""
    print("🧪 Testing GUI Autorefresh Functionality")
    print("=" * 50)

    print("🧪 Running GUI autorefresh test")

    try:
        # Create runner
        runner = SimulationRunner()
        print("✅ SimulationRunner created")

        # Test initial state
        print(f"Initial is_running: {runner.is_running}")
        print(f"Initial is_running: {runner.is_running}")
        print(f"Initial thread: {runner.thread}")

        # Start a short simulation
        print("\n🚀 Starting simulation...")
        success = runner.start_simulation(
            config=DEFAULT_QLEARNING_CONFIG,
            duration_hours=0.005,  # ~18 seconds
            n_cells=3,
            electrode_area_m2=0.001,
            target_conc=25.0,
            gui_refresh_interval=2.0,
        )

        if success:
            print("✅ Simulation started successfully")
            print(f"After start - is_running: {runner.is_running}")
            print(f"After start - is_running: {runner.is_running}")
            print(f"After start - thread alive: {runner.thread and runner.thread.is_alive()}")

            # Monitor for a while
            for i in range(10):  # Monitor for 20 seconds
                time.sleep(2)

                status = runner.get_status()
                is_running = runner.is_running
                thread_alive = runner.thread and runner.thread.is_alive()

                print(f"Step {i+1}:")
                print(f"  Status: {status[0] if status else 'None'} - {status[1] if status else 'None'}")
                print(f"  is_running: {is_running}")
                print(f"  thread_alive: {thread_alive}")

                # If simulation finished, break
                if status and status[0] in ["completed", "stopped", "error"] and not thread_alive:
                    print("  ✅ Simulation properly finished")
                    break

                if not is_running and not thread_alive:
                    print("  ✅ Simulation properly stopped")
                    break

            # Stop if still running
            if runner.is_running:
                print("\n⏹️ Stopping simulation...")
                runner.stop_simulation()
                time.sleep(3)  # Wait for cleanup

            print("\nFinal state:")
            print(f"  is_running: {runner.is_running}")
            print(f"  is_running: {runner.is_running}")
            print(f"  thread_alive: {runner.thread and runner.thread.is_alive()}")

        else:
            print("❌ Failed to start simulation")

    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n✅ Test completed!")

if __name__ == "__main__":
    test_gui_autorefresh()
