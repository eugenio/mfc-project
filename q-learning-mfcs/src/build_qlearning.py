#!/usr/bin/env python3
"""
Build script for Mojo Q-learning MFC controller.

This script compiles the Mojo Q-learning implementation and creates
Python bindings for easy integration.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            if result.stdout:
                print("Output:", result.stdout)
        else:
            print(f"✗ {description} failed")
            print("Error:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ {description} timed out")
        return False
    except Exception as e:
        print(f"✗ {description} failed with exception: {e}")
        return False
    
    return True

def main():
    """Main build process"""
    print("=== Mojo Q-Learning MFC Controller Build Script ===")
    
    # Check if we're in the correct directory
    if not os.path.exists("odes.mojo"):
        print("Error: odes.mojo not found. Please run this script from the correct directory.")
        sys.exit(1)
    
    # Build steps
    build_steps = [
        # Step 1: Build the basic MFC model
        {
            "cmd": "mojo build odes.mojo --emit='shared-lib' -o odes.so",
            "desc": "Building MFC model shared library"
        },
        
        # Step 2: Compile the Q-learning module (if possible)
        {
            "cmd": "mojo build mfc_qlearning.mojo -o mfc_qlearning",
            "desc": "Building Q-learning standalone executable"
        }
    ]
    
    print("Starting build process...")
    
    success_count = 0
    for i, step in enumerate(build_steps, 1):
        print(f"\nStep {i}/{len(build_steps)}: {step['desc']}")
        
        if run_command(step["cmd"], step["desc"]):
            success_count += 1
        else:
            print(f"Build step {i} failed, but continuing...")
    
    print("\n=== Build Summary ===")
    print(f"Successful steps: {success_count}/{len(build_steps)}")
    
    if success_count == len(build_steps):
        print("✓ All build steps completed successfully!")
        print("\nNext steps:")
        print("1. Run the Q-learning demo: python mfc_qlearning_demo.py")
        print("2. Test the MFC model: python mfc_model.py")
        
    else:
        print("⚠ Some build steps failed, but you can still run the demo")
        print("The demo will work with the Python fallback implementation")
    
    print("\nAvailable files:")
    for filename in ["odes.so", "mfc_qlearning", "mfc_qlearning_demo.py"]:
        if os.path.exists(filename):
            print(f"✓ {filename}")
        else:
            print(f"✗ {filename}")

if __name__ == "__main__":
    main()