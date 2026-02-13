#!/usr/bin/env python3
"""Test script for file creation hook functionality."""

import sys
import json
import time

def create_test_file(filename: str, lines: int):
    """Create a test file with specified number of lines."""
    content = "\n".join([f"Line {i+1}: Test content for {filename}" for i in range(lines)])
    
    # This will trigger the hook
    with open(filename, 'w') as f:
        f.write(content)
    
    print(f"Created {filename} with {lines} lines")

def main():
    print("Testing file creation hook...")
    print("-" * 50)
    
    # Test 1: Create a small file (should not trigger)
    print("\nTest 1: Creating small file (50 lines)")
    create_test_file("test_small.txt", 50)
    time.sleep(1)
    
    # Test 2: Create a large file (should trigger threshold)
    print("\nTest 2: Creating large file (150 lines)")
    create_test_file("test_large.txt", 150)
    time.sleep(1)
    
    # Test 3: Create multiple small files
    print("\nTest 3: Creating multiple small files")
    for i in range(3):
        create_test_file(f"test_multi_{i}.txt", 30)
        time.sleep(0.5)
    
    # Test 4: Test pattern matching
    print("\nTest 4: Testing pattern exclusion")
    create_test_file("test_file.log", 200)  # Should be excluded if .log is in exclude patterns
    
    print("\n" + "-" * 50)
    print("Test completed!")
    print("\nNote: Check the hook logs to see if thresholds were triggered correctly.")

if __name__ == "__main__":
    main()