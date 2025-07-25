#!/usr/bin/env python3
import sys
import os

print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")

try:
    print("Attempting to import odes...")
    import odes
    print("Import successful!")
    print(f"Module type: {type(odes)}")
    print(f"Module attributes: {dir(odes)}")
    
    if hasattr(odes, 'MFCModel'):
        print("MFCModel found!")
        model = odes.MFCModel()
        print(f"Model created: {model}")
    else:
        print("MFCModel NOT found in module")
        
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Other error: {e}")
    