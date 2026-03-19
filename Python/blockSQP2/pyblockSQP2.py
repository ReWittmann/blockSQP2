from ctypes import *

import os
import sys

def load_library(lib_name):
    if sys.platform.startswith('linux'):
        lib_path = f"lib{lib_name}.so"
    elif sys.platform == 'darwin':
        lib_path = f"lib{lib_name}.dylib"
    elif sys.platform == 'win32':
        lib_path = f"{lib_name}.dll"
    else:
        raise OSError(f"Unsupported platform: {sys.platform}")
    
    try:
        lib = CDLL(lib_path)
        print(f"Library {lib_name} loaded successfully.")
        return lib
    except OSError as e:
        raise OSError(f"Error loading library {lib_name}: {e}")

# Example Usage:
lib_name = "example"  # Replace with the name of your library (without extension)
lib = load_library(lib_name)