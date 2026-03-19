# import os
# import sys
# if os.name == 'nt':
#     exe_dir = os.path.dirname(sys.executable)
#     dll_dir = os.path.join(exe_dir, f"python{sys.version_info.major}{sys.version_info.minor}.dll")
#     os.add_dll_directory(dll_dir)
# try:
#     from .pyblockSQP2 import *
# except ImportError as IERR:
#     if IERR.msg[:len("generic_type: type ")] == "generic_type: type ":
#         raise ImportError(IERR.msg + "\n**Note**: The above error likely ocurred because a different version of py_blockSQP was previously loaded. This is due to how Python handles pybind11/boost::python modules. Start a new Python session.") from None
#     else:
#         raise IERR
# from .Problemspec import Problemspec

from ctypes import *
from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()
import sys

def load_library(lib_name):
    if sys.platform.startswith('linux'):
        libname = Path(f"lib{lib_name}.so")
    elif sys.platform == 'darwin':
        libname = Path(f"lib{lib_name}.dylib")
    elif sys.platform == 'win32':
        libname = Path(f"{lib_name}.dll")
    else:
        raise OSError(f"Unsupported platform: {sys.platform}")
    
    libpath = str(cD/libname)
    try:
        lib = CDLL(libpath)
        print(f"Library {lib_name} loaded successfully.")
        return lib
    except OSError as e:
        raise OSError(f"Error loading library {lib_name}: {e}")

BSQP = load_library("pyblockSQP2")

from .function_signatures import add_function_signatures
add_function_signatures(BSQP)



from .solver import Solver
Solver.BSQP = BSQP
from .problem import Problem

