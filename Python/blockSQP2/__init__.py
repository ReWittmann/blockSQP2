import os
import sys
if os.name == 'nt':
    exe_dir = os.path.dirname(sys.executable)
    dll_dir = os.path.join(exe_dir, f"python{sys.version_info.major}{sys.version_info.minor}.dll")
    os.add_dll_directory(dll_dir)
try:
    from .pyblockSQP import *
except ImportError as IERR:
    if IERR.msg[:len("generic_type: type ")] == "generic_type: type ":
        raise ImportError(IERR.msg + "\n**Note**: The above error likely ocurred because a different version of py_blockSQP was previously loaded. This is due to how Python handles pybind11/boost::python modules. Start a new Python session.") from None
    else:
        raise IERR
from .Problemspec import Problemspec