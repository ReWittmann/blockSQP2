import os
import sys
if os.name == 'nt':
	exe_dir = os.path.dirname(sys.executable)
	dll_dir = os.path.join(exe_dir, f"python{sys.version_info.major}{sys.version_info.minor}.dll")
	os.add_dll_directory(dll_dir)
from .py_blockSQP import *
from .blockSQP_Problemspec import blockSQP_Problemspec as Problemspec