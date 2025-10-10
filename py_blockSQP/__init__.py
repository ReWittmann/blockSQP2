# py_blockSQP -- A python interface to the blockSQP nonlinear 
#                solver developed by Dennis Janka and extended by
#                Reinhold Wittmann
# Copyright (C) 2022-2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
#
# Licensed under the zlib license. See LICENSE for more details.


# \file py_blockSQP.cpp
# \author Reinhold Wittmann
# \date 2022-2025
#
# Implementation of a python interface to the blockSQP 
# nonlinear solver - Python module init file

import os
import sys
if os.name == 'nt':
	exe_dir = os.path.dirname(sys.executable)
	dll_dir = os.path.join(exe_dir, f"python{sys.version_info.major}{sys.version_info.minor}.dll")
	os.add_dll_directory(dll_dir)
from .py_blockSQP import *
from .blockSQP_Problemspec import blockSQP_Problemspec as Problemspec