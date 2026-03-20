
import sys
from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()
sys.path += [str(cD.parent)]

import blockSQP2
import numpy as np
import time
from ctypes import *


BSQP = blockSQP2.BSQP

BSQP.TEST2.restype = c_void_p
ptr = c_void_p(BSQP.TEST2())

BSQP.TEST3.argtypes = [c_void_p]


BSQP.CREATE_TESTCLS.restype = c_void_p
BSQP.CREATE_TESTCLS.argtypes = ()

BSQP.PRINT_TESTCLS.argtypes = (c_void_p,)

Cptr = BSQP.CREATE_TESTCLS()
BSQP.PRINT_TESTCLS(Cptr)

BSQP.DELETE_TESTCLS.restype = None
BSQP.DELETE_TESTCLS.argtypes = (c_void_p,)


BSQP.create_Problemspec.restype = c_void_p
BSQP.create_Problemspec.argtypes = (c_int, c_int)

BSQP.delete_Problemspec.argtypes = (c_void_p,)
BSQP.delete_Problemspec.restype = None

a = BSQP.create_Problemspec(3,3)
