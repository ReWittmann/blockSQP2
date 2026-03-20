from ctypes import CDLL
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
        # print(f"Library {lib_name} loaded successfully.")
        return lib
    except OSError as e:
        raise OSError(f"Error loading library {lib_name}: {e}") from e

BSQP = load_library("pyblockSQP2")

from .function_signatures import add_BSQP_signatures
add_BSQP_signatures(BSQP)


from .solver import Solver, SQPresults
Solver.BSQP = BSQP

from .problem import Problem

from .options import qpOASESoptions, Options
qpOASESoptions.BSQP = BSQP
Options.BSQP = BSQP

from .condenser import vblock, cblock, condensing_target, Condenser
Condenser.BSQP = BSQP

from .stats import Stats
Stats.BSQP = BSQP

#Some backwards compatibility to original pybind11 based interface
Problemspec = Problem
qpOASES_options = qpOASESoptions
SQPoptions = Options
SQPstats = Stats
SQPmethod = Solver

def vblock_array(size):
    return [None]*size
def cblock_array(size):
    return [None]*size
def int_array(size):
    return [0]*size
def condensing_targets(size):
    return [None]*size