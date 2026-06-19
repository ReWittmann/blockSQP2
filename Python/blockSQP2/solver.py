from ctypes import c_double, c_char_p, c_void_p, POINTER, cast
c_double_p = POINTER(c_double)

from .cxxwrappers import CXXobjWrapper, CXXobjHolder
from .problem import Problem
from .stats import Stats
from .options import Options
import numpy as np
from enum import Enum

class SQPresults(Enum):
    it_finished = 0
    partial_success = 1
    success = 2
    super_success = 3
    local_infeasibility = -1
    restoration_failure = -2
    linesearch_failure = -3
    qp_failure = -4
    eval_failure = -5
    misc_error = -10
    sensitivity_eval_failure = -100

class Solver(CXXobjWrapper):
    #Python side arguments
    Py_Problem : Problem
    Py_Opts : Options
    Py_Stats : Stats
    
    #Py_Problem may change, so save required data
    prob_nVar : int
    prob_nCon : int
    
    # C++ side objects
    Problemspec_hld : CXXobjHolder
    SQPoptions_hld : CXXobjHolder
    SQPstats_obj : c_void_p         # Owned by Py_Stats
    
    def __init__(self, arg_prob, arg_opts, arg_stats):
        self.Py_Problem = arg_prob
        self.Py_Opts = arg_opts
        self.Py_Stats = arg_stats
        
        self.prob_nVar = self.Py_Problem.nVar
        self.prob_nCon = self.Py_Problem.nCon
        
        BSQP = self.BSQP
        self.Problemspec_hld = self.Py_Problem.create_cxx_obj()
        self.SQPoptions_hld = self.Py_Opts.create_cxx_obj()        
        self.SQPstats_obj = self.Py_Stats.get_cxx_obj()
        
        self.cxx_obj = BSQP.create_SQPmethod(self.Problemspec_hld.cxx_obj, self.SQPoptions_hld.cxx_obj, self.SQPstats_obj)

        if not self.cxx_obj:
            error_message = BSQP.get_error_message()
            raise RuntimeError(cast(error_message, c_char_p).value.decode())
    
    def __del__(self):
        self.BSQP.delete_SQPmethod(self.cxx_obj)
    
    def init(self):
        self.BSQP.SQPmethod_init(self.cxx_obj)
    
    def run(self, maxIt: int, warmStart: int = 0):
        ret = self.BSQP.SQPmethod_run(self.cxx_obj, maxIt, warmStart)

        if ret == -1000:
            error_message = self.BSQP.get_error_message()
            raise RuntimeError(cast(error_message, c_char_p).value.decode())
        return SQPresults(ret)

    def finish(self):
        self.BSQP.SQPmethod_finish(self.cxx_obj)

    def get_itCount(self):
        return self.Py_Stats.ItCount

    def get_primal_solution(self):
        xi_arr = np.zeros(self.prob_nVar, dtype = c_double)
        self.BSQP.SQPmethod_get_xi(self.cxx_obj, xi_arr.ctypes.data_as(c_double_p))
        return xi_arr

    def get_dual_solution(self):
        lam_arr = np.zeros(self.prob_nVar + self.prob_nCon, dtype = c_double)
        self.BSQP.SQPmethod_get_lambda(self.cxx_obj, lam_arr.ctypes.data_as(c_double_p))
        return lam_arr[self.prob_nVar:]
    
    def get_dual_solution_full(self):
        lam_arr = np.zeros(self.prob_nVar + self.prob_nCon, dtype = c_double)
        self.BSQP.SQPmethod_get_lambda(self.cxx_obj, lam_arr.ctypes.data_as(c_double_p))
        return lam_arr
    
    def get_xi(self):
        return self.get_primal_solution()
    def get_lambda(self):
        return self.get_dual_solution_full()
    
    def get_hess1_block(self, ind):
        nBlocks = self.BSQP.Problemspec_get_nBlocks(self.Problemspec_hld.cxx_obj)
        blockIdx = np.ctypeslib.as_array(self.BSQP.Problemspec_get_blockIdx(self.Problemspec_hld.cxx_obj), shape = (nBlocks + 1,))
        n = blockIdx[ind + 1] - blockIdx[ind]
        hess1_block = np.zeros((n,n), dtype = c_double) #Note: This is row major, but treated as column major. Doesn't matter here due to symmetry.
        hess1_lower = self.BSQP.SQPmethod_get_hess1_block(self.cxx_obj, ind)
        self.BSQP.lower_to_full(hess1_block.ctypes.data_as(c_double_p), hess1_lower, n)
        return hess1_block
    
    def get_hess2_block(self, ind):
        nBlocks = self.BSQP.Problemspec_get_nBlocks(self.Problemspec_hld.cxx_obj)
        blockIdx = np.ctypeslib.as_array(self.BSQP.Problemspec_get_blockIdx(self.Problemspec_hld.cxx_obj), shape = (nBlocks + 1,))
        n = blockIdx[ind + 1] - blockIdx[ind]
        hess2_block = np.zeros((n,n), dtype = c_double) #Note: This is row major, but treated as column major. Doesn't matter here due to symmetry.
        hess2_lower = self.BSQP.SQPmethod_get_hess2_block(self.cxx_obj, ind)
        self.BSQP.lower_to_full(hess2_block.ctypes.data_as(c_double_p), hess2_lower, n)
        return hess2_block

    
class BoundCorrectionSolver(Solver):
    def __init__(self, arg_prob, arg_opts, arg_stats):
        self.Py_Problem = arg_prob
        self.Py_Opts = arg_opts
        self.Py_Stats = arg_stats
        
        self.prob_nVar = self.Py_Problem.nVar
        self.prob_nCon = self.Py_Problem.nCon
        
        BSQP = self.BSQP
        self.Problemspec_hld = self.Py_Problem.create_cxx_obj()
        self.SQPoptions_hld = self.Py_Opts.create_cxx_obj()        
        self.SQPstats_obj = self.Py_Stats.get_cxx_obj()
        
        self.cxx_obj = BSQP.create_BoundCorrectionSolver(self.Problemspec_hld.cxx_obj, self.SQPoptions_hld.cxx_obj, self.SQPstats_obj)
        if not self.cxx_obj:
            error_message = BSQP.get_error_message()
            raise RuntimeError(cast(error_message, c_char_p).value.decode())
    