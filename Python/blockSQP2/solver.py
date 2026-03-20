from ctypes import CDLL, c_void_p, c_int, c_double, c_char, c_char_p, POINTER, cast
from .problem import Problem
from .stats import Stats
from .options import Options#, create_cxx_options
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

class Solver:
    BSQP : CDLL
    
    #C++ side objects
    SQPmethod_obj : c_void_p = c_void_p(None)
    Problemspec_obj : c_void_p = c_void_p(None)
    SQPoptions_obj : c_void_p = c_void_p(None)
    QPsolver_options_obj : c_void_p = c_void_p(None)
    
    #Python side objects
    Py_Problem : Problem
    Py_Opts : Options
    Py_Stats : Stats
    
    def __init__(self, arg_prob, arg_opts, arg_stats):
        self.Py_Problem = arg_prob
        self.Py_Opts = arg_opts
        self.Py_Stats = arg_stats
        
        self.Problemspec_obj = self.BSQP.create_Problemspec(c_int(self.Py_Problem.nVar), c_int(self.Py_Problem.nCon))
        
        # Register callbacks
        self.BSQP.Problemspec_set_closure(self.Problemspec_obj, c_void_p(None))
        self.BSQP.Problemspec_set_dense_init(self.Problemspec_obj, self.Py_Problem.PTR_initialize_dense)
        self.BSQP.Problemspec_set_sparse_init(self.Problemspec_obj, self.Py_Problem.PTR_initialize_sparse)
        self.BSQP.Problemspec_set_dense_eval(self.Problemspec_obj, self.Py_Problem.PTR_evaluate_dense)
        self.BSQP.Problemspec_set_sparse_eval(self.Problemspec_obj, self.Py_Problem.PTR_evaluate_sparse)
        self.BSQP.Problemspec_set_simple_eval(self.Problemspec_obj, self.Py_Problem.PTR_evaluate_simple)
        self.BSQP.Problemspec_set_reduce_constr_vio(self.Problemspec_obj, self.Py_Problem.PTR_reduce_constr_vio)
        self.BSQP.Problemspec_set_modify_step(self.Problemspec_obj, self.Py_Problem.PTR_modify_step)
        
        if self.Py_Problem.blockIdx is not None:
            self.BSQP.Problemspec_set_blockIdx(
                self.Problemspec_obj,
                self.Py_Problem.blockIdx.ctypes.data_as(POINTER(c_int)),
                c_int(len(self.Py_Problem.blockIdx) - 1)
            )
        
        self.BSQP.Problemspec_set_nnz(
            self.Problemspec_obj,
            self.Py_Problem.nnz
        )

        self.BSQP.Problemspec_set_bounds(
            self.Problemspec_obj,
            self.Py_Problem.lb_var.ctypes.data_as(POINTER(c_double)),
            self.Py_Problem.ub_var.ctypes.data_as(POINTER(c_double)),
            self.Py_Problem.lb_con.ctypes.data_as(POINTER(c_double)),
            self.Py_Problem.ub_con.ctypes.data_as(POINTER(c_double)),
            c_double(self.Py_Problem.lb_obj),
            c_double(self.Py_Problem.ub_obj)
        )

        if len(self.Py_Problem.vblocks) > 0:
            vblock_array = self.BSQP.create_vblock_array(c_int(len(self.Py_Problem.vblocks)))

            for i, vb in enumerate(self.Py_Problem.vblocks):
                self.BSQP.vblock_array_set(
                    vblock_array,
                    c_int(i),
                    c_int(vb.size),
                    c_char(vb.dependent)
                )

            self.BSQP.Problemspec_pass_vblocks(
                self.Problemspec_obj,
                vblock_array,
                c_int(len(self.Py_Problem.vblocks))
            )

        if self.Py_Problem.condenser is not None:
            self.BSQP.Problemspec_set_cond(
                self.Problemspec_obj,
                self.Py_Problem.condenser.Condenser_obj
            )

        self.SQPoptions_obj, self.QPsolver_options_obj = self.Py_Opts.cxx_obj()
        # create_cxx_options(
        #     self.BSQP, self.Py_Opts
        # )

        self.SQPmethod_obj = self.BSQP.create_SQPmethod(
            self.Problemspec_obj,
            self.SQPoptions_obj,
            self.Py_Stats.SQPstats_obj
        )

        if not self.SQPmethod_obj:
            err = self.BSQP.get_error_message()
            raise RuntimeError(cast(err, c_char_p).value.decode())
    
    def finalize(self):
        pass
        # self.BSQP.delete_SQPmethod(self.SQPmethod_obj)
        # self.BSQP.delete_Problemspec(self.Problemspec_obj)
        # self.BSQP.delete_SQPoptions(self.SQPoptions_obj)
        # self.BSQP.delete_QPsolver_options(self.QPsolver_options_obj)
        
    def __del__(self):
        self.finalize()
    
    def init(self):
        self.BSQP.SQPmethod_init(self.SQPmethod_obj)

    def run(self, maxIt: int, warmStart: int = 0):
        ret = self.BSQP.SQPmethod_run(self.SQPmethod_obj, maxIt, warmStart)

        if ret == -1000:
            error_message = self.BSQP.get_error_message()
            raise Exception(error_message.value.decode('utf-8'))
        return SQPresults(ret)

    def finish(self):
        self.BSQP.SQPmethod_finish(self.SQPmethod_obj)

    def get_itCount(self):
        return self.BSQP.SQPstats_get_itCount(self.Py_Stats.SQPstats_obj)

    def get_primal_solution(self):
        xi_arr = np.zeros(self.Py_Problem.nVar, dtype = c_double)
        self.BSQP.SQPmethod_get_xi(self.SQPmethod_obj, xi_arr.ctypes.data_as(POINTER(c_double)))
        return xi_arr

    def get_dual_solution(self):
        lam_arr = np.zeros(self.Py_Problem.nVar + self.Py_Problem.nCon)
        self.BSQP.SQPmethod_get_lambda(self.SQPmethod_obj, lam_arr.ctypes.data_as(POINTER(c_double)))
        return lam_arr[self.Py_Problem.nVar:]
    
    def get_dual_solution_full(self):
        lam_arr = np.zeros(self.Py_Problem.nVar + self.Py_Problem.nCon)
        self.BSQP.SQPmethod_get_lambda(self.SQPmethod_obj, lam_arr.ctypes.data_as(POINTER(c_double)))
        return lam_arr
    
    def get_xi(self):
        return self.get_primal_solution()
    def get_lambda(self):
        return self.get_dual_solution_full()



    