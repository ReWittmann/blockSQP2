from ctypes import *
from .problem import Problem
from .options import Options, qpOASESoptions, create_cxx_options

class Solver:
    BSQP : ctypes.CDLL
    
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
        
        self.Problemspec_obj = self.BSQP.create_Problemspec(self.Py_Problem.nVar, self.Py_Problem.nCon)
        
        # Register callbacks
        self.BSQP.Problemspec_set_closure(c_void_p(None))
        self.BSQP.Problemspec_set_dense_init(self.Problemspec_obj, Py_Problem.PTR_initialize_dense)
        self.BSQP.Problemspec_set_sparse_init(self.Problemspec_obj, Py_Problem.PTR_initialize_sparse)
        self.BSQP.Problemspec_set_sparse_eval(self.Problemspec_obj, Py_Problem.PTR_evaluate_sparse)
        
        self.BSQP.Problemspec_set_dense_eval(self.Problemspec_obj, Py_Problem.PTR_evaluate_dense)
        self.BSQP.Problemspec_set_simple_eval(self.Problemspec_obj, Py_Problem.PTR_avaluate_simple)
        self.BSQP.Problemspec_set_continuity_restoration(self.Problemspec_obj, Py_Problem.PTR_reduceConstrVio)
        
        self.BSQP.Problemspec_set_blockIdx(
            self.Problemspec_obj,
            Py_Problem.blockIdx.ctypes.data_as(POINTER(c_int)),
            c_int(len(Py_Problem.blockIdx) - 1)
        )

        self.BSQP.Problemspec_set_nnz(
            self.Problemspec_obj,
            c_int(Py_Problem.nnz)
        )

        self.BSQP.Problemspec_set_bounds(
            self.Problemspec_obj,
            Py_Problem.lb_var.ctypes.data_as(POINTER(c_double)),
            Py_Problem.ub_var.ctypes.data_as(POINTER(c_double)),
            Py_Problem.lb_con.ctypes.data_as(POINTER(c_double)),
            Py_Problem.ub_con.ctypes.data_as(POINTER(c_double)),
            c_double(Py_Problem.lb_obj),
            c_double(Py_Problem.ub_obj)
        )

        if len(Py_Problem.vblocks) > 0:
            vblock_array = self.BSQP.create_vblock_array(c_int(len(Py_Problem.vblocks)))

            for i, vb in enumerate(Py_Problem.vblocks):
                self.BSQP.vblock_array_set(
                    vblock_array,
                    c_int(i),
                    c_int(vb.size),
                    c_char(vb.dependent)
                )

            self.BSQP.Problemspec_pass_vblocks(
                self.Problemspec_obj,
                vblock_array,
                c_int(len(Py_Problem.vblocks))
            )

        if Py_Problem.condenser is not None:
            self.BSQP.Problemspec_set_cond(
                self.Problemspec_obj,
                Py_Problem.condenser.Condenser_obj
            )

        self.SQPoptions_obj, self.QPsolver_options_obj = create_cxx_options(
            self.BSQP, Py_Opts
        )

        self.SQPmethod_obj = self.BSQP.create_SQPmethod(
            self.Problemspec_obj,
            self.SQPoptions_obj,
            Py_Stats.SQPstats_obj
        )

        if not self.SQPmethod_obj:
            err = self.BSQP.get_error_message()
            raise RuntimeError(ctypes.cast(err, c_char_p).value.decode())
    
    def finalize(self):
        self.BSQP.delete_SQPmethod(self.SQPmethod_obj)
        self.BSQP.delete_Problemspec(self.Problemspec_obj)
        self.BSQP.delete_SQPoptions(self.SQPoptions_obj)
        self.BSQP.delete_SQPstats(self.SQPstats_obj)
    
    def __del__(self):
        self.finalize()
    
    