# blockSQP2 -- A structure-exploiting nonlinear programming solver based
#              on blockSQP by Dennis Janka.
# Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>

# Licensed under the zlib license. See LICENSE for more details.


# \file Problemspec.py
# \author Reinhold Wittmann
# \date 2022-2025
#
# Python-side problem data structure of the python-interface to blockSQP2

import numpy as np
import typing
from ctypes import c_double, c_int, c_char, POINTER, c_void_p
from .function_signatures import BSQP_callback_signatures as CB_sigs
from .cxxwrappers import CXXobjCreator, CXXobjHolder
from .condenser import vblock, Condenser

c_double_p = POINTER(c_double)
c_int_p = POINTER(c_int)

class Problem(CXXobjCreator):
    nVar: int  # Number of variables
    nCon: int  # Number of constraints
    nnz: int   # Number of non-zeros
    sparse : bool
    
    _blockIdx: typing.Optional[np.ndarray[c_int]]
    vblocks: typing.Optional[typing.List[vblock]]
    condenser: typing.Optional[Condenser]
    
    lb_var: np.ndarray  # Lower bounds for variables
    ub_var: np.ndarray  # Upper bounds for variables
    lb_con: np.ndarray  # Lower bounds for constraints
    ub_con: np.ndarray  # Upper bounds for constraints
    lb_obj: c_double    # Lower bound for objective
    ub_obj: c_double    # Upper bound for objective
    
    f: typing.Optional[typing.Callable[[np.ndarray], c_double]]  # Objective function
    g: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]  # Constraints function
    grad_f: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]  # Gradient of objective
    jac_g: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]  # Jacobian of constraints
    last_hessBlock: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]  # Last Hessian block
    hess: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]  # Hessian
    _costrVioReducer: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]  # Feasibility restoration heuristic
    _stepModifier: typing.Callable[[np.ndarray[np.float64], np.ndarray[np.float64]], int]
    
    # Jacobian CCS sparsity structure and evaluation function
    jac_g_nz: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]
    jac_g_row: np.ndarray
    jac_g_colind: np.ndarray

    x_start : np.ndarray[np.float64]
    lam_start : np.ndarray[np.float64]
    
    def __init__(self, nVar = 0, nCon = 0):
        self.nVar = nVar
        self.nCon = nCon
        self.nnz = -1
        
        self._stepModifier = None
        self.sparse = False
        self.rest_cont = False
        self.blockIdx = None
        self.vblocks = []
        self.condenser = None
        
        self.C_initialize_dense = lambda scope, xi, lam, constrJac: self.initialize_dense(scope, xi, lam, constrJac)
        self.C_initialize_sparse = lambda scope, xi, lam, jac_nz, jac_row, jac_colind: self.initialize_sparse(scope, xi, lam, jac_nz, jac_row, jac_colind)
        self.C_evaluate_dense = lambda scope, xi, lam, objval, constr, gradObj, constrJac, hess, dmode, info: self.evaluate_dense(scope, xi, lam, objval, constr, gradObj, constrJac, hess, dmode, info)
        self.C_evaluate_sparse = lambda scope, xi, lam, objval, constr, gradObj, jac_nz, jac_row, jac_colind, hess, dmode, info: self.evaluate_sparse(scope, xi, lam, objval, constr, gradObj, jac_nz, jac_row, jac_colind, hess, dmode, info)
        self.C_evaluate_simple = lambda scope, xi, objval, constr, info: self.evaluate_simple(scope, xi, objval, constr, info)
        
        self.C_reduce_constr_vio = lambda scope, xi, info: self.call_constrVioReducer(scope, xi, info)
        self.C_modify_step = lambda scope, xi, lam, info: self.call_stepModifier(scope, xi, lam, info)
        
        self.PTR_initialize_dense = CB_sigs["initialize_dense"](self.C_initialize_dense)
        self.PTR_initialize_sparse = CB_sigs["initialize_sparse"](self.C_initialize_sparse)
        self.PTR_evaluate_dense = CB_sigs["evaluate_dense"](self.C_evaluate_dense)
        self.PTR_evaluate_sparse = CB_sigs["evaluate_sparse"](self.C_evaluate_sparse)
        self.PTR_evaluate_simple = CB_sigs["evaluate_simple"](self.C_evaluate_simple)
        
        self.PTR_reduce_constr_vio = CB_sigs["reduce_constr_vio"](self.C_reduce_constr_vio)
        self.PTR_modify_step = CB_sigs["modify_step"](self.C_modify_step)
        
    # Some setters
    def set_bounds(self, lb_x, ub_x, lb_g, ub_g, objLo = -np.inf, objUp = np.inf):
        self.lb_obj = objLo
        self.ub_obj = objUp
        
        self.lb_var = np.asarray(lb_x, dtype = c_double)
        self.ub_var = np.asarray(ub_x, dtype = c_double)
        
        self.lb_con = np.asarray(lb_g, dtype = c_double)
        self.ub_con = np.asarray(ub_g, dtype = c_double)
    
    def make_sparse(self, nnz : int, jacIndRow : typing.Union[list, np.ndarray], jacIndCol : typing.Union[list, np.ndarray]):
        self.sparse = True
        self.nnz = nnz
        assert len(jacIndRow) == nnz
        self.jac_g_row = jacIndRow
        self.jac_g_colind = jacIndCol
    
    def set_blockIndex(self, idx : typing.Iterable):
        self.blockIdx = idx
    
    @property
    def blockIdx(self):
        return self._blockIdx
    
    @blockIdx.setter
    def blockIdx(self, idx : typing.Optional[typing.Iterable]):
        self._blockIdx = np.array(idx, dtype = c_int) if idx is not None else None
    
    
    @property
    def costrVioReducer(self):
        return self._costrVioReducer
    
    @costrVioReducer.setter
    def costrVioReducer(self, rest_func):
        self._costrVioReducer = rest_func
        self.rest_cont = True
    
    @property
    def stepModifier(self):
        return self._stepModifier
    
    @stepModifier.setter
    def set_stepModifier(self, stepM_func : typing.Optional[typing.Callable[[np.ndarray[np.float64], np.ndarray[np.float64]], int]]):
        self._stepModifier = stepM_func
    
    
    def initialize_dense(self, _, xi : c_double_p, lam : c_double_p, constrJac : c_double_p):
        xi_arr = np.ctypeslib.as_array(xi, shape=(self.nVar,))
        lam_arr = np.ctypeslib.as_array(lam, shape=(self.nVar + self.nCon,))
        
        xi_arr[:] = self.x_start
        lam_arr[:] = self.lam_start

    def initialize_sparse(self, _, xi : c_double_p, lam : c_double_p, jac_nz : c_double_p, jac_row : c_int_p, jac_colind : c_int_p):
        xi_arr = np.ctypeslib.as_array(xi, shape=(self.nVar,))
        lam_arr = np.ctypeslib.as_array(lam, shape=(self.nVar + self.nCon,))
        jac_row_arr = np.ctypeslib.as_array(jac_row, shape=(self.nnz,))
        jac_colind_arr = np.ctypeslib.as_array(jac_colind, shape=(self.nVar + 1,))

        # Copy initial values from Problem to arrays
        xi_arr[:] = self.x_start
        lam_arr[:] = self.lam_start
        jac_row_arr[:] = self.jac_g_row
        jac_colind_arr[:] = self.jac_g_colind
    
    def evaluate_dense(self, _, xi: c_double_p, lam : c_double_p, objval : c_double_p, constr : c_double_p, gradObj : c_double_p, constrJac : c_double_p, hess : POINTER(c_double_p), dmode : c_int, info : c_int_p):
        try:
            xi_arr = np.ctypeslib.as_array(xi, shape=(self.nVar,))            
            lam_arr = np.ctypeslib.as_array(lam, shape=(self.nVar + self.nCon,))
            constr_arr = np.ctypeslib.as_array(constr, shape=(self.nCon,))
        
            objval[0] = self.f(xi_arr)
            constr_arr[:] = self.g(xi_arr)
            if dmode > 0:
                gradObj_arr = np.ctypeslib.as_array(gradObj, shape=(self.nVar,))
                constrJac_arr = np.ctypeslib.as_array(constrJac, shape=(self.nCon, self.nVar))
        
                gradObj_arr[:] = self.grad_f(xi_arr)
                constrJac_arr[:, :] = self.jac_g(xi_arr)
                if dmode == 2:
                    hess_arr = np.ctypeslib.as_array(hess, shape=(self.n_hessblocks,))
                    s = self.blockIdx[self.n_hessblocks] - self.blockIdx[self.n_hessblocks - 1]
                    hess_last = np.ctypeslib.as_array(hess_arr[self.n_hessblocks - 1], shape=(s * (s + 1) // 2,))
                    hess_last[:] = self.last_hessBlock(xi_arr, lam_arr[self.nVar: self.nVar + self.nCon])
        
                elif dmode == 3:
                    hess_list = []
                    for i in range(len(self.blockIdx) - 1):
                        Bsize = self.blockIdx[i+1] - self.blockIdx[i]
                        hess_list.append(np.ctypeslib.as_array(hess[i], shape = ((Bsize*(Bsize + 1))//2,)))
                    hess_eval = self.hess(xi_arr, lam_arr[self.nVar:self.nVar+self.nCon])
                    for i in range(len(self.blockIdx) - 1):
                        hess_list[i][:] = hess_eval[i]
        except Exception:
            info[0] = 1
        else:
            info[0] = 0
    
    def evaluate_sparse(self, _, xi : c_double_p, lam : c_double_p, objval : c_double_p, constr : c_double_p, gradObj : c_double_p, jac_nz : c_double_p, jac_row : c_int_p, jac_colind : c_int_p, hess : POINTER(c_double_p), dmode : c_int, info : c_int_p):
        try:
            xi_arr = np.ctypeslib.as_array(xi, shape=(self.nVar,))
            lam_arr = np.ctypeslib.as_array(lam, shape=(self.nVar + self.nCon,))
            constr_arr = np.ctypeslib.as_array(constr, shape=(self.nCon,))
            jac_nz_arr = np.ctypeslib.as_array(jac_nz, shape=(self.nnz,))
        
            objval[0] = self.f(xi_arr)
            constr_arr[:] = self.g(xi_arr)
        
            if dmode > 0:
                gradObj_arr = np.ctypeslib.as_array(gradObj, shape=(self.nVar,))
                gradObj_arr[:] = self.grad_f(xi_arr)
        
                jac_g_nz_eval = self.jac_g_nz(xi_arr)
                jac_nz_arr[:] = jac_g_nz_eval
        
                if dmode == 2:
                    hess_arr = np.ctypeslib.as_array(hess, shape=(self.n_hessblocks,))
                    s_last = self.blockIdx[self.n_hessblocks] - self.blockIdx[self.n_hessblocks - 1]
                    hess_last = np.ctypeslib.as_array(hess_arr[self.n_hessblocks - 1], shape=(s_last * (s_last + 1) // 2,))
                    hess_last[:] = self.last_hessBlock(xi_arr, lam_arr[self.nVar: self.nVar + self.nCon])
        
                elif dmode == 3:
                    hess_list = []
                    for i in range(len(self.blockIdx) - 1):
                        Bsize = self.blockIdx[i+1] - self.blockIdx[i]
                        hess_list.append(np.ctypeslib.as_array(hess[i], shape = ((Bsize*(Bsize + 1))//2,)))
                    hess_eval = self.hess(xi_arr, lam_arr[self.nVar:self.nVar+self.nCon])
                    for i in range(len(self.blockIdx) - 1):
                        hess_list[i][:] = hess_eval[i]
        except Exception:
            info[0] = 1
        else:
            info[0] = 0
    
    def evaluate_simple(self, _, xi : c_double_p, objval : c_double_p, constr : c_double_p, info : c_int_p):
        try:
            xi_arr = np.ctypeslib.as_array(xi, shape=(self.nVar,))
            constr_arr = np.ctypeslib.as_array(constr, shape=(self.nCon,))
            
            objval[0] = self.f(xi_arr)
            constr_arr[:] = self.g(xi_arr)
        except Exception:
            info[0] = 1
        else:
            info[0] = 0
        
    def call_constrVioReducer(self, _, xi : c_double_p, info : c_int_p):
        xi_arr = np.ctypeslib.as_array(xi, shape=(self.nVar,))
        if self.rest_cont:
            xi_arr[:] = self._costrVioReducer(xi_arr)
            info[0] = 0
            return
        info[0] = 1
    
    def call_stepModifier(self, _, xi : c_double_p, lam : c_double_p, info : c_int_p):
        xi_arr = np.ctypeslib.as_array(xi, shape=(self.nVar,))
        lam_arr = np.ctypeslib.as_array(lam, shape=(self.nVar + self.nCon,))
        if self._stepModifier is not None:
            info[0] = self._stepModifier(xi_arr, lam_arr)
            return
        info[0] = 1
    
    
    def create_cxx_obj(self):
        BSQP = self.BSQP
        
        cxx_obj = BSQP.create_Problemspec(self.nVar, self.nCon)
        self.CXX_Problem = cxx_obj
        
        BSQP.Problemspec_set_closure(cxx_obj, c_void_p(None))
        BSQP.Problemspec_set_dense_init(cxx_obj, self.PTR_initialize_dense)
        BSQP.Problemspec_set_sparse_init(cxx_obj, self.PTR_initialize_sparse)
        BSQP.Problemspec_set_dense_eval(cxx_obj, self.PTR_evaluate_dense)
        BSQP.Problemspec_set_sparse_eval(cxx_obj, self.PTR_evaluate_sparse)
        BSQP.Problemspec_set_simple_eval(cxx_obj, self.PTR_evaluate_simple)
        BSQP.Problemspec_set_reduce_constr_vio(cxx_obj, self.PTR_reduce_constr_vio)
        BSQP.Problemspec_set_modify_step(cxx_obj, self.PTR_modify_step)
        
        if self.blockIdx is not None:
            BSQP.Problemspec_set_blockIdx(cxx_obj, self.blockIdx.ctypes.data_as(POINTER(c_int)), c_int(len(self.blockIdx) - 1))
        BSQP.Problemspec_set_nnz(cxx_obj, self.nnz)
        
        BSQP.Problemspec_set_bounds(
            cxx_obj,
            self.lb_var.ctypes.data_as(POINTER(c_double)),
            self.ub_var.ctypes.data_as(POINTER(c_double)),
            self.lb_con.ctypes.data_as(POINTER(c_double)),
            self.ub_con.ctypes.data_as(POINTER(c_double)),
            c_double(self.lb_obj),
            c_double(self.ub_obj)
        )
        
        if self.vblocks is not None and len(self.vblocks) > 0:
            vblock_array = BSQP.create_vblock_array(c_int(len(self.vblocks)))
            for i, vb in enumerate(self.vblocks):
                BSQP.vblock_array_set(vblock_array, c_int(i), c_int(vb.size), c_char(vb.dependent), c_char(vb.bounds_implicit))
            BSQP.Problemspec_pass_vblocks(cxx_obj, vblock_array, c_int(len(self.vblocks)))
        
        if self.condenser is not None:
            BSQP.Problemspec_set_condenser(cxx_obj, self.condenser.get_cxx_obj())
        
        deleter = lambda ptr: self.BSQP.delete_Problemspec(ptr)
        return CXXobjHolder(cxx_obj, deleter)
    
    def complete(self):
        print("Calling *.complete is no longer required.")
    
class ScaledProblem(Problem):
    parent : Problem
    scaling_factors : np.ndarray[c_double]
    
    def __init__(self, parent):
        self.parent = parent
        self.scaling_factors = np.ones(parent.nVar, dtype = c_double)
        
        self.nVar = parent.nVar
        self.nCon = parent.nCon
        self.nnz = parent.nnz
        
    def set_scale(self, scaling_factors):
        self.scaling_factors = np.asarray(scaling_factors, dtype = c_double)
    def rescale(self, rescaling_factors):
        self.scaling_factors *= np.asarray(rescaling_factors, dtype = c_double)
    
    def create_cxx_obj(self):
        parent_hld = self.parent.create_cxx_obj()
        cxx_obj = self.BSQP.create_scaled_Problemspec(parent_hld.cxx_obj)
        self.BSQP.scaled_Problemspec_set_scale(cxx_obj, self.scaling_factors.ctypes.data_as(c_double_p))
        deleter = lambda ptr: self.BSQP.delete_Problemspec(ptr)
        return CXXobjHolder(cxx_obj, deleter, parent_hld)
    
    def unscale_variables(self, xi):
        ret = np.array(xi, copy = True)
        for i, s in enumerate(self.scaling_factors):
            ret[i] = xi[i]/s
        return ret
    
class TCfeasibilityProblem(Problem):
    parent : Problem
    def __init__(self, parent):
        self.parent = parent
        
        ntc = self.BSQP.Condenser_num_true_cons(parent.condenser.cxx_obj)
        self.nVar = parent.nVar + ntc
        self.nCon = parent.nCon
        self.nnz = parent.nnz + ntc
        
    def create_cxx_obj(self):
        parent_hld = self.parent.create_cxx_obj()
        cxx_obj = self.BSQP.create_TCfeasibilityProblem(parent_hld.cxx_obj)
        deleter = lambda ptr: self.BSQP.delete_Problemspec(ptr)
        return CXXobjHolder(cxx_obj, deleter, parent_hld)