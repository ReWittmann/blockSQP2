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
from ctypes import c_double, c_int, CFUNCTYPE, POINTER, c_void_p
from .function_signatures import callbacktype


c_double_p = POINTER(c_double)
c_int_p = POINTER(c_int)

callback_signatures = {
        'initialize_dense': CFUNCTYPE(None, c_void_p, c_double_p, c_double_p, c_double_p),
        'initialize_sparse': CFUNCTYPE(None, c_void_p,
                                c_double_p, c_double_p,
                                c_double_p, c_int_p, c_int_p),
        'evaluate_dense': CFUNCTYPE(None, c_void_p,
                               c_double_p, c_double_p,
                               c_double_p, c_double_p,
                               c_double_p, c_double_p,
                               POINTER(c_double_p), c_int, c_int_p),
        'evaluate_sparse': CFUNCTYPE(None, c_void_p,
                                c_double_p, c_double_p,
                                c_double_p, c_int_p),
        'evaluate_simple': CFUNCTYPE(None, c_void_p,
                                c_double_p, c_double_p,
                                c_double_p, c_double_p,
                                c_double_p, c_double_p,
                                c_int_p, c_int_p,
                                POINTER(c_double_p), c_int, c_int_p),
        'restore_continuity': CFUNCTYPE(None, c_void_p, c_double_p, c_int_p)
    }

class Problem:
    nVar: int  # Number of variables
    nCon: int  # Number of constraints
    nnz: int   # Number of non-zeros
    blockIdx: np.ndarray  # Block indices (array of int32)
    vblocks: typing.List['vblock']  # List of vblock objects (you should define the 'vblock' class)
    condenser: typing.Optional['Condenser']  # Optional condenser object

    # Additional fields from Julia struct
    lb_var: np.ndarray  # Lower bounds for variables
    ub_var: np.ndarray  # Upper bounds for variables
    lb_con: np.ndarray  # Lower bounds for constraints
    ub_con: np.ndarray  # Upper bounds for constraints
    lb_obj: c_double    # Lower bound for objective
    ub_obj: c_double    # Upper bound for objective

    # Function references (initialized as None here, but they will be set later in usage)
    f: typing.Optional[typing.Callable[[np.ndarray], c_double]]  # Objective function
    g: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]  # Constraints function
    grad_f: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]  # Gradient of objective
    jac_g: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]  # Jacobian of constraints
    last_hessBlock: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]  # Last Hessian block
    hess: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]  # Hessian
    _costrVioReducer: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]  # Continuity restoration
    _stepModifier: typing.Callable[[np.ndarray[np.float64], np.ndarray[np.float64]], int]
    
    # Jacobian sparsity structure
    jac_g_nz: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]
    jac_g_row: np.ndarray       # Row indices for Jacobian non-zeros
    jac_g_colind: np.ndarray    # Column indices for Jacobian non-zeros

    #Primal start point for optimization
    x_start : np.ndarray[np.float64]
    #Dual start point for optimization
    lam_start : np.ndarray[np.float64]
    
    #Model bounds on inputs (variables) for functions and derivatives
    # lb_input : np.ndarray[np.float64]
    # ub_input : np.ndarray[np.float64]
    
    sparse : bool
    
    
    def __init__(self, nVar = 0, nCon = 0):
        self.nVar = nVar
        self.nCon = nCon
        
        self._stepModifier = None
        self.sparse = False
        self.rest_cont = False
        
        
        self.C_initialize_dense = lambda *args: self.initialize_dense(*args)
        self.C_initialize_sparse = lambda *args: self.initialize_sparse(*args)
        self.C_evaluate_dense = lambda *args: self.evaluate_dense(*args)
        self.C_evaluate_sparse = lambda *args: self.evaluate_sparse(*args)
        self.C_evaluate_simple = lambda *args: self.evaluate_simple(*args)
        
        self.C_restore_continuity = lambda *args: self.restore_continuity(*args)
        self.C_modify_step = lambda *args: self.modify_step(*args)
        
        
        self.PTR_initialize_dense = callbacktype("initialize_dense")(self._initialize_dense)
        self.PTR_initialize_sparse = callbacktype("initialize_sparse")(self._initialize_sparse)
        self.PTR_evaluate_dense = callbacktype("evaluate_dense")(self.evaluate_dense)
        self.PTR_evaluate_sparse = callbacktype("evaluate_sparse")(self.evaluate_sparse)
        self.PTR_evaluate_simple = callbacktype("evaluate_simple")(self.evaluate_simple)
        
        self.PTR_reduceConstrVio = callbacktype("reduceConstrVio")(self.C_restore_continuity)
        self.PTR_modify_step = callbacktype("modify_step")(self.C_modify_step)
        
    ##Some setter methods##
    def set_bounds(self, lb_x, ub_x, lb_g, ub_g, objLo = -np.inf, objUp = np.inf):
        self.objLo = objLo
        self.objUp = objUp
        
        self.lb_var = lb_x
        self.ub_Var = ub_x
        
        self.lb_con = lb_g
        self.ub_con = ub_g
    
    def make_sparse(self, nnz : int, jacIndRow : typing.Union[list, np.ndarray], jacIndCol : typing.Union[list, np.ndarray]):
        self.sparse = True
        self.nnz = nnz
        assert len(jacIndRow) == nnz
        self.jacIndRow = jacIndRow
        self.jacIndCol = jacIndCol
    
    def set_blockIndex(self, idx : typing.Iterable):
        idx = np.array(idx, dtype = np.int32)
        self.blockIdx = idx
    
    
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
    
    def initialize_dense(self,_, xi : c_double_p, lam : c_double_p, constrJac : c_double_p):
        xi_arr = np.ctypeslib.as_array(xi, shape=(self.nVar,))
        lam_arr = np.ctypeslib.as_array(lam, shape=(self.nVar + self.nCon,))
        
        xi_arr[:] = self.x_start
        lam_arr[:] = self.lam_start

    def initialize_sparse(self, _, xi : c_double_p, lam : c_double_p, jac_nz : c_double_p, jac_row : c_int_p, jac_colind : c_int_p, info : c_int_p):
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
        info[0] = 0
    
    def evaluate_sparse(self, _, xi : c_double_p, lam : c_double_p, objval : c_double_p, constr : c_double_p, gradObj : c_double_p, jac_nz : c_double_p, jac_row : c_int_p, jac_colind : c_int_p, hess : POINTER(c_double_p), dmode : c_int, info : c_int_p):
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
        info[0] = 0
    
    def evaluate_simple(self, _, xi : c_double_p, objval : c_double_p, constr : c_double_p, info : c_int_p):
        xi_arr = np.ctypeslib.as_array(xi, shape=(self.nVar,))
        constr_arr = np.ctypeslib.as_array(constr, shape=(self.nCon,))
        
        objval[0] = self.f(xi_arr)
        constr_arr[:] = self.g(xi_arr)
        
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



