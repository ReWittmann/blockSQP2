# py_blockSQP -- A python interface to blockSQP 2, a nonlinear programming
#                solver based on blockSQP by Dennis Janka.
# Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
#
# Licensed under the zlib license. See LICENSE for more details.


# \file blockSQP_Problemspec.py
# \author Reinhold Wittmann
# \date 2022-2025
#
# Implementation of a python interface to the blockSQP_2 
# nonlinear programming solver - Python side problem specification

import numpy as np
from . import py_blockSQP
import typing

class blockSQP_Problemspec(py_blockSQP.Problemform):
    #Primal start point for optimization
    x_start : np.ndarray[np.float64]
    #Dual start point for optimization
    lam_start : np.ndarray[np.float64]
    
    #Model bounds on inputs (variables) for functions and derivatives
    lb_input : np.ndarray[np.float64]
    ub_input : np.ndarray[np.float64]
    
    #Objective function
    f : typing.Callable[[np.ndarray[np.float64]], float]
    #Nonlinear constraints function
    g : typing.Callable[[np.ndarray[np.float64]], np.ndarray[np.float64]]
    #Objective gradient
    grad_f : typing.Callable[[np.ndarray[np.float64]], np.ndarray[np.float64]]
    #Constraint jacobian
    jac_g : typing.Callable[[np.ndarray[np.float64]], np.ndarray[np.float64]]
    #Nonzero elements of sparse constraint jacobian
    jac_g_nz : typing.Callable[[np.ndarray[np.float64]], np.ndarray[np.float64]]
    #Column-major, lower-triangular elements of last exact hessian block
    last_hessBlock : typing.Callable[[np.ndarray[np.float64]], np.ndarray[np.float64]]
    #List of column-major, lower-triangular elements of all exact hessian blocks
    hess : typing.Callable[[np.ndarray[np.float64]], typing.Iterable[np.ndarray[np.float64]]]
    
    
    #Flag for sparse mode (set by calling make_sparse)
    Sparse_QP = False
    #Row indices of sparse constraint jacobian in CCS-format (must be fixed, set by calling make_sparse)
    jacIndRow : typing.Union[list, np.ndarray[np.int32]]
    #Column indices of sparse constraint jacobian in CCS-format (must be fixed, set by calling make_sparse)
    jacIndCol : typing.Union[list, np.ndarray[np.int32]]
    
    #Integrate states in an attempt to reduce infeasibility. May prevent resorting to a restoration phase
    _continuity_restoration: typing.Callable[[np.ndarray[np.float64]], np.ndarray[np.float64]]
    
    def __init__(self, nVar = 0, nCon = 0):
        py_blockSQP.Problemform.__init__(self)
        self.nVar = nVar
        self.nCon = nCon
    
    ##Some setter methods##
    
    def set_bounds(self, lb_x, ub_x, lb_g, ub_g, objLo = -np.inf, objUp = np.inf):
        self.objLo = objLo
        self.objUp = objUp
        
        self.lb_var.Dimension(len(lb_x))
        self.ub_var.Dimension(len(ub_x))
        np.array(self.lb_var, copy = False)[:,0] = lb_x
        np.array(self.ub_var, copy = False)[:,0] = ub_x
        
        self.lb_con.Dimension(len(lb_g))
        self.ub_con.Dimension(len(ub_g))
        np.array(self.lb_con, copy = False)[:,0] = lb_g
        np.array(self.ub_con, copy = False)[:,0] = ub_g
        
        #Save variable bounds to ensure functions are evaluated
        #only with data inside the model bounds
        self.lb_input = lb_x
        self.ub_input = ub_x
    
    #nnz: number of nonzero entries in sparse constraint jacobian. Other arguments: see above.
    def make_sparse(self, nnz : int, jacIndRow : typing.Union[list, np.ndarray], jacIndCol : typing.Union[list, np.ndarray]):
        self.Sparse_QP = True
        self.nnz = nnz
        assert len(jacIndRow) == nnz
        self.jacIndRow = jacIndRow
        self.jacIndCol = jacIndCol
    
    #Set indices of hessian block starts and ends. idx[0] == 0 and idx[-1] == nVar and len(idx) = nBlocks + 1 must hold.  
    def set_blockIndex(self, idx : typing.Iterable):
        #assert isinstance(idx, np.ndarray)
        idx = np.array(idx, dtype = np.int32)
        self.blockIdx = idx
    
    #Setter for _continuity restoration, see above
    @property
    def continuity_restoration(self):
        return self._continuity_restoration
    
    @continuity_restoration.setter
    def continuity_restoration(self, rest_func):
        self._continuity_restoration = rest_func
        self.rest_cont = True

    ##IMPORTANT: Must be called before passing to SQPmethod##
    #Finalize the the problem specification
    def complete(self):
        assert(len(self.x_start) == self.nVar)
        self.init_Cpp_Data(self.Sparse_QP, self.nnz)
    
    
    #############################
    ##Internal data and methods##
    #############################
    
    rest_cont : bool = False
    class Data:
        objval : float
        xi : np.ndarray[np.float64]
        lam : np.ndarray[np.float64]
        constr : np.ndarray[np.float64]
        gradObj : np.ndarray[np.float64]
        constrJac : np.ndarray[np.float64]
        jacNz : np.ndarray[np.float64]
        jacIndRow : np.ndarray[np.int32]
        jacIndCol : np.ndarray[np.int32]
        dmode : int
        hess_arr : list[np.ndarray[np.float64]]
        hess_last : np.ndarray[np.float64]
        
    def get_objval(self):
        self.Cpp_Data.objval = self.Data.objval

    def update_inits(self):
        self.Data.xi = np.array(self.Cpp_Data.xi, copy = False)
        self.Data.xi.shape = (-1)
        self.Data.lam = np.array(self.Cpp_Data.lam, copy = False)
        self.Data.lam.shape = (-1)
        if not self.Sparse_QP:
            self.Data.constrJac = np.array(self.Cpp_Data.constrJac, copy = False)
        else:
            self.Data.jacNz = np.array(self.Cpp_Data.jacNz, copy = False)
            self.Data.jacIndRow = np.array(self.Cpp_Data.jacIndRow, copy = False)
            self.Data.jacIndCol = np.array(self.Cpp_Data.jacIndCol, copy = False)        
    
    def update_evals(self):
        self.Data.xi = np.array(self.Cpp_Data.xi, copy = False)
        self.Data.xi.shape = (-1)
        self.Data.lam = np.array(self.Cpp_Data.lam, copy = False)
        self.Data.lam.shape = (-1)
        self.Data.dmode = self.Cpp_Data.dmode
        self.Data.constr = np.array(self.Cpp_Data.constr, copy = False)
        self.Data.constr.shape = (-1)
        self.Data.gradObj = np.array(self.Cpp_Data.gradObj, copy = False)
        self.Data.gradObj.shape = (-1)
        if not self.Sparse_QP:
            self.Data.constrJac = np.array(self.Cpp_Data.constrJac, copy = False)
        else:
            self.Data.jacNz = np.array(self.Cpp_Data.jacNz, copy = False)
            self.Data.jacIndRow = np.array(self.Cpp_Data.jacIndRow, copy = False)
            self.Data.jacIndCol = np.array(self.Cpp_Data.jacIndCol, copy = False)
        
        if self.Data.dmode == 2:
            self.Data.hess_last = np.array(self.Cpp_Data.hess_arr[self.nBlocks - 1], copy = False)
            self.Data.hess_last.shape = (-1)
        elif self.Data.dmode == 3:
            self.Data.hess_arr = []
            for k in range(self.nBlocks):
                hk = np.array(self.Cpp_Data.hess_arr[k], copy = False)
                hk.shape = (-1)
                self.Data.hess_arr.append(hk)
    
    def update_simple(self):
        self.Data.xi = np.array(self.Cpp_Data.xi, copy = False)
        self.Data.xi.shape = (-1)
        self.Data.constr = np.array(self.Cpp_Data.constr, copy = False)
        self.Data.constr.shape = (-1)
    
    def update_xi(self):
        self.Data.xi = np.array(self.Cpp_Data.xi, copy = False)
        self.Data.xi.shape = (-1)
    
    def initialize_dense(self):
        self.Data.xi[:] = self.x_start
        self.Data.lam[:] = self.lam_start


    def initialize_sparse(self):
        self.Data.xi[:] = self.x_start
        self.Data.lam[:] = self.lam_start
        self.Data.jacIndRow[:] = self.jacIndRow 
        self.Data.jacIndCol[:] = self.jacIndCol
        
    def evaluate_dense(self):
        xi_ = np.maximum(self.Data.xi, self.lb_input)
        xi_ = np.minimum(xi_, self.ub_input)
        
        try:
            self.Data.objval = self.f(self.Data.xi)
            self.Data.constr[:] = self.g(self.Data.xi)
            
            if self.Data.dmode > 0:
                self.Data.gradObj[:] = self.grad_f(self.Data.xi)
                self.Data.constrJac[:,:] = self.jac_g(self.Data.xi)
            if self.Data.dmode == 2:
                self.Data.hess_last[:] = self.last_hessBlock(self.Data.xi, self.Data.lam[self.nVar:self.nVar + self.nCon])
            if self.Data.dmode == 3:
                hess_eval = self.hess(self.Data.xi, self.Data.lam[self.nVar:self.nVar + self.nCon])
                for j in range(self.nBlocks):
                    self.Data.hess_arr[j][:] = hess_eval[j]
        except Exception:
            self.Cpp_Data.info = 1
        else:
            self.Cpp_Data.info = 0
    
    def evaluate_sparse(self):
        xi_ = np.maximum(self.Data.xi, self.lb_input)
        xi_ = np.minimum(xi_, self.ub_input)
        
        try:
            self.Data.objval = self.f(xi_)
            self.Data.constr[:] = self.g(xi_)
            
            if self.Data.dmode > 0:
                self.Data.gradObj[:] = self.grad_f(xi_)
                self.Data.jacNz[:] = self.jac_g_nz(xi_)
            if self.Data.dmode == 2:
                self.Data.hess_last[:] = self.last_hessBlock(self.Data.xi, self.Data.lam[self.nVar : self.nVar + self.nCon])
            if self.Data.dmode == 3:
                hess_eval = self.hess(self.Data.xi, self.Data.lam[self.nVar : self.nVar + self.nCon])
                for j in range(self.nBlocks):
                    self.Data.hess_arr[j][:] = hess_eval[j]
        except:
            self.Cpp_Data.info = 1
        else:
            self.Cpp_Data.info = 0
    
    def evaluate_simple(self):
        xi_ = np.maximum(self.Data.xi, self.lb_input)
        xi_ = np.minimum(xi_, self.ub_input)
        
        try:
            self.Data.objval = self.f(xi_)
            self.Data.constr[:] = self.g(xi_)
        except Exception:
            self.Cpp_Data.info = 1
        else:
            self.Cpp_Data.info = 0
        
    def restore_continuity(self):
        if self.rest_cont:
            self.Data.xi[:] = self._continuity_restoration(self.Data.xi)
            self.Cpp_Data.info = 0
        else:
            self.Cpp_Data.info = 1
        return
    


