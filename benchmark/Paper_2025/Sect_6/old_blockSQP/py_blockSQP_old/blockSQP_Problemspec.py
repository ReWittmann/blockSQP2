# blockSQP_reference_build - build system and Python interface for blockSQP
# Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
# Licensed under the zlib license. See LICENSE for more details.

import numpy as np
from . import py_blockSQP
import typing

class BlockSQP_Problem(py_blockSQP.Problemform):
    
    Sparse_QP = False
    nnz = 0
    jacIndRow : typing.Union[list, np.ndarray]
    jacIndCol : typing.Union[list, np.ndarray]
    x_start : np.array
    lam_start : np.array
    bl_x : np.array
    bu_x : np.array
    
    def f(xi):
        return None
    def g(xi):
        return None
    def grad_f(xi):
        return None
    def jac_g(xi):
        return None
    def jac_g_nz(xi):
        	return None
    
    class Data:
        objval : float
        xi : np.array
        lam : np.array
        constr : np.array
        gradObj : np.array
        constrJac : np.array
        jacNz : np.array
        jacIndRow : np.array
        jacIndCol : np.array
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
        
        if self.Data.dmode > 0:
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
    
    def initialize_dense(self):
        self.Data.xi[:] = self.x_start
        self.Data.lam[:] = self.lam_start


    def initialize_sparse(self):
        self.Data.xi[:] = self.x_start
        self.Data.lam[:] = self.lam_start
        self.Data.jacIndRow[:] = self.jacIndRow 
        self.Data.jacIndCol[:] = self.jacIndCol
        
    def evaluate_dense(self):
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
        xi_ = np.maximum(self.Data.xi, self.bl_x)
        xi_ = np.minimum(xi_, self.bu_x)
        
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
        except Exception:
            self.Cpp_Data.info = 1
        else:
            self.Cpp_Data.info = 0
        
    def evaluate_simple(self):
        xi_ = np.maximum(self.Data.xi, self.bl_x)
        xi_ = np.minimum(xi_, self.bu_x)
        
        try:
            self.Data.objval = self.f(xi_)
            self.Data.constr[:] = self.g(xi_)
        except Exception:
            self.Cpp_Data.info = 1
        else:
            self.Cpp_Data.info = 0
    
    
    def set_bounds(self, bl_x, bu_x, bl_g, bu_g, objLo = -np.inf, objUp = np.inf):
        lowbound = py_blockSQP.Matrix(len(bl_x) + len(bl_g))
        upbound = py_blockSQP.Matrix(len(bu_x) + len(bu_g))
        np.array(lowbound, copy = False)[:,0] = np.concatenate([bl_x, bl_g], axis = 0)
        np.array(upbound, copy = False)[:,0] = np.concatenate([bu_x, bu_g], axis = 0)
        self.bl_x = bl_x
        self.bu_x = bu_x
        self.bl = lowbound
        self.bu = upbound
        self.objLo = objLo
        self.objUp = objUp
        
    def make_sparse(self, nnz : int, jacIndRow : typing.Union[list, np.ndarray], jacIndCol : typing.Union[list, np.ndarray]):
        self.Sparse_QP = True
        self.nnz = nnz
        assert len(jacIndRow) == nnz
        self.jacIndRow = jacIndRow
        self.jacIndCol = jacIndCol
    	
    
    def complete(self):
        self.init_Cpp_Data(self.Sparse_QP, self.nnz)
        
    def set_blockIndex(self, idx : np.array):
        assert isinstance(idx, np.ndarray)
        if idx.dtype != np.int32:
            raise Exception("block index array has wrong dtype! numpy.array(., dtype = np.int32) required")
        else:
            self.blockIdx = idx
        
        


