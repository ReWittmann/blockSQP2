import numpy as np
import BlockSQP
import typing
import time
from typing import Callable

class BlockSQP_Problem(BlockSQP.Problemform):
    
    Sparse_QP = False
    jacIndRow : typing.Union[list, np.ndarray]
    jacIndCol : typing.Union[list, np.ndarray]
    x_start : np.array
    lam_start : np.array
    
    #Model bounds on inputs (variables) for functions and derivatives
    lb_input : np.array
    ub_input : np.array
    
    f : Callable[[np.array], float]
    g : Callable[[np.array], np.array]
    grad_f : Callable[[np.array], np.array]
    jac_g : Callable[[np.array], np.array]
    jac_gNz : Callable[[np.array], np.array]
    
    _continuity_restoration: Callable[[np.array], np.array]
    rest_cont : bool = False
    
    # def f(xi):
    #     return None
    # def g(xi):
    #     return None
    # def grad_f(xi):
    #     return None
    # def jac_g(xi):
    #     return None
    # def jac_gNz(xi):
    # 	return None
    # def close_continuity_gaps(xi):
    # 	return None
    
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
        
        self.Data.objval = self.f(self.Data.xi)
        self.Data.constr[:] = self.g(self.Data.xi)
        
        if self.Data.dmode > 0:
           self.Data.gradObj[:] = self.grad_f(self.Data.xi)
           self.Data.constrJac[:,:] = self.jac_g(self.Data.xi)
           
    
    def evaluate_sparse(self):
        xi_ = np.maximum(self.Data.xi, self.lb_input)
        xi_ = np.minimum(xi_, self.ub_input)
        
        np.savez('last_input', xi_)
        
        for j in range(len(xi_)):
            if np.isnan(self.Data.xi[j]):
                raise ValueError("Received nan")
        
        
        self.Data.objval = self.f(xi_)
        self.Data.constr[:] = self.g(xi_)
        
        if self.Data.dmode > 0:
            t0 = time.time()
            print("Evaluating constraint-jacobian\n")
            self.Data.gradObj[:] = self.grad_f(xi_)
            self.Data.jacNz[:] = self.jac_gNz(xi_)
            t1 = time.time()
            print("Evaluated constraint jacobian in ", t1 - t0, "seconds\n")
        
    def evaluate_simple(self):
        xi_ = np.maximum(self.Data.xi, self.lb_input)
        xi_ = np.minimum(xi_, self.ub_input)
        
        self.Data.objval = self.f(xi_)
        self.Data.constr[:] = self.g(xi_)
        
    def restore_continuity(self):
        if self.rest_cont:
            self.Data.xi[:] = self._continuity_restoration(self.Data.xi)
            self.Cpp_Data.info = 0
        else:
            self.Cpp_Data.info = 1
        print("Python side: Cpp_Data.info = ", self.Cpp_Data.info, "\n")
        return
    
    
    def set_bounds(self, lb_x, ub_x, lb_g, ub_g, objLo = -np.inf, objUp = np.inf):
        # lowbound = BlockSQP.Matrix(len(bl_x) + len(bl_g))
        # upbound = BlockSQP.Matrix(len(bu_x) + len(bu_g))
        # np.array(lowbound, copy = False)[:,0] = np.concatenate([bl_x, bl_g], axis = 0)
        # np.array(upbound, copy = False)[:,0] = np.concatenate([bu_x, bu_g], axis = 0)
        # self.bl_x = bl_x
        # self.bu_x = bu_x
        # self.bl = lowbound
        # self.bu = upbound
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
    
    @property
    def continuity_restoration(self):
        return self._continuity_restoration
    
    @continuity_restoration.setter
    def continuity_restoration(self, rest_func):
        self._continuity_restoration = rest_func
        self.rest_cont = True
        
        
        


