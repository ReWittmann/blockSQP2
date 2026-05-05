from ctypes import c_void_p, c_char, c_int, c_double, POINTER, cast
c_double_p = POINTER(c_double)
c_char_p = POINTER(c_char)

import typing
import numpy as np
from .cxxwrappers import CXXobjWrapper

class vblock:
    size: int
    dependent : bool
    def __init__(self, arg_size, arg_dep):
        self.size = arg_size
        self.dependent = arg_dep

class cblock:
    size: int
    def __init__(self, arg_size):
        self.size = arg_size

class condensing_target:
    n_stages: int
    first_free: int
    vblock_end: int
    first_cond: int
    cblock_end: int
    def __init__(self, arg_n_stages, arg_first_free, arg_vblock_end, arg_first_cond, arg_cblock_end):
        self.n_stages = arg_n_stages
        self.first_free = arg_first_free
        self.vblock_end = arg_vblock_end
        self.first_cond = arg_first_cond
        self.cblock_end = arg_cblock_end

#TODO make CXXobjWrapper
class Sparse_Matrix:
    m: int
    n: int
    nz: np.ndarray
    row: np.ndarray
    colind: np.ndarray

    def __init__(self, m : int = 0, n : int = 0, nz = None, row = None, colind = None):
        self.m = int(m)
        self.n = int(n)
        
        self.nz = np.array([], dtype = np.float64) if nz is None else np.asarray(nz, dtype = np.float64)
        self.row = np.array([], dtype = np.int32) if row is None else np.asarray(row, dtype = np.int32)
        self.colind = np.array([], dtype = np.int32) if colind is None else np.asarray(colind, dtype = np.int32)
    def nnz(self):
        return self.colind[self.n]
def Sparse_Matrix_structure(m : int, n : int, nnz : int):
    return Sparse_Matrix(m, n, np.zeros(nnz, dtype = np.double), np.zeros(nnz, dtype = np.int32), np.zeros(n+1, dtype = np.int32))


def full_to_lower(arr1: np.ndarray, arr2: np.ndarray, n: int):
    for i in range(n):
        for j in range(i):
            arr1[i + j * n - (j * (j + 1)) // 2] = arr2[i + j * n]

def lower_to_full(arr1: np.ndarray, arr2: np.ndarray, n: int):
    for i in range(n):
        for j in range(i):
            arr1[ i + j * n] = arr2[i + j * n - (j * (j + 1)) // 2]
    for i in range(n):
        for j in range(i, n):
            arr1[i + j * n] = arr2[(j + (i * n)) - (i * (i + 1)) // 2]

def as_array(ptr, length, dtype):
    return np.ctypeslib.as_array(cast(ptr, POINTER(dtype)), shape=(length,))


class Condenser(CXXobjWrapper):
    # Condenser constructor data
    vblock_array_obj: c_void_p
    cblock_array_obj: c_void_p
    hsize_array_obj: c_void_p
    target_array_obj: c_void_p
    
    # Condensing input
    Matrix_grad_obj: c_void_p
    Sparse_Matrix_constr_jac: c_void_p
    SymMatrix_array_hess: c_void_p
    Matrix_lb_var: c_void_p
    Matrix_ub_var: c_void_p
    Matrix_lb_con: c_void_p
    Matrix_ub_con: c_void_p
    
    # Condensing output
    Matrix_condensed_grad_obj: c_void_p
    Sparse_Matrix_condensed_constr_jac: c_void_p
    SymMatrix_array_condensed_hess: c_void_p
    Matrix_condensed_lb_var: c_void_p
    Matrix_condensed_ub_var: c_void_p
    Matrix_condensed_lb_con: c_void_p
    Matrix_condensed_ub_con: c_void_p
    
    # Condensed QP solution
    Matrix_xi_cond: c_void_p
    Matrix_lambda_cond: c_void_p
    
    # Restored full QP solution
    Matrix_xi_rest: c_void_p
    Matrix_lambda_rest: c_void_p
    
    def __init__(self, vblocks : typing.List['vblock'], cblocks : typing.List['cblock'], hsizes : typing.List['int'], targets : typing.List['condensing_target'], dep_bounds : int = 2):
        BSQP = self.BSQP
        self.vblock_array_obj = BSQP.create_vblock_array((len(vblocks)))
        for i, vb in enumerate(vblocks):
            BSQP.vblock_array_set(self.vblock_array_obj, i, vb.size, c_char(vb.dependent))
        
        self.cblock_array_obj = BSQP.create_cblock_array(len(cblocks))
        for i, cb in enumerate(cblocks):
            BSQP.cblock_array_set(self.cblock_array_obj, i, cb.size)
        
        self.hsize_array_obj = BSQP.create_hsize_array(len(hsizes))
        for i, hs in enumerate(hsizes):
            BSQP.hsize_array_set(self.hsize_array_obj, i, hs)
        
        self.target_array_obj = BSQP.create_target_array(len(targets))
        for i, t in enumerate(targets):
            BSQP.target_array_set(self.target_array_obj, i, t.n_stages, t.first_free, t.vblock_end, t.first_cond, t.cblock_end)
        
        self.cxx_obj = BSQP.create_Condenser(self.vblock_array_obj, len(vblocks), self.cblock_array_obj, len(cblocks), self.hsize_array_obj, len(hsizes), self.target_array_obj, len(targets), dep_bounds)
        if not self.cxx_obj:
            err = BSQP.get_error_message()
            raise RuntimeError(cast(err, c_char_p).value.decode())
        
        nVar = BSQP.Condenser_nVar(self.cxx_obj)
        nCon = BSQP.Condenser_nCon(self.cxx_obj)
        nBlocks = BSQP.Condenser_nBlocks(self.cxx_obj)
        
        self.Matrix_grad_obj = BSQP.create_Matrix(nVar, 1)
        self.Sparse_Matrix_constr_jac = BSQP.create_Sparse_Matrix_default()
        self.SymMatrix_array_hess = BSQP.create_SymMatrix_array(nBlocks)
        
        for i in range(nBlocks):
            BSQP.SymMatrix_array_index_resize(self.SymMatrix_array_hess, i, hsizes[i])
        
        self.Matrix_lb_var = BSQP.create_Matrix(nVar, 1)
        self.Matrix_ub_var = BSQP.create_Matrix(nVar, 1)
        self.Matrix_lb_con = BSQP.create_Matrix(nCon, 1)
        self.Matrix_ub_con = BSQP.create_Matrix(nCon, 1)
        
        
        condensed_nVar = BSQP.Condenser_condensed_nVar(self.cxx_obj)
        condensed_nCon = BSQP.Condenser_condensed_nCon(self.cxx_obj)
        condensed_nBlocks = BSQP.Condenser_condensed_nBlocks(self.cxx_obj)
        
        self.Matrix_condensed_grad_obj = BSQP.create_Matrix(condensed_nVar, 1)
        self.Sparse_Matrix_condensed_constr_jac = BSQP.create_Sparse_Matrix_default()
        self.SymMatrix_array_condensed_hess = BSQP.create_SymMatrix_array(condensed_nBlocks)
        
        self.Matrix_condensed_lb_var = BSQP.create_Matrix_default()
        self.Matrix_condensed_ub_var = BSQP.create_Matrix_default()
        self.Matrix_condensed_lb_con = BSQP.create_Matrix_default()
        self.Matrix_condensed_ub_con = BSQP.create_Matrix_default()
        
        # self.Matrix_condensed_lb_var = BSQP.create_Matrix(condensed_nVar, 1)
        # self.Matrix_condensed_ub_var = BSQP.create_Matrix(condensed_nVar, 1)
        # self.Matrix_condensed_lb_con = BSQP.create_Matrix(condensed_nCon, 1)
        # self.Matrix_condensed_ub_con = BSQP.create_Matrix(condensed_nCon, 1)
        
        self.Matrix_xi_cond = BSQP.create_Matrix(condensed_nVar, 1)
        self.Matrix_lambda_cond = BSQP.create_Matrix(condensed_nVar + condensed_nCon, 1)
        
        self.Matrix_xi_rest = BSQP.create_Matrix(nVar, 1)
        self.Matrix_lambda_rest = BSQP.create_Matrix(nVar + nCon, 1)

    def __del__(self):
        BSQP = self.BSQP
        BSQP.delete_Matrix(self.Matrix_lambda_rest)
        BSQP.delete_Matrix(self.Matrix_xi_rest)
        BSQP.delete_Matrix(self.Matrix_lambda_cond)
        BSQP.delete_Matrix(self.Matrix_xi_cond)

        BSQP.delete_Matrix(self.Matrix_condensed_ub_con)
        BSQP.delete_Matrix(self.Matrix_condensed_lb_con)
        BSQP.delete_Matrix(self.Matrix_condensed_ub_var)
        BSQP.delete_Matrix(self.Matrix_condensed_lb_var)
        BSQP.delete_SymMatrix_array(self.SymMatrix_array_condensed_hess)
        BSQP.delete_Sparse_Matrix(self.Sparse_Matrix_condensed_constr_jac)
        BSQP.delete_Matrix(self.Matrix_condensed_grad_obj)

        BSQP.delete_Matrix(self.Matrix_ub_con)
        BSQP.delete_Matrix(self.Matrix_lb_con)
        BSQP.delete_Matrix(self.Matrix_ub_var)
        BSQP.delete_Matrix(self.Matrix_lb_var)
        BSQP.delete_SymMatrix_array(self.SymMatrix_array_hess)
        BSQP.delete_Sparse_Matrix(self.Sparse_Matrix_constr_jac)
        BSQP.delete_Matrix(self.Matrix_grad_obj)
        
        BSQP.delete_Condenser(self.cxx_obj)
        
        BSQP.delete_target_array(self.target_array_obj)
        BSQP.delete_hsize_array(self.hsize_array_obj)
        BSQP.delete_cblock_array(self.cblock_array_obj)
        BSQP.delete_vblock_array(self.vblock_array_obj)
    
    def print_info(self):
        self.BSQP.Condenser_print_info(self.cxx_obj)
    
    def condensed_nBlocks(self):
        self.BSQP.Condenser_condensed_nBlocks(self.cxx_obj)
    
    def full_condense(self, grad_obj : typing.Iterable, constr_jac : Sparse_Matrix, hess : typing.Iterable[typing.Iterable], lb_var : typing.Iterable, ub_var : typing.Iterable, lb_con : typing.Iterable, ub_con : typing.Iterable):
        BSQP = self.BSQP
        
        nVar = BSQP.Condenser_nVar(self.cxx_obj)
        nCon = BSQP.Condenser_nCon(self.cxx_obj)
        nBlocks = BSQP.Condenser_nBlocks(self.cxx_obj)
    
        nnz = len(constr_jac.nz)
        assert nnz == constr_jac.colind[constr_jac.n]
        
        as_array(BSQP.Matrix_array(self.Matrix_grad_obj), nVar, c_double)[:] = grad_obj
        
        BSQP.Sparse_Matrix_set_structure(self.Sparse_Matrix_constr_jac, nCon, nVar, nnz)
        as_array(BSQP.Sparse_Matrix_nz(self.Sparse_Matrix_constr_jac), nnz, c_double)[:] = constr_jac.nz
        as_array(BSQP.Sparse_Matrix_row(self.Sparse_Matrix_constr_jac), nnz, c_int)[:] = constr_jac.row
        as_array(BSQP.Sparse_Matrix_colind(self.Sparse_Matrix_constr_jac), nVar + 1, c_int)[:] = constr_jac.colind
    
        hsizes = as_array(BSQP.Condenser_hsizes(self.cxx_obj), nBlocks, c_int)
        for i in range(nBlocks):
            hsize = hsizes[i]
            hessblock_data = as_array(BSQP.SymMatrix_array_index_array(self.SymMatrix_array_hess, c_int(i)), int((hsize * (hsize + 1)) // 2), c_double)
            hessColMajor = np.asarray(hess[i], dtype=np.float64).reshape(-1, order = 'F')
            full_to_lower(hessblock_data, hessColMajor, hsize)
        
        as_array(BSQP.Matrix_array(self.Matrix_lb_var), nVar, c_double)[:] = lb_var
        as_array(BSQP.Matrix_array(self.Matrix_ub_var), nVar, c_double)[:] = ub_var
        as_array(BSQP.Matrix_array(self.Matrix_lb_con), nCon, c_double)[:] = lb_con
        as_array(BSQP.Matrix_array(self.Matrix_ub_con), nCon, c_double)[:] = ub_con
        
        BSQP.Condenser_full_condense(
            self.cxx_obj,
            self.Matrix_grad_obj,
            self.Sparse_Matrix_constr_jac,
            self.SymMatrix_array_hess,
            self.Matrix_lb_var,
            self.Matrix_ub_var,
            self.Matrix_lb_con,
            self.Matrix_ub_con,
            
            self.Matrix_condensed_grad_obj,
            self.Sparse_Matrix_condensed_constr_jac,
            self.SymMatrix_array_condensed_hess,
            self.Matrix_condensed_lb_var,
            self.Matrix_condensed_ub_var,
            self.Matrix_condensed_lb_con,
            self.Matrix_condensed_ub_con
        )
        
        condensed_nVar = BSQP.Condenser_condensed_nVar(self.cxx_obj)
        condensed_nCon = BSQP.Condenser_condensed_nCon(self.cxx_obj)
        condensed_nnz = BSQP.Sparse_Matrix_nnz(self.Sparse_Matrix_condensed_constr_jac)
        
        condensed_grad_obj = np.array(as_array(BSQP.Matrix_array(self.Matrix_condensed_grad_obj), condensed_nVar, c_double), copy = True)
    
        condensed_constr_jac = Sparse_Matrix_structure(condensed_nCon, condensed_nVar, condensed_nnz)
        condensed_constr_jac.nz[:] = as_array(BSQP.Sparse_Matrix_nz(self.Sparse_Matrix_condensed_constr_jac), condensed_nnz, c_double)
        condensed_constr_jac.row[:] = as_array(BSQP.Sparse_Matrix_row(self.Sparse_Matrix_condensed_constr_jac), condensed_nnz, c_int)
        condensed_constr_jac.colind[:] = as_array(BSQP.Sparse_Matrix_colind(self.Sparse_Matrix_condensed_constr_jac), condensed_nVar + 1, c_int)
    
        condensed_nBlocks = BSQP.Condenser_condensed_nBlocks(self.cxx_obj)
        condensed_hsizes = as_array(BSQP.Condenser_condensed_hsizes(self.cxx_obj), condensed_nBlocks, c_int)
    
        condensed_hess = []
        for i in range(condensed_nBlocks):
            hsize = condensed_hsizes[i]
            full = np.empty((hsize, hsize), dtype=np.float64)
            lower = as_array(BSQP.SymMatrix_array_index_array(self.SymMatrix_array_condensed_hess, c_int(i)), (hsize * (hsize + 1)) // 2, c_double)
            lower_to_full(full.reshape(-1, order = 'F'), lower, hsize)
            condensed_hess.append(full)
    
        condensed_lb_var = np.array(as_array(BSQP.Matrix_array(self.Matrix_condensed_lb_var), condensed_nVar, c_double), copy = True)
        condensed_ub_var = np.array(as_array(BSQP.Matrix_array(self.Matrix_condensed_ub_var), condensed_nVar, c_double), copy = True)
        condensed_lb_con = np.array(as_array(BSQP.Matrix_array(self.Matrix_condensed_lb_con), condensed_nCon, c_double), copy = True)
        condensed_ub_con = np.array(as_array(BSQP.Matrix_array(self.Matrix_condensed_ub_con), condensed_nCon, c_double), copy = True)
        
        return condensed_grad_obj, condensed_constr_jac, condensed_hess, condensed_lb_var, condensed_ub_var, condensed_lb_con, condensed_ub_con

    def recover_var_mult(self, xi_cond, lambda_cond):
        BSQP = self.BSQP
        nVar = BSQP.Condenser_nVar(self.cxx_obj)
        nCon = BSQP.Condenser_nCon(self.cxx_obj)
        condensed_nVar = BSQP.Condenser_condensed_nVar(self.cxx_obj)
        condensed_nCon = BSQP.Condenser_condensed_nCon(self.cxx_obj)
        
        as_array(BSQP.Matrix_array(self.Matrix_xi_cond), condensed_nVar, c_double)[:] = np.asarray(xi_cond, dtype=np.float64)
        as_array(BSQP.Matrix_array(self.Matrix_lambda_cond), condensed_nVar + condensed_nCon, c_double)[:] = np.asarray(lambda_cond, dtype=np.float64)
        
        BSQP.Condenser_recover_var_mult(self.cxx_obj, self.Matrix_xi_cond, self.Matrix_lambda_cond, self.Matrix_xi_rest, self.Matrix_lambda_rest)
        xi_rest = np.array(as_array(BSQP.Matrix_array(self.Matrix_xi_rest), nVar, c_double), copy = True)
        lambda_rest = np.array(as_array(BSQP.Matrix_array(self.Matrix_lambda_rest), nVar + nCon, c_double), copy = True)
        
        return xi_rest, lambda_rest
    
    
    