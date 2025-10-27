import numpy as np
import time
import sys
from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()
sys.path += [str(cD.parents[2])]

import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import py_blockSQP


structure_data = np.load(cD / Path('structure_data.npz'))

vblock_sizes = structure_data['vblock_sizes']
vblock_dependencies = structure_data['vblock_dependencies']
cblock_sizes = structure_data['cblock_sizes']
hsizes = structure_data['hsizes']
targets_data = []
for i in range(5):
    targets_data.append(structure_data['target_' + str(i)])


#Condensing information
vblocks = py_blockSQP.vblock_array(len(vblock_sizes))
for i in range(len(vblock_sizes)):
    vblocks[i] = py_blockSQP.vblock(vblock_sizes[i], vblock_dependencies[i])

cblocks = py_blockSQP.cblock_array(len(cblock_sizes))
for i in range(len(cblock_sizes)):
    cblocks[i] = py_blockSQP.cblock(cblock_sizes[i])

targets = py_blockSQP.condensing_targets(5)
for i in range(5):
    targets[i] = py_blockSQP.condensing_target(*(targets_data[i]))

hessblock_sizes = py_blockSQP.int_array(len(hsizes))
np.array(hessblock_sizes, copy = False)[:] = hsizes

cond_nobounds = py_blockSQP.Condenser(vblocks, cblocks, hessblock_sizes, targets, 0)
cond_bounds = py_blockSQP.Condenser(vblocks, cblocks, hessblock_sizes, targets, 2)

#Prepare calculation of the condensed jacobian

###Create py_blockSQP.condensing_args object to pass to Condenser
def identity_S(n, scale = 1.0):
    M = py_blockSQP.SymMatrix(n)
    M.Initialize(0)
    for i in range(n):
        M[i,i] = scale
    return M

prob_vectors = np.load(cD / Path('prob_vectors.npz'))

lb_var = prob_vectors['lb_var'].reshape(-1)
ub_var = prob_vectors['ub_var'].reshape(-1)
lb_con = prob_vectors['lb_con'].reshape(-1)
ub_con = prob_vectors['ub_con'].reshape(-1)
grad_obj = prob_vectors['grad_obj'].reshape(-1)

M_lb_var = py_blockSQP.Matrix(len(lb_var))
M_ub_var = py_blockSQP.Matrix(len(ub_var))
M_lb_con = py_blockSQP.Matrix(len(lb_con))
M_ub_con = py_blockSQP.Matrix(len(ub_con))
M_grad_obj = py_blockSQP.Matrix(len(grad_obj))

np.array(M_lb_var, copy = False)[:,0] = lb_var
np.array(M_ub_var, copy = False)[:,0] = ub_var
np.array(M_lb_con, copy = False)[:,0] = lb_con
np.array(M_ub_con, copy = False)[:,0] = ub_con
np.array(M_grad_obj, copy = False)[:,0] = grad_obj

Jacobian = np.load(cD / Path('Jacobian.npz'))

nnz = int(Jacobian['nnz'])
m = int(Jacobian['m'])
n = int(Jacobian['n'])
nz = np.array(Jacobian['nz'])
row = np.array(Jacobian['row'])
colind = np.array(Jacobian['colind'])

A_nz = py_blockSQP.double_array(nnz)
A_row = py_blockSQP.int_array(nnz)
A_colind = py_blockSQP.int_array(n + 1)
np.array(A_nz, copy = False)[:] = nz
np.array(A_row, copy = False)[:] = row
np.array(A_colind, copy = False)[:] = colind

SM_Jacobian = py_blockSQP.Sparse_Matrix(m, n, A_nz, A_row, A_colind)

#Set Hessian as identity scaled by 1e-4, causes the step 
#to potentially violate implicit bounds
hess = py_blockSQP.SymMat_array(len(hsizes))
for i in range(len(hsizes)):
    hess[i] = identity_S(hsizes[i], 1e-4)


cond_args_nobounds = py_blockSQP.condensing_args()
cond_args_nobounds.grad_obj = M_grad_obj
cond_args_nobounds.con_jac = SM_Jacobian 
cond_args_nobounds.hess = hess
cond_args_nobounds.lb_var = M_lb_var
cond_args_nobounds.ub_var = M_ub_var
cond_args_nobounds.lb_con = M_lb_con
cond_args_nobounds.ub_con = M_ub_con

#Condense a QP to obtain the condensed jacobian
cond_nobounds.condense_args(cond_args_nobounds)

#Optional: Solve both QPs and compare the solution (-times)
sys.stdout.flush()
print("Solving the full QP and condensed QP (without implicit bounds), this may take up to half a minute ...")
sys.stdout.flush()
time.sleep(0.01)
cond_args_nobounds.solve_QPs()
#QP solution
deltaXi = np.array(cond_args_nobounds.deltaXi).reshape(-1)
#QP solution restored from condensed QP solution
deltaXi_rest_nobounds = np.array(cond_args_nobounds.deltaXi_rest).reshape(-1)
print("||deltaXi - deltaXi_rest_nobounds|| = ", np.linalg.norm(deltaXi - deltaXi_rest_nobounds, np.inf), "\n\n")

cond_args_bounds = py_blockSQP.condensing_args()
cond_args_bounds.grad_obj = M_grad_obj
cond_args_bounds.con_jac = SM_Jacobian 
cond_args_bounds.hess = hess
cond_args_bounds.lb_var = M_lb_var
cond_args_bounds.ub_var = M_ub_var
cond_args_bounds.lb_con = M_lb_con
cond_args_bounds.ub_con = M_ub_con

#Condense with condenser that includes dependent variable bounds
cond_bounds.condense_args(cond_args_bounds)

#Optional: Solve both QPs and compare the solution (-times)
sys.stdout.flush()
print("Solving the full QP and condensed QP (with implicit bounds), this may take up to half a minute ...")
sys.stdout.flush()
time.sleep(0.01)
cond_args_bounds.solve_QPs()
#QP solution
deltaXi = np.array(cond_args_bounds.deltaXi).reshape(-1)
#QP solution restored from condensed QP solution
deltaXi_rest_bounds = np.array(cond_args_bounds.deltaXi_rest).reshape(-1)
print("||deltaXi - deltaXi_rest_bounds|| = ", np.linalg.norm(deltaXi - deltaXi_rest_bounds, np.inf))


#Plot the sparsity structure of the condensed jacobian
J_full = SM_Jacobian 
J_cond_NB = cond_args_nobounds.condensed_Jacobian
J_cond_B = cond_args_bounds.condensed_Jacobian


nnz = J_full.nnz
nz = [1]*nnz
m = J_full.m
n = J_full.n
row = np.array(J_full.ROW)
colind = np.array(J_full.COLIND)
col = []
for j in range(n):
    for i in range(colind[j], colind[j+1]):
        col.append(j)

J_full_coo = coo_matrix((nz, (row, col)), shape=(m,n))


nnz_cond = J_cond_NB.nnz
nz_cond = [1]*nnz_cond
m_cond = J_cond_NB.m
n_cond = J_cond_NB.n
row_cond = np.array(J_cond_NB.ROW)
colind_cond = np.array(J_cond_NB.COLIND)
col_cond = []
for j in range(n_cond):
    for i in range(colind_cond[j], colind_cond[j+1]):
        col_cond.append(j)

J_cond_NB_coo = coo_matrix((nz_cond, (row_cond, col_cond)), shape=(m_cond,n_cond))


nnz_cond_2 = J_cond_B.nnz
nz_cond_2 = [1]*nnz_cond_2
m_cond_2 = J_cond_B.m
n_cond_2 = J_cond_B.n
row_cond_2 = np.array(J_cond_B.ROW)
colind_cond_2 = np.array(J_cond_B.COLIND)
col_cond_2 = []
for j in range(n_cond_2):
    for i in range(colind_cond_2[j], colind_cond_2[j+1]):
        col_cond_2.append(j)

J_cond_B_coo = coo_matrix((nz_cond_2, (row_cond_2, col_cond_2)), shape=(m_cond_2,n_cond_2))


#Full constraint matrix
plt.rcParams["text.usetex"] = True
fig, ax = plt.subplots(dpi = 200, figsize = (12*0.65, 9*0.65))
ax.spy(J_full_coo, markersize = 0.05)
ax.tick_params(labelsize = 'x-large')
plt.show()

#Condensed constraint matrix with no implicit bounds
fig, ax = plt.subplots(dpi = 200, figsize = (12*0.55, 9*0.55))
ax.spy(J_cond_NB_coo, markersize = 0.05)
ax.set_xticks(np.array([0,1737]))
ax.set_yticks(np.array([0,1844]))
ax.tick_params(labelsize = 'x-large')
plt.show()

#Condensed constraint matrix with included implicit bounds
fig, ax = plt.subplots(dpi = 200, figsize = (12*0.75, 9*0.75))
ax.spy(J_cond_B_coo, markersize = 0.05)
ax.set_xticks(np.array([0,1737]))
# ax.tick_params(labelsize = 'x-large')
ax.tick_params(labelsize = 16.0)
plt.show()
