import numpy as np
import time
import sys
from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()
sys.path += [str(cD.parents[2]/Path("Python"))]

import matplotlib.pyplot as plt
# plt.rcParams["text.usetex"] = True

from scipy.sparse import coo_matrix
import blockSQP2


structure_data = np.load(cD / Path('structure_data.npz'))

vblock_sizes = structure_data['vblock_sizes']
vblock_dependencies = structure_data['vblock_dependencies']
cblock_sizes = structure_data['cblock_sizes']
hsizes = structure_data['hsizes']
targets_data = []
for i in range(5):
    targets_data.append(structure_data['target_' + str(i)])

vblocks = [blockSQP2.vblock(size, bool(dep)) for size, dep in zip(vblock_sizes, vblock_dependencies)]
cblocks = [blockSQP2.cblock(size) for size in cblock_sizes]
hessblock_sizes = hsizes
targets = [blockSQP2.condensing_target(*tdata) for tdata in targets_data]

cond_nobounds = blockSQP2.Condenser(vblocks, cblocks, hessblock_sizes, targets, 0)
cond_bounds = blockSQP2.Condenser(vblocks, cblocks, hessblock_sizes, targets, 2)


prob_vectors = np.load(cD / Path('prob_vectors.npz'))

lb_var = prob_vectors['lb_var'].reshape(-1)
ub_var = prob_vectors['ub_var'].reshape(-1)
lb_con = prob_vectors['lb_con'].reshape(-1)
ub_con = prob_vectors['ub_con'].reshape(-1)
grad_obj = prob_vectors['grad_obj'].reshape(-1)

Jacobian = np.load(cD / Path('Jacobian.npz'))

nnz = int(Jacobian['nnz'])
m = int(Jacobian['m'])
n = int(Jacobian['n'])
nz = np.array(Jacobian['nz'])
row = np.array(Jacobian['row'])
colind = np.array(Jacobian['colind'])


constrJac = blockSQP2.Sparse_Matrix(m, n, nz, row, colind)
hess = [np.eye(hsize)*1e-4 for hsize in hsizes]
c_grad_n, c_jac_n, c_hess_n, c_lb_var_n, c_ub_var_n, c_lb_con_n, c_ub_con_n = cond_nobounds.full_condense(grad_obj, constrJac, hess, lb_var, ub_var, lb_con, ub_con)
c_grad_b, c_jac_b, c_hess_b, c_lb_var_b, c_ub_var_b, c_lb_con_b, c_ub_con_b = cond_bounds.full_condense(grad_obj, constrJac, hess, lb_var, ub_var, lb_con, ub_con)

# TODO: Re-enable solving the condensed QPs with qpOASES


#Plot the sparsity structure of the condensed jacobian
J_full = constrJac
J_cond_NB = c_jac_n
J_cond_B = c_jac_b


nnz = J_full.nnz()
nz = [1]*nnz
m = J_full.m
n = J_full.n
row = np.array(J_full.row)
colind = np.array(J_full.colind)
col = []
for j in range(n):
    for i in range(colind[j], colind[j+1]):
        col.append(j)

J_full_coo = coo_matrix((nz, (row, col)), shape=(m,n))


nnz_cond = J_cond_NB.nnz()
nz_cond = [1]*nnz_cond
m_cond = J_cond_NB.m
n_cond = J_cond_NB.n
row_cond = np.array(J_cond_NB.row)
colind_cond = np.array(J_cond_NB.colind)
col_cond = []
for j in range(n_cond):
    for i in range(colind_cond[j], colind_cond[j+1]):
        col_cond.append(j)

J_cond_NB_coo = coo_matrix((nz_cond, (row_cond, col_cond)), shape=(m_cond,n_cond))


nnz_cond_2 = J_cond_B.nnz()
nz_cond_2 = [1]*nnz_cond_2
m_cond_2 = J_cond_B.m
n_cond_2 = J_cond_B.n
row_cond_2 = np.array(J_cond_B.row)
colind_cond_2 = np.array(J_cond_B.colind)
col_cond_2 = []
for j in range(n_cond_2):
    for i in range(colind_cond_2[j], colind_cond_2[j+1]):
        col_cond_2.append(j)

J_cond_B_coo = coo_matrix((nz_cond_2, (row_cond_2, col_cond_2)), shape=(m_cond_2,n_cond_2))


#Full constraint matrix
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
