import numpy as np
import os
import sys
import time
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
except:
    sys.path.append(os.getcwd() + "/..")

import py_blockSQP
from blockSQP_pyProblem import blockSQP_pyProblem as Problemspec
import matplotlib.pyplot as plt

itMax = 8

step_plots = False
plot_title = True


import OCProblems
OCprob = OCProblems.Lotka_Volterra_Fishing(nt = 3, refine=8, parallel = False, integrator = 'RK4')

################################
opts = py_blockSQP.SQPoptions()
opts.max_QP_it = 10000
opts.max_QP_secs = 5.0

opts.max_conv_QPs = 4
opts.conv_strategy = 2
opts.exact_hess = 0
opts.hess_approx = 1
opts.sizing = 2
opts.fallback_approx = 2
opts.fallback_sizing = 4
opts.BFGS_damping_factor = 0.2

opts.lim_mem = True
opts.mem_size = 20
opts.opt_tol = 1e-6
opts.feas_tol = 1e-6
opts.conv_kappa_max = 100.0

opts.automatic_scaling = False

opts.max_extra_steps = 0
opts.enable_premature_termination = False
opts.max_filter_overrides = 0


opts.qpsol = 'qpOASES'
QPopts = py_blockSQP.qpOASES_options()
QPopts.terminationTolerance = 1e-10
QPopts.printLevel = 0
QPopts.sparsityLevel = 2
opts.qpsol_options = QPopts

################################

#Create condenser, chose SCQPmethod below to enable condensing
vBlocks = py_blockSQP.vblock_array(len(OCprob.vBlock_sizes))
cBlocks = py_blockSQP.cblock_array(len(OCprob.cBlock_sizes))
hBlocks = py_blockSQP.int_array(len(OCprob.hessBlock_sizes))
targets = py_blockSQP.condensing_targets(1)
for i in range(len(OCprob.vBlock_sizes)):
    vBlocks[i] = py_blockSQP.vblock(OCprob.vBlock_sizes[i], OCprob.vBlock_dependencies[i])
for i in range(len(OCprob.cBlock_sizes)):
    cBlocks[i] = py_blockSQP.cblock(OCprob.cBlock_sizes[i])
for i in range(len(OCprob.hessBlock_sizes)):
    hBlocks[i] = OCprob.hessBlock_sizes[i]
targets[0] = py_blockSQP.condensing_target(*OCprob.ctarget_data)
cond = py_blockSQP.Condenser(vBlocks, cBlocks, hBlocks, targets)



#Define blockSQP Problemspec
prob = Problemspec()
prob.nVar = OCprob.nVar
prob.nCon = OCprob.nCon

prob.f = lambda x: OCprob.f(x)
prob.grad_f = lambda x: OCprob.grad_f(x)
prob.g = lambda x: OCprob.g(x)
prob.make_sparse(OCprob.jac_g_nnz, OCprob.jac_g_row, OCprob.jac_g_colind)
prob.jac_g_nz = lambda x: OCprob.jac_g_nz(x)
prob.hess = OCprob.hess_lag

prob.set_blockIndex(OCprob.hessBlock_index)
prob.set_bounds(OCprob.lb_var, OCprob.ub_var, OCprob.lb_con, OCprob.ub_con)

prob.vblocks = vBlocks

prob.x_start = OCprob.start_point
prob.lam_start = np.zeros(prob.nVar + prob.nCon, dtype = np.float64).reshape(-1)
prob.complete()

scale_arr = 1.0;
#####################
stats = py_blockSQP.SQPstats("./solver_outputs")
optimizer = py_blockSQP.SQPmethod(prob, opts, stats)
optimizer.init()
#####################
ret = int(optimizer.run(itMax))
xi = np.array(optimizer.get_xi()).reshape(-1)/scale_arr
#####################

time.sleep(0.2)
A = OCprob.jac_g(xi)
u = OCprob.get_control_arrays(xi)
x1,x2 = OCprob.get_state_arrays(xi)

print("\n\n\nu=", u)
print("\nx1 = ", x1, ", x2 = ", x2)
print("\nConstraint Jacobian is\n", A)




