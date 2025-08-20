import numpy as np
import os
import sys
import time
import copy
try:
    cD = os.path.dirname(os.path.abspath(__file__))
except:
    cD = os.getcwd()
sys.path += [cD + "/../../..", cD + "/../../../examples"]

import py_blockSQP
import matplotlib.pyplot as plt

itMax = 100
step_plots = True
plot_title = True

import OCProblems

OCprob = OCProblems.Three_Tank_Multimode(nt = 100, 
                    refine = 1, 
                    parallel = True, 
                    integrator = 'RK4'
                    )

################################
opts = py_blockSQP.SQPoptions()
opts.max_QP_it = 10000
opts.max_QP_secs = 5.0

opts.max_conv_QPs = 6
opts.conv_strategy = 2
opts.par_QPs = True
opts.enable_QP_cancellation = True
opts.indef_delay = 3

opts.exact_hess = 0
opts.hess_approx = 1
opts.sizing = 2
opts.fallback_approx = 2
opts.fallback_sizing = 4
opts.BFGS_damping_factor = 1/3

opts.lim_mem = True
opts.mem_size = 20
opts.opt_tol = 1e-6
opts.feas_tol = 1e-6

opts.automatic_scaling = True

opts.max_extra_steps = 0
opts.enable_premature_termination = False
opts.max_filter_overrides = 2

opts.qpsol = 'qpOASES'
QPopts = py_blockSQP.qpOASES_options()
QPopts.printLevel = 0
QPopts.sparsityLevel = 2
opts.qpsol_options = QPopts
################################

#Create condenser, pass as cond attribute of problem specification to enable condensing
vBlocks = py_blockSQP.vblock_array(len(OCprob.vBlock_sizes))
for i in range(len(OCprob.vBlock_sizes)):
    vBlocks[i] = py_blockSQP.vblock(OCprob.vBlock_sizes[i], OCprob.vBlock_dependencies[i])


#Define blockSQP Problemspec
prob = py_blockSQP.Problemspec()
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


stats = py_blockSQP.SQPstats("./solver_outputs")

optimizer = py_blockSQP.SQPmethod(prob, opts, stats)
optimizer.init()


ret = optimizer.run(100)
xi = np.array(optimizer.get_xi()).reshape(-1)
OCprob.plot(xi)


#Enable new termination features
opts.max_extra_steps = 10
opts.enable_premature_termination = True
opts.max_filter_overrides = 2


optimizer2 = py_blockSQP.SQPmethod(prob, opts, stats)
optimizer2.init()
ret = optimizer2.run(100)
xi_accurate = np.array(optimizer2.get_xi()).reshape(-1)
OCprob.plot(xi_accurate)

