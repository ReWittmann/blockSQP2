import numpy as np
import os
import sys
import time
try:
    sys.path += [os.path.dirname(os.path.abspath(__file__)) + "/..", os.path.dirname(os.path.abspath(__file__)) + "/../.."]
except:
    sys.path += [os.getcwd() + "/..", os.getcwd() + "/../.."]

import py_blockSQP
from blockSQP_pyProblem import blockSQP_pyProblem as Problemspec
import matplotlib.pyplot as plt

itMax = 200

step_plots = True
plot_title = False


import OCProblems


OCprob = OCProblems.Lotka_Volterra_Fishing(nt = 100, parallel = False)

################################
opts = py_blockSQP.SQPoptions()
opts.max_QP_iter = 10000
opts.max_QP_seconds = 5.0

opts.max_conv_QPs = 1
opts.conv_strategy = 2
opts.hess_approx = 1
opts.sizing_strategy = 2
opts.fallback_approx = 2
opts.fallback_sizing_strategy = 4
opts.BFGS_damping_factor = 0.2
opts.automatic_scaling = False
opts.block_hess = True


opts.qpsol = 'qpOASES'
QPopts = py_blockSQP.qpOASES_options()
QPopts.terminationTolerance = 1e-10
QPopts.printLevel = 0
QPopts.sparsityLevel = 2
opts.qpsol_options = QPopts


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


prob.x_start = OCprob.start_point
prob.lam_start = np.zeros(prob.nVar + prob.nCon, dtype = np.float64).reshape(-1)


vblocks = py_blockSQP.vblock_array(len(OCprob.vBlock_sizes))
for i in range(len(OCprob.vBlock_sizes)):
    vblocks[i] = py_blockSQP.vblock(OCprob.vBlock_sizes[i], OCprob.vBlock_dependencies[i])
prob.vblocks = vblocks

prob.complete()


stats = py_blockSQP.SQPstats("./solver_outputs")

#No condensing
optimizer = py_blockSQP.SQPmethod(prob, opts, stats)

optimizer.init()
#####################
t0 = time.time()
if (step_plots):
    OCprob.plot(OCprob.start_point, dpi = 150, it = 0, title=plot_title)
    ret = int(optimizer.run(1))
    xi = np.array(optimizer.get_xi()).reshape(-1)
    i = 1
    OCprob.plot(xi, dpi = 150, it = i, title=plot_title)
    # OCprob.plot(xi, dpi = 200, it = i, title=False)
    while ret == 0 and i < itMax:
        ret = int(optimizer.run(1,1))
        xi = np.array(optimizer.get_xi()).reshape(-1)
        i += 1
        OCprob.plot(xi, dpi = 150, it = i, title=plot_title)
        # OCprob.plot(xi, dpi = 200, it = i, title=False)
else:
    ret = int(optimizer.run(itMax))
    xi = np.array(optimizer.get_xi()).reshape(-1)
t1 = time.time()
#####################
