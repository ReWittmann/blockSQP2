import numpy as np
import sys
from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()
sys.path += [str(cD.parents[1]), str(cD.parents[2]/Path("Python"))]
import time
import blockSQP2

import OCProblems
OCprob = OCProblems.Three_Tank_Multimode(
                    nt = 100, 
                    refine = 1, 
                    parallel = True, 
                    integrator = 'RK4'
                    )
itMax = 100
################################
opts = blockSQP2.SQPoptions()
opts.max_QP_it = 10000
opts.max_QP_secs = 5.0

opts.max_conv_QPs = 4
opts.conv_strategy = 2
opts.par_QPs = True
opts.enable_QP_cancellation = True
opts.indef_delay = 3

opts.hess_approx = 'SR1'
opts.sizing = 'OL'
opts.fallback_approx = 'BFGS'
opts.fallback_sizing = 'COL'
opts.BFGS_damping_factor = 1/3

opts.lim_mem = True
opts.mem_size = 20
opts.opt_tol = 1e-6
opts.feas_tol = 1e-6

opts.automatic_scaling = True

opts.max_extra_steps = 0
opts.enable_premature_termination = False
opts.max_filter_overrides = 3

opts.qpsol = 'qpOASES'
QPopts = blockSQP2.qpOASES_options()
QPopts.printLevel = 0
QPopts.sparsityLevel = 2
opts.qpsol_options = QPopts
################################

#Create condenser, pass as cond attribute of problem specification to enable condensing
vBlocks = blockSQP2.vblock_array(len(OCprob.vBlock_sizes))
for i in range(len(OCprob.vBlock_sizes)):
    vBlocks[i] = blockSQP2.vblock(OCprob.vBlock_sizes[i], OCprob.vBlock_dependencies[i])


#Define blockSQP Problemspec
prob = blockSQP2.Problemspec()
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


stats = blockSQP2.SQPstats("./solver_outputs")

optimizer = blockSQP2.SQPmethod(prob, opts, stats)
optimizer.init()


ret = optimizer.run(100)
xi = np.array(optimizer.get_xi()).reshape(-1)
OCprob.plot(xi, dpi=200)


#Enable new termination features
opts.max_extra_steps = 10
opts.enable_premature_termination = True
opts.max_filter_overrides = 3


optimizer2 = blockSQP2.SQPmethod(prob, opts, stats)
optimizer2.init()
ret = optimizer2.run(100)
xi_accurate = np.array(optimizer2.get_xi()).reshape(-1)
OCprob.plot(xi_accurate,dpi=200)

time.sleep(0.1)
print("Optimality and feasibility error without additional steps: ", optimizer.vars.tol, ", ", optimizer.vars.cNorm, "\n")
print("Optimality and feasibility error with additional steps: ", optimizer2.vars.tol, ", ", optimizer2.vars.cNorm, "\n")

