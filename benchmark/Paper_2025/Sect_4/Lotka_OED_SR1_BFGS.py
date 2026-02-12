import numpy as np
import sys
from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()
sys.path += [str(cD.parents[1]), str(cD.parents[2]/Path("Python"))]
import blockSQP2

itMax = 250
step_plots = False
plot_title = False

import OCProblems
OCprob = OCProblems.Lotka_OED(
                    nt = 100, 
                    refine = 1, 
                    parallel = False, 
                    integrator = 'RK4', 
                    )

# Note: Due to randomness in the sparse linear solver, 
# several runs may be needed to reproduce the exact plots 
# in the paper (114 total iterations, some runs lead to 150).
# The overall iteration behavior is unaffected.


################################
opts = blockSQP2.SQPoptions()
opts.max_QP_it = 10000
opts.max_QP_secs = 5.0

opts.max_conv_QPs = 1
opts.conv_strategy = 0
opts.par_QPs = False
opts.enable_QP_cancellation = False
opts.indef_delay = 1

opts.hess_approx = 'SR1'
opts.sizing = 'OL'
opts.fallback_approx = 'BFGS'
opts.fallback_sizing = 'COL'
opts.BFGS_damping_factor = 0.2

opts.lim_mem = True
opts.mem_size = 20
opts.opt_tol = 1e-6
opts.feas_tol = 1e-6

opts.automatic_scaling = False
opts.max_extra_steps = 0
opts.enable_premature_termination = False
opts.max_filter_overrides = 0

opts.qpsol = 'qpOASES'
QPopts = blockSQP2.qpOASES_options()
QPopts.printLevel = 0
QPopts.sparsityLevel = 2
opts.qpsol_options = QPopts
################################

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
prob.x_start = OCprob.start_point
prob.lam_start = np.zeros(prob.nVar + prob.nCon, dtype = np.float64).reshape(-1)
prob.complete()

stats = blockSQP2.SQPstats("./solver_outputs")

xi_arr = []
optimizer = blockSQP2.SQPmethod(prob, opts, stats)
optimizer.init()
################################

ret = optimizer.run(1)
i = 1
if step_plots:
    OCprob.plot(OCprob.start_point, dpi = 200, it = 0, title=plot_title)
    xi = np.array(optimizer.get_xi()).reshape(-1)
    OCprob.plot(xi, dpi = 200, it = i, title=plot_title)
    while ret == blockSQP2.SQPresults.it_finished and i < itMax:
        ret = optimizer.run(1,1)
        xi = np.array(optimizer.get_xi()).reshape(-1)
        i += 1
        if i in (15, 80, 105, 106):
            xi_arr.append(xi)
        OCprob.plot(xi, dpi=200, title=plot_title)
else:
    while ret == blockSQP2.SQPresults.it_finished and i < itMax:
        ret = optimizer.run(1,1)
        i += 1
        if i in (15, 80, 105, 106):
            xi_arr.append(np.array(optimizer.get_xi()).reshape(-1))

for sol in xi_arr:
    OCprob.plot(sol, dpi=200, title=plot_title)
