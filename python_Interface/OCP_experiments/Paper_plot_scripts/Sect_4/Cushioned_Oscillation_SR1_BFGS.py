import numpy as np
import os
import sys
try:
    cD = os.path.dirname(os.path.abspath(__file__))
except:
    cD = os.getcwd()
sys.path += [cD + "/../../..", cD + "/../../../examples"]

import py_blockSQP

itMax = 100
step_plots = True
plot_title = False

import OCProblems
OCprob = OCProblems.Electric_Car(nt = 100, 
                    refine = 1, 
                    parallel = True, 
                    integrator = 'RK4', 
                    )


################################
opts = py_blockSQP.SQPoptions()
opts.max_QP_it = 10000
opts.max_QP_secs = 5.0

opts.max_conv_QPs = 1
opts.conv_strategy = 0
opts.par_QPs = False
opts.enable_QP_cancellation = False
opts.indef_delay = 1

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

opts.automatic_scaling = False
opts.max_extra_steps = 0
opts.enable_premature_termination = False
opts.max_filter_overrides = 0

opts.qpsol = 'qpOASES'
QPopts = py_blockSQP.qpOASES_options()
QPopts.printLevel = 0
QPopts.sparsityLevel = 2
# QPopts.terminationTolerance = 1e-10
opts.qpsol_options = QPopts
################################

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
prob.x_start = OCprob.start_point
prob.lam_start = np.zeros(prob.nVar + prob.nCon, dtype = np.float64).reshape(-1)
prob.complete()

stats = py_blockSQP.SQPstats("./solver_outputs")

optimizer = py_blockSQP.SQPmethod(prob, opts, stats)
optimizer.init()
#####################
if (step_plots):
    OCprob.plot(OCprob.start_point, dpi = 200, it = 0, title=plot_title)
    ret = int(optimizer.run(1))
    xi = np.array(optimizer.get_xi()).reshape(-1)
    i = 1
    OCprob.plot(xi, dpi = 200, it = i, title=plot_title)
    while ret == 0 and i < itMax:
        ret = int(optimizer.run(1,1))
        xi = np.array(optimizer.get_xi()).reshape(-1)
        i += 1
        OCprob.plot(xi, dpi = 200, it = i, title=plot_title)
else:
    ret = int(optimizer.run(itMax))
    xi = np.array(optimizer.get_xi()).reshape(-1)
OCprob.plot(xi, dpi=200, title=plot_title)
