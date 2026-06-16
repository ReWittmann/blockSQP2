# blockSQP2 -- A structure-exploiting nonlinear programming solver based
#              on blockSQP by Dennis Janka.
# Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>

# Licensed under the zlib license. See LICENSE for more details.


# \file run_blockSQP2.py
# \author Reinhold Wittmann
# \date 2025
#
# Script to invoke blockSQP2 for an example problem.

import numpy as np
import time
import sys
from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()
sys.path += [str(cD.parent/Path("Python"))]

import blockSQP2
import OCProblems


#Check OCProblems.py for available examples
OCprob = OCProblems.Goddard_Rocket(
                    nt = 100,               #number of shooting intervals
                    refine = 1,             #number of control intervals per shooting interval
                    integrator = 'RK4',     #ODE integrator
                    parallel = True,        #run ODE integration in parallel
                    N_threads = 4,          #number of threads for parallelization
                                            #problem specific keyword parameters, e.g. c0, c1, x_init, t0, tf for Lotka_Volterra_Fishing, see default_params of problems
                    )

itMax = 100                                   #max number of steps
step_plots = True                           #Plot each iterate?
step_delay_ms = 0
plot_title = True                           #Put name of problem in plot?


start = OCprob.start_point                  #Start point for problem, can use, e.g. OCprob.perturbed_start_point(k)
################################
opts = blockSQP2.SQPoptions()
opts.max_QP_it = 10000
opts.max_QP_secs = 20.0

opts.max_conv_QPs = 4                       #max number of additional QPs per SQP iteration including fallback Hess QP
opts.conv_strategy = 2                      #Convexification strategy, 2 requires passing vblocks
opts.par_QPs = False                         #Enable parallel solution of QPs
opts.enable_QP_cancellation = True          #Enable cancellation of long running QP threads
opts.indef_delay = 3                        #Only use fallback Hessian in first # iterations

opts.hess_approx = 'SR1'                    #'SR1'/'BFGS'/'exact'
opts.sizing = 'OL'                          #'SP' - Shanno-Phua, 'OL' - Oren-Luenberger, 'GM_SP_OL' - geometric mean of SP and OL, 'COL' - centered Oren-Luenberger
opts.fallback_approx = 'BFGS'               # ''   ''
opts.fallback_sizing = 'COL'                # ''   ''
opts.BFGS_damping_factor = 1/3

opts.lim_mem = True
opts.mem_size = 20
opts.opt_tol = 1e-6                         #Tolerances for termination
opts.feas_tol = 1e-6
opts.conv_kappa_max = 8.                    #Maximum Hess regularization factor for conv. strategy, default 8.0

opts.automatic_scaling = True

opts.max_extra_steps = 0                    #Extra steps for improved accuracy
opts.enable_premature_termination = False    #Enable early termination at acceptable tolerance
opts.max_filter_overrides = 0

opts.test_opt_1 = True
# opts.qpsol = 'qpOASES'
# QPopts = blockSQP2.qpOASES_options()
# QPopts.printLevel = 0                     
# QPopts.sparsityLevel = 2                  #0-dense QPs, 1-sparse matrices + dense factorizations, 2 (default) - sparse matrices and factorizations
# opts.qpsol_options = QPopts
################################

#Create condenser, enable condensing by passing setting it as cond attribute of Problemspec
#Currently not recommended due to qpOASES only supporting sparse matrices when allowing indefinite Hessians
vblocks = [blockSQP2.vblock(size, dep) for size, dep in zip(OCprob.vBlock_sizes, OCprob.vBlock_dependencies)]
cblocks = [blockSQP2.cblock(size) for size in OCprob.cBlock_sizes]
hblocks = [size for size in OCprob.hessBlock_sizes]
targets = [blockSQP2.condensing_target(*OCprob.ctarget_data)]
condenser = blockSQP2.Condenser(vblocks, cblocks, hblocks, targets, 2)


#Define blockSQP Problemspec
#See class OCProblems.OCProblem and blockSQP2/Problem.py for field specifications
prob = blockSQP2.Problemspec(OCprob.nVar, OCprob.nCon)
prob.nVar = OCprob.nVar
prob.nCon = OCprob.nCon

#Pass necessary callbacks to Problemspec
prob.f = OCprob.f                           #objective
prob.grad_f = OCprob.grad_f
prob.g = OCprob.g                           #constraint function

#jac_g sparsity structure: # nonzeros, row and colind of CCS format
prob.make_sparse(OCprob.jac_g_nnz, OCprob.jac_g_row, OCprob.jac_g_colind)
prob.jac_g_nz = OCprob.jac_g_nz

prob.hess = OCprob.hess_lag
prob.blockIdx = OCprob.hessBlock_index
prob.set_bounds(OCprob.lb_var, OCprob.ub_var, OCprob.lb_con, OCprob.ub_con)

#Recommended: Dont pass condenser to activate condensing, 
#but pass vblocks to enable convexification strategy 2 and automatic scaling
prob.vblocks = vblocks
# prob.condenser = condenser

prob.x_start = start
prob.lam_start = np.zeros(prob.nVar + prob.nCon, dtype = np.float64).reshape(-1)

# scaledProb = blockSQP2.ScaledProblem(prob)
# scalingFactors = np.ones(prob.nVar, dtype = np.float64)
# for i in range(OCprob.ntS + 1):
#     OCprob.set_stage_control(scalingFactors, i, 1.0)
# scaledProb.set_scale(scalingFactors)

stats = blockSQP2.SQPstats("./solver_outputs")

t0 = time.monotonic()
optimizer = blockSQP2.SQPmethod(prob, opts, stats)
optimizer.init()

if (step_plots):
    OCprob.plot(OCprob.start_point, dpi = 150, it = 0, title=plot_title)
    ret = optimizer.run(1)
    xi = np.array(optimizer.get_xi()).reshape(-1)
    lam = optimizer.get_dual_solution()
    i = 1
    OCprob.plot(xi, dpi = 150, it = i, title=plot_title)
    while ret == blockSQP2.SQPresults.it_finished and i < itMax:
        ret = optimizer.run(1,1)
        xi = np.array(optimizer.get_xi()).reshape(-1)
        i += 1
        time.sleep(step_delay_ms/1000)
        OCprob.plot(xi, dpi = 150, it = i, title=plot_title)
else:
    ret = optimizer.run(itMax)
t1 = time.monotonic()
if not step_plots:
    xi = np.array(optimizer.get_xi()).reshape(-1)
    OCprob.plot(xi, dpi=200, title=plot_title)

time.sleep(0.01)
print(t1 - t0, "s")