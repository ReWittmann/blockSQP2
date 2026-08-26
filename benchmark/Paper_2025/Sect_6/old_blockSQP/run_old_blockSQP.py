# blockSQP_reference_build - build system and Python interface for blockSQP
# Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
# Licensed under the zlib license. See LICENSE for more details.

import numpy as np
import time

import py_blockSQP_old as py_blockSQP

import sys
from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()
sys.path.append(str(cD.parents[2]))
import OCP_experiment
import OCProblems

OCprob = OCProblems.Lotka_Volterra_Fishing(
                    nt = 100,           #number of shooting intervals
                    refine = 1,         #number of control intervals per shooting interval
                    integrator = 'RK4', #ODE integrator
                    parallel = True,    #run ODE integration in parallel
                    N_threads = 4,      #number of threads for parallelization
                    )

itMax = 200                             #maximum number of steps
step_plots = False                      #plot every iteration
plot_title = False                      #put name of problem in plots


################################
opts = py_blockSQP.SQPoptions()
opts.maxItQP = 10000
opts.maxTimeQP = 10.0

opts.maxConvQP = 1                      #number of additional QPs per SQP iteration, includes fallback BFGS
opts.convStrategy = 0                   #convexification strategy, only 0: convex combinations is available

opts.whichSecondDerv = 0                #2: broken in this version of blockSQP
opts.hessUpdate = 1                     #1: SR1, 2: damped BFGS
opts.hessScaling = 2                    #2: Oren-Luenberger, 4: Selective COL sizing 
opts.fallbackUpdate = 2                 # ''    ''    ''    ''
opts.fallbackScaling = 4                # ''    ''    ''    ''

opts.hessLimMem = 1                     #0: Full memory, 1: limited memory
opts.hessMemsize = 20
opts.opttol = 1e-6
opts.nlinfeastol = 1e-6
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

blockIndex = np.array(OCprob.hessBlock_index, dtype=np.int32)
prob.set_blockIndex(blockIndex)
prob.set_bounds(OCprob.lb_var, OCprob.ub_var, OCprob.lb_con, OCprob.ub_con)

prob.x_start = OCprob.start_point

prob.lam_start = np.zeros(prob.nVar + prob.nCon, dtype = np.float64).reshape(-1)
prob.complete()


#####################
stats = py_blockSQP.SQPstats("./solver_outputs")
t0 = time.monotonic()
optimizer = py_blockSQP.SQPmethod(prob, opts, stats)
optimizer.init()

#####################
if (step_plots):
    OCprob.plot(OCprob.start_point, dpi = 150, it = 0, title=plot_title)
    ret = int(optimizer.run(1))
    xi = np.array(optimizer.vars.xi).reshape(-1)
    i = 1
    OCprob.plot(xi, dpi = 150, it = i, title=plot_title)
    while ret == 1 and i < itMax:
        ret = int(optimizer.run(1,1))
        xi = np.array(optimizer.vars.xi).reshape(-1)
        i += 1
        OCprob.plot(xi, dpi = 150, it = i, title=plot_title)
else:
    ret = int(optimizer.run(itMax))
t1 = time.monotonic()
xi = np.array(optimizer.vars.xi).reshape(-1)
OCprob.plot(xi, dpi=150, title=plot_title)
#####################

time.sleep(0.01)
print(t1 - t0, "s")