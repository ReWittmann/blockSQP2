import numpy as np
import time
import os
import sys
try:
    sys.path += [os.path.dirname(os.path.abspath(__file__)) + "/../..", os.path.dirname(os.path.abspath(__file__)) + "/.."]
except:
    sys.path += [os.getcwd() + "/..", os.getcwd() + "/../.."]
import py_blockSQP
from blockSQP_pyProblem import blockSQP_pyProblem as Problemspec
import OCProblems

OCprob = OCProblems.Lotka_Volterra_Fishing(nt=100, parallel = False)


#Scale the controls from [0, 1] to [0, uscale]
uscale = 100´


################################
opts = py_blockSQP.SQPoptions()



opts.convStrategy = 1
opts.maxConvQP = 1
################################
opts.maxItQP = 100000
opts.maxTimeQP = 5.0

opts.hessUpdate = 1         #SR1
opts.hessScaling = 2        #Oren-Luenberger sizing
opts.fallbackUpdate = 2     #damped BFGS
opts.fallbackScaling = 4    #centered Oren-Luenberger sizing

opts.opttol = 1e-6
opts.nlinfeastol = 1e-6

opts.QPsol = 'qpOASES'
QPopts = py_blockSQP.qpOASES_options()
QPopts.terminationTolerance = 1e-10
QPopts.printLevel = 0
opts.QPsol_opts = QPopts
################################






# Scaling heuristic: Scale u such that (γ_u, γ_x = γ Lagrange-gradient difference)
#   ||γ_x||_1/N_x <= ||γ_u||_1/N_u <= 2*||γ_x||_1/N_x

opts.autoScaling = True





################################

vblocks = py_blockSQP.vblock_array(len(OCprob.vBlock_sizes))
for i in range(len(OCprob.vBlock_sizes)):
    vblocks[i] = py_blockSQP.vblock(OCprob.vBlock_sizes[i], OCprob.vBlock_dependencies[i])

prob = Problemspec()
prob.vblocks = vblocks


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
################################
scale_arr = 1.0;
prob_unscaled = prob
prob = py_blockSQP.scaled_Problemspec(prob)
scale = py_blockSQP.double_array(OCprob.nVar)
scale_arr = np.array(scale, copy = False)
scale_arr[:] = 1.0
for i in range(OCprob.ntS):
    OCprob.set_stage_control(scale_arr, i, [uscale])
prob.arr_set_scale(scale)
################################
stats = py_blockSQP.SQPstats("./solver_outputs")
optimizer = py_blockSQP.SQPmethod(prob, opts, stats)
################################

itMax = 100
step_plots = True
plot_title = True

optimizer.init()
t0 = time.time()
if (step_plots):
    OCprob.plot(OCprob.start_point, dpi = 200, it = 0, title=plot_title)
    
    ret = int(optimizer.run(1))
    
    xi = np.array(optimizer.get_xi()).reshape(-1)/scale_arr
    OCprob.plot(xi, dpi = 200, it = 1, title=plot_title)
    i = 1
    while ret == 0 and i < itMax:
        ret = int(optimizer.run(1,1))
        
        
        xi = np.array(optimizer.get_xi()).reshape(-1)/scale_arr
        i += 1
        OCprob.plot(xi, dpi = 200, it = i, title=plot_title)
else:
    ret = int(optimizer.run(itMax))
    xi = np.array(optimizer.get_xi()).reshape(-1)/scale_arr
    OCprob.plot(xi, dpi=200, it = stats.itCount - 1, title=plot_title)
    t1 = time.time()
    print("Solved OCP in ", t1 - t0, "s\n")
################################
