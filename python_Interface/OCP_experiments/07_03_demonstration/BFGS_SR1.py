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

#See mintoc.de for problem definitions
#1.
# OCprob = OCProblems.Goddard_Rocket(nt = 100, parallel = False)

#2.
# OCprob = OCProblems.Egerstedt_Standard(nt = 100, parallel = False)

#3. 
# OCprob = OCProblems.Three_Tank_Multimode(nt = 100, parallel = False)

#4.
# OCprob = OCProblems.Catalyst_Mixing(nt=100, parallel = False)

#5.
# OCprob = OCProblems.Goddard_Rocket(nt = 100, parallel = False)
# OCprob.set_stage_control(OCprob.start_point, 13, 0.9)


#
# OCprob = OCProblems.Batch_Reactor(nt = 100, parallel = False)

################################
opts = py_blockSQP.SQPoptions()
opts.maxItQP = 100000
opts.maxTimeQP = 5.0

# opts.whichSecondDerv = 2
opts.maxConvQP = 1          #SR1 - BFGS
opts.hessUpdate = 1         #SR1
opts.hessScaling = 2        #Oren-Luenberger sizing
opts.fallbackUpdate = 2     #damped BFGS
opts.fallbackScaling = 4    #centered Oren-Luenberger sizing

opts.opttol = 1e-6
opts.nlinfeastol = 1e-6

opts.allow_premature_termination = False

opts.QPsol = 'qpOASES'
QPopts = py_blockSQP.qpOASES_options()
QPopts.terminationTolerance = 1e-10
QPopts.printLevel = 0
opts.QPsol_opts = QPopts
################################

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
prob.complete()

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
    
    xi = np.array(optimizer.get_xi()).reshape(-1)
    OCprob.plot(xi, dpi = 200, it = 1, title=plot_title)
    i = 1
    while ret == 0 and i < itMax:
        ret = int(optimizer.run(1,1))
        
        
        xi = np.array(optimizer.get_xi()).reshape(-1)
        i += 1
        OCprob.plot(xi, dpi = 200, it = i, title=plot_title)
else:
    ret = int(optimizer.run(itMax))
    xi = np.array(optimizer.get_xi()).reshape(-1)
    OCprob.plot(xi, dpi=200, it = stats.itCount - 1, title=plot_title)
    t1 = time.time()
    print("Solved OCP in ", t1 - t0, "s\n")
################################
