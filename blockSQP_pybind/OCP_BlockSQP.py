import numpy as np
import BlockSQP
from Py_BlockSQP import BlockSQP_Problem
import OCProblems

######################
##Available problems##
######################

OCprob = OCProblems.Lotka_Volterra_Fishing(Tgrid = np.linspace(0,12,101,endpoint=True))
# OCprob = OCProblems.Lotka_Shared_Resources(x_init = [1.5,0.5,1.0])
# OCprob = OCProblems.Lotka_Competitive(x_init = [0.5, 1.5])
# OCprob = OCProblems.Three_Tank_Multimode()

# OCprob = OCProblems.Goddart_Rocket(nt = 200)
#NOTE: Use fallbackScaling = 2 or exact hessian (whichSecondDerv = 2)

# OCprob = OCProblems.Hanging_Chain(Tgrid = np.linspace(0,1,201,endpoint=True))
#NOTE: Using exact hessian (opts.whichSecondDerv = 2) or downscaled initial hessian (opts.iniHessDiag = 1e-2) recommended


prob = BlockSQP_Problem()
prob.nVar = OCprob.nVar
prob.nCon = OCprob.nCon

prob.f = OCprob.f
prob.grad_f = OCprob.grad_f
prob.g = OCprob.g
prob.make_sparse(OCprob.jac_g_nnz, OCprob.jac_g_row, OCprob.jac_g_colind)
prob.jac_g_nz = OCprob.jac_g_nz
prob.hess = OCprob.hess_lag

prob.set_blockIndex(OCprob.hessBlock_index)
prob.set_bounds(OCprob.lb_var, OCprob.ub_var, OCprob.lb_con, OCprob.ub_con)

prob.x_start = OCprob.start_point

prob.lam_start = np.zeros(prob.nVar + prob.nCon, dtype = np.float64).reshape(-1)

prob.complete()


opts = BlockSQP.SQPoptions();
opts.maxItQP = 100000000
opts.maxConvQP = 1
opts.restoreFeas = 1
opts.maxTimeQP = 4
opts.hessMemsize = 20
opts.iniHessDiag = 1.0
opts.hessUpdate = 1
opts.hessScaling = 2
opts.fallbackUpdate = 2
opts.fallbackScaling = 4

opts.whichSecondDerv = 0

opts.hessLimMem = 1
opts.hessMemsize = 20
opts.opttol = 1e-6
opts.nlinfeastol = 1e-6
opts.skipFirstGlobalization = 0
opts.maxLineSearch = 7
opts.max_bound_refines = 3
opts.max_correction_steps = 6

###################
opts.which_QPsolver = 'qpOASES'
###################

####################
###gurobi options###
####################
# opts.gurobi_OptimalityTol = 1e-9
# opts.gurobi_FeasibilityTol = 1e-9
# opts.gurobi_NumericFocus = 3
# opts.gurobi_Method = 1
# opts.gurobi_BarHomogeneous = 1
# opts.gurobi_OutputFlag = 0

#####################
###qpOASES options###
#####################
opts.qpOASES_terminationTolerance = 1e-10
opts.qpOASES_printLevel = 0

#####################
stats = BlockSQP.SQPstats("./solver_outputs")
optimizer = BlockSQP.SQPmethod(prob, opts, stats)
optimizer.init()
#####################
#Standard run

ret = optimizer.run(500)
xi = np.array(optimizer.vars.xi).reshape(-1)
OCprob.plot(xi)
#####################
#Plot all steps

# OCprob.plot(OCprob.start_point)
# ret = optimizer.run(1)
# while ret != 0 and ret != -1:
#     ret = optimizer.run(1,1)
#     xi = np.array(optimizer.vars.xi).reshape(-1)
#     OCprob.plot(xi)



