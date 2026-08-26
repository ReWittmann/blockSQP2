#Python version of blockSQP example1.cpp (sparse variant)
import py_blockSQP_old as py_blockSQP
import numpy as np
import time

opts = py_blockSQP.SQPoptions();
opts.opttol = 1.0e-12
opts.nlinfeastol = 1.0e-12
opts.globalization = 0
opts.hessUpdate = 1
opts.hessScaling = 2
opts.fallbackScaling = 4
opts.hessLimMem = 1
opts.maxConsecSkippedUpdates = 200
opts.blockHess = 1
opts.whichSecondDerv = 0
opts.sparseQP = 2


stats = py_blockSQP.SQPstats("./")

prob = py_blockSQP.Problemspec()
prob.nVar = 2
prob.nCon = 1
prob.set_blockIndex(np.array([0,1,2], dtype = np.int32))
prob.set_bounds([-np.inf, -np.inf], [np.inf, np.inf], [0.], [0.])
#######
prob.x_start = [10.,10.]
prob.lam_start = [0.,0.,0.]
#######
prob.f = lambda x: x[0]**2 - 0.5*x[1]**2
prob.g = lambda x: x[0] - x[1]
prob.grad_f = lambda x: [2*x[0], -x[1]]
prob.jac_g_nz = lambda x: np.array([1,-1])
#######
jac_g_nnz = 2
jac_g_ind_row = np.array([0,0])
jac_g_ind_col = np.array([0,1,2])

prob.make_sparse(jac_g_nnz, jac_g_ind_row, jac_g_ind_col)

prob.complete()


meth = py_blockSQP.SQPmethod(prob, opts, stats)
meth.init()
time.sleep(0.01)
print("starting run")

ret = meth.run(100)
meth.finish()

time.sleep(0.25)
print("\nPrimal solution:\n")
print(np.array(meth.vars.xi))
print("\nDual solution:\n")
print(np.array(meth.vars.lam))
