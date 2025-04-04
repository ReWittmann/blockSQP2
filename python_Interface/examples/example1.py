import os
import sys
sys.path.append(os.path.abspath('') + "/..")

import py_blockSQP as blockSQP
from blockSQP_pyProblem import blockSQP_pyProblem as Problemspec
import numpy as np
import time

opts = blockSQP.SQPoptions()
opts.optimality_tol = 1.0e-12
opts.feasibility_tol = 1.0e-12
opts.enable_linesearch = 0
opts.hess_approximation = 2
opts.fallback_approximation = 2
opts.sizing_strategy = 0
opts.fallback_sizing_strategy = 0
opts.limited_memory = 1
opts.memory_size = 20
opts.max_consec_skipped_updates = 200
opts.block_hess = 1
opts.exact_hess_usage = 0
opts.sparse_mode = 0
opts.print_level = 2
opts.debug_level = 0
opts.qpsol = "qpOASES"


stats = blockSQP.SQPstats("./")

prob = Problemspec()
prob.nVar = 2
prob.nCon = 1
prob.set_blockIndex(np.array([0,1,2],dtype = np.int32))
prob.set_bounds([-np.inf, -np.inf], [np.inf, np.inf], [0.], [0.])
#######
prob.x_start = [10.,10.]
prob.lam_start = [0.,0.,0.]
#######
prob.f = lambda x: x[0]**2 - 0.5*x[1]**2
prob.g = lambda x: x[0] - x[1]
prob.grad_f = lambda x: [2*x[0], -x[1]]
prob.jac_g = lambda x: [[1,-1]]
#######
prob.complete()


meth = blockSQP.SQPmethod(prob, opts, stats)
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
