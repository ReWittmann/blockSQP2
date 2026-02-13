# blockSQP2 -- A structure-exploiting nonlinear programming solver based
#              on blockSQP by Dennis Janka.
# Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>

# Licensed under the zlib license. See LICENSE for more details.


import sys
from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()
sys.path += [str(cD.parent)]

import blockSQP2
import numpy as np
import time

opts = blockSQP2.SQPoptions()
opts.opt_tol = 1.0e-12
opts.feas_tol = 1.0e-12
opts.enable_linesearch = 0
opts.hess_approx = 'BFGS'
opts.fallback_approx = 'BFGS'
opts.sizing = 'None'
opts.fallback_sizing = 'None'
opts.lim_mem = 1
opts.mem_size = 20
opts.block_hess = 1
opts.sparse = False
opts.print_level = 2
opts.debug_level = 0
opts.qpsol = "qpOASES"


stats = blockSQP2.SQPstats("./")

prob = blockSQP2.Problemspec()
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


meth = blockSQP2.SQPmethod(prob, opts, stats)
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
