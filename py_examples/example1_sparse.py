# blockSQP 2 -- Condensing, convexification strategies, scaling heuristics and more
#               for blockSQP, the nonlinear programming solver by Dennis Janka.
# Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
# 
# Licensed under the zlib license. See LICENSE for more details.

import sys
from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()
sys.path += [str(cD.parent)]

import py_blockSQP as blockSQP
import numpy as np
import time

opts = blockSQP.SQPoptions()
opts.opt_tol = 1.0e-12
opts.feas_tol = 1.0e-12
opts.enable_linesearch = False
opts.hess_approx = 1
opts.fallback_approx = 2
opts.sizing = 0
opts.fallback_sizing = 0
opts.lim_mem = True
opts.mem_size = 20
opts.block_hess = 1
opts.exact_hess = 0
opts.sparse = True
opts.print_level = 2
opts.debug_level = 0
opts.qpsol = "qpOASES"
opts.indef_delay = 1

stats = blockSQP.SQPstats("./")

prob = blockSQP.Problemspec()
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
