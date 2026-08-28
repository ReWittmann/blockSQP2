# blockSQP2 -- A structure-exploiting nonlinear programming solver based
#              on blockSQP by Dennis Janka.
# Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>

# Licensed under the zlib license. See LICENSE for more details.


# \file run_casadi_solver.py
# \author Reinhold Wittmann
# \date 2025
#
# Script to invoke a solver available through casadi,
# for comparing the performance to py_blockSQP.

import casadi as cs
import numpy as np
import OCProblems
import time

import sys
from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()
sys.path += [str(cD/Path("experiments"))]
import OCProblems_fatrop

itMax = 500

OCprob = OCProblems.Satellite_Deorbiting_2(
                    nt = 100,
                    refine = 1,
                    # integrator = 'RK4',
                    parallel = True,
                    N_threads = 4, 
                    # **OCProblems.Cart_Pendulum.param_set_2,
                    )

ipopts = dict()
ipopts['hessian_approximation'] = 'exact'
ipopts['tol'] = 1e-6
ipopts['constr_viol_tol'] = 1e-6
ipopts['max_iter'] = itMax

sp = OCprob.start_point


S = cs.nlpsol('S', 'ipopt', OCprob.NLP, {'ipopt':ipopts, 'jit': False})

#For fatrop, see "run_fatrop_solver.py"


# S = cs.nlpsol('S', 'sqpmethod', OCprob.NLP)

# worhp_opts = {}#'TolOpti':1e-9}
# worhp_opts = {'TolOpti':1e-6, 'ScaledKKT':False}#'TolOpti':1e-9}
# worhp_opts = {
#     'BFGSmethod' : 100,
#     'BFGSmaxblockSize': 20,
#     'UserHM' : False,
#     'TolOpti': 1e-6,
#     'ScaledKKT' : False,
#     'FidifHM' : False
#     }

# S = cs.nlpsol('S', 'worhp', OCprob.NLP, {'worhp':worhp_opts})

# blocksqp_opts = {'linsol':'ma27', 'warmstart':False}
# blocksqp_opts = {}
# S = cs.nlpsol('S', 'blocksqp', OCprob.NLP, blocksqp_opts)

t0 = time.monotonic()
out = S(x0=sp, lbx=OCprob.lb_var,ubx=OCprob.ub_var, lbg=OCprob.lb_con, ubg=OCprob.ub_con)
t1 = time.monotonic()
stats = S.stats()

xi = out['x']
OCprob.plot(np.array(xi).reshape(-1), dpi=200)

time.sleep(0.1)
print(t1 - t0, "s")