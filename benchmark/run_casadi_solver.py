# py_blockSQP -- A python interface to blockSQP 2, a nonlinear programming
#                solver based on blockSQP by Dennis Janka.
# Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
#
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

itMax = 1000

OCprob = OCProblems.Lotka_Volterra_Fishing(
                    nt = 100,
                    refine = 1,
                    parallel = True,
                    integrator = 'RK4',
                    
                    )

ipopts = dict()
ipopts['hessian_approximation'] = 'exact'
ipopts['tol'] = 1e-6
ipopts['constr_viol_tol'] = 1e-6
ipopts['max_iter'] = itMax

sp = OCprob.start_point
S = cs.nlpsol('S', 'ipopt', OCprob.NLP, {'ipopt':ipopts})


# S = cs.nlpsol('S', 'fatrop', OCprob.NLP, {'structure_detection' : None, "jit" : False, "fatrop.print_level":10, "jit_options": {"flags": "-Os", "verbose": False}})#, 'nx':[len([x for x in OCprob.x_init if x is None])] + [OCprob.nx]*OCprob.ntS, 'nu': [OCprob.nu]*OCprob.ntS + [0], 'ng': [0]*101, 'N':OCprob.ntS, "jit":False, "fatrop.print_level":10, "jit_options": {"flags": "-O3", "verbose": True}})#, {'ipopt':ipopts})
#See fatrop source code - legacy/src/OCPCInterface.cpp for options
# S = cs.nlpsol('S', 'fatrop', OCprob.NLP, {'structure_detection' : 'manual', 'nx':[len([x for x in OCprob.x_init if x is None])] + [OCprob.nx]*OCprob.ntS, 'nu': [OCprob.nu]*OCprob.ntS + [0], 'ng': [0]*101, 'N':OCprob.ntS, "jit":False, "fatrop.print_level":10, "jit_options": {"flags": "-Os", "verbose": False}, \
#                                           'fatrop':{'tol':1e-6, 'acceptable_tol':1e-4}
#                                           })

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