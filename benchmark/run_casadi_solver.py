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

itMax = 1000

OCprob = OCProblems.Goddard_Rocket_noParams(
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

sp = OCprob.perturbed_start_point(0)


# S = cs.nlpsol('S', 'ipopt', OCprob.NLP, {'ipopt':ipopts})

# S = cs.nlpsol('S', 'fatrop', OCprob.NLP, {'structure_detection' : None, "jit" : False, "fatrop.print_level":10, "jit_options": {"flags": "-Os", "verbose": False}})#, 'nx':[len([x for x in OCprob.x_init if x is None])] + [OCprob.nx]*OCprob.ntS, 'nu': [OCprob.nu]*OCprob.ntS + [0], 'ng': [0]*101, 'N':OCprob.ntS, "jit":False, "fatrop.print_level":10, "jit_options": {"flags": "-O3", "verbose": True}})#, {'ipopt':ipopts})
#See fatrop source code - c_interface/src/OCPCInterface.cpp for options


# def reorder_constr_for_fatrop(constr_expr, lb_con, ub_con, ntS, nx, n_path_constr, n_term_constr):
#     if n_path_constr == 0:
#         return constr_expr, lb_con, ub_con
    
#     match_arr = []
#     lb_match_arr = []
#     ub_match_arr = []
    
#     path_constr_arr = []
#     lb_path_constr_arr = []
#     ub_path_constr_arr = []
    
#     offset = 0
#     for i in range(ntS):
#         match_arr.append(constr_expr[offset:offset+nx])
#         lb_match_arr.append(lb_con[offset:offset+nx])
#         ub_match_arr.append(ub_con[offset:offset+nx])
#         offset += nx
#     for i in range(ntS):
#         path_constr_arr.append(constr_expr[offset:offset+n_path_constr])
#         lb_path_constr_arr.append(lb_con[offset:offset+n_path_constr])
#         ub_path_constr_arr.append(ub_con[offset:offset+n_path_constr])
#         offset += n_path_constr
#     term_constr = constr_expr[offset:offset+n_term_constr]
#     lb_term_constr = lb_con[offset:offset+n_term_constr]
#     ub_term_constr = ub_con[offset:offset+n_term_constr]
    
    
#     constr_arr = []
#     lb_con_arr = []
#     ub_con_arr = []
#     constr_arr.append(match_arr[0])
#     lb_con_arr.append(lb_match_arr[0])
#     ub_con_arr.append(ub_match_arr[0])
#     for i in range(ntS-1):
#         constr_arr.append(match_arr[1+i])
#         lb_con_arr.append(lb_match_arr[1+i])
#         ub_con_arr.append(ub_match_arr[1+i])
        
#         constr_arr.append(path_constr_arr[i])
#         lb_con_arr.append(lb_path_constr_arr[i])
#         ub_con_arr.append(ub_path_constr_arr[i])
        
#     constr_arr.append(path_constr_arr[ntS-1])
#     lb_con_arr.append(lb_path_constr_arr[ntS-1])
#     ub_con_arr.append(ub_path_constr_arr[ntS-1])
    
#     constr_arr.append(term_constr)
#     lb_con_arr.append(lb_term_constr)
#     ub_con_arr.append(ub_term_constr)
    
#     return cs.vertcat(*constr_arr), np.concatenate(lb_con_arr), np.concatenate(ub_con_arr)


def reorder_constr_for_fatrop(constr_expr, lb_con, ub_con, ntS, nx, n_path_constr, n_term_constr, path_constr_0 = False, path_constr_F = True):
    if n_path_constr == 0:
        return constr_expr, lb_con, ub_con
    
    match_arr = []
    lb_match_arr = []
    ub_match_arr = []
    
    path_constr_arr = []
    lb_path_constr_arr = []
    ub_path_constr_arr = []
    
    offset = 0
    for i in range(ntS):
        match_arr.append(constr_expr[offset:offset+nx])
        lb_match_arr.append(lb_con[offset:offset+nx])
        ub_match_arr.append(ub_con[offset:offset+nx])
        offset += nx
    for i in range(ntS + int(path_constr_0)):
        path_constr_arr.append(constr_expr[offset:offset+n_path_constr])
        lb_path_constr_arr.append(lb_con[offset:offset+n_path_constr])
        ub_path_constr_arr.append(ub_con[offset:offset+n_path_constr])
        offset += n_path_constr
    term_constr = constr_expr[offset:offset+n_term_constr]
    lb_term_constr = lb_con[offset:offset+n_term_constr]
    ub_term_constr = ub_con[offset:offset+n_term_constr]
    
    
    constr_arr = []
    lb_con_arr = []
    ub_con_arr = []
    constr_arr.append(match_arr[0])
    lb_con_arr.append(lb_match_arr[0])
    ub_con_arr.append(ub_match_arr[0])
    if path_constr_0:
        constr_arr.append(path_constr_arr[0])
        lb_con_arr.append(lb_path_constr_arr[0])
        ub_con_arr.append(ub_path_constr_arr[0])
    
    for i in range(ntS-1):
        constr_arr.append(match_arr[1+i])
        lb_con_arr.append(lb_match_arr[1+i])
        ub_con_arr.append(ub_match_arr[1+i])
        
        constr_arr.append(path_constr_arr[int(path_constr_0) + i])
        lb_con_arr.append(lb_path_constr_arr[int(path_constr_0) + i])
        ub_con_arr.append(ub_path_constr_arr[int(path_constr_0) + i])
        
    constr_arr.append(path_constr_arr[int(path_constr_0) + ntS-1])
    lb_con_arr.append(lb_path_constr_arr[int(path_constr_0) + ntS-1])
    ub_con_arr.append(ub_path_constr_arr[int(path_constr_0) + ntS-1])
    
    constr_arr.append(term_constr)
    lb_con_arr.append(lb_term_constr)
    ub_con_arr.append(ub_term_constr)
    
    return cs.vertcat(*constr_arr), np.concatenate(lb_con_arr), np.concatenate(ub_con_arr)

g_expr = OCprob.NLP['g']

g_expr_new, lb_con_new, ub_con_new = reorder_constr_for_fatrop(g_expr, OCprob.lb_con, OCprob.ub_con, 100, 4, 1, 1)
OCprob.NLP['g'] = g_expr_new

# S = cs.nlpsol('S', 'ipopt', OCprob.NLP, {'ipopt':ipopts})

S = cs.nlpsol('S', 'fatrop', OCprob.NLP, {'structure_detection' : 'manual', 
                                          'nx':[len([x for x in OCprob.x_init if x is None])] + [OCprob.nx]*OCprob.ntS, 'nu': [OCprob.nu]*OCprob.ntS + [0], 'ng': [0] + [1]*(OCprob.ntS-1) + [1 + 1], 'N':OCprob.ntS, 
                                          "jit":False,
                                          "expand": False,
                                          "jit_options": {"flags": "-Os", "verbose": False},
                                          "fatrop.print_level":10,
                                          'fatrop':{'tol':1e-6, 'constr_viol_tol':1e-4},
                                          'debug': False
                                          })


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
out = S(x0=sp, lbx=OCprob.lb_var,ubx=OCprob.ub_var, lbg = lb_con_new, ubg = ub_con_new)#, lbg=OCprob.lb_con, ubg=OCprob.ub_con)
t1 = time.monotonic()
stats = S.stats()

xi = out['x']
OCprob.plot(np.array(xi).reshape(-1), dpi=200)

time.sleep(0.1)
print(t1 - t0, "s")