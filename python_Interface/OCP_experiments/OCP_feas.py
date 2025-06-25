import numpy as np
import sys
import time
import os
import casadi as cs
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
except:
    sys.path.append(os.getcwd() + "/..")

import py_blockSQP
from blockSQP_pyProblem import blockSQP_pyProblem as Problemspec
import matplotlib.pyplot as plt


itMax = 100

step_plots = True
plot_title = True


import OCProblems

Lprob = OCProblems.Gravity_Turn(nt = 100, parallel = True, integrator = 'rk4')

#NOTE: use fallbackScaling = 4

# Lprob = OCProblem.Lotka_Volterra_Fishing(Tgrid = np.linspace(0,12,101,endpoint=True))
# Lprob = OCProblem.Lotka_Shared_Resources()
# Lprob = OCProblem.Lotka_Competitive()
# Lprob = OCProblem.Three_Tank_Multimode()
# Lprob = OCProblem.Goddart_Rocket()

x = Lprob.NLP['x']
g_expr = Lprob.NLP['g']
g = cs.Function('g', [x], [g_expr])

s = cs.MX.sym('s', g_expr.numel())
feas_var = cs.vertcat(x, s)
feas_g_expr = g_expr - s
feas_g = cs.Function('feas_g', [feas_var], [feas_g_expr])
feas_jac_g_expr = cs.jacobian(feas_g_expr, feas_var)
feas_jac_g = cs.Function('feas_jac_g', [feas_var], [feas_jac_g_expr])


feas_hessBlock_index = np.concatenate((Lprob.hessBlock_index, Lprob.hessBlock_index[-1] + 1 + np.array(range(g_expr.numel()))), dtype = np.int32)
feas_obj_expr = cs.sumsqr(s)
feas_obj = cs.Function('feas_obj', [feas_var], [feas_obj_expr])
feas_grad_expr = cs.jacobian(feas_obj_expr, feas_var)
feas_grad = cs.Function('feas_grad', [feas_var], [feas_grad_expr])

g0 = g(Lprob.start_point)
s_start = cs.DM.zeros(g_expr.numel())
for j in range(g_expr.numel()):
    if (g0[j] < Lprob.lb_con[j]):
        s_start[j] = g0[j] - Lprob.lb_con[j]
    
    if (g0[j] > Lprob.ub_con[j]):
        s_start[j] = g0[j] - Lprob.ub_con[j];


feas_start = cs.vertcat(Lprob.start_point, s_start)

feas_lb_var = np.concatenate((Lprob.lb_var, np.array([-np.inf] * g_expr.numel())), dtype = np.float64).reshape(-1)
feas_ub_var = np.concatenate((Lprob.ub_var, np.array([np.inf] * g_expr.numel())), dtype = np.float64).reshape(-1)

feas_jac_g_0 = feas_jac_g(feas_start)



prob = Problemspec()
prob.nVar = Lprob.nVar + Lprob.nCon
prob.nCon = Lprob.nCon

prob.f = lambda x: np.array(feas_obj(x), dtype = np.float64).reshape(-1)
prob.grad_f = lambda x: np.array(feas_grad(x), dtype = np.float64).reshape(-1)
prob.g = lambda x: np.array(feas_g(x), dtype = np.float64).reshape(-1)
prob.make_sparse(feas_jac_g_0.nnz(), np.array(feas_jac_g_0.row(), dtype = np.int32).reshape(-1), np.array(feas_jac_g_0.colind(), dtype = np.int32).reshape(-1))
prob.jac_g_nz = lambda x: np.array(feas_jac_g(x).nz[:], dtype = np.float64).reshape(-1)

prob.set_blockIndex(feas_hessBlock_index)
prob.set_bounds(feas_lb_var, feas_ub_var, Lprob.lb_con, Lprob.ub_con)

# L = np.load('lotka_feas_acc.npz')
# prob.x_start = L['arr_0.npy'].reshape(-1)

prob.x_start = np.array(feas_start, dtype = np.float64).reshape(-1)
prob.lam_start = np.zeros(prob.nVar + prob.nCon, dtype = np.float64).reshape(-1)

prob.complete()

# feas_prob = BlockSQP.feasibility_problem(prob)

opts = py_blockSQP.SQPoptions()
opts.max_QP_it = 10000
opts.max_QP_secs = 5.0

opts.max_conv_QPs = 1
opts.conv_strategy = 0
opts.exact_hess = 0
opts.hess_approx = 1
opts.sizing = 2
opts.fallback_approx = 2
opts.fallback_sizing = 4
opts.BFGS_damping_factor = 1/3
opts.test_opt_1 = False
opts.test_opt_2 = True
opts.test_qp_hotstart = 0

opts.lim_mem = True
opts.mem_size = 20
opts.opt_tol = 1e-6
opts.feas_tol = 1e-6
opts.conv_kappa_max = 8.0

opts.automatic_scaling = False

opts.max_extra_steps = 0
opts.enable_premature_termination = True
opts.max_filter_overrides = 0

opts.max_QP_secs = 5.0

####################
###gurobi options###
####################
# opts.gurobi_OptimalityTol = 1e-9
# opts.gurobi_FeasibilityTol = 1e-9
# opts.gurobi_NumericFocus = 3
# opts.gurobi_Method = 1
# opts.gurobi_BarHomogeneous = 1
# # opts.gurobi_Presolve = 0
# # opts.gurobi_Aggregate = 0
# opts.gurobi_OutputFlag = 0

#####################
###qpOASES options###
#####################

stats = py_blockSQP.SQPstats("./solver_outputs")
optimizer = py_blockSQP.SQPmethod(prob, opts, stats)
optimizer.init()



ret = optimizer.run(200)


x_feas = np.array(optimizer.vars.xi, dtype = np.float64).reshape(-1)[0:Lprob.nVar]
s_feas = np.array(optimizer.vars.xi, dtype = np.float64).reshape(-1)[Lprob.nVar:]


# dt_feas = x_feas[0::5]
# u_feas = x_feas[1::5]

# feas_param = cs.vertcat(cs.DM(dt_feas).T, cs.DM(u_feas).T)
############
# Lprob.plot(Lprob.start_point)
# ret = optimizer.run(1)
# while ret != 0 and ret != -1:
#     ret = optimizer.run(1,1)
#     xi = np.array(optimizer.vars.xi).reshape(-1)
#     Lprob.plot(xi)
    
#################
# t0 = time.time()
# ret = optimizer.run(20)

# xi = np.array(optimizer.vars.xi).reshape(-1)
# Lprob.plot(xi)

# ret = optimizer.run(80,1)
# xi = np.array(optimizer.vars.xi).reshape(-1)
# Lprob.plot(xi)

# ret = optimizer.run(100,1)
# xi = np.array(optimizer.vars.xi).reshape(-1)
# Lprob.plot(xi)

# ret = optimizer.run(100,1)
# xi = np.array(optimizer.vars.xi).reshape(-1)
# Lprob.plot(xi)

# ret = optimizer.run(100,1)
# xi = np.array(optimizer.vars.xi).reshape(-1)
# Lprob.plot(xi)

# t1 = time.time()
# print("finished iterations after ", t1 - t0, " seconds\n")






