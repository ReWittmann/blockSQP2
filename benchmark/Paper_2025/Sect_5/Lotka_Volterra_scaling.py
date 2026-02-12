import numpy as np
import time
import sys
from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()
sys.path += [str(cD.parents[1]), str(cD.parents[2]/Path("Python"))]
import blockSQP2
import matplotlib.pyplot as plt

itMax = 100
step_plots = True
plot_title = True

import OCProblems
OCprob = OCProblems.Lotka_Volterra_Fishing(
                    nt = 100, 
                    refine = 1, 
                    parallel = True, 
                    integrator = 'RK4'
                    )

################################
opts = blockSQP2.SQPoptions()
opts.max_QP_it = 10000
opts.max_QP_secs = 5.0

opts.max_conv_QPs = 1
opts.conv_strategy = 2
opts.par_QPs = False
opts.enable_QP_cancellation = True
opts.indef_delay = 1

opts.hess_approx = 'SR1'
opts.sizing = 'OL'
opts.fallback_approx = 'BFGS'
opts.fallback_sizing = 'COL'
opts.BFGS_damping_factor = 1/3

opts.lim_mem = True
opts.mem_size = 20
opts.opt_tol = 1e-6
opts.feas_tol = 1e-6

opts.automatic_scaling = False

opts.max_extra_steps = 0
opts.enable_premature_termination = False
opts.max_filter_overrides = 0


vBlocks = blockSQP2.vblock_array(len(OCprob.vBlock_sizes))
for i in range(len(OCprob.vBlock_sizes)):
    vBlocks[i] = blockSQP2.vblock(OCprob.vBlock_sizes[i], OCprob.vBlock_dependencies[i])


#Define blockSQP Problemspec
prob = blockSQP2.Problemspec()
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

prob.vblocks = vBlocks

prob.x_start = OCprob.start_point
prob.lam_start = np.zeros(prob.nVar + prob.nCon, dtype = np.float64).reshape(-1)
prob.complete()

stats = blockSQP2.SQPstats("./solver_outputs")
#No condensing
optimizer = blockSQP2.SQPmethod(prob, opts, stats)
optimizer.init()

ret = optimizer.run(10)
xi10 = np.array(optimizer.get_xi()).reshape(-1)
OCprob.plot(xi10, dpi=200)

ret = optimizer.run(90,1)
xi = np.array(optimizer.get_xi()).reshape(-1)
OCprob.plot(xi, dpi=200)


### Scale x1 by 100.0, x2 by 0.01 ###
prob2 = blockSQP2.scaled_Problemspec(prob)
scale = blockSQP2.double_array(OCprob.nVar)
scale_arr = np.array(scale, copy = False)
scale_arr[:] = 1.0
for i in range(OCprob.ntS + 1):
    OCprob.set_stage_state(scale_arr, i, [100.0, 0.01])
prob2.arr_set_scale(scale)
stats_100_001 = blockSQP2.SQPstats("./solver_outputs")
optimizer2 = blockSQP2.SQPmethod(prob2, opts, stats_100_001)
optimizer2.init()
ret = optimizer2.run(100)

### Vice versa ###
prob3 = blockSQP2.scaled_Problemspec(prob)
scale = blockSQP2.double_array(OCprob.nVar)
scale_arr = np.array(scale, copy = False)
scale_arr[:] = 1.0
for i in range(OCprob.ntS + 1):
    OCprob.set_stage_state(scale_arr, i, [0.01, 100.0])
prob3.arr_set_scale(scale)
stats_001_100 = blockSQP2.SQPstats("./solver_outputs")
optimizer3 = blockSQP2.SQPmethod(prob3, opts, stats_001_100)
optimizer3.init()
ret = optimizer3.run(100)


### u scaled by 10 ###
prob4 = blockSQP2.scaled_Problemspec(prob)
scale = blockSQP2.double_array(OCprob.nVar)
scale_arr = np.array(scale, copy = False)
scale_arr[:] = 1.0
for i in range(OCprob.ntS):
    OCprob.set_stage_control(scale_arr, i, 10.0)
prob4.arr_set_scale(scale)
stats = blockSQP2.SQPstats("./solver_outputs")
optimizer4 = blockSQP2.SQPmethod(prob4, opts, stats)
optimizer4.init()
ret = optimizer4.run(10)
xi = np.array(optimizer4.get_xi()).reshape(-1)

x1,x2 = OCprob.get_state_arrays(xi)
u = OCprob.get_control_plot_arrays(xi)
fig, ax = plt.subplots(dpi=200)
ax.plot(OCprob.time_grid, x1, 'tab:green', linestyle='-.', label = '$x_1$')
ax.plot(OCprob.time_grid, x2, 'tab:blue', linestyle='--', label = '$x_2$')
ax.step(OCprob.time_grid_ref, u/10, 'tab:red', linestyle='-', label = r'$u\cdot 0.1$')
ax.legend(fontsize='large')
ax.set_xlabel('t', fontsize = 17.5)
ax.xaxis.set_label_coords(1.015,-0.006)
plt.show()
plt.close()

ret = optimizer4.run(90,1)


### u scaled by 100 ###
prob5 = blockSQP2.scaled_Problemspec(prob)
scale = blockSQP2.double_array(OCprob.nVar)
scale_arr = np.array(scale, copy = False)
scale_arr[:] = 1.0
for i in range(OCprob.ntS + 1):
    OCprob.set_stage_control(scale_arr, i, 100.0)
prob5.arr_set_scale(scale)
stats = blockSQP2.SQPstats("./solver_outputs")
optimizer5 = blockSQP2.SQPmethod(prob5, opts, stats)
optimizer5.init()
ret = optimizer5.run(10)
xi = np.array(optimizer5.get_xi()).reshape(-1)

x1,x2 = OCprob.get_state_arrays(xi)
u = OCprob.get_control_plot_arrays(xi)
fig, ax = plt.subplots(dpi=200)
ax.plot(OCprob.time_grid, x1, 'tab:green', linestyle='-.', label = '$x_1$')
ax.plot(OCprob.time_grid, x2, 'tab:blue', linestyle='--', label = '$x_2$')
ax.step(OCprob.time_grid_ref, u/100, 'tab:red', linestyle='-', label = r'$u\cdot 0.01$')
ax.legend(fontsize='large')
ax.set_xlabel('t', fontsize = 17.5)
ax.xaxis.set_label_coords(1.015,-0.006)
plt.show()
plt.close()

ret = optimizer5.run(90,1)


time.sleep(0.25)
print("#################################################################")
print("With x1, x2 scaled by 100, 0.01, the number of iterations was ", stats_100_001.itCount)
print("With x1, x2 scaled by 0.01, 100, the number of iterations was ", stats_001_100.itCount)
print("#################################################################")

