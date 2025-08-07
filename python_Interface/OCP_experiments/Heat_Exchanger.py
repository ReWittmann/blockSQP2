import casadi as cs
import numpy as np

import os
import sys
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
except:
    sys.path.append(os.getcwd() + "/..")

import py_blockSQP
import matplotlib.pyplot as plt
import time
################
## Parameters ##
################

# Physical constants
mh_total     = 100                       # Total mass of milk (hot fluid) that is to be cooled down [kg]
Cpc          = 4180                      # Specific heat capacity of water (cold fluid) [J/kg*K]
Cph          = 3700                      # Specific heat capacity of milk (hot fluid) [J/kg*K]
D            = 15e-3                     # Tube diameter [m]
L            = 2                         # Tube length [m]
N_tubes      = 20                        # Number of tubes [-]
U            = 500                       # Overall heat transfer coefficient [W/(m^2*K)]
A            = np.pi * D * L * N_tubes   # Heat transfer area [m^2]

# Boundary conditions
mc_0         = 0                         # Accumulated water (cold fluid) mass used at start of simulation [kg]
Tc_in        = 10                        # Temperature of water (cold fluid) at the inlet [°C]
Th_in        = 75                        # Temperature of milk (hot fluid) at the inlet [°C]
Th_out       = 35                        # Temperature of milk (hot fluid) at the outlet [°C]
T_env        = 20                        # Room temperature [°C]

# Constraints
mc_dot_max   = 7                         # Max. mass flow rate of water (cold fluid) [kg/s]
mh_dot_max   = 2                         # Max. mass flow rate of milk (hot fluid) [kg/s]
delta_T_min  = 3                         # Minimum temperature difference [°C]

##################
## Optimization ##
##################

# Time discretization
N      = 100                             # Number of control intervals [-]
tf     = mh_total / mh_dot_max           # Final time (t ∈ [0, tf]) [s]
dt     = tf / N                          # Time step [s] 
t_ramp = 5                               # Ramp-up time [s]

# Create optimization variables

# State variables
x_arr = [[mc_0, Tc_in, T_env]]
u_arr = []
s_arr = [None]
for i in range(N):
    x_arr.append(cs.MX.sym(f"x_{i+1}", 3))
    u_arr.append(cs.MX.sym(f"u_{i}", 1))
    s_arr.append(cs.MX.sym(f"s_{i+1}", 1))

# Softplus is a smooth approximation of the ReLU (Rectified Linear Unit) function.
# It outputs a value that is always positive and smoothes the transition between negative and positive values
def softplus(x, beta=30):
    return cs.if_else(x > 0,
                      x + (1/beta) * cs.log1p(cs.exp(-beta * x)),
                      (1/beta) * cs.log1p(cs.exp(beta * x)))

# Linear ramp function (using a simple linear interpolation)
def linear_ramp(t, t_end, T_start, T_end):
    t = cs.fmin(t, t_end)
    return T_start + (T_end - T_start) * (t / t_end)

def system_dynamics(X, u):
    _, Tc, Th = cs.vertsplit(X)
    mc_dot      =  mc_dot_max * (softplus(u) / softplus(1))  # Smoothly map u ∈ [0,1] to mc_dot ∈ [0, mc_dot_max] without discontinuities
    mh_dot      =  mh_dot_max
    delta_T_eff =  softplus(Th - Tc - delta_T_min)
    Tc_dot      =  (U * A / (mc_dot * Cpc)) * delta_T_eff
    Th_dot      = -(U * A / (mh_dot * Cph)) * delta_T_eff
    return cs.vertcat(mc_dot, Tc_dot, Th_dot)

# Initialize the temperature lock flag (initially, not locked (0 = not locked))
temperature_locked = cs.MX(0)  

# RK4 integration (Euler wasn't fit for larger timesteps)
con_arr = []
lb_con_arr = []
ub_con_arr = []
lb_var_arr = []
ub_var_arr = []

x_opt_arr = []
x_start_arr = []
hsize_arr = []
###################################
u_0 = u_arr[0]
x_0 = x_arr[0]
time_k = 0*dt

x_opt_arr.append(u_0)
lb_var_arr.append(0.)
ub_var_arr.append(1.0)
x_start_arr.append(0.)
hsize_arr.append(1)

mc_0, Tc_0, Th_0 = cs.vertsplit(x_0)
x_kp1 = x_arr[1]

k1_x = dt*system_dynamics(x_0, u_0)
k2_x = dt*system_dynamics(x_0 + 0.5*k1_x, u_0)
k3_x = dt*system_dynamics(x_0 + 0.5*k2_x, u_0)
k4_x = dt*system_dynamics(x_0 + k3_x, u_0)
x_kp1_ = x_0 + 1./6.*(1*k1_x + 2*k2_x + 2*k3_x + 1*k4_x)

Th_desired = linear_ramp(time_k + dt, t_ramp, T_env, Th_in)

# Constraints
con_arr.append(x_kp1 - cs.vertcat(x_kp1_[:-1], Th_desired))
lb_con_arr.extend([0.]*3)
ub_con_arr.extend([0.]*3)

for k in range(1,N):
    u_k    = u_arr[k]
    time_k = k * dt
    
    x_k = x_arr[k]
    mc_k, Tc_k, Th_k = cs.vertsplit(x_k)
    x_kp1 = x_arr[k+1]
    
    s_k = s_arr[k]
    
    x_opt_arr.extend([x_k, s_k, u_k])
    hsize_arr.append(3+1+1)
    lb_var_arr.extend([0.]*5)
    ub_var_arr.extend([np.inf]*4 + [1.0])
    x_start_arr.extend(x_arr[0] + [0.,0.])
    
    k1_x = dt*system_dynamics(x_k, u_k)
    k2_x = dt*system_dynamics(x_k + 0.5*k1_x, u_k)
    k3_x = dt*system_dynamics(x_k + 0.5*k2_x, u_k)
    k4_x = dt*system_dynamics(x_k + k3_x, u_k)
    x_kp1_ = x_k + 1./6.*(1*k1_x + 2*k2_x + 2*k3_x + 1*k4_x)
    
    # Conditional logic for Th[k+1]:
    # If temperature is not locked, use linear ramp
    # Once Th[k] reaches 90, lock the temperature behavior to RK4
    Th_desired = cs.if_else(temperature_locked == 0,
                            linear_ramp(time_k + dt, t_ramp, T_env, Th_in), 
                            x_kp1_[2])
        
    # Lock the temperature behavior once Th[k] reaches 90 for the first time
    temperature_locked = cs.if_else(Th_k >= Th_in, 1, temperature_locked)
    
    # Constraints
    con_arr.append(Th_k - Tc_k + s_k)
    lb_con_arr.extend([delta_T_min])
    ub_con_arr.extend([np.inf])
    
    con_arr.append(x_kp1 - cs.vertcat(x_kp1_[:-1], Th_desired))
    lb_con_arr.extend([0.]*3)
    ub_con_arr.extend([0.]*3)
x_N = x_arr[N]
s_N = s_arr[N]
mc_N, Tc_N, Th_N = cs.vertsplit(x_N)
con_arr.append(Th_N - Tc_N + s_N)
lb_con_arr.extend([delta_T_min])
ub_con_arr.extend([np.inf])

x_opt_arr.extend([x_arr[N], s_arr[N]])
lb_var_arr.extend([0.]*4)
ub_var_arr.extend([np.inf]*4)
x_start_arr.extend(x_arr[0] + [0.])
hsize_arr.append(4)

# Penalty weight for deviation from target temperature at final time
# Enforces terminal constraint Th[N] ≈ Th_out, guarantees feasability
w1 = 1e4
# Weight for slack variable penalty (constraint relaxation), doesn't need to be too strict
# Shall simply ensure that Th and Tc are never equal, guarantees feasibility
w2 = 1






# Objective function: Minimize cooling water usage at t = tf
x_opt = cs.vertcat(*x_opt_arr)
f_expr = mc_N + w1 * (Th_N - Th_out)**2 + w2 * cs.sumsqr(cs.vertcat(*s_arr[1:]))
g_expr = cs.vertcat(*con_arr)

NLP = {'x':x_opt, 'f':f_expr, 'g':g_expr}

f_ = cs.Function('cs_f', [x_opt], [f_expr])
f = lambda xi: np.array(f_(xi), dtype = np.float64).reshape(-1)

grad_f_expr = cs.jacobian(f_expr, x_opt)
grad_f_ = cs.Function('cs_grad_f', [x_opt], [grad_f_expr])
grad_f = lambda xi: np.array(grad_f_(xi), dtype = np.float64).reshape(-1)

g_ = cs.Function('cs_g', [x_opt], [g_expr])
g = lambda xi: np.array(g_(xi), dtype = np.float64).reshape(-1)
jac_g_expr = cs.jacobian(g_expr, x_opt)
jac_g_ = cs.Function('cs_jac_g', [x_opt], [jac_g_expr])
jac_g = lambda xi: np.array(jac_g_(xi), dtype = np.float64)

jac_g_nnz = jac_g_expr.nnz()
jac_g_row = jac_g_expr.row()
jac_g_colind = jac_g_expr.colind()
jac_g_nz = lambda xi: np.array(jac_g_(xi).nz[:], dtype = np.float64).reshape(-1)

lam = cs.MX.sym('lambda', g_expr.numel())
lag_expr = f_expr - lam.T @ g_expr
grad_lag_expr = cs.jacobian(lag_expr, x_opt)
hess_lag_expr = cs.jacobian(grad_lag_expr, x_opt)
hess_lag_ = cs.Function('hess_lag', [x_opt, lam], [hess_lag_expr])

def to_blocks_LT(sparse_hess : cs.DM):
    blocks = []
    for j in range(len(hsize_arr)):
       blocks.append(np.array(cs.tril(sparse_hess[blockIdx[j]:blockIdx[j+1], blockIdx[j]:blockIdx[j+1]].full()).nz[:], dtype = np.float64).reshape(-1))
    return blocks
hess_lag = lambda xi, lambd: to_blocks_LT(hess_lag_(xi, lambd))


lb_var = np.array(lb_var_arr)
ub_var = np.array(ub_var_arr)
lb_con = np.array(lb_con_arr)
ub_con = np.array(ub_con_arr)
x_start = np.array(x_start_arr)

blockIdx = np.cumsum(np.concatenate([np.array([0]), np.array(hsize_arr)]))



# ##################
# ## Plot results ##
# ##################

def plot(xi, dpi = None, title = None, it = None):
    time_grid = np.linspace(0, tf, N+1, endpoint = True)
    
    p_x_arr = [np.array(x_arr[0]).reshape(-1)]
    p_u_arr = [xi[0]]
    ind = 1
    for i in range(N-1):
        p_x_arr.append(xi[ind:ind + 3].reshape(-1))
        ind += 3
        ind += 1 #skip s
        p_u_arr.extend(xi[ind:ind + 1])
        ind += 1
    p_x_arr.append(xi[ind:ind+3].reshape(-1))
    p_x = np.concatenate(p_x_arr).reshape((3,-1), order = 'F')
    p_mc, p_Tc, p_Th = p_x
    p_u = np.array(p_u_arr)
    
    plt.figure(dpi = dpi)
    plt.plot(time_grid, p_mc*0.5, 'r-', label = 'mc*0.5')
    plt.plot(time_grid, p_Tc, 'g-', label = 'Tc')
    plt.plot(time_grid, p_Th, 'g-', label = 'Th')
    plt.step(time_grid[:-1], p_u*50, 'y-', label = 'u*50')
    plt.legend(fontsize='large')
    
    ttl = None
    if isinstance(title,str):
        ttl = title
    elif title == True:
        ttl = 'D\'Onofrio chemotherapy problem'
    if ttl is not None:
        if isinstance(it, int):
            ttl = ttl + f', iteration {it}'
        plt.title(ttl)
    else:
        plt.title('')
        
    plt.show()    


opts = py_blockSQP.SQPoptions()
opts.max_QP_it = 10000
opts.max_QP_secs = 5.0

opts.max_conv_QPs = 6
opts.conv_strategy = 1
opts.par_QPs = True
opts.enable_QP_cancellation=True
opts.test_opt_2 = 3

opts.exact_hess = 2
opts.hess_approx = 1
opts.sizing = 2
opts.fallback_approx = 2
opts.fallback_sizing = 4
opts.BFGS_damping_factor = 1/3

opts.lim_mem = True
opts.mem_size = 20
opts.opt_tol = 1e-6
opts.feas_tol = 1e-6
opts.conv_kappa_max = 8.0

opts.automatic_scaling = False

opts.max_extra_steps = 10
opts.enable_premature_termination = True
opts.max_filter_overrides = 0

opts.qpsol = 'qpOASES'
QPopts = py_blockSQP.qpOASES_options()
QPopts.printLevel = 0
QPopts.sparsityLevel = 2
opts.qpsol_options = QPopts
################################


#Define blockSQP Problemspec
prob = py_blockSQP.Problemspec()
prob.nVar = x_opt.size1()
prob.nCon = g_expr.size1()

prob.f = f
prob.grad_f = grad_f
prob.g = g
prob.make_sparse(jac_g_nnz, jac_g_row, jac_g_colind)
prob.jac_g_nz = jac_g_nz

prob.hess = hess_lag
prob.set_blockIndex(blockIdx)
prob.set_bounds(lb_var, ub_var, lb_con, ub_con)

prob.x_start = x_start





prob.lam_start = np.zeros(prob.nVar + prob.nCon, dtype = np.float64).reshape(-1)
prob.complete()

scale_arr = 1.0;

#####################
stats = py_blockSQP.SQPstats("./solver_outputs")
#No condensing
optimizer = py_blockSQP.SQPmethod(prob, opts, stats)
optimizer.init()

t0 = time.time()
# xi = x_start
for j in range(100):
    # plot(xi)
    ret = optimizer.run(1,1)
    if int(ret) != 0:
        break
    # xi = np.array(optimizer.get_xi())
    # plot(xi)
xi = np.array(optimizer.get_xi()).reshape(-1)
t1 = time.time()
plot(xi)
print("blockSQP required ", t1 - t0, "s\n")

# S = cs.nlpsol('S', 'ipopt', NLP, {'ipopt':{
#         'hessian_approximation' : 'exact'
#     }})
# out = S(x0=x_start, lbx=lb_var,ubx=ub_var, lbg=lb_con, ubg=ub_con)
# xi = np.array(out['x']).reshape(-1)
# plot(xi)


