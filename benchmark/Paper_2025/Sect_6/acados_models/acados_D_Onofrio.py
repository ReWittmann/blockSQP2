from acados_template import AcadosModel, AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver, plot_trajectories, ACADOS_INFTY
import numpy as np
import casadi as ca
from casadi import SX, vertcat
import time

# Parameters
Tf = 6.0
zeta = 0.192
b = 5.85
mu = 0.0
d = 0.00873
G = 0.15
F = 1.0
eta = 1.0
alpha = 0.0
u0_max = 75.0
x2_max = 300.0

#Param set 1
# x00 = 12000.0
# x10 = 15000.0
# u1_max = 1.0
# x3_max = 2.0

#Param set 2
x00 = 12000
x10 = 15000
u1_max = 2
x3_max = 10

#Param set 3
# x00 = 14000
# x10 = 5000
# u1_max = 1
# x3_max = 2

#Param set 4
# x00 = 14000
# x10 = 5000
# u1_max = 2
# x3_max = 10


def export_chemotherapy_model() -> AcadosModel:
    model_name = 'D_Onofrio_Chemotherapy'

    x0 = SX.sym('x0')
    x1 = SX.sym('x1')
    x2 = SX.sym('x2')
    x3 = SX.sym('x3')
    X = vertcat(x0, x1, x2, x3)

    u0 = SX.sym('u0')
    u1 = SX.sym('u1')
    U = vertcat(u0, u1)

    x0_dot = SX.sym('x0_dot')
    x1_dot = SX.sym('x1_dot')
    x2_dot = SX.sym('x2_dot')
    x3_dot = SX.sym('x3_dot')
    Xdot = vertcat(x0_dot, x1_dot, x2_dot, x3_dot)
    
    f_expl = vertcat(
        -zeta * x0 * ca.log(x0 / x1) - F * x0 * u1,
        b * x0 - mu * x1 - d * (x0**(2/3)) * x1 - G * u0 * x1 - eta * x1 * u1,
        u0,
        u1
    )
    f_impl = Xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = X
    model.xdot = Xdot
    model.u = U
    model.name = model_name
    
    # Stage cost: alpha * u0^2
    # model.cost_expr_ext_cost = alpha * u0**2
    model.cost_expr_ext_cost_e = x0

    model.x_labels = ['Tumor Vol', 'Vessel Vol', 'Cumul u0', 'Cumul u1']
    model.u_labels = ['Drug u0', 'Drug u1']
    model.t_label = 'Time [days]'

    return model

# --- OCP Setup ---
ocp = AcadosOcp()
model = export_chemotherapy_model()
ocp.model = model

N = 100
nx = model.x.rows()
nu = model.u.rows()

ocp.solver_options.N_horizon = N
ocp.solver_options.tf = Tf
ocp.solver_options.qp_solver_cond_N = 4

# Cost: min x0(tF) + integral(alpha * u0^2)
ocp.cost.cost_type_e = 'EXTERNAL' # For the integral part

ocp.cost.cost_type = 'LINEAR_LS'
ocp.cost.yref = np.array([0.])

ocp.cost.Vx = np.array([[0.0, 0.0, 0.0, 0.0]])
ocp.cost.Vu = np.array([[1.0,0.]])
ocp.cost.W = np.array([[alpha]])




# ocp.cost.cost_type_x_e = 'LINEAR'  # For the final state x0(tF)
# ocp.cost.Vx_e = np.array([1.0, 0.0, 0.0, 0.0]) # Weight for x0 at tF

# Constraints
# Control constraints: 0 <= u <= u_max
ocp.constraints.lbu = np.array([0.0, 0.0])
ocp.constraints.ubu = np.array([u0_max, u1_max])
ocp.constraints.idxbu = np.array([0, 1])

# Initial state
ocp.constraints.lbx_0 = np.array([x00, x10, 0.0, 0.0])
ocp.constraints.ubx_0 = np.array([x00, x10, 0.0, 0.0])
ocp.constraints.idxbx_0 = np.arange(4)

# Final state constraints: x2(tF) <= x2_max, x3(tF) <= x3_max
ocp.constraints.lbx_e = np.array([-ACADOS_INFTY, -ACADOS_INFTY]) # No lower bound
ocp.constraints.ubx_e = np.array([x2_max, x3_max])
ocp.constraints.idxbx_e = np.array([2, 3])

# Solver Options
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
ocp.solver_options.hessian_approx = 'EXACT'
ocp.solver_options.integrator_type = 'IRK'
ocp.solver_options.nlp_solver_type = 'SQP'
ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
ocp.solver_options.nlp_solver_max_iter = 200

tm2 = time.time()
ocp_solver = AcadosOcpSolver(ocp)
tm1 = time.time()

u0_init_val = x2_max / Tf
u1_init_val = x3_max / Tf

sim = AcadosSim()
sim.model = model
sim.solver_options.integrator_type = 'ERK'
sim.solver_options.T = Tf / N
sim.solver_options.num_steps = 2
sim_sol = AcadosSimSolver(sim)

x_current = np.array([x00, x10, 0.0, 0.0])
u_init_traj = np.zeros((N, nu))
for i in range(N):
    u_init_traj[i, 0] = u0_init_val
    u_init_traj[i, 1] = u1_init_val

sim_x = np.zeros((N + 1, nx))
sim_x[0, :] = x_current

for i in range(N):
    x_next = sim_sol.simulate(x=x_current, u=u_init_traj[i])
    sim_x[i+1, :] = x_next
    x_current = x_next

# Set initial guess
for i in range(N):
    ocp_solver.set(i, "x", sim_x[i, :])
    ocp_solver.set(i, "u", u_init_traj[i])
ocp_solver.set(N, "x", sim_x[N, :])

# Solve
t0 = time.time()
status = ocp_solver.solve()
t1 = time.time()

ocp_solver.print_statistics()
if status != 0:
    print(f"Warning: Solver returned status {status}")


simX = np.zeros((N+1, nx))
simU = np.zeros((N, nu))
for i in range(N):
    simX[i,:] = ocp_solver.get(i, "x")
    simU[i,:] = ocp_solver.get(i, "u")
simX[N,:] = ocp_solver.get(N, "x")

plot_trajectories(
    x_traj_list=[simX[:,0:2]],
    u_traj_list=[simU],
    time_traj_list=[np.linspace(0, Tf, N+1)],
    time_label=model.t_label,
    labels_list=['Chemotherapy Model'],
    x_labels=model.x_labels,
    u_labels=model.u_labels,
    idxbu=ocp.constraints.idxbu,
    lbu=ocp.constraints.lbu,
    ubu=ocp.constraints.ubu,
    fig_filename='chemotherapy_ocp.png',
)