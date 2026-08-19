from acados_template import AcadosModel, AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver, ACADOS_INFTY, plot_trajectories
import numpy as np
import casadi as ca
from casadi import SX, vertcat
import time

# Parameters
T = 12.0
c1 = 1.0
c2 = 2.0
c3 = 0.8
k1 = 2.0
k2 = 3.0
k3 = 1.0
k4 = 3.0

def export_three_tank_model() -> AcadosModel:
    model_name = 'three_tank'

    # States: x1, x2, x3 (fluid levels)
    x1 = SX.sym('x1')
    x2 = SX.sym('x2')
    x3 = SX.sym('x3')
    X = vertcat(x1, x2, x3)

    # Controls: w1, w2, w3 (flow rates)
    w1 = SX.sym('w1')
    w2 = SX.sym('w2')
    w3 = SX.sym('w3')
    U = vertcat(w1, w2, w3)

    # xdot symbols
    Xdot = SX.sym('Xdot', 3)
    
    # Dynamics
    # Note: we use ca.sqrt(x + 1e-6) to avoid gradients of sqrt(0)
    dx1 = -ca.sqrt(x1 + 1e-8) + c1*w1 + c2*w2 - w3*ca.sqrt(c3*x1 + 1e-8)
    dx2 = ca.sqrt(x1 + 1e-8) - ca.sqrt(x2 + 1e-8)
    dx3 = ca.sqrt(x2 + 1e-8) - ca.sqrt(x3 + 1e-8) + w3*ca.sqrt(c3*x1 + 1e-8)

    f_expl = vertcat(dx1, dx2, dx3)
    f_impl = Xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = X
    model.xdot = Xdot
    model.u = U
    model.name = model_name
    
    # Stage Cost: k1*(x2 - k2)^2 + k3*(x3 - k4)^2
    model.cost_expr_ext_cost_e = k1 * (x2 - k2)**2 + k3 * (x3 - k4)**2

    model.x_labels = ['Tank 1', 'Tank 2', 'Tank 3']
    model.u_labels = ['w1', 'w2', 'w3']
    model.t_label = 'Time [s]'

    return model

# --- OCP Setup ---
ocp = AcadosOcp()
model = export_three_tank_model()
ocp.model = model

N = 50
nx = model.x.rows()
nu = model.u.rows()

ocp.solver_options.N_horizon = N
ocp.solver_options.tf = T
ocp.solver_options.qp_solver_cond_N = 50


ocp.cost.cost_type = 'LINEAR_LS'
ocp.cost.yref = np.array([k2, k4])
ocp.cost.Vx = np.array([[0.,1.,0.],[0.,0.,1.]])
ocp.cost.Vu = np.array([[0.,0.,0.],[0.,0.,0.]])
ocp.cost.W = np.diag([k1, k3])

ocp.cost.cost_type_e = 'LINEAR_LS'
ocp.cost.yref_e = np.array([k2, k4])
ocp.cost.Vx_e = np.array([[0.,1.,0.],[0.,0.,1.]])
ocp.cost.Vu_e = np.array([[0.],[0.]])
ocp.cost.W_e = np.diag([k1, k3])


# Constraints
# Control bounds: 0 <= w_i <= 1
ocp.constraints.lbu = np.array([0.0, 0.0, 0.0])
ocp.constraints.ubu = np.array([1.0, 1.0, 1.0])
ocp.constraints.idxbu = np.array([0, 1, 2])

# State bounds: 0 <= x_i
ocp.constraints.lbx = np.array([0.0, 0.0, 0.0])
ocp.constraints.ubx = np.array([ACADOS_INFTY, ACADOS_INFTY, ACADOS_INFTY]) # Large upper bound
ocp.constraints.idxbx = np.arange(nx)

# Initial state: x(0) = (2, 2, 2)
x0 = np.array([2.0, 2.0, 2.0])
ocp.constraints.lbx_0 = x0
ocp.constraints.ubx_0 = x0
ocp.constraints.idxbx_0 = np.arange(nx)


ocp.model.con_h_expr = ca.sum(ocp.model.u)
ocp.constraints.lh = np.array([1.0])
ocp.constraints.uh = np.array([1.0])


# Solver Options
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
ocp.solver_options.hessian_approx = 'EXACT'
ocp.solver_options.integrator_type = 'ERK'
ocp.solver_options.nlp_solver_type = 'SQP'
ocp.solver_options.globalization = 'MERIT_BACKTRACKING'

ocp_solver = AcadosOcpSolver(ocp)

# --- Automatic Initialization x = S(w) ---
sim = AcadosSim()
sim.model = model
sim.solver_options.integrator_type = 'ERK'
sim.solver_options.T = T / N
sim.solver_options.num_steps = 2
sim_sol = AcadosSimSolver(sim)

w_init_val = np.array([1/3, 1/3, 1/3])
u_init_traj = np.tile(w_init_val, (N, 1))

x_current = x0
sim_x = np.zeros((N + 1, nx))
sim_x[0, :] = x_current

for i in range(N):
    x_next = sim_sol.simulate(x=x_current, u=u_init_traj[i])
    sim_x[i+1, :] = x_next
    x_current = x_next

for i in range(N):
    ocp_solver.set(i, "x", sim_x[i, :])
    ocp_solver.set(i, "u", u_init_traj[i])
ocp_solver.set(N, "x", sim_x[N, :])

# Solve
t0 = time.time()
status = ocp_solver.solve()
t1 = time.time()
ocp_solver.print_statistics()

# --- Extract and Plot ---
simX = np.zeros((N+1, nx))
simU = np.zeros((N, nu))
for i in range(N):
    simX[i,:] = ocp_solver.get(i, "x")
    simU[i,:] = ocp_solver.get(i, "u")
simX[N,:] = ocp_solver.get(N, "x")

plot_trajectories(
    x_traj_list=[simX],
    u_traj_list=[simU],
    time_traj_list=[np.linspace(0, T, N+1)],
    time_label=model.t_label,
    labels_list=['Three Tank System'],
    x_labels=model.x_labels,
    u_labels=model.u_labels,
    idxbu=ocp.constraints.idxbu,
    lbu=ocp.constraints.lbu,
    ubu=ocp.constraints.ubu,
    fig_filename='three_tank_ocp.png',
)