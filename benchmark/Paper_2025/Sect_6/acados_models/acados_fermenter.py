from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, plot_trajectories, AcadosSim, AcadosSimSolver, ACADOS_INFTY
import numpy as np
import casadi as ca
from casadi import SX, vertcat
import time

# Parameters
Tf = 1.0
mu_x = 2e5
mu_p = 5000
gamma_xg = 5e4
gamma_x1 = 1e5
gamma_p1 = 2e4
gamma_x2 = 1500
gamma_p2 = 5e4

def export_fermenter_model() -> AcadosModel:
    model_name = 'fermenter'

    # States
    P = SX.sym('P')
    S1 = SX.sym('S1')
    S2 = SX.sym('S2')
    E = SX.sym('E')
    V = SX.sym('V')
    G = SX.sym('G')
    P_acc = SX.sym('P_acc')
    S1_acc = SX.sym('S1_acc')
    S2_acc = SX.sym('S2_acc')
    X = vertcat(P, S1, S2, E, V, G, P_acc, S1_acc, S2_acc)

    # Controls
    uS1 = SX.sym('uS1')
    uS2 = SX.sym('uS2')
    uP = SX.sym('uP')
    U = vertcat(uS1, uS2, uP)

    # xdot symbols
    Xdot = SX.sym('Xdot', 9)
    
    # Intermediate terms for readability
    u_sum = uS1 + uS2
    dilution = u_sum / (25 * V)
    reaction_rate = E * S1 * S2
    
    # Dynamics
    dP = mu_p * reaction_rate - P * dilution
    dS1 = -gamma_x1 * reaction_rate * G - gamma_p1 * reaction_rate + (0.42 * uS1 - S1 * u_sum) / (25 * V)
    dS2 = -gamma_x2 * reaction_rate * G - gamma_p2 * reaction_rate + (0.333 * uS2 - S2 * u_sum) / (25 * V)
    dE = mu_x * reaction_rate * G - E * dilution
    dV = uS1 + uS2 - uP
    dG = -gamma_xg * reaction_rate * G - G * dilution
    dP_acc = uP * P + ((uS1 + uS2 - uP) / 25) * P + V * dP
    dS1_acc = 0.0168 * uS1
    dS2_acc = 0.01332 * uS2

    f_expl = vertcat(dP, dS1, dS2, dE, dV, dG, dP_acc, dS1_acc, dS2_acc)
    f_impl = Xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = X
    model.xdot = Xdot
    model.u = U
    model.name = model_name
    
    # Final Cost: (2 * S1_acc * S2_acc) / P_acc
    # We use a small epsilon to avoid division by zero during initialization
    model.cost_expr_ext_cost_e = (2 * S1_acc * S2_acc) / (P_acc + 1e-6)

    model.x_labels = ['P', 'S1', 'S2', 'E', 'V', 'G', 'P_acc', 'S1_acc', 'S2_acc']
    model.u_labels = ['uS1', 'uS2', 'uP']
    model.t_label = 'Time [h]'

    return model

# --- OCP Setup ---
ocp = AcadosOcp()
model = export_fermenter_model()
ocp.model = model

N = 80
nx = model.x.rows()
nu = model.u.rows()

ocp.solver_options.N_horizon = N
ocp.solver_options.tf = Tf
# ocp.solver_options.qp_solver_cond_N = 10


# Cost: Only final cost (Mayer term)
ocp.cost.cost_type_e = 'EXTERNAL'

# Constraints
# Control constraints: 0 <= uS1 <= 15, 0 <= uS2 <= 1, 0 <= uP <= 30
ocp.constraints.lbu = np.array([0.0, 0.0, 0.0])
ocp.constraints.ubu = np.array([15.0, 1.0, 30.0])
ocp.constraints.idxbu = np.array([0, 1, 2])

# State constraints: 0 <= P, S1, S2, E, V, G <= ...
# Indices: P(0), S1(1), S2(2), E(3), V(4), G(5)
lbx = np.array([0.0, 0.0, 0.0, 0.0, 0.3, 0.0])
ubx = np.array([0.1, 0.04, 0.03, 0.1, 0.45, 0.1])
# ocp.constraints.lbx = lbx
# ocp.constraints.ubx = ubx
# ocp.constraints.idxbx = np.arange(6)

# # Acc constraints: 0 <= P_acc, S1_acc, S2_acc <= ...
# # Indices: P_acc(6), S1_acc(7), S2_acc(8)
lbx_acc = np.array([0.0, 0.0, 0.0])
ubx_acc = np.array([0.05, 0.2, 0.025])
ocp.constraints.lbx = np.concatenate([lbx, lbx_acc])
ocp.constraints.ubx = np.concatenate([ubx, ubx_acc])
ocp.constraints.idxbx = np.arange(9)

# Initial state
x0 = np.array([0, 0.03, 0.03, 0.01, 0.3, 0.1, 0, 0.009, 0.009])
ocp.constraints.lbx_0 = x0
ocp.constraints.ubx_0 = x0
ocp.constraints.idxbx_0 = np.arange(nx)

# Solver Options
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
# ocp.solver_options.qp_solver_iter_max = 80
ocp.solver_options.hessian_approx = 'EXACT'
ocp.solver_options.integrator_type = 'IRK'
# ocp.solver_options.sim_method_num_steps = 10
ocp.solver_options.nlp_solver_type = 'SQP'
ocp.solver_options.globalization = 'MERIT_BACKTRACKING'

ocp_solver = AcadosOcpSolver(ocp)

# --- Automatic Initialization x = S(u) ---
# 1. Setup the Simulator
sim = AcadosSim()
sim.model = model
sim.solver_options.integrator_type = 'IRK'
sim.solver_options.T = Tf / N
sim.solver_options.num_steps = 2
sim_sol = AcadosSimSolver(sim)

# 2. Define initial control guess (u_S1=0, u_S2=0, u_P=0)
u_init_val = np.zeros(nu) 
u_init_traj = np.tile(u_init_val, (N, 1))

# 3. Simulate the trajectory
x_current = x0 # Initial state [0, 0.03, 0.03, 0.01, 0.3, 0.1, 0, 0.009, 0.009]
sim_x = np.zeros((N + 1, nx))
sim_x[0, :] = x_current

for i in range(N):
    # Simulate one step forward
    x_next = sim_sol.simulate(x=x_current, u=u_init_traj[i])
    sim_x[i+1, :] = x_next
    x_current = x_next

# 4. Set the simulated trajectory as the initial guess for the OCP solver
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

# --- Extract and Plot ---
simX = np.zeros((N+1, nx))
simU = np.zeros((N, nu))
for i in range(N):
    simX[i,:] = ocp_solver.get(i, "x")
    simU[i,:] = ocp_solver.get(i, "u")
simX[N,:] = ocp_solver.get(N, "x")

plot_trajectories(
    x_traj_list=[simX[:,:6]],
    u_traj_list=[simU],
    time_traj_list=[np.linspace(0, Tf, N+1)],
    time_label=model.t_label,
    labels_list=['Fermenter Model'],
    x_labels=model.x_labels,
    u_labels=model.u_labels,
    idxbu=ocp.constraints.idxbu,
    lbu=ocp.constraints.lbu,
    ubu=ocp.constraints.ubu,
    fig_filename='fermenter_ocp.png',
)