from acados_template import AcadosModel, AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver, plot_trajectories, ACADOS_INFTY
import numpy as np
import casadi as ca
from casadi import SX, vertcat
import time

# Parameters
x00 = 1.0
xF = 2.0
alpha = -0.75
c_param = 1.0
u_init_val = 1.0
Tf_init = 5.0

def export_dielectrophoretic_model() -> AcadosModel:
    model_name = 'dielectrophoretic_particle'

    # States: x0 (position), x1 (velocity), T (time accumulator)
    x0 = SX.sym('x0')
    x1 = SX.sym('x1')
    T = SX.sym('T')
    X = vertcat(x0, x1, T)

    u = SX.sym('u')

    # xdot symbols
    x0_dot = SX.sym('x0_dot')
    x1_dot = SX.sym('x1_dot')
    T_dot = SX.sym('T_dot')
    Xdot = vertcat(x0_dot, x1_dot, T_dot)
    
    f_expl = T*vertcat(x1 * u + alpha * u**2, 
                      -c_param * x1 + u, 
                       0.)
    
    f_impl = Xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = X
    model.xdot = Xdot
    model.u = u
    model.name = model_name
    
    # Objective: min Tf -> minimize the final value of state T
    model.cost_expr_ext_cost_e = T

    model.x_labels = ['Position (x0)', 'Velocity (x1)', 'Time (T)']
    model.u_labels = ['Voltage (u)']
    model.t_label = 'Normalized Time'

    return model

# --- OCP Setup ---
ocp = AcadosOcp()
model = export_dielectrophoretic_model()
ocp.model = model

# We use a fixed horizon N, but the "real" time is scaled by the state T
Tf_scaled = 1.0 
N = 100
nx = model.x.rows()
nu = model.u.rows()

ocp.solver_options.N_horizon = N
ocp.solver_options.tf = Tf_scaled

ocp.cost.cost_type_e = 'EXTERNAL'

# Constraints
ocp.constraints.lbu = np.array([0])
ocp.constraints.ubu = np.array([1.0])
ocp.constraints.idxbu = np.array([0])

# Initial state: x(0) = [x00, 0, 0]
ocp.constraints.lbx_0 = np.array([x00, 0.0, 1.0])
ocp.constraints.ubx_0 = np.array([x00, 0.0, ACADOS_INFTY])
ocp.constraints.idxbx_0 = np.arange(3)

# Final state: x0(tF) = xF
ocp.constraints.lbx_e = np.array([xF])
ocp.constraints.ubx_e = np.array([xF])
ocp.constraints.idxbx_e = np.array([0])

# Solver Options
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
ocp.solver_options.hessian_approx = 'EXACT'
ocp.solver_options.integrator_type = 'ERK'
ocp.solver_options.nlp_solver_type = 'SQP'
ocp.solver_options.globalization = 'MERIT_BACKTRACKING'

ocp_solver = AcadosOcpSolver(ocp)

# --- Automatic Initialization x = S(u) ---
# We simulate the system with u = 1.0 for Tf_init seconds
sim = AcadosSim()
sim.model = model
sim.solver_options.integrator_type = 'ERK'
sim.solver_options.T = Tf_init / N
sim.solver_options.num_steps = 2
sim_sol = AcadosSimSolver(sim)

x_current = np.array([x00, 0.0, Tf_init])
u_init_traj = np.full((N, nu), u_init_val)
sim_x = np.zeros((N + 1, nx))
sim_x[0, :] = x_current

for i in range(N):
    x_next = sim_sol.simulate(x=x_current, u=u_init_traj[i])
    sim_x[i+1, :] = x_next
    x_current = x_next

# Set initial guess to the solver
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

# The actual time axis is the values of the T state

plot_trajectories(
    x_traj_list=[simX[:,:-1]],
    u_traj_list=[simU],
    time_traj_list=[np.linspace(0, Tf_scaled, N+1) * simX[0,-1]],
    time_label='Time [s]',
    labels_list=['Dielectrophoretic Particle'],
    x_labels=model.x_labels,
    u_labels=model.u_labels,
    idxbu=ocp.constraints.idxbu,
    lbu=ocp.constraints.lbu,
    ubu=ocp.constraints.ubu,
    fig_filename='dielectrophoretic_ocp.png',
)