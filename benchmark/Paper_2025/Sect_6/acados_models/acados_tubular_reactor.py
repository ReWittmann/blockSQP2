from acados_template import AcadosModel, AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver, plot_trajectories
import numpy as np
import casadi as ca
from casadi import SX, vertcat
import time

# Parameters
Tf = 1.0

def export_tubular_reactor_model() -> AcadosModel:
    model_name = 'tubular_reactor'

    # State: x (concentration)
    x = SX.sym('x')
    X = vertcat(x)

    # Control: w
    w = SX.sym('w')
    U = vertcat(w)

    # xdot symbols
    Xdot = SX.sym('Xdot', 1)
    
    # Dynamics: dx/dt = -(w + 0.5*w^2) * x
    f_expl = -(w + 0.5 * w**2) * x
    f_impl = Xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = X
    model.xdot = Xdot
    model.u = U
    model.name = model_name
    
    # Objective: min - integral(w * x) dt
    model.cost_expr_ext_cost_e = -w * x

    model.x_labels = ['Concentration (x)']
    model.u_labels = ['Control (w)']
    model.t_label = 'Time'

    return model

# --- OCP Setup ---
ocp = AcadosOcp()
model = export_tubular_reactor_model()
ocp.model = model

N = 100
nx = model.x.rows()
nu = model.u.rows()

ocp.solver_options.N_horizon = N
ocp.solver_options.tf = Tf

ocp.cost.cost_type_e = 'EXTERNAL'

# Constraints
# Control: 0 <= w <= 5.0
ocp.constraints.lbu = np.array([0.0])
ocp.constraints.ubu = np.array([5.0])
ocp.constraints.idxbu = np.array([0])

# Initial state: x(0) = 1
ocp.constraints.lbx_0 = np.array([1.0])
ocp.constraints.ubx_0 = np.array([1.0])
ocp.constraints.idxbx_0 = np.arange(nx)

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
sim.solver_options.T = Tf / N
sim.solver_options.num_steps = 2
sim_sol = AcadosSimSolver(sim)

# Start point: w = 5, x = 0 (Note: x(0)=1 is enforced by constraints, 
# but the trajectory guess can start at 0 as per prompt)
u_init_val = 5.0
u_init_traj = np.full((N, nu), u_init_val)

x_current = np.array([1.0]) # Using x(0)=1 for simulation consistency
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
status = ocp_solver.solve()
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
    time_traj_list=[np.linspace(0, Tf, N+1)],
    time_label=model.t_label,
    labels_list=['Tubular Reactor'],
    x_labels=model.x_labels,
    u_labels=model.u_labels,
    idxbu=ocp.constraints.idxbu,
    lbu=ocp.constraints.lbu,
    ubu=ocp.constraints.ubu,
    fig_filename='tubular_reactor_ocp.png',
)