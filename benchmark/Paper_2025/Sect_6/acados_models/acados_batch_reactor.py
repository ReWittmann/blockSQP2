from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, plot_trajectories
import numpy as np
import casadi as ca
from casadi import SX, vertcat, exp
import time

def export_reactor_model() -> AcadosModel:
    model_name = 'batch_reactor'

    # States: x1 (Substance A), x2 (Substance B)
    x1 = SX.sym('x1')
    x2 = SX.sym('x2')
    x = vertcat(x1, x2)

    # Control: u (Temperature)
    u_sym = SX.sym('u')
    u = vertcat(u_sym)

    # xdot symbols
    x1_dot = SX.sym('x1_dot')
    x2_dot = SX.sym('x2_dot')
    xdot = vertcat(x1_dot, x2_dot)

    # Rate constants (Arrhenius equations)
    k1 = 4000 * exp(-2500 / u_sym)
    k2 = 620000 * exp(-5000 / u_sym)

    # Dynamics
    f_expl = vertcat(
        -k1 * x1**2,
        k1 * x1**2 - k2 * x2
    )

    f_impl = xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    # Terminal cost expression: we want to maximize x2, so we minimize -x2.
    # In acados NONLINEAR_LS: 0.5 * ||W_e * (y_e - y_ref_e)||^2
    # To get a linear cost, we can't directly. 
    # However, we can set y_e = x2 and y_ref_e = a very large number.
    # Or, we can use the trick: y_e = sqrt(x2) if x2 > 0.
    # For this specific problem, we will define y_e = x2 and use a target.
    model.cost_y_expr_e = x2

    model.x_labels = ['Substance A', 'Substance B']
    model.u_labels = ['Temperature']
    model.t_label = 'Time [s]'

    return model

ocp = AcadosOcp()
model = export_reactor_model()
ocp.model = model

Tf = 1.0
N = 100
nx = model.x.rows()
nu = model.u.rows()

ocp.solver_options.N_horizon = N
ocp.solver_options.tf = Tf

# --- Path Cost ---
# No integral cost in this problem
ocp.cost.cost_type = 'NONLINEAR_LS'
ocp.model.cost_y_expr = ca.vertcat(0) # Dummy
ocp.cost.yref = np.array([0.0])
ocp.cost.W = np.array([[0.0]])

# --- Terminal Cost ---
# Goal: Maximize x2(tF)  => Minimize -x2(tF)
# Since acados uses 0.5 * W * (y - yref)^2, we can't do linear directly.
# Trick: Set y_ref_e to a value higher than any possible x2 (e.g., 1.0)
# and W_e = 1. This minimizes (x2 - 1)^2, which is equivalent to maximizing x2.
ocp.cost.cost_type_e = 'NONLINEAR_LS'
ocp.cost.yref_e = np.array([1.0]) 
ocp.cost.W_e = np.array([[1.0]])

# --- Constraints ---
# Control constraints: 298 <= u <= 398
ocp.constraints.lbu = np.array([298.0])
ocp.constraints.ubu = np.array([398.0])
ocp.constraints.idxbu = np.array([0])

# Initial state: x(0) = [1, 0]
ocp.constraints.x0 = np.array([1.0, 0.0])

# --- Solver Options ---
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
ocp.solver_options.hessian_approx = 'EXACT'
ocp.solver_options.integrator_type = 'ERK'
ocp.solver_options.nlp_solver_type = 'SQP'
ocp.solver_options.globalization = 'MERIT_BACKTRACKING'

ocp_solver = AcadosOcpSolver(ocp)

# Initial guess
for i in range(ocp.solver_options.N_horizon):
    ocp_solver.set(i, "u", 298)

t0 = time.time()
status = ocp_solver.solve()
t1 = time.time()

ocp_solver.print_statistics()

if status != 0:
    print(f"Warning: Solver returned status {status}")

# Extract solution
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
    labels_list=['Batch Reactor'],
    x_labels=model.x_labels,
    u_labels=model.u_labels,
    idxbu=ocp.constraints.idxbu,
    lbu=ocp.constraints.lbu,
    ubu=ocp.constraints.ubu,
    fig_filename='reactor_ocp.png',
)
