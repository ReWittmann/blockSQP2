from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, plot_trajectories
# from lotka_model import export_lotka_volterra_model
import numpy as np
import casadi as ca
import time

from casadi import SX, vertcat

def export_lotka_volterra_model() -> AcadosModel:
    model_name = 'lotka_volterra_fishing'

    # Parameters
    p1, p2, p3, p4 = 1.0, 1.0, 1.0, 1.0
    c1, c2 = 0.4, 0.2

    # States: x1 (prey), x2 (predator)
    x1 = SX.sym('x1')
    x2 = SX.sym('x2')
    x = vertcat(x1, x2)

    # Control: u (fishing effort)
    u_sym = SX.sym('u')
    u = vertcat(u_sym)

    # xdot symbols
    x1_dot = SX.sym('x1_dot')
    x2_dot = SX.sym('x2_dot')
    xdot = vertcat(x1_dot, x2_dot)

    # Dynamics: f_expl
    f_expl = vertcat(
        p1 * x1 - p2 * x1 * x2 - c1 * u_sym * x1,
        -p3 * x2 + p4 * x1 * x2 - c2 * u_sym * x2
    )

    # Implicit dynamics for acados
    f_impl = xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    # Meta information for plotting
    model.x_labels = ['Prey $x_1$', 'Predator $x_2$']
    model.u_labels = ['Effort $u$']
    model.t_label = 'Time [s]'

    return model


# Create OCP object
ocp = AcadosOcp()

# Set model
model = export_lotka_volterra_model()
ocp.model = model

Tf = 12.0
nx = model.x.rows()
nu = model.u.rows()
N = 100

# Set prediction horizon
ocp.solver_options.N_horizon = N
ocp.solver_options.tf = Tf
ocp.solver_options.qp_solver_cond_N = 3

# Cost matrices
# Objective: (x1-1)^2 + (x2-1)^2. 
# In NLS: 0.5 * ||W(y - yref)||^2. 
# To get 1.0 * (x-1)^2, W must be sqrt(2)
Q_mat = np.eye(2)
# We don't penalize u in the integral, but a tiny R prevents singularity
R_mat = np.diag([1e-4 * 0])

# Path cost
ocp.cost.cost_type = 'NONLINEAR_LS'
ocp.model.cost_y_expr = ca.vertcat(model.x, model.u)
ocp.cost.yref = np.array([1.0, 1.0, 0.0])
ocp.cost.W = ca.diagcat(Q_mat, R_mat).full()

# Terminal cost
ocp.cost.cost_type_e = 'NONLINEAR_LS'
ocp.model.cost_y_expr_e = model.x
ocp.cost.yref_e = np.array([1.0, 1.0])
ocp.cost.W_e = Q_mat

# Constraints
ocp.constraints.lbu = np.array([0.0])
ocp.constraints.ubu = np.array([1.0])
ocp.constraints.idxbu = np.array([0])

# Initial state x(0) = [0.5, 0.7]
ocp.constraints.x0 = np.array([0.5, 0.7])

# Solver options
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
ocp.solver_options.hessian_approx = 'EXACT'
ocp.solver_options.integrator_type = 'ERK'
ocp.solver_options.nlp_solver_type = 'SQP'
ocp.solver_options.globalization = 'MERIT_BACKTRACKING'

ocp.solver_options.nlp_solver_max_iter = 100
ocp.solver_options.tol = 1e-6

ocp_solver = AcadosOcpSolver(ocp)
for i in range(ocp.solver_options.N_horizon + 1):
    ocp_solver.set(i, "x", np.array([0.5,0.7]))


simX = np.zeros((N+1, nx))
simU = np.zeros((N, nu))

t0 = time.time()
status = ocp_solver.solve()
t1 = time.time()

ocp_solver.print_statistics()

# if status != 0:
#     raise Exception(f'acados returned status {status}.')

# Get solution
for i in range(N):
    simX[i,:] = ocp_solver.get(i, "x")
    simU[i,:] = ocp_solver.get(i, "u")
simX[N,:] = ocp_solver.get(N, "x")

plot_trajectories(
    x_traj_list=[simX],
    u_traj_list=[simU],
    time_traj_list=[np.linspace(0, Tf, N+1)],
    time_label=model.t_label,
    labels_list=['OCP result'],
    x_labels=model.x_labels,
    u_labels=model.u_labels,
    idxbu=ocp.constraints.idxbu,
    lbu=ocp.constraints.lbu,
    ubu=ocp.constraints.ubu,
    fig_filename='lotka_ocp.png',
)
