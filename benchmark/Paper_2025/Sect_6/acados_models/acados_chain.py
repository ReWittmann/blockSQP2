from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, plot_trajectories, ACADOS_INFTY
import numpy as np
import casadi as ca
from casadi import SX, vertcat
import time

# Parameters
z0 = 0.0
zF = 1.0
a = 1.0
b = 3.0
L = 4.0

def export_hanging_chain_model() -> AcadosModel:
    model_name = 'hanging_chain'

    # States: x1 (height), x2 (potential energy), x3 (arc length)
    x1 = SX.sym('x1')
    x2 = SX.sym('x2')
    x3 = SX.sym('x3')
    X = vertcat(x1, x2, x3)

    # Control: u (slope dx1/dz)
    u = SX.sym('u')
    U = vertcat(u)

    # xdot symbols
    Xdot = SX.sym('Xdot', 3)
    
    # Dynamics
    # dx1/dz = u
    # dx2/dz = x1 * sqrt(1 + u^2)
    # dx3/dz = sqrt(1 + u^2)
    sqrt_term = ca.sqrt(1 + u**2)
    f_expl = vertcat(
        u,
        x1 * sqrt_term,
        sqrt_term
    )
    f_impl = Xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = X
    model.xdot = Xdot
    model.u = U
    model.name = model_name
    
    # Objective: min x2(zF)
    model.cost_expr_ext_cost_e = x2

    model.x_labels = ['Height (x1)', 'Energy (x2)', 'Length (x3)']
    model.u_labels = ['Slope (u)']
    model.t_label = 'Horizontal Position (z)'

    return model

# --- OCP Setup ---
ocp = AcadosOcp()
model = export_hanging_chain_model()
ocp.model = model

N = 100
nx = model.x.rows()
nu = model.u.rows()

ocp.solver_options.N_horizon = N
ocp.solver_options.tf = zF - z0

ocp.cost.cost_type_e = 'EXTERNAL'

# Constraints
# Control: -10 <= u <= 20
ocp.constraints.lbu = np.array([-10.0])
ocp.constraints.ubu = np.array([20.0])
ocp.constraints.idxbu = np.array([0])

# State constraints: 0 <= x <= 10
ocp.constraints.lbx = np.array([0.0])
ocp.constraints.ubx = np.array([10.0])
ocp.constraints.idxbx = np.array([0])

# Initial state: x(z0) = [a, 0, 0]
# ocp.constraints.lbx_0 = np.array([a, 0.0, 0.0])
# ocp.constraints.ubx_0 = np.array([a, 0.0, 0.0])
ocp.constraints.x0 = np.array([a, 0.0, 0.0])

# Final state constraints: x1(zF) = b, x3(zF) = L
ocp.constraints.lbx_e = np.array([b, L])
ocp.constraints.ubx_e = np.array([b, L])
ocp.constraints.idxbx_e = np.array([0, 2])

# Solver Options
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
ocp.solver_options.hessian_approx = 'EXACT'
ocp.solver_options.integrator_type = 'ERK'
ocp.solver_options.nlp_solver_type = 'SQP'
ocp.solver_options.globalization = 'MERIT_BACKTRACKING'

ocp_solver = AcadosOcpSolver(ocp)

# --- Initial Guess based on provided formula ---
zm = 0.25 if b > a else 0.75
z_vals = np.linspace(z0, zF, N + 1)

# x1_init(z) = (2 * |b-a|) * z * (z - 2*zm)
x1_init = (2 * abs(b - a)) * z_vals * (z_vals - 2 * zm)
# u_init = dx1/dz = (2 * |b-a|) * (2*z - 2*zm)
u_init = (2 * abs(b - a)) * (2 * z_vals - 2 * zm)

# Integrate for x2 and x3 initial guesses
x2_init = np.zeros(N + 1)
x3_init = np.zeros(N + 1)
for i in range(N):
    dz = (zF - z0) / N
    sqrt_u = np.sqrt(1 + u_init[i]**2)
    x2_init[i+1] = x2_init[i] + x1_init[i] * sqrt_u * dz
    x3_init[i+1] = x3_init[i] + sqrt_u * dz

# Set initial guess to solver
for i in range(N):
    ocp_solver.set(i, "x", np.array([x1_init[i], x2_init[i], x3_init[i]]))
    ocp_solver.set(i, "u", u_init[i])
ocp_solver.set(N, "x", np.array([x1_init[N], x2_init[N], x3_init[N]]))

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
    time_traj_list=[np.linspace(z0, zF, N+1)],
    time_label=model.t_label,
    labels_list=['Hanging Chain'],
    x_labels=model.x_labels,
    u_labels=model.u_labels,
    idxbu=ocp.constraints.idxbu,
    lbu=ocp.constraints.lbu,
    ubu=ocp.constraints.ubu,
    fig_filename='hanging_chain_ocp.png',
)