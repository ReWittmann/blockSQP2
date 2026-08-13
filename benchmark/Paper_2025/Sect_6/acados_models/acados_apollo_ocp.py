from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, plot_trajectories
from casadi import SX, vertcat, sin, cos, sqrt, exp
import numpy as np
import casadi as ca

def export_apollo_model() -> AcadosModel:
    model_name = 'apollo_reentry'

    # Parameters
    R = 209.0
    beta = 4.26
    rho0 = 2.704e-3
    g = 3.2172e-4
    Sm = 53200.0
    c1 = 1.175
    c2 = 0.9
    c3 = 0.6

    # States: v (velocity), gamma (flight path angle), xi (altitude/R)
    v = SX.sym('v')
    gamma = SX.sym('gamma')
    xi = SX.sym('xi')
    T = SX.sym('T')
    x = vertcat(v, gamma, xi, T)

    # Control: u (bank angle)
    u_sym = SX.sym('u')
    u = vertcat(u_sym)

    # xdot symbols
    v_dot = SX.sym('v_dot')
    gamma_dot = SX.sym('gamma_dot')
    xi_dot = SX.sym('xi_dot')
    T_dot = SX.sym('T_dot')
    xdot = vertcat(v_dot, gamma_dot, xi_dot, T_dot)

    # Intermediate expressions
    rho = rho0 * exp(-beta * R * xi)
    CD = c1 - c2 * cos(u_sym)
    CL = c3 * sin(u_sym)

    # Dynamics f_expl
    f_expl = T*vertcat(
        -0.5 * Sm * rho * v**2 * CD - (g * sin(gamma)) / (1 + xi)**2,
        0.5 * Sm * rho * v * CL + (v * cos(gamma)) / (R * (1 + xi)) - (g * cos(gamma)) / (v * (1 + xi)**2),
        (v * sin(gamma)) / R,
        0.
    )

    f_impl = xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    # Cost: Minimize integral of 10 * v^3 * sqrt(rho)
    # acados cost = 0.5 * W * (y - yref)^2. 
    # To get 10 * v^3 * sqrt(rho), we set y = sqrt(20 * v^3 * sqrt(rho)) and W = 1.
    model.cost_y_expr = ca.sqrt(20 * v**3 * ca.sqrt(rho))
    
    # Terminal cost: we use hard constraints instead, but acados requires 
    # a terminal cost expression if cost_type_e is set.
    model.cost_y_expr_e = x[0:3]

    model.x_labels = [r'$v$', r'$\gamma$', r'$\xi$', r'$T$']
    model.u_labels = [r'$u$']
    model.t_label = r'$t$ [s]'

    return model

###############################################################################

ocp = AcadosOcp()
model = export_apollo_model()
ocp.model = model

# Problem constants
R = 209.0
Tf = 1.0 #230.0 
xf = np.array([0.27, 0.0, 2.5/R])

N = 40
nx = model.x.rows()
nu = model.u.rows()

ocp.solver_options.N_horizon = N
ocp.solver_options.tf = Tf

# --- Path Cost ---
ocp.cost.cost_type = 'NONLINEAR_LS'
ocp.cost.yref = np.array([0.0])
ocp.cost.W = np.array([[1.0]])
ocp.cost.y_expr = ocp.model.x[0:3]

# --- Terminal Cost ---
# We set this to 0 because we use hard terminal constraints (xf)
# ocp.cost.cost_type_e = 'NONLINEAR_LS'
# ocp.cost.yref_e = xf
# ocp.cost.y_expr_e = model.x[0:3]
# ocp.cost.W_e = np.eye(3)*100

# --- Constraints ---
# Control constraints: -pi/2 <= u <= pi/2
ocp.constraints.lbu = np.array([-np.pi/2])
ocp.constraints.ubu = np.array([np.pi/2])
ocp.constraints.idxbu = np.array([0])

# State constraints: 0.2 <= v <= 0.4, -0.2 <= gamma <= 0.1, 0.006 <= xi <= 0.02
ocp.constraints.lbx = np.array([0.2, -0.2, 0.006])
ocp.constraints.ubx = np.array([0.4, 0.1, 0.02])
ocp.constraints.idxbx = np.array([0, 1, 2])

# Initial state
ocp.constraints.lbx_0 = np.array([0.36, -8.1 * np.pi / 180, 4.0 / R, 220.0])
ocp.constraints.ubx_0 = np.array([0.36, -8.1 * np.pi / 180, 4.0 / R, 240.0])
ocp.constraints.idxbx_0 = np.array([0,1,2,3])

# ocp.constraints.x0 = np.array([0.36, -8.1 * np.pi / 180, 4.0 / R])

# Terminal constraints (Hard equalities)
# v(tF) = 0.27, gamma(tF) = 0, xi(tF) = 2.5/R
# xf = np.array([0.27, 0.0, 2.5/R])
# ocp.constraints.idxbx_e = np.array([0, 1, 2])
# ocp.constraints.lbx_e = xf
# ocp.constraints.ubx_e = xf

# --- Solver Options ---
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
ocp.solver_options.hessian_approx = 'EXACT'
ocp.solver_options.integrator_type = 'ERK'
ocp.solver_options.nlp_solver_type = 'SQP'
ocp.solver_options.globalization = 'MERIT_BACKTRACKING'

ocp_solver = AcadosOcpSolver(ocp)
# Initial guess
for i in range(ocp.solver_options.N_horizon):
    ocp_solver.set(i, 'u', 0.5)
for i in range(ocp.solver_options.N_horizon+1):
    ocp_solver.set(i, 'x', np.array([0.36, -8.1 * np.pi / 180, 4.0 / R, 220.0]))

status = ocp_solver.solve()
ocp_solver.print_statistics()

if status != 0:
    print(f"Solver returned status {status}. Try adjusting N or initial guess.")

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
    labels_list=['Apollo Reentry'],
    x_labels=model.x_labels,
    u_labels=model.u_labels,
    idxbu=ocp.constraints.idxbu,
    lbu=ocp.constraints.lbu,
    ubu=ocp.constraints.ubu,
    fig_filename='apollo_reentry.png',
)
