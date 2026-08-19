from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, plot_trajectories
from casadi import SX, vertcat, sin, cos, sqrt, exp
import numpy as np
import casadi as ca

def export_apollo_model() -> AcadosModel:
    model_name = 'apollo_reentry'

    R = 209.0
    beta = 4.26
    rho0 = 2.704e-3
    g = 3.2172e-4
    Sm = 53200.0
    c1 = 1.175
    c2 = 0.9
    c3 = 0.6

    v = SX.sym('v')
    gamma = SX.sym('gamma')
    xi = SX.sym('xi')
    q = SX.sym('q')
    T = SX.sym('T')
    x = vertcat(v, gamma, xi, q, T)

    u_sym = SX.sym('u')
    u = vertcat(u_sym)

    v_dot = SX.sym('v_dot')
    gamma_dot = SX.sym('gamma_dot')
    xi_dot = SX.sym('xi_dot')
    q_dot = SX.sym('q_dot')
    T_dot = SX.sym('T_dot')
    xdot = vertcat(v_dot, gamma_dot, xi_dot, q_dot, T_dot)

    rho = rho0 * exp(-beta * R * xi)
    CD = c1 - c2 * cos(u_sym)
    CL = c3 * sin(u_sym)

    f_expl = T*vertcat(
        -0.5 * Sm * rho * v**2 * CD - (g * sin(gamma)) / (1 + xi)**2,
        0.5 * Sm * rho * v * CL + (v * cos(gamma)) / (R * (1 + xi)) - (g * cos(gamma)) / (v * (1 + xi)**2),
        (v * sin(gamma)) / R,
        20 * v**3 * ca.sqrt(rho),
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

    model.cost_expr_ext_cost_e = q

    model.x_labels = [r'$v$', r'$\gamma$', r'$\xi$', r'$q$', r'$T$']
    model.u_labels = [r'$u$']
    model.t_label = r'$t$ [s]'

    return model

###############################################################################

ocp = AcadosOcp()
model = export_apollo_model()
ocp.model = model

# Problem constants
R = 209.0
Tf = 1.0
xf = np.array([0.27, 0.0, 2.5/R])

N = 60
nx = model.x.rows()
nu = model.u.rows()

ocp.solver_options.N_horizon = N
ocp.solver_options.tf = Tf


ocp.cost.cost_type_e = 'EXTERNAL'


ocp.constraints.lbu = np.array([-np.pi/2])
ocp.constraints.ubu = np.array([np.pi/2])
ocp.constraints.idxbu = np.array([0])

ocp.constraints.lbx = np.array([0.2, -0.2, 0.006])
ocp.constraints.ubx = np.array([0.4, 0.1, 0.02])
ocp.constraints.idxbx = np.array([0, 1, 2])

ocp.constraints.lbx_0 = np.array([0.36, -8.1 * np.pi / 180, 4.0 / R, 0., 220.0])
ocp.constraints.ubx_0 = np.array([0.36, -8.1 * np.pi / 180, 4.0 / R, 0., 240.0])
ocp.constraints.idxbx_0 = np.array([0,1,2,3,4])


ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
ocp.solver_options.hessian_approx = 'EXACT'
ocp.solver_options.integrator_type = 'ERK'
ocp.solver_options.nlp_solver_type = 'SQP'
ocp.solver_options.globalization = 'MERIT_BACKTRACKING'

ocp_solver = AcadosOcpSolver(ocp)
for i in range(ocp.solver_options.N_horizon):
    ocp_solver.set(i, 'u', 0.5)
for i in range(ocp.solver_options.N_horizon+1):
    ocp_solver.set(i, 'x', np.array([0.36, -8.1 * np.pi / 180, 4.0 / R, 0., 220.0]))

status = ocp_solver.solve()
ocp_solver.print_statistics()

if status != 0:
    print(f"Solver returned status {status}. Try adjusting N or initial guess.")

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
