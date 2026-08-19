from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, plot_trajectories
import numpy as np
import casadi as ca
from casadi import SX, vertcat, exp
import time

def export_catalyst_mixing_model() -> AcadosModel:
    model_name = 'catalyst_mixing'

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
    
    alpha = 10
    f_expl = ca.vertcat(u*(alpha*x2-x1), u*(x1 - alpha*x2) - (1-u)*x2)
    f_impl = xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name
    
    # model.cost_y_expr_e = x2
    model.cost_expr_ext_cost_e = -1 + x1 + x2

    model.x_labels = ['Substance A', 'Substance B']
    model.u_labels = ['Control']
    model.t_label = 'Time [s]'

    return model

ocp = AcadosOcp()
model = export_catalyst_mixing_model()
ocp.model = model

Tf = 1.0
N = 30
nx = model.x.rows()
nu = model.u.rows()

ocp.solver_options.N_horizon = N
ocp.solver_options.tf = Tf

ocp.cost.cost_type_e = 'EXTERNAL'

# --- Constraints ---
# Control constraints: 298 <= u <= 398
ocp.constraints.lbu = np.array([0.0])
ocp.constraints.ubu = np.array([1.0])
ocp.constraints.idxbu = np.array([0])

# Initial state: x(0) = [1, 0]
ocp.constraints.x0 = np.array([1.0, 0.0])

# --- Solver Options ---
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
# ocp.solver_options.qp_solver_iter_max = 50
ocp.solver_options.hessian_approx = 'EXACT'
ocp.solver_options.integrator_type = 'ERK'
ocp.solver_options.nlp_solver_type = 'SQP'
ocp.solver_options.globalization = 'MERIT_BACKTRACKING'

ocp_solver = AcadosOcpSolver(ocp)

# Initial guess
for i in range(ocp.solver_options.N_horizon):
    ocp_solver.set(i, "u", 0.)

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
    labels_list=['Catalyst Mixing'],
    x_labels=model.x_labels,
    u_labels=model.u_labels,
    idxbu=ocp.constraints.idxbu,
    lbu=ocp.constraints.lbu,
    ubu=ocp.constraints.ubu,
    fig_filename='catalyst_mixing_ocp.png',
)
