from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, plot_trajectories
import numpy as np
import casadi as ca
from casadi import SX, vertcat
import time


m = 5.
c = 10.
x0 = 2.
v0 = 5.
umm = 5.

def export_cushioned_oscillation_model() -> AcadosModel:
    model_name = 'cushioned_oscillation'

    x = SX.sym('x')
    v = SX.sym('v')
    T = SX.sym('T')
    X = vertcat(x, v, T)

    u_sym = SX.sym('u')
    u = vertcat(u_sym)

    # xdot symbols
    x_dot = SX.sym('x_dot')
    v_dot = SX.sym('v_dot')
    T_dot = SX.sym('T_dot')
    Xdot = vertcat(x_dot, v_dot, T_dot)
    
    f_expl = T*ca.vertcat(v, 1/m * (u - c*x), 0.)
    f_impl = Xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = X
    model.xdot = Xdot
    model.u = u
    model.name = model_name
    
    # model.cost_y_expr_e = x2
    model.cost_expr_ext_cost_e = T

    model.x_labels = ['position', 'velocity', 'Time']
    model.u_labels = ['Control']
    model.t_label = 'Time [s]'

    return model

ocp = AcadosOcp()
model = export_cushioned_oscillation_model()
ocp.model = model

Tf = 1.0
N = 40
nx = model.x.rows()
nu = model.u.rows()

ocp.solver_options.N_horizon = N
ocp.solver_options.tf = Tf
# ocp.solver_options.qp_solver_cond_N = 10

ocp.cost.cost_type_e = 'EXTERNAL'

ocp.constraints.lbu = np.array([-umm])
ocp.constraints.ubu = np.array([umm])
ocp.constraints.idxbu = np.array([0])

ocp.constraints.lbx_0 = np.array([x0, v0] + [8.0])
ocp.constraints.ubx_0 = np.array([x0, v0] + [20.0])
ocp.constraints.idxbx_0 = np.arange(3)

ocp.constraints.lbx_e = np.array([0.,0.])
ocp.constraints.ubx_e = np.array([0.,0.])
ocp.constraints.idxbx_e = np.arange(2)


ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
ocp.solver_options.hessian_approx = 'EXACT'
ocp.solver_options.integrator_type = 'ERK'
ocp.solver_options.nlp_solver_type = 'SQP'
ocp.solver_options.globalization = 'MERIT_BACKTRACKING'

ocp_solver = AcadosOcpSolver(ocp)

# Initial guess
for i in range(N):
    ocp_solver.set(i, "u", 0.)
for i in range(N+1):
    ocp_solver.set(i, "x", np.array([x0, v0, 10.0]))

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
    x_traj_list=[simX[:,:-1]],
    u_traj_list=[simU],
    time_traj_list=[np.linspace(0, Tf, N+1) * simX[0,-1]],
    time_label=model.t_label,
    labels_list=['Cushioned Oscillation'],
    x_labels=model.x_labels,
    u_labels=model.u_labels,
    idxbu=ocp.constraints.idxbu,
    lbu=ocp.constraints.lbu,
    ubu=ocp.constraints.ubu,
    fig_filename='cushioned_oscillation_ocp.png',
)
