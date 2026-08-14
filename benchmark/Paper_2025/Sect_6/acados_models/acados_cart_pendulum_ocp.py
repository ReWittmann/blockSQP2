from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, plot_trajectories, ACADOS_INFTY
import numpy as np
import casadi as ca
from casadi import SX, vertcat, cos, sin
import time

def export_cart_pendulum_model(lambda_u = 0.5) -> AcadosModel:
    model_name = 'cart_pendulum'

    # States: x (position), theta (angle), v (velocity), w (angular velocity)
    x_pos = SX.sym('x_pos')
    theta = SX.sym('theta')
    v = SX.sym('v')
    w = SX.sym('w')
    q = SX.sym('q')
    x = vertcat(x_pos, theta, v, w, q)

    # Control: u (acceleration)
    u_sym = SX.sym('u')
    u = vertcat(u_sym)

    # xdot symbols
    x_dot = SX.sym('x_dot')
    theta_dot = SX.sym('theta_dot')
    v_dot = SX.sym('v_dot')
    w_dot = SX.sym('w_dot')
    q_dot = SX.sym('q_dot')
    
    xdot = vertcat(x_dot, theta_dot, v_dot, w_dot, q_dot)

    # Parameters
    M = 1.0
    m = 0.1
    l = 1.0
    g = 9.81

    # Dynamics
    # 1. x_dot = v
    f0 = v
    # 2. theta_dot = w
    f1 = w
    # 3. v_dot calculation
    # Note: 1 - cos(theta)^2 is sin(theta)^2
    v_dot_expr = (u_sym + m * g * cos(theta) * sin(theta) + m * l * w**2 * sin(theta)) / (M + m * (1 - cos(theta)**2))
    f2 = v_dot_expr
    # 4. w_dot calculation (depends on v_dot)
    f3 = (-g * sin(theta) - v_dot_expr * cos(theta)) / l

    f4 = 10*x_pos**2 + 50*(theta - ca.pi)**2 + lambda_u*u**2

    f_expl = vertcat(f0, f1, f2, f3, f4)
    f_impl = xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    # Path Cost: 10*x^2 + 50*(theta - pi)^2 + lambda_u * u^2
    # We define y = [x, theta, u]
    # model.cost_y_expr = vertcat(x_pos, theta, u_sym)
    model.cost_expr_ext_cost_e = q

    model.x_labels = [r'Position [m]', r'Angle [rad]', r'Velocity [m/s]', r'Ang. Vel [rad/s]', r'q']
    model.u_labels = [r'$Acceleration [m/s^2]$']
    model.t_label = r'Time [s]'

    return model

u_max = 30
lambda_u = 0.5
# --- OCP Setup ---
ocp = AcadosOcp()
model = export_cart_pendulum_model(lambda_u)
ocp.model = model

Tf = 4.0
N = 100
nx = model.x.rows()
nu = model.u.rows()

ocp.solver_options.N_horizon = N
ocp.solver_options.tf = Tf
ocp.solver_options.qp_solver_cond_N = 20


ocp.cost.cost_type_e = 'EXTERNAL'

ocp.constraints.lbu = np.array([-u_max])
ocp.constraints.ubu = np.array([u_max])
ocp.constraints.idxbu = np.array([0])


ocp.constraints.lbx = np.array([-2.0])
ocp.constraints.ubx = np.array([2.0])
ocp.constraints.idxbx = np.array([0])

x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) 
ocp.constraints.x0 = x0

# --- Solver Options ---
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
ocp.solver_options.hessian_approx = 'EXACT'
ocp.solver_options.integrator_type = 'IRK'
ocp.solver_options.nlp_solver_type = 'SQP'
ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
ocp.solver_options.nlp_solver_max_iter = 1000

ocp_solver = AcadosOcpSolver(ocp)

for i in range(N):
    ocp_solver.set(i, "u", 0.0)
for i in range(N+1):
    ocp_solver.set(i, "x", np.array([0.,0.,0.,0.,0.]))

t0 = time.time()
status = ocp_solver.solve()
t1 = time.time()


ocp_solver.print_statistics()


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
    labels_list=['Cart Pendulum'],
    x_labels=model.x_labels,
    u_labels=model.u_labels,
    idxbu=ocp.constraints.idxbu,
    lbu=ocp.constraints.lbu,
    ubu=ocp.constraints.ubu,
    fig_filename='cart_pendulum_ocp.png',
)