from acados_template import AcadosModel, AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver, plot_trajectories, ACADOS_INFTY
import numpy as np
import casadi as ca
from casadi import SX, vertcat
import time

# Parameters
m = 2.2
J = 0.05
r = 0.2
mg = 4.0
mu = 4.0
Tf_init = 5.0

def export_ducted_fan_model() -> AcadosModel:
    model_name = 'ducted_fan'

    x1 = SX.sym('x1')
    x2 = SX.sym('x2')
    alpha = SX.sym('alpha')
    v1 = SX.sym('v1')
    v2 = SX.sym('v2')
    va = SX.sym('va')
    q = SX.sym('q')
    T = SX.sym('T')
    X = vertcat(x1, x2, alpha, v1, v2, va, q, T)

    u1 = SX.sym('u1')
    u2 = SX.sym('u2')
    U = vertcat(u1, u2)

    x1_dot = SX.sym('x1_dot')
    x2_dot = SX.sym('x2_dot')
    alpha_dot = SX.sym('alpha_dot')
    v1_dot = SX.sym('v1_dot')
    v2_dot = SX.sym('v2_dot')
    va_dot = SX.sym('va_dot')
    q_dot = SX.sym('q_dot')
    T_dot = SX.sym('T_dot')
    Xdot = vertcat(x1_dot, x2_dot, alpha_dot, v1_dot, v2_dot, va_dot, q_dot, T_dot)
    
    f_expl = T * vertcat(
        v1,
        v2,
        va,
        (1/m) * (u1 * ca.cos(alpha) - u2 * ca.sin(alpha)),
        (1/m) * (-mg + u1 * ca.sin(alpha) + u2 * ca.cos(alpha)),
        (r/J) * u1,
        2*u1**2 + u2**2,
        0.0
    )
    
    f_impl = Xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = X
    model.xdot = Xdot
    model.u = U
    model.name = model_name
    
    model.cost_expr_ext_cost_e = 1/T * q + T*mu

    model.x_labels = ['x1', 'x2', 'alpha', 'v1', 'v2', 'va', 'q', 'T']
    model.u_labels = ['u1', 'u2']
    model.t_label = 'Normalized Time'

    return model

# --- OCP Setup ---
ocp = AcadosOcp()
model = export_ducted_fan_model()
ocp.model = model

Tf_scaled = 1.0 
N = 100
nx = model.x.rows()
nu = model.u.rows()

ocp.solver_options.N_horizon = N
ocp.solver_options.tf = Tf_scaled

# Cost
ocp.cost.cost_type_e = 'EXTERNAL'
# ocp.cost.cost_type = 'LINEAR_LS'

# ocp.cost.yref = np.zeros(2)
# ocp.cost.Vx = np.zeros((2, 7))
# ocp.cost.Vu = np.eye(2)
# ocp.cost.W = np.diag([2.0, 1.0])

# ocp.cost.cost_type_0 = 'LINEAR_LS'
# ocp.cost.yref_0 = np.zeros(2)
# ocp.cost.Vx_0 = np.zeros((2,7))
# ocp.cost.Vu_0 = np.eye(2)
# ocp.cost.W_0 = np.diag([2.0,1.0])

ocp.constraints.lbu = np.array([-5.0, 0.0])
ocp.constraints.ubu = np.array([5.0, 17.0])
ocp.constraints.idxbu = np.array([0, 1])


ocp.constraints.lbx = np.array([-30.0])
ocp.constraints.ubx = np.array([30.0])
ocp.constraints.idxbx = np.array([2])


ocp.constraints.lbx_0 = np.array([0, 0, 0, 0, 0, 0] + [0] + [1.0])
ocp.constraints.ubx_0 = np.array([0, 0, 0, 0, 0, 0] + [9] + [8.0])
ocp.constraints.idxbx_0 = np.arange(nx)


ocp.constraints.lbx_e = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
ocp.constraints.ubx_e = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
ocp.constraints.idxbx_e = np.arange(6)

# Solver Options
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
ocp.solver_options.hessian_approx = 'EXACT'
ocp.solver_options.integrator_type = 'ERK'
ocp.solver_options.nlp_solver_type = 'SQP'
ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
ocp.solver_options.nlp_solver_max_iter = 100


ocp_solver = AcadosOcpSolver(ocp)

#####

sim = AcadosSim()
sim.model = model
sim.solver_options.integrator_type = 'ERK'
sim.solver_options.T = Tf_scaled / N
sim.solver_options.num_steps = 2
sim_sol = AcadosSimSolver(sim)

x_current = np.array([0, 0, 0, 0, 0, 0,] + [0] + [Tf_init])
u_init_traj = np.full((N, nu), 1.0)
sim_x = np.zeros((N + 1, nx))
sim_x[0, :] = x_current

for i in range(N):
    x_next = sim_sol.simulate(x=x_current, u=u_init_traj[i])
    sim_x[i+1, :] = x_next
    x_current = x_next


#####

for i in range(N):
    ocp_solver.set(i, "x", np.array([0.]*6 + [0.] + [5.0]))
    ocp_solver.set(i, "u", np.array([1.0,1.0]))
ocp_solver.set(N, "x", np.array([0.]*6 + [0.] + [5.0]))

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
    x_traj_list=[simX[:,0:3]],
    u_traj_list=[simU],
    time_traj_list=[np.linspace(0, Tf_scaled, N+1) * simX[0,-1]],
    time_label='Time [s]',
    labels_list=['Ducted Fan'],
    x_labels=model.x_labels,
    u_labels=model.u_labels,
    idxbu=ocp.constraints.idxbu,
    lbu=ocp.constraints.lbu,
    ubu=ocp.constraints.ubu,
    fig_filename='ducted_fan_ocp.png',
)