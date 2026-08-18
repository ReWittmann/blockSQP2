from acados_template import AcadosModel, AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver, plot_trajectories, ACADOS_INFTY
import numpy as np
import casadi as ca
from casadi import SX, vertcat
import time

# Parameters
a_accel = 100.0
Tf_init = 1.0

def export_particle_steering_model() -> AcadosModel:
    model_name = 'particle_steering'

    # States: x1, v1, x2, v2, T
    x1 = SX.sym('x1')
    v1 = SX.sym('v1')
    x2 = SX.sym('x2')
    v2 = SX.sym('v2')
    T = SX.sym('T')
    X = vertcat(x1, v1, x2, v2, T)

    # Control: u (thrust angle)
    u = SX.sym('u')
    U = vertcat(u)

    # xdot symbols
    Xdot = SX.sym('Xdot', 5)
    
    # Dynamics scaled by T
    # dx1/dt = v1
    # dv1/dt = a * cos(u)
    # dx2/dt = v2
    # dv2/dt = a * sin(u)
    # dT/dt = 0
    f_expl = T * vertcat(
        v1,
        a_accel * ca.cos(u),
        v2,
        a_accel * ca.sin(u),
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
    
    # Objective: min Tf -> minimize final value of state T
    model.cost_expr_ext_cost_e = T

    model.x_labels = ['x1', 'v1', 'x2', 'v2', 'T']
    model.u_labels = ['u']
    model.t_label = 'Normalized Time'

    return model

# --- OCP Setup ---
ocp = AcadosOcp()
model = export_particle_steering_model()
ocp.model = model

Tf_scaled = 1.0 
N = 100
nx = model.x.rows()
nu = model.u.rows()

ocp.solver_options.N_horizon = N
ocp.solver_options.tf = Tf_scaled

ocp.cost.cost_type_e = 'EXTERNAL'

# Constraints
# Control: -pi/2 <= u <= pi/2
ocp.constraints.lbu = np.array([-np.pi/2])
ocp.constraints.ubu = np.array([np.pi/2])
ocp.constraints.idxbu = np.array([0])

# Initial state: [0, 0, 0, 0, Tf_init]
ocp.constraints.lbx_0 = np.array([0.0, 0.0, 0.0, 0.0, 1e-3])
ocp.constraints.ubx_0 = np.array([0.0, 0.0, 0.0, 0.0, ACADOS_INFTY])
ocp.constraints.idxbx_0 = np.arange(nx)

# Final state constraints:
# x2(tF) = 5, v1(tF) = 45, v2(tF) = 0
# Indices in X: x1(0), v1(1), x2(2), v2(3), T(4)
ocp.constraints.lbx_e = np.array([45.0, 5.0, 0.0])
ocp.constraints.ubx_e = np.array([45.0, 5.0, 0.0])
ocp.constraints.idxbx_e = np.array([1, 2, 3])

# Solver Options
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
ocp.solver_options.hessian_approx = 'EXACT'
ocp.solver_options.integrator_type = 'ERK'
ocp.solver_options.nlp_solver_type = 'SQP'
ocp.solver_options.globalization = 'MERIT_BACKTRACKING'

ocp_solver = AcadosOcpSolver(ocp)

# --- Automatic Initialization x = S(u) ---
sim = AcadosSim()
sim.model = model
sim.solver_options.integrator_type = 'ERK'
sim.solver_options.T = Tf_scaled / N
sim.solver_options.num_steps = 2
sim_sol = AcadosSimSolver(sim)

u_init_traj = np.zeros(N) # u_init = 0
x_current = np.array([0.0, 0.0, 0.0, 0.0, Tf_init])
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
    x_traj_list=[simX[:,:-1]],
    u_traj_list=[simU],
    time_traj_list=[np.linspace(0, Tf_scaled, N+1) * simX[0,-1]],
    time_label='Time [s]',
    labels_list=['Particle Steering'],
    x_labels=model.x_labels,
    u_labels=model.u_labels,
    idxbu=ocp.constraints.idxbu,
    lbu=ocp.constraints.lbu,
    ubu=ocp.constraints.ubu,
    fig_filename='particle_steering_ocp.png',
)