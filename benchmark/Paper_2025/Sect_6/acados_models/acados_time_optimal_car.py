from acados_template import AcadosModel, AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver, plot_trajectories, ACADOS_INFTY
import numpy as np
import casadi as ca
from casadi import SX, vertcat
import time

# Parameters
v_max = 33.0
z_target = 300.0
Tf_init = 10.0

def export_time_optimal_car_model() -> AcadosModel:
    model_name = 'time_optimal_car'

    # States: z (position), v (velocity), T (time scaling)
    z = SX.sym('z')
    v = SX.sym('v')
    T = SX.sym('T')
    X = vertcat(z, v, T)

    # Control: u (acceleration)
    u = SX.sym('u')
    U = vertcat(u)

    # xdot symbols
    Xdot = SX.sym('Xdot', 3)
    
    # Dynamics scaled by T
    # dz/dt = v
    # dv/dt = u
    # dT/dt = 0
    f_expl = T * vertcat(
        v,
        u,
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

    model.x_labels = ['Position (z)', 'Velocity (v)', 'Time (T)']
    model.u_labels = ['Acceleration (u)']
    model.t_label = 'Normalized Time'

    return model

# --- OCP Setup ---
ocp = AcadosOcp()
model = export_time_optimal_car_model()
ocp.model = model

Tf_scaled = 1.0 
N = 100
nx = model.x.rows()
nu = model.u.rows()

ocp.solver_options.N_horizon = N
ocp.solver_options.tf = Tf_scaled

ocp.cost.cost_type_e = 'EXTERNAL'

# Constraints
# Control: -2 <= u <= 1
ocp.constraints.lbu = np.array([-2.0])
ocp.constraints.ubu = np.array([1.0])
ocp.constraints.idxbu = np.array([0])

# State constraints: 0 <= z <= 330, 0 <= v <= v_max
ocp.constraints.lbx = np.array([0.0, 0.0])
ocp.constraints.ubx = np.array([330.0, v_max])
ocp.constraints.idxbx = np.array([0, 1])

# Initial state: [0, 0, Tf_init]
ocp.constraints.lbx_0 = np.array([0.0, 0.0, 1e-3])
ocp.constraints.ubx_0 = np.array([0.0, 0.0, ACADOS_INFTY])
ocp.constraints.idxbx_0 = np.arange(nx)

# Final state constraints: z(tF) = 300, v(tF) = 0
ocp.constraints.lbx_e = np.array([z_target, 0.0])
ocp.constraints.ubx_e = np.array([z_target, 0.0])
ocp.constraints.idxbx_e = np.array([0, 1])

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
x_current = np.array([0.0, 0.0, Tf_init])
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
    labels_list=['Time Optimal Car'],
    x_labels=model.x_labels,
    u_labels=model.u_labels,
    idxbu=ocp.constraints.idxbu,
    lbu=ocp.constraints.lbu,
    ubu=ocp.constraints.ubu,
    fig_filename='time_optimal_car_ocp.png',
)