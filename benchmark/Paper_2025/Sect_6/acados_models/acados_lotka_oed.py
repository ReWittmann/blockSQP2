from acados_template import AcadosModel, AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver, plot_trajectories, ACADOS_INFTY
import numpy as np
import casadi as ca
from casadi import SX, vertcat
import time

# Parameters
Tf = 12.0
p1 = p2 = p3 = p4 = 1.0
c1 = 0.4
c2 = 0.2
M_max = 4.0

def export_lotka_oed_model() -> AcadosModel:
    model_name = 'lotka_oed'

    x1 = SX.sym('x1')
    x2 = SX.sym('x2')
    G11 = SX.sym('G11')
    G12 = SX.sym('G12')
    G21 = SX.sym('G21')
    G22 = SX.sym('G22')
    F11 = SX.sym('F11')
    F12 = SX.sym('F12')
    F22 = SX.sym('F22')
    z1 = SX.sym('z1')
    z2 = SX.sym('z2')
    X = vertcat(x1, x2, G11, G12, G21, G22, F11, F12, F22, z1, z2)

    # Controls: u (harvesting), w1, w2 (measurement weights)
    u = SX.sym('u')
    w1 = SX.sym('w1')
    w2 = SX.sym('w2')
    U = vertcat(u, w1, w2)

    Xdot = SX.sym('Xdot', 11)
    
    # Dynamics
    # Population dynamics
    dx1 = p1*x1 - p2*x1*x2 - c1*u*x1
    dx2 = -p3*x2 + p4*x1*x2 - c2*u*x2
    
    # Sensitivity dynamics (G)
    # dG/dt = (df/dx)G
    dfdx11 = p1 - p2*x2 - c1*u
    dfdx12 = -p2*x1
    dfdx21 = p4*x2
    dfdx22 = -p3 + p4*x1
    
    dx1dp2 = -x1*x2
    dx1dp4 = 0
    dx2dp2 = 0
    dx2dp4 = x1*x2
    
    dG11 = dfdx11 * G11 + dfdx12 * G21 + dx1dp2
    dG12 = dfdx11 * G12 + dfdx12 * G22 + dx1dp4
    dG21 = dfdx21 * G11 + dfdx22 * G21 + dx2dp2
    dG22 = dfdx21 * G12 + dfdx22 * G22 + dx2dp4
    
    # Fisher Information Matrix dynamics (F)
    dF11 = w1 * G11**2 + w2 * G21**2
    dF12 = w1 * G11 * G12 + w2 * G21 * G22
    dF22 = w1 * G12**2 + w2 * G22**2
    
    # Measurement effort integration
    dz1 = w1
    dz2 = w2

    f_expl = vertcat(dx1, dx2, dG11, dG12, dG21, dG22, dF11, dF12, dF22, dz1, dz2)
    f_impl = Xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = X
    model.xdot = Xdot
    model.u = U
    model.name = model_name
    
    # Objective: trace(F^-1)
    # F = [[F11, F12], [F12, F22]]
    # F^-1 = (1/det) * [[F22, -F12], [-F12, F11]]
    # trace(F^-1) = (F11 + F22) / (F11*F22 - F12**2)
    det_F = F11 * F22 - F12**2
    model.cost_expr_ext_cost_e = (F11 + F22) / (det_F)

    model.x_labels = ['x1', 'x2', 'G11', 'G12', 'G21', 'G22', 'F11', 'F12', 'F22', 'z1', 'z2']
    model.u_labels = ['u', 'w1', 'w2']
    model.t_label = 'Time'

    return model

# --- OCP Setup ---
ocp = AcadosOcp()
model = export_lotka_oed_model()
ocp.model = model

N = 75
nx = model.x.rows()
nu = model.u.rows()

ocp.solver_options.N_horizon = N
ocp.solver_options.tf = Tf

ocp.cost.cost_type_e = 'EXTERNAL'

# Constraints
# 0 <= u <= 1, 0 <= w1 <= 1, 0 <= w2 <= 1
ocp.constraints.lbu = np.array([0.0, 0.0, 0.0])
ocp.constraints.ubu = np.array([1.0, 1.0, 1.0])
ocp.constraints.idxbu = np.array([0, 1, 2])

# Initial state: x=(0.5, 0.7), G=0, F=0, z=0
x0 = np.array([0.5, 0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0])
ocp.constraints.x0 = x0

ocp.constraints.lbx = np.array([0.,0.])
ocp.constraints.ubx = np.array([ACADOS_INFTY, ACADOS_INFTY])
ocp.constraints.idxbx = np.arange(2)

ocp.constraints.lbx_e = np.array([0., 0.])
ocp.constraints.ubx_e = np.array([M_max, M_max])
ocp.constraints.idxbx_e = np.arange(nx-2, nx)

# Solver Options
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
ocp.solver_options.hessian_approx = 'EXACT'
ocp.solver_options.integrator_type = 'ERK'
ocp.solver_options.nlp_solver_type = 'SQP'
ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
ocp.solver_options.nlp_solver_max_iter = 300

ocp_solver = AcadosOcpSolver(ocp)

# --- Automatic Initialization x = S(u) ---
sim = AcadosSim()
sim.model = model
sim.solver_options.integrator_type = 'ERK'
sim.solver_options.T = Tf / N
sim.solver_options.num_steps = 2
sim_sol = AcadosSimSolver(sim)

u_init_val = np.array([0.0, 1/3, 1/3])
u_init_traj = np.tile(u_init_val, (N, 1))

x_current = x0
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
t0 = time.time()
status = ocp_solver.solve()
t1 = time.time()
ocp_solver.print_statistics()

# --- Extract and Plot ---
simX = np.zeros((N+1, nx))
simU = np.zeros((N, nu))
for i in range(N):
    simX[i,:] = ocp_solver.get(i, "x")
    simU[i,:] = ocp_solver.get(i, "u")
simX[N,:] = ocp_solver.get(N, "x")

plot_trajectories(
    x_traj_list=[simX[:,0:2]],
    u_traj_list=[simU],
    time_traj_list=[np.linspace(0, Tf, N+1)],
    time_label=model.t_label,
    labels_list=['Lotka OED'],
    x_labels=model.x_labels,
    u_labels=model.u_labels,
    idxbu=ocp.constraints.idxbu,
    lbu=ocp.constraints.lbu,
    ubu=ocp.constraints.ubu,
    fig_filename='lotka_oed_ocp.png',
)