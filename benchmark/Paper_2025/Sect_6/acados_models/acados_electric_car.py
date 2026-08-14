from acados_template import AcadosModel, AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver, plot_trajectories
import numpy as np
import casadi as ca
from casadi import SX, vertcat, exp
import time

def export_electric_car_model() -> AcadosModel:
    model_name = 'electric_car'

    Kr = 10
    rho = 1.293
    Cx = 0.4
    S = 2
    r = 0.33
    Kf = 0.03
    Km = 0.27
    Rm = 0.03
    Lm = 0.05
    M = 250
    g = 9.81
    Valim = 150
    Rbat = 0.05


    # States: x1 (Substance A), x2 (Substance B)
    x0 = SX.sym('x0')
    x1 = SX.sym('x1')
    x2 = SX.sym('x2')
    x3 = SX.sym('x3')
    x = vertcat(x0, x1, x2, x3)
    
    u = SX.sym('u')
    
    # xdot symbols
    x0_dot = SX.sym('x0_dot')
    x1_dot = SX.sym('x1_dot')
    x2_dot = SX.sym('x2_dot')
    x3_dot = SX.sym('x3_dot')
    xdot = vertcat(x0_dot, x1_dot, x2_dot, x3_dot)
    
    f_expl = ca.vertcat((Valim*u - Rm*x0-Km*x1)/Lm,
                         (Kr**2)/(M*r**2) * (Km*x0 - r/Kr*(M*g*Kf + 0.5*rho*S*Cx*r**2/Kr**2 * x1**2)),
                         r/Kr * x1,
                         Valim*u*x0 + Rbat*x0**2
                         )
    f_impl = xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name
    
    model.cost_expr_ext_cost_e = x3

    model.x_labels = ['x0', 'x1', 'x2', 'x3']
    model.u_labels = ['Control']
    model.t_label = 'Time [s]'

    return model

ocp = AcadosOcp()
model = export_electric_car_model()
ocp.model = model

Tf = 10.0
N = 100
nx = model.x.rows()
nu = model.u.rows()

ocp.solver_options.N_horizon = N
ocp.solver_options.tf = Tf

ocp.cost.cost_type_e = 'EXTERNAL'

# --- Constraints ---
# Control constraints: 298 <= u <= 398
ocp.constraints.lbu = np.array([-1.0])
ocp.constraints.ubu = np.array([ 1.0])
ocp.constraints.idxbu = np.array([0.])

# Initial state: x(0) = [1, 0]
ocp.constraints.x0 = np.array([0.,0.,0.,0.])

ocp.constraints.lbx = np.array([-150.])
ocp.constraints.ubx = np.array([150.])
ocp.constraints.idxbx = np.array([0.])

ocp.constraints.lbx_e = np.array([100.])
ocp.constraints.ubx_e = np.array([100.])
ocp.constraints.idxbx_e = np.array([2])


# --- Solver Options ---
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
ocp.solver_options.hessian_approx = 'EXACT'
ocp.solver_options.integrator_type = 'ERK'
ocp.solver_options.nlp_solver_type = 'SQP'
ocp.solver_options.globalization = 'MERIT_BACKTRACKING'

ocp_solver = AcadosOcpSolver(ocp)

# Initial guess
for i in range(ocp.solver_options.N_horizon):
    ocp_solver.set(i, "u", 0.1 + 0.9*i/(ocp.solver_options.N_horizon))

sim = AcadosSim()
sim.model = model

sim.solver_options.integrator_type = 'ERK'
sim.solver_options.T = Tf/N
sim.solver_options.num_steps = 2

x_current = np.array([0., 0., 0., 0.])
u_init = np.zeros((N, nu))
for i in range(N):
    u_init[i, 0] = 0.1 + 0.9 * i / N


sim_x = np.zeros((N + 1, nx))
sim_x[0, :] = x_current
sim_sol = AcadosSimSolver(sim)

for i in range(N):
    x_next = sim_sol.simulate(x = x_current, u = u_init[i])
    sim_x[i+1, :] = x_next
    x_current = x_next

for i in range(N):
    if sim_x[i, 0] > 150:
        sim_x[i, 0] = 150
    ocp_solver.set(i, "x", sim_x[i, :])
    ocp_solver.set(i, "u", u_init[i])
ocp_solver.set(N, "x", sim_x[N, :])

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
    labels_list=['Electric Car'],
    x_labels=model.x_labels,
    u_labels=model.u_labels,
    idxbu=ocp.constraints.idxbu,
    lbu=ocp.constraints.lbu,
    ubu=ocp.constraints.ubu,
    fig_filename='catalyst_mixing_ocp.png',
)
