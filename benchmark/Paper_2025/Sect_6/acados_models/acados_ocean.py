from acados_template import AcadosModel, AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver, ACADOS_INFTY, plot_trajectories
import numpy as np
import casadi as ca
from casadi import SX, vertcat
from pathlib import Path

def export_ocean_model() -> AcadosModel:
    model_name = 'ocean'

    rho = 0.03
    gamma = 0.001
    omega = 0.1
    b = 50.
    mu = 0.5
    a1 = 2.
    a2 = 2.
    nu = 1.
    c1 = 50.
    c2 = 0.004
    Spreind = 600.
    S0 = 2000.
    R0 = 1e4
    DL0 = 2.3e4

    S = SX.sym('S')
    R = SX.sym('R')
    t = SX.sym('t')
    q = SX.sym('q')
    X = vertcat(S, R, t, q)

    u1 = SX.sym('u1')
    u2 = SX.sym('u2')
    u = vertcat(u1, u2)

    
    DL = DL0 + R0 + S0 - R - S

    U = b*u1 - mu*u1**2
    A = a1*u2 + a2*u2**2
    C = c1 - c2*R
    D = nu*(0.3*S - Spreind)**2
    
    f_expl = vertcat(
        u1 - u2 - gamma*(S - omega*DL),
        -u1,
        1.0,
        ca.exp(-rho*t)*(U - A - u1*C - D)
    )
    Xdot = SX.sym('Xdot', 4)
    f_impl = Xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = X
    model.xdot = Xdot
    model.u = u
    model.name = model_name
    
    model.cost_expr_ext_cost_e = -q

    model.x_labels = ['S', 'R', 't', 'q']
    model.u_labels = ['u1', 'u2']
    model.t_label = 't'

    return model

def setup_ocean_ocp():
    ocp = AcadosOcp()
    model = export_ocean_model()
    ocp.model = model
    
    S0 = 2000.
    R0 = 1e4
    Tf = 400.0

    N = 50    
    nx = model.x.rows()
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf
    # ocp.solver_options.qp_solver_cond_N = 4
    
    try:
        cD = Path(__file__).parent
    except:
        cD = Path.cwd()
    ocp.code_gen_options.code_export_directory = str(cD/Path(f"acados_codegen/{model.name}"))

    
    ocp.cost.cost_type_e = 'EXTERNAL'
    
    ocp.constraints.lbu = np.array([0.0, 0.0])
    ocp.constraints.ubu = np.array([40.0, 40.0])
    ocp.constraints.idxbu = np.array([0, 1])
    
    ocp.constraints.x0 = np.array([S0, R0, 0.0, 0.0])
    
    ocp.constraints.lbx = np.array([0.0, 0.0])
    ocp.constraints.ubx = np.array([1e5, 1e5])
    ocp.constraints.idxbx = np.arange(2)
    
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    
    ocp_solver = AcadosOcpSolver(ocp)
    
    sim = AcadosSim()
    sim.model = model
    sim.code_gen_options.code_export_directory = str(cD/Path(f"acados_codegen/{model.name}_sim"))
    
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.T = Tf / N
    sim.solver_options.num_steps = 2
    sim_sol = AcadosSimSolver(sim)
    
    u_init = np.array([30.0, 10.0])
    u_init_traj = np.tile(u_init, (N, 1))
    
    x_current = np.array(ocp.constraints.x0)
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
    
    # for i in range(N):
    #     ocp_solver.set(i, "x", np.array([S0, R0, 0.0, 0.0]))
    #     ocp_solver.set(i, "u", np.array([30.0, 10.0]))
    # ocp_solver.set(N, "x", np.array([S0, R0, 0.0, 0.0]))
    
    return ocp_solver

def main():
    ocp_solver = setup_ocean_ocp()
    status = ocp_solver.solve()
    ocp_solver.print_statistics()
    
    N = ocp_solver.ocp.solver_options.N_horizon
    tf = ocp_solver.ocp.solver_options.tf
    nx = ocp_solver.ocp.model.x.numel()
    nu = ocp_solver.ocp.model.u.numel()
    
    simX = np.zeros((N+1, nx))
    simU = np.zeros((N, nu))
    for i in range(N):
        simX[i,:] = ocp_solver.get(i, "x")
        simU[i,:] = ocp_solver.get(i, "u")
    simX[N,:] = ocp_solver.get(N, "x")
    
    plot_trajectories(
        x_traj_list=[simX[:, :2]],
        u_traj_list=[simU],
        time_traj_list=[np.linspace(0, tf, N+1)],
        time_label=ocp_solver.ocp.model.t_label,
        labels_list=['Ocean Model'],
        x_labels=['S', 'R'],
        u_labels=ocp_solver.ocp.model.u_labels,
        idxbu=ocp_solver.ocp.constraints.idxbu,
        lbu=ocp_solver.ocp.constraints.lbu,
        ubu=ocp_solver.ocp.constraints.ubu,
        fig_filename='ocean_ocp.png',
    )
    
if __name__ == '__main__':
    main()