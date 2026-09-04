from acados_template import AcadosModel, AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver, plot_trajectories
import numpy as np
import casadi as ca
from pathlib import Path


def export_catalyst_mixing_oed_model() -> AcadosModel:
    model_name = 'catalyst_mixing_oed'
    
    x = ca.MX.sym('x', 2)
    x1, x2 = ca.vertsplit(x)
    
    theta = ca.MX.sym('theta', 2)
    p1_s, p2_s = ca.vertsplit(theta)
    
    p1val = 1.0
    p2val = 10.0
    p3val = 1.0
    
    u = ca.MX.sym('u')
    w = ca.MX.sym('w', 2)
    w1, w2 = ca.vertsplit(w)
    
    f_expr = ca.vertcat(
        u * (p2_s * x2 - p1_s * x1),
        u * (p1_s * x1 - p2_s * x2) - (1 - u) * p3val * x2
    )
    
    f_x_expr = ca.jacobian(f_expr, x)
    f_theta_expr = ca.jacobian(f_expr, theta)
    
    f = ca.Function('f', [x,u,theta], [f_expr])
    f_x = ca.Function('f', [x,u,theta], [f_x_expr])
    f_theta = ca.Function('f', [x,u,theta], [f_theta_expr])
    
    f_expr = f(x, u, ca.DM([p1val, p2val]))
    f_x_expr = f_x(x, u, ca.DM([p1val, p2val]))
    f_theta_expr = f_theta(x, u, ca.DM([p1val, p2val]))
    
    G = ca.MX.sym('G', x.numel(), theta.numel())
    dG = f_x_expr@G + f_theta_expr
    G_rhs = ca.vec(dG)
    
    w = ca.MX.sym('w', 2)
    w1,w2 = ca.vertsplit(w)
    
    F = ca.MX.sym('F', (theta.numel()*(theta.numel() + 1))//2)
    dh1, dh2 = ca.DM([1,0]), ca.DM([0,1])
    dF = w1*(dh1.T@G).T @ (dh1.T@G) + w2*(dh2.T@G).T @ (dh2.T@G)
    
    
    F_rhs = ca.vertcat(dF[0,0], dF[1,0], dF[1,1])
    
    qstates = ca.MX.sym('qstates', 2)
    quad_rhs = w
    
    f_expl = ca.vertcat(f_expr, G_rhs, F_rhs, quad_rhs)
    
    X = ca.vertcat(x, ca.vec(G), F, qstates)
    Xdot = ca.MX.sym('Xdot', f_expl.numel())
    f_impl = Xdot - f_expl
    
    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = X
    model.xdot = Xdot
    model.u = ca.vertcat(u, w)
    model.name = model_name
    
    F_tf = ca.MX.zeros(2, 2)
    idx = 0
    for j in range(2):
        for i in range(j + 1):
            F_tf[i, j] = F[idx]
            F_tf[j, i] = F[idx]
            idx += 1
    
    F_reg = F_tf + 1e-2 * ca.diag(np.ones(2))
    
    
    model.cost_expr_ext_cost_e = ca.trace(ca.inv(F_reg))
    
    
    model.x_labels = ['x1', 'x2', 'G', 'F', 'z1', 'z2']
    model.u_labels = ['u', 'w1', 'w2']
    model.t_label = 't'
    
    return model


def setup_catalyst_mixing_oed_ocp():
    ocp = AcadosOcp()
    model = export_catalyst_mixing_oed_model()
    ocp.model = model
    
    ocp.solver_options.N_horizon = 40          #N = 40 seems to work
    ocp.solver_options.tf = 1.0
    ocp.solver_options.qp_solver_cond_N = 10     #10 seems to work
    
    N = ocp.solver_options.N_horizon
    nx = model.x.rows()
    Tf = ocp.solver_options.tf

    try:
        cD = Path(__file__).parent
    except:
        cD = Path.cwd()
    ocp.code_gen_options.code_export_directory = str(cD/Path(f"acados_codegen/{model.name}"))

    
    M1 = 0.2
    M2 = 0.2
    
    ocp.cost.cost_type_e = 'EXTERNAL'
    
    ocp.constraints.lbu = np.array([0.0, 0.0, 0.0])
    ocp.constraints.ubu = np.array([1.0, 1.0, 1.0])
    ocp.constraints.idxbu = np.array([0, 1, 2])
    
    ocp.constraints.x0 = np.array([1.0, 0.0] + [0.0]*4 + [0.0]*3 + [0.0]*2)
    
    ocp.constraints.lbx_e = np.array([0.0, 0.0])
    ocp.constraints.ubx_e = np.array([M1, M2])
    ocp.constraints.idxbx_e = np.array([nx-2, nx-1])
    
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    # ocp.solver_options.qp_solver_iter_max = 200  #Usually hits the default limit of 50, but this does not seem to matter
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    ocp.solver_options.nlp_solver_max_iter = 15 
    
    ocp_solver = AcadosOcpSolver(ocp)
    
    # --- Automatic Initialization ---
    sim = AcadosSim()
    sim.model = model
    sim.code_gen_options.code_export_directory = str(cD/Path(f"acados_codegen/{model.name}_sim"))
    
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.T = Tf / N
    sim.solver_options.num_steps = 2
    sim_sol = AcadosSimSolver(sim)
    
    # Start point: u=0.5, w = M/Tf
    u_init_val = np.array([0.5, M1/Tf, M2/Tf])
    u_init_traj = np.tile(u_init_val, (N, 1))
    
    x_current = ocp.constraints.x0
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
    return ocp_solver

def main():
    
    ocp_solver = setup_catalyst_mixing_oed_ocp()

    status = ocp_solver.solve()
    ocp_solver.print_statistics()
    
    N = ocp_solver.ocp.solver_options.N_horizon
    tf = ocp_solver.ocp.solver_options.tf
    nx = ocp_solver.ocp.model.x.numel()
    nu = ocp_solver.ocp.model.u.numel()
    
    # --- Extract and Plot ---
    simX = np.zeros((N+1, nx))
    simU = np.zeros((N, nu))
    for i in range(N):
        simX[i,:] = ocp_solver.get(i, "x")
        simU[i,:] = ocp_solver.get(i, "u")
    simX[N,:] = ocp_solver.get(N, "x")
    
    plot_trajectories(
        x_traj_list=[simX[:, :2]], # Plot only concentrations
        u_traj_list=[simU],
        time_traj_list=[np.linspace(0, tf, N+1)],
        time_label=ocp_solver.ocp.model.t_label,
        labels_list=['Catalyst Mixing OED'],
        x_labels=['x1', 'x2'],
        u_labels=ocp_solver.ocp.model.u_labels,
        idxbu=ocp_solver.ocp.constraints.idxbu,
        lbu=ocp_solver.ocp.constraints.lbu,
        ubu=ocp_solver.ocp.constraints.ubu,
        fig_filename='acados_plots/catalyst_mixing_oed_ocp.png',
    )
    
if __name__ == '__main__':
	main()