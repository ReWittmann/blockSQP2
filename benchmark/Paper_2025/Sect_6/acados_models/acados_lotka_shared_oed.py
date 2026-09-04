from acados_template import AcadosModel, AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver, plot_trajectories
import numpy as np
import casadi as ca
from pathlib import Path

def export_lotka_shared_oed_model() -> AcadosModel:
    model_name = 'lotka_shared_oed'
    
    x = ca.MX.sym('x', 3)
    x0, x1, x2 = ca.vertsplit(x)
    
    theta = ca.MX.sym('theta', 3)
    alpha0_s, alpha1_s, alpha2_s = ca.vertsplit(theta)
    
    alpha0 = 1.0
    alpha1 = 1.0
    alpha2 = 1.2
    c1 = 0.1
    c2 = 0.4
    
    reg_init = 0.1
    
    u = ca.MX.sym('u', 1)
    w = ca.MX.sym('w', 3)
    w1, w2, w3 = ca.vertsplit(w)
    
    f_expr = ca.vertcat(
        x0 - alpha0_s * x0 * x1 - x0 * x2,
        -x1 + alpha1_s * x0 * x1 - c1 * x1 * u, 
        -x2 + alpha2_s * x0 * x2 - c2 * x2 * u
    )
    
    f_x_expr = ca.jacobian(f_expr, x)
    f_theta_expr = ca.jacobian(f_expr, theta)
    
    f = ca.Function('f', [x, u, theta], [f_expr])
    f_x = ca.Function('f_x', [x, u, theta], [f_x_expr])
    f_theta = ca.Function('f_theta', [x, u, theta], [f_theta_expr])
    
    f_expr = f(x, u, ca.DM([alpha0, alpha1, alpha2]))
    f_x_expr = f_x(x, u, ca.DM([alpha0, alpha1, alpha2]))
    f_theta_expr = f_theta(x, u, ca.DM([alpha0, alpha1, alpha2]))
    
    G = ca.MX.sym('G', 3, 3)
    dG = f_x_expr @ G + f_theta_expr
    G_rhs = ca.vec(dG)
    
    F = ca.MX.sym('F', 6)
    dh1, dh2, dh3 = ca.DM([1,0,0]), ca.DM([0,1,0]), ca.DM([0,0,1])
    
    dF_mat = w1 * (dh1.T @ G).T @ (dh1.T @ G) + \
             w2 * (dh2.T @ G).T @ (dh2.T @ G) + \
             w3 * (dh3.T @ G).T @ (dh3.T @ G)
    
    F_rhs = ca.vertcat(dF_mat[0,0], dF_mat[1,0], dF_mat[2,0], 
                                    dF_mat[1,1], dF_mat[2,1], 
                                                 dF_mat[2,2])
    
    f_expl = ca.vertcat(f_expr, G_rhs, F_rhs, w)
    
    X = ca.vertcat(x, ca.vec(G), F, ca.MX.sym('z', 3))
    Xdot = ca.MX.sym('Xdot', f_expl.numel())
    f_impl = Xdot - f_expl
    
    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = X
    model.xdot = Xdot
    model.u = ca.vertcat(u, w)
    model.name = model_name
    
    F_tf = ca.MX.zeros(3,3)
    for j in range(3):
        for i in range(0, j):
            F_tf[i,j] = F[j + i*3 - (i*(i+1))//2]
        F_tf[j,j] = F[j*4 - (j*(j+1))//2] + reg_init
        for i in range(j + 1, 3):
            F_tf[i,j] = F[i + j*3 - (j*(j+1))//2]
    model.cost_expr_ext_cost_e = ca.trace(ca.inv(F_tf)) / theta.numel()
    
    model.x_labels = ['x', 'G', 'F', 'z']
    model.u_labels = ['u', 'w1', 'w2', 'w3']
    model.t_label = 't'
    
    return model

def setup_lotka_shared_oed_ocp():
    ocp = AcadosOcp()
    model = export_lotka_shared_oed_model()
    ocp.model = model
    
    Tf = 20.0
    N = 50
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf
    ocp.solver_options.qp_solver_cond_N = 1                #Only 1 seems to (somewhat) work
    ocp.solver_options.nlp_solver_max_iter = 300
    
    try:
        cD = Path(__file__).parent
    except:
        cD = Path.cwd()
    ocp.code_gen_options.code_export_directory = str(cD/Path(f"acados_codegen/{model.name}"))

    
    nx = model.x.rows()
    
    ocp.cost.cost_type_e = 'EXTERNAL'
    
    ocp.constraints.lbu = np.zeros(4)
    ocp.constraints.ubu = np.ones(4)
    ocp.constraints.idxbu = np.arange(4)
    
    x_init = np.array([1.5, 0.5, 1.0])
    ocp.constraints.x0 = np.concatenate([x_init, np.zeros(9 + 6 + 3)])
    
    M = np.array([4.0, 4.0, 4.0])
    ocp.constraints.lbx_e = np.zeros(3)
    ocp.constraints.ubx_e = M
    ocp.constraints.idxbx_e = np.arange(nx-3, nx)
    
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
    
    u_init_val = np.array([0.0, M[0]/Tf, M[1]/Tf, M[2]/Tf])
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
    ocp_solver = setup_lotka_shared_oed_ocp()
    status = ocp_solver.solve()
    ocp_solver.print_statistics()
    
    N = ocp_solver.ocp.solver_options.N_horizon
    nx = ocp_solver.ocp.model.x.numel()
    simX = np.zeros((N+1, nx))
    for i in range(N+1):
        simX[i,:] = ocp_solver.get(i, "x")
        
    plot_trajectories(
        x_traj_list=[simX[:, :3]],
        u_traj_list=[np.array([ocp_solver.get(i, "u") for i in range(N)])],
        time_traj_list=[np.linspace(0, 20.0, N+1)],
        time_label=ocp_solver.ocp.model.t_label,
        labels_list=['Lotka Shared OED'],
        x_labels=['x0', 'x1', 'x2'],
        u_labels=['u', 'w1', 'w2', 'w3'],
        idxbu=ocp_solver.ocp.constraints.idxbu,
        lbu=ocp_solver.ocp.constraints.lbu,
        ubu=ocp_solver.ocp.constraints.ubu,
        fig_filename='lotka_shared_oed.png',
    )
    
if __name__ == '__main__':
    main()