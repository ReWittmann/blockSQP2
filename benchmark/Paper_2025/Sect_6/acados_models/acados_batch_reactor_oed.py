from acados_template import AcadosModel, AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver, plot_trajectories
import numpy as np
import casadi as ca
import time
from pathlib import Path

def export_batch_reactor_oed_model() -> AcadosModel:
    model_name = 'batch_reactor_oed'
    
    p1val = 4000.0
    p2val = 2500.0
    p3val = 620000.0
    p4val = 5000.0
    
    p1scale = 400/4000
    p2scale = 400/2500
    p3scale = 400/620000
    p4scale = 400/5000
    
    x = ca.MX.sym('x', 2)
    x1,x2 = ca.vertsplit(x)
    
    theta = ca.MX.sym('theta', 4)
    p1_s, p2_s, p3_s, p4_s = ca.vertsplit(theta)
    
    p1_s, p2_s, p3_s, p4_s = p1_s/p1scale, p2_s/p2scale, p3_s/p3scale, p4_s/p4scale
    p1, p2, p3, p4 = p1val*p1scale, p2val*p2scale, p3val*p3scale, p4val*p4scale
    
    
    T = ca.MX.sym('T', 1)
    k1 = p1_s*ca.exp(-p2_s/T)
    k2 = p3_s*ca.exp(-p4_s/T)
    
    f_expr = ca.vertcat(-k1*x1**2, 
                         k1*x1**2 - k2*x2
                         )
    f_x_expr = ca.jacobian(f_expr, x)
    f_theta_expr = ca.jacobian(f_expr, theta)
    
    
    f = ca.Function('f', [x,T,theta], [f_expr])
    f_x = ca.Function('f', [x,T,theta], [f_x_expr])
    f_theta = ca.Function('f', [x,T,theta], [f_theta_expr])
    
    f_expr = f(x,T,ca.DM([p1, p2, p3, p4]))
    f_x_expr = f_x(x,T,ca.DM([p1, p2, p3, p4]))
    f_theta_expr = f_theta(x,T,ca.DM([p1, p2, p3, p4]))
    
    
    G = ca.MX.sym('G', x.numel(), theta.numel())
    dG = f_x_expr@G + f_theta_expr
    G_rhs = ca.vec(dG)
    
    w = ca.MX.sym('w', 2)
    w1,w2 = ca.vertsplit(w)
    
    F = ca.MX.sym('F', (theta.numel()*(theta.numel() + 1))//2)
    dh1, dh2 = ca.DM([1,0]), ca.DM([0,1])
    dF = w1*(dh1.T@G).T @ (dh1.T@G) + w2*(dh2.T@G).T @ (dh2.T@G)
    
    F_rhs = ca.vertcat(dF[0,0], dF[1,0], dF[2,0], dF[3,0], 
                                dF[1,1], dF[2,1], dF[3,1], 
                                         dF[2,2], dF[3,2], 
                                                  dF[3,3])
    
    
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
    model.u = ca.vertcat(T, w)
    model.name = model_name
    
    F_tf = ca.MX.zeros(4, 4)
    idx = 0
    for j in range(4):
        for i in range(j + 1):
            F_tf[i, j] = F[idx]
            F_tf[j, i] = F[idx]
            idx += 1
    
    F_reg = F_tf + 1e-3 * ca.diag(np.ones(4))
    model.cost_expr_ext_cost_e = 0.25 * ca.trace(ca.inv(F_reg))
    
    model.x_labels = ['x1', 'x2', 'G', 'F', 'z1', 'z2']
    model.u_labels = ['T', 'w1', 'w2']
    model.t_label = 't'

    return model


def setup_batch_reactor_oed_ocp():
    
    ocp = AcadosOcp()
    model = export_batch_reactor_oed_model()
    ocp.model = model
    
    nx = model.x.rows()
    nu = model.u.rows()
    
    M1 = 0.4
    M2 = 0.4
    
    ocp.solver_options.N_horizon = 80        #80 works reasonably well
    ocp.solver_options.tf = 1.0
    ocp.solver_options.qp_solver_cond_N = 10
    
    try:
        cD = Path(__file__).parent
    except:
        cD = Path.cwd()
    ocp.code_gen_options.code_export_directory = str(cD/Path(f"acados_codegen/{model.name}"))

    
    ocp.cost.cost_type_e = 'EXTERNAL'
    
    ocp.constraints.lbu = np.array([298.0 + 60.0, 0.0, 0.0])
    ocp.constraints.ubu = np.array([398.0, 1.0, 1.0])
    ocp.constraints.idxbu = np.array([0, 1, 2])
    
    x0 = np.array([1.0, 0.0] + [0.0]*8 + [0.0]*10 + [0.0]*2)
    ocp.constraints.lbx_0 = x0
    ocp.constraints.ubx_0 = x0
    ocp.constraints.idxbx_0 = np.arange(nx)
    
    ocp.constraints.lbx_e = np.array([0.0, 0.0])
    ocp.constraints.ubx_e = np.array([M1, M2])
    ocp.constraints.idxbx_e = np.array([nx-2, nx-1])
    
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    # ocp.solver_options.qp_solver_iter_max = 1000  
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    
    ocp_solver = AcadosOcpSolver(ocp)
    
    sim = AcadosSim()
    sim.model = model
    sim.code_gen_options.code_export_directory = str(cD/Path(f"acados_codegen/{model.name}_sim"))
    
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.T = ocp.solver_options.tf / ocp.solver_options.N_horizon
    sim.solver_options.num_steps = 2
    sim_sol = AcadosSimSolver(sim)
    
    u_init_val = np.array([398.0, M1/ocp.solver_options.tf, M2/ocp.solver_options.tf])
    u_init_traj = np.tile(u_init_val, (ocp.solver_options.N_horizon, 1))
    
    x_current = x0
    sim_x = np.zeros((ocp.solver_options.N_horizon + 1, nx))
    sim_x[0, :] = x_current
    
    for i in range(ocp.solver_options.N_horizon):
        x_next = sim_sol.simulate(x=x_current, u=u_init_traj[i])
        sim_x[i+1, :] = x_next
        x_current = x_next
    
    for i in range(ocp.solver_options.N_horizon):
        ocp_solver.set(i, "x", sim_x[i, :])
        ocp_solver.set(i, "u", u_init_traj[i])
    ocp_solver.set(ocp.solver_options.N_horizon, "x", sim_x[ocp.solver_options.N_horizon, :])
    
    return ocp_solver
    
def main():
    
    ocp_solver = setup_batch_reactor_oed_ocp()
    
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
        labels_list=['Batch Reactor OED'],
        x_labels=['x1', 'x2'],
        u_labels=ocp_solver.ocp.model.u_labels,
        idxbu=ocp_solver.ocp.constraints.idxbu,
        lbu=ocp_solver.ocp.constraints.lbu,
        ubu=ocp_solver.ocp.constraints.ubu,
        fig_filename='acados_plots/batch_reactor_oed_ocp.png',
    )
    
        
if __name__ == '__main__':
    	main()