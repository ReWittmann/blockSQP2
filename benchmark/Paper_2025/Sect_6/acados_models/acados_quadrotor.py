from acados_template import AcadosModel, AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver, ACADOS_INFTY, plot_trajectories
import numpy as np
import casadi as ca
from pathlib import Path

def export_quadrotor_model() -> AcadosModel:
    model_name = 'quadrotor_helicopter'
    
    g = 9.8
    M = 1.3
    L = 0.305
    I = 0.0605
    
    x = ca.MX.sym('x', 7)
    x1, x2, x3, x4, x5, x6, q = ca.vertsplit(x)
    
    U = ca.MX.sym('U', 4)
    w1, w2, w3, u = ca.vertsplit(U)
    

    f_expl = ca.vertcat(
        x2,
        g*ca.sin(x5) + w1*u*ca.sin(x5)/M,
        x4,
        g*ca.cos(x5) - g + w1*u*ca.cos(x5)/M,
        x6,
        -w2*L*u/I + w3*L*u/I,
        5*u**2
    )
    
    Xdot = ca.MX.sym('Xdot', 7)
    f_impl = Xdot - f_expl
    
    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = Xdot
    model.u = U
    model.name = model_name
    
    model.cost_expr_ext_cost_e = 5*(x1 - 6)**2 + 5*(x3 - 1)**2 + (0.5*ca.sin(x5))**2 + q
    
    model.x_labels = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'q']
    model.u_labels = ['w1', 'w2', 'w3', 'u']
    model.t_label = 't'
    
    return model

def setup_quadrotor_ocp():
    ocp = AcadosOcp()
    model = export_quadrotor_model()
    ocp.model = model
    
    Tf = 7.5
    N = 100
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf
    # ocp.solver_options.qp_solver_cond_N = 4   #seem to work only for OFF or 1
    
    try:
        cD = Path(__file__).parent
    except:
        cD = Path.cwd()
    ocp.code_gen_options.code_export_directory = str(cD/Path(f"acados_codegen/{model.name}"))
    
    
    nx = model.x.rows()
    nu = model.u.rows()
    
    
    ocp.cost.cost_type_e = 'EXTERNAL'
    

    ocp.constraints.lbu = np.array([0.0, 0.0, 0.0, 0.0])
    ocp.constraints.ubu = np.array([1.0, 1.0, 1.0, 0.001])
    ocp.constraints.idxbu = np.arange(nu)
    
    ocp.constraints.x0 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    
    ocp.model.con_h_expr = ca.sum(ocp.model.u[:3])
    ocp.constraints.lh = np.array([1.0])
    ocp.constraints.uh = np.array([1.0])
    
    ocp.constraints.lbx = np.array([0.0])
    ocp.constraints.ubx = np.array([ACADOS_INFTY])
    ocp.constraints.idxbx = np.array([2])
    
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
    
    u_init = np.array([0.5, 0.0, 0.5, 0.001])
    u_init_traj = np.tile(u_init, (N, 1))
    
    x_current = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
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
    ocp_solver = setup_quadrotor_ocp()
    status = ocp_solver.solve()
    ocp_solver.print_statistics()
    
    N = ocp_solver.ocp.solver_options.N_horizon
    nx = ocp_solver.ocp.model.x.numel()
    nu = ocp_solver.ocp.model.u.numel()
    simX = np.zeros((N+1, nx))
    simU = np.zeros((N, nu))
    for i in range(N):
        simX[i,:] = ocp_solver.get(i, "x")
        simU[i,:] = ocp_solver.get(i, "u")
    simX[N,:] = ocp_solver.get(N, "x")

    plot_trajectories(
        x_traj_list=[simX[:, (0,2,4)]],
        u_traj_list=[simU],
        time_traj_list=[np.linspace(0, 7.5, N+1)],
        time_label='t',
        labels_list=['Quadrotor Helicopter'],
        x_labels=[ocp_solver.ocp.model.x_labels[i] for i in (0,2,4)],
        u_labels=ocp_solver.ocp.model.u_labels,
        idxbu=ocp_solver.ocp.constraints.idxbu,
        lbu=ocp_solver.ocp.constraints.lbu,
        ubu=ocp_solver.ocp.constraints.ubu,
        fig_filename='quadrotor_ocp.png',
    )
    
if __name__ == '__main__':
    main()