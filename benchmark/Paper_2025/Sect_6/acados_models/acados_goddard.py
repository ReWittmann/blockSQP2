from acados_template import AcadosModel, AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver, plot_trajectories, ACADOS_INFTY
import numpy as np
import casadi as ca
from casadi import SX, vertcat
import time
from pathlib import Path

def export_goddard_model() -> AcadosModel:
    model_name = 'goddard_rocket'

    r0 = 1.0
    b = 7.0
    Tmax = 3.5
    A = 310.0
    k = 500.0
    
    r = SX.sym('r')
    v = SX.sym('v')
    m = SX.sym('m')
    T = SX.sym('T')
    X = vertcat(r, v, m, T)

    u = SX.sym('u')


    r_dot = SX.sym('r_dot')
    v_dot = SX.sym('v_dot')
    m_dot = SX.sym('m_dot')
    T_dot = SX.sym('T_dot')
    Xdot = vertcat(r_dot, v_dot, m_dot, T_dot)
    

    rho = ca.exp(-k * (r - r0))
    drag = A * v**2 * rho
    
    f_expl = T * vertcat(
        v,
        -1.0/r**2 + (1.0/m) * (Tmax * u - drag),
        -b * u,
        0.0
    )
    
    f_impl = Xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = X
    model.xdot = Xdot
    model.u = u
    model.name = model_name
    
    model.cost_expr_ext_cost_e = -m
    
    
    model.x_labels = ['r', 'v', 'm', 't']
    model.u_labels = ['u']
    model.t_label = 't'


    return model

def setup_goddard_ocp():
    ocp = AcadosOcp()
    model = export_goddard_model()
    ocp.model = model
    
    r0 = 1.0
    v0 = 0.0
    m0 = 1.0
    rT = 1.01
    b = 7.0
    A = 310.0
    k = 500.0

    
    Tf_scaled = 1.0 
    N = 100
    nx = model.x.rows()
    nu = model.u.rows()
    
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf_scaled
    # ocp.solver_options.qp_solver_cond_N = 1
    
    try:
        cD = Path(__file__).parent
    except:
        cD = Path.cwd()
    ocp.code_gen_options.code_export_directory = str(cD/Path(f"acados_codegen/{model.name}"))

    
    ocp.cost.cost_type_e = 'EXTERNAL'
    
    
    ocp.constraints.lbu = np.array([0.0])
    ocp.constraints.ubu = np.array([1.0])
    ocp.constraints.idxbu = np.array([0])
    
    rho_expr = ca.exp(-k * (model.x[0] - r0))
    drag_expr = A * model.x[1]**2 * rho_expr
    ocp.model.con_h_expr = drag_expr
    ocp.constraints.lh = np.array([-ACADOS_INFTY])
    ocp.constraints.uh = np.array([0.6])
    
    
    Tf_init = (0.4 / b) * 2.5
    ocp.constraints.lbx_0 = np.array([r0, v0, m0, 1e-3])
    ocp.constraints.ubx_0 = np.array([r0, v0, m0, ACADOS_INFTY])
    ocp.constraints.idxbx_0 = np.arange(nx)
    
    
    ocp.constraints.lbx_e = np.array([rT])
    ocp.constraints.ubx_e = np.array([rT])
    ocp.constraints.idxbx_e = np.array([0])
    
    
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
    sim.solver_options.T = Tf_scaled / N
    sim.solver_options.num_steps = 2
    sim_sol = AcadosSimSolver(sim)
    
    u_init_traj = np.zeros(N)
    for i in range(N):
        if i / N <= 0.4:
            u_init_traj[i] = 1.0
        else:
            u_init_traj[i] = 0.0
    
    x_current = np.array([r0, v0, m0, Tf_init])
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
    ocp_solver = setup_goddard_ocp()
    
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
        x_traj_list=[simX[:,:-1]],
        u_traj_list=[simU],
        time_traj_list=[np.linspace(0, 1.0, N+1) * simX[0,-1]],
        time_label=ocp_solver.ocp.model.t_label,
        labels_list=['Goddard Rocket'],
        x_labels=ocp_solver.ocp.model.x_labels,
        u_labels=ocp_solver.ocp.model.u_labels,
        idxbu=ocp_solver.ocp.constraints.idxbu,
        lbu=ocp_solver.ocp.constraints.lbu,
        ubu=ocp_solver.ocp.constraints.ubu,
        fig_filename='acados_plots/goddard_rocket_ocp.png',
    )
    
if __name__ == '__main__':
	main()