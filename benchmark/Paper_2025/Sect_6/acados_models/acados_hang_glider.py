from acados_template import AcadosModel, AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver, plot_trajectories, ACADOS_INFTY
import numpy as np
import casadi as ca
from casadi import SX, vertcat
import time



def export_hang_glider_model() -> AcadosModel:
    model_name = 'hang_glider'
    
    
    c0 = 0.034
    c1 = 0.069662
    S_area = 14.0
    rho = 1.13
    uC = 2.5
    rC = 100.0
    m = 100.0
    g = 9.81


    # States: x, vx, y, vy, T (following CasADi order: x, dx, y, dy)
    x = SX.sym('x')
    vx = SX.sym('vx')
    y = SX.sym('y')
    vy = SX.sym('vy')
    T = SX.sym('T')
    X = vertcat(x, vx, y, vy, T)

    cL = SX.sym('cL')
    U = vertcat(cL)

    Xdot = SX.sym('Xdot', 5)
    
    r_val = (x / rC - 2.5)**2
    U_updraft = uC * (1 - r_val) * ca.exp(-r_val)
    
    w = vy - U_updraft
    v_rel = ca.sqrt(vx**2 + w**2)
    
    L = 0.5 * cL * rho * S_area * v_rel**2
    D = 0.5 * (c0 + c1 * cL**2) * rho * S_area * v_rel**2
    
    f_expl = T * vertcat(
        vx,
        (1/m) * (-L * (w / v_rel) - D * (vx / v_rel)),
        vy,
        (1/m) * (L * (vx / v_rel) - D * (w / v_rel)) - g,
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
    
    model.cost_expr_ext_cost_e = -x

    model.x_labels = ['x', 'vx', 'y', 'vy', 'T']
    model.u_labels = ['cL']
    model.t_label = 'Normalized Time'

    return model

def setup_hang_glider_ocp():
    ocp = AcadosOcp()
    model = export_hang_glider_model()
    ocp.model = model
    
    cmax = 1.4
    Tf_init = 100.0
    
    N = 100
    nx = model.x.rows()
    nu = model.u.rows()
    
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = 1.0
    ocp.solver_options.qp_solver_cond_N = 4
    
    ocp.cost.cost_type_e = 'EXTERNAL'
    
    
    ocp.constraints.lbu = np.array([0.0])
    ocp.constraints.ubu = np.array([cmax])
    ocp.constraints.idxbu = np.array([0])
    
    ocp.constraints.lbx_0 = np.array([0.0, 13.23, 1000.0, -1.288, 75])
    ocp.constraints.ubx_0 = np.array([0.0, 13.23, 1000.0, -1.288, 1500])
    ocp.constraints.idxbx_0 = np.arange(nx)
    
    ocp.constraints.lbx_e = np.array([13.23, 900.0, -1.288])
    ocp.constraints.ubx_e = np.array([13.23, 900.0, -1.288])
    ocp.constraints.idxbx_e = np.array([1, 2, 3])
    
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    
    ocp_solver = AcadosOcpSolver(ocp)
    
    
    sim = AcadosSim()
    sim.model = model
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.T = 1.0 / N
    sim.solver_options.num_steps = 2
    sim_sol = AcadosSimSolver(sim)
    
    u_init_traj = np.full((N, nu), cmax)
    x_current = np.array([0.0, 13.23, 1000.0, -1.288, Tf_init])
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
    ocp_solver = setup_hang_glider_ocp()
    
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
        time_label='Time [s]',
        labels_list=['Hang Glider'],
        x_labels=ocp_solver.ocp.model.x_labels,
        u_labels=ocp_solver.ocp.model.u_labels,
        idxbu=ocp_solver.ocp.constraints.idxbu,
        lbu=ocp_solver.ocp.constraints.lbu,
        ubu=ocp_solver.ocp.constraints.ubu,
        fig_filename='hang_glider_ocp.png',
    )
    
if __name__ == '__main__':
	main()