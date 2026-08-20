from acados_template import AcadosModel, AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver, plot_trajectories, ACADOS_INFTY
import numpy as np
import casadi as ca
from casadi import SX, vertcat
import time
from pathlib import Path



def export_chemotherapy_model() -> AcadosModel:
    model_name = 'D_Onofrio_Chemotherapy'

    zeta = 0.192
    b = 5.85
    mu = 0.0
    d = 0.00873
    G = 0.15
    F = 1.0
    eta = 1.0
    
    
    x0 = SX.sym('x0')
    x1 = SX.sym('x1')
    x2 = SX.sym('x2')
    x3 = SX.sym('x3')
    X = vertcat(x0, x1, x2, x3)

    u0 = SX.sym('u0')
    u1 = SX.sym('u1')
    U = vertcat(u0, u1)

    x0_dot = SX.sym('x0_dot')
    x1_dot = SX.sym('x1_dot')
    x2_dot = SX.sym('x2_dot')
    x3_dot = SX.sym('x3_dot')
    Xdot = vertcat(x0_dot, x1_dot, x2_dot, x3_dot)
    
    f_expl = vertcat(
        -zeta * x0 * ca.log(x0 / x1) - F * x0 * u1,
        b * x0 - mu * x1 - d * (x0**(2/3)) * x1 - G * u0 * x1 - eta * x1 * u1,
        u0,
        u1
    )
    f_impl = Xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = X
    model.xdot = Xdot
    model.u = U
    model.name = model_name
    
    model.cost_expr_ext_cost_e = x0

    model.x_labels = ['x0', 'x1', 'x2', 'x3']
    model.u_labels = ['u0', 'u1']
    model.t_label = 't'

    return model


def setup_D_Onofrio_ocp(x00 = 12000.0, x10 = 15000.0, u1_max = 1.0, x3_max = 2.0):
    ocp = AcadosOcp()
    model = export_chemotherapy_model()
    ocp.model = model
    
    alpha = 0.0
    u0_max = 75.0
    x2_max = 300.0
    
    ocp.solver_options.N_horizon = 100
    ocp.solver_options.tf = 6.0
    ocp.solver_options.qp_solver_cond_N = 4
    
    try:
        cD = Path(__file__).parent
    except:
        cD = Path.cwd()
    ocp.code_gen_options.code_export_directory = str(cD/Path(f"acados_codegen/{model.name}"))

    
    N = ocp.solver_options.N_horizon
    nx = model.x.numel()
    nu = model.u.numel()
    
    ocp.cost.cost_type_e = 'EXTERNAL'
    
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.yref = np.array([0.])
    
    ocp.cost.Vx = np.array([[0.0, 0.0, 0.0, 0.0]])
    ocp.cost.Vu = np.array([[1.0,0.]])
    ocp.cost.W = np.array([[alpha]])
    
    
    ocp.constraints.lbu = np.array([0.0, 0.0])
    ocp.constraints.ubu = np.array([u0_max, u1_max])
    ocp.constraints.idxbu = np.array([0, 1])
    
    
    ocp.constraints.lbx_0 = np.array([x00, x10, 0.0, 0.0])
    ocp.constraints.ubx_0 = np.array([x00, x10, 0.0, 0.0])
    ocp.constraints.idxbx_0 = np.arange(4)
    

    ocp.constraints.lbx_e = np.array([-ACADOS_INFTY, -ACADOS_INFTY]) # No lower bound
    ocp.constraints.ubx_e = np.array([x2_max, x3_max])
    ocp.constraints.idxbx_e = np.array([2, 3])
    

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    ocp.solver_options.nlp_solver_max_iter = 200


    ocp_solver = AcadosOcpSolver(ocp)
    
    u0_init_val = x2_max / ocp.solver_options.tf
    u1_init_val = x3_max / ocp.solver_options.tf
    
    sim = AcadosSim()
    sim.model = model
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.T = ocp.solver_options.tf / N
    sim.solver_options.num_steps = 2
    sim_sol = AcadosSimSolver(sim)
    
    x_current = np.array([x00, x10, 0.0, 0.0])
    u_init_traj = np.zeros((N, nu))
    for i in range(N):
        u_init_traj[i, 0] = u0_init_val
        u_init_traj[i, 1] = u1_init_val
    
    sim_x = np.zeros((N + 1, nx))
    sim_x[0, :] = x_current
    
    for i in range(N):
        x_next = sim_sol.simulate(x=x_current, u=u_init_traj[i])
        sim_x[i+1, :] = x_next
        x_current = x_next
    
    # Set initial guess
    for i in range(N):
        ocp_solver.set(i, "x", sim_x[i, :])
        ocp_solver.set(i, "u", u_init_traj[i])
    ocp_solver.set(N, "x", sim_x[N, :])
    
    return ocp_solver

def setup_D_Onofrio_ocp_2(x00 = 12000.0, x10 = 15000.0, u1_max = 2.0, x3_max = 10.0):
    return setup_D_Onofrio_ocp(x00 = x00, x10 = x10, u1_max = u1_max, x3_max = x3_max)
def setup_D_Onofrio_ocp_3(x00 = 14000.0, x10 = 5000.0, u1_max = 1.0, x3_max = 2.0):
    return setup_D_Onofrio_ocp(x00 = x00, x10 = x10, u1_max = u1_max, x3_max = x3_max)
def setup_D_Onofrio_ocp_4(x00 = 14000.0, x10 = 5000.0, u1_max = 2.0, x3_max = 10.0):
    return setup_D_Onofrio_ocp(x00 = x00, x10 = x10, u1_max = u1_max, x3_max = x3_max)

def main():
    ocp_solver = setup_D_Onofrio_ocp()
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
        x_traj_list=[simX[:,0:2]],
        u_traj_list=[simU],
        time_traj_list=[np.linspace(0, tf, N+1)],
        time_label=ocp_solver.ocp.model.t_label,
        labels_list=['Chemotherapy Model'],
        x_labels=ocp_solver.ocp.model.x_labels,
        u_labels=ocp_solver.ocp.model.u_labels,
        idxbu=ocp_solver.ocp.constraints.idxbu,
        lbu=ocp_solver.ocp.constraints.lbu,
        ubu=ocp_solver.ocp.constraints.ubu,
        fig_filename='acados_plots/chemotherapy_ocp.png',
    )
    
if __name__ == '__main__':
	main()