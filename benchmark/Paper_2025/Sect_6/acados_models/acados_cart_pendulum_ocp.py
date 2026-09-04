from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, plot_trajectories, ACADOS_INFTY
import numpy as np
import casadi as ca
from casadi import SX, vertcat, cos, sin
import time
from pathlib import Path

def export_cart_pendulum_model(lambda_u = 0.5) -> AcadosModel:
    model_name = 'cart_pendulum'

    x_pos = SX.sym('x_pos')
    theta = SX.sym('theta')
    v = SX.sym('v')
    w = SX.sym('w')
    q = SX.sym('q')
    x = vertcat(x_pos, theta, v, w, q)

    u_sym = SX.sym('u')
    u = vertcat(u_sym)

    x_dot = SX.sym('x_dot')
    theta_dot = SX.sym('theta_dot')
    v_dot = SX.sym('v_dot')
    w_dot = SX.sym('w_dot')
    q_dot = SX.sym('q_dot')
    
    xdot = vertcat(x_dot, theta_dot, v_dot, w_dot, q_dot)

    M = 1.0
    m = 0.1
    l = 1.0
    g = 9.81

    f0 = v
    f1 = w
    v_dot_expr = (u_sym + m * g * cos(theta) * sin(theta) + m * l * w**2 * sin(theta)) / (M + m * (1 - cos(theta)**2))
    f2 = v_dot_expr
    f3 = (-g * sin(theta) - v_dot_expr * cos(theta)) / l
    f4 = 10*x_pos**2 + 50*(theta - ca.pi)**2 + lambda_u*u**2

    f_expl = vertcat(f0, f1, f2, f3, f4)
    f_impl = xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    model.cost_expr_ext_cost_e = q

    model.x_labels = [r'x', r'theta', r'v', r'w', r'q']
    model.u_labels = [r'$u$']
    model.t_label = r't'

    return model


def setup_cart_pendulum_ocp(u_max = 30, lambda_u = 0.5):
    
    ocp = AcadosOcp()
    model = export_cart_pendulum_model(lambda_u)
    ocp.model = model


    ocp.solver_options.N_horizon = 100
    ocp.solver_options.tf = 4.0
    ocp.solver_options.qp_solver_cond_N = 20
    
    try:
        cD = Path(__file__).parent
    except:
        cD = Path.cwd()
    ocp.code_gen_options.code_export_directory = str(cD/Path(f"acados_codegen/{model.name}"))

    
    ocp.cost.cost_type_e = 'EXTERNAL'
    
    ocp.constraints.lbu = np.array([-u_max])
    ocp.constraints.ubu = np.array([u_max])
    ocp.constraints.idxbu = np.array([0])
    
    
    ocp.constraints.lbx = np.array([-2.0])
    ocp.constraints.ubx = np.array([2.0])
    ocp.constraints.idxbx = np.array([0])
    
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) 
    ocp.constraints.x0 = x0
    
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    # ocp.solver_options.qp_solver_iter_max = 50
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    
    ocp_solver = AcadosOcpSolver(ocp)
    
    for i in range(ocp.solver_options.N_horizon):
        ocp_solver.set(i, "u", 0.0)
    for i in range(ocp.solver_options.N_horizon+1):
        ocp_solver.set(i, "x", np.array([0.,0.,0.,0.,0.]))
    
    return ocp_solver

def setup_cart_pendulum_ocp_2(u_max = 15, lambda_u = 0.05):
    return setup_cart_pendulum_ocp(u_max = u_max, lambda_u = lambda_u)

def main():
    ocp_solver = setup_cart_pendulum_ocp()

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
        x_traj_list=[simX],
        u_traj_list=[simU],
        time_traj_list=[np.linspace(0, tf, N+1)],
        time_label=ocp_solver.ocp.model.t_label,
        labels_list=['Cart Pendulum'],
        x_labels=ocp_solver.ocp.model.x_labels,
        u_labels=ocp_solver.ocp.model.u_labels,
        idxbu=ocp_solver.ocp.constraints.idxbu,
        lbu=ocp_solver.ocp.constraints.lbu,
        ubu=ocp_solver.ocp.constraints.ubu,
        fig_filename='acados_plots/cart_pendulum_ocp.png',
    )
    
if __name__ == '__main__':
	main()