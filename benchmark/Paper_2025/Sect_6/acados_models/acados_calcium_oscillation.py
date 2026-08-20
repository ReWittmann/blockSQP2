from acados_template import AcadosModel, AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver, ACADOS_INFTY, plot_trajectories
import numpy as np
import casadi as ca
from casadi import SX, vertcat
import time
from pathlib import Path

def export_calcium_model() -> AcadosModel:
    model_name = 'calcium_oscillation'
    
    p1 = 100.0
    k1, k2, k3, K4, k5, K6 = 0.09, 2.30066, 0.64, 0.19, 4.88, 1.18
    k7, k8, K9 = 2.08, 32.24, 29.09
    k10, K11, k12, k13, k14, K15, k16, K17 = 5.0, 2.67, 0.7, 13.58, 153.0, 0.16, 4.85, 0.05
    x_bar = np.array([6.78677, 22.65836, 0.384306, 0.28977])
    w_max = 1.3

    x0 = SX.sym('x0')
    x1 = SX.sym('x1')
    x2 = SX.sym('x2')
    x3 = SX.sym('x3')
    q = SX.sym('q')
    X = vertcat(x0, x1, x2, x3, q)

    w = SX.sym('w')
    U = vertcat(w)

    Xdot = SX.sym('Xdot', 5)
    
    dx0 = k1 + k2*x0 - (k3*x0*x1)/(x0 + K4) - (k5*x0*x2)/(x0 + K6)
    dx1 = k7*x0 - (k8*x1)/(x1 + K9)
        
    dx2 = (k10*x1*x2*x3)/(x3 + K11) + k12*x1 + k13*x0 - (k14*x2)/((1.0 + w * (w_max - 1.0))*x2 + K15) - (k16*x2)/(x2 + K17) + x3/10.0
    dx3 = -(k10*x1*x2*x3)/(x3 + K11) + (k16*x2)/(x2 + K17) - x3/10.0
    dq = ca.sumsqr(X[:-1] - x_bar) + p1*w # Constant parameter treated as state
    
    f_expl = vertcat(dx0, dx1, dx2, dx3, dq)
    f_impl = Xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = X
    model.xdot = Xdot
    model.u = U
    model.name = model_name
    
    model.cost_expr_ext_cost_e = q

    model.x_labels = ['x0', 'x1', 'x2', 'x3', 'w_max']
    model.u_labels = ['w']
    model.t_label = 't'

    return model

def setup_calcium_oscillation_ocp():
    ocp = AcadosOcp()
    model = export_calcium_model()
    ocp.model = model
    
    nx = model.x.rows()

    ocp.solver_options.N_horizon = 100
    ocp.solver_options.tf = 22.0
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
    
    ocp.constraints.lbx = np.zeros(nx)
    ocp.constraints.ubx = np.full(nx, ACADOS_INFTY)
    ocp.constraints.idxbx = np.arange(nx)
    
    x0_vec = np.array([0.03966, 1.09799, 0.00142, 1.65431, 0.])
    ocp.constraints.x0 = x0_vec
    
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.qp_solver_iter_max = 1000
    
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.integrator_type = 'IRK' 
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    ocp.solver_options.sim_method_num_steps = 100
    
    ocp_solver = AcadosOcpSolver(ocp)
    
    N = ocp.solver_options.N_horizon
    sim = AcadosSim()
    sim.model = model
    sim.solver_options.integrator_type = 'IRK' # Must match OCP
    sim.solver_options.T = ocp.solver_options.tf / N
    sim.solver_options.num_steps = 100
    sim_sol = AcadosSimSolver(sim)
    
    u_init_val = np.array([1.0])
    u_init_traj = np.tile(u_init_val, (N, 1))
    
    x_current = x0_vec
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
    ocp_solver = setup_calcium_oscillation_ocp()
    
    N = ocp_solver.ocp.solver_options.N_horizon
    tf = ocp_solver.ocp.solver_options.tf
    nx = ocp_solver.ocp.model.x.numel()
    nu = ocp_solver.ocp.model.u.numel()
    # Solve
    status = ocp_solver.solve()
    ocp_solver.print_statistics()
    
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
        labels_list=['Calcium Oscillation'],
        x_labels=ocp_solver.ocp.model.x_labels,
        u_labels=ocp_solver.ocp.model.u_labels,
        idxbu=ocp_solver.ocp.constraints.idxbu,
        lbu=ocp_solver.ocp.constraints.lbu,
        ubu=ocp_solver.ocp.constraints.ubu,
        fig_filename='acados_plots/calcium_ocp.png',
    )
    
if __name__ == '__main__':
    	main()