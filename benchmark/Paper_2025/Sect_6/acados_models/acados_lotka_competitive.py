from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, plot_trajectories
# from lotka_model import export_lotka_volterra_model
import numpy as np
import casadi as ca
import time
from casadi import SX, vertcat
from pathlib import Path

def export_lotka_volterra_model() -> AcadosModel:
    model_name = 'lotka_volterra_competitive'

    c1 = 0.1 
    c2 = 0.4

    x1 = SX.sym('x1')
    x2 = SX.sym('x2')
    x = vertcat(x1, x2)

    u_sym = SX.sym('u')
    u = vertcat(u_sym)

    x1_dot = SX.sym('x1_dot')
    x2_dot = SX.sym('x2_dot')
    xdot = vertcat(x1_dot, x2_dot)

    f_expl = ca.vertcat(
            x1 * (1 - (x1 + 1.2 * x2)/1.8) - c1 * x1 * u,
            x2 * (1 - (x1 + x2)/1.8) - c2 * x2 * u)

    f_impl = xdot - f_expl
    
    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    model.x_labels = ['x1', 's2']
    model.u_labels = ['u']
    model.t_label = 't'

    return model


def setup_lotka_competitive_ocp(x_init = np.array([0.5, 1.5])):
    ocp = AcadosOcp()
    
    model = export_lotka_volterra_model()
    ocp.model = model
    
    Tf = 40.0
    nx = model.x.rows()
    nu = model.u.rows()
    N = 100
    
    
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf
    ocp.solver_options.qp_solver_cond_N = 4
    
    try:
        cD = Path(__file__).parent
    except:
        cD = Path.cwd()
    ocp.code_gen_options.code_export_directory = str(cD/Path(f"acados_codegen/{model.name}"))
    
    Q_mat = np.eye(2)
    R_mat = np.diag([1e-4])
    
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.yref = np.array([1.0, 1.0, 0.0])
    ocp.cost.W = ca.diagcat(Q_mat, R_mat).full()
    ocp.cost.Vx = np.array([[1.,0.],[0.,1.],[0.,0.]])
    ocp.cost.Vu = np.array([[0.],[0.],[1.0]])
    
    ocp.cost.cost_type_e = 'LINEAR_LS'
    ocp.cost.W_e = Q_mat*(Tf/N)
    ocp.cost.Vx_e = np.eye(2)
    ocp.cost.yref_e = np.array([1.,1.])
    
    
    ocp.constraints.lbu = np.array([0.0])
    ocp.constraints.ubu = np.array([1.0])
    ocp.constraints.idxbu = np.array([0])
    
    ocp.constraints.x0 = x_init
    
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    
    ocp.solver_options.nlp_solver_max_iter = 100
    ocp.solver_options.tol = 1e-6
    
    
    ocp_solver = AcadosOcpSolver(ocp)
    for i in range(ocp.solver_options.N_horizon + 1):
        ocp_solver.set(i, "x", x_init)
    
    return ocp_solver
    

def setup_lotka_competitive_ocp_2(x_init = np.array([1.5, 0.5])):
    return setup_lotka_competitive_ocp(x_init = x_init)        
    
def main():
    ocp_solver = setup_lotka_competitive_ocp()

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
        labels_list=['OCP result'],
        x_labels=ocp_solver.ocp.model.x_labels,
        u_labels=ocp_solver.ocp.model.u_labels,
        idxbu=ocp_solver.ocp.constraints.idxbu,
        lbu=ocp_solver.ocp.constraints.lbu,
        ubu=ocp_solver.ocp.constraints.ubu,
        fig_filename='acados_plots/lotka_competitive.png',
    )

if __name__ == '__main__':
	main()