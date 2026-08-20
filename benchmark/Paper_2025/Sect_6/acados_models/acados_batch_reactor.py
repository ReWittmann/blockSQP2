from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, plot_trajectories
import numpy as np
import casadi as ca
from casadi import SX, vertcat, exp
import time
from pathlib import Path

def export_reactor_model() -> AcadosModel:
    model_name = 'batch_reactor'

    x1 = SX.sym('x1')
    x2 = SX.sym('x2')
    x = vertcat(x1, x2)

    u_sym = SX.sym('u')
    u = vertcat(u_sym)

    x1_dot = SX.sym('x1_dot')
    x2_dot = SX.sym('x2_dot')
    xdot = vertcat(x1_dot, x2_dot)

    k1 = 4000 * exp(-2500 / u_sym)
    k2 = 620000 * exp(-5000 / u_sym)

    f_expl = vertcat(
        -k1 * x1**2,
        k1 * x1**2 - k2 * x2
    )

    f_impl = xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name


    model.cost_y_expr_e = x2
    
    model.cost_expr_ext_cost_e = -x2

    model.x_labels = ['x1', 'x2']
    model.u_labels = ['Temperature']
    model.t_label = 't'

    return model

def setup_batch_reactor_ocp():
    ocp = AcadosOcp()
    model = export_reactor_model()
    ocp.model = model
    
    Tf = 1.0
    N = 100
    
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf
    
    try:
        cD = Path(__file__).parent
    except:
        cD = Path.cwd()
    ocp.code_gen_options.code_export_directory = str(cD/Path(f"acados_codegen/{model.name}"))

    
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.model.cost_y_expr = ca.vertcat(0)
    ocp.cost.yref = np.array([0.0])
    ocp.cost.W = np.array([[0.0]])
    
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.cost.yref_e = np.array([1.0]) 
    ocp.cost.W_e = np.array([[1.0]])
    
    # ocp.cost.cost_type_e = 'EXTERNAL'
    
    ocp.constraints.lbu = np.array([298.0])
    ocp.constraints.ubu = np.array([398.0])
    ocp.constraints.idxbu = np.array([0])
    
    ocp.constraints.x0 = np.array([1.0, 0.0])
    
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    # ocp.solver_options.qp_solver_iter_max = 1000
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    
    
    ocp_solver = AcadosOcpSolver(ocp)
    for i in range(ocp.solver_options.N_horizon):
        ocp_solver.set(i, "u", 298)
    return ocp_solver
    
def main():
    ocp_solver = setup_batch_reactor_ocp()
    status = ocp_solver.solve()
    ocp_solver.print_statistics()
    
    if status != 0:
        print(f"Warning: Solver returned status {status}")
    
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
        labels_list=['Batch Reactor'],
        x_labels=ocp_solver.ocp.model.x_labels,
        u_labels=ocp_solver.ocp.model.u_labels,
        idxbu=ocp_solver.ocp.constraints.idxbu,
        lbu=ocp_solver.ocp.constraints.lbu,
        ubu=ocp_solver.ocp.constraints.ubu,
        fig_filename='acados_plots/reactor_ocp.png',
    )

    
if __name__ == '__main__':
    	main()