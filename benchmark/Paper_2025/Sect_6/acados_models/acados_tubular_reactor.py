from acados_template import AcadosModel, AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver, plot_trajectories
import numpy as np
import casadi as ca
from casadi import SX, vertcat
from pathlib import Path

def export_tubular_reactor_model() -> AcadosModel:
    model_name = 'tubular_reactor'

    x1 = SX.sym('x1')
    q = SX.sym('q')
    x = vertcat(x1, q)

    w = SX.sym('w')
    U = vertcat(w)
    
    x1dot = SX.sym('x1dot', 1)
    qdot = SX.sym('qdot', 1)
    
    xdot = ca.vertcat(x1dot, qdot)

    f_expl = ca.vertcat(-(w + 0.5 * w**2) * x1, w*x1)
    f_impl = xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = U
    model.name = model_name
    
    model.cost_expr_ext_cost_e = -q

    model.x_labels = ['x', 'q']
    model.u_labels = ['w']
    model.t_label = 't'

    return model

def setup_tubular_reactor_ocp():
    ocp = AcadosOcp()
    model = export_tubular_reactor_model()
    ocp.model = model
    
    N = 100
    nx = model.x.rows()
    nu = model.u.rows()
    
    Tf = 1.0
    
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf
    ocp.qp_solver_cond_N = 4
    
    try:
        cD = Path(__file__).parent
    except:
        cD = Path.cwd()
    ocp.code_gen_options.code_export_directory = str(cD/Path(f"acados_codegen/{model.name}"))

    
    ocp.cost.cost_type_e = 'EXTERNAL'
    
    ocp.constraints.lbu = np.array([0.0])
    ocp.constraints.ubu = np.array([5.0])
    ocp.constraints.idxbu = np.array([0])
    ocp.constraints.x0 = np.array([1.0, 0.0])
    
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    
    ocp_solver = AcadosOcpSolver(ocp)
    
    for i in range(N):
        ocp_solver.set(i, "x", np.array([1.0, 0.0]))
        ocp_solver.set(i, "u", 5.0)
    ocp_solver.set(N, "x", np.array([1.0,0.0]))
    
    return ocp_solver

def main():
    ocp_solver = setup_tubular_reactor_ocp()
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
        time_traj_list=[np.linspace(0, 1.0, N+1)],
        time_label=ocp_solver.ocp.model.t_label,
        labels_list=['Tubular Reactor'],
        x_labels=ocp_solver.ocp.model.x_labels,
        u_labels=ocp_solver.ocp.model.u_labels,
        idxbu=ocp_solver.ocp.constraints.idxbu,
        lbu=ocp_solver.ocp.constraints.lbu,
        ubu=ocp_solver.ocp.constraints.ubu,
        fig_filename='acados_plots/tubular_reactor_ocp.png',
    )
    
if __name__ == '__main__':
	main()