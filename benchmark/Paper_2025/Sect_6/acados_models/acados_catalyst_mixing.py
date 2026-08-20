from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, plot_trajectories
import numpy as np
import casadi as ca
from casadi import SX, vertcat, exp
import time
from pathlib import Path

def export_catalyst_mixing_model() -> AcadosModel:
    model_name = 'catalyst_mixing'

    x1 = SX.sym('x1')
    x2 = SX.sym('x2')
    x = vertcat(x1, x2)

    u = SX.sym('u')

    x1_dot = SX.sym('x1_dot')
    x2_dot = SX.sym('x2_dot')
    xdot = vertcat(x1_dot, x2_dot)
    
    alpha = 10
    f_expl = ca.vertcat(u*(alpha*x2-x1), u*(x1 - alpha*x2) - (1-u)*x2)
    f_impl = xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name
    
    model.cost_expr_ext_cost_e = -1 + x1 + x2

    model.x_labels = ['Substance A', 'Substance B']
    model.u_labels = ['Control']
    model.t_label = 'Time [s]'

    return model


def setup_catalyst_mixing_ocp():
    ocp = AcadosOcp()
    model = export_catalyst_mixing_model()
    ocp.model = model
    

    ocp.solver_options.N_horizon = 40
    ocp.solver_options.tf = 1.0
    ocp.solver_options.qp_solver_cond_N = 10
    
    try:
        cD = Path(__file__).parent
    except:
        cD = Path.cwd()
    ocp.code_gen_options.code_export_directory = str(cD/Path(f"acados_codegen/{model.name}"))

    
    ocp.cost.cost_type_e = 'EXTERNAL'
    
    ocp.constraints.lbu = np.array([0.0])
    ocp.constraints.ubu = np.array([1.0])
    ocp.constraints.idxbu = np.array([0])
    
    ocp.constraints.x0 = np.array([1.0, 0.0])
    
    
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    # ocp.solver_options.qp_solver_iter_max = 50
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    
    ocp_solver = AcadosOcpSolver(ocp)
    
    
    for i in range(ocp.solver_options.N_horizon):
        ocp_solver.set(i, "u", 0.)
    
    return ocp_solver

def main():
    ocp_solver = setup_catalyst_mixing_ocp()

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
        labels_list=['Catalyst Mixing'],
        x_labels=ocp_solver.ocp.model.x_labels,
        u_labels=ocp_solver.ocp.model.u_labels,
        idxbu=ocp_solver.ocp.constraints.idxbu,
        lbu=ocp_solver.ocp.constraints.lbu,
        ubu=ocp_solver.ocp.constraints.ubu,
        fig_filename='acados_plots/catalyst_mixing_ocp.png',
    )

if __name__ == '__main__':
	main()