from acados_template import AcadosModel, AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver, plot_trajectories, ACADOS_INFTY
import numpy as np
from casadi import SX, vertcat
import time
from pathlib import Path

def export_dielectrophoretic_model() -> AcadosModel:
    model_name = 'dielectrophoretic_particle'

    alpha = -0.75
    c_param = 1.0
    
    x0 = SX.sym('x0')
    x1 = SX.sym('x1')
    T = SX.sym('T')
    X = vertcat(x0, x1, T)
    
    u = SX.sym('u')

    x0_dot = SX.sym('x0_dot')
    x1_dot = SX.sym('x1_dot')
    T_dot = SX.sym('T_dot')
    Xdot = vertcat(x0_dot, x1_dot, T_dot)
    
    f_expl = T*vertcat(x1 * u + alpha * u**2, 
                      -c_param * x1 + u, 
                       0.)
    
    f_impl = Xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = X
    model.xdot = Xdot
    model.u = u
    model.name = model_name
    
    model.cost_expr_ext_cost_e = T

    model.x_labels = ['x0', 'x1', 't']
    model.u_labels = ['u']
    model.t_label = 't'

    return model

def setup_dielectrophoretic_ocp():
    ocp = AcadosOcp()
    model = export_dielectrophoretic_model()
    ocp.model = model
    
    x00 = 1.0
    xF = 2.0
    u_init_val = 1.0
    Tf_init = 5.0
    
    ocp.solver_options.N_horizon = 100
    ocp.solver_options.tf = 1.0
    ocp.solver_options.qp_solver_cond_N = 4
    
    try:
        cD = Path(__file__).parent
    except:
        cD = Path.cwd()
    ocp.code_gen_options.code_export_directory = str(cD/Path(f"acados_codegen/{model.name}"))

    
    N = ocp.solver_options.N_horizon
    nx = model.x.rows()
    nu = model.u.rows()
    
    ocp.cost.cost_type_e = 'EXTERNAL'
    
    ocp.constraints.lbu = np.array([0])
    ocp.constraints.ubu = np.array([1.0])
    ocp.constraints.idxbu = np.array([0])
    
    ocp.constraints.lbx_0 = np.array([x00, 0.0, 1.0])
    ocp.constraints.ubx_0 = np.array([x00, 0.0, ACADOS_INFTY])
    ocp.constraints.idxbx_0 = np.arange(3)
    
    ocp.constraints.lbx_e = np.array([xF])
    ocp.constraints.ubx_e = np.array([xF])
    ocp.constraints.idxbx_e = np.array([0])
    
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    
    ocp_solver = AcadosOcpSolver(ocp)
    
    sim = AcadosSim()
    sim.model = model
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.T = Tf_init / N      #This should be 1.0/N, but then nlp solver is somehow no longer successful
    sim.solver_options.num_steps = 2
    sim.code_gen_options.code_export_directory = str(cD/Path(f"acados_codegen/{model.name}_sim"))
    
    sim_sol = AcadosSimSolver(sim)
    
    x_current = np.array([x00, 0.0, Tf_init])
    u_init_traj = np.full((N, nu), u_init_val)
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
    
    ocp_solver = setup_dielectrophoretic_ocp()
    
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
        time_traj_list=[np.linspace(0, tf, N+1) * simX[0,-1]],
        time_label='Time [s]',
        labels_list=['Dielectrophoretic Particle'],
        x_labels=ocp_solver.ocp.model.x_labels,
        u_labels=ocp_solver.ocp.model.u_labels,
        idxbu=ocp_solver.ocp.constraints.idxbu,
        lbu=ocp_solver.ocp.constraints.lbu,
        ubu=ocp_solver.ocp.constraints.ubu,
        fig_filename='acados_plots/dielectrophoretic_ocp.png',
    )
if __name__ == '__main__':
	main()