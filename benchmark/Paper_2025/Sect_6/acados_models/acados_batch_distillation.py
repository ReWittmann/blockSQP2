from acados_template import AcadosModel, AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver, plot_trajectories, ACADOS_INFTY
import numpy as np
import casadi as ca
from casadi import SX, vertcat, exp
import time
from pathlib import Path

def export_batch_distillation_model() -> AcadosModel:
    model_name = 'batch_distillation'

    alpha = 0.2
    V = 100
    m = 0.1
    mC = 0.1

    X = ca.SX.sym('X',10)
    M0,x0,x1,x2,x3,x4,x5,xC,xD,MD = ca.vertsplit(X)   
    T = ca.SX.sym('T')    
    xdot = ca.SX.sym('xdot', 11)

    R = ca.SX.sym('R')

    L = R/(1+R) * V        
    y = lambda x: x*(1+alpha)/(x+alpha)
    
    f_expl = T*ca.vertcat(-V + L,
                          1/M0 * (L*x1 - V*y(x0) + (V - L)*x0),
                          1/m * (L*x2 - V*y(x1) + V*y(x0) - L*x1),
                          1/m * (L*x3 - V*y(x2) + V*y(x1) - L*x2),
                          1/m * (L*x4 - V*y(x3) + V*y(x2) - L*x3),
                          1/m * (L*x5 - V*y(x4) + V*y(x3) - L*x4),
                          1/m * (L*xD - V*y(x5) + V*y(x4) - L*x5),
                          V/mC * (y(x5) - xC),
                          (V - L)/MD * (xC - xD),
                          V - L,
                          0)

    f_impl = xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = ca.vertcat(X, T)
    model.xdot = xdot
    model.u = R
    model.name = model_name
    
    model.cost_expr_ext_cost_e = T - MD

    model.x_labels = ['M0','x0','x1','x2','x3','x4','x5','xC','xD','MD','T']
    model.u_labels = ['Reflux ratio']
    model.t_label = 'Time [s]'
    
    return model

def setup_batch_distillation_ocp():
    M0init = 100.
    MDinit = 0.1
    x0init = 0.5
    xinit = 1.0
    xCinit = 1.0
    xDinit = 1.0
    
    ocp = AcadosOcp()
    model = export_batch_distillation_model()
    ocp.model = model
    
    Tf = 1.0
    N = 100
    nx = model.x.rows()
    nu = model.u.rows()
    
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf
    
    try:
        cD = Path(__file__).parent
    except:
        cD = Path.cwd()
    ocp.code_gen_options.code_export_directory = str(cD/Path(f"acados_codegen/{model.name}"))

    
    ocp.cost.cost_type_e = 'EXTERNAL'
    
    ocp.constraints.lbu = np.array([0.])
    ocp.constraints.ubu = np.array([ 15.])
    ocp.constraints.idxbu = np.array([0.])
    
    X_init = [M0init,x0init] + [xinit]*5+[xCinit,xDinit,MDinit]
    
    ocp.constraints.lbx_0 = np.array(X_init + [0.5])
    ocp.constraints.ubx_0 = np.array(X_init + [10.0])
    ocp.constraints.idxbx_0 = np.arange(nx)
    
    ocp.constraints.lbx = np.array([0.]*9 + [MDinit] + [0.5])
    ocp.constraints.ubx = np.array([0.]*9 + [ACADOS_INFTY] + [10.0])
    ocp.constraints.idxbx = np.arange(nx)
    
    ocp.constraints.lbx_e = np.array([0.]*8 + [0.99] + [MDinit] + [0.5])
    ocp.constraints.ubx_e = np.array([ACADOS_INFTY]*8 + [ACADOS_INFTY] + [ACADOS_INFTY] + [10.0])
    ocp.constraints.idxbx_e = np.arange(nx)
    
    
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    ocp.solver_options.qp_solver_iter_max = 10000
    ocp.solver_options.sim_method_num_steps = 10
    
    ocp_solver = AcadosOcpSolver(ocp)
    
    # Initial guess
    for i in range(N//2):
        ocp_solver.set(i, "u", 1.)
        
    for i in range(N//2, N):
        ocp_solver.set(i, "u", 15.)
    
    ocp_solver.set(0, "x", np.array(X_init + [1.0]))
    
    sim = AcadosSim()
    sim.model = model
    
    sim.solver_options.integrator_type = 'IRK'
    sim.solver_options.T = Tf/N
    sim.solver_options.num_steps = 10
    
    x_current =  np.array(X_init + [1.0])
    u_init = np.zeros((N, nu))
    for i in range(N//2):
        u_init[i, 0] = 1.0
    for i in range(N//2, N):
        u_init[i, 0] = 15.0
    
    
    sim_x = np.zeros((N + 1, nx))
    sim_x[0, :] = x_current
    for i in range(1,N+1):
        sim_x[i,:] = x_current
        # sim_x[i,-1] = sim_x[i-1,-1]
    
    sim_sol = AcadosSimSolver(sim)
    
    for i in range(N):
        x_next = sim_sol.simulate(x = x_current, u = u_init[i])
        sim_x[i+1, :] = x_next
        x_current = x_next
    
    for i in range(N):
        ocp_solver.set(i, "x", sim_x[i, :])
        ocp_solver.set(i, "u", u_init[i])
    ocp_solver.set(N, "x", sim_x[N, :])
    return ocp_solver


def main():
    ocp_solver = setup_batch_distillation_ocp()
    t0 = time.time()
    status = ocp_solver.solve()
    t1 = time.time()
    
    ocp_solver.print_statistics()
    
    if status != 0:
        print(f"Warning: Solver returned status {status}")
    
    N = ocp_solver.ocp.solver_options.N_horizon
    tf = ocp_solver.ocp.solver_options.tf
    nx = ocp_solver.ocp.model.x.numel()
    nu = ocp_solver.ocp.model.u.numel()
    
    # Extract solution
    simX = np.zeros((N+1, nx))
    simU = np.zeros((N, nu))
    for i in range(N):
        simX[i,:] = ocp_solver.get(i, "x")
        simU[i,:] = ocp_solver.get(i, "u")
    simX[N,:] = ocp_solver.get(N, "x")
    
    plot_trajectories(
        x_traj_list=[simX],
        u_traj_list=[simU],
        time_traj_list=[np.linspace(0, tf, N+1)*simX[0,-1]],
        time_label=ocp_solver.ocp.model.t_label,
        labels_list=['Batch_Distillation'],
        x_labels=ocp_solver.ocp.model.x_labels,
        u_labels=ocp_solver.ocp.model.u_labels,
        idxbu=ocp_solver.ocp.constraints.idxbu,
        lbu=ocp_solver.ocp.constraints.lbu,
        ubu=ocp_solver.ocp.constraints.ubu,
        fig_filename='acados_plots/batch_distillation_ocp.png',
    )
    
if __name__ == '__main__':
    main()
