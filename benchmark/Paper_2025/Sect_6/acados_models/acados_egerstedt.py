from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, plot_trajectories, ACADOS_INFTY
import numpy as np
import casadi as ca
from casadi import SX, vertcat, exp
import time

def export_egerstedt_model() -> AcadosModel:
    model_name = 'egerstedt_standard'

    x1 = SX.sym('x1')
    x2 = SX.sym('x2')
    x3 = SX.sym('x3')
    x = vertcat(x1, x2,x3)

    u1, u2, u3 = SX.sym('u1'), SX.sym('u2'), SX.sym('u3')
    u = vertcat(u1, u2, u3)

    x1_dot = SX.sym('x1_dot')
    x2_dot = SX.sym('x2_dot')
    x3_dot = SX.sym('x3_dot')
    xdot = vertcat(x1_dot, x2_dot, x3_dot)
    
    
    f_expl = ca.vertcat(-x1*u1 + (x1+x2)*u2 + (x1-x2)*u3,
                         (x1+2*x2)*u1 + (x1 - 2*x2)*u2 + (x1 + x2)*u3,
                         x1**2 + x2**2
                         )
    f_impl = xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name
    
    # model.cost_y_expr_e = x2
    model.cost_expr_ext_cost_e = x3

    model.x_labels = ['state 1', 'state 1', 'quadrature']
    model.u_labels = ['Control 1', 'Control 2', 'Control 3']
    model.t_label = 'Time [s]'

    return model

def setup_egerstedt_ocp():
    ocp = AcadosOcp()
    model = export_egerstedt_model()
    ocp.model = model
    
    ocp.solver_options.N_horizon = 100
    ocp.solver_options.tf = 1.0
    
    # ocp.solver_options.qp_solver_cond_N = 50  #Doesnt work for this problem
    
    ocp.cost.cost_type_e = 'LINEAR_LS'
    
    ocp.cost.W = np.eye(2)
    ocp.cost.yref = np.zeros(2)
    ocp.cost.Vx = np.array([[1.0,0,0],[0,1.0,0]])
    ocp.cost.Vu = np.array([[0.,0.,0.],[0.,0.,0.]])
    
    
    ocp.constraints.lbu = np.array([0.0, 0.0, 0.0])
    ocp.constraints.ubu = np.array([1.0, 1.0, 1.0])
    ocp.constraints.idxbu = np.array([0, 1, 2])
    
    
    ocp.constraints.x0 = np.array([0.5, 0.5, 0.])
    ocp.constraints.lbx = np.array([0.4])
    ocp.constraints.ubx = np.array([ACADOS_INFTY])
    ocp.constraints.idxbx = np.array([1])
    
    
    ocp.model.con_h_expr = ca.sum(ocp.model.u)
    ocp.constraints.lh = np.array([1.0])
    ocp.constraints.uh = np.array([1.0])
    

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.integrator_type = 'ERK'
    # ocp.solver_options.sim_method_num_steps = 2
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    ocp.solver_options.tol = 1e-6
    ocp.solver_options.nlp_solver_max_iter = 300
    
    ocp_solver = AcadosOcpSolver(ocp)
    
    for i in range(ocp.solver_options.N_horizon):
        ocp_solver.set(i, "u", np.array([1/3, 1/3, 1/3]))
    for i in range(ocp.solver_options.N_horizon+1):
        ocp_solver.set(i, "x", np.array([0.5, 0.5, 0.]))
        
    return ocp_solver

def main():
    ocp_solver = setup_egerstedt_ocp()

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
        labels_list=['Egerstedt Standard'],
        x_labels=ocp_solver.ocp.model.x_labels,
        u_labels=ocp_solver.ocp.model.u_labels,
        idxbu=ocp_solver.ocp.constraints.idxbu,
        lbu=ocp_solver.ocp.constraints.lbu,
        ubu=ocp_solver.ocp.constraints.ubu,
        fig_filename='egerstedt_standard_ocp.png',
    )

if __name__ == '__main__':
	main()