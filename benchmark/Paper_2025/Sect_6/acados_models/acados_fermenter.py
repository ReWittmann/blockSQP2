from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, plot_trajectories, AcadosSim, AcadosSimSolver, ACADOS_INFTY
import numpy as np
import casadi as ca
from casadi import SX, vertcat
import time
from pathlib import Path


def export_fermenter_model() -> AcadosModel:
    model_name = 'fermenter'

    mu_x = 2e5
    mu_p = 5000
    gamma_xg = 5e4
    gamma_x1 = 1e5
    gamma_p1 = 2e4
    gamma_x2 = 1500
    gamma_p2 = 5e4

    P = SX.sym('P')
    S1 = SX.sym('S1')
    S2 = SX.sym('S2')
    E = SX.sym('E')
    V = SX.sym('V')
    G = SX.sym('G')
    P_acc = SX.sym('P_acc')
    S1_acc = SX.sym('S1_acc')
    S2_acc = SX.sym('S2_acc')
    X = vertcat(P, S1, S2, E, V, G, P_acc, S1_acc, S2_acc)

    uS1 = SX.sym('uS1')
    uS2 = SX.sym('uS2')
    uP = SX.sym('uP')
    U = vertcat(uS1, uS2, uP)

    Xdot = SX.sym('Xdot', 9)
    
    u_sum = uS1 + uS2
    dilution = u_sum / (25 * V)
    reaction_rate = E * S1 * S2
    

    dP = mu_p * reaction_rate - P * dilution
    dS1 = -gamma_x1 * reaction_rate * G - gamma_p1 * reaction_rate + (0.42 * uS1 - S1 * u_sum) / (25 * V)
    dS2 = -gamma_x2 * reaction_rate * G - gamma_p2 * reaction_rate + (0.333 * uS2 - S2 * u_sum) / (25 * V)
    dE = mu_x * reaction_rate * G - E * dilution
    dV = uS1 + uS2 - uP
    dG = -gamma_xg * reaction_rate * G - G * dilution
    dP_acc = uP * P + ((uS1 + uS2 - uP) / 25) * P + V * dP
    dS1_acc = 0.0168 * uS1
    dS2_acc = 0.01332 * uS2

    f_expl = vertcat(dP, dS1, dS2, dE, dV, dG, dP_acc, dS1_acc, dS2_acc)
    f_impl = Xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = X
    model.xdot = Xdot
    model.u = U
    model.name = model_name
    
    model.cost_expr_ext_cost_e = (2 * S1_acc * S2_acc) / (P_acc + 1e-6)

    model.x_labels = ['P', 'S1', 'S2', 'E', 'V', 'G', 'P_acc', 'S1_acc', 'S2_acc']
    model.u_labels = ['uS1', 'uS2', 'uP']
    model.t_label = 't'

    return model

def setup_fermenter_ocp():
    # --- OCP Setup ---
    ocp = AcadosOcp()
    model = export_fermenter_model()
    ocp.model = model
    
    N = 80
    nx = model.x.rows()
    nu = model.u.rows()
    Tf = 1.0
    
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf
    # ocp.solver_options.qp_solver_cond_N = 10

    try:
        cD = Path(__file__).parent
    except:
        cD = Path.cwd()
    ocp.code_gen_options.code_export_directory = str(cD/Path(f"acados_codegen/{model.name}"))

    
    ocp.cost.cost_type_e = 'EXTERNAL'
    
    ocp.constraints.lbu = np.array([0.0, 0.0, 0.0])
    ocp.constraints.ubu = np.array([15.0, 1.0, 30.0])
    ocp.constraints.idxbu = np.array([0, 1, 2])

    lbx = np.array([0.0, 0.0, 0.0, 0.0, 0.3, 0.0])
    ubx = np.array([0.1, 0.04, 0.03, 0.1, 0.45, 0.1])

    lbx_acc = np.array([0.0, 0.0, 0.0])
    ubx_acc = np.array([0.05, 0.2, 0.025])
    ocp.constraints.lbx = np.concatenate([lbx, lbx_acc])
    ocp.constraints.ubx = np.concatenate([ubx, ubx_acc])
    ocp.constraints.idxbx = np.arange(9)

    x0 = np.array([0, 0.03, 0.03, 0.01, 0.3, 0.1, 0, 0.009, 0.009])
    ocp.constraints.lbx_0 = x0
    ocp.constraints.ubx_0 = x0
    ocp.constraints.idxbx_0 = np.arange(nx)
    

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    
    ocp_solver = AcadosOcpSolver(ocp)
    

    sim = AcadosSim()
    sim.model = model
    sim.code_gen_options.code_export_directory = str(cD/Path(f"acados_codegen/{model.name}_sim"))
    
    sim.solver_options.integrator_type = 'IRK'
    sim.solver_options.T = Tf / N
    sim.solver_options.num_steps = 2
    sim_sol = AcadosSimSolver(sim)
    
    u_init_val = np.zeros(nu) 
    u_init_traj = np.tile(u_init_val, (N, 1))
    
    x_current = x0
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
    ocp_solver = setup_fermenter_ocp()

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
        x_traj_list=[simX[:,:6]],
        u_traj_list=[simU],
        time_traj_list=[np.linspace(0, tf, N+1)],
        time_label=ocp_solver.ocp.model.t_label,
        labels_list=['Fermenter Model'],
        x_labels=ocp_solver.ocp.model.x_labels,
        u_labels=ocp_solver.ocp.model.u_labels,
        idxbu=ocp_solver.ocp.constraints.idxbu,
        lbu=ocp_solver.ocp.constraints.lbu,
        ubu=ocp_solver.ocp.constraints.ubu,
        fig_filename='acados_plots/fermenter_ocp.png',
    )
    
if __name__ == '__main__':
	main()