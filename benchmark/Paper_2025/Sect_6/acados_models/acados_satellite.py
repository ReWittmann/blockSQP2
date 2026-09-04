from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, plot_trajectories
import numpy as np
import casadi as ca
from casadi import SX, vertcat
from pathlib import Path


def safe_sqrt(x):
    return ca.sqrt(ca.fmax(x, 1e-12))


def export_satellite_model() -> AcadosModel:
    model_name = 'satellite_deorbiting'

    mu = 3.986e14
    RE = 6.371e6
    rho0 = 1.225
    H = 8500
    CD = 2.2
    A = 1.0
    Isp = 220
    g0 = 9.81
    mdry = 100
    omegaE = 7.2921e-5
    
    rscale = 1e-4
    vrscale = 1.0
    thetascale = 1.0
    vthetascale = 1e-4
    mscale = 1.0
    TSCALE = 1.0
    
    def atmospheric_density(r_val):
        h = r_val - RE
        h_safe = ca.fmax(h, -100000)
        return rho0 * ca.exp(-h_safe / H)

    r_ = SX.sym('r_')
    theta_ = SX.sym('theta_')
    vr_ = SX.sym('vr_')
    vtheta_ = SX.sym('vtheta_')
    m_ = SX.sym('m_')
    T_ = SX.sym('T_')
    X = vertcat(r_, theta_, vr_, vtheta_, m_, T_)

    
    r = r_ / rscale + RE
    vr = vr_ / vrscale
    vtheta = vtheta_ / vthetascale
    m = m_ / mscale
    T = T_ / TSCALE
    
    ur = SX.sym('ur')
    utheta = SX.sym('utheta')
    U = vertcat(ur, utheta)

    Xdot = SX.sym('Xdot', 6)
    
    rsafe = ca.fmax(r, RE + 10000)
    msafe = ca.fmax(m, mdry)
    
    hsafe = ca.fmax(rsafe - RE, -100000)
    rho = rho0 * ca.exp(-hsafe / H)
    
    vrelr = vr
    vreltheta = vtheta - omegaE * rsafe
    vrel = safe_sqrt(vrelr**2 + vreltheta**2)
    
    centrifugal = vtheta**2 / rsafe
    gravity = mu / (rsafe**2)
    drag = 0.5 * CD * A / msafe * rho * vrel
    rthrust = ur / msafe
    thetathrust = utheta / msafe
    
    f_expl = T * vertcat(
        vr * rscale,
        (vtheta / rsafe) * thetascale,
        (centrifugal - gravity + rthrust - drag*vrelr) * vrscale,
        (-vr*vtheta/rsafe + thetathrust - drag*vreltheta)*vthetascale,
        (-ca.sqrt(ur**2 + utheta**2) / (Isp * g0)) * mscale,
        0.0
    )
    
    f_impl = Xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = X
    model.xdot = Xdot
    model.u = U
    model.name = model_name
    

    model.x_labels = ['r_s', 'theta_s', 'vr_s', 'vtheta_s', 'm_s', 't']
    model.u_labels = ['ur', 'utheta']
    model.t_label = 'Time'
    
    model.cost_expr_ext_cost_e = T
    return model

def setup_satellite_ocp():
    ocp = AcadosOcp()
    model = export_satellite_model()
    ocp.model = model
    
    mu = 3.986e14
    RE = 6.371e6
    umax = 20
    m0 = 150
    mdry = 100
    
    h0 = 450000
    hreentry = 120000
    
    
    rscale = 1e-4
    vrscale = 1.0
    thetascale = 1.0
    vthetascale = 1e-4
    mscale = 1.0
    TSCALE = 1.0
    
    
    Tf_init = 1800.0 
    N = 100
    nx = model.x.rows()
    nu = model.u.rows()
    
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = 1.0
    # ocp.solver_options.qp_solver_cond_N = 4    #doesnt work for this problem
    try:
        cD = Path(__file__).parent
    except:
        cD = Path.cwd()
    ocp.code_gen_options.code_export_directory = str(cD/Path(f"acados_codegen/{model.name}"))


    ocp.cost.cost_type_e = 'EXTERNAL'
    
    
    ocp.constraints.lbu = np.array([-umax, -umax])
    ocp.constraints.ubu = np.array([umax, umax])
    ocp.constraints.idxbu = np.array([0, 1])
    
    
    r0 = RE + h0
    theta0 = 0.
    vr0 = 0.
    vorb = np.sqrt(mu/r0)
            
    rfinal = RE + hreentry
    
    x0 = np.array([
        (r0 - RE) * rscale, 
        theta0 * thetascale, 
        vr0 * vrscale, 
        vorb * vthetascale, 
        m0 * mscale
    ])
    ocp.constraints.lbx_0 = np.array([*x0, 300*TSCALE])
    ocp.constraints.ubx_0 = np.array([*x0, 21600*TSCALE])
    ocp.constraints.idxbx_0 = np.arange(nx)
    
    ocp.constraints.lbx = np.array([(RE+5000. - RE)*rscale, -2*np.pi*thetascale, -10000.*vrscale, 0.*vthetascale, (mdry - 0.1)*mscale])
    ocp.constraints.ubx = np.array([(r0 + 100000. - RE)*rscale, 2*np.pi*thetascale, 10000.*vrscale, 20000*vthetascale, (m0 + 0.1)*mscale])
    ocp.constraints.idxbx = np.array(np.arange(nx-1))
    
    
    ocp.constraints.lbx_e = np.array([0.])
    ocp.constraints.ubx_e = np.array([(rfinal - RE)*rscale])
    ocp.constraints.idxbx_e = np.array([0])
    
    ocp.model.con_h_expr = safe_sqrt(ocp.model.u[0]**2 + ocp.model.u[1]**2)
    ocp.constraints.lh = np.array([0.])
    ocp.constraints.uh = np.array([umax])
    
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    
    ocp_solver = AcadosOcpSolver(ocp)
    
    
    r_init = (np.linspace(r0, rfinal, N + 1) - RE) * rscale
    theta_init = np.linspace(theta0, 2*np.pi, N + 1) * thetascale
    vr_init = np.zeros(N + 1) * rscale
    vtheta_init = np.ones(N + 1)*vorb*0.9 * rscale
    m_init = np.linspace(m0, mdry + 10, N + 1) * mscale
    
    for i in range(N+1):
        ocp_solver.set(i, "x", np.array([r_init[i], theta_init[i], vr_init[i], vtheta_init[i], m_init[i], Tf_init]))
        if i == N:
            break
        ocp_solver.set(i, "u", np.array([-5., -10.]))
    return ocp_solver

def main():
    ocp_solver = setup_satellite_ocp()
    
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
        time_traj_list=[np.linspace(0, 1.0, N+1) * simX[0,-1]],
        time_label='Time [s]',
        labels_list=['Satellite Deorbiting'],
        x_labels=ocp_solver.ocp.model.x_labels,
        u_labels=ocp_solver.ocp.model.u_labels,
        idxbu=ocp_solver.ocp.constraints.idxbu,
        lbu=ocp_solver.ocp.constraints.lbu,
        ubu=ocp_solver.ocp.constraints.ubu,
        fig_filename='acados_plots/satellite_deorbiting_ocp.png',
    )
    
if __name__ == '__main__':
	main()