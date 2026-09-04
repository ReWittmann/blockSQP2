# blockSQP2 -- A structure-exploiting nonlinear programming solver based
#              on blockSQP by Dennis Janka.
# Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>

# Licensed under the zlib license. See LICENSE for more details.

# \file OCProblems.cpp
# \author Reinhold Wittmann
# \date 2024-2026
#
# Fatrop solver compatible versions of benchmark problems found in OCProblems.py
#  ...

import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
import math
import OCProblems


class Apollo_Reentry_noParams(OCProblems.Apollo_Reentry):
    default_params = {
        'R': 209,
        'beta': 4.26,
        'rho0': 2.704e-3,
        'g': 3.2172e-4,
        'Sm': 53200,
        'c1': 1.175,
        'c2': 0.9,
        'c3': 0.6
        }
    
    def build_problem(self):
        R, beta, rho0, g, Sm, c1, c2, c3 = (self.model_params[key] for key in ('R', 'beta', 'rho0', 'g', 'Sm', 'c1', 'c2', 'c3'))
        
        self.set_OCP_data(3+1,0,1,1,[0.2, -0.2, 0.006]+[220/self.ntS],[0.4, 0.1, 0.02]+[240/self.ntS],[],[],[-np.pi/2],[np.pi/2])
        self.fix_initial_value([0.36, -8.1*(np.pi/180), 4./R, None])
        self.fix_time_horizon(0, 1)
        
        x = cs.MX.sym('x', 4)
        v, gamma, xi, dt = cs.vertsplit(x, 1)
        u = cs.MX.sym('u')
        
        C_D = c1 - c2*cs.cos(u)
        C_L = c3*cs.sin(u)
        rho = rho0 * cs.exp(-beta*R*xi)

        vdot = -0.5*Sm*rho*v**2 * C_D - g*cs.sin(gamma)/(1 + xi)**2
        gammadot = 0.5*Sm*rho*v*C_L + v*cs.cos(gamma)/(R*(1 + xi)) - g*cs.cos(gamma)/(v*(1 + xi)**2)
        xidot = v*cs.sin(gamma)/R
        
        ode_rhs = cs.vertcat(vdot, gammadot, xidot, 0)
        quad_rhs = 10 * v**3 * cs.sqrt(rho)
        
        dt_ = cs.MX.sym('dt_', 1)
        self.ODE = {'x': x, 'p':cs.vertcat(dt_, u),'ode': dt*ode_rhs, 'quad': dt*quad_rhs}
        self.multiple_shooting()
        self.set_objective(self.q_tf)
        self.add_constraint(self.x_eval[:-1,-1], [0.27, 0., 2.5/R], [0.27, 0., 2.5/R])
        self.build_NLP()
        
        self.start_point = np.zeros(self.nVar)
        for i in range(self.ntS):
            self.set_stage_param(self.start_point, i, 230.0/self.ntS)
            self.set_stage_control(self.start_point, i, 0.5)
            self.set_stage_state(self.start_point, i, self.x_init[:3] + [230.0/self.ntS])
        self.set_stage_state(self.start_point, self.ntS, self.x_init[:3] + [230.0/self.ntS])
    
    def plot(self, xi, dpi = None, title = None, it = None):
        v, gamma, xivar, dt_arr = self.get_state_arrays_expanded(xi)
        u = self.get_control_plot_arrays(xi)
        time_grid_ref = np.cumsum(dt_arr).reshape(-1)
        
        fig,ax = plt.subplots(dpi=dpi)
        ax.plot(time_grid_ref, v, 'tab:green', linestyle='-.', label = r'$v$')
        ax.plot(time_grid_ref, gamma, 'tab:blue', linestyle='--', label = r'$\gamma$')
        ax.plot(time_grid_ref, xivar*10, 'tab:olive', linestyle='-.', label = r'$\xi\cdot 10$')
        
        ax.step(time_grid_ref, u/5, 'tab:red', linestyle='-', label = r'$u/5$')
        ax.legend(fontsize='x-large', loc = 'upper right')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Reentry problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
        
        ax.set_xlabel('t', fontsize = 17.5)
        ax.xaxis.set_label_coords(1.015,-0.006)
        
        plt.show()
        plt.close()


class Batch_Distillation_noParams(OCProblems.Batch_Distillation):
    def build_problem(self):
        M0init, MDinit, x0init, xinit, xCinit, xDinit, alpha, V, m, mC = (self.model_params[key] for key in ['M0init', 'MDinit', 'x0init', 'xinit', 'xCinit', 'xDinit', 'alpha', 'V', 'm', 'mC'])
        self.set_OCP_data(10+1,0,1,0, [0.]*8 + [0.] + [MDinit*self.MDscale] + [0.5/self.ntS * self.tscale], [np.inf] + [self.x0scale] + [1.0]*5 + [self.xCscale, self.xDscale] + [np.inf] + [10/self.ntS * self.tscale], [], [], [0. * self.Rscale], [15. * self.Rscale])
        self.fix_initial_value([M0init*self.M0scale,x0init*self.x0scale] + [xinit]*5+[xCinit*self.xCscale,xDinit*self.xDscale,MDinit*self.MDscale] + [None])
        self.fix_time_horizon(0, 1)
        
        X = cs.MX.sym('X',10+1)
        M0_,x0_,x1,x2,x3,x4,x5,xC_,xD_,MD_, dt_ = cs.vertsplit(X)
        M0 = M0_/self.M0scale
        MD = MD_/self.MDscale
        x0 = x0_/self.x0scale
        xC = xC_/self.xCscale
        xD = xD_/self.xDscale
        dt = dt_/self.tscale
        
        R_ = cs.MX.sym('R')
        R = R_/self.Rscale
        dt_dummy = cs.MX.sym('dt_dummy')
        
        L = R/(1+R) * V
        
        y = lambda x: x*(1+alpha)/(x+alpha)
        
        ode_rhs = cs.vertcat(self.M0scale*(-V + L),
                             self.x0scale*(1/M0 * (L*x1 - V*y(x0) + (V - L)*x0)),
                             1/m * (L*x2 - V*y(x1) + V*y(x0) - L*x1),
                             1/m * (L*x3 - V*y(x2) + V*y(x1) - L*x2),
                             1/m * (L*x4 - V*y(x3) + V*y(x2) - L*x3),
                             1/m * (L*x5 - V*y(x4) + V*y(x3) - L*x4),
                             1/m * (L*xD - V*y(x5) + V*y(x4) - L*x5),
                             self.xCscale*(V/mC * (y(x5) - xC)),
                             self.xDscale*((V - L)/MD * (xC - xD)),
                             self.MDscale*(V - L),
                             0
                             )
        
        self.ODE = {'x':X, 'p':cs.vertcat(dt_dummy,R_), 'ode':dt*ode_rhs}
        self.multiple_shooting()
        self.set_objective((1/self.tscale * self.x_eval[-1,-1]*self.ntS - self.x_eval[9,-1]/self.MDscale))
        self.add_constraint(self.x_eval[8,-1], 0.99*self.xDscale, np.inf)
        self.build_NLP()
        
        self.set_stage_state(self.start_point, 0, 1/self.ntS * self.tscale)
        for j in range(math.floor(0.5*self.ntS)):
            self.set_stage_control(self.start_point, j, 1.0*self.Rscale)
        for j in range(math.floor(0.5*self.ntS), self.ntS):
            self.set_stage_control(self.start_point, j, 15.0*self.Rscale)
        self.integrate_full(self.start_point)
    
    def plot(self, xi, dpi = None, title = None, it = None):
        M0_,x0_,x1,x2,x3,x4,x5,xC_,xD_,MD_, dt_arr = self.get_state_arrays_expanded(xi)
        
        M0 = M0_/self.M0scale
        MD = MD_/self.MDscale
        x0 = x0_/self.x0scale
        xC = xC_/self.xCscale
        xD = xD_/self.xDscale
        
        R = self.get_control_plot_arrays(xi)
        time_grid = np.cumsum(dt_arr/self.tscale).reshape(-1)
        
        fix, ax = plt.subplots(dpi=dpi)
        ax.plot(time_grid, M0/100, 'tab:red', linestyle = '--', label = 'M0/100')
        ax.plot(time_grid, x0, 'g', linestyle = '-.', label = 'x0')
        ax.plot(time_grid, x1, 'b', linestyle = ':', label = 'x1')
        ax.plot(time_grid, x2, 'y', linestyle = '-', label = 'x2')
        ax.plot(time_grid, x3, 'c', linestyle = '-', label = 'x3')
        ax.plot(time_grid, x4, 'm', linestyle = '-', label = 'x4')
        ax.plot(time_grid, x5, 'r', linestyle = '--', label = 'x5')
        ax.plot(time_grid, xC, 'g', linestyle = '--', label = 'xC')
        ax.plot(time_grid, (xD-0.99)*100, 'b', linestyle = '--', label = '(xD-0.99)*100')
        ax.plot(time_grid, MD/100, 'y', linestyle = '--', label = 'MD/100')
        
        ax.step(time_grid, R/10 / self.Rscale, 'tab:red', label = 'R/10')
        ax.legend(fontsize='large')
        
        self.finish_plot(ax, title, it, 'Batch distillation problem')


class Goddard_Rocket_noParams(OCProblems.Goddard_Rocket):
    def build_problem(self):
        #                                                                   Set upper bound to time, so fatrop does not enter region of local infeasibility
        self.set_OCP_data(3 + 1,0,1,0,[1.0,0.,0.,0.],[np.inf,np.inf,np.inf,0.25/self.ntS],[],[],[0],[1])
        self.fix_initial_value(self.model_params['x_init'] + [None])
        self.fix_time_horizon(0, 1)
        
        x = cs.MX.sym('x', self.nx)
        r,v,m,dt = cs.vertsplit(x)
        r0,v0,m0,_ = self.x_init
        
        u = cs.MX.sym('u', self.nu)
        Tmax, A, b, k, rT, C = (self.model_params[key] for key in ('Tmax', 'A', 'b', 'k', 'rT', 'C'))
        
        ode_rhs = cs.vertcat(v,\
                            -1/(r**2) + (1/m) * (Tmax*u - A*(v**2) * cs.exp(-k * (r - r0))),\
                            -b*u,
                            0
                            )
        dt_ = cs.MX.sym('dt_')
        self.ODE = {'x': x, 'p':cs.vertcat(dt_, u),'ode': dt*ode_rhs}
        self.multiple_shooting()
        
        r_eval = self.x_eval[0,1:]
        v_eval = self.x_eval[1,1:]
        
        max_drag_expr = A*(v_eval**2) * cs.exp(-k * (r_eval - r0))
        self.add_constraint(max_drag_expr, -np.inf, C)
        self.add_constraint(r_eval[-1], rT, np.inf)
        
        self.start_point = np.zeros(self.nVar)
        nt_acc = math.ceil(self.ntS*2/5)
        nt_dec = math.floor(self.ntS*3/5)
        for i in range(nt_acc):
            self.set_stage_control(self.start_point, i, [1.0])
        for i in range(nt_acc,nt_acc+nt_dec):
            self.set_stage_control(self.start_point, i, [0.0])
        self.set_stage_state(self.start_point, 0, 0.4/(b*0.4)/self.ntS)
        
        self.integrate_full(self.start_point)
        self.set_objective(-self.x_eval[2,-1])
        self.build_NLP()
    
    def plot(self, xi, dpi = None, title = None, it = None):
        u = self.get_control_plot_arrays(xi)
        r,v,m,t_arr = self.get_state_arrays_expanded(xi)
        time_grid = np.cumsum(t_arr).reshape(-1)
        
        fig, ax = plt.subplots(dpi=dpi)
        ax.plot(time_grid, (r - 1)*100, 'tab:blue', linestyle = ':', label = r'$(r-1)\cdot 100$')
        ax.plot(time_grid, v*20, 'tab:green', linestyle = '--', label = r'$v\cdot 20$')
        ax.plot(time_grid, m, 'tab:olive', linestyle = '-.', label = '$m$')
        
        
        ax.step(time_grid, u, 'tab:red', label = '$u$')
        ax.legend(fontsize = 'large')
        
        self.finish_plot(ax, title, it, 'Goddard\'s rocket problem')


class Batch_Reactor_OED_noQuads(OCProblems.Batch_Reactor_OED):
    def build_problem(self):                                                                                                 # Increase lower control bound to force good? local optimum
        self.set_OCP_data(2 + 4*2 + 10 + 2, 0, 1 + 2, 2-2, [-np.inf,-np.inf] + [-np.inf]*18 + [-np.inf]*2, [np.inf,np.inf] + [np.inf]*18 + [np.inf]*2, [], [], [298 + 60] + [0.]*2, [398] + [1.]*2)
        self.mark_state_bounds_implicit()
        
        p1, p2, p3, p4, M1, M2, reg_init, p1scale, p2scale, p3scale, p4scale = (self.model_params[key] for key in ['p1', 'p2', 'p3', 'p4', 'M1', 'M2', 'reg_init', 'p1scale', 'p2scale', 'p3scale', 'p4scale'])
        self.fix_initial_value([1.0,0.0] + [0.]*18 + [0.]*2)
        self.fix_time_horizon(0,1)
        
        x = cs.MX.sym('x', 2)
        x1,x2 = cs.vertsplit(x)
        
        theta = cs.MX.sym('theta', 4)
        p1_s, p2_s, p3_s, p4_s = cs.vertsplit(theta)
        
        p1_s, p2_s, p3_s, p4_s = p1_s/p1scale, p2_s/p2scale, p3_s/p3scale, p4_s/p4scale
        p1, p2, p3, p4 = p1*p1scale, p2*p2scale, p3*p3scale, p4*p4scale
        
        
        T = cs.MX.sym('T', 1)
        k1 = p1_s*cs.exp(-p2_s/T)
        k2 = p3_s*cs.exp(-p4_s/T)
        
        f_expr = cs.vertcat(-k1*x1**2, 
                             k1*x1**2 - k2*x2
                             )
        f_x_expr = cs.jacobian(f_expr, x)
        f_theta_expr = cs.jacobian(f_expr, theta)
        
        
        f = cs.Function('f', [x,T,theta], [f_expr])
        f_x = cs.Function('f', [x,T,theta], [f_x_expr])
        f_theta = cs.Function('f', [x,T,theta], [f_theta_expr])
        
        f_expr = f(x,T,cs.DM([p1, p2, p3, p4]))
        f_x_expr = f_x(x,T,cs.DM([p1, p2, p3, p4]))
        f_theta_expr = f_theta(x,T,cs.DM([p1, p2, p3, p4]))
        
        
        G = cs.MX.sym('G', x.numel(), theta.numel())
        dG = f_x_expr@G + f_theta_expr
        G_rhs = cs.vec(dG)
        
        w = cs.MX.sym('w', 2)
        w1,w2 = cs.vertsplit(w)
        
        F = cs.MX.sym('F', (theta.numel()*(theta.numel() + 1))//2)
        dh1, dh2 = cs.DM([1,0]), cs.DM([0,1])
        dF = w1*(dh1.T@G).T @ (dh1.T@G) + w2*(dh2.T@G).T @ (dh2.T@G)
        
        F_rhs = cs.vertcat(dF[0,0], dF[1,0], dF[2,0], dF[3,0], dF[1,1], dF[2,1], dF[3,1], dF[2,2], dF[3,2], dF[3,3])
        
        quad_expr = w
        ode_rhs = cs.vertcat(f_expr, G_rhs, F_rhs, quad_expr)
        
        dt = cs.MX.sym('dt', 1)
        qstates = cs.MX.sym('qstates', 2)
        
        self.ODE = {'x': cs.vertcat(x, cs.vec(G), F, qstates), 'p':cs.vertcat(dt, T, w),'ode': dt*ode_rhs, 'quad':dt*quad_expr}
        self.multiple_shooting()
        
        F_rhs_tf = self.x_eval[2 + 4*2 : 2 + 4*2 + 10, -1]
        
        F_tf = cs.MX.zeros(4,4)
        for j in range(4):
            for i in range(0, j):
                F_tf[i,j] = F_rhs_tf[j + i*4 - (i*(i+1))//2]
            F_tf[j,j] = F_rhs_tf[j*5 - (j*(j+1))//2] + reg_init
            for i in range(j + 1, 4):
                F_tf[i,j] = F_rhs_tf[i + j*4 - (j*(j+1))//2]
        
        self.set_objective(cs.trace(cs.inv(F_tf))/4)
        # self.set_objective(self.q_tf[3,-1])
        self.add_constraint(self.x_eval[-2:,-1], [0., 0.], [M1, M2])
        
        self.build_NLP()
        
        self.start_point = np.zeros(self.nVar)
        for i in range(self.ntS):
            self.set_stage_control(self.start_point, i, 398)
            self.set_stage_state(self.start_point, i, self.x_init)
        self.set_stage_state(self.start_point, self.ntS, self.x_init)
        self.integrate_full(self.start_point)


class Calcium_Oscillation_noParams(OCProblems.Calcium_Oscillation):
    def build_problem(self):
        self.set_OCP_data(4+1,0,1,1,[0,0,0,0] + [1.1],[np.inf,np.inf,np.inf,np.inf] + [1.3],[], [], [1], [np.inf])
        self.fix_initial_value([0.03966, 1.09799, 0.00142, 1.65431] + [None])
        self.mark_state_bounds_implicit()
        
        x = cs.MX.sym('x', 4+1)
        x0,x1,x2,x3, wmax = cs.vertsplit(x)
        w = cs.MX.sym('w')
        # wmax = cs.MX.sym('wmax')
        dt = cs.MX.sym('dt')
        
        t0, tf, k1, k2, k3, K4, k5, K6, k7, k8, K9, k10, K11, k12, k13, k14, K15, k16, K17, p1, tx0, tx1, tx2, tx3 = (self.model_params[key] for key in ('t0', 'tf', 'k1', 'k2', 'k3', 'K4', 'k5', 'K6', 'k7', 'k8', 'K9', 'k10', 'K11', 'k12', 'k13', 'k14', 'K15', 'k16', 'K17', 'p1', 'tx0', 'tx1', 'tx2', 'tx3'))
        self.fix_time_horizon(t0,tf)
        
        ode_rhs = cs.vertcat(
            k1 + k2*x0 - (k3*x0*x1)/(x0 + K4) - (k5*x0*x2)/(x0 + K6),
            k7*x0 - (k8*x1)/(x1+K9),
            (k10*x1*x2*x3)/(x3 + K11) + k12*x1 + k13*x0 - (k14*x2)/((1 + w*(wmax-1.0))*x2 + K15) - (k16*x2)/(x2 + K17) + x3/10,
            -(k10*x1*x2*x3)/(x3 + K11) + (k16*x2)/(x2+K17) - x3/10,
            0
            )
        quad_expr = (x0 - tx0)**2 + (x1 - tx1)**2 +(x2 - tx2)**2 + (x3 - tx3)**2 + p1*w
        
        self.ODE = {'x':x, 'p':cs.vertcat(dt,w), 'ode':dt*ode_rhs, 'quad': dt*quad_expr}
        
        self.multiple_shooting()
        
        self.set_objective(self.q_tf)
        self.add_constraint(self.u_eval - self.x_eval[-1,:-1], -np.inf, 0)
        
        self.build_NLP()
        self.start_point = np.zeros(self.nVar)
        
        self.set_stage_state(self.start_point, 0, 1.3)
        for i in range(self.ntS):
            # self.set_stage_param(self.start_point, i, 1.3)
            self.set_stage_control(self.start_point, i, 1.0)
        
            
        self.integrate_full(self.start_point)
        
        #Prevent local minimum with second stimulus (but better objective)
        for i in range(math.floor(0.4*self.ntS), self.ntS):
            self.set_stage_control(self.ub_var, i, 1.0)
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x0, x1, x2, x3,_ = self.get_state_arrays_expanded(xi)
        w = self.get_control_plot_arrays(xi)        
        
        fig, ax = plt.subplots(dpi = dpi)
        ax.plot(self.time_grid_ref, x0, 'tab:olive', linestyle = '--', label = r'$x_0$')
        ax.plot(self.time_grid_ref, x1, 'tab:green', linestyle = '-.', label = r'$x_1$')
        ax.plot(self.time_grid_ref, x2, 'tab:cyan', linestyle = ':', label = r'$x_2$')
        ax.plot(self.time_grid_ref, x3, 'tab:blue', linestyle = '-.', label = r'$x_3$')
        ax.step(self.time_grid_ref, (w-1.0)*20, 'tab:red', label = r'$(w-1)\cdot 20$')
        ax.legend(fontsize='large')
        
        self.finish_plot(ax, title, it, 'Calcium Oscillation problem')


class Catalyst_Mixing_OED_noQuads(OCProblems.Catalyst_Mixing_OED):
    def build_problem(self):
        self.set_OCP_data(2 + 4 + 3 + 2,0,3,2-2,[-np.inf,-np.inf] + [-np.inf]*7 + [-np.inf]*2,[np.inf,np.inf] + [np.inf]*7 + [np.inf]*2,[],[],[0.] + [0.]*2,[1.] + [1.]*2)
        self.fix_time_horizon(0,1)
        T_l = 1.0
        self.fix_initial_value([1.,0.] + [0.]*7 + [0.]*2)
        self.mark_state_bounds_implicit()
        
        p1,p2,p3,M1,M2,reg_init, p1scale, p2scale = (self.model_params[key] for key in ('p1', 'p2', 'p3', 'M1', 'M2', 'reg_init', 'p1scale', 'p2scale'))
        
        x = cs.MX.sym('x', 2)
        x1,x2 = cs.vertsplit(x)
        u = cs.MX.sym('u',1)
        p = cs.MX.sym('p', 2)
        p1_s, p2_s = cs.vertsplit(p)
        
        p1_s, p2_s = p1_s/p1scale, p2_s/p2scale
        p1, p2 = p1*p1scale, p2*p2scale
        
        dt = cs.MX.sym('dt', 1)
        f_expr = cs.vertcat(u*(p2_s*x2 - p1_s*x1), 
                            u*(p1_s*x1 - p2_s*x2) - (1-u)*p3*x2
                            )
        f_x_expr = cs.jacobian(f_expr, x)
        f_p_expr = cs.jacobian(f_expr, p)
        
        
        f = cs.Function('f', [x,u,p], [f_expr])
        f_x = cs.Function('f', [x,u,p], [f_x_expr])
        f_p = cs.Function('f', [x,u,p], [f_p_expr])
        
        f_expr = f(x,u,cs.DM([p1,p2]))
        f_x_expr = f_x(x,u,cs.DM([p1,p2]))
        f_p_expr = f_p(x,u,cs.DM([p1,p2]))
        
        
        G = cs.MX.sym('G', x.numel(), p.numel())
        dG = f_x_expr@G + f_p_expr
        G_rhs = cs.vec(dG)
        
        w = cs.MX.sym('w', 2)
        w1,w2 = cs.vertsplit(w)
        F = cs.MX.sym('F', (p.numel()*(p.numel() + 1))//2)
        dh1, dh2 = cs.DM([1,0]), cs.DM([0,1])
        dF = w1*(dh1.T@G).T @ (dh1.T@G) + w2*(dh2.T@G).T @ (dh2.T@G)
        
        F11scale = 1.0
        F21scale = 1.0
        F22scale = 1.0
        F_rhs = cs.vertcat(F11scale*dF[0,0], F21scale*dF[1,0], F22scale*dF[1,1])
        
        quad_expr = w
        ode_rhs = cs.vertcat(f_expr, G_rhs, F_rhs, quad_expr)
        
        dt = cs.MX.sym('dt', 1)
        qstates = cs.MX.sym('qstates', 2)
        self.ODE = {'x': cs.vertcat(x, cs.vec(G), F, qstates), 'p':cs.vertcat(dt, u, w),'ode': dt*ode_rhs}
        self.multiple_shooting()
        
        F_rhs_tf_11 = self.x_eval[2 + 2*2 + 0,-1]/F11scale
        F_rhs_tf_21 = self.x_eval[2 + 2*2 + 1,-1]/F21scale
        F_rhs_tf_22 = self.x_eval[2 + 2*2 + 2,-1]/F22scale
        F_rhs_tf = cs.vertcat(F_rhs_tf_11, F_rhs_tf_21, F_rhs_tf_22)
        
        F_tf = cs.MX.zeros(2,2)
        for j in range(2):
            for i in range(0, j):
                F_tf[i,j] = F_rhs_tf[j + i*2 - (i*(i+1))//2]
            F_tf[j,j] = F_rhs_tf[j*3 - (j*(j+1))//2] + reg_init
            for i in range(j + 1, 2):
                F_tf[i,j] = F_rhs_tf[i + j*2 - (j*(j+1))//2]
        
        self.set_objective(cs.trace(cs.inv(F_tf)))
        self.add_constraint(self.x_eval[-2:,-1] - cs.DM([M1,M2]), -np.inf, 0.)
        self.build_NLP()
        for i in range(self.ntS):
            self.set_stage_control(self.start_point, i, [0.5,M1/T_l,M2/T_l])
        self.integrate_full(self.start_point)
    
    def plot(self, xi, dpi = None, title = None, it = None):
        fig, ax = plt.subplots(dpi=dpi)
        x1,x2, G11, G21, G12, G22, F11, F21, F22,_,_ = self.get_state_arrays_expanded(xi)
        u, w1, w2 = self.get_control_plot_arrays(xi)
        ax.plot(self.time_grid_ref, x1, 'tab:green', linestyle='-.', label = r'$x_1$')
        ax.plot(self.time_grid_ref, x2, 'tab:blue', linestyle='--', label = r'$x_2$')
        ax.step(self.time_grid_ref, u, 'tab:red', linestyle='-', label = r'$u$')
        ax.step(self.time_grid_ref, w1, 'tab:green', linestyle='--', label = r'$w_1$')
        ax.step(self.time_grid_ref, w2, 'tab:blue', linestyle='-.', label = r'$w_2$')
        
        ax.legend(fontsize = 'large')
        
        self.finish_plot(ax, title, it, "Catalyst mixing OED")

class Cushioned_Oscillation_noParams(OCProblems.Cushioned_Oscillation):    
    def build_problem(self):
        m,c,x0,v0,umm = (self.model_params[key] for key in ['m', 'c', 'x0', 'v0', 'umm'])
        self.set_OCP_data(2+1,1-1,1,0,[-np.inf,-np.inf] + [8/self.ntS], [np.inf,np.inf] + [20/self.ntS], [], [], [-umm], [umm])
        self.mark_state_bounds_implicit()
        self.fix_time_horizon(0, 1)
        
        X = cs.MX.sym('X',3)
        x,v,dt = cs.vertsplit(X)
        u = cs.MX.sym('u',1)
        self.fix_initial_value([x0,v0, None])
        
        ode_rhs = cs.vertcat(v, 1/m * (u - c*x), 0)
        
        dt_dummy = cs.MX.sym('dt_dummy')
        self.ODE = {'x':X, 'p':cs.vertcat(dt_dummy,u), 'ode':dt*ode_rhs}
        self.multiple_shooting()
        self.set_objective(self.ntS*self.x_eval[2,-1])
        self.add_constraint(cs.vec(self.x_eval[:2,-1] - cs.DM([0.,0.])), [0.,0.], [0.,0.])
        
        self.build_NLP()
        self.set_stage_state(self.start_point, 0, 10/self.ntS)
        for i in range(1,self.ntS):
            self.set_stage_state(self.start_point, i, self.x_init[:2] + [10/self.ntS])
        self.set_stage_state(self.start_point, self.ntS, self.x_init[:2] + [10/self.ntS])
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x,v, t_arr = self.get_state_arrays(xi)
        u = self.get_control_plot_arrays(xi)
        time_grid = np.cumsum(t_arr)
        
        fig, ax = plt.subplots(dpi=dpi)
        ax.plot(time_grid, x, 'tab:blue', linestyle = '--', label = r'$x$')
        ax.plot(time_grid, v, 'tab:green', linestyle = '-.', label = r'$v$')
        ax.step(time_grid, u, 'tab:red', label = r'$u$')
        ax.legend(loc='upper right', fontsize = 'large')
        
        ax.set_xlabel('t', fontsize = 17.5)
        ax.xaxis.set_label_coords(1.015,-0.006)
        
        self.finish_plot(ax, title, it, 'Cushioned oscillation problem')


class Dielectrophoretic_Particle_noParams(OCProblems.Dielectrophoretic_Particle):
    def build_problem(self):
        self.set_OCP_data(2+1,1-1,1,0,[-np.inf,-np.inf] + [0.01],[np.inf, np.inf] + [np.inf],[],[],[-1],[1])
        x00,xf,alpha,c = (self.model_params[key] for key in ('x00', 'xf', 'alpha', 'c'))
        self.fix_initial_value([x00, 0., None])
        self.mark_state_bounds_implicit()
        self.fix_time_horizon(0, 1)
        
        x = cs.MX.sym('x', 2+1)
        u = cs.MX.sym('u', 1)
        x0, x1, dt = cs.vertsplit(x)
        ode_rhs = cs.vertcat(x1*u + alpha*u**2, -c*x1 + u, 0)
        dt_dummy = cs.MX.sym('dt_dummy', 1)
        self.ODE = {'x': x, 'p':cs.vertcat(dt_dummy, u),'ode': dt*ode_rhs}
        self.multiple_shooting()
        self.set_objective(self.x_eval[2,-1]*self.ntS)
        self.add_constraint(self.x_eval[0,-1], xf, xf)
        self.build_NLP()
        
        self.start_point = np.zeros(self.nVar)
        
        self.set_stage_state(self.start_point, 0, 5.0/self.ntS)
        for i in range(self.ntS):
            self.set_stage_control(self.start_point, i, [1.0])
        self.integrate_full(self.start_point)
        
    def plot(self, xi, dpi = None, title = None, it = None):
        x0, x1, dt_arr = self.get_state_arrays_expanded(xi)
        u = self.get_control_plot_arrays(xi)
        time_grid_ref = np.cumsum(dt_arr)
        
        # plt.figure(dpi = dpi)
        fig,ax = plt.subplots(dpi=dpi)
        ax.plot(time_grid_ref, x0, 'tab:green', linestyle='-.', label = '$x_0$')
        ax.plot(time_grid_ref, x1, 'tab:blue', linestyle='--', label = '$x_1$')
        ax.step(time_grid_ref, u, 'tab:red', linestyle='-', label = r'$u$')
        ax.legend(fontsize='x-large')
        self.finish_plot(ax, title, it, 'Dielectrophoretic Particle problem')


class D_Onofrio_Chemotherapy_noQuads(OCProblems.D_Onofrio_Chemotherapy):
    def build_problem(self):
        zeta, b, mu, d, G, x20, x30, u0max, x2max, x00, x10, u1max, x3max, F, eta, alpha, tF = (self.model_params[key] for key in ('zeta','b','mu','d','G','x20','x30','u0max','x2max','x00','x10','u1max','x3max','F','eta', 'alpha', 'tF'))
        self.set_OCP_data(2 + 3,0,2,3,[0.,0.] + [-np.inf]*3, [np.inf,np.inf] + [np.inf]*3, [], [], [0.,0.],[u0max,u1max])
        self.fix_initial_value([x00,x10] + [0.]*3)
        self.fix_time_horizon(0., tF)
        # self.mark_state_bounds_implicit()
        
        x = cs.MX.sym('x', 2)
        x0,x1 = cs.vertsplit(x)
        u = cs.MX.sym('u', 2)
        u0,u1 = cs.vertsplit(u)
        dt = cs.MX.sym('dt')
        
        ode_rhs = cs.vertcat(-zeta*x0*cs.log(x0/x1) - F*x0*u1,
                             b*x0 - mu*x1 - d*x0**(2./3.)*x1 - G*u0*x1 - eta*x1*u1,
                             )
        quad = cs.vertcat(u0**2,u0,u1)
        
        qstates = cs.MX.sym('qstates', 3)
        self.ODE = {'x':cs.vertcat(x, qstates), 'p':cs.vertcat(dt,u), 'ode': dt*cs.vertcat(ode_rhs, quad)}
        self.multiple_shooting()
        self.set_objective(self.x_eval[0,-1] + alpha*self.x_eval[2+0,-1])
        self.add_constraint(self.x_eval[2+1:2+3,-1] - cs.DM([x2max,x3max]), [-np.inf,-np.inf], [0.,0.])
        self.build_NLP()
        
        for i in range(self.ntS):
            self.set_stage_control(self.start_point, i, [x2max/tF, x3max/tF])
        self.integrate_full(self.start_point)
    
    def plot(self, xi, dpi = None, title = None, it = None):
        time_grid = self.time_grid
        x0,x1,_,_,_ = self.get_state_arrays(xi)
        u0,u1 = self.get_control_plot_arrays(xi)
        
        fig, ax = plt.subplots(dpi = dpi)
        ax.plot(time_grid, x0/100., 'tab:green', linestyle = '--', label = r'$x_0/100$')
        ax.plot(time_grid, x1/100., 'tab:blue', linestyle = ':', label = r'$x_1/100$')
        ax.step(time_grid, u0, 'tab:red', label = r'$u_0$')
        ax.step(time_grid, u1*75, 'tab:blue', linestyle = '-.', label = r'$u_1\cdot75$')
        ax.legend(fontsize='large', loc = 'upper right')
        
        self.finish_plot(ax, title, it, 'D\'Onofrio chemotherapy problem')


class Ducted_Fan_noParams(OCProblems.Ducted_Fan):
    
    def build_problem(self):
        self.set_OCP_data(6 + 1,0,2,1,[-np.inf]*2 + [-30] + [-np.inf]*3 + [1.0/self.ntS],[np.inf]*2 + [30] + [np.inf]*3 + [8.0/self.ntS],[],[],[-5., 0.],[5., 17.])
        m, J, r, mg, mu = (self.model_params[key] for key in self.default_params.keys())
        self.fix_initial_value([0.]*6 + [None])
        self.mark_state_bounds_implicit([i != 2 for i in range(self.nx)])
        self.fix_time_horizon(0, 1)
        
        x = cs.MX.sym('x', 6 + 1)
        u = cs.MX.sym('u', 2)
        x1, x2, alpha, dx1, dx2, dalpha, dt = cs.vertsplit(x)
        u1, u2 = cs.vertsplit(u)
        ode_rhs = cs.vertcat(dx1, 
                             dx2,
                             dalpha,
                             1/m*(u1*cs.cos(alpha) - u2*cs.sin(alpha)),
                             1/m * (-mg + u1*cs.sin(alpha) + u2*cs.cos(alpha)),
                             r/J * u1,
                             0
                             )
        quad = 2*u1**2 + u2**2
        dt_dummy = cs.MX.sym('dt_dummy', 1)
        self.ODE = {'x': x, 'p':cs.vertcat(dt_dummy, u),'ode': dt*ode_rhs, 'quad': dt*quad}
        self.multiple_shooting()
        self.set_objective(1/(self.x_eval[6,-1]*self.ntS) * self.q_tf + mu*self.x_eval[6,-1]*self.ntS)
        self.add_constraint(self.x_eval[:6,-1], [1] + [0.]*5, [1.] + [0.]*5)
        self.build_NLP()
        
        self.start_point = np.zeros(self.nVar)
        # self.set_stage_state(self.start_point, 0, 5.0/self.ntS)
        for i in range(self.ntS):
            # self.set_stage_param(self.start_point, i, [5.0/self.ntS])
            self.set_stage_state(self.start_point, i, [0.]*6 + [5.0/self.ntS])
            self.set_stage_control(self.start_point, i, [1., 1.])
        # self.integrate_full(self.start_point)
        self.set_stage_state(self.start_point, self.ntS, [0.]*6 + [5.0/self.ntS])
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x1, x2, alpha, _, _, _, dt_arr = self.get_state_arrays_expanded(xi)
        u1,u2 = self.get_control_plot_arrays(xi)
        time_grid_ref = np.cumsum(dt_arr)
        
        # plt.figure(dpi = dpi)
        fig,ax = plt.subplots(dpi=dpi)
        ax.plot(time_grid_ref, x1*5, 'tab:green', linestyle='-.', label = r'$x_1\cdot 5$')
        ax.plot(time_grid_ref, x2*20, 'tab:blue', linestyle='--', label = r'$x_2\cdot 20$')
        ax.plot(time_grid_ref, alpha, 'tab:olive', linestyle='--', label = r'$\alpha$')
        
        ax.step(time_grid_ref, u1, 'tab:red', linestyle='-', label = r'$u_1$')
        ax.step(time_grid_ref, u2, 'tab:cyan', linestyle='-', label = r'$u_2$')

        ax.legend(fontsize='x-large')
        
        self.finish_plot(ax, title, it, 'Ducted fan problem')

# class Fermenter_noQuads(OCProblems.Fermenter):
#     def build_problem(self):
#         self.set_OCP_data(6 + 3, 0, 3, 3-3, [0.,0.,0.,0.,0.3,0.] + [0.,0.,0.], [0.1,0.04,0.03,0.1,0.45,0.1] + [0.05,0.2,0.025], [], [], [0.,0.,0.], [15.,1.,30.])
#         mux, mup, gxg, gx1, gp1, gx2, gp2 = (self.model_params[key] for key in ['mux', 'mup', 'gxg', 'gx1', 'gp1', 'gx2', 'gp2'])
#         self.fix_time_horizon(0.,1.)
#         self.fix_initial_value([0.,0.03,0.03,0.01,0.3,0.1] + [0., 0.009, 0.009])
#         x = cs.MX.sym('x', 6)
#         P,S1,S2,E,V,G = cs.vertsplit(x)
#         u = cs.MX.sym('u', 3)
#         uS1,uS2,uP = cs.vertsplit(u)
#         dt = cs.MX.sym('dt', 1)
        
#         #In Le, first term in rhs for S1, S2 and G enters with positive sign, negative in Janka and MUSCOD
#         #Janka and MUSCOD seem to be correct
        
#         Pdot = mup*E*S1*S2 - P*(uS1+uS2)/(25*V)
#         ode_rhs = cs.vertcat(
#                 Pdot,
#                 -gx1*E*S1*S2*G - gp1*E*S1*S2 + (0.42*uS1 - S1*(uS1 + uS2))/(25*V),
#                 -gx2*E*S1*S2*G - gp2*E*S1*S2 + (0.333*uS2 - S2*(uS1 + uS2))/(25*V),
#                 mux*E*S1*S2*G - E*(uS1 + uS2)/(25*V),
#                 uS1 + uS2 - uP,
#                 -gxg*E*S1*S2*G - G*(uS1+uS2)/(25*V),
#         )
        
#         qstates = cs.MX.sym('qstates', 3)
#         quad = cs.vertcat(uP*P + (uS1 + uS2 - uP)/25 * P + V*Pdot,
#                 0.0168*uS1,
#                 0.01332*uS2)
        
#         self.ODE = {'x':cs.vertcat(x, qstates), 'p':cs.vertcat(dt, u), 'ode':dt*cs.vertcat(ode_rhs, quad)}
#         self.multiple_shooting()
        
#         # P_acc, S1_acc, S2_acc = cs.vertsplit(self.q_tf + cs.DM([0., 0.009, 0.009]))
#         P_acc, S1_acc, S2_acc = cs.vertsplit(self.x_eval[6:9,-1])
        
#         # self.set_objective(2*(self.x_eval[7,-1]*self.x_eval[8,-1])/self.x_eval[6,-1])
#         self.set_objective(2*(S1_acc*S2_acc)/P_acc)

#         self.build_NLP()
#         for i in range(self.ntS):
#             self.set_stage_control(self.start_point, i, [0., 0., 0.])
#         self.integrate_full(self.start_point)
    
    
#     def plot(self, xi, dpi = None, title = None, it = None):
#         P,S1,S2,E,V,G, _, _, _ = self.get_state_arrays(xi)
#         uS1,uS2,uP = self.get_control_plot_arrays(xi)
        
#         fig, ax = plt.subplots(dpi=dpi)
#         ax.plot(self.time_grid, P*10., 'tab:red', linestyle = '--', label = r'$P\cdot 10$')
#         ax.plot(self.time_grid, S1*2, 'tab:green', linestyle = '--', label = r'$S1\cdot 2$')
#         ax.plot(self.time_grid, S2*2, 'tab:brown', linestyle = '--', label = r'$S2\cdot 2$')
#         ax.plot(self.time_grid, E, 'tab:olive', linestyle = '--', label = 'E')
#         ax.plot(self.time_grid, V/3, 'tab:cyan', linestyle = '--', label = 'V/3')
#         ax.plot(self.time_grid, G, 'tab:purple', linestyle = '--', label = 'G')
        
#         ax.step(self.time_grid_ref, uS1/5., 'tab:red', label = r'$u_{S1}/5$')
#         ax.step(self.time_grid_ref, uS2/15., 'tab:green', label = r'$u_{S2}/15$')
#         ax.step(self.time_grid_ref, uP/60., 'tab:grey', label = r'$u_{P}/60$')
        
#         ax.legend(fontsize='medium', loc = 'upper center')
#         ax.set_ylim(0, 0.16)
        
#         self.finish_plot(ax, title, it, "Fermenter problem")


class Hang_Glider_noParams(OCProblems.Hang_Glider):
    def build_problem(self):
        x0, y0, ytf, dxbc, dybc, c0, c1, S, rho, cmax, m, g, uC, rC = (self.model_params[key] for key in ['x0', 'y0', 'ytf', 'dxbc', 'dybc', 'c0', 'c1', 'S', 'rho', 'cmax', 'm', 'g', 'uC', 'rC'])
        self.set_OCP_data(4+1,0,1,0, [0.,0.,-np.inf,-np.inf] + [75/self.ntS], [np.inf,np.inf,np.inf,np.inf] + [1500/self.ntS], [], [], [0], [cmax])
        self.fix_initial_value([x0, dxbc, y0, dybc] + [None])
        self.mark_state_bounds_implicit(False,False,True,True,True)
        self.fix_time_horizon(0, 1)
        
        XY = cs.MX.sym('XY', 4)
        x,dx,y,dy = cs.vertsplit(XY)
        cL = cs.MX.sym('cL')
        dt = cs.MX.sym('dt')
        
        r = (x/rC - 2.5)**2
        u = uC*(1 - r)*cs.exp(-r)
        w = dy - u
        v = cs.sqrt(dx**2 + w**2)
        
        D = 1/2 * (c0 + c1*cL**2)*rho*S*v**2
        L = 1/2 * cL*rho*S*v**2
        
        ode_rhs = cs.vertcat(
                dx,
                1/m * (-L*w/v - D*dx/v),
                dy,
                1/m * (L*dx/v - D*w/v) - g
                )
        
        dt_dummy = cs.MX.sym('dt_dummy')
        self.ODE = {'x':cs.vertcat(XY, dt), 'p':cs.vertcat(dt_dummy,cL), 'ode':dt*cs.vertcat(ode_rhs, 0)}
        self.multiple_shooting()
        self.add_constraint(self.x_eval[1:4,-1] - cs.vertcat(dxbc, ytf, dybc), 0., 0.)
        self.set_objective(-self.x_eval[0,-1])
        self.build_NLP()
        
        self.set_stage_state(self.start_point, 0, 100/self.ntS)
        for j in range(self.ntS):
            self.set_stage_control(self.start_point, j, cmax)
            # self.set_stage_param(self.start_point, j, 100/self.ntS)
        self.integrate_full(self.start_point)
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x, dx, y, dy, dt_arr = self.get_state_arrays_expanded(xi)
        cL = self.get_control_plot_arrays(xi)
        time_grid = np.cumsum(dt_arr).reshape(-1)
        
        fig, ax = plt.subplots(dpi=dpi)
        ax.step(time_grid, cL, 'tab:red', label = r'$c_L$')
        ax.plot(time_grid, x/500, 'tab:green', linestyle = '-', label = r'$x/500$')
        ax.plot(time_grid, (y-900)/100, 'tab:blue', linestyle = '-', label = r'$(y-900)/100$')
        ax.plot(time_grid, dx/10, 'tab:green', linestyle = ':', label = r'$v_x/10$')
        ax.plot(time_grid, dy/10, 'tab:blue', linestyle = ':', label = r'$v_y/10$')
        ax.legend(fontsize='large', loc = 'upper right')
        
        self.finish_plot(ax, title, it, 'Hang glider problem')


class Hanging_Chain_noQuads(OCProblems.Hanging_Chain):
    def build_problem(self):
        self.set_OCP_data(1+1,0,1,2-1, [0.] + [-np.inf], [10.] + [np.inf], [], [], [-10.], [20.])
        
        a,b,Lp = (self.model_params[key] for key in ['a', 'b', 'Lp'])
        self.fix_initial_value([a] + [0.])
        self.fix_time_horizon(0,1)
        self.mark_state_bounds_implicit()
        
        x1 = cs.MX.sym('x1',1)
        u = cs.MX.sym('u',1)
        dt = cs.MX.sym('dt',1)
        ode_rhs = u
        # quad = cs.vertcat(x1*(1.0+u**2)**0.5, (1.0+u**2)**0.5)
        quad = x1*(1.0 + u**2)**0.5
        
        qstates = cs.MX.sym('qstates')
        statequad = (1.0 + u**2)**0.5
        
        self.ODE = {'x':cs.vertcat(x1, qstates), 'p':cs.vertcat(dt,u), 'ode':dt*cs.vertcat(ode_rhs, statequad), 'quad':dt*quad}
        self.multiple_shooting()
        
        self.set_objective(self.q_tf[0])
        # self.add_constraint(self.q_tf[1] - Lp, 0., 0.)
        self.add_constraint(self.x_eval[1,-1] - Lp, 0., 0.)
        
        self.add_constraint(self.x_eval[0,-1] - b, 0.,0.)
        
        self.build_NLP()
        
        if b > a:
            tm = 0.25
        else:
            tm = 0.75
        x1_start = []
        for i in range(self.ntS+1):
            t = self.time_grid[i]
            x1_start.append(2*abs(b - a)*t*(t - 2*tm) + a)
        x1_start = np.array(x1_start)
        u_start = np.diff(x1_start, 1, 0)/np.diff(self.time_grid, 1, 0)
        self.set_stage_control(self.start_point, 0, u_start[0])
        for i in range(1,self.ntS):
            self.set_stage_control(self.start_point, i, u_start[i])
            self.set_stage_state(self.start_point, i, [x1_start[i], u_start[i-1]])
        self.set_stage_state(self.start_point, self.ntS, [x1_start[self.ntS], u_start[self.ntS-1]]) 
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x, _ = self.get_state_arrays(xi)
        
        fig, ax = plt.subplots(dpi=dpi)
        ax.plot(self.time_grid, x, 'k-', label = 'chain')
        ax.legend(fontsize='large')
        
        self.finish_plot(ax, title, it, 'Hanging chain problem')

class Lotka_OED_noQuads(OCProblems.Lotka_OED):
    
    def build_problem(self):
        self.set_OCP_data(9 + 2, 0, 3, 2 - 2, [0.,0.]+[-np.inf]*7 + [-np.inf]*2, [np.inf]*9 + [np.inf]*2,[],[],[0.] + [0.]*2, [float(self.model_params['fishing'])] + [1.]*2)
        tf,p1,p2,p3,p4,c1,c2,x_init,M,epsilon, transform_obj= (self.model_params[key] for key in ['tf', 'p1', 'p2', 'p3', 'p4', 'c1', 'c2','x_init', 'M', 'epsilon', 'transform_obj'])
        self.fix_time_horizon(0.,tf)
        self.fix_initial_value(x_init + [0.]*4 + [epsilon, 0., epsilon] + [0.,0.])
        self.mark_state_bounds_implicit()
        
        S = cs.MX.sym('S', 9)
        x1, x2, G11, G12, G21, G22, F11, F12, F22 = cs.vertsplit(S)
        
        C = cs.MX.sym('C', 3)
        u, w1, w2 = cs.vertsplit(C)
        
        dt = cs.MX.sym('dt', 1)
        ode_rhs = cs.vertcat(
                p1*x1 - p2*x1*x2 - c1*u*x1,
                -p3*x2 + p4*x1*x2 - c2*u*x2,
                (p1 - p2*x2 - c1*u)*G11 + (-p2*x1)*G21 - x1*x2,
                (p1 - p2*x2 - c1*u)*G12 + (-p2*x1)*G22,
                (p4*x2)*G11 + (-p3 + p4*x1 - c2*u)*G21,
                (p4*x2)*G12 + (-p3 + p4*x1 - c2*u)*G22  + x1*x2,
                w1*(G11**2) + w2*(G21**2),
                w1*G11*G12 + w2*G21*G22,
                w1*(G12**2) + w2*(G22**2)
        )
        qstates = cs.MX.sym('qstates', 2)
        quad_expr = cs.vertcat(w1, w2)
        
        
        self.ODE = {'x': cs.vertcat(S, qstates), 'p':cs.vertcat(dt, C),'ode': dt*cs.vertcat(ode_rhs, quad_expr)}
        self.multiple_shooting()
        F11T,F12T,F22T = cs.vertsplit(self.x_eval[6:9,-1])
        
        obj_expr = (1/(F11T*F22T - F12T*F12T))*(F22T + F11T)
        if transform_obj:
            self.set_objective(-obj_expr**-2)
        else:
            self.set_objective(obj_expr)
            
        self.add_constraint(self.x_eval[-2:,-1] - M, -np.inf, 0.)
        # self.add_constraint(self.q_tf - M, -np.inf, 0.)
        self.build_NLP()
        for i in range(self.ntS):
            self.set_stage_control(self.start_point, i, [0.,1/3,1/3])
        self.integrate_full(self.start_point)
    
    def plot(self, xi, dpi = None, title = None, it = None):
        u,w1,w2 = self.get_control_plot_arrays(xi)
        x1, x2, G11, G12, G21, G22, F11, F12, F22, _, _ = self.get_state_arrays_expanded(xi)
        
        fig, ax = plt.subplots(dpi=dpi)
        ax.plot(self.time_grid_ref, x1, 'tab:olive', linestyle='-.', label = r'$x_1$')
        ax.plot(self.time_grid_ref, x2, 'tab:cyan', linestyle='-.', label = r'$x_2$')
        ax.step(self.time_grid_ref, u, 'tab:red', linestyle='-', label = r'$u$')
        ax.step(self.time_grid_ref, w1, 'tab:blue', linestyle=':', label = r'$w_1$')
        ax.step(self.time_grid_ref, w2, 'tab:green', linestyle='--', label = r'$w_2$')
        
        ax.set_ylim(0.,4.)
        ax.legend(fontsize = 'large', loc = 'upper left')
        
        self.finish_plot(ax, title, it, 'Lotka OED problem')



class Lotka_Shared_OED_noQuads(OCProblems.Lotka_Shared_OED):
    
    def build_problem(self):
        self.set_OCP_data(3+9+6 + 3,0,4,3 - 3, [0.,0.,0.] + [-np.inf]*15 + [-np.inf]*3, [np.inf, np.inf, np.inf] + [np.inf]*15 + [np.inf]*3,[],[],[0.]*4,[1.]*4)
        
        alpha0, alpha1, alpha2, c1, c2, t0, tf, x_init, M1, M2, M3, reg_init = (self.model_params[key] for key in ['alpha0', 'alpha1', 'alpha2', 'c1', 'c2', 't0', 'tf', 'x_init', 'M1', 'M2', 'M3', 'reg_init'])
        self.fix_time_horizon(self.model_params['t0'], self.model_params['tf'])
        self.fix_initial_value(self.model_params['x_init'] + [0.]*15 + [0.]*3)
        self.mark_state_bounds_implicit()
        
        x = cs.MX.sym('x', 3)
        u = cs.MX.sym('u', 1)
        x0, x1, x2 = cs.vertsplit(x)
        theta = cs.MX.sym('theta', 3)
        alpha0_s, alpha1_s, alpha2_s = cs.vertsplit(theta)
        
        f_expr = cs.vertcat( x0 - alpha0_s * x0 * x1 - x0 * x2,
                            -x1 + alpha1_s * x0 * x1 - c1 * x1 * u, 
                            -x2 + alpha2_s * x0 * x2 - c2 * x2 * u
                            )
        
        f_x_expr = cs.jacobian(f_expr, x)
        f_theta_expr = cs.jacobian(f_expr, theta)
        
        f = cs.Function('f', [x, u, theta], [f_expr])
        f_x = cs.Function('f_x', [x, u, theta], [f_x_expr])
        f_theta = cs.Function('f_p', [x,u,theta], [f_theta_expr])
        
        f_expr = f(x, u, cs.DM([alpha0, alpha1, alpha2]))
        f_x_expr = f_x(x, u, cs.DM([alpha0, alpha1, alpha2]))
        f_theta_expr = f_theta(x, u, cs.DM([alpha0, alpha1, alpha2]))
        
        
        G = cs.MX.sym('G', x.numel(), theta.numel())
        dG = f_x_expr@G + f_theta_expr
        G_rhs = cs.vec(dG)
        
        w = cs.MX.sym('w', 3)
        w1,w2,w3 = cs.vertsplit(w)
        
        F = cs.MX.sym('F', (theta.numel()*(theta.numel() + 1))//2)
        dh1, dh2, dh3 = cs.DM([1,0,0]), cs.DM([0,1,0]), cs.DM([0,0,1])
        dF = w1*(dh1.T@G).T @ (dh1.T@G) + w2*(dh2.T@G).T @ (dh2.T@G) + w3*(dh3.T@G).T @ (dh3.T@G)
        
        
        F_rhs = cs.vertcat(dF[0,0], dF[1,0], dF[2,0], dF[1,1], dF[2,1], dF[2,2])
        ode_rhs = cs.vertcat(f_expr, G_rhs, F_rhs)
        
        qstates = cs.MX.sym('qstates', 3)
        quad_expr = w
        dt = cs.MX.sym('dt', 1)
        self.ODE = {'x': cs.vertcat(x, cs.vec(G), F, qstates), 'p':cs.vertcat(dt, u, w),'ode': dt*cs.vertcat(ode_rhs, quad_expr)}
        self.multiple_shooting()
        
        F_rhs_tf = self.x_eval[3+9:3+9+6,-1]
        F_tf = cs.MX.zeros(3,3)
        for j in range(3):
            for i in range(0, j):
                F_tf[i,j] = F_rhs_tf[j + i*3 - (i*(i+1))//2]
            F_tf[j,j] = F_rhs_tf[j*4 - (j*(j+1))//2] + reg_init
            for i in range(j + 1, 3):
                F_tf[i,j] = F_rhs_tf[i + j*3 - (j*(j+1))//2]
        
        self.set_objective(cs.trace(cs.inv(F_tf))/theta.numel())
        
        q_tf = self.x_eval[3+9+6:3+9+6+3,-1]
        self.add_constraint(q_tf, [0.,0.,0.], [M1,M2,M3])
        self.build_NLP()
        
        L_t = tf - t0
        for i in range(self.ntS):
            self.set_stage_control(self.start_point, i, [0, M1/L_t, M2/L_t, M3/L_t])
        self.integrate_full(self.start_point)
        
        #Hack: Prevent bad local optimum
        for i in range(self.ntS // 40):
            self.set_stage_control(self.ub_var, i, [0.] + [1.]*3)
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x0, x1, x2 = self.get_state_arrays_expanded(xi)[0:3]
        u, w1, w2, w3 = self.get_control_plot_arrays(xi)
        
        fig, ax = plt.subplots(dpi = dpi)
        ax.plot(self.time_grid_ref, x0, 'tab:green', linestyle = '-.', label = '$x_0$')
        ax.plot(self.time_grid_ref, x1, 'tab:blue', linestyle = '--', label = '$x_1$')
        ax.plot(self.time_grid_ref, x2, 'tab:olive', linestyle = ':', label = '$x_2$')
        
        ax.step(self.time_grid_ref, u, 'tab:red', label = r'$u$')
        ax.step(self.time_grid_ref, w1, 'tab:grey', linestyle = '--', label = r'$w_1$')
        ax.step(self.time_grid_ref, w2, 'tab:blue', linestyle = ':', label = r'$w_2$')
        ax.step(self.time_grid_ref, w3, 'tab:cyan', linestyle = '-.', label = r'$w_3$')
        ax.legend(fontsize='x-large')
        
        self.finish_plot(ax, title, it, "Lotka shared OED")
        

class Particle_Steering_noParams(OCProblems.Particle_Steering):
    def build_problem(self):
        self.set_OCP_data(4+1, 0, 1, 0, [-np.inf]*4 + [0.01/self.ntS], [np.inf]*4 + [100/self.ntS], [], [], [-np.pi/2], [np.pi/2])
        self.fix_initial_value([0.,0.,0.,0.] + [None])
        self.mark_state_bounds_implicit()
        self.fix_time_horizon(0, 1)
        
        a = self.model_params['a']
        x = cs.MX.sym('x',4+1)
        x1,x2,dx1,dx2,dt = cs.vertsplit(x)
        u = cs.MX.sym('u')
        dt_dummy = cs.MX.sym('dt_dummy')
        
        ode_rhs = cs.vertcat(dx1,
                             dx2,
                             a*cs.cos(u),
                             a*cs.sin(u),
                             0
                             )
        
        self.ODE = {'x': x, 'p': cs.vertcat(dt_dummy,u), 'ode': dt*ode_rhs}
        self.multiple_shooting()
        self.set_objective(self.x_eval[4,-1]*self.ntS)
        self.add_constraint(self.x_eval[1,-1] - 5, 0., 0.)
        self.add_constraint(self.x_eval[2:4,-1] - cs.DM([45,0]), 0.,0.)
        self.build_NLP()
        
        for i in range(self.ntS+1):
            self.set_stage_state(self.start_point, i, self.x_init[0:4] + [1/self.ntS])
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x1,x2,dx1,dx2,dt_arr = self.get_state_arrays_expanded(xi)
        u = self.get_control_plot_arrays(xi)
        time_grid = np.cumsum(dt_arr).reshape(-1)
        
        fig, ax = plt.subplots(dpi = dpi)
        ax.plot(time_grid, x1, 'tab:green', linestyle = '--', label = r'$x_1$')
        ax.plot(time_grid, x2, 'tab:blue', linestyle = '-.', label = r'$x_2$')
        # plt.plot(time_grid, y1, 'tab:green', linestyle = '-.', label = r'$v_1$')
        # plt.plot(time_grid, y2, 'tab:blue', linestyle = '-.', label = r'$v_2$')
        ax.step(time_grid, u*10, 'tab:red', label = r'$u\cdot 10$')
        ax.legend(fontsize='large')
        
        self.finish_plot(ax, title, it, 'Particle steering problem')


class Satellite_Deorbiting_noParams(OCProblems.Satellite_Deorbiting):
    def build_problem(self):
        mu, RE, rho0, H, CD, A, Isp, g0, umax, m0, mdry, omegaE, h0, hreentry, rscale, thetascale, mscale, vrscale, vthetascale, TSCALE = (self.model_params[key] for key in ['mu', 'RE', 'rho0', 'H', 'CD', 'A', 'Isp', 'g0', 'umax', 'm0', 'mdry', 'omegaE', 'h0', 'hreentry', 'rscale', 'thetascale', 'mscale', 'vrscale', 'vthetascale', 'TSCALE'])
        
        r0 = RE + h0
        theta0 = 0.
        vr0 = 0.
        vorb = np.sqrt(mu/r0)
        
        rfinal = RE + hreentry
        
        self.set_OCP_data(5+1,0,2,0, [(RE+5000. - RE)*rscale, -2*np.pi*thetascale, -10000.*vrscale, 0.*vthetascale, (mdry - 0.1)*mscale, 300/self.ntS * TSCALE], [(r0 + 100000. - RE)*rscale, 2*np.pi*thetascale, 10000.*vrscale, 20000*vthetascale, (m0 + 0.1)*mscale, 21600/self.ntS * TSCALE], [], [], [-umax, -umax], [umax, umax])
        self.fix_initial_value([(r0 - RE)*rscale, theta0*thetascale, vr0*vrscale, vorb*vthetascale, m0*mscale, None])
        
        self.fix_time_horizon(0., 1.)
        
        def safe_sqrt(x):
            return cs.sqrt(cs.fmax(x, 1e-12))
        
        # Atmospheric model
        def atmospheric_density(r_val):
            h = r_val - RE
            h_safe = cs.fmax(h, -100000)
            return rho0 * cs.exp(-h_safe / H)
        
        X = cs.MX.sym('X', 5+1)
        r_, theta_, vr_, vtheta_, m_,dt_ = cs.vertsplit(X)
        r = r_/rscale + RE
        theta = theta_/thetascale
        vr = vr_/vrscale
        vtheta = vtheta_/vthetascale
        m = m_/mscale
        dt = dt_/TSCALE
        
        U = cs.MX.sym('U', 2)
        ur, utheta = cs.vertsplit(U)
        dt_dummy = cs.MX.sym('dt_dummy', 1)
        
        rsafe = cs.fmax(r, RE + 10000)
        msafe = cs.fmax(m, mdry)
        
        hsafe = cs.fmax(rsafe - RE, -100000)
        rho = rho0 * cs.exp(-hsafe/H)

        vrelr = vr
        vreltheta = vtheta - omegaE*rsafe
        vrel = safe_sqrt(vrelr**2 + vreltheta**2)
        
        centrifugal = vtheta**2/rsafe
        gravity = mu/(rsafe**2)
        drag = 0.5*CD*A/msafe*rho*vrel
        rthrust = ur/msafe
        thetathrust = utheta/msafe
        
        ode_rhs = cs.vertcat(
            vr * rscale,
            vtheta/rsafe * thetascale,
            (centrifugal - gravity + rthrust - drag*vrelr) * vrscale,
            (-vr*vtheta/rsafe + thetathrust - drag*vreltheta) * vthetascale,
            (-cs.sqrt(ur**2 + utheta**2)/(Isp*g0)) * mscale,
            0.
        )
        
        self.ODE = {'x':X, 'p':cs.vertcat(dt_dummy,U), 'ode': dt*ode_rhs}
        self.multiple_shooting()
        self.set_objective(self.ntS*self.x_eval[5,-1]/TSCALE)
        
        rT = self.x_eval[0,-1]
        urt, uthetat = cs.vertsplit(self.u_eval)
        
        self.add_constraint(safe_sqrt(urt**2 + uthetat**2), 0., umax)
        self.add_constraint(rT/rscale, 0., rfinal - RE)
        
        self.build_NLP()
        
        for j in range(self.ntS):
            # self.set_stage_param(self.start_point, j, 1800/self.ntS * TSCALE)
            self.set_stage_control(self.start_point, j, [-5.0,-10.0])
        
        r_init = (np.linspace(r0, rfinal, self.ntS + 1) - RE) * rscale
        theta_init = np.linspace(0, 2*np.pi, self.ntS + 1) * thetascale
        vr_init = np.zeros(self.ntS + 1) * rscale
        vtheta_init = np.ones(self.ntS + 1)*vorb*0.9 * rscale
        m_init = np.linspace(m0, mdry + 10, self.ntS + 1) * mscale
        for j in range(self.ntS + 1):
            self.set_stage_state(self.start_point, j, [r_init[j], theta_init[j], vr_init[j], vtheta_init[j], m_init[j], 1800/self.ntS * TSCALE])
    
    def plot(self, xi, dpi = None, title = None, it = None):
        RE, rscale, thetascale, mscale, TSCALE, vrscale, vthetascale, mu = [self.model_params[key] for key in ['RE', 'rscale', 'thetascale', 'mscale', 'TSCALE', 'vrscale', 'vthetascale', 'mu']]
        
        h0 = 450000
        r0 = RE + h0
        vorb = np.sqrt(mu/r0)
        
        r_, theta_, vr_, vtheta_, m_, dt_arr = self.get_state_arrays_expanded(xi)
        r = r_/rscale + RE
        theta = theta_/thetascale
        vr = vr_/vrscale
        vtheta = vtheta_/vthetascale
        m = m_/mscale
        
        ur, utheta = self.get_control_plot_arrays(xi)
        time_grid = np.cumsum(dt_arr).reshape(-1)/TSCALE
        
        fix, ax = plt.subplots(dpi=dpi)
        ax.plot(time_grid, (r - RE)/1000, 'tab:cyan', linestyle = '--', label = r'(r - RE)/1000')
        ax.plot(time_grid, theta*50, 'tab:green', linestyle = ':', label = r'$\theta\cdot 50$')
        ax.plot(time_grid, vr, 'tab:blue', linestyle = '--', label = r'$v_r$')
        ax.plot(time_grid, vtheta - vorb, 'tab:olive', linestyle = '-.', label = r'$v_\theta - v_\theta(0)$')
        ax.plot(time_grid, (m - 100.)*4, 'tab:blue', linestyle = ':', label = r'(m - 100)$\cdot 4$')
        
        ax.step(time_grid, ur*20, 'tab:red', label = r'$u_r \cdot 20$')
        ax.step(time_grid, utheta*20, 'tab:green', label = r'$u_\theta \cdot 20$')
        
        ax.legend(fontsize='large')
        
        self.finish_plot(ax, title, it, "Satellite Deorbiting problem")


class Time_Optimal_Car_noParams(OCProblems.Time_Optimal_Car):    
    def build_problem(self):
        self.set_OCP_data(2+1,1-1,1,0,[0.,0.] + [0.1/self.ntS],[330.,self.model_params['vmax']] + [500/self.ntS],[], [], [-2.], [1.])
        self.fix_initial_value([0.,0.] + [None])
        self.fix_time_horizon(0, 1)
        
        x = cs.MX.sym('x',2+1)
        z1,z2, dt = cs.vertsplit(x)
        u = cs.MX.sym('u')
        dt_dummy = cs.MX.sym('dt_dummy')
        
        ode_rhs = cs.vertcat(z2,u,0)
        self.ODE = {'x':x, 'p':cs.vertcat(dt_dummy,u), 'ode':dt*ode_rhs}
        self.multiple_shooting()
        self.set_objective(self.x_eval[2,-1]*self.ntS)
        self.add_constraint(self.x_eval[:2,-1] - cs.DM([300,0]),0.,0.)
        self.build_NLP()
        for i in range(self.ntS+1):
            self.set_stage_state(self.start_point, i, self.x_init[:2] + [10/self.ntS])
    
    def plot(self, xi, dpi = None, title = None, it = None):
        z1,z2,dt_arr = self.get_state_arrays_expanded(xi)
        u = self.get_control_plot_arrays(xi)
        time_grid = np.cumsum(dt_arr).reshape(-1)
        
        fig, ax = plt.subplots(dpi = dpi)
        ax.plot(time_grid, z1, 'tab:blue', linestyle = '--', label = r'$z_1$')
        ax.plot(time_grid, z2*5, 'tab:green', linestyle = '-.', label = r'$z_2\cdot5$')
        ax.step(time_grid, u*20, 'tab:red', label = r'$u\cdot20$')
        ax.legend(fontsize='large')
        
        self.finish_plot(ax, title, it, 'Time optimal car problem')

#Note: Dont use refinement > 1 for variable time problems with duration encoded as state, messes up initial duration.
constr_data = {
            Apollo_Reentry_noParams: (0, 3, False, True),
            Batch_Distillation_noParams: (0, 1, False, True),    
            OCProblems.Batch_Reactor: (0, 0, False, True),
            Batch_Reactor_OED_noQuads: (0, 2, False, True),
            Calcium_Oscillation_noParams: (1, 0, True, False),   
            OCProblems.Cart_Pendulum: (0, 0, False, True),
            OCProblems.Catalyst_Mixing: (0, 0, False, True),
            Catalyst_Mixing_OED_noQuads: (0, 2, False, True),
            Cushioned_Oscillation_noParams: (0, 2, False, True),
            Dielectrophoretic_Particle_noParams: (0, 1, False, True),
            D_Onofrio_Chemotherapy_noQuads: (0, 2, False, True),
            Ducted_Fan_noParams: (0, 6, False, True),
            OCProblems.Egerstedt_Standard: (1, 0, True, False),
            OCProblems.Electric_Car: (0, 1, False, True),
            OCProblems.Fermenter: (0, 0, False, True),
            Goddard_Rocket_noParams: (1, 1, False, True),
            Hang_Glider_noParams: (0, 3, False, True),
            Hanging_Chain_noQuads: (0, 2, False, True),
            OCProblems.Lotka_Volterra_Fishing: (0, 0, False, True),
            Lotka_OED_noQuads: (0, 2, False, True),
            OCProblems.Lotka_Volterra_Competitive:  (0, 0, False, True),
            OCProblems.Lotka_Volterra_Shared: (0, 0, False, True),
            Lotka_Shared_OED_noQuads: (0, 3, False, True),
            OCProblems.Ocean: (0, 0, False, True),
            Particle_Steering_noParams: (0, 3, False, True),
            OCProblems.Quadrotor_Helicopter: (1, 0, True, False),
            Satellite_Deorbiting_noParams: (1, 1, True, False),
            OCProblems.Three_Tank_Multimode: (1, 0, True, False),
            Time_Optimal_Car_noParams: (0, 2, True, False),
            OCProblems.Tubular_Reactor: (0, 0, False, True),
}

def get_constr_data(OCprob : OCProblems.OCProblem):
    try:
        return constr_data[type(OCprob)]
    except KeyError as E:
        raise Exception(f"No constraint data available for {type(OCprob).__name__}") from E