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
                F_tf[i,j] = F_rhs_tf[i + j*4 - (j*(j+1))//2]
            F_tf[j,j] = F_rhs_tf[j*5 - (j*(j+1))//2] + reg_init
            for i in range(j + 1, 4):
                F_tf[i,j] = F_rhs_tf[j + i*4 - (i*(i+1))//2]
        
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
                F_tf[i,j] = F_rhs_tf[i + j*2 - (j*(j+1))//2]
            F_tf[j,j] = F_rhs_tf[j*3 - (j*(j+1))//2] + reg_init
            for i in range(j + 1, 2):
                F_tf[i,j] = F_rhs_tf[j + i*2 - (i*(i+1))//2]
        
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
            # (OCProblems.Ducted_Fan, dict(), None),
            # (OCProblems.Egerstedt_Standard, dict(), None),
            OCProblems.Electric_Car: (0, 1, False, True),
            # (OCProblems.Fermenter, dict(), None),
            Goddard_Rocket_noParams: (1, 1, False, True),
            # (OCProblems.Hang_Glider, dict(), None),
            # (OCProblems.Hanging_Chain, dict(), None),
            OCProblems.Lotka_Volterra_Fishing: (0, 0, False, True),
            # (OCProblems.Lotka_OED, dict(), None),
            OCProblems.Lotka_Volterra_Competitive:  (0, 0, False, True),
            OCProblems.Lotka_Volterra_Shared: (0, 0, False, True),
            # (OCProblems.Lotka_Shared_OED, dict(), None),
            OCProblems.Ocean: (0,0, False, True),
            # (OCProblems.Particle_Steering, dict(), None),
            OCProblems.Quadrotor_Helicopter: (1, 0, True, False),
            OCProblems.Satellite_Deorbiting: (1, 1, True, False),
            OCProblems.Three_Tank_Multimode: (1, 0, True, False),
            # (OCProblems.Time_Optimal_Car, dict(), None),
            OCProblems.Tubular_Reactor: (0, 0, False, True),
}

def get_constr_data(OCprob : OCProblems.OCProblem):
    try:
        return constr_data[type(OCprob)]
    except KeyError as E:
        raise Exception(f"No constraint data available for {type(OCprob).__name__}") from E