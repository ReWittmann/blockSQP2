import numpy as np
import casadi as cs
import typing
import matplotlib.pyplot as plt
import math


class OCProblem:
    
    nVar : int
    nCon : int
    time_grid: np.array
    
    #########################################################
    #Fields that a subclass implementing a problem should set
    #########################################################
    
    #NLP dict as required for casadi NLP solvers
    NLP : {str : cs.MX}
    
    #Objective
    f : typing.Callable[[np.ndarray[np.float64]], float]
    #Constraint function
    g : typing.Callable[[np.ndarray[np.float64]], np.ndarray[np.float64]]
    #Objective gradient
    grad_f : typing.Callable[[np.ndarray[np.float64]], np.ndarray[np.float64]]
    #Constraint jacobian (either sparse or dense should be implemented)
    ##SPARSE: number of nonzeros, nonzeros, row indices, column starts (CCS format)
    jac_g_nnz : int
    jac_g_nz : typing.Callable[[np.ndarray[np.float64]], np.ndarray[np.float64]]
    jac_g_row : np.ndarray[np.int32]
    jac_g_colind: np.ndarray[np.int32]
    ##DENSE
    jac_g : typing.Callable[[np.ndarray[np.float64]], np.ndarray[2, np.float64]]
    
    #Bounds
    lb_var : np.ndarray[np.float64]
    ub_var : np.ndarray[np.float64]
    lb_con : np.ndarray[np.float64]
    ub_con : np.ndarray[np.float64]
    
    #Starting point for optimization
    start_point : np.ndarray[np.float64]
    
    
    ##Structure and variable type information (integrality, dependency, hessian blocks, ...)##
    hessBlock_sizes : list[int]
    hessBlock_index : np.ndarray[np.int32]
    vBlock_sizes : np.ndarray[np.int32]
    vBlock_dependencies : np.ndarray[np.dtype('bool')]
    cBlock_sizes : np.ndarray[np.int32]
    ctarget_data : list
    
    
    model_params : dict
    time_grid : np.ndarray[np.int32]
    
    def to_blocks_LT(self, sparse_hess : cs.DM):
        blocks = []
        for j in range(len(self.hessBlock_sizes)):
           blocks.append(np.array(cs.tril(sparse_hess[self.hessBlock_index[j]:self.hessBlock_index[j+1], self.hessBlock_index[j]:self.hessBlock_index[j+1]].full()).nz[:], dtype = np.float64).reshape(-1))
        return blocks
    
    
    
        
class Lotka_Volterra_Fishing(OCProblem):
    def __init__(self, Tgrid = np.linspace(0.0, 12.0, 101, endpoint = True), c0 = 0.4, c1 = 0.2, x_init = [0.5,0.7]):
        self.time_grid = Tgrid
        self.x_init = x_init
        x = cs.MX.sym('x', 2)
        x0,x1 = cs.vertsplit(x)
        w = cs.MX.sym('w', 1)
        dt = cs.MX.sym('dt', 1)
        nt = len(self.time_grid) - 1
        
        ode_rhs = cs.vertcat(x0 - x0*x1 - c0*x0*w, -x1 + x0*x1 - c1*x1*w)
        quad_expr = (x0 - 1)**2 + (x1 - 1)**2
        
        ode = {'x': x, 'p':cs.vertcat(dt, w),'ode': dt*ode_rhs, 'quad': dt*quad_expr}
        
        integrator_opts = {'linear_solver': 'csparse', 'augmented_options' : {'linear_solver' : 'csparse'}}
        odesol_single = cs.integrator('odesol_single', 'cvodes', ode, 0.0, 1.0, integrator_opts)
        
        x_arr = []
        for i in range(1, nt):
            x_arr.append(cs.MX.sym(f'x_s_{i}', 2, 1))
        x_stages = cs.horzcat(*x_arr)
        x_s = cs.horzcat(x_init, x_stages)
        
        w_arr = []
        for i in range(nt):
            w_arr.append(cs.MX.sym(f'w_{i}'))
        w = cs.horzcat(*w_arr)
        
        odesol = odesol_single.map(nt, 'thread', 4)
        
        out = odesol(x0 = x_s, p = cs.vertcat(cs.diff(cs.DM(self.time_grid).T, 1, 1), w))
        F_xf = out['xf']
        F_q = out['qf']
        obj_expr = cs.sum2(F_q)
        cont_cond_expr = x_stages - F_xf[:,:-1]
        
        xopt_arr = [w_arr[0]]
        _hessBlock_sizes = [1]
        lbv_arr = [cs.DM([0])]
        ubv_arr = [cs.DM([1])]
        x_start_arr = [cs.DM([0])]
        _vBlock_sizes = [1]
        _vBlock_dependencies = [False]
        
        for i in range(1, nt):
            xopt_arr.append(x_arr[i-1])
            xopt_arr.append(w_arr[i])
            _hessBlock_sizes += [3]
            _vBlock_sizes += [2,1]
            _vBlock_dependencies += [True, False]
            lbv_arr.append(cs.DM([0,0,0]))
            ubv_arr.append(cs.DM([cs.inf, cs.inf, 1]))
            x_start_arr.append(cs.vertcat(x_init, cs.DM(0)))
        
        self.hessBlock_sizes = np.array(_hessBlock_sizes, dtype = np.int32)
        self.hessBlock_index = np.cumsum([0] + _hessBlock_sizes, dtype = np.int32)
        self.vBlock_sizes = np.array(_vBlock_sizes, dtype = np.int32)
        self.vBlock_dependencies = np.array(_vBlock_dependencies, dtype = np.dtype('bool'))
        self.cBlock_sizes = np.array([2]*(nt - 1), dtype = np.int32)
        xopt = cs.vertcat(*xopt_arr)
        self.lb_var = np.array(cs.vertcat(*lbv_arr), dtype = np.float64).reshape(-1)
        self.ub_var = np.array(cs.vertcat(*ubv_arr), dtype = np.float64).reshape(-1)
        self.lb_con = np.array(cs.DM.zeros(cont_cond_expr.numel(), 1), dtype = np.float64).reshape(-1)
        self.ub_con = np.array(cs.DM.zeros(cont_cond_expr.numel(), 1), dtype = np.float64).reshape(-1)
        self.start_point = np.array(cs.vertcat(*x_start_arr), dtype = np.float64).reshape(-1)
        
        self.NLP = {'x' : xopt, 'f' : obj_expr, 'g' : cs.vec(cont_cond_expr)}
        self.nVar = xopt.numel()
        self.nCon = self.NLP['g'].numel()
        
        self._f = cs.Function('cs_f', [xopt], [obj_expr])
        self.f = lambda xi: np.array(self._f(xi), dtype = np.float64).reshape(-1)
        
        grad_f_expr = cs.jacobian(obj_expr, xopt)
        self._grad_f = cs.Function('cs_grad_f', [xopt], [grad_f_expr])
        self.grad_f = lambda xi: np.array(self._grad_f(xi), dtype = np.float64).reshape(-1)
        
        self._g = cs.Function('cs_g', [xopt], [self.NLP['g']])
        self.g = lambda xi: np.array(self._g(xi), dtype = np.float64).reshape(-1)
        jac_g_expr = cs.jacobian(self.NLP['g'], xopt)
        self._jac_g = cs.Function('cs_jac_g', [xopt], [jac_g_expr])
        self.jac_g = lambda xi: np.array(self._jac_g(xi), dtype = np.float64)
        
        self.jac_g_nnz = jac_g_expr.nnz()
        self.jac_g_row = jac_g_expr.row()
        self.jac_g_colind = jac_g_expr.colind()
        self.jac_g_nz = lambda xi: np.array(self._jac_g(xi).nz[:], dtype = np.float64).reshape(-1)
        
        lam = cs.MX.sym('lambda', self.NLP['g'].numel())
        lag_expr = self.NLP['f'] - lam.T @ self.NLP['g']
        grad_lag_expr = cs.jacobian(lag_expr, self.NLP['x'])
        hess_lag_expr = cs.jacobian(grad_lag_expr, self.NLP['x'])
        self._hess_lag = cs.Function('hess_lag', [self.NLP['x'], lam], [hess_lag_expr])
        self.hess_lag = lambda xi, lambd: self.to_blocks_LT(self._hess_lag(xi, lambd))
        
        
        # self.problem_structure = {'hblock_sizes': hessblock_sizes, 'vblock_sizes': vblock_sizes,
        #                           'cblock_sizes': cblock_sizes, 'vblock_dependencies': vblock_dependencies,
        #                           'ctarget_data': [self.model_params['nt'], 0, len(vblock_sizes) + 1, 0, len(cblock_sizes) + 1]
        #                           }
        
        
    def plot(self, xi:np.ndarray[np.float64]):
        x0 = np.concatenate(([self.x_init[0]], xi[1::3]), axis = 0)
        x1 = np.concatenate(([self.x_init[1]], xi[2::3]), axis = 0)
        u = xi[0::3]
        
        
        plt.figure()
        plt.plot(self.time_grid[:-1], x0, 'r-', label = 'x0')#, self.time_grid[:,-1], x1, '--', self.time_grid[:,-1], u, 'o')
        plt.plot(self.time_grid[:-1], x1, 'b--', label = 'x1')
        # plt.plot(self.time_grid[:-1], u, 'g:', label = 'u')
        plt.bar(self.time_grid[:-1], u, color = 'g', label = 'u')
        plt.legend()
        plt.show()
        
        
class Hanging_Chain(OCProblem):
    def __init__(self, Tgrid = np.linspace(0,1,101,endpoint = True), a = 1.0, b = 3.0, Lp = 4.0, integrator = "RK4"):
        x = cs.MX.sym('x', 2)
        x1, x3 = cs.vertsplit(x)
        u = cs.MX.sym('u', 1)
        dt = cs.MX.sym('dt', 1)
        ode = {'x':x, 'p': cs.vertcat(dt,u), 'ode': dt * cs.vertcat(u, (1+u**2)**0.5), 'quad': dt*x1*(1+u**2)**0.5}
        nt = len(Tgrid) - 1
        self.time_grid = Tgrid
        
        if integrator == "cvodes":
            integrator_opts = {'linear_solver': 'csparse', 'augmented_options' : {'linear_solver' : 'csparse'}, 'abstol':1e-6, 'reltol':1e-6}
            odesol_single = cs.integrator('odesol_single', 'cvodes', ode, 0.0, 1.0, integrator_opts)
        elif integrator == "RK4":
            #Fixed step Runge-Kutta 4 integrator
            M = 2 # RK4 steps per interval
            DT = 1/M
            f = cs.Function('f', [x, cs.vertcat(dt,u)], [ode['ode'], ode['quad']])
            X0 = x
            U = cs.vertcat(dt,u)
            X = X0
            Q = 0
            for j in range(M):
                k1, k1_q = f(X, U)
                k2, k2_q = f(X + DT/2 * k1, U)
                k3, k3_q = f(X + DT/2 * k2, U)
                k4, k4_q = f(X + DT * k3, U)
                X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
                Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
            odesol_single = cs.Function('F', [X0, U], [X, Q],['x0','p'],['xf','qf'])
        
        x_init = [a,0]
        self.x_init = x_init
        x_arr = []
        for i in range(1, nt):
            x_arr.append(cs.MX.sym(f'x_s_{i}', 2, 1))
        x_stages = cs.horzcat(*x_arr)
        x_s = cs.horzcat(cs.DM(x_init), x_stages)
        
        u_arr = []
        for i in range(nt):
            u_arr.append(cs.MX.sym(f'u_{i}'))
        u = cs.horzcat(*u_arr)
        
        odesol = odesol_single.map(nt, 'thread', 4)
        
        out = odesol(x0 = x_s, p = cs.vertcat(cs.diff(cs.DM(self.time_grid).T, 1, 1), u))
        F_xf = out['xf']
        F_q = out['qf']
        obj_expr = cs.sum2(F_q)
        cont_cond_expr = x_stages - F_xf[:,:-1]
        t_cond_expr = cs.vertcat(F_xf[0,-1] - b, F_xf[1,-1] - Lp)
        
        xopt_arr = [u_arr[0]]
        _hessBlock_sizes = [1]
        lbv_arr = [cs.DM([-10])]
        ubv_arr = [cs.DM([20])]
        # x_start_arr = [cs.DM([0])]
        _vBlock_sizes = [1]
        _vBlock_dependencies = [False]
        
        for i in range(1, nt):
            xopt_arr.append(x_arr[i-1])
            xopt_arr.append(u_arr[i])
            _hessBlock_sizes += [3]
            _vBlock_sizes += [2,1]
            _vBlock_dependencies += [True, False]
            lbv_arr.append(cs.DM([0,0,-10]))
            ubv_arr.append(cs.DM([10,10,20]))
            # x_start_arr.append(cs.vertcat(x_init, cs.DM(0)))
        
        self.hessBlock_sizes = np.array(_hessBlock_sizes, dtype = np.int32)
        self.hessBlock_index = np.cumsum([0] + _hessBlock_sizes, dtype = np.int32)
        self.vBlock_sizes = np.array(_vBlock_sizes, dtype = np.int32)
        self.vBlock_dependencies = np.array(_vBlock_dependencies, dtype = np.dtype('bool'))
        self.cBlock_sizes = np.array([2]*(nt - 1) + [1] + [1], dtype = np.int32)
        xopt = cs.vertcat(*xopt_arr)
        self.lb_var = np.array(cs.vertcat(*lbv_arr), dtype = np.float64).reshape(-1)
        self.ub_var = np.array(cs.vertcat(*ubv_arr), dtype = np.float64).reshape(-1)
        
        self.lb_con = np.array([0]*cont_cond_expr.numel() + [0] + [0], dtype = np.float64)
        # self.ub_con = np.array(cs.DM.zeros(cont_cond_expr.numel(), 1), dtype = np.float64).reshape(-1)
        self.ub_con = np.array([0]*cont_cond_expr.numel() + [0] + [0], dtype = np.float64)
        
        # self.start_point = np.array(cs.vertcat(*x_start_arr), dtype = np.float64).reshape(-1)
        
        if b > a:
            tm = 0.25
        else:
            tm = 0.75
        x1_start = []
        for j in range(len(self.time_grid)):
            t = self.time_grid[j]
            x1_start.append(2*abs(b - a)*t*(t - 2*tm) + a)
        x1_start = np.array(x1_start)
        u_start = np.diff(x1_start, 1, 0)/np.diff(self.time_grid, 1, 0)
        x2_start = x1_start[1:-1]*u_start[:-1]
        xi_start = np.zeros(xopt.numel())
        xi_start[0::3] = u_start
        xi_start[1::3] = x1_start[1:-1]
        xi_start[2::3] = x2_start
        self.start_point = xi_start
        
        
        self.NLP = {'x' : xopt, 'f' : obj_expr, 'g' : cs.vertcat(cs.vec(cont_cond_expr), t_cond_expr)}
        self.nVar = xopt.numel()
        self.nCon = self.NLP['g'].numel()
        
        self._f = cs.Function('cs_f', [xopt], [obj_expr])
        self.f = lambda xi: np.array(self._f(xi), dtype = np.float64).reshape(-1)
        
        grad_f_expr = cs.jacobian(obj_expr, xopt)
        self._grad_f = cs.Function('cs_grad_f', [xopt], [grad_f_expr])
        self.grad_f = lambda xi: np.array(self._grad_f(xi), dtype = np.float64).reshape(-1)
        
        self._g = cs.Function('cs_g', [xopt], [self.NLP['g']])
        self.g = lambda xi: np.array(self._g(xi), dtype = np.float64).reshape(-1)
        jac_g_expr = cs.jacobian(self.NLP['g'], xopt)
        self._jac_g = cs.Function('cs_jac_g', [xopt], [jac_g_expr])
        self.jac_g = lambda xi: np.array(self._jac_g(xi), dtype = np.float64)
        
        self.jac_g_nnz = jac_g_expr.nnz()
        # self.jac_g_row = np.array(jac_g_expr.row(), dtype = np.int32).reshape(-1)
        self.jac_g_row = jac_g_expr.row()
        # self.jac_g_colind = np.array(jac_g_expr.colind(), dtype = np.int32).reshape(-1)
        self.jac_g_colind = jac_g_expr.colind()
        self.jac_g_nz = lambda xi: np.array(self._jac_g(xi).nz[:], dtype = np.float64).reshape(-1)
        
        lam = cs.MX.sym('lambda', self.NLP['g'].numel())
        lag_expr = self.NLP['f'] - lam.T @ self.NLP['g']
        grad_lag_expr = cs.jacobian(lag_expr, self.NLP['x'])
        hess_lag_expr = cs.jacobian(grad_lag_expr, self.NLP['x'])
        self._hess_lag = cs.Function('hess_lag', [self.NLP['x'], lam], [hess_lag_expr])
        self.hess_lag = lambda xi, lambd: self.to_blocks_LT(self._hess_lag(xi, lambd))        
        
        
    def plot(self, xi:np.ndarray[np.float64]):
        x1 = np.concatenate(([self.x_init[0]], xi[1::3]), axis = 0)
        
        
        plt.figure()
        plt.plot(self.time_grid[:-1], x1, 'r-', label = 'x1')#, self.time_grid[:,-1], x1, '--', self.time_grid[:,-1], u, 'o')
        # plt.plot(self.time_grid[:-1], x1, 'b--', label = 'x1')
        # # plt.plot(self.time_grid[:-1], u, 'g:', label = 'u')
        # plt.bar(self.time_grid[:-1], u, color = 'g', label = 'u')
        plt.legend()
        plt.show()

        
class Lotka_Shared_Resources(OCProblem):
    def __init__(self, Tgrid = np.linspace(0.0, 40.0, 101, endpoint = True), c1 = 0.1, c2 = 0.4, alpha = 1.2, x_init = [1.5, 0.5, 1.0]):
        
        self.time_grid = Tgrid
        self.x_init = x_init
        x = cs.MX.sym('x', 3)
        x0,x1,x2 = cs.vertsplit(x)
        u = cs.MX.sym('u', 1)
        dt = cs.MX.sym('dt', 1)
        nt = len(self.time_grid) - 1
        
        ode_rhs = cs.vertcat(x0 - x0*x1 - x0*x2, -x1 + x0*x1 - c1*x1*u, -x2 + alpha*x0*x2 - c2*x2*u)
        quad_expr = (x0 - 1.5)**2 + (x1 - 1)**2 + (x2 - 1)**2
        
        ode = {'x': x, 'p':cs.vertcat(dt, u),'ode': dt*ode_rhs, 'quad': dt*quad_expr}
        
        integrator_opts = {'linear_solver': 'csparse', 'augmented_options' : {'linear_solver' : 'csparse'}}
        # integrator_opts = {}
        odesol_single = cs.integrator('odesol_single', 'cvodes', ode, 0.0, 1.0, integrator_opts)
        
        
        # # Fixed step Runge-Kutta 4 integrator
        # M = 2 # RK4 steps per interval
        # DT = 1/M
        # f = cs.Function('f', [x, cs.vertcat(dt,u)], [ode['ode'], ode['quad']])
        # X0 = x
        # U = cs.vertcat(dt,u)
        # X = X0
        # Q = 0
        # for j in range(M):
        #     k1, k1_q = f(X, U)
        #     k2, k2_q = f(X + DT/2 * k1, U)
        #     k3, k3_q = f(X + DT/2 * k2, U)
        #     k4, k4_q = f(X + DT * k3, U)
        #     X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
        #     Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
        # odesol_single = cs.Function('F', [X0, U], [X, Q],['x0','p'],['xf','qf'])
        
        
        
        
        x_arr = []
        for i in range(1, nt):
            x_arr.append(cs.MX.sym(f'x_s_{i}', 3, 1))
        x_stages = cs.horzcat(*x_arr)
        x_s = cs.horzcat(x_init, x_stages)
        
        u_arr = []
        for i in range(nt):
            u_arr.append(cs.MX.sym(f'u_{i}'))
        u = cs.horzcat(*u_arr)
        
        odesol = odesol_single.map(nt, 'thread', 4)
        # odesol = odesol_single.map(nt)
        
        out = odesol(x0 = x_s, p = cs.vertcat(cs.diff(cs.DM(self.time_grid).T, 1, 1), u))
        F_xf = out['xf']
        F_q = out['qf']
        obj_expr = cs.sum2(F_q)
        cont_cond_expr = x_stages - F_xf[:,:-1]
        
        xopt_arr = [u_arr[0]]
        _hessBlock_sizes = [1]
        lbv_arr = [cs.DM([0])]
        ubv_arr = [cs.DM([1])]
        x_start_arr = [cs.DM([0])]
        _vBlock_sizes = [1]
        _vBlock_dependencies = [False]
        
        for i in range(1, nt):
            xopt_arr.append(x_arr[i-1])
            xopt_arr.append(u_arr[i])
            _hessBlock_sizes += [4]
            _vBlock_sizes += [3,1]
            _vBlock_dependencies += [True, False]
            lbv_arr.append(cs.DM([0,0,0,0]))
            ubv_arr.append(cs.DM([cs.inf, cs.inf, cs.inf, 1]))
            x_start_arr.append(cs.vertcat(x_init, cs.DM(0)))
        
        self.hessBlock_sizes = np.array(_hessBlock_sizes, dtype = np.int32)
        self.hessBlock_index = np.cumsum([0] + _hessBlock_sizes, dtype = np.int32)
        self.vBlock_sizes = np.array(_vBlock_sizes, dtype = np.int32)
        self.vBlock_dependencies = np.array(_vBlock_dependencies, dtype = np.dtype('bool'))
        self.cBlock_sizes = np.array([3]*(nt - 1), dtype = np.int32)
        xopt = cs.vertcat(*xopt_arr)
        self.lb_var = np.array(cs.vertcat(*lbv_arr), dtype = np.float64).reshape(-1)
        self.ub_var = np.array(cs.vertcat(*ubv_arr), dtype = np.float64).reshape(-1)
        self.lb_con = np.array(cs.DM.zeros(cont_cond_expr.numel(), 1), dtype = np.float64).reshape(-1)
        self.ub_con = np.array(cs.DM.zeros(cont_cond_expr.numel(), 1), dtype = np.float64).reshape(-1)
        self.start_point = np.array(cs.vertcat(*x_start_arr), dtype = np.float64).reshape(-1)
        
        self.NLP = {'x' : xopt, 'f' : obj_expr, 'g' : cs.vec(cont_cond_expr)}
        self.nVar = xopt.numel()
        self.nCon = self.NLP['g'].numel()
        
        self._f = cs.Function('cs_f', [xopt], [obj_expr])
        self.f = lambda xi: np.array(self._f(xi), dtype = np.float64).reshape(-1)
        
        grad_f_expr = cs.jacobian(obj_expr, xopt)
        self._grad_f = cs.Function('cs_grad_f', [xopt], [grad_f_expr])
        self.grad_f = lambda xi: np.array(self._grad_f(xi), dtype = np.float64).reshape(-1)
        
        self._g = cs.Function('cs_g', [xopt], [self.NLP['g']])
        self.g = lambda xi: np.array(self._g(xi), dtype = np.float64).reshape(-1)
        jac_g_expr = cs.jacobian(self.NLP['g'], xopt)
        self._jac_g = cs.Function('cs_jac_g', [xopt], [jac_g_expr])
        self.jac_g = lambda xi: np.array(self._jac_g(xi), dtype = np.float64)
        
        self.jac_g_nnz = jac_g_expr.nnz()
        # self.jac_g_row = np.array(jac_g_expr.row(), dtype = np.int32).reshape(-1)
        self.jac_g_row = jac_g_expr.row()
        # self.jac_g_colind = np.array(jac_g_expr.colind(), dtype = np.int32).reshape(-1)
        self.jac_g_colind = jac_g_expr.colind()
        self.jac_g_nz = lambda xi: np.array(self._jac_g(xi).nz[:], dtype = np.float64).reshape(-1)

        lam = cs.MX.sym('lambda', self.NLP['g'].numel())
        lag_expr = self.NLP['f'] - lam.T @ self.NLP['g']
        grad_lag_expr = cs.jacobian(lag_expr, self.NLP['x'])
        hess_lag_expr = cs.jacobian(grad_lag_expr, self.NLP['x'])
        self._hess_lag = cs.Function('hess_lag', [self.NLP['x'], lam], [hess_lag_expr])
        self.hess_lag = lambda xi, lambd: self.to_blocks_LT(self._hess_lag(xi, lambd))


    def plot(self, xi:np.ndarray[np.float64]):
        x0 = np.concatenate(([self.x_init[0]], xi[1::4]), axis = 0)
        x1 = np.concatenate(([self.x_init[1]], xi[2::4]), axis = 0)
        x2 = np.concatenate(([self.x_init[2]], xi[3::4]), axis = 0)
        u = xi[0::4]
        
        
        plt.figure()
        plt.plot(self.time_grid[:-1], x0, 'r-', label = 'resource')#, self.time_grid[:,-1], x1, '--', self.time_grid[:,-1], u, 'o')
        plt.plot(self.time_grid[:-1], x1, 'b-', label = 'x1')
        plt.plot(self.time_grid[:-1], x2, 'c-', label = 'x2')
        # plt.plot(self.time_grid[:-1], u, 'g:', label = 'u')
        plt.bar(self.time_grid[:-1], u, color = 'g', label = 'u')
        plt.legend()
        plt.show()

class Lotka_Competitive(OCProblem):
    def __init__(self, Tgrid = np.linspace(0.0, 40.0, 201, endpoint = True), c1 = 0.1, c2 = 0.3, alpha = 1.2, K = 1.8, x_init = [0.5, 1.5]):
        
        self.time_grid = Tgrid
        self.x_init = x_init
        x = cs.MX.sym('x', 2)
        x0,x1 = cs.vertsplit(x)
        u = cs.MX.sym('u', 1)
        dt = cs.MX.sym('dt', 1)
        nt = len(self.time_grid) - 1
        
        ode_rhs = cs.vertcat(x0 * (1 - (x0 + alpha*x1)/K) - c1*x0*u, x1*(1 - (x0 + x1)/K) - c2*x1*u)
        quad_expr = (x0 - 1)**2 + (x1 - 1)**2
        
        ode = {'x': x, 'p':cs.vertcat(dt, u),'ode': dt*ode_rhs, 'quad': dt*quad_expr}
        
        integrator_opts = {'linear_solver': 'csparse', 'augmented_options' : {'linear_solver' : 'csparse'}, 'abstol':1e-6, 'reltol':1e-6}
        odesol_single = cs.integrator('odesol_single', 'cvodes', ode, 0.0, 1.0, integrator_opts)
        
        
        
        # M = 2 # RK4 steps per interval
        # DT = 1/M
        # f = cs.Function('f', [x, cs.vertcat(dt,u)], [ode['ode'], ode['quad']])
        # X0 = x
        # U = cs.vertcat(dt,u)
        # X = X0
        # Q = 0
        # for j in range(M):
        #     k1, k1_q = f(X, U)
        #     k2, k2_q = f(X + DT/2 * k1, U)
        #     k3, k3_q = f(X + DT/2 * k2, U)
        #     k4, k4_q = f(X + DT * k3, U)
        #     X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
        #     Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
        # odesol_single = cs.Function('F', [X0, U], [X, Q],['x0','p'],['xf','qf'])
        
        
        x_arr = []
        for i in range(1, nt):
            x_arr.append(cs.MX.sym(f'x_s_{i}', 2, 1))
        x_stages = cs.horzcat(*x_arr)
        x_s = cs.horzcat(x_init, x_stages)
        
        u_arr = []
        for i in range(nt):
            u_arr.append(cs.MX.sym(f'u_{i}'))
        u = cs.horzcat(*u_arr)
        
        odesol = odesol_single.map(nt, 'thread', 4)
        
        out = odesol(x0 = x_s, p = cs.vertcat(cs.diff(cs.DM(self.time_grid).T, 1, 1), u))
        F_xf = out['xf']
        F_q = out['qf']
        obj_expr = cs.sum2(F_q)
        cont_cond_expr = x_stages - F_xf[:,:-1]
        
        xopt_arr = [u_arr[0]]
        _hessBlock_sizes = [1]
        lbv_arr = [cs.DM([0])]
        ubv_arr = [cs.DM([1])]
        x_start_arr = [cs.DM([0])]
        _vBlock_sizes = [1]
        _vBlock_dependencies = [False]
        
        for i in range(1, nt):
            xopt_arr.append(x_arr[i-1])
            xopt_arr.append(u_arr[i])
            _hessBlock_sizes += [3]
            _vBlock_sizes += [2,1]
            _vBlock_dependencies += [True, False]
            lbv_arr.append(cs.DM([0,0,0]))
            ubv_arr.append(cs.DM([2, 2, 1]))
            x_start_arr.append(cs.vertcat(x_init, cs.DM(0)))
        
        self.hessBlock_sizes = np.array(_hessBlock_sizes, dtype = np.int32)
        self.hessBlock_index = np.cumsum([0] + _hessBlock_sizes, dtype = np.int32)
        self.vBlock_sizes = np.array(_vBlock_sizes, dtype = np.int32)
        self.vBlock_dependencies = np.array(_vBlock_dependencies, dtype = np.dtype('bool'))
        self.cBlock_sizes = np.array([2]*(nt - 1), dtype = np.int32)
        xopt = cs.vertcat(*xopt_arr)
        self.lb_var = np.array(cs.vertcat(*lbv_arr), dtype = np.float64).reshape(-1)
        self.ub_var = np.array(cs.vertcat(*ubv_arr), dtype = np.float64).reshape(-1)
        self.lb_con = np.array(cs.DM.zeros(cont_cond_expr.numel(), 1), dtype = np.float64).reshape(-1)
        self.ub_con = np.array(cs.DM.zeros(cont_cond_expr.numel(), 1), dtype = np.float64).reshape(-1)
        self.start_point = np.array(cs.vertcat(*x_start_arr), dtype = np.float64).reshape(-1)
        
        self.NLP = {'x' : xopt, 'f' : obj_expr, 'g' : cs.vec(cont_cond_expr)}
        self.nVar = xopt.numel()
        self.nCon = self.NLP['g'].numel()
        
        self._f = cs.Function('cs_f', [xopt], [obj_expr])
        self.f = lambda xi: np.array(self._f(xi), dtype = np.float64).reshape(-1)
        
        grad_f_expr = cs.jacobian(obj_expr, xopt)
        self._grad_f = cs.Function('cs_grad_f', [xopt], [grad_f_expr])
        self.grad_f = lambda xi: np.array(self._grad_f(xi), dtype = np.float64).reshape(-1)
        
        self._g = cs.Function('cs_g', [xopt], [self.NLP['g']])
        self.g = lambda xi: np.array(self._g(xi), dtype = np.float64).reshape(-1)
        jac_g_expr = cs.jacobian(self.NLP['g'], xopt)
        self._jac_g = cs.Function('cs_jac_g', [xopt], [jac_g_expr])
        self.jac_g = lambda xi: np.array(self._jac_g(xi), dtype = np.float64)
        
        self.jac_g_nnz = jac_g_expr.nnz()
        # self.jac_g_row = np.array(jac_g_expr.row(), dtype = np.int32).reshape(-1)
        self.jac_g_row = jac_g_expr.row()
        # self.jac_g_colind = np.array(jac_g_expr.colind(), dtype = np.int32).reshape(-1)
        self.jac_g_colind = jac_g_expr.colind()
        self.jac_g_nz = lambda xi: np.array(self._jac_g(xi).nz[:], dtype = np.float64).reshape(-1)

        lam = cs.MX.sym('lambda', self.NLP['g'].numel())
        lag_expr = self.NLP['f'] - lam.T @ self.NLP['g']
        grad_lag_expr = cs.jacobian(lag_expr, self.NLP['x'])
        hess_lag_expr = cs.jacobian(grad_lag_expr, self.NLP['x'])
        self._hess_lag = cs.Function('hess_lag', [self.NLP['x'], lam], [hess_lag_expr])
        self.hess_lag = lambda xi, lambd: self.to_blocks_LT(self._hess_lag(xi, lambd))        
    
    
    def plot(self, xi:np.ndarray[np.float64]):
        x0 = np.concatenate(([self.x_init[0]], xi[1::3]), axis = 0)
        x1 = np.concatenate(([self.x_init[1]], xi[2::3]), axis = 0)
        u = xi[0::3]
        
        
        plt.figure()
        plt.plot(self.time_grid[:-1], x0, 'r-', label = 'x0')#, self.time_grid[:,-1], x1, '--', self.time_grid[:,-1], u, 'o')
        plt.plot(self.time_grid[:-1], x1, 'b-', label = 'x1')
        # plt.plot(self.time_grid[:-1], u, 'g:', label = 'u')
        plt.bar(self.time_grid[:-1], u, color = 'g', label = 'u')
        # plt.step(self.time_grid[:-1], u, color = 'g', label = 'u')
        plt.legend()
        plt.show()
    

class Three_Tank_Multimode(OCProblem):
    def __init__(self, Tgrid = np.linspace(0.0, 12.0, 101, endpoint = True), c1 = 1.0, c2 = 2.0, c3 = 0.8, k1 = 2.0, k2 = 3.0, k3 = 1.0, k4 = 3.0, x_init = [2.0, 2.0, 2.0]):
        
        self.time_grid = Tgrid
        self.x_init = x_init
        x = cs.MX.sym('x', 3)
        x1,x2,x3 = cs.vertsplit(x)
        w = cs.MX.sym('w', 3)
        w1, w2, w3 = cs.vertsplit(w)
        
        dt = cs.MX.sym('dt', 1)
        nt = len(self.time_grid) - 1
        
        ode_rhs = cs.vertcat(-cs.sqrt(x1) + c1*w1 + c2*w2 - w3*cs.sqrt(c3*x3),\
                             cs.sqrt(x1) - cs.sqrt(x2),\
                             cs.sqrt(x2) - cs.sqrt(x3) + w3*cs.sqrt(c3*x3))
        quad_expr = k1*(x2 - k2)**2 + k3*(x3 - k4)**2
        
        ode = {'x': x, 'p':cs.vertcat(dt, w),'ode': dt*ode_rhs, 'quad': dt*quad_expr}
        
        integrator_opts = {'linear_solver': 'csparse', 'augmented_options' : {'linear_solver' : 'csparse'}}
        odesol_single = cs.integrator('odesol_single', 'cvodes', ode, 0.0, 1.0, integrator_opts)
        
        x_arr = []
        for i in range(1, nt):
            x_arr.append(cs.MX.sym(f'x_s_{i}', 3, 1))
        x_stages = cs.horzcat(*x_arr)
        x_s = cs.horzcat(x_init, x_stages)
        
        w_arr = []
        for i in range(nt):
            w_arr.append(cs.MX.sym(f'u_{i}', 3, 1))
        w = cs.horzcat(*w_arr)
        
        odesol = odesol_single.map(nt, 'thread', 4)
        
        out = odesol(x0 = x_s, p = cs.vertcat(cs.diff(cs.DM(self.time_grid).T, 1, 1), w))
        F_xf = out['xf']
        F_q = out['qf']
        obj_expr = cs.sum2(F_q)
        cont_cond_expr = x_stages - F_xf[:,:-1]
        constr_expr = cs.sum1(w)
        
        xopt_arr = [w_arr[0]]
        _hessBlock_sizes = [3]
        lbv_arr = [cs.DM([0,0,0])]
        ubv_arr = [cs.DM([1,1,1])]
        x_start_arr = [cs.DM([1/3, 1/3, 1/3])]
        _vBlock_sizes = [3]
        _vBlock_dependencies = [False]
        
        for i in range(1, nt):
            xopt_arr.append(x_arr[i-1])
            xopt_arr.append(w_arr[i])
            _hessBlock_sizes += [6]
            _vBlock_sizes += [3,3]
            _vBlock_dependencies += [True, False]
            lbv_arr.append(cs.DM([0, 0, 0, 0, 0, 0]))
            ubv_arr.append(cs.DM([cs.inf, cs.inf, cs.inf, 1,1,1]))
            x_start_arr.append(cs.vertcat(x_init, cs.DM([1/3, 1/3, 1/3])))
        
        self.hessBlock_sizes = np.array(_hessBlock_sizes, dtype = np.int32)
        self.hessBlock_index = np.cumsum([0] + _hessBlock_sizes, dtype = np.int32)
        self.vBlock_sizes = np.array(_vBlock_sizes, dtype = np.int32)
        self.vBlock_dependencies = np.array(_vBlock_dependencies, dtype = np.dtype('bool'))
        self.cBlock_sizes = np.array([3]*(nt - 1) + [constr_expr.numel()], dtype = np.int32)
        xopt = cs.vertcat(*xopt_arr)
        self.lb_var = np.array(cs.vertcat(*lbv_arr), dtype = np.float64).reshape(-1)
        self.ub_var = np.array(cs.vertcat(*ubv_arr), dtype = np.float64).reshape(-1)
        self.lb_con = np.array([0]*cont_cond_expr.numel() + [1.0]*constr_expr.numel(), dtype = np.float64).reshape(-1)
        self.ub_con = np.array([0]*cont_cond_expr.numel() + [1.0]*constr_expr.numel(), dtype = np.float64).reshape(-1)
        self.start_point = np.array(cs.vertcat(*x_start_arr), dtype = np.float64).reshape(-1)
        
        self.NLP = {'x' : xopt, 'f' : obj_expr, 'g' : cs.vertcat(cs.vec(cont_cond_expr), cs.vec(constr_expr))}
        self.nVar = xopt.numel()
        self.nCon = self.NLP['g'].numel()
        
        self._f = cs.Function('cs_f', [xopt], [obj_expr])
        self.f = lambda xi: np.array(self._f(xi), dtype = np.float64).reshape(-1)
        
        grad_f_expr = cs.jacobian(obj_expr, xopt)
        self._grad_f = cs.Function('cs_grad_f', [xopt], [grad_f_expr])
        self.grad_f = lambda xi: np.array(self._grad_f(xi), dtype = np.float64).reshape(-1)
        
        self._g = cs.Function('cs_g', [xopt], [self.NLP['g']])
        self.g = lambda xi: np.array(self._g(xi), dtype = np.float64).reshape(-1)
        jac_g_expr = cs.jacobian(self.NLP['g'], xopt)
        self._jac_g = cs.Function('cs_jac_g', [xopt], [jac_g_expr])
        self.jac_g = lambda xi: np.array(self._jac_g(xi), dtype = np.float64)
        
        self.jac_g_nnz = jac_g_expr.nnz()
        # self.jac_g_row = np.array(jac_g_expr.row(), dtype = np.int32).reshape(-1)
        self.jac_g_row = jac_g_expr.row()
        # self.jac_g_colind = np.array(jac_g_expr.colind(), dtype = np.int32).reshape(-1)
        self.jac_g_colind = jac_g_expr.colind()
        self.jac_g_nz = lambda xi: np.array(self._jac_g(xi).nz[:], dtype = np.float64).reshape(-1)

        lam = cs.MX.sym('lambda', self.NLP['g'].numel())
        lag_expr = self.NLP['f'] - lam.T @ self.NLP['g']
        grad_lag_expr = cs.jacobian(lag_expr, self.NLP['x'])
        hess_lag_expr = cs.jacobian(grad_lag_expr, self.NLP['x'])
        self._hess_lag = cs.Function('hess_lag', [self.NLP['x'], lam], [hess_lag_expr])
        self.hess_lag = lambda xi, lambd: self.to_blocks_LT(self._hess_lag(xi, lambd))
        
        
    def plot(self, xi:np.ndarray[np.float64]):
        x1 = np.concatenate(([self.x_init[0]], xi[3::6]), axis = 0)
        x2 = np.concatenate(([self.x_init[1]], xi[4::6]), axis = 0)
        x3 = np.concatenate(([self.x_init[2]], xi[5::6]), axis = 0)
        w1 = xi[0::6]
        w2 = xi[1::6]
        w3 = xi[2::6]
        
        plt.figure()
        plt.plot(self.time_grid[:-1], x1, 'r-', label = 'x1')#, self.time_grid[:,-1], x1, '--', self.time_grid[:,-1], u, 'o')
        plt.plot(self.time_grid[:-1], x2, 'b-', label = 'x2')
        plt.plot(self.time_grid[:-1], x3, 'c-', label = 'x3')
        # plt.plot(self.time_grid[:-1], u, 'g:', label = 'u')
        # plt.bar(self.time_grid[:-1], u, color = 'g', label = 'u')
        plt.step(self.time_grid[:-1], w1, color = 'g', label = 'u')
        plt.step(self.time_grid[:-1], w2, color = 'r', label = 'u')
        plt.step(self.time_grid[:-1], w3, color = 'b', label = 'u')
        plt.legend()
        plt.show()
    

class Goddart_Rocket(OCProblem):
    def __init__(self, nt = 100, rT = 1.01, b = 7.0, Tmax = 3.5, A = 310.0, k = 500.0, C = 0.6, x_init = [1.0,0.0,1.0], integrator = "cvodes"):
        
        self.x_init = x_init
        x = cs.MX.sym('x', 3)
        r, v, m = cs.vertsplit(x)
        r0, v0, m0 = x_init
        u = cs.MX.sym('u', 1)
        dt = cs.MX.sym('dt', 1)
        
        
        ode_rhs = cs.vertcat(v,\
                            -1/(r**2) + (1/m) * (Tmax*u - A*(v**2) * cs.exp(-k * (r - r0))),\
                            -b*u) #-b*Tmax*u)
        
        ode = {'x': x, 'p':cs.vertcat(dt, u),'ode': dt*ode_rhs}
        
        if integrator == "cvodes":
            integrator_opts = {'linear_solver': 'csparse', 'augmented_options' : {'linear_solver' : 'csparse'}, 'abstol':1e-8, 'reltol':1e-8}
            odesol_single = cs.integrator('odesol_single', 'cvodes', ode, 0.0, 1.0, integrator_opts)
        elif integrator == "RK4":
            # Fixed step Runge-Kutta 4 integrator
            M = 4 # RK4 steps per interval
            DT = 1/M
            f = cs.Function('f', [x, cs.vertcat(dt,u)], [ode['ode']])
            X0 = x
            U = cs.vertcat(dt,u)
            X = X0
            for j in range(M):
                k1 = f(X, U)
                k2 = f(X + DT/2 * k1, U)
                k3 = f(X + DT/2 * k2, U)
                k4 = f(X + DT * k3, U)
                X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
            odesol_single = cs.Function('F', [X0, U], [X],['x0','p'], ['xf'])
        
        
        
        
        x_arr = []
        for i in range(1, nt):
            x_arr.append(cs.MX.sym(f'x_s_{i}', 3, 1))
        x_stages = cs.horzcat(*x_arr)
        x_s = cs.horzcat(x_init, x_stages)
        
        u_arr = []
        dt_arr = []
        for i in range(nt):
            u_arr.append(cs.MX.sym(f'w_{i}'))
            dt_arr.append(cs.MX.sym(f'dt_{i}'))
        u = cs.horzcat(*u_arr)
        dt = cs.horzcat(*dt_arr)
        
        odesol_full = odesol_single.mapaccum(nt)
        
        odesol = odesol_single.map(nt, 'thread', 4)
        
        out = odesol(x0 = x_s, p = cs.vertcat(dt, u))
        F_xf = out['xf']
        F_rf = F_xf[0,:]
        F_vf = F_xf[1,:]
        F_mf = F_xf[2,:]
        obj_expr = -F_mf[-1]
        cont_cond_expr = x_stages - F_xf[:,:-1]
        dt_equality_expr = cs.diff(dt, 1, 1)*(nt*10)
        

        max_drag_expr = A*(F_vf**2) * cs.exp(-k * (F_rf - r0))
        term_alt_expr = F_rf[-1] - rT
        
        xopt_arr = [dt_arr[0], u_arr[0]]
        _hessBlock_sizes = [2]
        lbv_arr = [cs.DM([1/(nt*1000),0])]
        ubv_arr = [cs.DM([1/nt,1])]
        # x_start_arr = [Tmax/nt, cs.DM([0])]
        _vBlock_sizes = [2]
        _vBlock_dependencies = [False]
        
        for i in range(1, nt):
            xopt_arr.append(x_arr[i-1])
            xopt_arr.extend([dt_arr[i], u_arr[i]])
            _hessBlock_sizes += [5]
            _vBlock_sizes += [3,2]
            _vBlock_dependencies += [True, False]
            lbv_arr.append(cs.DM([1.0, 0.0, C, 1/(nt*1000), 0]))
            ubv_arr.append(cs.DM([cs.inf, cs.inf, cs.inf, 1/nt, 1.0]))
        
        #Initial point
        # T_start = 0.05
        T_start = 0.4/7 * 2.5
        
        nt_acc = math.ceil(nt*2/5)
        nt_dec = math.floor(nt*3/5)
        
        
        out_full = odesol_full(x0 = x_init, p = cs.vertcat(cs.DM([T_start/nt]*nt).T, cs.DM([1.0]*nt_acc + [0.0]*nt_dec).T))
        xf_start = out_full['xf']
        x_start_arr = [cs.DM([T_start/nt, 1.0])]
        for i in range(1, nt_acc):
            x_start_arr.append(xf_start[:, i-1])
            x_start_arr.append(cs.DM([T_start/nt, 1.0]))
        for i in range(nt_acc, nt):
            x_start_arr.append(xf_start[:, i-1])
            x_start_arr.append(cs.DM([T_start/nt, 0.0]))
        
        self.x_start_arr = x_start_arr
        self.start_point = np.array(cs.vertcat(*x_start_arr), dtype = np.float64).reshape(-1)
        self.odesol = odesol_full
        
        self.hessBlock_sizes = np.array(_hessBlock_sizes, dtype = np.int32)
        self.hessBlock_index = np.cumsum([0] + _hessBlock_sizes, dtype = np.int32)
        self.vBlock_sizes = np.array(_vBlock_sizes, dtype = np.int32)
        self.vBlock_dependencies = np.array(_vBlock_dependencies, dtype = np.dtype('bool'))
        self.cBlock_sizes = np.array([3]*(nt - 1) + [nt - 1] + [max_drag_expr.numel()] + [1], dtype = np.int32)
        xopt = cs.vertcat(*xopt_arr)
        self.lb_var = np.array(cs.vertcat(*lbv_arr), dtype = np.float64).reshape(-1)
        self.ub_var = np.array(cs.vertcat(*ubv_arr), dtype = np.float64).reshape(-1)
        self.lb_con = np.array([0] * cont_cond_expr.numel() + [0]*(nt - 1) + [-np.inf]*max_drag_expr.numel() + [0], dtype = np.float64).reshape(-1)
        self.ub_con = np.array([0] * cont_cond_expr.numel() + [0]*(nt - 1) + [C]*max_drag_expr.numel() + [np.inf], dtype = np.float64).reshape(-1)

        self.NLP = {'x' : xopt, 'f' : obj_expr, 'g' : cs.vertcat(cs.vec(cont_cond_expr), cs.vec(dt_equality_expr), cs.vec(max_drag_expr), cs.vec(term_alt_expr))}
        self.nVar = xopt.numel()
        self.nCon = self.NLP['g'].numel()
        
        self._f = cs.Function('cs_f', [xopt], [obj_expr])
        self.f = lambda xi: np.array(self._f(xi), dtype = np.float64).reshape(-1)
        
        grad_f_expr = cs.jacobian(obj_expr, xopt)
        self._grad_f = cs.Function('cs_grad_f', [xopt], [grad_f_expr])
        self.grad_f = lambda xi: np.array(self._grad_f(xi), dtype = np.float64).reshape(-1)
        
        self._g = cs.Function('cs_g', [xopt], [self.NLP['g']])
        self.g = lambda xi: np.array(self._g(xi), dtype = np.float64).reshape(-1)
        jac_g_expr = cs.jacobian(self.NLP['g'], xopt)
        self._jac_g = cs.Function('cs_jac_g', [xopt], [jac_g_expr])
        self.jac_g = lambda xi: np.array(self._jac_g(xi), dtype = np.float64)
        
        self.jac_g_nnz = jac_g_expr.nnz()
        # self.jac_g_row = np.array(jac_g_expr.row(), dtype = np.int32).reshape(-1)
        self.jac_g_row = jac_g_expr.row()
        # self.jac_g_colind = np.array(jac_g_expr.colind(), dtype = np.int32).reshape(-1)
        self.jac_g_colind = jac_g_expr.colind()
        self.jac_g_nz = lambda xi: np.array(self._jac_g(xi).nz[:], dtype = np.float64).reshape(-1)

        lam = cs.MX.sym('lambda', self.NLP['g'].numel())
        lag_expr = self.NLP['f'] - lam.T @ self.NLP['g']
        grad_lag_expr = cs.jacobian(lag_expr, self.NLP['x'])
        hess_lag_expr = cs.jacobian(grad_lag_expr, self.NLP['x'])
        self._hess_lag = cs.Function('hess_lag', [self.NLP['x'], lam], [hess_lag_expr])
        self.hess_lag = lambda xi, lambd: self.to_blocks_LT(self._hess_lag(xi, lambd))
        
        
    def plot(self, xi:np.ndarray[np.float64]):
        time_grid = np.cumsum(np.concatenate(([0],xi[0::5]))).reshape(-1)
        u = xi[1::5]
        r = np.concatenate(([self.x_init[0]], xi[2::5]), axis = 0)
        v = np.concatenate(([self.x_init[1]], xi[3::5]), axis = 0)
        m = np.concatenate(([self.x_init[2]], xi[4::5]), axis = 0)
        
        plt.figure()
        plt.plot(time_grid[:-1], (r - 1)*100, 'b-', label = '(r-1)*100')
        plt.plot(time_grid[:-1], v*20, 'g-', label = 'v*20')
        plt.plot(time_grid[:-1], m, 'y-', label = 'm')
        
        plt.step(time_grid[:-1], u, color = 'r', label = 'u')
        plt.legend()
        plt.show()














