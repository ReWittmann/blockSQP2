#Collection of optimal control problems implemented using casadi.

import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
import typing
import math
import copy
plt.rcParams["text.usetex"] = True

def RK4_integrator(ODE, M = 2):
    #M: RK4 steps per interval
    DT = 1/M
    if not 'quad' in ODE:
        q = cs.MX.sym('q', 0)
    else:
        q = ODE['quad']
    if not 'p' in ODE:
        p = cs.MX.sym('p', 0)
    else:
        p = ODE['p']
    
    f = cs.Function('f', [ODE['x'], p], [ODE['ode'], q])
    X0 = ODE['x']
    U = p
    X = X0
    Q = 0
    for j in range(M):
        k1, k1_q = f(X, U)
        k2, k2_q = f(X + DT/2 * k1, U)
        k3, k3_q = f(X + DT/2 * k2, U)
        k4, k4_q = f(X + DT * k3, U)
        X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
        Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
    return cs.Function('F', [X0, U], [X, Q],['x0','p'],['xf','qf'])


def explicit_euler_integrator(ODE, M = 2):
    #M: RK4 steps per interval
    DT = 1/M
    if not 'quad' in ODE:
        q = cs.MX.sym('q', 0)
    else:
        q = ODE['quad']
    if not 'p' in ODE:
        p = cs.MX.sym('p', 0)
    else:
        p = ODE['p']
    
    f = cs.Function('f', [ODE['x'], p], [ODE['ode'], q])
    X0 = ODE['x']
    U = p
    X = X0
    Q = 0
    for j in range(M):
        dX, dQ = f(X, U)
        X = X + DT*dX
        Q = Q + DT*dQ
    return cs.Function('F', [X0, U], [X, Q],['x0','p'],['xf','qf'])

########################################
###Optimal control problem base class###
########################################

class OCProblem:    
    ##############################################################
    #Fields that a subclass implementing a problem should populate
    ##############################################################
    nVar : int #number of variables
    nCon : int #number of constraints
    
    #NLP dict as required for casadi NLP solvers
    NLP : {str : cs.MX}
    
    #Objective
    f : typing.Callable[[np.ndarray[1, np.float64]], float]
    #Constraint function
    g : typing.Callable[[np.ndarray[1, np.float64]], np.ndarray[1, np.float64]]
    #Objective gradient
    grad_f : typing.Callable[[np.ndarray[1, np.float64]], np.ndarray[1, np.float64]]
    #Constraint jacobian (either sparse or dense should be implemented)
    ##SPARSE: number of nonzeros, nonzeros, row indices, column starts (CCS format)
    jac_g_nnz : int
    jac_g_nz : typing.Callable[[np.ndarray[1, np.float64]], np.ndarray[1, np.float64]]
    jac_g_row : np.ndarray[1, np.int32]
    jac_g_colind: np.ndarray[1, np.int32]
    ##DENSE
    jac_g : typing.Callable[[np.ndarray[1, np.float64]], np.ndarray[2, np.float64]]
    
    _hess_lag : typing.Callable
    # x, lambda -> list(hessBlocks lower triangle elements)
    hess_lag : typing.Callable[[np.ndarray[1, np.float64], np.ndarray[1, np.float64]], typing.List[np.ndarray[1, np.float64]]]
    
    #Bounds
    lb_var : np.ndarray[1, np.float64]
    ub_var : np.ndarray[1, np.float64]
    lb_con : np.ndarray[1, np.float64]
    ub_con : np.ndarray[1, np.float64]
    
    #Starting point for optimization
    start_point : np.ndarray[1, np.float64]
    
    #Structure and variable type information (integrality, dependency, hessian blocks, ...)##
    #set by multiple_shooting helper method
    hessBlock_sizes : list[int]
    hessBlock_index : list[int]
    vBlock_sizes : list[int]            #partition of variables into blocks
    vBlock_dependencies : list[bool]    #Which blocks are free/dependent
            
    cBlock_sizes : list[int]            #Partition of constraints, used to distinguish individual continuity conditions
    ctarget_data : list[int]            #Specifies target for condensing, see blockSQP/src/blockSQP_condensing.hpp
    
    #########################################################
    #Internal fields
    #########################################################
    
    #Basic problem data, set in constructor
    nx : int #number of states
    nu : int #number of true controls
    np : int #total of parameters
    np_m : int #number of non-true controls (parameters, time interval lengths)
    
    
    nuS : int #number of control variables per stage
    nq : int #number of quadratures
    nfree : int #number of free initial values
    x_init : typing.Iterable[typing.Optional[float]] #(partially) fixed initial values
    fix_time : bool #is time horizon fixed?
    ntS : int #number of time (shooting) stages
    ntR : int #refinement multiplier, how many controls and possibly constraint evaluations per stage
    
    lbu : list #lower control bound
    ubu : list #upper control bound
    lbx : list #lower state bound
    ubx : list #upper state bound
    
    #Integrator data
    integration_method : str
    parallel : bool
    N_threads : int
    
    
    #Model specific data, set in problem subclass
    model_params : typing.Dict[str, typing.Any]
    time_grid : typing.Optional[np.ndarray[np.int32]]
    time_grid_ref : typing.Optional[np.ndarray[np.int32]]
    ODE : typing.Dict[str,cs.MX]

    
    odesol_single : object #single shooting integrator x_0, [u_k] -> x1,x2,...
    odesol_refC : object #Control-refined shooting interval integrator x_k, [u_k0,u_k1,...] -> x_{k+1}
    odesol_refined : object #Control + state - refined shooting interval integrator x_k, [u_k0,u_k1,...] -> x_k0, x_k1, ... , x_{k+1}
    # odesol_fill : object
    odesol_multi : object #odesol_refined mapped over all shooting intervals
    odesol_full : object #odesol_refC map-accumulated over all shooting intervals
    
    #Expressions for building the objective and constraints
    x_eval : cs.MX # State values composed of state variables x_k and intermediate state values F_k(t_k_j, x_k), excluding a fixed initial value
    q_eval : cs.MX # Quadrature values for each control interval
    u_eval : cs.MX # The controls
    p_eval : cs.MX # The localized parameters, which include dt as first parameter if the time horizon is not fixed
    
    q_tf : cs.MX # The quadratures over the whole time horizon
    p_tf : cs.MX # The localized parameter of the last interval
    
    cont_cond_expr : cs.MX
    
    #List of constraint expressions and the associated bounds
    constr_arr : list[cs.MX]
    lbc_arr : list
    ubc_arr : list
    
    #Symbolic integrator output
    F_xf : cs.MX
    F_qf : cs.MX
    
    #nt - number of shooting intervals
    #refine - control intervals per shooting interval
    #integrator (RK4, explicit_euler, cvodes, collocation) - ODE integrator
    #parallel - parallelize ODE integration over shooting intervals
    #N_threads - number of threads for integration parallelization
    #kwargs - problem specific parameters, see problem default parameters
    def __init__(self, nt = 100, refine = 1, integrator = 'rk4', parallel = True, N_threads = 4, **kwargs):
        if hasattr(self, 'default_params'):
            self.model_params = copy.copy(self.default_params)
        else:
            self.model_params = dict()
        self.model_params.update(**kwargs)
        self.integration_method = integrator
        self.parallel = parallel
        self.N_threads = N_threads
        
        self.NLP = dict()       
        self.time_grid = None        
        self.time_grid_ref = None        
        self.odesol_single = None
        self.odesol_refc = None
        self.odesol_multi = None        
        
        self.ntS = nt
        self.ntR = refine
        
        self.build_problem()
        
    #See existing implementations
    def build_problem():
        raise NotImplementedError('Optimal control problem must be implemented in subclass via build_problem method')
    
    
    def set_OCP_data(self,    
                     nx : int,              # Number of states
                     np : int,              # Number of parameters
                     nu : int,              # Number of controls
                     nq : int,              # Number of quadratures
                     lbx : typing.Iterable, # Lower state bounds
                     ubx : typing.Iterable, # Upper state bounds
                     lbp : typing.Iterable, # Lower parameter bounds
                     ubp : typing.Iterable, # Upper parameter bounds
                     lbu : typing.Iterable, # Lower control bounds
                     ubu : typing.Iterable  # Upper control bounds
                     ):
        self.nx = nx
        self.nu = nu
        self.np = np
        self.nq = nq
        self.nfree = self.nx
        self.x_init = [None]*self.nx
        self.fix_time = False
        
        self.lbx = list(lbx)
        self.ubx = list(ubx)
        self.lbp = list(lbp)
        self.ubp = list(ubp)
        self.lbu = list(lbu)
        self.ubu = list(ubu)
        
        self.cBlock_sizes = []
        self.constr_arr = []
        self.lbc_arr = []
        self.ubc_arr = []
    
    def to_blocks_LT(self, sparse_hess : cs.DM):
        blocks = []
        for j in range(len(self.hessBlock_sizes)):
           blocks.append(np.array(cs.tril(sparse_hess[self.hessBlock_index[j]:self.hessBlock_index[j+1], self.hessBlock_index[j]:self.hessBlock_index[j+1]].full()).nz[:], dtype = np.float64).reshape(-1))
        return blocks
    
    def to_blocks(self, sparse_hess : cs.DM):
        blocks = []
        for j in range(len(self.hessBlock_sizes)):
            blocks.append(np.array(sparse_hess[self.hessBlock_index[j]:self.hessBlock_index[j+1], self.hessBlock_index[j]:self.hessBlock_index[j+1]].full(), dtype = np.float64))
        return blocks
    
    def set_model_params(self, **kwargs):
        if hasattr(self, 'default_params'):
            self.model_params = self.default_params
        else:
            self.model_params = dict()
        self.model_params.update(kwargs)
    
    def fix_initial_value(self,initval):
        self.x_init = initval
        self.nfree = len([0 for x in self.x_init if x is None])
    
    def fix_time_horizon(self, t0, tf):
        self.time_grid = np.linspace(t0,tf,self.ntS+1,endpoint=True)
        self.time_grid_ref = np.linspace(t0,tf,self.ntS*self.ntR + 1, endpoint = True)
        self.fix_time = True
    
    def build_integrator(self):
        if self.integration_method.lower() == 'cvodes':
            # print('cvodes')
            self.odesol_single = cs.integrator('odesol_single', 'cvodes', self.ODE, {'linear_solver': 'csparse', 'augmented_options' : {'linear_solver' : 'csparse'}})
        elif self.integration_method.lower() == 'collocation':
            self.odesol_single = cs.integrator('odesol_single', 'collocation', self.ODE, {'number_of_finite_elements': 2})
        elif self.integration_method.lower() == 'explicit_euler':
            self.odesol_single = explicit_euler_integrator(self.ODE, M = 1)
        else:
            # print('rk4')
            self.odesol_single = RK4_integrator(self.ODE, M = 2)
        self.odesol_refined = self.odesol_single.mapaccum('odesol_refined',self.ntR, ['x0'], ['xf'])
        
        if self.ntR > 1:
            self.odesol_refC = self.odesol_single.fold(self.ntR)
            # self.odesol_fill = self.odesol_single.mapaccum(self.ntR)
        else:
            self.odesol_refC = self.odesol_single
            # self.odesol_fill = self.odesol_single
        self.odesol_full = self.odesol_refC.mapaccum('odesol_full', self.ntS, ['x0'], ['xf'])
        
    def integrate_full(self, xi):
        p_arr = [self.get_stage_param(xi, i) for i in range(self.ntS)]
        # if self.fix_time:
        #     p = cs.vertcat(cs.diff(self.time_grid_ref.reshape((1,-1)),1,1), p)
        p_exp_arr = []
        for i in range(self.ntS):
            for j in range(self.ntR):
                p_exp_arr.append(p_arr[i])
        p_exp = cs.horzcat(*p_exp_arr)
        if self.fix_time:
            p_exp = cs.vertcat(cs.diff(self.time_grid_ref.reshape((1,-1)),1,1), p_exp)
        else:
            p_exp[0,:]/=self.ntR
        
        u = cs.horzcat(*(self.get_stage_control(xi, i) for i in range(self.ntS)))
        # if self.fix_init:
        #     start = self.x_init
        # else:
        #     start = self.get_stage_state(xi, 0)
        start = self.get_stage_state(xi, 0)
        out = self.odesol_full(x0 = start, p = cs.vertcat(p_exp, u))
        x_stages = out['xf']
        for i in range(1,self.ntS + 1):
            self.set_stage_state(xi, i, x_stages[:,i-1])
        
        
    def add_constraint(self, constr : cs.MX, lbc : typing.Union[typing.Iterable, float, int], ubc : typing.Union[typing.Iterable, float, int], block_sizes : typing.Optional[list[int]] = None):
        if constr.numel() == 0:
            return
        
        if block_sizes is None:
            self.cBlock_sizes.append(constr.numel())
        else:
            self.cBlock_sizes += block_sizes
        self.constr_arr.append(constr)
        
        if isinstance(lbc, (int, float)):
            lbc_t = np.array([lbc], dtype = np.float64)
        else:
            lbc_t = np.array(lbc)
        if isinstance(ubc, (int, float)):
            ubc_t = np.array([ubc], dtype = np.float64)
        else:
            ubc_t = np.array(ubc)
        
        assert constr.numel()%len(lbc_t) == 0
        self.lbc_arr.append(np.concatenate([lbc_t]*(constr.numel()//len(lbc_t))))
        assert constr.numel()%len(ubc_t) == 0
        self.ubc_arr.append(np.concatenate([ubc_t]*(constr.numel()//len(ubc_t))))
        
        # self.lbc_arr.append(np.array(lbc))
        # self.ubc_arr.append(np.array(ubc))
    
    def set_objective(self, obj : cs.MX):
        self.NLP['f'] = obj
    
    
    def multiple_shooting(self):
        if self.odesol_single is None:
            self.build_integrator()
        
        x_arr = []
        
        x_init_free = cs.MX.sym('x_s_0_free', self.nfree, 1)
        x_init_arr = []
        x_init_lb = []
        x_init_ub = []
        j = 0
        for i in range(self.nx):
            if self.x_init[i] is not None:
                x_init_arr.append(self.x_init[i])
            else:
                x_init_arr.append(x_init_free[j])
                j += 1
                x_init_lb.append(self.lbx[i])
                x_init_ub.append(self.ubx[i])
        x_init = cs.vertcat(*x_init_arr)
        
        x_arr.append(x_init)
        for i in range(1, self.ntS + 1):
            x_arr.append(cs.MX.sym(f'x_s_{i}', self.nx, 1))
        x_stages = cs.horzcat(*x_arr[1:])
        x_starts = cs.horzcat(*x_arr[0:self.ntS])
        
        u_arr = []
        for i in range(self.ntS):
            u_arr.append(cs.MX.sym(f'u_{i}', self.nu, 1))
            for j in range(1, self.ntR):
                u_arr.append(cs.MX.sym(f'u_{i}_{j}', self.nu, 1))
        u = cs.horzcat(*u_arr)
        
        #If time horizon is not fixed, p[0,:] is assumed to be the variable stage duration
        p_arr = []
        p_exp_arr = []
        for i in range(self.ntS):
            p_arr.append(cs.MX.sym(f'p_{i}', self.np))
            for j in range(self.ntR):
                p_exp_arr.append(p_arr[i])
        p_exp = cs.horzcat(*p_exp_arr)
        if not self.fix_time:
            p_exp[0,:] /= self.ntR
            p_t_exp = p_exp
        else:
            assert self.time_grid is not None
            p_t_exp = cs.vertcat(cs.diff(cs.DM(self.time_grid_ref).T, 1, 1), p_exp)
        
        
        if self.parallel:
            self.odesol_multi = self.odesol_refined.map(self.ntS, 'thread', self.N_threads)
        else:
            self.odesol_multi = self.odesol_refined.map(self.ntS)
        
        # if self.fix_time:
        #     assert self.time_grid is not None
        #     out = self.odesol_multi(x0 = x_starts, p = cs.vertcat(cs.diff(cs.DM(self.time_grid_ref).T, 1, 1), p_m_exp, u))
        # else:
        out = self.odesol_multi(x0 = x_starts, p = cs.vertcat(p_t_exp, u))
        self.F_xf = out['xf']
        self.F_qf = out['qf']
        
        # self.add_constraint(cs.vec(x_stages - self.F_xf[:,self.ntR-1:-self.ntR:self.ntR]), 0., 0., [self.nx]*(self.ntS - 1))
        self.add_constraint(cs.vec(x_stages - self.F_xf[:,self.ntR-1:self.ntR*self.ntS:self.ntR]), 0., 0., [self.nx]*self.ntS)

        #Evaluate state bounds at intermediate values of refinement is used
        if self.ntR > 1:
            F_x_eval = []
            for i in range(self.ntS):
                F_x_eval.append(self.F_xf[:,i*self.ntR:(i+1)*self.ntR-1])
            self.add_constraint(cs.vec(cs.horzcat(*F_x_eval)), self.lbx, self.ubx)
        
        
        # self.x_eval = copy.copy(self.F_xf)
        self.x_eval = cs.horzcat(x_init, self.F_xf)
        # self.x_eval[:,self.ntR-1:-self.ntR*:self.ntR] = x_s[:,1:]
        self.x_eval[:,self.ntR:self.ntR*self.ntS + 1:self.ntR] = x_stages
        
        self.q_eval = self.F_qf
        self.q_tf = cs.sum2(self.q_eval)
        self.u_eval = u
        self.p_eval = cs.horzcat(*p_arr)
        self.p_tf = self.p_eval[:,-1]
        self.p_exp_eval = p_exp
        
        if self.np > 0:
            self.add_constraint(cs.diff(self.p_eval, 1, 1), 0, 0)
        
        xopt_arr = []
        lbv_arr = []
        ubv_arr = []
        
        self.hessBlock_sizes = [0]
        self.vBlock_sizes = [0]
        self.vBlock_dependencies = [False]
        # if not self.fix_init:
        #     xopt_arr.append(x_arr[0])
        #     self.hessBlock_sizes[0] += self.nx
        #     self.vBlock_sizes[0] += self.nx
        #     lbv_arr.append(self.lbx)
        #     ubv_arr.append(self.ubx)
        xopt_arr.append(x_init_free)
        self.hessBlock_sizes[0] += self.nfree
        self.vBlock_sizes[0] += self.nfree
        lbv_arr.append(cs.DM(x_init_lb))
        ubv_arr.append(cs.DM(x_init_ub))
        
        # lbv_arr += self.lbu*self.ntR
        # ubv_arr += self.ubu*self.ntR
        xopt_arr.append(p_arr[0])
        for j in range(self.ntR):
            xopt_arr.append(u_arr[j])
        lbv_arr += [cs.DM(self.lbp), cs.DM(self.lbu * self.ntR)]
        ubv_arr += [cs.DM(self.ubp), cs.DM(self.ubu * self.ntR)]
        
        self.hessBlock_sizes[0] += self.np + self.ntR * self.nu
        self.vBlock_sizes[0] += self.np + self.ntR * self.nu
        self.vBlock_dependencies = [False]
        
        for i in range(1, self.ntS):
            xopt_arr.append(x_arr[i])
            xopt_arr.append(p_arr[i])
            for j in range(self.ntR):
                xopt_arr.append(u_arr[i*self.ntR + j])
            self.hessBlock_sizes += [self.nx + self.np + self.ntR*self.nu]
            self.vBlock_sizes += [self.nx, self.np + self.ntR*self.nu]
            self.vBlock_dependencies += [True, False]
            lbv_arr.append(cs.DM(self.lbx + self.lbp + self.lbu*self.ntR))
            ubv_arr.append(cs.DM(self.ubx + self.ubp + self.ubu*self.ntR))        
        
        #Terminal state is a shooting variable
        xopt_arr.append(x_arr[self.ntS])
        self.hessBlock_sizes += [self.nx]
        self.vBlock_sizes += [self.nx]
        self.vBlock_dependencies += [True]
        lbv_arr.append(cs.DM(self.lbx))
        ubv_arr.append(cs.DM(self.ubx))
        
        
        self.hessBlock_index = list(np.cumsum([0] + self.hessBlock_sizes, dtype = np.int32))
        # self.cBlock_sizes = [self.nx]*(self.ntS - 1)
        self.NLP['x'] = cs.vertcat(*xopt_arr)
        self.nVar = self.NLP['x'].numel()
        self.start_point = np.zeros(self.nVar)
        self.lb_var = np.array(cs.vertcat(*lbv_arr), dtype = np.float64).reshape(-1)
        self.ub_var = np.array(cs.vertcat(*ubv_arr), dtype = np.float64).reshape(-1)
        self.ctarget_data = [self.ntS, 0, 2*self.ntS, 0, self.ntS]
    
    #Finalize NLP and populate NLP function fields
    def build_NLP(self):
        if 'x' not in self.NLP.keys() or 'f' not in self.NLP.keys():
            raise Exception('Error, multiple_shooting and set_objective need to be called before NLP dict can be built')
        self.NLP['g'] = cs.vertcat(*[cs.vec(constr) for constr in self.constr_arr])
        self.lb_con = np.concatenate(self.lbc_arr)
        self.ub_con = np.concatenate(self.ubc_arr)
        self.nVar = self.NLP['x'].numel()
        self.nCon = self.NLP['g'].numel()
        
        xopt = self.NLP['x']
        obj_expr = self.NLP['f']
        g_expr = self.NLP['g']
        
        self._f = cs.Function('cs_f', [xopt], [obj_expr])
        self.f = lambda xi: np.array(self._f(xi), dtype = np.float64).reshape(-1)
        
        grad_f_expr = cs.jacobian(obj_expr, xopt)
        self._grad_f = cs.Function('cs_grad_f', [xopt], [grad_f_expr])
        self.grad_f = lambda xi: np.array(self._grad_f(xi), dtype = np.float64).reshape(-1)
        
        self._g = cs.Function('cs_g', [xopt], [g_expr])
        self.g = lambda xi: np.array(self._g(xi), dtype = np.float64).reshape(-1)
        jac_g_expr = cs.jacobian(self.NLP['g'], xopt)
        self._jac_g = cs.Function('cs_jac_g', [xopt], [jac_g_expr])
        self.jac_g = lambda xi: np.array(self._jac_g(xi), dtype = np.float64)
        
        self.jac_g_nnz = jac_g_expr.nnz()
        self.jac_g_row = jac_g_expr.row()
        self.jac_g_colind = jac_g_expr.colind()
        self.jac_g_nz = lambda xi: np.array(self._jac_g(xi).nz[:], dtype = np.float64).reshape(-1)
        
        lam = cs.MX.sym('lambda', g_expr.numel())
        self.lag_expr = self.NLP['f'] - lam.T @ g_expr
        self.grad_lag_expr = cs.jacobian(self.lag_expr, xopt)
        self.grad_lag = cs.Function('grad_lag', [xopt, lam], [self.grad_lag_expr])
        
        self.hess_lag_expr = cs.jacobian(self.grad_lag_expr, xopt)
        self._hess_lag = cs.Function('hess_lag', [xopt, lam], [self.hess_lag_expr])
        self.hess_lag = lambda xi, lambd: self.to_blocks_LT(self._hess_lag(xi, lambd))

    
    def get_stage_state(self, xi, i:int):
        if i == 0:
            x_init_arr = []
            j = 0
            for k in range(self.nx):
                if self.x_init[k] is not None:
                    x_init_arr.append(self.x_init[k])
                else:
                    x_init_arr.append(xi[j])
                    j += 1
            return np.array(x_init_arr).reshape((self.nx, -1), order = 'F')
        else:
            return xi[i*(self.np + self.ntR*self.nu) + (i-1)*self.nx + self.nfree: i*(self.np + self.ntR*self.nu) + i*self.nx + self.nfree].reshape((self.nx, -1), order = 'F')
    
    def get_stage_param(self, xi, i:int):
        if self.np == 0:
            return np.array([])
        
        return xi[i*(self.np + self.ntR*self.nu) + i*self.nx + self.nfree:(i+1)*self.np + i*self.ntR*self.nu+ i*self.nx + self.nfree].reshape((self.np, -1), order = 'F')
    
    def get_stage_control(self, xi, i:int):
        if self.nu == 0:
            return np.array([])
        return xi[(i+1)*self.np + i*self.ntR*self.nu + i*self.nx + self.nfree:(i+1)*(self.np + self.ntR*self.nu) + i*self.nx + self.nfree].reshape((self.nu,-1), order = 'F')
    
    def set_stage_state(self, xi, i:int, val):
        val = np.array(val).reshape(-1)
        if i == 0:
            if len(val) == self.nfree:
                val_free = val
            else:
                val_free = np.array([val[i] for i in range(self.nx) if self.x_init[i] is None])
            xi[0:self.nfree] = val_free
            return
        xi[i*(self.np + self.ntR*self.nu) + (i-1)*self.nx + self.nfree: i*(self.np + self.ntR*self.nu) + i*self.nx + self.nfree] = np.array(val).reshape(-1)
        return
        
    def set_stage_param(self, xi, i:int, val):
        xi[i*(self.np + self.ntR*self.nu) + i*self.nx + self.nfree:(i+1)*self.np + i*self.ntR*self.nu + i*self.nx + self.nfree] = np.array(val).reshape(-1)
            
    def set_stage_control(self, xi, i:int, val):
        val = np.array(val).reshape(-1)
        if self.nu*self.ntR/len(val) > 1:
            assert self.nu*self.ntR % len(val) == 0
            val = np.tile(val, int((self.nu*self.ntR)/len(val)))
        xi[(i+1)*self.np + i*self.ntR*self.nu + i*self.nx + self.nfree:(i+1)*(self.np + self.ntR*self.nu) + i*self.nx + self.nfree] = val
    
    #Get all state state variables, including terminal state
    def get_state_arrays(self, xi):
        x = np.hstack([self.get_stage_state(xi,i) for i in range(self.ntS + 1)])
        if self.nx == 1:
            return x.reshape(-1)
        else:
            return tuple(x[i,:].reshape(-1) for i in range(self.nx))
        
    def get_control_arrays(self, xi):
        u = np.hstack([self.get_stage_control(xi,i) for i in range(self.ntS)])
        if self.nu == 1:
            return u.reshape(-1)
        else:
            return tuple(u[i,:].reshape(-1) for i in range(self.nu))
    
    def get_control_plot_arrays(self, xi):
        u_arr = [self.get_stage_control(xi,i) for i in range(self.ntS)]
        u_arr = [u_arr[0][:,0].reshape((self.nu, -1))] + u_arr
        u = np.hstack(u_arr)
        if self.nu == 1:
            return u.reshape(-1)
        else:
            return tuple(u[i,:].reshape(-1) for i in range(self.nu))
    
    def get_param_arrays(self, xi):
        p = np.hstack([self.get_stage_param(xi,i) for i in range(self.ntS)])
        if self.np == 1:
            return p.reshape(-1)
        else:
            return tuple(p[i,:].reshape(-1) for i in range(self.np))
    
    def get_param_arrays_expanded(self, xi):
        p = np.hstack([self.get_stage_param(xi,i) for i in range(self.ntS) for j in range(self.ntR)])#.reshape((self.np,-1), order = 'F')
        if not self.fix_time:
            p[0,:]/=self.ntR
        if self.np == 1:
            return p.reshape(-1)
        else:
            return tuple(p[i,:].reshape(-1) for i in range(self.np))
    
    def get_state_arrays_expanded(self, xi):
        x_arr = []
        for i in range(self.ntS):
            x_i = cs.DM(self.get_stage_state(xi, i))
            p_i = cs.DM(cs.repmat(self.get_stage_param(xi, i), 1, self.ntR))
            u_i = cs.DM(self.get_stage_control(xi, i))
            if not self.fix_time:
                p_i[0,:]/=self.ntR
            else:
                p_i = cs.vertcat(cs.diff(cs.DM(self.time_grid_ref[i*self.ntR:(i+1)*self.ntR + 1]).T, 1, 1), p_i)
            # out = self.odesol_fill(x0 = x_i, p = cs.vertcat(p_i,u_i))
            out = self.odesol_refined(x0 = x_i, p = cs.vertcat(p_i,u_i))
            x_arr.append(x_i)
            x_arr.append(out['xf'][:,:-1])
        #Terminal state
        x_arr.append(self.get_stage_state(xi, self.ntS))
        x = np.array(cs.horzcat(*x_arr))
        if self.nx == 1:
            return x.reshape(-1)
        else:
            return tuple(x[i,:].reshape(-1) for i in range(self.nx))
        
    #For overwriting
    # def __str__():
    #     return "OCProblem"
    
    def plot(self, xi, dpi = None, title = None, it = None):
        raise NotImplementedError('No plot functionality implemented for this problem')
    
    def perturbed_start_point(self, ind):
        raise NotImplementedError('No perturbed start points implemented for this problem')
    
    
def from_block_LT(HLT, dim):
    H = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(i,dim):
            H[j,i]= HLT[i*dim + j - int((i*(i+1))/2)]
        for j in range(i+1,dim):
            H[i,j] = HLT[i*dim + j - int((i*(i+1))/2)]
    return H



######################################
###Optimal control problem template###
######################################

# class example_optimalControl(OCProblem):
#     default_params = {'some param':some value}
    
#     def build_problem(self):
         ###Set nx, np, nu, nq, lbx, ubx, lbp, ubp, lbu, ubu###
         ###number + lower/upper bounds of states,controls,parameters,quadratures###
#         self.set_OCP_data(2,1,0,1,[0,0],[np.inf, np.inf],[],[],[0],[1])
        
        ###fix time horizon if it is not subject to optimization
#         self.fix_time_horizon(t0, tf)
        ###fix initial state if it is not subject to optimization
#         self.fix_initial_value(initial_value)
        
        ###Define casadi ODE dictionary
        ##create casadi MX symbols and symbolic dynamics
#         x = cs.MX.sym('x', self.nx)
#         u = cs.MX.sym('u', self.nu)
#         p = cs.MX.sym('p', self.np)
#       ##REQUIRED: create time interval length for scaling, may be part of parameters if time horizon is not fixed
#         dt = cs.MX.sym('dt') or dt = p[0]
#         ode_rhs = ...
#         quad_expr = ...
#         self.ODE = {'x': x, 'p': , 'ode': dt*ode_rhs, 'quad': dt*quad_expr}
        
        ##Helper function for multiple shooting discretization
#         self.multiple_shooting()
        ##Stage variables are set as self.x_eval, self.p_eval, self.u_eval
        ##self.p_exp_eval is the same as self.p_eval, but repeated once for each additional refinement point (self.ntR)
#         self.set_objective(...)
#         self.add_constraint(...)

        ##Finish NLP after adding all constraints and objective
#         self.build_NLP()
        
        ##Provide an starting point for optimization, use set_stage_(state/param/control) to access the desired variables
#         self.start_point = np.zeros(self.nVar)
#         for i in range(self.ntS):
#             self.set_stage_state(self.start_point, i, ...)
    
    # def plot(self, xi, dpi = None, title = None, it = None):
        #Get states, controls and parameters as arrays
        # x0,x1,... = self.get_state_arrays(xi)
        # u0,u1,... = self.set_control_arrays(xi)
        # p0,p1,... = self.get_param_arrays(xi)
        
        #plot using e.g. matplotlib



#######################################
###Optimal control problem instances###
#######################################

class Lotka_Volterra_Fishing(OCProblem):
    default_params = {
            'c0':0.4, 
            'c1':0.2, 
            'x_init':[0.5,0.7], 
            't0':0., 
            'tf':12.
            }
    
    def build_problem(self):
        self.set_OCP_data(2,0,1,1,[0,0],[np.inf, np.inf],[],[],[0],[1])
        self.fix_time_horizon(self.model_params['t0'],self.model_params['tf'])
        self.fix_initial_value(self.model_params['x_init'])
        
        x = cs.MX.sym('x', 2)
        w = cs.MX.sym('w', 1)
        x0, x1 = cs.vertsplit(x)
        ode_rhs = cs.vertcat(x0 - x0*x1 - self.model_params['c0']*x0*w, -x1 + x0*x1 - self.model_params['c1']*x1*w)
        quad_expr = (x0 - 1)**2 + (x1 - 1)**2
        dt = cs.MX.sym('dt', 1)
        self.ODE = {'x': x, 'p':cs.vertcat(dt, w),'ode': dt*ode_rhs, 'quad': dt*quad_expr}
        self.multiple_shooting()
        self.set_objective(self.q_tf)
        self.build_NLP()
        
        self.start_point = np.zeros(self.nVar)
        for i in range(self.ntS+1):
            self.set_stage_state(self.start_point, i, self.model_params['x_init'])
        for i in range(self.ntS):
            self.set_stage_control(self.start_point, i, [0])
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, 0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x0, x1 = self.get_state_arrays_expanded(xi)
        u = self.get_control_plot_arrays(xi)
        
        # plt.figure(dpi = dpi)
        fig,ax = plt.subplots(dpi=dpi)
        ax.plot(self.time_grid_ref, x0, 'tab:green', linestyle='-.', label = '$x_0$')
        ax.plot(self.time_grid_ref, x1, 'tab:blue', linestyle='--', label = '$x_1$')
        ax.step(self.time_grid_ref, u, 'tab:red', linestyle='-', label = r'$u$')
        ax.legend(fontsize='x-large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Lotka Volterra fishing problem'
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


class Lotka_Volterra_Fishing_MAYER(OCProblem):
    default_params = {'c0':0.4, 'c1':0.2, 'x_init':[0.5,0.7], 't0':0., 'tf':12.}
    
    def build_problem(self):
        self.set_OCP_data(3,0,1,0,[0,0,0],[np.inf, np.inf, np.inf],[],[],[0],[1])
        self.fix_time_horizon(self.model_params['t0'],self.model_params['tf'])
        self.fix_initial_value(self.model_params['x_init']+[0])
        
        x = cs.MX.sym('x', 3)
        w = cs.MX.sym('w', 1)
        x0, x1, q = cs.vertsplit(x)
        ode_rhs = cs.vertcat(x0 - x0*x1 - self.model_params['c0']*x0*w, 
                             -x1 + x0*x1 - self.model_params['c1']*x1*w, 
                             ((x0 - 1)**2 + (x1 - 1)**2))
        dt = cs.MX.sym('dt', 1)
        self.ODE = {'x': x, 'p':cs.vertcat(dt, w),'ode': dt*ode_rhs}
        self.multiple_shooting()
        self.set_objective(self.x_eval[2,-1])
        self.build_NLP()
        
        self.start_point = np.zeros(self.nVar)
        for i in range(self.ntS+1):
            self.set_stage_state(self.start_point, i, self.model_params['x_init'] + [i/100 * 2.4])
        for i in range(self.ntS):
            self.set_stage_control(self.start_point, i, [0])
        self.integrate_full(self.start_point)    
        
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, 0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x0, x1, _ = self.get_state_arrays(xi)
        # x0, x1 = self.get_state_arrays_expanded(xi)
        u = self.get_control_plot_arrays(xi)
        
        
        plt.figure(dpi = dpi)
        plt.plot(self.time_grid, x0, 'r-', label = '$x_0$')#, self.time_grid[:,-1], x1, '--', self.time_grid[:,-1], u, 'o')
        plt.plot(self.time_grid, x1, 'b--', label = '$x_1$')
        
        plt.step(self.time_grid_ref, u, 'g', label = r'$u\cdot 10$')
        plt.legend(fontsize='large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Lotka Volterra fishing problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
            
        plt.show()
        plt.close()


class Lotka_Volterra_multimode(OCProblem):
    default_params = {'t0':0., 'tf':12., 'c01': 0.2, 'c02':0.4, 'c03':0.01, 'c11':0.1, 'c12':0.2, 'c13':0.1}
    def build_problem(self):
        self.set_OCP_data(2,0,3,1, [0.,0.], [np.inf,np.inf], [], [], [0.,0.,0.], [1.,1.,1.])
        t0, tf, c01, c02, c03, c11, c12, c13 = (self.model_params[key] for key in ['t0', 'tf', 'c01', 'c02', 'c03', 'c11', 'c12', 'c13'])
        self.fix_initial_value([0.5,0.7])
        self.fix_time_horizon(t0,tf)
        
        x = cs.MX.sym('x',2)
        x0,x1 = cs.vertsplit(x)
        w = cs.MX.sym('w',3)
        w1,w2,w3 = cs.vertsplit(w)
        dt = cs.MX.sym('dt')
        ode_rhs = cs.vertcat(x0 - x0*x1 - c01*x0*w1 - c02*x0*w2 - c03*x0*w3,
                             -x1 + x0*x1 - c11*x1*w1 - c12*x1*w2 - c13*x1*w3
                             )
        quad = (x0-1)**2 + (x1-1)**2
        self.ODE = {'x':x, 'p':cs.vertcat(dt,w), 'ode':dt*ode_rhs, 'quad':dt*quad}
        self.multiple_shooting()
        self.set_objective(self.q_tf)
        self.add_constraint(cs.sum1(self.u_eval),1.,1.)
        for i in range(self.ntS+1):
            self.set_stage_state(self.start_point, i, self.x_init)
        for i in range(self.ntS):
            self.set_stage_control(self.start_point, i, [1/3,1/3,1/3])
        self.build_NLP()
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, [0.5, 0.25, 0.25])
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x0,x1 = self.get_state_arrays(xi)
        w1,w2,w3 = self.get_control_plot_arrays(xi)
        
        plt.figure(dpi = dpi)
        plt.plot(self.time_grid, x0, 'r-', label = '$x_0$')#, self.time_grid[:,-1], x1, '--', self.time_grid[:,-1], u, 'o')
        plt.plot(self.time_grid, x1, 'b--', label = '$x_1$')
        
        plt.step(self.time_grid_ref, w1, 'g', label = '$w_1$')
        plt.step(self.time_grid_ref, w2, 'c', label = '$w_2$')
        plt.step(self.time_grid_ref, w3, 'y', label = '$w_3$')
        
        plt.legend(fontsize='large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Lotka Volterra multimode problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
        plt.show()
        plt.close()


class Goddard_Rocket(OCProblem):
    default_params = {
        'rT':1.01, 
        'b':7.0, 
        'A':310.0, 
        'k':500.0, 
        'Tmax':3.5, 
        'C':0.6, 
        'x_init':[1.0,0.0,1.0]
        }
    
    def build_problem(self):
        
        self.set_OCP_data(3,1,1,0,[1.0,0.,0.],[np.inf,np.inf,np.inf],[0],[np.inf],[0],[1])
        self.fix_initial_value(self.model_params['x_init'])
        
        x = cs.MX.sym('x', self.nx)
        r,v,m = cs.vertsplit(x)
        r0,v0,m0 = cs.vertsplit(cs.DM(self.x_init))
        
        u = cs.MX.sym('u', self.nu)
        p = cs.MX.sym('p', self.np)
        
        dt = p
        
        Tmax, A, b, k, rT, C = (self.model_params[key] for key in ('Tmax', 'A', 'b', 'k', 'rT', 'C'))
        
        ode_rhs = cs.vertcat(v,\
                            -1/(r**2) + (1/m) * (Tmax*u - A*(v**2) * cs.exp(-k * (r - r0))),\
                            -b*u)
        
        self.ODE = {'x': x, 'p':cs.vertcat(dt, u),'ode': dt*ode_rhs}
        self.multiple_shooting()
        
        v_eval = self.x_eval[1,:]
        r_eval = self.x_eval[0,:]
        
        max_drag_expr = A*(v_eval**2) * cs.exp(-k * (r_eval - r0))
        term_alt_expr = r_eval[-1] - rT
        self.add_constraint(max_drag_expr, -np.inf, C)
        self.add_constraint(term_alt_expr, 0., np.inf)
        
        self.start_point = np.zeros(self.nVar)
        nt_acc = math.ceil(self.ntS*2/5)
        nt_dec = math.floor(self.ntS*3/5)
        for i in range(nt_acc):
            self.set_stage_control(self.start_point, i, [1.0])
            self.set_stage_param(self.start_point, i, [0.4/(b*0.4)/self.ntS])
        for i in range(nt_acc,nt_acc+nt_dec):
            self.set_stage_control(self.start_point, i, [0.0])
            self.set_stage_param(self.start_point, i, [0.4/(b*0.4)/self.ntS])
                
        self.integrate_full(self.start_point)
        self.set_objective(-self.x_eval[2,-1])
        self.build_NLP()
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        if ind < math.ceil(self.ntS*2/5):
            self.set_stage_control(s, ind, 0.9)
        else:
            self.set_stage_control(s, ind, 0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        t_arr = self.get_param_arrays_expanded(xi)
        time_grid = np.cumsum(np.concatenate(([0], t_arr))).reshape(-1)
        u = self.get_control_plot_arrays(xi)
        r,v,m = self.get_state_arrays_expanded(xi)
        
        fig, ax = plt.subplots(dpi=dpi)
        ax.plot(time_grid, (r - 1)*100, 'tab:blue', linestyle = ':', label = r'$(r-1)\cdot 100$')
        ax.plot(time_grid, v*20, 'tab:green', linestyle = '--', label = r'$v\cdot 20$')
        ax.plot(time_grid, m, 'tab:olive', linestyle = '-.', label = '$m$')
        
        
        ax.step(time_grid, u, 'tab:red', label = '$u$')
        ax.legend(fontsize = 'large')
        
        ax.set_xlabel('t', fontsize = 17.5)
        ax.xaxis.set_label_coords(1.015,-0.006)
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Goddard\'s rocket problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
            
        plt.show()
        plt.close()

#Goddard rocket with only state, control bounds and terminal constraints by adding the air friction as a differential state.
#Created by a fellow PhD.
class Goddard_Rocket_MOD(Goddard_Rocket):
    default_params = {
        'rT':1.01, 
        'b':7.0, 
        'A':310.0, 
        'k':500.0, 
        'Tmax':3.5, 
        'C':0.6, 
        'x_init':[1.0,0.0,1.0]
        }
    
    def build_problem(self):
        Tmax, A, b, k, rT, C = (self.model_params[key] for key in ('Tmax', 'A', 'b', 'k', 'rT', 'C'))
        
        self.set_OCP_data(4, 1, 1, 0, [1.0,0,0., -np.inf], [np.inf,np.inf,np.inf, C],[0],[np.inf],[0],[1])
        self.fix_initial_value(self.model_params['x_init'] + [0.])
        
        x = cs.MX.sym('x', self.nx)
        r,v,m,D = cs.vertsplit(x)
        r0,v0,m0,_ = cs.vertsplit(cs.DM(self.x_init))
        
        u = cs.MX.sym('u', self.nu)
        p = cs.MX.sym('p', self.np)
        dt = p
        
        ode_rhs = cs.vertcat(v,
                            -1/(r**2) + (1/m) * (Tmax*u - A*(v**2) * cs.exp(-k * (r - r0))),
                            -b*u,
                            2*A*v*cs.exp(-k*(r-r0))*(-1/(r**2) + (1/m) * (Tmax*u - A*(v**2) * cs.exp(-k * (r - r0)))) + A*v**2 * cs.exp(-k*(r-r0))*(-k)*(v)
                            )
        
        self.ODE = {'x': x, 'p':cs.vertcat(dt, u),'ode': dt*ode_rhs}
        self.multiple_shooting()
        
        r_eval = self.x_eval[0,:]
        
        term_alt_expr = r_eval[-1] - rT
        self.add_constraint(term_alt_expr, 0., np.inf)
        
        self.start_point = np.zeros(self.nVar)
        nt_acc = math.ceil(self.ntS*2/5)
        nt_dec = math.floor(self.ntS*3/5)
        for i in range(nt_acc):
            self.set_stage_control(self.start_point, i, [1.0])
            self.set_stage_param(self.start_point, i, [0.4/(b*0.4)/self.ntS])
        for i in range(nt_acc,nt_acc+nt_dec):
            self.set_stage_control(self.start_point, i, [0.0])
            self.set_stage_param(self.start_point, i, [0.4/(b*0.4)/self.ntS])
        self.integrate_full(self.start_point)
        
        
        self.set_objective(-self.x_eval[2,-1])
        self.build_NLP()
        
    def plot(self, xi, dpi = None, title = None, it = None):
        t_arr = self.get_param_arrays_expanded(xi)
        time_grid = np.cumsum(np.concatenate(([0], t_arr))).reshape(-1)
        u = self.get_control_plot_arrays(xi)
        r,v,m,_ = self.get_state_arrays_expanded(xi)
        
        plt.figure(dpi = dpi)
        plt.plot(time_grid, (r - 1)*100, 'b--', label = r'$(r-1)\cdot 100$')
        plt.plot(time_grid, v*20, 'g:', label = r'$v\cdot 20$')
        plt.plot(time_grid, m, 'y-.', label = '$m$')
        
        plt.step(time_grid, u, 'r', label = '$u$')
        plt.legend(fontsize = 'large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Goddard\'s rocket problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
            
        plt.show()
        plt.close()
        

class Calcium_Oscillation(OCProblem):
    default_params = {
    't0': 0,
    'tf': 22,
    'k1': 0.09,
    'k2': 2.30066,
    'k3': 0.64,
    'K4': 0.19,
    'k5': 4.88,
    'K6': 1.18,
    'k7': 2.08,
    'k8': 32.24,
    'K9': 29.09,
    'k10': 5.0,
    'K11': 2.67,
    'k12': 0.7,
    'k13': 13.58,
    'k14': 153.0,
    'K15': 0.16,
    'k16': 4.85,
    'K17': 0.05,
    'p1': 100,
    'tx0': 6.78677,
    'tx1': 22.65836,
    'tx2': 0.384306,
    'tx3': 0.28977
    }
    def __init__(self, nt = 100, refine = 1, integrator = 'cvodes', parallel = False, N_thread = 4, **kwargs):
        OCProblem.__init__(self, nt=nt, refine=refine, integrator=integrator, parallel=parallel, N_thread=4, **kwargs)
    
    def build_problem(self):
        self.set_OCP_data(4,1,1,1,[0,0,0,0],[np.inf,np.inf,np.inf,np.inf],[1.1], [1.3], [1], [np.inf])
        self.fix_initial_value([0.03966, 1.09799, 0.00142, 1.65431])
        x = cs.MX.sym('x', 4)
        x0,x1,x2,x3 = cs.vertsplit(x)
        w = cs.MX.sym('w')
        wmax = cs.MX.sym('wmax')
        dt = cs.MX.sym('dt')
        
        t0, tf, k1, k2, k3, K4, k5, K6, k7, k8, K9, k10, K11, k12, k13, k14, K15, k16, K17, p1, tx0, tx1, tx2, tx3 = (self.model_params[key] for key in ('t0', 'tf', 'k1', 'k2', 'k3', 'K4', 'k5', 'K6', 'k7', 'k8', 'K9', 'k10', 'K11', 'k12', 'k13', 'k14', 'K15', 'k16', 'K17', 'p1', 'tx0', 'tx1', 'tx2', 'tx3'))
        self.fix_time_horizon(t0,tf)
        
        ode_rhs = cs.vertcat(
            k1 + k2*x0 - (k3*x0*x1)/(x0 + K4) - (k5*x0*x2)/(x0 + K6),
            k7*x0 - (k8*x1)/(x1+K9),
            (k10*x1*x2*x3)/(x3 + K11) + k12*x1 + k13*x0 - (k14*x2)/((1 + w*(wmax-1.0))*x2 + K15) - (k16*x2)/(x2 + K17) + x3/10,
            -(k10*x1*x2*x3)/(x3 + K11) + (k16*x2)/(x2+K17) - x3/10
            )
        quad_expr = (x0 - tx0)**2 + (x1 - tx1)**2 +(x2 - tx2)**2 + (x3 - tx3)**2 + p1*w
        
        self.ODE = {'x':x, 'p':cs.vertcat(dt,wmax,w), 'ode':dt*ode_rhs, 'quad': dt*quad_expr}
        
        self.multiple_shooting()
        
        self.set_objective(self.q_tf)
        self.add_constraint(self.u_eval - self.p_exp_eval, -np.inf, 0)
        
        self.build_NLP()
        self.start_point = np.zeros(self.nVar)
        for i in range(self.ntS):
            self.set_stage_param(self.start_point, i, 1.3)
            self.set_stage_control(self.start_point, i, 1.0)
        
            
        self.integrate_full(self.start_point)
        
        for i in range(math.floor(0.4*self.ntS), self.ntS):
            self.set_stage_control(self.ub_var, i, 1.0)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        val = self.get_stage_control(s, ind)
        self.set_stage_control(s, ind, val + 0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x0, x1, x2, x3 = self.get_state_arrays(xi)
        w = self.get_control_plot_arrays(xi)        
        
        plt.figure(dpi = dpi)
        plt.plot(self.time_grid, x0, 'b-', label = 'x0')
        plt.plot(self.time_grid, x1, 'r-', label = 'x1')
        plt.plot(self.time_grid, x2, 'g-', label = 'x2')
        plt.plot(self.time_grid, x3, 'y-', label = 'x3')
        plt.step(self.time_grid, (w-1.0)*20, 'g', label = '(w-1)*20')
        
        plt.legend(fontsize='large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Calcium oscillation'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
            
        plt.show()
        plt.close()


class Batch_Reactor(OCProblem):
    default_params = {}
    def build_problem(self):
        self.set_OCP_data(2, 0, 1, 0, [-np.inf,-np.inf], [np.inf,np.inf], [], [], [298],[398])
        
        x = cs.MX.sym('x', 2)
        x1,x2 = cs.vertsplit(x)
        
        T = cs.MX.sym('T', 1)
        k1 = 4000*cs.exp(-2500/T)
        k2 = 620000*cs.exp(-5000/T)
        ode_rhs = cs.vertcat(-k1*x1**2, k1*x1**2 - k2*x2)
        self.fix_initial_value([1.0,0.0])
        self.fix_time_horizon(0,1)
        dt = cs.MX.sym('dt', 1)
        
        self.ODE = {'x':x, 'p': cs.vertcat(dt,T), 'ode':dt*ode_rhs}
        
        self.multiple_shooting()
        
        self.set_objective(-self.x_eval[1,-1])
        
        self.build_NLP()
        
        self.start_point = np.zeros(self.nVar)
        for i in range(self.ntS):
            self.set_stage_control(self.start_point, i, 298)
            # self.set_stage_state(self.start_point, i, self.x_init)
        # self.integrate_full(self.start_point)

    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, 300)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x1,x2 = self.get_state_arrays(xi)
        T = self.get_control_plot_arrays(xi)
        
        plt.figure(dpi = dpi)
        plt.plot(self.time_grid, x1, 'tab:green', linestyle = '--', label = r'$x_1$')
        plt.plot(self.time_grid, x2, 'tab:blue', linestyle = '-.', label = r'$x_2$')
        plt.step(self.time_grid_ref, (T-298)*0.05, 'tab:red', label = r'$(u-298)\cdot 0.05$')
        
        plt.legend(fontsize='large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Batch reactor'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
            
        plt.show()
        plt.close()
        
    
class Bioreactor(OCProblem):
    default_params = {
    'D': 0.15,
    'Ki': 22,
    'Km': 1.2,
    'Pm': 50,
    'Yxs': 0.4,
    'alpha': 2.2,
    'beta': 0.2,
    'mum': 0.48
    }
    def build_problem(self):
        self.set_OCP_data(3, 0, 1, 1,[0.,0.,0.],[np.inf,np.inf,np.inf],[],[],[28.7],[40.])
        self.fix_initial_value([6.5,12,22])
        self.fix_time_horizon(0,48)
        
        D, Ki, Km, Pm, Yxs, alpha, beta, mum = (self.model_params[key] for key in ['D', 'Ki', 'Km', 'Pm', 'Yxs', 'alpha', 'beta', 'mum'])
        
        x = cs.MX.sym('x', 3)
        X,S,P = cs.vertsplit(x)
        Sf = cs.MX.sym('Sf', 1)
        dt = cs.MX.sym('dt', 1)
        
        mu = mum*(1-P/Pm)*S/(Km + S + S**2/Ki) 
        ode_rhs = cs.vertcat(-D*X + mu*X,
                             D*(Sf - S) - (mu/Yxs)*X,
                             -D*P + (alpha*mu+beta)*X
                             )
        quad = D*(Sf-P)**2
        
        self.ODE = {'x':x, 'p':cs.vertcat(dt,Sf), 'ode': dt*ode_rhs, 'quad':dt*quad}
        
        self.multiple_shooting()
        self.set_objective(self.q_tf)
        
        self.build_NLP()
        
        self.start_point = np.zeros(self.nVar)
        for i in range(self.ntS):
            self.set_stage_state(self.start_point, i, self.x_init)
            self.set_stage_control(self.start_point, i, 28.7)
        self.set_stage_state(self.start_point, self.ntS, self.x_init)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, 30)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        plt.figure(dpi=dpi)
        
        X,S,P = self.get_state_arrays(xi)
        Sf = self.get_control_plot_arrays(xi)
        if title is not None:
            plt.title(title)
        
        plt.plot(self.time_grid, (X-5)*10, 'r-', label = r'$(X-5)\cdot 10$')
        plt.plot(self.time_grid, (S-10)*10, 'c-', label = r'$(S-10)\cdot 10$')
        plt.plot(self.time_grid, (P-20)*5, 'b-', label = r'$(P-20)\cdot 5$')
        
        
        plt.step(self.time_grid, Sf, 'g', label = 'Sf')
        
        plt.legend(fontsize = 'large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Bioreactor'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
            
        plt.show()
        plt.close()


class Hanging_Chain(OCProblem):
    default_params = {'a':1, 'b':3, 'Lp': 4}
    def build_problem(self):
        self.set_OCP_data(1,0,1,2,[0.], [10.], [], [], [-10.], [20.])
        
        a,b,Lp = (self.model_params[key] for key in ['a', 'b', 'Lp'])
        self.fix_initial_value([a])
        self.fix_time_horizon(0,1)
        
        x1 = cs.MX.sym('x1',1)
        u = cs.MX.sym('u',1)
        dt = cs.MX.sym('dt',1)
        ode_rhs = u
        quad = cs.vertcat(x1*(1.0+u**2)**0.5, (1.0+u**2)**0.5)
        self.ODE = {'x':x1, 'p':cs.vertcat(dt,u), 'ode':dt*ode_rhs, 'quad':dt*quad}
        self.multiple_shooting()
        self.set_objective(self.q_tf[0])
        self.add_constraint(self.q_tf[1] - Lp, 0., 0.)
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
            self.set_stage_state(self.start_point, i, [x1_start[i]])
        self.set_stage_state(self.start_point, self.ntS, x1_start[self.ntS])
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        val = self.get_stage_control(s, ind)
        self.set_stage_control(s, ind, val + 0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x = self.get_state_arrays(xi)
        # u = self.get_control_plot_arrays(xi)
        
        fig, ax = plt.subplots(dpi=dpi)
        ax.plot(self.time_grid, x, 'k-', label = 'chain')
        # plt.plot(self.time_grid_ref, u*0.1, 'g-', label = 'u*0.1')
        ax.legend(fontsize='large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Hanging chain problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
            
        plt.show()
        plt.close()


class Hanging_Chain_MAYER(Hanging_Chain):
    default_params = {'a':1, 'b':3, 'Lp': 4}
    def build_problem(self):
        self.set_OCP_data(3,0,1,0,[0.], [10.], [], [], [-10., -np.inf, -np.inf], [20., np.inf, np.inf])
        
        a,b,Lp = (self.model_params[key] for key in ['a', 'b', 'Lp'])
        self.fix_initial_value([a, 0, 0])
        self.fix_time_horizon(0,1)
        
        x_ = cs.MX.sym('x_', 3)
        x1, _ , _ = cs.vertsplit(x_)
        u = cs.MX.sym('u',1)
        dt = cs.MX.sym('dt',1)
        # ode_rhs = u
        ode_rhs = cs.vertcat(u, x1*(1.0+u**2)**0.5, (1.0+u**2)**0.5)
        self.ODE = {'x':x_, 'p':cs.vertcat(dt,u), 'ode':dt*ode_rhs}
        self.multiple_shooting()
        self.set_objective(self.x_eval[1, -1])
        self.add_constraint(self.x_eval[2, -1] - Lp, 0., 0.)
        self.add_constraint(self.x_eval[0, -1] - b, 0.,0.)
        
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
        
        x2_start = x1_start[1:]*u_start[:]
        x3_start = u_start[:]
        self.set_stage_control(self.start_point, 0, u_start[0])
        for i in range(1,self.ntS):
            self.set_stage_control(self.start_point, i, u_start[i])
            self.set_stage_state(self.start_point, i, [x1_start[i], x2_start[i-1], x3_start[i-1]])
        self.set_stage_state(self.start_point, self.ntS, [x1_start[self.ntS], x2_start[self.ntS - 1], x3_start[self.ntS - 1]])
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        val = self.get_stage_control(s, ind)
        self.set_stage_control(s, ind, val + 0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x, _, _ = self.get_state_arrays(xi)
        # u = self.get_control_plot_arrays(xi)
        
        plt.figure(dpi = dpi)
        plt.plot(self.time_grid, x, 'r-', label = 'chain')
        # plt.plot(self.time_grid_ref, u*0.1, 'g-', label = 'u*0.1')
        plt.legend(fontsize='large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Hanging chain problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
            
        plt.show()
        plt.close()



class Catalyst_Mixing(OCProblem):
    
    def build_problem(self):
        self.set_OCP_data(2,0,1,0,[-np.inf,-np.inf],[np.inf,np.inf],[],[],[0.],[1.])
        self.fix_time_horizon(0,1)
        self.fix_initial_value([1.,0.])
        
        x = cs.MX.sym('x', 2)
        x1,x2 = cs.vertsplit(x)
        w = cs.MX.sym('w',1)
        dt = cs.MX.sym('dt', 1)
        ode_rhs = cs.vertcat(w*(10*x2-x1), w*(x1 - 10*x2) - (1-w)*x2)
        
        self.ODE = {'x':x, 'p':cs.vertcat(dt,w), 'ode': dt*ode_rhs}
        self.multiple_shooting()
        
        self.set_objective((-1 + self.x_eval[0,-1] + self.x_eval[1,-1]))
        
        self.build_NLP()
        
        for j in range(self.ntS+1):
            self.set_stage_state(self.start_point, j, self.x_init)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, 0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        fig, ax = plt.subplots(dpi=dpi)
        x1,x2 = self.get_state_arrays(xi)
        u = self.get_control_plot_arrays(xi)
        ax.plot(self.time_grid, x1, 'tab:green', linestyle='-.', label = r'$x_1$')
        ax.plot(self.time_grid, x2, 'tab:blue', linestyle='--', label = r'$x_2$')
        ax.step(self.time_grid, u, 'tab:red', linestyle='-', label = r'$u$')
        ax.legend(fontsize = 'large')
        
        ax.set_xlabel('t', fontsize = 17.5)
        ax.xaxis.set_label_coords(1.015,-0.006)
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Catalyst mixing'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(copy.deepcopy(ttl))
        else:
            plt.title('')
        
        plt.show()
        plt.close()
        

class Cushioned_Oscillation(OCProblem):
    default_params = {'m':5.,'c':10.,'x0':2.,'v0':5.,'umm':5.}
    def build_problem(self):
        m,c,x0,v0,umm = (self.model_params[key] for key in ['m', 'c', 'x0', 'v0', 'umm'])
        self.set_OCP_data(2,1,1,0,[-np.inf,-np.inf], [np.inf,np.inf], [8/self.ntS],[20/self.ntS], [-umm], [umm])
        
        X = cs.MX.sym('X',2)
        x,v = cs.vertsplit(X)
        u = cs.MX.sym('u',1)
        p = cs.MX.sym('p')
        dt = p
        self.fix_initial_value([x0,v0])
        
        ode_rhs = cs.vertcat(v, 1/m * (u - c*x))
        self.ODE = {'x':X, 'p':cs.vertcat(p,u), 'ode':dt*ode_rhs}
        self.multiple_shooting()
        self.set_objective(self.ntS*self.p_tf)
        self.add_constraint(cs.vec(self.x_eval[:,-1] - cs.DM([0.,0.])),[0.,0.],[0.,0.])
        
        self.build_NLP()
        self.set_stage_param(self.start_point, 0, 10/self.ntS)
        for i in range(1,self.ntS):
            self.set_stage_state(self.start_point, i, self.x_init)
            self.set_stage_param(self.start_point, i, 10/self.ntS)
        self.set_stage_state(self.start_point, self.ntS, self.x_init)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, 0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x,v = self.get_state_arrays(xi)
        u = self.get_control_plot_arrays(xi)
        p = self.get_param_arrays(xi)
        time_grid = np.cumsum(np.concatenate([[0], p.reshape(-1)]))
        
        fig, ax = plt.subplots(dpi=dpi)
        ax.plot(time_grid, x, 'tab:blue', linestyle = '--', label = r'$x$')
        ax.plot(time_grid, v, 'tab:green', linestyle = '-.', label = r'$v$')
        ax.step(time_grid, u, 'tab:red', label = r'$u$')
        ax.legend(loc='upper right', fontsize = 'large')
        
        ax.set_xlabel('t', fontsize = 17.5)
        ax.xaxis.set_label_coords(1.015,-0.006)
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'cushioned oscillation problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
        plt.show()
        plt.close()
        
        
class Cushioned_Oscillation_TSCALE(Cushioned_Oscillation):
    default_params = {'m':5.,'c':10.,'x0':2.,'v0':5.,'umm':5., 'TSCALE':100.0}
    def build_problem(self):
        m,c,x0,v0,umm,TSCALE = (self.model_params[key] for key in ['m', 'c', 'x0', 'v0', 'umm', 'TSCALE'])
        self.set_OCP_data(2,1,1,0,[-np.inf,-np.inf], [np.inf,np.inf], [8/self.ntS * TSCALE],[20/self.ntS * TSCALE], [-umm], [umm])
        
        X = cs.MX.sym('X',2)
        x,v = cs.vertsplit(X)
        u = cs.MX.sym('u',1)
        p = cs.MX.sym('p')
        dt_ = p
        dt = dt_/TSCALE
        self.fix_initial_value([x0,v0])
        
        ode_rhs = cs.vertcat(v, 1/m * (u - c*x))
        self.ODE = {'x':X, 'p':cs.vertcat(p,u), 'ode':dt*ode_rhs}
        self.multiple_shooting()
        self.set_objective(self.ntS*self.p_tf/TSCALE)
        self.add_constraint(cs.vec(self.x_eval[:,-1] - cs.DM([0.,0.])),[0.,0.],[0.,0.])
        
        self.build_NLP()
        self.set_stage_param(self.start_point, 0, TSCALE*10.0/self.ntS)
        for i in range(1,self.ntS):
            self.set_stage_state(self.start_point, i, self.x_init)
            self.set_stage_param(self.start_point, i, TSCALE*10.0/self.ntS)
        self.set_stage_state(self.start_point, self.ntS, self.x_init)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, 0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        TSCALE = self.model_params['TSCALE']
        x,v = self.get_state_arrays(xi)
        u = self.get_control_plot_arrays(xi)
        p = self.get_param_arrays(xi)
        time_grid = np.cumsum(np.concatenate([[0], p.reshape(-1)]))/TSCALE
        
        plt.figure(dpi = dpi)
        plt.plot(time_grid, x, 'tab:blue', linestyle = '--', label = 'x')
        plt.plot(time_grid, v, 'tab:green', linestyle = '-.', label = 'v')
        plt.step(time_grid, u, 'tab:red', label = 'u')
        plt.legend(loc='upper right')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'cushioned oscillation problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
        plt.show()
        plt.close()
        
#Cvodes recommended
class D_Onofrio_Chemotherapy(OCProblem):
    default_params = {
        'zeta':0.192, 
        'b':5.85, 
        'mu': 0.0, 
        'd':0.00873, 
        'G':0.15, 
        'x20':0.0, 
        'x30':0.0, 
        'u0max':75., 
        'x2max':300., 
        'x00':12000., 
        'x10':15000., 
        'u1max':1., 
        'x3max':2., 
        'F':1., 
        'eta':1., 
        'alpha':0., 
        'duration': 6.
    }
    
    param_set_1 = {
        'x00': 12000,
        'x10': 15000,
        'u1max': 1,
        'x3max': 2
    }
    
    param_set_2 = {
        'x00': 12000,
        'x10': 15000,
        'u1max': 2,
        'x3max': 10
    }
    
    param_set_3 = {
        'x00': 14000,
        'x10': 5000,
        'u1max': 1,
        'x3max': 2
    }
    
    param_set_4 = {
        'x00': 14000,
        'x10': 5000,
        'u1max': 2,
        'x3max': 10
    }
    
    def __init__(self, nt = 100, refine = 1, integrator = 'cvodes', parallel = True, N_threads = 4, **kwargs):
        OCProblem.__init__(self,nt=nt,refine=refine,integrator=integrator,parallel=parallel,**kwargs)
    
    def build_problem(self):
        zeta, b, mu, d, G, x20, x30, u0max, x2max, x00, x10, u1max, x3max, F, eta, alpha = (self.model_params[key] for key in ('zeta','b','mu','d','G','x20','x30','u0max','x2max','x00','x10','u1max','x3max','F','eta', 'alpha'))
        self.set_OCP_data(2,0,2,3,[0.1,0.1], [np.inf,np.inf], [], [], [0.,0.],[u0max,u1max])
        #Note: Lower bounds 0.1,0.1 for differential states required as integrations fails at 0, 0 due to numerical errors causing negative states
        self.fix_initial_value([x00,x10])
        self.fix_time_horizon(0., self.model_params['duration'])
        
        x = cs.MX.sym('x', 2)
        x0,x1 = cs.vertsplit(x)
        u = cs.MX.sym('u', 2)
        u0,u1 = cs.vertsplit(u)
        dt = cs.MX.sym('dt')
        
        ode_rhs = cs.vertcat(-zeta*x0*cs.log(x0/x1) - F*x0*u1,
                             b*x0 - mu*x1 - d*x0**(2./3.)*x1 - G*u0*x1 - eta*x1*u1,
                             )
        quad = cs.vertcat(u0**2,u0,u1)
        self.ODE = {'x':x, 'p':cs.vertcat(dt,u), 'ode': dt*ode_rhs, 'quad':dt*quad}
        self.multiple_shooting()
        self.set_objective(self.x_eval[0,-1] + alpha*self.q_tf[0])
        self.add_constraint(self.q_tf[1:3] - cs.DM([x2max,x3max]), [-np.inf,-np.inf], [0.,0.])
        self.build_NLP()
        self.integrate_full(self.start_point)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, [0.1, 0.1])
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        time_grid = self.time_grid
        x0,x1 = self.get_state_arrays(xi)
        u0,u1 = self.get_control_plot_arrays(xi)
        
        plt.figure(dpi = dpi)
        plt.plot(time_grid, x0/100., 'r-', label = 'x0/100')
        plt.plot(time_grid, x1/100., 'g-', label = 'x1/100')
        plt.step(time_grid, u0, 'y-', label = 'u0')
        plt.step(time_grid, u1*75, 'c-', label = 'u1*75')
        plt.legend(fontsize='large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'D\'Onofrio chemotherapy problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
            
        plt.show()   
        plt.close()
    
class D_Onofrio_Chemotherapy_VT(OCProblem):
    default_params = {'zeta':0.192, 'b':5.85, 'mu': 0.0, 'd':0.00873, 'G':0.15, 'x20':0.0, 'x30':0.0, 'u0max':75., 'x2max':300., 'x00':12000., 'x10':15000., 'u1max':1., 'x3max':2., 'F':1., 'eta':1., 'alpha':0.}
    param_set_1 = {
        'x00': 12000,
        'x10': 15000,
        'u1max': 1,
        'x3max': 2
    }
    
    param_set_2 = {
        'x00': 12000,
        'x10': 15000,
        'u1max': 2,
        'x3max': 10
    }
    
    param_set_3 = {
        'x00': 14000,
        'x10': 5000,
        'u1max': 1,
        'x3max': 2
    }
    
    param_set_4 = {
        'x00': 14000,
        'x10': 5000,
        'u1max': 2,
        'x3max': 10
    }
    
    def __init__(self, nt = 20, refine = 1, integrator = 'rk4', parallel = True, N_threads = 4, **kwargs):
        OCProblem.__init__(self,nt=nt,refine=refine,integrator=integrator,parallel=parallel,**kwargs)

    
    def build_problem(self):
        zeta, b, mu, d, G, x20, x30, u0max, x2max, x00, x10, u1max, x3max, F, eta, alpha = (self.model_params[key] for key in ('zeta','b','mu','d','G','x20','x30','u0max','x2max','x00','x10','u1max','x3max','F','eta', 'alpha'))
        self.set_OCP_data(4,1,2,1,[0.1,0.1,0.,0.], [np.inf,np.inf,x2max,x3max], [4/self.ntS], [20/self.ntS], [0.,0.],[u0max,u1max])
        #Note: Lower bounds 0.1,0.1 for differential states required as integrations fails at 0, 0 due to numerical errors causing negative states
        self.fix_initial_value([x00,x10,x20,x30])
        # self.fix_time_horizon(0, 6.0)
        
        x = cs.MX.sym('x', 4)
        x0,x1,x2,x3 = cs.vertsplit(x)
        u = cs.MX.sym('u', 2)
        u0,u1 = cs.vertsplit(u)
        p = cs.MX.sym('p', 1)
        dt = p
        # dt = cs.MX.sym('dt')
        
        ode_rhs = cs.vertcat(-zeta*x0*cs.log(x0/x1) - F*x0*u1,
                             b*x0 - mu*x1 - d*x0**(2./3.)*x1 - G*u0*x1 - eta*x1*u1,
                             u0,
                             u1
                             )
        quad = u0**2
        self.ODE = {'x':x, 'p':cs.vertcat(dt,u), 'ode': dt*ode_rhs, 'quad':dt*quad}
        self.multiple_shooting()
        self.set_objective(20*self.p_tf*self.ntS+self.x_eval[0,-1] + alpha*self.q_tf)
        self.build_NLP()
        self.set_stage_param(self.start_point, 0, 4/self.ntS)
        for i in range(1,self.ntS):
            self.set_stage_param(self.start_point, i, 6/self.ntS)
        self.integrate_full(self.start_point)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, [0.1,0.1])
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        p = self.get_param_arrays(xi)
        time_grid = np.cumsum(np.concatenate([[0], p]))
        # time_grid = self.time_grid
        x0,x1,x2,x3 = self.get_state_arrays(xi)
        u0,u1 = self.get_control_plot_arrays(xi)
        
        plt.figure(dpi = dpi)
        plt.plot(time_grid, x0/100., 'r-', label = 'x0/100')
        plt.plot(time_grid, x1/100., 'g-', label = 'x1/100')
        plt.plot(time_grid, x2, 'b-', label = 'x2')
        plt.plot(time_grid, x3, 'c-', label = 'x3')
        plt.step(time_grid, u0, 'r-', label = 'u0')
        plt.step(time_grid, u1*75, 'g-', label = 'u1*75')
        plt.legend(fontsize='large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'D\'Onofrio chemotherapy problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
            
        plt.show()  
        plt.close()
             
        

class Egerstedt_Standard(OCProblem):
    
    def build_problem(self):
        self.set_OCP_data(2,0,3,1, [-np.inf, 0.4], [np.inf,np.inf], [], [], [0.,0.,0.], [1.,1.,1.])
        self.fix_time_horizon(0.,1.)
        self.fix_initial_value([0.5,0.5])
        
        x = cs.MX.sym('x', 2)
        x1,x2 = cs.vertsplit(x)
        w = cs.MX.sym('w', 3)
        w1,w2,w3 = cs.vertsplit(w)
        dt = cs.MX.sym('dt')
        
        ode_rhs = cs.vertcat(-x1*w1 + (x1+x2)*w2 + (x1-x2)*w3,
                             (x1+2*x2)*w1 + (x1 - 2*x2)*w2 + (x1 + x2)*w3
                             )
        quad = x1**2 + x2**2
        
        self.ODE = {'x':x, 'p':cs.vertcat(dt,w), 'ode': dt*ode_rhs, 'quad': dt*quad}
        self.multiple_shooting()
        self.set_objective(self.q_tf)
        self.add_constraint(cs.sum1(self.u_eval), 1.0, 1.0)
        
        self.build_NLP()
        
        for i in range(self.ntS):
            #Usually runs into the good local optimum f(x_opt) ~ 0.989
            self.set_stage_control(self.start_point, i, [1/3]*3)
            
            #Likely to run into the bad local optimum f(x_opt) ~ 1.1054
            # self.set_stage_control(self.start_point, i, [0.5,0.5,0.])
            
        for i in range(1, self.ntS + 1):
            self.set_stage_state(self.start_point, i, self.x_init)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, [0.5, 0.25, 0.25])
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x1,x2 = self.get_state_arrays(xi)
        w1,w2,w3 = self.get_control_plot_arrays(xi)
        
        plt.figure(dpi = dpi)
        plt.plot(self.time_grid, x1, 'tab:green', linestyle = '--', label = r'$x_1$')
        plt.plot(self.time_grid, x2, 'tab:blue', linestyle = '--', label = r'$x_2$')
        plt.step(self.time_grid, w1, 'tab:red', label = r'$w_1$')
        plt.step(self.time_grid, w2, 'tab:olive', linestyle = '-.', label = r'$w_2$')
        plt.step(self.time_grid, w3, 'tab:cyan', linestyle = ':', label = r'$w_3$')
        plt.legend(loc = 'center left', fontsize = 'large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Egerstedt standard problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
            
        plt.show()
        plt.close()


class Egerstedt_Standard_MAYER(Egerstedt_Standard):
    
    def build_problem(self):
        self.set_OCP_data(3,0,3,0, [-np.inf, 0.4, -np.inf], [np.inf,np.inf,np.inf], [], [], [0.,0.,0.], [1.,1.,1.])
        self.fix_time_horizon(0.,1.)
        self.fix_initial_value([0.5,0.5,0.])
        
        x = cs.MX.sym('x', 3)
        x1,x2, q = cs.vertsplit(x)
        w = cs.MX.sym('w', 3)
        w1,w2,w3 = cs.vertsplit(w)
        dt = cs.MX.sym('dt')
        
        ode_rhs = cs.vertcat(-x1*w1 + (x1+x2)*w2 + (x1-x2)*w3,
                             (x1+2*x2)*w1 + (x1 - 2*x2)*w2 + (x1 + x2)*w3,
                             x1**2 + x2**2
                             )
        
        self.ODE = {'x':x, 'p':cs.vertcat(dt,w), 'ode': dt*ode_rhs}
        self.multiple_shooting()
        self.set_objective(self.x_eval[2,-1])
        self.add_constraint(cs.sum1(self.u_eval)-1.0, 0., 0.)
        
        self.build_NLP()
        
        self.set_stage_control(self.start_point, 0, [1/3]*3)
        for i in range(1,self.ntS):
            self.set_stage_control(self.start_point, i, [1/3]*3)
            self.set_stage_state(self.start_point, i, self.x_init)
        self.set_stage_state(self.start_point, self.ntS, self.x_init)
        self.set_stage_control(self.lb_var, 10, [0., 0., 0.4])
        self.set_stage_control(self.lb_var, 11, [0., 0., 0.4])
        self.set_stage_control(self.lb_var, 12, [0., 0., 0.4])
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, [0.5, 0.25, 0.25])
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x1,x2,_ = self.get_state_arrays(xi)
        w1,w2,w3 = self.get_control_plot_arrays(xi)
        
        plt.figure(dpi = dpi)
        plt.plot(self.time_grid, x1, 'c-', label = r'$x_1$')
        plt.plot(self.time_grid, x2, 'y-', label = r'$x_2$')
        plt.step(self.time_grid, w1, 'r-', label = r'$w_1$')
        plt.step(self.time_grid, w2, 'b-', label = r'$w_2$')
        plt.step(self.time_grid, w3, 'g-', label = r'$w_3$')
        plt.legend(loc = 'center left', fontsize = 'large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Egerstedt standard problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
        plt.show()
        plt.close()


class Fullers(OCProblem):
    
    def build_problem(self):
        self.set_OCP_data(2, 0, 1, 1, [-np.inf,-np.inf], [np.inf,np.inf], [], [], [0.], [1.])
        self.fix_initial_value([0.01, 0.])
        self.fix_time_horizon(0.,1.)
        
        x = cs.MX.sym('x', 2)
        x0,x1 = cs.vertsplit(x)
        u = cs.MX.sym('w')
        dt = cs.MX.sym('dt')
        
        ode_rhs = cs.vertcat(x1, 1-2*u)
        quad = x0**2
        self.ODE = {'x':x, 'p':cs.vertcat(dt,u), 'ode':dt*ode_rhs, 'quad':dt*quad}
        self.multiple_shooting()
        self.set_objective(self.q_tf)
        self.add_constraint(self.x_eval[0,-1]-0.01, 0.,0.)
        self.build_NLP()
        
        self.set_stage_control(self.start_point, 0, 0.5)
        for i in range(1,self.ntS):
            self.set_stage_control(self.start_point, i, 0.5)
            self.set_stage_state(self.start_point, i, self.x_init)
        self.set_stage_state(self.start_point, self.ntS, self.x_init)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        val = self.get_stage_control(s, ind)
        self.set_stage_control(s, ind, val - 0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x1,x2 = self.get_state_arrays(xi)
        u = self.get_control_plot_arrays(xi)
        
        plt.figure(dpi = dpi)
        plt.plot(self.time_grid, x1*20, 'r-', label = r'$x_1\cdot 20$')
        plt.plot(self.time_grid, x2, 'b-', label = r'$x_2 $')
        plt.step(self.time_grid, u, 'g-', label = r'$u$')
        plt.legend(loc = 'right', fontsize = 'large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Fuller\'s problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
            
        plt.show()
        plt.close()


class Electric_Car(OCProblem):
    default_params = {
    "Kr": 10,
    "rho": 1.293,
    "Cx": 0.4,
    "S": 2,
    "r": 0.33,
    "Kf": 0.03,
    "Km": 0.27,
    "Rm": 0.03,
    "Lm": 0.05,
    "M": 250,
    "g": 9.81,
    "Valim": 150,
    "Rbat": 0.05
    }
    
    def build_problem(self):
        self.set_OCP_data(3,0,1,1,[-150,-np.inf,-np.inf], [150,np.inf,np.inf], [], [], [-1.], [1.])
        self.fix_time_horizon(0.,10.)
        self.fix_initial_value([0.,0.,0.])
        
        x = cs.MX.sym('x', 3)
        x0,x1,x2 = cs.vertsplit(x)
        u = cs.MX.sym('u')
        dt = cs.MX.sym('dt')
        
        Kr, rho, Cx, S, r, Kf, Km, Rm, Lm, M, g, Valim, Rbat = (self.model_params[key] for key in ['Kr', 'rho', 'Cx', 'S', 'r', 'Kf', 'Km', 'Rm', 'Lm', 'M', 'g', 'Valim', 'Rbat'])
        
        ode_rhs = cs.vertcat((Valim*u - Rm*x0-Km*x1)/Lm,
                             (Kr**2)/(M*r**2) * (Km*x0 - r/Kr*(M*g*Kf + 0.5*rho*S*Cx*r**2/Kr**2 * x1**2)),
                             r/Kr * x1
                             )
        quad = Valim*u*x0 + Rbat*x0**2
        
        self.ODE = {'x':x, 'p':cs.vertcat(dt,u), 'ode': dt*ode_rhs, 'quad':dt*quad}
        self.multiple_shooting()
        self.set_objective(self.q_tf)
        self.add_constraint(self.x_eval[2,-1] - 100., 0., 0.)
        self.build_NLP()
        
        for i in range(self.ntS):
            self.set_stage_control(self.start_point, i, 0.1 + 0.9*i/self.ntS)
        self.integrate_full(self.start_point)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        val = self.get_stage_control(s, ind)
        self.set_stage_control(s, ind, val - 0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x0,x1,x2 = self.get_state_arrays(xi)
        u = self.get_control_plot_arrays(xi)
        
        fig, ax = plt.subplots(dpi = dpi)
        ax.plot(self.time_grid, x0, 'tab:olive', linestyle='--', label = r'$x_0$')
        ax.plot(self.time_grid, x1, 'tab:green', linestyle='-.', label = r'$x_1$')
        ax.plot(self.time_grid, x2, 'tab:blue', linestyle=':', label = r'$x_2$')
        ax.step(self.time_grid_ref, u*100, 'tab:red', label = r'$u\cdot 100$')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Electric car problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
        
        ax.legend(fontsize='large')
        
        ax.set_xlabel('t', fontsize = 17.5)
        ax.xaxis.set_label_coords(1.015,-0.006)
        
        plt.show()
        plt.close()
        

class F8_Aircraft(OCProblem):
    
    def build_problem(self):
        self.set_OCP_data(3,1,1,0,[-np.inf,-np.inf,-np.inf], [np.inf,np.inf,np.inf], [1/self.ntS], [100/self.ntS], [-0.05236], [0.05236])
        self.fix_initial_value([0.4655,0.,0.])
        
        x = cs.MX.sym('x', 3)
        x0,x1,x2 = cs.vertsplit(x)
        w = cs.MX.sym('w')
        dt = cs.MX.sym('dt')
        
        ode_rhs = cs.vertcat(-0.877*x0 + x2 - 0.088*x0*x2 + 0.47*x0**2 - 0.019*x1**2 - x0**2*x2 + 3.846*x0**3 - 0.215*w + 0.28*x0**2*w + 0.47*x0*w**2 + 0.63*w**3,
                             x2,
                             -4.208*x0 - 0.396*x2 - 0.47*x0**2 - 3.564*x0**3 - 20.967*w + 6.265*x0**2*w + 46.*x0*w**2 + 61.4*w**3
                             )
        self.ODE = {'x':x, 'p':cs.vertcat(dt,w), 'ode':dt*ode_rhs}
        self.multiple_shooting()
        self.set_objective(self.p_tf*self.ntS)
        self.add_constraint(self.x_eval[:,-1] - cs.DM([0.,0.,0.]), 0., 0.)
        self.build_NLP()
        
        for i in range(0, self.ntS):
            self.set_stage_state(self.start_point, i, self.x_init)
            self.set_stage_param(self.start_point, i, 5./self.ntS)
        self.set_stage_state(self.start_point, self.ntS, self.x_init)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, 0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x0,x1,x2 = self.get_state_arrays(xi)
        w = self.get_control_plot_arrays(xi)
        p = self.get_param_arrays(xi)
        time_grid = np.cumsum(np.concatenate([[0.], p]))
        
        plt.figure(dpi = dpi)
        plt.plot(time_grid, x0, 'r-', label = r'$x_0$')
        plt.plot(time_grid, x1, 'g-', label = r'$x_1$')
        plt.plot(time_grid, x2, 'b-', label = r'$x_2$')
        plt.step(time_grid, w*20, 'y-', label = r'$w\cdot20$')
        plt.legend(fontsize='large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'F8 aircraft problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
        
        plt.show()
        plt.close()


#HINT: Try solving for hT = 70.0 or nt = 50 first
#      Cvodes recommended
class Gravity_Turn(OCProblem):
    default_params = {
    "m0": 11.3,
    "m1": 1.3,
    "Isp": 300,
    "Fmax": 0.6,
    "cd": 0.021,
    "A": 1.,
    "g0": 9.81e-3,
    "r0": 600.0,
    "H": 5.6,
    "rho0": 1.2230948554874,
    "betaT": 3.141592653589793 / 2,
    "vT": 2.287,
    "hT": 75.,
    "Tmin": 120.,
    "Tmax": 600.,
    "eps": 1e-6
    }
    
    def __init__(self, nt = 100, refine = 1, integrator = 'cvodes', parallel = False, N_threads = 4, **kwargs):
        OCProblem.__init__(self, nt=nt, refine=refine, integrator=integrator, parallel=parallel, N_thread=4, **kwargs)

    def build_problem(self):
        m0, m1, Isp, Fmax, cd, A, g0, r0, H, rho0, betaT, vT, hT, Tmin, Tmax, eps= (self.model_params[key] for key in ['m0', 'm1', 'Isp', 'Fmax', 'cd', 'A', 'g0', 'r0', 'H', 'rho0', 'betaT', 'vT', 'hT', 'Tmin', 'Tmax', 'eps'])
        self.set_OCP_data(4,1,1,1,[m1, eps, 0, 0], [m0, np.inf, np.pi/2., np.inf], [Tmin/self.ntS], [Tmax/self.ntS * 0.66], [0.], [1.])
        self.fix_initial_value([m0, eps, None, 0.])
        # self.fix_initial_value([m0, eps, 5e-6, 0.])
        
        x = cs.MX.sym('x', 4)
        m,v,beta,h = cs.vertsplit(x)
        dt = cs.MX.sym('dt')
        u = cs.MX.sym('u')
        r = r0 + h
        
        ode_rhs = cs.vertcat(-Fmax/(Isp*g0) * u,
                             (Fmax*u - 0.5e3*A*cd*rho0*cs.exp(-h/H)*v**2)/m - g0*(r0/r)**2 * cs.cos(beta),
                             g0*(r0/r)**2 * cs.sin(beta)/v - v * cs.sin(beta)/r,
                             v*cs.cos(beta)
                             )
        quad = v*cs.sin(beta)/r
        
        self.ODE = {'x':x, 'p':cs.vertcat(dt,u), 'ode':dt*ode_rhs, 'quad':dt*quad}
        self.multiple_shooting()
        
        self.set_objective(m0 - self.x_eval[0,-1])
        self.add_constraint(cs.cumsum(self.q_eval), 0., np.inf)
        self.add_constraint(self.x_eval[1:4,-1] - cs.DM([vT, betaT, hT]), 0., 0.)
        
        self.build_NLP()
        
        for i in range(self.ntS):
            self.set_stage_param(self.start_point, i, 125./self.ntS)
        self.set_stage_state(self.start_point, 0, [5e-6])
        for i in range(math.floor(self.ntS*0.5)):
            self.set_stage_control(self.start_point, i, 0.8)
        self.integrate_full(self.start_point)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        val = self.get_stage_control(s, ind)
        self.set_stage_control(s, ind, val + 0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        m,v,beta,h = self.get_state_arrays(xi)
        u = self.get_control_plot_arrays(xi)
        p = self.get_param_arrays(xi)
        p_exp = self.get_param_arrays_expanded(xi)
        time_grid = np.cumsum(np.concatenate([[0.],p]))
        time_grid_ref = np.cumsum(np.concatenate([[0.],p_exp]))
        
        plt.figure(dpi = dpi)
        plt.plot(time_grid, m, 'g-', label = 'm')
        plt.plot(time_grid, v*10, 'b-', label = 'v*10')
        plt.plot(time_grid, beta*20, 'r-', label = 'beta*20')
        plt.plot(time_grid, h, 'y-', label = 'h')
        plt.step(time_grid_ref, u*20, 'c-', label = 'u*20')
        plt.legend(fontsize='large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Gravity turn maneuver problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
        plt.show()
        plt.close()


class Oil_Shale_Pyrolysis(OCProblem):
    default_params = {'b1':20.3,
                      'b2':37.4,
                      'b3':33.8,
                      'b4':28.2,
                      'b5':31.0,
                      'a1':np.exp(8.86),
                      'a2':np.exp(24.25),
                      'a3':np.exp(23.67),
                      'a4':np.exp(18.75),
                      'a5':np.exp(20.7)
                      }
    def build_problem(self):
        self.set_OCP_data(4,1,1,0, [0.,0.,0.,0.], [np.inf,np.inf,np.inf,np.inf], [0.1/self.ntS], [20./self.ntS], [698.15], [748.15])
        self.fix_initial_value([1.,0.,0.,0.])
        
        x = cs.MX.sym('x', 4)
        x1,x2,x3,x4 = cs.vertsplit(x)
        T = cs.MX.sym('T')
        dt = cs.MX.sym('dt')
        
        a1,a2,a3,a4,a5,b1,b2,b3,b4,b5 = (self.model_params[key] for key in ['a1', 'a2', 'a3', 'a4', 'a5', 'b1', 'b2', 'b3', 'b4', 'b5'])
        k1,k2,k3,k4,k5 = (ai*cs.exp(-bi/(1.9858775e-3 * T)) for ai,bi in zip([a1,a2,a3,a4,a5], [b1,b2,b3,b4,b5]))
        ode_rhs = cs.vertcat(-k1*x1 - (k3+k4+k5)*x1*x2,
                             k1*x1 - k2*x2 + k3*x1*x2,
                             k2*x2 + k4*x1*x2,
                             k5*x1*x2
                             )
        
        self.ODE = {'x':x, 'p':cs.vertcat(dt,T), 'ode':dt*ode_rhs}
        self.multiple_shooting()
        
        self.set_objective(-self.x_eval[1,-1])
        self.build_NLP()
        
        for i in range(self.ntS):
            self.set_stage_param(self.start_point, i, 20./self.ntS)
            self.set_stage_state(self.start_point, i, self.x_init)
            self.set_stage_control(self.start_point, i, 698.15)
        self.set_stage_state(self.start_point, self.ntS, self.x_init)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, 710)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x1,x2,x3,x4 = self.get_state_arrays(xi)
        T = self.get_control_plot_arrays(xi)
        p = self.get_param_arrays(xi)
        p_exp = self.get_param_arrays_expanded(xi)
        time_grid = np.cumsum(np.concatenate([[0.],p]))
        time_grid_ref = np.cumsum(np.concatenate([[0.],p_exp]))
        
        plt.figure(dpi = dpi)
        plt.plot(time_grid, x1, 'b-', label = 'x1')
        plt.plot(time_grid, x2, 'g-', label = 'x2')
        plt.plot(time_grid, x3, 'r-', label = 'x3')
        plt.plot(time_grid, x4, 'c-', label = 'x4')
        
        plt.step(time_grid_ref, (T - 698.15)/50, 'r', label = '(T-698.15)/50')
        plt.legend(fontsize='large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Oil shale pyrolysis problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
        
        plt.show()
        plt.close()


class Particle_Steering(OCProblem):
    default_params = {'a': 100}
    def build_problem(self):
        self.set_OCP_data(4, 1, 1, 0, [-np.inf]*4, [np.inf]*4, [0.01/self.ntS], [100/self.ntS], [-np.pi/2], [np.pi/2])
        self.fix_initial_value([0.,0.,0.,0.])
        
        a = self.model_params['a']
        x = cs.MX.sym('x',4)
        x1,x2,y1,y2 = cs.vertsplit(x)
        u = cs.MX.sym('u')
        dt = cs.MX.sym('dt')
        
        ode_rhs = cs.vertcat(y1,
                             y2,
                             a*cs.cos(u),
                             a*cs.sin(u),
                             )
        
        self.ODE = {'x':x, 'p':cs.vertcat(dt,u), 'ode': dt*ode_rhs}
        self.multiple_shooting()
        self.set_objective(self.p_tf[0]*self.ntS)
        self.add_constraint(self.x_eval[1,-1] - 5, 0., 0.)
        self.add_constraint(self.x_eval[2:4,-1] - cs.DM([45,0]), 0.,0.)
        self.build_NLP()
        
        for i in range(self.ntS):
            self.set_stage_param(self.start_point, i, 1/self.ntS)

    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, 0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x1,x2,y1,y2 = self.get_state_arrays(xi)
        u = self.get_control_plot_arrays(xi)
        p = self.get_param_arrays(xi)
        p_exp = self.get_param_arrays_expanded(xi)
        time_grid = np.cumsum(np.concatenate([[0.],p]))
        time_grid_ref = np.cumsum(np.concatenate([[0.],p_exp]))
        
        plt.figure(dpi = dpi)
        plt.plot(time_grid, x1, 'tab:green', linestyle = '--', label = r'$x_1$')
        plt.plot(time_grid, x2, 'tab:blue', linestyle = '-.', label = r'$x_2$')
        # plt.plot(time_grid, y1, 'tab:green', linestyle = '-.', label = r'$v_1$')
        # plt.plot(time_grid, y2, 'tab:blue', linestyle = '-.', label = r'$v_2$')
        plt.step(time_grid_ref, u*10, 'tab:red', label = r'$u\cdot 10$')
        plt.legend(fontsize='large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Particle steering problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
        
        plt.show()
        plt.close()
    
class Quadrotor_Helicopter(OCProblem):
    default_params = {'g':9.8,'M':1.3,'L':0.305,'I':0.0605}
    def build_problem(self):
        self.set_OCP_data(6, 0, 4, 1, [-np.inf,-np.inf,0.,-np.inf,-np.inf,-np.inf], [np.inf]*6, [], [], [0.,0.,0.,0.], [1.,1.,1.,0.001])
        self.fix_time_horizon(0,7.5)
        self.fix_initial_value([0.,0.,1.,0.,0.,0.])
        
        g,M,L,I = (self.model_params[key] for key in ['g', 'M', 'L', 'I'])
        
        x = cs.MX.sym('x', 6)
        x1,x2,x3,x4,x5,x6 = cs.vertsplit(x)
        U = cs.MX.sym('u', 4)
        w1,w2,w3,u = cs.vertsplit(U)
        dt = cs.MX.sym('dt')
        
        ode_rhs = cs.vertcat(x2,
                             g*cs.sin(x5) + w1*u*cs.sin(x5)/M,
                             x4,
                             g*cs.cos(x5) - g + w1*u*cs.cos(x5)/M,
                             x6,
                             -w2*L*u/I + w3*L*u/I
                             )
        quad = 5*u**2
        self.ODE = {'x':x, 'p': cs.vertcat(dt, U), 'ode': dt*ode_rhs, 'quad': dt*quad}
        self.multiple_shooting()
        
        x1tf,_,x3tf,_,x5tf,_ = cs.vertsplit(self.x_eval[:,-1])
        self.set_objective(5*(x1tf - 6)**2 + 5*(x3tf - 1)**2 + (cs.sin(x5tf)*0.5)**2 + self.q_tf)
        self.add_constraint(cs.sum1(self.u_eval[0:3,:]), 1., 1.)
        self.build_NLP()
        
        for i in range(self.ntS):
            self.set_stage_control(self.start_point, i, [1/2,0.,1/2, 0.001])
        self.integrate_full(self.start_point)

    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, [9/20, 0.1, 9/20, 0.001])
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x1,_,x3,_,x5,_ = self.get_state_arrays(xi)
        w1,w2,w3,u = self.get_control_plot_arrays(xi)
        
        plt.figure(dpi = dpi)
        plt.plot(self.time_grid, x1, 'tab:blue', linestyle = '--', label = r'$x_1$')
        plt.plot(self.time_grid, x3*3, 'tab:green', linestyle = '-.', label = r'$x_3\cdot 3$')
        plt.plot(self.time_grid, x5*20, 'tab:olive', linestyle = ':', label = r'$x_5\cdot 20$')
        
        plt.step(self.time_grid, w1, 'tab:red', linestyle = '--', label = r'$w_1$')
        plt.step(self.time_grid, w2, 'tab:green', label = r'$w_2$')
        plt.step(self.time_grid, w3, 'tab:blue', linestyle = '-.', label = r'$w_3$')
        plt.step(self.time_grid, u*2000, 'tab:cyan', linestyle = ':', label = r'$u\cdot 2000$')
        plt.legend(fontsize='large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Quadrotor helicopter problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
        
        plt.show()
        plt.close()
    

class Supermarket_Refrigeration(OCProblem):
    default_params = {
    "Qairload": 3000.00,
    "mrefconst": 0.20,
    "Mgoods": 200.00,
    "Cpgoods": 1000.00,
    "UAgoodsair": 300.00,
    "Mwall": 260.00,
    "Cpwall": 385.00,
    "UAairwall": 500.00,
    "Mair": 50.00,
    "Cpair": 1000.00,
    "UAwallrefmax": 4000.00,
    "taufill": 40.00,
    "TSH": 10.00,
    "Mrefmax": 1.00,
    "Vsuc": 5.00,
    "Vsl": 0.08,
    "etavol": 0.81
    }
    
    def __init__(self, nt = 100, refine = 1, integrator = 'cvodes', parallel = False, N_threads = 4, **kwargs):
        OCProblem.__init__(self, nt=nt, refine=refine, integrator=integrator, parallel=parallel, N_thread=4, **kwargs)

    def build_problem(self):
        self.set_OCP_data(10, 1, 3, 0, [0., -5.,-20.,2.0,0.01, -5.,-20.0, 2.0, 0.01] + [-np.inf], [1.7, 10.,0.,5.0,20.,10.,0.,5.0,20.] + [np.inf], [650./self.ntS], [750./self.ntS], [0.,0.,0.], [1.,1.,1.])
        self.fix_initial_value([None]*9 + [0.])
        
        Qairload, mrefconst, Mgoods, Cpgoods, UAgoodsair, Mwall, Cpwall, UAairwall, Mair, Cpair, UAwallrefmax, taufill, TSH, Mrefmax, Vsuc, Vsl, etavol = (self.model_params[key] for key in ['Qairload', 'mrefconst', 'Mgoods', 'Cpgoods', 'UAgoodsair', 'Mwall', 'Cpwall', 'UAairwall', 'Mair', 'Cpair', 'UAwallrefmax', 'taufill', 'TSH', 'Mrefmax', 'Vsuc', 'Vsl', 'etavol'])
        
        x = cs.MX.sym('x', 10)
        u = cs.MX.sym('u', 3)
        dt = cs.MX.sym('dt')
        
        x0,x1,x2,x3,x4,x5,x6,x7,x8,_ = cs.vertsplit(x)
        u0,u1,u2 = cs.vertsplit(u)
        
        Te = -4.3544 * x0**2 + 29.224 * x0 - 51.2005
        Deltahlg = (0.0217 * x0**2 - 0.1704 * x0 + 2.2988) * 10**5
        rhosuc = 4.6073 * x0 + 0.3798
        drhosucdPsuc = -0.0329 * x0**3 + 0.2161 * x0**2 - 0.4742 * x0 + 5.4817
        f = (0.0265 * x0**3 - 0.4346 * x0**2 + 2.4923 * x0 + 1.2189) * 10**5
        
        ode_rhs = cs.vertcat(
            1/(Vsuc * drhosucdPsuc) * ( (UAwallrefmax/(Mrefmax*Deltahlg)) * (x4*(x2 - Te) + x8*(x6 - Te)) + mrefconst - etavol*Vsl*0.5*u2*rhosuc),
            -(UAgoodsair*(x1 - x3))/(Mgoods * Cpgoods),
            (UAairwall*(x3 - x2) - UAwallrefmax/Mrefmax * x4 * (x2 - Te))/(Mwall * Cpwall),
            (UAgoodsair*(x1 - x3) + Qairload - UAairwall*(x3 - x2))/(Mair * Cpair),
            (Mrefmax - x4)/taufill * u0 - (UAwallrefmax/(Mrefmax*Deltahlg)) * x4 * (x2 - Te) * (1 - u0),
            -(UAgoodsair*(x5 - x7))/(Mgoods * Cpgoods),
            (UAairwall*(x7 - x6) - UAwallrefmax/Mrefmax * x8 * (x6 - Te))/(Mwall * Cpwall),
            (UAgoodsair*(x5 - x7) + Qairload - UAairwall*(x7 - x6))/(Mair * Cpair),
            (Mrefmax - x8)/taufill * u1 - (UAwallrefmax/(Mrefmax*Deltahlg)) * x8 * (x6 - Te) * (1 - u1),
            u2*0.5*etavol*Vsl*f
        )
        
        self.ODE = {'x':x, 'p':cs.vertcat(dt, u), 'ode': dt*ode_rhs}
        self.multiple_shooting()
        self.set_objective(self.x_eval[9,-1]/(self.p_tf*self.ntS))
        self.add_constraint(self.x_eval[0:9,0] - self.x_eval[0:9,-1], 0.,0.)
        self.build_NLP()
        
        self.set_stage_state(self.start_point, 0, [1.] + [2.]*2 + [0.2] + [2.]*5)
        for i in range(self.ntS):
            self.set_stage_control(self.start_point, i, [1.,1.,1.])
            self.set_stage_param(self.start_point, i, [650/self.ntS])
        self.integrate_full(self.start_point)
        
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, [0.9]*3)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x0,x1,x2,x3,x4,x5,x6,x7,x8,_ = self.get_state_arrays(xi)
        u0,u1,u2 = self.get_control_plot_arrays(xi)
        p = self.get_param_arrays(xi)
        p_exp = self.get_param_arrays_expanded(xi)
        time_grid = np.cumsum(np.concatenate([[0.],p]))
        time_grid_ref = np.cumsum(np.concatenate([[0.],p_exp]))
        
        plt.figure(dpi = dpi)
        for val, clr, lbl in zip([x0,x1,x2,x3,x4,x5,x6,x7,x8], ['y-','c-','m-','r--','g--','b--','r.','g.','b.'], ['x0','x1','x2','x3','x4','x5','x6','x7','x8']):
            plt.plot(time_grid, val, clr, label = lbl)
        plt.step(time_grid_ref, u0, 'r', label = 'u0')
        plt.step(time_grid_ref, u1, 'g', label = 'u1')
        plt.step(time_grid_ref, u2, 'b', label = 'u2')
        plt.legend(fontsize='large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Supermarket refrigeration problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
        
        plt.show()
        plt.close()
        
    
class Three_Tank_Multimode(OCProblem):
    default_params = {'T':12, 'c1':1, 'c2':2, 'c3':0.8, 'k1':2, 'k2':3, 'k3':1, 'k4':3}
    def build_problem(self):
        self.set_OCP_data(3,0,3,1,[0.,0.,0.], [np.inf,np.inf,np.inf], [],[], [0.,0.,0.], [1.,1.,1.])
        self.fix_time_horizon(0, self.model_params['T'])
        self.fix_initial_value([2.,2.,2.])
        
        c1, c2, c3, k1, k2, k3, k4 = (self.model_params[key] for key in ['c1', 'c2', 'c3', 'k1', 'k2', 'k3', 'k4'])
        
        x = cs.MX.sym('x',3)
        x1,x2,x3 = cs.vertsplit(x)
        u = cs.MX.sym('u',3)
        w1,w2,w3 = cs.vertsplit(u)
        dt = cs.MX.sym('dt')
        
        ode_rhs = cs.vertcat(-cs.sqrt(x1) + c1*w1+c2*w2 - w3*cs.sqrt(c3*x3),
                              cs.sqrt(x1) - cs.sqrt(x2),
                              cs.sqrt(x2) - cs.sqrt(x3) + w3*cs.sqrt(c3*x3)
                              )
        
        quad = k1*(x2-k2)**2 + k3*(x3-k4)**2
        self.ODE = {'x':x, 'p':cs.vertcat(dt,u), 'ode':dt*ode_rhs, 'quad':dt*quad}
        self.multiple_shooting()
        self.set_objective(self.q_tf)
        self.add_constraint(cs.sum1(self.u_eval),1.,1.)
        self.build_NLP()
        for i in range(self.ntS):
            self.set_stage_control(self.start_point, i, [1/3,1/3,1/3])
            self.set_stage_state(self.start_point, i, self.x_init)
        self.set_stage_state(self.start_point, self.ntS, self.x_init)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, [0.5, 0.25, 0.25])
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x1,x2,x3 = self.get_state_arrays(xi)
        w1,w2,w3 = self.get_control_plot_arrays(xi)
        
        fig, ax = plt.subplots(dpi=dpi)
        ax.plot(self.time_grid, x1, 'tab:olive', linestyle='--', label = r'$x_1$')#, self.time_grid[:,-1], x1, '--', self.time_grid[:,-1], u, 'o')
        ax.plot(self.time_grid, x2, 'tab:purple', linestyle='-.', label = r'$x_2$')
        ax.plot(self.time_grid, x3, 'tab:cyan', linestyle=':', label = r'$x_3$')
        ax.step(self.time_grid_ref, w1, 'tab:olive', linestyle='-', label = r'$w_1$')
        ax.step(self.time_grid_ref, w2, 'tab:red', linestyle='-', label = r'$w_2$')
        ax.step(self.time_grid_ref, w3, 'grey', label = r'$w_3$')
        ax.legend(prop={'size': 13.4}, loc = 'upper right')
        
        ax.set_xlabel('t', fontsize = 17.5)
        ax.xaxis.set_label_coords(1.015,-0.006)
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Three tank problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
            
        plt.show()
        plt.close()


class Three_Tank_Multimode_MAYER(Three_Tank_Multimode):
    default_params = {'T':12, 'c1':1, 'c2':2, 'c3':0.8, 'k1':2, 'k2':3, 'k3':1, 'k4':3}
    def build_problem(self):
        self.set_OCP_data(4,0,3,0,[0.,0.,0.,-np.inf], [np.inf,np.inf,np.inf,np.inf], [],[], [0.,0.,0.], [1.,1.,1.])
        self.fix_time_horizon(0, self.model_params['T'])
        self.fix_initial_value([2.,2.,2.,0.])
        
        c1, c2, c3, k1, k2, k3, k4 = (self.model_params[key] for key in ['c1', 'c2', 'c3', 'k1', 'k2', 'k3', 'k4'])
        
        x = cs.MX.sym('x',4)
        x1,x2,x3,q = cs.vertsplit(x)
        u = cs.MX.sym('u',3)
        w1,w2,w3 = cs.vertsplit(u)
        dt = cs.MX.sym('dt')
        
        ode_rhs = cs.vertcat(-cs.sqrt(x1) + c1*w1+c2*w2 - w3*cs.sqrt(c3*x3),
                              cs.sqrt(x1) - cs.sqrt(x2),
                              cs.sqrt(x2) - cs.sqrt(x3) + w3*cs.sqrt(c3*x3),
                              k1*(x2-k2)**2 + k3*(x3-k4)**2
                              )
        
        self.ODE = {'x':x, 'p':cs.vertcat(dt,u), 'ode':dt*ode_rhs}
        self.multiple_shooting()
        self.set_objective(self.x_eval[3,-1])
        self.add_constraint(cs.sum1(self.u_eval),1.,1.)
        self.build_NLP()
        for i in range(self.ntS):
            self.set_stage_control(self.start_point, i, [1/3,1/3,1/3])
            self.set_stage_state(self.start_point, i, self.x_init)
        self.set_stage_state(self.start_point, self.ntS, self.x_init)

    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, [0.5, 0.25, 0.25])
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x1,x2,x3,_ = self.get_state_arrays(xi)
        w1,w2,w3 = self.get_control_plot_arrays(xi)
        
        plt.figure(dpi = dpi)
        plt.plot(self.time_grid, x1, 'y-', label = r'$x_1$')#, self.time_grid[:,-1], x1, '--', self.time_grid[:,-1], u, 'o')
        plt.plot(self.time_grid, x2, 'm-', label = r'$x_2$')
        plt.plot(self.time_grid, x3, 'c-', label = r'$x_3$')
        plt.step(self.time_grid_ref, w1, 'g', label = r'$w_1$')
        plt.step(self.time_grid_ref, w2, 'r', label = r'$w_2$')
        plt.step(self.time_grid_ref, w3, 'b', label = r'$w_3$')
        plt.legend(fontsize = 'large', loc = 'upper right')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Three tank problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
            
        plt.show()
        plt.close()


class Time_Optimal_Car(OCProblem):
    default_params = {'vmax':33.}
    def build_problem(self):
        self.set_OCP_data(2,1,1,0,[0.,0.],[330.,self.model_params['vmax']],[0.1/self.ntS], [500/self.ntS], [-2.], [1.])
        self.fix_initial_value([0.,0.])
        
        x = cs.MX.sym('x',2)
        z1,z2 = cs.vertsplit(x)
        u = cs.MX.sym('u')
        dt = cs.MX.sym('dt')
        
        ode_rhs = cs.vertcat(z2,u)
        self.ODE = {'x':x, 'p':cs.vertcat(dt,u), 'ode':dt*ode_rhs}
        self.multiple_shooting()
        self.set_objective(self.p_tf*self.ntS)
        self.add_constraint(self.x_eval[:,-1] - cs.DM([300,0]),0.,0.)
        self.build_NLP()
        for i in range(self.ntS):
            self.set_stage_param(self.start_point, i, 10/self.ntS)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, 0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        z1,z2 = self.get_state_arrays(xi)
        u = self.get_control_plot_arrays(xi)
        p = self.get_param_arrays(xi)
        p_exp = self.get_param_arrays_expanded(xi)
        time_grid = np.cumsum(np.concatenate([[0.],p]))
        time_grid_ref = np.cumsum(np.concatenate([[0.],p_exp]))
        
        plt.figure(dpi = dpi)
        plt.plot(time_grid, z1, 'tab:blue', linestyle = '--', label = r'$z_1$')
        plt.plot(time_grid, z2*5, 'tab:green', linestyle = '-.', label = r'$z_2\cdot5$')
        plt.step(time_grid_ref, u*20, 'tab:red', label = r'$u\cdot20$')
        plt.legend(fontsize='large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Time optimal car problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
            
        plt.show()
        plt.close()
        
class Van_der_Pol_Oscillator(OCProblem):
    def __init__(self, nt = 100, refine = 1, integrator = 'cvodes', parallel = False, N_threads = 4, **kwargs):
        OCProblem.__init__(self, nt=nt, refine=refine, integrator=integrator, parallel=parallel, N_thread=4, **kwargs)

    default_params = dict()
    def build_problem(self):
        self.set_OCP_data(2,0,1,1,[-10.,-10.], [10.,10.], [], [], [-np.inf], [0.75])
        self.fix_time_horizon(0,20)
        self.fix_initial_value([1.,0.])
        X = cs.MX.sym('X',2)
        x,y = cs.vertsplit(X)
        u = cs.MX.sym('u')
        dt = cs.MX.sym('dt')
        ode_rhs = cs.vertcat(y,u*(1-x**2)*y - x)
        quad = x**2 + y**2+u**2
        self.ODE = {'x':X, 'p':cs.vertcat(dt,u), 'ode':dt*ode_rhs, 'quad':dt*quad}
        self.multiple_shooting()
        self.set_objective(self.q_tf)
        self.build_NLP()
        for i in range(self.ntS+1):
            self.set_stage_state(self.start_point, i, self.x_init)
        
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, 0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x,y = self.get_state_arrays(xi)
        u = self.get_control_plot_arrays(xi)
        
        plt.figure(dpi = dpi)
        plt.plot(self.time_grid, x, 'g-', label = 'x')
        plt.plot(self.time_grid, y, 'b-', label = 'y')
        plt.step(self.time_grid_ref, u, 'r', label = 'u')
        plt.legend(fontsize='large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Van der Pol problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
            
        plt.show()
        plt.close()

class Van_der_Pol_Oscillator_2(OCProblem):
    def __init__(self, nt = 100, refine = 1, integrator = 'cvodes', parallel = False, N_threads = 4, **kwargs):
        OCProblem.__init__(self, nt=nt, refine=refine, integrator=integrator, parallel=parallel, N_thread=4, **kwargs)

    default_params = dict()
    def build_problem(self):
        self.set_OCP_data(2,0,1,1,[-10.,-10.], [10.,10.], [], [], [-np.inf], [0.75])
        self.fix_time_horizon(0,20)
        self.fix_initial_value([1.,0.])
        X = cs.MX.sym('X',2)
        x,y = cs.vertsplit(X)
        u = cs.MX.sym('u')
        dt = cs.MX.sym('dt')
        ode_rhs = cs.vertcat(y,(1-x**2)*y - x + u)
        quad = x**2 + y**2 + u**2
        self.ODE = {'x':X, 'p':cs.vertcat(dt,u), 'ode':dt*ode_rhs, 'quad':dt*quad}
        self.multiple_shooting()
        self.set_objective(self.q_tf)
        self.build_NLP()
        for i in range(self.ntS+1):
            self.set_stage_state(self.start_point, i, self.x_init)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, 0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x,y = self.get_state_arrays(xi)
        u = self.get_control_plot_arrays(xi)
        
        plt.figure(dpi = dpi)
        plt.plot(self.time_grid, x, 'g-', label = 'x')
        plt.plot(self.time_grid, y, 'b-', label = 'y')
        plt.step(self.time_grid_ref, u, 'r', label = 'u')
        plt.legend(fontsize='large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Van der Pol problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
            
        plt.show()    
        plt.close()

class Van_der_Pol_Oscillator_3(OCProblem):
    def __init__(self, nt = 100, refine = 1, integrator = 'cvodes', parallel = False, N_threads = 4, **kwargs):
        OCProblem.__init__(self, nt=nt, refine=refine, integrator=integrator, parallel=parallel, N_thread=4, **kwargs)

    default_params = dict()
    def build_problem(self):
        self.set_OCP_data(2,0,1,1,[-0.25,-0.25], [10.,10.], [], [], [-1.], [1.])
        self.fix_time_horizon(0,10)
        self.fix_initial_value([1.,0.])
        X = cs.MX.sym('X',2)
        x,y = cs.vertsplit(X)
        u = cs.MX.sym('u')
        dt = cs.MX.sym('dt')
        ode_rhs = cs.vertcat(y,(1-x**2)*y - x + u)
        quad = x**2 + y**2 + u**2
        self.ODE = {'x':X, 'p':cs.vertcat(dt,u), 'ode':dt*ode_rhs, 'quad':dt*quad}
        self.multiple_shooting()
        self.set_objective(self.q_tf)
        self.build_NLP()
        for i in range(self.ntS):
            self.set_stage_state(self.start_point, i, self.x_init)
            self.set_stage_control(self.start_point, i, [0.])
        self.set_stage_state(self.start_point, self.ntS, self.x_init)
        # self.integrate_full(self.start_point)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        val = self.get_stage_control(s, ind)
        self.set_stage_control(s, ind, val - 0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x,y = self.get_state_arrays(xi)
        u = self.get_control_plot_arrays(xi)
        
        plt.figure(dpi = dpi)
        plt.plot(self.time_grid, x, 'g-', label = 'x')
        plt.plot(self.time_grid, y, 'b-', label = 'y')
        plt.step(self.time_grid_ref, u, 'r', label = 'u')
        plt.legend(fontsize='large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Van der Pol problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
            
        plt.show()    
        plt.close()
        


class Van_der_Pol_Oscillator_3_MAYER(OCProblem):
    def __init__(self, nt = 100, refine = 1, integrator = 'cvodes', parallel = False, N_threads = 4, **kwargs):
        OCProblem.__init__(self, nt=nt, refine=refine, integrator=integrator, parallel=parallel, N_thread=4, **kwargs)

    default_params = dict()
    def build_problem(self):
        self.set_OCP_data(3,0,1,0,[-0.25,-0.25, -np.inf], [10.,10., np.inf], [], [], [-1.], [1.])
        self.fix_time_horizon(0,10)
        self.fix_initial_value([1.,0., 0.])
        X = cs.MX.sym('X',3)
        x,y,q = cs.vertsplit(X)
        u = cs.MX.sym('u')
        dt = cs.MX.sym('dt')
        ode_rhs = cs.vertcat(y,(1-x**2)*y - x + u, x**2 + y**2 + u**2)
        self.ODE = {'x':X, 'p':cs.vertcat(dt,u), 'ode':dt*ode_rhs}
        self.multiple_shooting()
        self.set_objective(self.x_eval[2,-1])
        self.build_NLP()
        for i in range(self.ntS):
            self.set_stage_state(self.start_point, i, self.x_init)
            self.set_stage_control(self.start_point, i, [0.5])
        self.set_stage_state(self.start_point, self.ntS, self.x_init)
        self.integrate_full(self.start_point)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        val = self.get_stage_control(s, ind)
        self.set_stage_control(s, ind, val - 0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x,y,_ = self.get_state_arrays(xi)
        u = self.get_control_plot_arrays(xi)
        
        plt.figure(dpi = dpi)
        plt.plot(self.time_grid, x, 'g-', label = 'x')
        plt.plot(self.time_grid, y, 'b-', label = 'y')
        plt.step(self.time_grid_ref, u, 'r', label = 'u')
        plt.legend(fontsize='large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Van der Pol problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
            
        plt.show()   
        plt.close()


# TODO check problem formulation
class Ocean(OCProblem):
    default_params = {
        'rho':0.03,
        'gamma':0.001,
        'omega':0.1,
        'b':50.,
        'mu':0.5,
        'a1':2.,
        'a2':2.,
        'nu':1.,
        'c1':50.,
        'c2':0.004,
        'Spreind':600.,
        'S0':2000.,
        'R0':1e4,
        'DL0':2.3e4
        }
    def __init__(self, nt = 100, refine = 1, integrator = 'cvodes', parallel = False, N_threads = 4, **kwargs):
        OCProblem.__init__(self, nt=nt, refine=refine, integrator=integrator, parallel=parallel, N_thread=4, **kwargs)

    
    def build_problem(self):
        self.set_OCP_data(3,0,2,1,[0.,0.,0.],[1e5,1e5,np.inf],[],[],[0.,0.],[40.,40.])
        
        rho, gamma, omega, b, mu, a1, a2, nu, c1, c2, Spreind, S0, R0, DL0 = (self.model_params[key] for key in ['rho', 'gamma', 'omega', 'b', 'mu', 'a1', 'a2', 'nu', 'c1', 'c2', 'Spreind', 'S0', 'R0', 'DL0'])
        self.fix_time_horizon(0.,400.)
        self.fix_initial_value([S0,R0,0.])
        
        x = cs.MX.sym('x',3)
        S,R,t = cs.vertsplit(x)
        u = cs.MX.sym('u',2)
        u1,u2 = cs.vertsplit(u)
        dt = cs.MX.sym('dt')
        
        U = b*u1 - mu*u1**2
        A = a1*u2 + a2*u2**2
        C = c1 - c2*R
        D = nu*(0.3*S-Spreind)**2
        DL = DL0 + R0 + S0 - R - S
        
        ode_rhs = cs.vertcat(u1 - u2 - gamma*(S - omega*DL), -u1, cs.DM(1.))
        
        quad = cs.exp(-rho*t)*(U - A - u1*C - D)
        self.ODE = {'x':x, 'p':cs.vertcat(dt,u), 'ode':dt*ode_rhs, 'quad':dt*quad}
        self.multiple_shooting()
        self.set_objective(-self.q_tf)
        self.build_NLP()
        
        for i in range(self.ntS):
            self.set_stage_control(self.start_point, i, [30.,10.])
            self.set_stage_state(self.start_point, i, self.x_init)
        self.set_stage_state(self.start_point, self.ntS, self.x_init)
        
    def plot(self, xi, dpi = None, title = None, it = None):
        S,R,_ = self.get_state_arrays(xi)
        u1,u2 = self.get_control_plot_arrays(xi)
        
        plt.figure(dpi = dpi)
        plt.plot(self.time_grid, S-2000, 'r-', label = 'S-2000')
        plt.plot(self.time_grid, R/1000, 'g-', label = 'R/1000')
        plt.step(self.time_grid_ref, u1, 'b', label = 'u1')
        plt.step(self.time_grid_ref, u2, 'c', label = 'u2')
        plt.legend(fontsize='large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Ocean problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
            
        plt.show()


class Lotka_OED(OCProblem):
    default_params = {'tf':12, 'p1':1,'p2':1,'p3':1,'p4':1,'p5':0.4, 'p6':0.2, 'x_init':[0.5,0.7], 'M':4.0, 'fishing':True, 'epsilon': 0.0}
    def build_problem(self):
        self.set_OCP_data(9, 0, 3, 2, [0.,0.]+[-np.inf]*7, [np.inf]*9,[],[],[0.] + [0.]*2, [float(self.model_params['fishing'])] + [1.]*2)
        tf,p1,p2,p3,p4,p5,p6,x_init,M,epsilon= (self.model_params[key] for key in ['tf', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6','x_init', 'M', 'epsilon'])
        self.fix_time_horizon(0.,tf)
        self.fix_initial_value(x_init + [0.]*4 + [epsilon, 0., epsilon])
        
        S = cs.MX.sym('S', 9)
        x1, x2, G11, G12, G21, G22, F11, F12, F22 = cs.vertsplit(S)
        
        C = cs.MX.sym('C', 3)
        u, w1, w2 = cs.vertsplit(C)
        
        dt = cs.MX.sym('dt', 1)
        ode_rhs = cs.vertcat(
                p1*x1 - p2*x1*x2 - p5*u*x1,
                -p3*x2 + p4*x1*x2 - p6*u*x2,
                (p1 - p2*x2 - p5*u)*G11 + (-p2*x1)*G21 - x1*x2,
                (p1 - p2*x2 - p5*u)*G12 + (-p2*x1)*G22,
                (p4*x2)*G11 + (-p3 + p4*x1 - p6*u)*G21,
                (p4*x2)*G12 + (-p3 + p4*x1 - p6*u)*G22  + x1*x2,
                w1*(G11**2) + w2*(G21**2),
                w1*G11*G12 + w2*G21*G22,
                w1*(G12**2) + w2*(G22**2)
        )
        quad_expr = cs.vertcat(w1, w2)
        self.ODE = {'x': S, 'p':cs.vertcat(dt, C),'ode': dt*ode_rhs, 'quad': dt*quad_expr}
        self.multiple_shooting()
        F11T,F12T,F22T = cs.vertsplit(self.x_eval[6:9,-1])
        self.set_objective((1/(F11T*F22T - F12T*F12T))*(F22T + F11T))
        self.add_constraint(self.q_tf - M, -np.inf, 0.)
        self.build_NLP()
        for i in range(self.ntS):
            self.set_stage_control(self.start_point, i, [0.,1/3,1/3])
        self.integrate_full(self.start_point)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, 0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        u,w1,w2 = self.get_control_plot_arrays(xi)
        x1, x2, G11, G12, G21, G22, F11, F12, F22 = self.get_state_arrays(xi)
        
        fig, ax = plt.subplots(dpi=dpi)
        ax.plot(self.time_grid, x1, 'tab:olive', linestyle='-.', label = r'$x_1$')
        ax.plot(self.time_grid, x2, 'tab:cyan', linestyle='-.', label = r'$x_2$')
        ax.step(self.time_grid_ref, u, 'tab:red', linestyle='-', label = r'$u$')
        ax.step(self.time_grid_ref, w1, 'tab:blue', linestyle=':', label = r'$w_1$')
        ax.step(self.time_grid_ref, w2, 'tab:green', linestyle='--', label = r'$w_2$')
        
        ax.set_ylim(0.,4.)
        ax.legend(fontsize = 'large', loc = 'upper left')
        ax.set_xlabel('t', fontsize = 17.5)
        ax.xaxis.set_label_coords(1.015,-0.006)
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Lotka OED problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
        plt.show()
        plt.close()
        
        
    def plot_sensitivities(self, xi, dpi=None, title=None, it=None):
        _, _, G11, G12, G21, G22, F11, F12, F22 = self.get_state_arrays(xi)
        
        plt.figure(dpi = dpi)
        plt.plot(self.time_grid, G11, 'y-', label = r'$G_{11}(t)$')
        plt.plot(self.time_grid, G12, 'c-', label = r'$G_{12}(t)$')
        plt.plot(self.time_grid, G21, 'r-', label = r'$G_{21}(t)$')
        plt.plot(self.time_grid, G22, 'b-', label = r'$G_{22}(t)$')
        
        plt.ylim(-9.,6.)
        plt.legend(fontsize = 'medium', loc = 'upper left')
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Lotka OED problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
        plt.show()
        plt.close()


class Lotka_OED_USCALE(OCProblem):
    default_params = {'tf':12, 'p1':1,'p2':1,'p3':1,'p4':1,'p5':0.4, 'p6':0.2, 'x_init':[0.5,0.7], 'M':4.0, 'fishing':True, 'epsilon': 0.0, 'USCALE': 0.1}
    def build_problem(self):
        tf,p1,p2,p3,p4,p5,p6,x_init,M,epsilon,USCALE= (self.model_params[key] for key in ['tf', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6','x_init', 'M', 'epsilon', 'USCALE'])
        self.set_OCP_data(9, 0, 3, 2, [0.,0.]+[-np.inf]*7, [np.inf]*9,[],[],[0.] + [0.]*2, [float(self.model_params['fishing'])*USCALE] + [1.]*2)
        self.fix_time_horizon(0.,tf)
        self.fix_initial_value(x_init + [0.]*4 + [epsilon, 0., epsilon])
        
        S = cs.MX.sym('S', 9)
        x1, x2, G11, G12, G21, G22, F11, F12, F22 = cs.vertsplit(S)
        #(Measurement -) Controls C
        C = cs.MX.sym('C', 3)
        u_, w1, w2 = cs.vertsplit(C)
        u = u_/USCALE
        # C_init = cs.DM([0., 1/3, 1/3])
        
        dt = cs.MX.sym('dt', 1)
        ode_rhs = cs.vertcat(
                p1*x1 - p2*x1*x2 - p5*u*x1,
                -p3*x2 + p4*x1*x2 - p6*u*x2,
                (p1 - p2*x2 - p5*u)*G11 + (-p2*x1)*G21 - x1*x2,
                (p1 - p2*x2 - p5*u)*G12 + (-p2*x1)*G22,
                (p4*x2)*G11 + (-p3 + p4*x1 - p6*u)*G21,
                (p4*x2)*G12 + (-p3 + p4*x1 - p6*u)*G22  + x1*x2,
                w1*(G11**2) + w2*(G21**2),
                w1*G11*G12 + w2*G21*G22,
                w1*(G12**2) + w2*(G22**2)
        )
        quad_expr = cs.vertcat(w1, w2)
        self.ODE = {'x': S, 'p':cs.vertcat(dt, C),'ode': dt*ode_rhs, 'quad': dt*quad_expr}
        self.multiple_shooting()
        F11T,F12T,F22T = cs.vertsplit(self.x_eval[6:9,-1])
        self.set_objective((1/(F11T*F22T - F12T*F12T))*(F22T + F11T))
        self.add_constraint(self.q_tf - M, -np.inf, 0.)
        self.build_NLP()
        for i in range(self.ntS):
            self.set_stage_control(self.start_point, i, [0.*USCALE,1/3,1/3])
        self.integrate_full(self.start_point)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, 0.1*0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        USCALE = self.model_params['USCALE']
        u_,w1,w2 = self.get_control_plot_arrays(xi)
        u = u_/USCALE
        x1, x2, G11, G12, G21, G22, F11, F12, F22 = self.get_state_arrays(xi)
        
        plt.figure(dpi = dpi)
        
        plt.plot(self.time_grid, x1, 'y-', label = r'Biomass prey $x_1(t)$')
        plt.plot(self.time_grid, x2, 'b-', label = r'Biomass predator $x_2(t)$')
        plt.step(self.time_grid_ref, u, 'r-', label = r'Fishing control $u$')
        plt.step(self.time_grid_ref, w1, 'c-', label = r'sampling $w^{(1)}$')
        plt.step(self.time_grid_ref, w2, 'g--', label = r'sampling $w^{(2)}$')
        
        plt.ylim(0.,4.)
        
        plt.legend(fontsize = 'medium', loc = 'upper left')
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Lotka OED problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
        plt.show()
        plt.close()
        
        
    def plot_sensitivities(self, xi, dpi=None, title=None, it=None):
        _, _, G11, G12, G21, G22, F11, F12, F22 = self.get_state_arrays(xi)
        
        plt.figure(dpi = dpi)
        plt.plot(self.time_grid, G11, 'y-', label = r'$G_{11}(t)$')
        plt.plot(self.time_grid, G12, 'c-', label = r'$G_{12}(t)$')
        plt.plot(self.time_grid, G21, 'r-', label = r'$G_{21}(t)$')
        plt.plot(self.time_grid, G22, 'b-', label = r'$G_{22}(t)$')
        
        plt.ylim(-9.,6.)
        plt.legend(fontsize = 'medium', loc = 'upper left')
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Lotka OED problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
        plt.show()
        plt.close()




class Fermenter(OCProblem):
    #Janka PhD and Le master's thesis params
    default_params = {'mux':2e5,
                      'mup':5000,
                      'gxg':5e4,
                      'gx1':1e5,
                      'gp1':2e4,
                      'gx2':1500,
                      'gp2':5e4
                      }
    
    #MUSCOD II example params
    # default_params = {'mux':2e5,
    #                   'mup':5000,
    #                   'gxg':5e4,
    #                   'gx1':1e5,
    #                   'gp1':2e4,
    #                   'gx2':0.5e4,
    #                   'gp2':1.5e3
    #                   }
    
    def __init__(self, nt = 100, refine = 1, integrator = 'cvodes', parallel = False, N_threads = 4, **kwargs):
        OCProblem.__init__(self, nt=nt, refine=refine, integrator=integrator, parallel=parallel, N_thread=4, **kwargs)

    def build_problem(self):
        self.set_OCP_data(9, 0, 3, 0, [0.,0.,0.,0.,0.3,0.] + [0.,0.,0.], [0.1,0.04,0.03,0.1,0.45,0.1] + [0.05,0.2,0.025], [], [], [0.,0.,0.], [15.,1.,30.])
        mux, mup, gxg, gx1, gp1, gx2, gp2 = (self.model_params[key] for key in ['mux', 'mup', 'gxg', 'gx1', 'gp1', 'gx2', 'gp2'])
        self.fix_time_horizon(0.,1.)
        self.fix_initial_value([0.,0.03,0.03,0.01,0.3,0.1] + [0., 0.009, 0.009])
        x = cs.MX.sym('x', 9)
        P,S1,S2,E,V,G, _,_,_ = cs.vertsplit(x)
        u = cs.MX.sym('u', 3)
        uS1,uS2,uP = cs.vertsplit(u)
        dt = cs.MX.sym('dt', 1)
        
        #In Le, first term in rhs for S1, S2 and G enters with positive sign, negative in Janka and MUSCOD
        #Janka and MUSCOD seem to be correct
        ode_rhs = cs.vertcat(
                mup*E*S1*S2 - P*(uS1+uS2)/(25*V),
                -gx1*E*S1*S2*G - gp1*E*S1*S2 + (0.42*uS1 - S1*(uS1 + uS2))/(25*V),
                -gx2*E*S1*S2*G - gp2*E*S1*S2 + (0.333*uS2 - S2*(uS1 + uS2))/(25*V),
                mux*E*S1*S2*G - E*(uS1 + uS2)/(25*V),
                uS1 + uS2 - uP,
                -gxg*E*S1*S2*G - G*(uS1+uS2)/(25*V),
                uP*P + (uS1 + uS2 - uP)/25 * P + V*(mup*E*S1*S2 - P*(uS1+uS2)/(25*V)),#P,
                0.0168*uS1,
                0.01332*uS2
        )
        
        self.ODE = {'x':x, 'p':cs.vertcat(dt, u), 'ode':dt*ode_rhs}
        self.multiple_shooting()
        
        _,_,_,_,_,_, P_acc, S1_acc, S2_acc = cs.vertsplit(self.x_eval[:,-1])
        # self.set_objective(2*(self.x_eval[7,-1]*self.x_eval[8,-1])/self.x_eval[6,-1])
        self.set_objective(2*(S1_acc*S2_acc)/P_acc)

        self.build_NLP()
        for i in range(self.ntS):
            self.set_stage_control(self.start_point, i, [0., 0., 0.])
        self.integrate_full(self.start_point)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        val0, val1, val2 = self.get_stage_control(s, ind)
        self.set_stage_control(s, ind, [val0 + 0.1, val1 + 0.1, val2 + 0.1])
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        P,S1,S2,E,V,G,_,_,_ = self.get_state_arrays(xi)
        uS1,uS2,uP = self.get_control_plot_arrays(xi)
        
        plt.figure(dpi=dpi)
        plt.step(self.time_grid_ref, uS1/5., 'r', label = 'uS1/5.')
        plt.step(self.time_grid_ref, uS2/15., 'g', label = 'uS2/15.')
        plt.step(self.time_grid_ref, uP/60., 'b', label = 'uP/60.')
        
        plt.plot(self.time_grid, P*10., 'r--', label = r'$P\cdot 10.$')
        plt.plot(self.time_grid, S1, 'g-.', label = 'S1')
        plt.plot(self.time_grid, S2, 'b:', label = 'S2')
        plt.plot(self.time_grid, E, 'y--', label = 'E')
        plt.plot(self.time_grid, V/3, 'c-.', label = 'V/3')
        plt.plot(self.time_grid, G, 'm:', label = 'G')
        plt.legend(fontsize='medium', loc = 'upper center')
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Fermenter problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
        plt.show()
        plt.close()

#Cvodes required
class Batch_Distillation(OCProblem):
    default_params = {'M0init':100.,
                      'MDinit':0.1,
                      'x0init':0.5,
                      'xinit':1.0,
                      'xCinit':1.0,
                      'xDinit':1.0,
                      'alpha':0.2,
                      'V':100,
                      'm':0.1,
                      'mC':0.1
                      }
    # tscale = 1e1 #1e2
    # uscale = 1.0 #1.0
    # M0scale = 1e-2
    # MDscale = 1e-2
    # xDscale = 1e2
    # xCscale = 2.0
    # x0scale = 2.0
    
    tscale = 1.0
    uscale = 1.0
    M0scale = 1.0
    MDscale = 1.0
    xDscale = 1.0
    xCscale = 1.0
    x0scale = 1.0
    
    def __init__(self, nt = 100, refine = 1, integrator = 'cvodes', parallel = False, N_threads = 4, **kwargs):
        OCProblem.__init__(self, nt=nt, refine=refine, integrator=integrator, parallel=parallel, N_thread=4, **kwargs)

    def build_problem(self):
        M0init, MDinit, x0init, xinit, xCinit, xDinit, alpha, V, m, mC = (self.model_params[key] for key in ['M0init', 'MDinit', 'x0init', 'xinit', 'xCinit', 'xDinit', 'alpha', 'V', 'm', 'mC'])
        # self.set_OCP_data(10,1,1,0, [0.]*9 + [MDinit],[np.inf]*10,[0.5/self.ntS * self.tscale],[10/self.ntS * self.tscale], [0. * self.uscale], [15. * self.uscale])
        self.set_OCP_data(10,1,1,0, [0.]*8 + [0.] + [MDinit*self.MDscale],[np.inf] + [self.x0scale] + [1.0]*5 + [self.xCscale, self.xDscale] + [np.inf],[0.5/self.ntS * self.tscale],[10/self.ntS * self.tscale], [0. * self.uscale], [15. * self.uscale])
        self.fix_initial_value([M0init*self.M0scale,x0init*self.x0scale] + [xinit]*5+[xCinit*self.xCscale,xDinit*self.xDscale,MDinit*self.MDscale])
        
        X = cs.MX.sym('X',10)
        M0_,x0_,x1,x2,x3,x4,x5,xC_,xD_,MD_ = cs.vertsplit(X)
        M0 = M0_/self.M0scale
        MD = MD_/self.MDscale
        x0 = x0_/self.x0scale
        xC = xC_/self.xCscale
        xD = xD_/self.xDscale
        
        R_ = cs.MX.sym('R')
        R = R_/self.uscale
        dt_ = cs.MX.sym('dt')
        dt = dt_/self.tscale
        
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
                             self.MDscale*(V - L)
                             )
        
        self.ODE = {'x':X, 'p':cs.vertcat(dt_,R_), 'ode':dt*ode_rhs}
        self.multiple_shooting()
        self.set_objective((1/self.tscale * self.p_tf*self.ntS - self.x_eval[9,-1]/self.MDscale))
        self.add_constraint(self.x_eval[8,-1], 0.99*self.xDscale, np.inf)
        self.build_NLP()
        for j in range(self.ntS):
            self.set_stage_param(self.start_point, j, 1/self.ntS * self.tscale)
        for j in range(math.floor(0.5*self.ntS)):
            self.set_stage_control(self.start_point, j, 1.0*self.uscale)
        for j in range(math.floor(0.5*self.ntS), self.ntS):
            self.set_stage_control(self.start_point, j, 15.0*self.uscale)
        self.integrate_full(self.start_point)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        val = self.get_stage_control(s, ind)
        if ind < math.floor(0.5*self.ntS):
            self.set_stage_control(s, ind, val + 1.0*self.uscale)
        else:
            self.set_stage_control(s, ind, val - 1.0*self.uscale)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        M0_,x0_,x1,x2,x3,x4,x5,xC_,xD_,MD_ = self.get_state_arrays_expanded(xi)
        
        M0 = M0_/self.M0scale
        MD = MD_/self.MDscale
        x0 = x0_/self.x0scale
        xC = xC_/self.xCscale
        xD = xD_/self.xDscale
        
        R = self.get_control_plot_arrays(xi)
        p = self.get_param_arrays_expanded(xi)
        time_grid = np.cumsum(np.concatenate([[0], p/self.tscale])).reshape(-1)
        
        plt.figure(dpi=dpi)
        plt.plot(time_grid, M0/100, 'r--', label = 'M0/100')
        plt.plot(time_grid, x0, 'g-.', label = 'x0')
        plt.plot(time_grid, x1, 'b:', label = 'x1')
        plt.plot(time_grid, x2, 'y-', label = 'x2')
        plt.plot(time_grid, x3, 'c-', label = 'x3')
        plt.plot(time_grid, x4, 'm-', label = 'x4')
        plt.plot(time_grid, x5, 'r--', label = 'x5')
        plt.plot(time_grid, xC, 'g--', label = 'xC')
        plt.plot(time_grid, (xD-0.99)*100, 'b--', label = '(xD-0.99)*100')
        plt.plot(time_grid, MD/100, 'y--', label = 'MD/100')
        
        plt.step(time_grid, R/10 / self.uscale, 'r', label = 'R/10')
        plt.legend(fontsize='large')
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Batch distillation problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
        plt.show()
        plt.close()
        

class Hang_Glider(OCProblem):
    default_params = {
            'x0':0,
            'y0':1000,
            'ytf':900,
            'dxbc':13.23,
            'dybc':-1.288,
            'c0':0.034,
            'c1':0.069662,
            'S':14.,
            'rho':1.13,
            'cmax':1.4,
            'm':100,
            'g':9.81,
            'uC':2.5,
            'rC':100
            }
    def build_problem(self):
        x0, y0, ytf, dxbc, dybc, c0, c1, S, rho, cmax, m, g, uC, rC = (self.model_params[key] for key in ['x0', 'y0', 'ytf', 'dxbc', 'dybc', 'c0', 'c1', 'S', 'rho', 'cmax', 'm', 'g', 'uC', 'rC'])
        self.set_OCP_data(4,1,1,0, [0.,0.,-np.inf,-np.inf], [np.inf,np.inf,np.inf,np.inf], [75/self.ntS], [250/self.ntS], [0], [cmax])
        self.fix_initial_value([x0, dxbc, y0, dybc])
        
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
        self.ODE = {'x':XY, 'p':cs.vertcat(dt,cL), 'ode':dt*ode_rhs}
        self.multiple_shooting()
        self.add_constraint(self.x_eval[1:4,-1] - cs.vertcat(dxbc, ytf, dybc), 0., 0.)
        self.set_objective(-self.x_eval[0,-1])
        self.build_NLP()
        for j in range(self.ntS):
            self.set_stage_control(self.start_point, j, cmax)
            self.set_stage_param(self.start_point, j, 100/self.ntS)
        self.integrate_full(self.start_point)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        val = self.get_stage_control(s, ind)
        self.set_stage_control(s, ind, val - 0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x, dx, y, dy = self.get_state_arrays(xi)
        cL = self.get_control_plot_arrays(xi)
        p = self.get_param_arrays_expanded(xi)
        time_grid = np.cumsum(np.concatenate([[0], p])).reshape(-1)
        
        plt.figure(dpi=dpi)
        plt.step(time_grid, cL, 'tab:red', label = r'$c_L$')
        plt.plot(time_grid, x/500, 'tab:green', linestyle = '-', label = r'$x/500$')
        plt.plot(time_grid, (y-900)/100, 'tab:blue', linestyle = '-', label = r'$(y-900)/1000$')
        plt.plot(time_grid, dx/10, 'tab:green', linestyle = ':', label = r'$v_x/10$')
        plt.plot(time_grid, dy/10, 'tab:blue', linestyle = ':', label = r'$v_y/10$')
        plt.legend(fontsize='large', loc = 'upper right')
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Hang glider problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
        plt.show()
        plt.close()


class Tubular_Reactor(OCProblem):
    def build_problem(self):
        self.set_OCP_data(1,0,1,1, [-np.inf], [np.inf], [], [], [0.], [5.])
        self.fix_time_horizon(0.,1.)
        self.fix_initial_value([1.0])
        x = cs.MX.sym('x', 1)
        u = cs.MX.sym('u', 1)
        dt = cs.MX.sym('dt', 1)
        ode_rhs = -(u + 0.5 * u**2) * x
        quad_expr = u * x
        self.ODE = {'x':x, 'p':cs.vertcat(dt,u), 'ode': dt*ode_rhs, 'quad': dt*quad_expr}
        self.multiple_shooting()
        self.set_objective(-self.q_tf[0])
        
        self.build_NLP()
        for j in range(self.ntS):
            self.set_stage_control(self.start_point, j, 5.)
            self.set_stage_state(self.start_point, j, self.x_init)
        self.set_stage_state(self.start_point, self.ntS, self.x_init)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        val = self.get_stage_control(s, ind)
        self.set_stage_control(s, ind, val - 0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x = self.get_state_arrays(xi)
        u = self.get_control_plot_arrays(xi)
        
        plt.figure(dpi=dpi)
        plt.step(self.time_grid_ref, u, 'tab:red', label = r'$u$')
        plt.plot(self.time_grid, x, 'tab:green', linestyle = '--', label = r'$x$')
        plt.legend(fontsize='large')
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Tubular reactor problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
        plt.show()
        plt.close()
        

class Tubular_Reactor_MAYER(Tubular_Reactor):
    
    def build_problem(self):
        self.set_OCP_data(2,0,1,0, [-np.inf, -np.inf], [np.inf, np.inf], [], [], [0.], [5.])
        self.fix_time_horizon(0.,1.)
        self.fix_initial_value([None, 0.])
        X = cs.MX.sym('X', 2)
        x,y = cs.vertsplit(X)
        u = cs.MX.sym('u', 1)
        dt = cs.MX.sym('dt', 1)
        ode_rhs = cs.vertcat(-(u + 0.5 * u**2) * x, u * x)
        self.ODE = {'x':X, 'p':cs.vertcat(dt,u), 'ode': dt*ode_rhs}
        self.multiple_shooting()
        self.set_objective(-self.x_eval[1,-1])
        
        self.build_NLP()
        self.set_stage_state(self.lb_var, 0, [0.])
        self.set_stage_state(self.ub_var, 0, [1.])
        for j in range(self.ntS):
            self.set_stage_control(self.start_point, j, 5.)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        val = self.get_stage_control(s, ind)
        self.set_stage_control(s, ind, val - 0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x,_ = self.get_state_arrays(xi)
        u = self.get_control_plot_arrays(xi)
        
        plt.figure(dpi=dpi)
        plt.step(self.time_grid_ref, u, 'r', label = 'u')
        plt.plot(self.time_grid, x, 'g-', label = 'x')
        plt.legend(fontsize='large')
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Hang glider problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
        plt.show()
        plt.close()


class Mountain_Car(OCProblem):
    default_params = {}
    
    def build_problem(self):
        self.set_OCP_data(2,1,1,0,[-np.inf, -np.inf],[np.inf, np.inf],[1.0/self.ntS],[np.inf],[-1.0],[1.0])
        self.fix_initial_value([-0.5, 0.])
        
        X = cs.MX.sym('X', 2)
        x,v = cs.vertsplit(X)
        u = cs.MX.sym('u', 1)
        ode_rhs = cs.vertcat(v, 0.001*u - 0.0025*cs.cos(3*x))
        dt = cs.MX.sym('dt', 1)
        self.ODE = {'x': X, 'p':cs.vertcat(dt, u),'ode': dt*ode_rhs}
        self.multiple_shooting()
        self.set_objective(self.p_tf*self.ntS)
        self.add_constraint(self.x_eval[:,-1], [0.5, 0.], [0.5, np.inf])
        self.build_NLP()
        
        self.start_point = np.zeros(self.nVar)
        for i in range(self.ntS+1):
            self.set_stage_state(self.start_point, i, [-0.5, 0.])
        for i in range(self.ntS):
            self.set_stage_control(self.start_point, i, [0.])
            self.set_stage_param(self.start_point, i, [100.0/self.ntS])
        self.integrate_full(self.start_point)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, 0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        t_arr = self.get_param_arrays_expanded(xi)
        time_grid_ref = np.cumsum(np.concatenate(([0], t_arr))).reshape(-1)
        
        x,v = self.get_state_arrays_expanded(xi)
        u = self.get_control_plot_arrays(xi)
        
        fig,ax = plt.subplots(dpi=dpi)
        ax.plot(time_grid_ref, x, 'g-.', label = '$x$')
        ax.plot(time_grid_ref, v, 'b-.', label = '$v$')
        ax.step(time_grid_ref, u, 'r', label = r'$u$')
        ax.legend(fontsize='x-large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Rao Maese problem'
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


class Rao_Mease(OCProblem):
    default_params = {}
    
    def build_problem(self):
        self.set_OCP_data(1,0,1,1,[-np.inf],[np.inf],[],[],[-np.inf],[np.inf])
        self.fix_time_horizon(0.,10.)
        self.fix_initial_value([1.0])
        
        x = cs.MX.sym('x', 1)
        w = cs.MX.sym('w', 1)
        ode_rhs = cs.vertcat(-x**3 + w)
        quad_expr = x**2 + w**2
        dt = cs.MX.sym('dt', 1)
        self.ODE = {'x': x, 'p':cs.vertcat(dt, w),'ode': dt*ode_rhs, 'quad': dt*quad_expr}
        self.multiple_shooting()
        self.set_objective(self.q_tf)
        self.add_constraint(self.x_eval[:,-1], 1.5, 1.5)
        self.build_NLP()
        
        self.start_point = np.zeros(self.nVar)
        for i in range(self.ntS+1):
            self.set_stage_state(self.start_point, i, 1.0)
        for i in range(self.ntS):
            self.set_stage_control(self.start_point, i, [0.])
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, 0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x = self.get_state_arrays_expanded(xi)
        w = self.get_control_plot_arrays(xi)
        
        fig,ax = plt.subplots(dpi=dpi)
        ax.plot(self.time_grid_ref, x, 'g-.', label = '$x$')
        ax.step(self.time_grid_ref, w, 'r', label = r'$w$')
        ax.legend(fontsize='x-large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Rao Maese problem'
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



# Problems """borrowed""" from MSO participants

class Lotka_Volterra_Foodsource(OCProblem):
    default_params = {'c1':0.1, 'c2':0.4, 't0':0., 'tf':40.0, 'x_init':[1.5,0.5,1.0]}
    def build_problem(self):
        self.set_OCP_data(3,0,1,1,[0.,0.,0.],[2.0, 2.0, 2.0],[],[],[0.],[1.])
        
        c1, c2 = (self.model_params[key] for key in ['c1', 'c2'])
        self.fix_time_horizon(self.model_params['t0'], self.model_params['tf'])
        self.fix_initial_value(self.model_params['x_init'])
        
        x = cs.MX.sym('x', 3)
        u = cs.MX.sym('w', 1)
        x0, x1, x2 = cs.vertsplit(x)
        ode_rhs = cs.vertcat(x0 - x0 * x1 - x0 * x2,
              -x1 + x0 * x1 - c1 * x1 * u, 
              -x2 + 1.2 * x0 * x2 - c2 * x2 * u)
        
        quad_expr = (x1 - 1)**2 + (x2 - 1)**2 + 1e-3*u**2 + (x0 - 1.7)**2
        dt = cs.MX.sym('dt', 1)
        self.ODE = {'x': x, 'p':cs.vertcat(dt, u),'ode': dt*ode_rhs, 'quad': dt*quad_expr}
        self.multiple_shooting()
        self.set_objective(self.q_tf)
        self.build_NLP()
        
        self.start_point = np.zeros(self.nVar)
        for i in range(self.ntS+1):
            self.set_stage_state(self.start_point, i, self.model_params['x_init'])
        for i in range(self.ntS):
            self.set_stage_control(self.start_point, i, [0])
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, 0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x0, x1, x2 = self.get_state_arrays_expanded(xi)
        u = self.get_control_plot_arrays(xi)
        
        plt.figure(dpi = dpi)
        plt.plot(self.time_grid_ref, x0, 'g-.', label = '$x_0$')
        plt.plot(self.time_grid_ref, x1, 'b--', label = '$x_1$')
        plt.plot(self.time_grid_ref, x2, 'y:', label = '$x_2$')
        #'y-.'
        
        plt.step(self.time_grid_ref, u, 'r', label = r'$u$')
        plt.legend(fontsize='x-large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Lotka Volterra fishing problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
            
        plt.show()
        plt.close()

class Lotka_Volterra_Competitive(OCProblem):
    default_params = {'c1':0.1, 'c2':0.4, 't0':0., 'tf':40.0, 'x_init':[0.5, 1.5]}
    def build_problem(self):
        self.set_OCP_data(2,0,1,1,[0.,0.],[2.0, 2.0],[],[],[0.],[1.])
        
        c1, c2 = (self.model_params[key] for key in ['c1', 'c2'])
        self.fix_time_horizon(self.model_params['t0'], self.model_params['tf'])
        self.fix_initial_value(self.model_params['x_init'])
        
        x = cs.MX.sym('x', 2)
        u = cs.MX.sym('w', 1)
        x0, x1 = cs.vertsplit(x)
        ode_rhs = cs.vertcat(
                x0 * (1 - (x0 + 1.2 * x1)/1.8) - c1 * x0 * u,#x[0] population suffers greater loss from competition with x[1] than vice versa
                x1 * (1 - (x0 + x1)/1.8) - c2 * x1 * u)
        
        quad_expr = (x1 - 1)**2 + (x0 - 1)**2 + 1e-4*u**2 
        dt = cs.MX.sym('dt', 1)
        self.ODE = {'x': x, 'p':cs.vertcat(dt, u),'ode': dt*ode_rhs, 'quad': dt*quad_expr}
        self.multiple_shooting()
        self.set_objective(self.q_tf)
        self.build_NLP()
        
        self.start_point = np.zeros(self.nVar)
        for i in range(self.ntS+1):
            self.set_stage_state(self.start_point, i, self.model_params['x_init'])
        for i in range(self.ntS):
            self.set_stage_control(self.start_point, i, [0])
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, 0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x0, x1 = self.get_state_arrays_expanded(xi)
        u = self.get_control_plot_arrays(xi)
        
        plt.figure(dpi = dpi)
        plt.plot(self.time_grid_ref, x0, 'g-.', label = '$x_0$')
        plt.plot(self.time_grid_ref, x1, 'b--', label = '$x_1$')
        #'y-.'
        
        plt.step(self.time_grid_ref, u, 'r', label = r'$u$')
        plt.legend(fontsize='x-large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Lotka Volterra fishing problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
            
        plt.show()
        plt.close()


class Cart_Pendulum(OCProblem):
    default_params = {
            'M':1.0,
            'm':0.1,
            'l':1.0,
            'g':9.81,
            'u_max':30,
            'lambda_u':0.5
            }
    
    param_set_1 = {'u_max': 30, 'lambda_u': 0.5}
    param_set_2 = {'u_max': 15, 'lambda_u': 0.05}
    
    def build_problem(self):
        M, m, l, g, u_max, lambda_u = (self.model_params[key] for key in ['M','m','l','g','u_max','lambda_u'])
        
        self.set_OCP_data(4,0,1,0, [-2.0, -np.inf, -np.inf, -np.inf], [2.0, np.inf, np.inf, np.inf], [], [], [-u_max], [u_max])
        self.fix_time_horizon(0., 4.0)
        self.fix_initial_value([0., 0., 0., 0.])
        
        w = cs.MX.sym('w', 4)
        x,xdot,theta,thetadot = cs.vertsplit(w)
        _,w2,w3,w4 = (x, xdot, theta, thetadot)
        u = cs.MX.sym('u', 1)
        dt = cs.MX.sym('dt', 1)
        
        w2dot = (u + m*g*cs.sin(w3)*cs.cos(w3) + m*l*w4**2 * cs.sin(w3))/(M + m*(1 - cs.cos(w3)**2))
        
        ode_rhs = cs.vertcat(
            w2,
            w2dot,
            w4,
            (-g*cs.sin(w3) - w2dot * cs.cos(w3))/l
        )
     
        quad_expr = 10*x**2 + 50*(theta - cs.pi)**2 + lambda_u*u**2
        self.ODE = {'x':w, 'p':cs.vertcat(dt,u), 'ode': dt*ode_rhs, 'quad': dt*quad_expr}
        self.multiple_shooting()
        self.set_objective(self.q_tf[0])
        
        self.build_NLP()
        for j in range(self.ntS):
            self.set_stage_control(self.start_point, j, 0.)
        self.integrate_full(self.start_point)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        val = self.get_stage_control(s, ind)
        self.set_stage_control(s, ind, val - 1.0)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x,xdot,theta,thetadot = self.get_state_arrays(xi)
        u = self.get_control_plot_arrays(xi)
        
        plt.figure(dpi=dpi)
        plt.step(self.time_grid_ref, 4*u/self.model_params['u_max'], 'tab:red', label = r'$4\cdot u$/$u_{max}$')
        plt.plot(self.time_grid, x, 'tab:green', linestyle = '--', label = r'$x$')
        plt.plot(self.time_grid, theta, 'tab:blue', linestyle = '-.', label = r'$\theta$')
        plt.legend(fontsize='large')
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Cart pendulum problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
        plt.show()
        plt.close()


#Differential state 1 is stiff, approximating its lower bound of 0, so an implicit integrator is required.
#In addition, the bound should be relaxed to something like 1e-[4 -- 6].

#Using one step explicit euler with lower bound 1e-6 also """""""works""""""" as the optimizer can adjust the controls such that the collocation does not overshoot 
class Denbigh_Reaction(OCProblem):
    default_params = {
            'E1':3000.0,
            'E2':6000.0,
            'E3':3000.0,
            'E4':0.,
            }
    
    def build_problem(self):
        E = [self.model_params[key] for key in ['E1','E2','E3','E4']]

        self.set_OCP_data(2,0,1,1, [-1e-6,0.], [1.0, 1.0], [], [], [273.0], [415.0])
        self.fix_time_horizon(0., 1000.0)
        self.fix_initial_value([1.0, 0.])
        
        x = cs.MX.sym('x', 2)
        x1, x2 = cs.vertsplit(x)
        # x1 = cs.fmax(x1_, 0.)
        
        T = cs.MX.sym('T', 1)
        dt = cs.MX.sym('dt', 1)
        
        k_s = [1e3,1e7,1e1,1e-3]
        
        k1, k2, k3, k4 = [k_s[i]*cs.exp(-E[i]/(T)) for i in range(4)]
        
        ode_rhs = cs.vertcat(
            -k1*x1 - k2*x1,
            k1*x1 - (k3 + k4)*x2
        )
     
        quad_expr = k3*x2
        self.ODE = {'x':x, 'p':cs.vertcat(dt,T), 'ode': dt*ode_rhs, 'quad': dt*quad_expr}
        self.multiple_shooting()
        self.set_objective(-self.q_tf[0])
        
        self.build_NLP()
        for j in range(self.ntS):
            self.set_stage_control(self.start_point, j, 273.0)
        self.integrate_full(self.start_point)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, 280.0)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x1, x2 = self.get_state_arrays_expanded(xi)
        T = self.get_control_plot_arrays(xi)
        
        x1_in, x2_in = self.get_state_arrays(xi)
        x_in = cs.vertcat(x1_in.reshape((1,-1)), x2_in.reshape((1,-1)))[:,:-1]
        dt = cs.diff(self.time_grid_ref.reshape((1,-1)),1,1)
        p_in = cs.vertcat(dt,T[:-1].reshape((1,-1)))
        # print(p_in.shape)
        
        out = self.odesol_refined(x0 = x_in, p = p_in)
        # print(out)
        q = np.array(cs.cumsum(cs.horzcat(cs.DM([[0]]), out['qf']), 1)).reshape(-1)
        # print(q)
        
        plt.figure(dpi=dpi)
        plt.step(self.time_grid_ref, (T - 273.0)/142.0, 'r', label = r'(T-273)/142')
        plt.plot(self.time_grid_ref, x1, 'g-', label = '$x_1$')
        plt.plot(self.time_grid_ref, x2, 'b--', label = '$x_2$')
        plt.plot(self.time_grid_ref, q, 'c', label = 'q')
        plt.legend(fontsize='large')
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Denbigh_Reaction'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
        plt.show()
        plt.close()


class Clinic_Scheduling(OCProblem):    
    default_params = {
        'alpha': 2.0,
        'beta': 3.0,
        'gamma': 1.5,
        'delta': 0.1,
        'lam_max': 15.0,    # Maximum appointment rate [patients/hour]
        'mu_min': 0.5,       # Minimum service rate factor
        'mu_max': 2.0,       # Maximum service rate factor
        'N_max': 8,       # Maximum number of staff
        'W_max': 30.0,       # Maximum acceptable waiting time [minutes]
        'U_min': 0.6,       # Minimum utilization requirement
        'N_total': 60,       # Total appointments to schedule
        'lam_0': 7.5,       # Target appointment rate [patients/hour]
        'w1': 1.0,
        'w2': 0.5,
        'w3': 0.1,
        'Q0': 10.0,
        'S0': 4.0,
        'W0': 5.0,
        'U0': 0.8
    }
    
    def build_problem(self):
        alpha, beta, gamma, delta, lam_max, mu_min, mu_max, N_max, W_max, U_min, N_total, lam_0, w1, w2, w3, Q0, S0, W0, U0 = (self.model_params[key] for key in ['alpha', 'beta', 'gamma', 'delta', 'lam_max', 'mu_min', 'mu_max', 'N_max', 'W_max', 'U_min', 'N_total', 'lam_0', 'w1', 'w2', 'w3', 'Q0', 'S0', 'W0', 'U0'])
        
        self.set_OCP_data(4,0,3,0, [0.,0.,0.,U_min], [np.inf,N_max,W_max,1.0], [], [], [0., mu_min, 1.0], [lam_max, mu_max, N_max])
        self.fix_time_horizon(0., 8.0)
        self.fix_initial_value([Q0, S0, W0, U0])
        
        X = cs.MX.sym('X', 4)
        Q, S, W, U = cs.vertsplit(X)
        
        u = cs.MX.sym('u', 3)
        lam, mu, N = cs.vertsplit(u)
        
        dt = cs.MX.sym('dt', 1)
        
        ode_rhs = cs.vertcat(
            lam - cs.fmin(S, Q)*mu,
            gamma*(N - S) - delta*S,
            alpha*(Q/cs.fmax(S, 0.1) - W),
            beta*(cs.fmin(Q,S)/cs.fmax(S, 0.1) - U)
        )
             
        quad_expr = w1*W**2 + w2 * (1 - U)**2 + w3*(lam - lam_0)**2
        self.ODE = {'x':X, 'p':cs.vertcat(dt,u), 'ode': dt*ode_rhs, 'quad': dt*quad_expr}
        self.multiple_shooting()
        self.set_objective(self.q_tf[0])
        self.add_constraint(self.u_eval[0,1:] - self.u_eval[0,:-1], -2.0, 2.0)
        
        
        self.build_NLP()
        for j in range(self.ntS+1):
            self.set_stage_state(self.start_point, j, self.x_init)
        for j in range(self.ntS):
            self.set_stage_control(self.start_point, j, [0., mu_min, 1.0])
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        val1, val2, val3 = self.get_stage_control(s, ind)
        self.set_stage_control(s, ind, [val1 + 0.1, val2 + 0.1, val3 + 1.0])
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        Q, S, W, U = self.get_state_arrays_expanded(xi)
        lam, mu, N = self.get_control_plot_arrays(xi)
        
        plt.figure(dpi=dpi)
        plt.plot(self.time_grid_ref, Q, 'g-', label = r'Q')
        plt.plot(self.time_grid_ref, S, 'b--', label = r'S')
        plt.plot(self.time_grid_ref, W, 'c', label = r'W')
        plt.plot(self.time_grid_ref, 4*U, 'y', label = r'$U\cdot 4$')
        
        plt.step(self.time_grid_ref, lam, 'r', label = r'$\lambda$')
        plt.step(self.time_grid_ref, mu, 'g', label = r'$\mu$')
        plt.step(self.time_grid_ref, N, 'b', label = r'N')
        
        plt.legend(fontsize='large')
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Clinic_Scheduling'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
        plt.show()
        plt.close()


class Rocket_Landing(OCProblem):    
    default_params = {
        'Isp': 350,
        'g0': 9.81,
        'g': 9.81,
        'rho0': 1.225,
        'H': 8400,
        'CD': 1.64,
        'A': 33.18,
        'Tmin': 2e5,
        'Tmax': 9e5,
        'thetamax': 15*(2*np.pi/360),
        'gammamax': 30*(2*np.pi/360),
        'mdry': 25e3,
        'objective': "max_fuel",
        
        # 'Tscale': 1e-5,
        # 'xscale': 1e-3,
        # 'zscale': 1e-3,
        # 'vscale': 1e-2,
        # 'mscale': 1e-5,
        # 'tscale': 1.0,
        
        # 'Tscale': 1.0,
        # 'xscale': 1.0,
        # 'zscale': 1.0,
        # 'vscale': 1.0,
        # 'mscale': 1.0,
        # 'tscale': 1.0,
        
        'Tscale': 1e-6,
        'xscale': 1e-3,
        'zscale': 1e-3,
        'vscale': 1e-2,
        'mscale': 1e-4,
        'tscale': 1.0, #leave at 1.0
    }
    
    def build_problem(self):
        Isp, g0, g, rho0, H, CD, A, Tmin, Tmax, thetamax, gammamax, mdry, objective, Tscale, xscale, zscale, vscale, mscale, tscale = (self.model_params[key] for key in ['Isp', 'g0', 'g', 'rho0', 'H', 'CD', 'A', 'Tmin', 'Tmax', 'thetamax', 'gammamax', 'mdry', 'objective', 'Tscale', 'xscale', 'zscale', 'vscale', 'mscale', 'tscale'])
        
        self.set_OCP_data(5,1,2,0, [-np.inf,0.,-np.inf,-np.inf,25e3 * mscale], [np.inf,np.inf,np.inf,np.inf,np.inf], [15/self.ntS * tscale], [45/self.ntS * tscale], [Tmin * Tscale, -thetamax], [Tmax * Tscale, thetamax])
        self.fix_initial_value([xscale*(-100), zscale*2000, vscale*30, vscale*(-236), mscale*30e3])
        
        X = cs.MX.sym('X', 5)
        X_ = X / [xscale, zscale, vscale, vscale, mscale]
        x,z,vx,vz,m = cs.vertsplit(X_)
        
        u = cs.MX.sym('u', 2)
        T_, theta = cs.vertsplit(u)
        T = T_/Tscale
        
        dt = cs.MX.sym('dt', 1)
        dt_s = dt/tscale
        
        vsq = vx**2 + vz**2
        rho = rho0*cs.exp(-z/H)
        D = 0.5*rho*CD*A*vsq
        ode_rhs = cs.vertcat(
            xscale*vx,
            zscale*vz,
            vscale*(T*cs.sin(theta)/m - D*vx/(m*cs.sqrt(vsq))),
            vscale*(T*cs.cos(theta)/m - g - D*vz/(m*cs.sqrt(vsq))),
            mscale*(-T/(Isp*g0))
        )
        
        self.ODE = {'x':X, 'p':cs.vertcat(dt,u), 'ode': dt_s*ode_rhs}
        self.multiple_shooting()
        
        _,_,_,_,m_tf = cs.vertsplit(self.x_eval[:,-1])
        
        if objective == "max_fuel":
            self.set_objective(-m_tf)
            con_relax_factor = 1.0
        elif objective == "max_performance":
            x_tf = self.x_eval[:-1,-1]
            dtf = self.p_eval[:,-1]/tscale
            self.set_objective(1000*(x_tf[0]**2 + x_tf[1]**2 + x_tf[2]**2 + (x_tf[3]+2*vscale)**2) + 0.1*dtf*self.ntS)
            con_relax_factor = 1.8
        else:
            raise Exception("Unknown objective")
        
        x_tf, z_tf, vx_tf, vz_tf, m_tf = cs.vertsplit(self.x_eval[:,-1])
        

        POS_TOL = 3.0
        VEL_TOL = 3.0
        Prlx = con_relax_factor * POS_TOL
        Vrlx = con_relax_factor * VEL_TOL
        self.add_constraint(self.x_eval[:,-1], [0. - xscale*Prlx, 
                                                0. - zscale*Prlx, 
                                                0. - vscale*Vrlx, 
                                                vscale*(-2. - Vrlx), 
                                                mscale*25e3], 
                                                [0. + xscale*Prlx, 
                                                 0. + zscale*Prlx, 
                                                 0. + vscale*Vrlx, 
                                                 vscale*(-2. + Vrlx), 
                                                 np.inf])
        
        self.build_NLP()
        for j in range(self.ntS+1):
            self.set_stage_state(self.start_point, j, self.x_init)
        for j in range(self.ntS):
            self.set_stage_param(self.start_point, j, [tscale*25/self.ntS])
        
        for j in range(0, self.ntS):
            self.set_stage_control(self.start_point, j, [Tmin*Tscale, 0.])
        
        self.integrate_full(self.start_point)
    
    def perturbed_start_point(self, ind):
        Tscale = self.model_params['Tscale']
        s = copy.copy(self.start_point)
        T_i, theta_i = self.get_stage_control(s, ind)
        self.set_stage_control(s, ind, [T_i + 1e5*Tscale, theta_i + (2*np.pi/360)])
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        Tscale, xscale, zscale, vscale, mscale, tscale = (self.model_params[key] for key in ['Tscale', 'xscale', 'zscale', 'vscale', 'mscale', 'tscale'])
        
        # x_,z_,vx_,vz_,m_ = self.get_state_arrays_expanded(xi)# / [xscale, zscale, vscale, vscale, mscale]
        # x,z,vx,vz,m = (x_/xscale, z_/zscale, vx_/vscale, vz_/vscale, m_/mscale)
        
        x,z,vx,vz,m = self.get_state_arrays_expanded(xi)
        
        T, theta = self.get_control_plot_arrays(xi)# / [Tscale, 1.0]
        dt_arr = self.get_param_arrays_expanded(xi)
        time_grid_ref = np.cumsum(np.concatenate([[0], dt_arr])).reshape(-1)
        
        Tscale = self.model_params['Tscale']
        
        plt.figure(dpi=dpi)
        plt.plot(time_grid_ref, x*20, 'm-', label = r'x $\cdot$ ' + str(xscale) + r'$\cdot 20$')
        plt.plot(time_grid_ref, z, 'g--', label = r'z $\cdot$ ' + str(zscale))
        plt.plot(time_grid_ref, vx, 'b:', label = r'vx $\cdot$ ' + str(vscale))
        plt.plot(time_grid_ref, vz, 'c-.', label = r'vz $\cdot$ ' + str(vscale))
        plt.plot(time_grid_ref, (m - self.model_params['mdry']*mscale)*5.0, 'y-.', label = r'(m - mdry)$\cdot$' + str(mscale) + r'$\cdot 5.0$')
        
        plt.step(time_grid_ref, T, 'r', label = r'T$\cdot$ ' + str(Tscale))
        plt.step(time_grid_ref, theta*10, 'g', label = r'theta$\cdot 10$')
        
        plt.legend(fontsize='small', loc = 'upper left')
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Rocket_Landing'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
        plt.show()
        plt.close()


class Satellite_Deorbiting_1(OCProblem):
    default_params = {
            'mu':3.986e14,
            'RE': 6.371e6,
            'rho0': 1.225,
            'H': 8500,
            'CD': 2.2,
            'A':1.0,
            'Isp': 220,
            'g0': 9.81,
            'umax': 20,
            'm0': 150,
            'omegaE':7.2921e-5,
            # 'rscale': 1.0e-2,
            # 'vscale': 1.0e-1,
            # 'thetascale': 1.0,
            # 'mscale': 1.0e1,
            # 'TSCALE': 1.0
            'rscale': 1.0e-4,
            'vscale': 1.0e-2,
            'thetascale': 1.0e-2,
            'mscale': 1.0e0,
            'TSCALE': 1.0
            }
    
    def build_problem(self):
        mu, RE, rho0, H, CD, A, Isp, g0, umax, omegaE, rscale, thetascale, mscale, vscale, TSCALE = (self.model_params[key] for key in ['mu', 'RE', 'rho0', 'H', 'CD', 'A', 'Isp', 'g0', 'umax', 'omegaE', 'rscale', 'thetascale', 'mscale', 'vscale', 'TSCALE'])
        
        h0 = 450000
        r0 = RE + h0
        theta0 = 0.
        vr0 = 0.
        vorb = np.sqrt(mu/r0)
        m0 = 150
        mdry = 100
        
        hreentry = 120000
        rfinal = RE + hreentry
        
        self.set_OCP_data(5,1,2,0, [(RE+5000. - RE)*rscale, -2*np.pi*thetascale, -10000.*vscale, 0.*vscale, (mdry - 0.1)*mscale], [(r0 + 100000. - RE)*rscale, 2*np.pi*thetascale, 10000.*vscale, 20000*vscale, (m0 + 0.1)*mscale], [300/self.ntS * TSCALE], [21600/self.ntS * TSCALE], [-umax, -umax], [umax, umax])
        self.fix_initial_value([(r0 - RE)*rscale, theta0*thetascale, vr0*vscale, vorb*vscale, m0*mscale])
        
        def safe_sqrt(x):
            return cs.sqrt(cs.fmax(x, 1e-12))
        
        # Atmospheric model
        def atmospheric_density(r_val):
            h = r_val - RE
            h_safe = cs.fmax(h, -100000)
            return rho0 * cs.exp(-h_safe / H)
        
        # Relative velocity components
        def relative_velocity(v_r_val, v_theta_val, r_val):
            v_rel_r = v_r_val
            v_rel_theta = v_theta_val - omegaE * r_val
            v_rel_sq = v_rel_r**2 + v_rel_theta**2
            v_rel = safe_sqrt(v_rel_sq)
            return v_rel_r, v_rel_theta, v_rel
        
        X = cs.MX.sym('X', 5)
        r_, theta_, vr_, vtheta_, m_ = cs.vertsplit(X)
        r = r_/rscale + RE
        theta = theta_/thetascale
        vr = vr_/vscale
        vtheta = vtheta_/vscale
        m = m_/mscale
        
        U = cs.MX.sym('U', 2)
        ur, utheta = cs.vertsplit(U)
        dt_ = cs.MX.sym('dt', 1)
        dt = dt_/TSCALE
        
        rsafe = cs.fmax(r, RE + 10000)
        msafe = cs.fmax(m, mdry)
        
        rho = atmospheric_density(rsafe)
        vrelr, vreltheta, vrel = relative_velocity(vr, vtheta, rsafe)
        
        centrifugal = vtheta**2/rsafe
        gravity = mu/(rsafe**2)
        drag = 0.5*CD*A/msafe*rho*vrel
        rthrust = ur/msafe
        thetathrust = utheta/msafe
        
        ode_rhs = cs.vertcat(
            vr * rscale,
            vtheta/rsafe * thetascale,
            (centrifugal - gravity + rthrust - drag*vrelr) * vscale,
            (-vr*vtheta/rsafe + thetathrust - drag*vreltheta) * vscale,
            # vtheta**2 / r_safe - mu / r**2 + ur/m - drag*vrelr,
            # -vr*vtheta/r + utheta/m - drag*vreltheta,
            (-cs.sqrt(ur**2 + utheta**2)/(Isp*g0)) * mscale
        )
        
        self.ODE = {'x':X, 'p':cs.vertcat(dt_,U), 'ode': dt*ode_rhs}
        self.multiple_shooting()
        self.set_objective(self.ntS*self.p_tf[-1]/TSCALE)
        
        rT,_,_,_,_ = cs.vertsplit(self.x_eval[:,-1])
        urt, uthetat = cs.vertsplit(self.u_eval)
        
        self.add_constraint(rT/rscale, 0., rfinal - RE)
        self.add_constraint(safe_sqrt(urt**2 + uthetat**2), 0., umax)
        
        self.build_NLP()
        
        for j in range(self.ntS):
            self.set_stage_param(self.start_point, j, 1800/self.ntS * TSCALE)
            self.set_stage_control(self.start_point, j, [-5.0,-10.0])
        
        r_init = (np.linspace(r0, rfinal, self.ntS + 1) - RE) * rscale
        theta_init = np.linspace(0, 2*np.pi, self.ntS + 1) * thetascale
        vr_init = np.zeros(self.ntS + 1) * rscale
        vtheta_init = np.ones(self.ntS + 1)*vorb*0.9 * rscale
        m_init = np.linspace(m0, mdry + 10, self.ntS + 1) * mscale
        for j in range(self.ntS + 1):
            self.set_stage_state(self.start_point, j, [r_init[j], theta_init[j], vr_init[j], vtheta_init[j], m_init[j]])
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        val0, val1 = self.get_stage_control(s, ind)
        self.set_stage_control(s, ind, [val0 + 1.0, val1 + 1.0])
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        RE, rscale, thetascale, mscale, TSCALE = [self.model_params[key] for key in ['RE', 'rscale', 'thetascale', 'mscale', 'TSCALE']]
        
        r_, theta_, vr_, vtheta_, m_ = self.get_state_arrays(xi)
        r = r_/rscale + RE
        theta = theta_/thetascale
        # vr = vr_/rscale
        # vtheta = vtheta_/rscale
        m = m_/mscale
        
        ur, utheta = self.get_control_plot_arrays(xi)
        dt = self.get_param_arrays(xi)
        time_grid = np.cumsum(np.concatenate([np.array([0]), dt]))/TSCALE
        dte = self.get_param_arrays_expanded(xi)
        time_grid_ref = np.cumsum(np.concatenate([np.array([0]), dte]))/TSCALE
        
        plt.figure(dpi=dpi)
        plt.plot(time_grid, (r - RE)/1000, 'r--', label = r'(r - Re)/1000')
        plt.plot(time_grid, theta*50, 'g:', label = r'$\theta\cdot 50$')
        # plt.plot(time_grid, vr, 'b-.', label = r'$v_r$')
        # plt.plot(time_grid, vtheta, 'y-', label = r$v_\theta$))
        plt.plot(time_grid, (m - 100.)*4, 'b-.', label = r'(m - 100)$\cdot 4$')
        
        # plt.axhline(y = 5)
        # plt.axhline(y = 120, color = 'g', linestyle = '--')
        
        plt.step(time_grid_ref, ur*20, 'r', label = r'$u_r \cdot 20$')
        plt.step(time_grid_ref, utheta*20, 'g', label = r'$u_\theta \cdot 20$')
        
        plt.legend(fontsize='large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Satellite Deorbiting min time problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
        plt.show()
        plt.close()


#Doent work well (flat objective due to very low control shifting sensitivity)
# TODO consider adding scaled duration penalty to objective
# Somehow also brings the MUMPS sparse linear solver to its knees inside qpOASES ...
class Satellite_Deorbiting_2(OCProblem):
    default_params = {
            # 'r0':6.821e6,
            # 'vr0':0.,
            # 'vtheta0':7650,
            # 'm0':150,
            # 'rT':6491e6,
            'mu': 3.986e14,
            'RE': 6.371e6,
            'rho0': 1.225,
            'H': 8500,
            'CD': 2.2,
            'A':1.0,
            'Isp': 220,
            'g0': 9.81,
            'umax': 20,
            'm0': 150,
            'omegaE':7.2921e-5,
            'MDTH': 1.0,
            'rscale': 1.0e-2, #1.0e-2
            'vscale': 1.0e-1, #1.0e-1
            'thetascale': 1.0,
            'mscale': 1.0e1   #1.0e1
            }
    
    def build_problem(self):
        mu, RE, rho0, H, CD, A, Isp, g0, umax, omegaE, MDTH, rscale, vscale, thetascale, mscale = (self.model_params[key] for key in ['mu', 'RE', 'rho0', 'H', 'CD', 'A', 'Isp', 'g0', 'umax', 'omegaE', 'MDTH', 'rscale', 'vscale', 'thetascale', 'mscale'])
        
        h0 = 450000
        r0 = RE + h0
        theta0 = 0.
        vr0 = 0.
        vorb = np.sqrt(mu/r0)
        m0 = 150
        mdry = 100
        
        hreentry = 120000
        rfinal = RE + hreentry
        
        # MISSION_DEORBIT_TIME_HOURS = 0.6
        MISSION_DEORBIT_TIME_HOURS = MDTH
        
        T_mission_fixed = MISSION_DEORBIT_TIME_HOURS * 3600
        orbital_period = 2 * np.pi * np.sqrt(r0**3 / mu)
        n_orbits = T_mission_fixed / orbital_period
        
        self.set_OCP_data(5,0,2,0, [(RE + 5000 - RE)*rscale, -np.ceil(n_orbits)*2*np.pi*thetascale, -8000.*vscale, 1000.*vscale, (mdry - 0.1)*mscale], [(r0 + 50000. - RE)*rscale, np.ceil(n_orbits)*2*np.pi*thetascale, 8000.*vscale, 15000.*vscale, (m0 + 0.1)*mscale], [], [], [-umax, -umax], [umax, umax])
        self.fix_time_horizon(0., T_mission_fixed)
        self.fix_initial_value([(r0 - RE)*rscale, (theta0)*thetascale, vr0*vscale, vorb*vscale, m0*mscale])
        
        def safe_sqrt(x):
            return cs.sqrt(cs.fmax(x, 1e-12))
        
        # Atmospheric model
        def atmospheric_density(r_val):
            h = r_val - RE
            h_safe = cs.fmax(h, -100000)
            return rho0 * cs.exp(-h_safe / H)
        
        # Relative velocity components
        def relative_velocity(v_r_val, v_theta_val, r_val):
            v_rel_r = v_r_val
            v_rel_theta = v_theta_val - omegaE * r_val
            v_rel_sq = v_rel_r**2 + v_rel_theta**2
            v_rel = safe_sqrt(v_rel_sq)
            return v_rel_r, v_rel_theta, v_rel
        
        X = cs.MX.sym('X', 5)
        r_, theta_, vr_, vtheta_, m_ = cs.vertsplit(X)
        r = r_/rscale + RE
        theta = theta_/thetascale
        vr = vr_/vscale
        vtheta = vtheta_/vscale
        m = m_/mscale
        
        U = cs.MX.sym('U', 2)
        ur, utheta = cs.vertsplit(U)
        dt = cs.MX.sym('dt', 1)
        
        rsafe = cs.fmax(r, RE + 10000)
        msafe = cs.fmax(m, mdry)
        
        rho = atmospheric_density(rsafe)
        vrelr, vreltheta, vrel = relative_velocity(vr, vtheta, rsafe)
        
        centrifugal = vtheta**2/rsafe
        gravity = mu/(rsafe**2)
        drag = 0.5*CD*A/msafe*rho*vrel
        rthrust = ur/msafe
        thetathrust = utheta/msafe
        
        ode_rhs = cs.vertcat(
            vr * rscale,
            vtheta/rsafe * thetascale,
            (centrifugal - gravity + rthrust - drag*vrelr) * vscale,
            (-vr*vtheta/rsafe + thetathrust - drag*vreltheta) * vscale,
            (-safe_sqrt(ur**2 + utheta**2)/(Isp*g0)) * mscale
        )
        
        
        self.ODE = {'x':X, 'p':cs.vertcat(dt,U), 'ode': dt*ode_rhs}
        self.multiple_shooting()
        
        _,_,_,_,m_tf_ = cs.vertsplit(self.x_eval[:,-1])
        m_tf = m_tf_/mscale
        self.set_objective(m0 - m_tf)
        
        rT_,_,_,_,_ = cs.vertsplit(self.x_eval[:,-1])
        self.add_constraint(rT_, 0., (rfinal - RE)*rscale)
        
        urt, uthetat = cs.vertsplit(self.u_eval)
        self.add_constraint(safe_sqrt(urt**2 + uthetat**2), 0., umax)
        
        self.build_NLP()
        
        fuel_estimate = 20
        r_init = (np.linspace(r0, rfinal, self.ntS + 1) - RE)*rscale
        theta_init = np.linspace(0, 2*np.pi * n_orbits * 0.8, self.ntS + 1) * thetascale
        vr_init = np.linspace(0., -500., self.ntS + 1) * vscale
        vtheta_init = np.linspace(vorb, vorb*0.85, self.ntS + 1) * vscale
        m_init = np.linspace(m0, m0 - fuel_estimate, self.ntS + 1) * mscale
        for j in range(self.ntS + 1):
            self.set_stage_state(self.start_point, j, [r_init[j], theta_init[j], vr_init[j], vtheta_init[j], m_init[j]])

        for j in range(self.ntS):
            self.set_stage_control(self.start_point, j, [-2.0,-3.0])
            
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        val0, val1 = self.get_stage_control(s, ind)
        self.set_stage_control(s, ind, [val0 + 0.5, val1 + 0.5])
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        RE, rscale, thetascale, vscale, mscale = [self.model_params[key] for key in ['RE', 'rscale', 'thetascale', 'vscale', 'mscale']]
        
        r_, theta_, vr_, vtheta_, m_ = self.get_state_arrays(xi)
        r = r_/rscale + RE
        theta = theta_/thetascale
        # vr = vr_/vscale
        # vtheta = vtheta_/vscale
        m = m_/mscale
        
        ur, utheta = self.get_control_plot_arrays(xi)

        plt.figure(dpi=dpi)
        plt.plot(self.time_grid, (r - RE)/1000, 'r--', label = r'(r - Re)/1000')
        plt.plot(self.time_grid, theta*50, 'g:', label = r'$\theta\cdot 50$')
        # plt.plot(time_grid, vr, 'b-.', label = r'$v_r$')
        # plt.plot(time_grid, vtheta, 'y-', label = r$v_\theta$))
        plt.plot(self.time_grid, (m - 100.)*4, 'b-.', label = r'(m - 100)$\cdot 4$')
        
        plt.axhline(y = 120, color = 'c', linestyle = '--')
        
        plt.step(self.time_grid_ref, ur*20, 'r', label = r'$u_r \cdot 20$')
        plt.step(self.time_grid_ref, utheta*20, 'g', label = r'$u_\theta \cdot 20$')
        
        plt.legend(fontsize='large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Satellite Deorbiting min fuel problem'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
        plt.show()
        plt.close()








