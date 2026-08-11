# blockSQP2 -- A structure-exploiting nonlinear programming solver based
#              on blockSQP by Dennis Janka.
# Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>

# Licensed under the zlib license. See LICENSE for more details.

# \file OCProblems.cpp
# \author Reinhold Wittmann
# \date 2024-2026
#
# Collection of optimal control problems implemented using casadi,
# employing a helper base class for multiple shooting parametrization.
# Used for testing and benchmarking blockSQP2
# 
# The problems originate from various sources, most notably
#  Dolan, E. D., Moré, J. J., & Munson, T. S. (2004). Benchmarking optimization software with COPS 3.0 (No. ANL/MCS-TM-273). Argonne National Lab., Argonne, IL (US).
#  Sager, S. (2005). Numerical methods for mixed-integer optimal control problems. Tönning: Der Andere Verlag.
#  mintoc.de
#  ...

import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
import typing
import math
import copy
# import shutil
# if shutil.which("latex") is not None:
#     plt.rcParams["text.usetex"] = True

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
    NLP : typing.Dict[str, cs.MX]
    
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
    
    #Structure and variable type information (dependency, hessian blocks, ...)##
    #set by multiple_shooting helper method
    hessBlock_sizes : list[int]
    hessBlock_index : list[int]
    vBlock_sizes : list[int]            #partition of variables into blocks
    vBlock_dependencies : list[bool]    #Which blocks are free/dependent
    vBlock_bounds_implicit : list[bool] #Which (dependent) blocks have implicit bounds
    
    cBlock_sizes : list[int]            #Partition of constraints, used to distinguish individual continuity conditions
    ctarget_data : list[int]            #Specifies target for condensing, see blockSQP2/include/blockSQP2/condensing.hpp
    
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
    
    state_bounds_implicit : typing.Iterable[bool]
    
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
    def __init__(self, nt = 100, refine = 1, integrator = 'RK4', parallel = True, N_threads = 4, **kwargs):
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
        self.state_bounds_implicit = [False]*nx
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
    
    def mark_state_bounds_implicit(self, *args):
        if len(args) == 0:
            self.state_bounds_implicit = [True]*self.nx
        elif len(args) == 1:
            if isinstance(args[0], bool):
                self.state_bounds_implicit = [args[0]]*self.nx
            elif isinstance(args[0], int):
                self.state_bounds_implicit = [i == args[0] for i in range(self.nx)]
            elif len(args[0]) == self.nx and all([isinstance(arg0, bool) for arg0 in args[0]]): 
                self.state_bounds_implicit = [*args[0]]
            elif all([isinstance(elem, int) for elem in args[0]]):
                self.state_bounds_implicit = [(i in args[0]) for i in range(self.nx)]
            else:
                raise Exception("Invalid argument")
        elif len(args) == self.nx and all([isinstance(arg, bool) for arg in args]):
            self.state_bounds_implicit = [*args]
        elif all([isinstance(arg, int) for arg in args]):
            self.state_bounds_implicit = [(i in args) for i in range(self.nx)]
        else:
            raise Exception("Invalid argument")
    
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
        self.vBlock_bounds_implicit = [False]
        
        
        vBlock_state_lt = []
        vBlock_state_impl = []
        vBlock_impl_current = self.state_bounds_implicit[0]
        count = 0
        for SBI in self.state_bounds_implicit:
            if SBI != vBlock_impl_current:
                vBlock_state_lt.append(count)
                vBlock_state_impl.append(vBlock_impl_current)
                vBlock_impl_current = SBI
                count = 0
            count += 1
        vBlock_state_lt.append(count)
        vBlock_state_impl.append(vBlock_impl_current)
        
        
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
        # self.vBlock_dependencies = [False]
        
        for i in range(1, self.ntS):
            xopt_arr.append(x_arr[i])
            xopt_arr.append(p_arr[i])
            for j in range(self.ntR):
                xopt_arr.append(u_arr[i*self.ntR + j])
            self.hessBlock_sizes += [self.nx + self.np + self.ntR*self.nu]
            
            # self.vBlock_sizes += [self.nx, self.np + self.ntR*self.nu]
            # self.vBlock_dependencies += [True, False]
            self.vBlock_sizes += vBlock_state_lt + [self.np + self.ntR*self.nu]
            self.vBlock_dependencies += [True]*len(vBlock_state_lt) + [False]
            self.vBlock_bounds_implicit += vBlock_state_impl + [False]
            
            lbv_arr.append(cs.DM(self.lbx + self.lbp + self.lbu*self.ntR))
            ubv_arr.append(cs.DM(self.ubx + self.ubp + self.ubu*self.ntR))        
        
        #Terminal state is a shooting variable
        xopt_arr.append(x_arr[self.ntS])
        self.hessBlock_sizes += [self.nx]
        
        # self.vBlock_sizes += [self.nx]
        # self.vBlock_dependencies += [True]
        self.vBlock_sizes += vBlock_state_lt
        self.vBlock_dependencies += [True]*len(vBlock_state_lt)
        self.vBlock_bounds_implicit += vBlock_state_impl
        
        lbv_arr.append(cs.DM(self.lbx))
        ubv_arr.append(cs.DM(self.ubx))
        
        
        self.hessBlock_index = list(np.cumsum([0] + self.hessBlock_sizes, dtype = np.int32))
        # self.cBlock_sizes = [self.nx]*(self.ntS - 1)
        self.NLP['x'] = cs.vertcat(*xopt_arr)
        self.nVar = self.NLP['x'].numel()
        self.start_point = np.zeros(self.nVar)
        self.lb_var = np.array(cs.vertcat(*lbv_arr), dtype = np.float64).reshape(-1)
        self.ub_var = np.array(cs.vertcat(*ubv_arr), dtype = np.float64).reshape(-1)
        # self.ctarget_data = [self.ntS, 0, 2*self.ntS, 0, self.ntS]
        self.ctarget_data = [self.ntS, 0, (1 + len(vBlock_state_lt))*self.ntS, 0, self.ntS]
    
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
        self.f = lambda xi: float(self._f(xi))
        
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
        
        #Inplace versions (for UNO solver)
        def grad_f_inplace(xi, ret):
            ret[:] = np.array(self._grad_f(xi), dtype = np.float64, copy = False)
        self.grad_f_inplace = grad_f_inplace
        def g_inplace(xi, ret):
            ret[:] = np.array(self._g(xi), dtype = np.float64, copy = False).reshape(-1)
        self.g_inplace = g_inplace
        def jac_g_nz_inplace(xi, ret):
            ret[:] = np.asarray(self._jac_g(xi).nz[:], dtype = np.float64).reshape(-1)
        self.jac_g_nz_inplace = jac_g_nz_inplace
        
        jac_g_col_arr = []
        for i in range(self.nVar):
            jac_g_col_arr.append(np.ones(self.jac_g_colind[i+1] - self.jac_g_colind[i], dtype = np.int64)*i)
        self.jac_g_col = np.concatenate(jac_g_col_arr)
        
        
        objmult = cs.MX.sym('objmult')
        lag_objmult_expr = self.NLP['f']*objmult - lam.T @ g_expr
        grad_lag_objmult_expr = cs.jacobian(lag_objmult_expr, xopt)
        hess_lag_objmult_expr = cs.jacobian(grad_lag_objmult_expr, xopt)
        self._hess_lag_objmult = cs.Function('hess_lag_objmult', [xopt, objmult, lam], [hess_lag_objmult_expr])
        def hess_lag_objmult_inplace(xi, arg_objmult, lambd, ret):
            hlBlocks = self.to_blocks_LT(self._hess_lag_objmult(xi, arg_objmult, lambd))
            ret[:] = np.concatenate(hlBlocks)
        self.hess_lag_objmult_inplace = hess_lag_objmult_inplace
        
        offset = 0
        hess_LT_row_arr = []
        hess_LT_col_arr = []
        self.hess_LT_nnz = 0
        for bsize in self.hessBlock_sizes:
            for j in range(bsize):
                hess_LT_row_arr.append(np.arange(j, bsize) + offset)
                hess_LT_col_arr.append(np.ones(bsize - j, dtype = np.int64)*(j + offset))
            offset += bsize
            self.hess_LT_nnz += (bsize*(bsize+1))//2
        self.hess_LT_row = np.concatenate(hess_LT_row_arr)
        self.hess_LT_col = np.concatenate(hess_LT_col_arr)
        
        
        
    
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
            
    def finish_plot(self, ax, title, it, default_title):
        ax.set_xlabel('t', fontsize = 17.5)
        ax.xaxis.set_label_coords(1.015,-0.006)
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = default_title
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
            
        plt.show()
        plt.close()
    
    
    def perturbed_start_point(self, ind):
        raise NotImplementedError('No perturbed start points implemented for this problem')
    

def add_title(ax, ttl, it = None, default = ""):
    sep = ", "
    if isinstance(ttl,str):
        title = ttl
    elif ttl == True:
        title = default
    else:
        title = ""
        sep = ""
    if isinstance(it, int) and title != "":
        title = title + sep + f'iteration {it}'
    plt.title(title)
    # if title is not None:
    #     if isinstance(it, int):
    #         ttl = ttl + f', iteration {it}'
    #     plt.title(ttl)
    # else:
    #     plt.title('')
    
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
        'tf':12.,
        'auto_init': False
        }
    
    param_set_1 = {
        'c0': 0.4,
        'c1': 0.2,
        'x_init':[0.5,0.7],
        't0':0.,
        'tf':12.0,
        'auto_init': False
        }
    
    param_set_2 = {
        'c0': 0.4,
        'c1': 0.2,
        'x_init':[1.0,0.1],
        't0':0.,
        'tf':16.0,
        'auto_init': False
        }
    
    param_set_3 = {
        'c0': 0.4,
        'c1': 0.2,
        'x_init':[1.0,0.01],
        't0':0.,
        'tf':24.0,
        'auto_init': True
        }
    
    def build_problem(self):
        self.set_OCP_data(2,0,1,1,[0,0],[np.inf, np.inf],[],[],[0],[1])
        
        c0, c1, x_init, t0, tf, auto_init = (self.model_params[key] for key in ('c0', 'c1', 'x_init', 't0', 'tf', 'auto_init'))
        self.fix_time_horizon(t0, tf)
        self.fix_initial_value(x_init)
        self.mark_state_bounds_implicit()
        
        x = cs.MX.sym('x', 2)
        w = cs.MX.sym('w', 1)
        x0, x1 = cs.vertsplit(x)
        ode_rhs = cs.vertcat( x0 - x0*x1 - c0*x0*w, 
                             -x1 + x0*x1 - c1*x1*w)
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
        if auto_init:
            self.integrate_full(self.start_point)
    
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
        
        # ttl = None
        # if isinstance(title,str):
        #     ttl = title
        # elif title == True:
        #     ttl = 'Lotka Volterra fishing problem'
        # if ttl is not None:
        #     if isinstance(it, int):
        #         ttl = ttl + f', iteration {it}'
        #     plt.title(ttl)
        # else:
        #     plt.title('')
        
        # ax.set_xlabel('t', fontsize = 17.5)
        # ax.xaxis.set_label_coords(1.015,-0.006)
        
        # plt.show()
        # plt.close()
        
        self.finish_plot(ax, title, it, "Lotka Volterra fishing problem")


class Lotka_Volterra_Fishing_MAYER(OCProblem):
    default_params = {'c0':0.4, 'c1':0.2, 'x_init':[0.5,0.7], 't0':0., 'tf':12.}
    
    def build_problem(self):
        self.set_OCP_data(3,0,1,0,[0,0,-np.inf],[np.inf, np.inf, np.inf],[],[],[0],[1])
        self.fix_time_horizon(self.model_params['t0'],self.model_params['tf'])
        self.fix_initial_value(self.model_params['x_init']+[0])
        self.mark_state_bounds_implicit()
        
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
        # self.integrate_full(self.start_point)    
        
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
        self.mark_state_bounds_implicit()
        
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
        
        v_eval = self.x_eval[1,1:]
        r_eval = self.x_eval[0,1:]
        
        max_drag_expr = A*(v_eval**2) * cs.exp(-k * (r_eval - r0))
        self.add_constraint(max_drag_expr, -np.inf, C)
        self.add_constraint(r_eval[-1], rT, np.inf)
        
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
        
        self.finish_plot(ax, title, it, 'Goddard\'s rocket problem')


#Goddard rocket with only state, control bounds and terminal constraints by adding the air friction as a differential state.
#Formulated by a fellow PhD.
class Goddard_Rocket_MOD(Goddard_Rocket):
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
        
        fig, ax = plt.subplots(dpi = dpi)
        ax.plot(time_grid, (r - 1)*100, 'b--', label = r'$(r-1)\cdot 100$')
        ax.plot(time_grid, v*20, 'g:', label = r'$v\cdot 20$')
        ax.plot(time_grid, m, 'y-.', label = '$m$')
        
        ax.step(time_grid, u, 'r', label = '$u$')
        ax.legend(fontsize = 'large')
        
        self.finish_plot(ax, title, it, "Goddard\'s rocket problem")
        

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
    def __init__(self, nt = 100, refine = 1, integrator = 'cvodes', parallel = False, N_threads = 4, **kwargs):
        OCProblem.__init__(self, nt=nt, refine=refine, integrator=integrator, parallel=parallel, N_threads = N_threads, **kwargs)
    
    def build_problem(self):
        self.set_OCP_data(4,1,1,1,[0,0,0,0],[np.inf,np.inf,np.inf,np.inf],[1.1], [1.3], [1], [np.inf])
        self.fix_initial_value([0.03966, 1.09799, 0.00142, 1.65431])
        self.mark_state_bounds_implicit()
        
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
        
        #Prevent local minimum with second stimulus (but better objective)
        for i in range(math.floor(0.4*self.ntS), self.ntS):
            self.set_stage_control(self.ub_var, i, 1.0)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        val = self.get_stage_control(s, ind)
        self.set_stage_control(s, ind, val + 0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x0, x1, x2, x3 = self.get_state_arrays_expanded(xi)
        w = self.get_control_plot_arrays(xi)        
        
        fig, ax = plt.subplots(dpi = dpi)
        ax.plot(self.time_grid_ref, x0, 'tab:olive', linestyle = '--', label = r'$x_0$')
        ax.plot(self.time_grid_ref, x1, 'tab:green', linestyle = '-.', label = r'$x_1$')
        ax.plot(self.time_grid_ref, x2, 'tab:cyan', linestyle = ':', label = r'$x_2$')
        ax.plot(self.time_grid_ref, x3, 'tab:blue', linestyle = '-.', label = r'$x_3$')
        ax.step(self.time_grid_ref, (w-1.0)*20, 'tab:red', label = r'$(w-1)\cdot 20$')
        ax.legend(fontsize='large')
        
        self.finish_plot(ax, title, it, 'Calcium Oscillation problem')


class Batch_Reactor(OCProblem):
    default_params = {}
    def build_problem(self):
        self.set_OCP_data(2, 0, 1, 0, [-np.inf,-np.inf], [np.inf,np.inf], [], [], [298],[398])
        self.mark_state_bounds_implicit()
        
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
        self.mark_state_bounds_implicit(True, False, True)
        
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
        self.set_OCP_data(1,0,1,2, [0.], [10.], [], [], [-10.], [20.])
        
        a,b,Lp = (self.model_params[key] for key in ['a', 'b', 'Lp'])
        self.fix_initial_value([a])
        self.fix_time_horizon(0,1)
        self.mark_state_bounds_implicit()
        
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
        self.set_OCP_data(3,0,1,0,[0., -np.inf, -np.inf], [10., np.inf, np.inf], [], [], [-10.], [20.])
        self.mark.state_bounds_implicit([False, True, True])
        
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
    default_params = {
        'alpha': 10
        }
    
    # param_set_1 = {
    #     'alpha': 10
    #     }
    
    # param_set_2 = {
    #     'alpha': 4
    #     }
    
    # param_set_3 = {
    #     'alpha': 25
    #     }
    def build_problem(self):
        self.set_OCP_data(2,0,1,0,[-np.inf,-np.inf],[np.inf,np.inf],[],[],[0.],[1.])
        self.fix_time_horizon(0,1)
        self.fix_initial_value([1.,0.])
        self.mark_state_bounds_implicit()
        
        alpha = self.model_params['alpha']
        
        x = cs.MX.sym('x', 2)
        x1,x2 = cs.vertsplit(x)
        w = cs.MX.sym('w',1)
        dt = cs.MX.sym('dt', 1)
        ode_rhs = cs.vertcat(w*(alpha*x2-x1), w*(x1 - alpha*x2) - (1-w)*x2)
        
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
        x1,x2 = self.get_state_arrays_expanded(xi)
        u = self.get_control_plot_arrays(xi)
        ax.plot(self.time_grid_ref, x1, 'tab:green', linestyle='-.', label = r'$x_1$')
        ax.plot(self.time_grid_ref, x2, 'tab:blue', linestyle='--', label = r'$x_2$')
        ax.step(self.time_grid_ref, u, 'tab:red', linestyle='-', label = r'$u$')
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
    default_params = {
        'm':5.,
        'c':10.,
        'x0':2.,
        'v0':5.,
        'umm':5.}
    
    def build_problem(self):
        m,c,x0,v0,umm = (self.model_params[key] for key in ['m', 'c', 'x0', 'v0', 'umm'])
        self.set_OCP_data(2,1,1,0,[-np.inf,-np.inf], [np.inf,np.inf], [8/self.ntS],[20/self.ntS], [-umm], [umm])
        self.mark_state_bounds_implicit()
        
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
        
        self.finish_plot(ax, title, it, 'Cushioned oscillation problem')
        
        
class Cushioned_Oscillation_TSCALE(Cushioned_Oscillation):
    default_params = {'m':5.,'c':10.,'x0':2.,'v0':5.,'umm':5., 'TSCALE':100.0}
    def build_problem(self):
        m,c,x0,v0,umm,TSCALE = (self.model_params[key] for key in ['m', 'c', 'x0', 'v0', 'umm', 'TSCALE'])
        self.set_OCP_data(2,1,1,0,[-np.inf,-np.inf], [np.inf,np.inf], [8/self.ntS * TSCALE],[20/self.ntS * TSCALE], [-umm], [umm])
        self.mark_state_bounds_implicit()
        
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
        'tF': 6.
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
    
    def build_problem(self):
        zeta, b, mu, d, G, x20, x30, u0max, x2max, x00, x10, u1max, x3max, F, eta, alpha, tF = (self.model_params[key] for key in ('zeta','b','mu','d','G','x20','x30','u0max','x2max','x00','x10','u1max','x3max','F','eta', 'alpha', 'tF'))
        self.set_OCP_data(2,0,2,3,[0.,0.], [np.inf,np.inf], [], [], [0.,0.],[u0max,u1max])
        self.fix_initial_value([x00,x10])
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
        self.ODE = {'x':x, 'p':cs.vertcat(dt,u), 'ode': dt*ode_rhs, 'quad':dt*quad}
        self.multiple_shooting()
        self.set_objective(self.x_eval[0,-1] + alpha*self.q_tf[0])
        self.add_constraint(self.q_tf[1:3] - cs.DM([x2max,x3max]), [-np.inf,-np.inf], [0.,0.])
        self.build_NLP()
        
        for i in range(self.ntS):
            self.set_stage_control(self.start_point, i, [x2max/tF, x3max/tF])
        self.integrate_full(self.start_point)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, [0.1, 0.1])
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        time_grid = self.time_grid
        x0,x1 = self.get_state_arrays(xi)
        u0,u1 = self.get_control_plot_arrays(xi)
        
        fig, ax = plt.subplots(dpi = dpi)
        ax.plot(time_grid, x0/100., 'tab:green', linestyle = '--', label = r'$x_0/100$')
        ax.plot(time_grid, x1/100., 'tab:blue', linestyle = ':', label = r'$x_1/100$')
        ax.step(time_grid, u0, 'tab:red', label = r'$u_0$')
        ax.step(time_grid, u1*75, 'tab:blue', linestyle = '-.', label = r'$u_1\cdot75$')
        ax.legend(fontsize='large', loc = 'upper right')
        
        self.finish_plot(ax, title, it, 'D\'Onofrio chemotherapy problem')
    
# class D_Onofrio_Chemotherapy_VT(OCProblem):
#     default_params = {'zeta':0.192, 'b':5.85, 'mu': 0.0, 'd':0.00873, 'G':0.15, 'x20':0.0, 'x30':0.0, 'u0max':75., 'x2max':300., 'x00':12000., 'x10':15000., 'u1max':1., 'x3max':2., 'F':1., 'eta':1., 'alpha':0.}
#     param_set_1 = {
#         'x00': 12000,
#         'x10': 15000,
#         'u1max': 1,
#         'x3max': 2
#     }
    
#     param_set_2 = {
#         'x00': 12000,
#         'x10': 15000,
#         'u1max': 2,
#         'x3max': 10
#     }
    
#     param_set_3 = {
#         'x00': 14000,
#         'x10': 5000,
#         'u1max': 1,
#         'x3max': 2
#     }
    
#     param_set_4 = {
#         'x00': 14000,
#         'x10': 5000,
#         'u1max': 2,
#         'x3max': 10
#     }
    
#     def __init__(self, nt = 20, refine = 1, integrator = 'rk4', parallel = True, N_threads = 4, **kwargs):
#         OCProblem.__init__(self,nt=nt,refine=refine,integrator=integrator,parallel=parallel, N_threads = N_threads, **kwargs)

    
#     def build_problem(self):
#         zeta, b, mu, d, G, x20, x30, u0max, x2max, x00, x10, u1max, x3max, F, eta, alpha = (self.model_params[key] for key in ('zeta','b','mu','d','G','x20','x30','u0max','x2max','x00','x10','u1max','x3max','F','eta', 'alpha'))
#         self.set_OCP_data(4,1,2,1,[0.1,0.1,0.,0.], [np.inf,np.inf,x2max,x3max], [4/self.ntS], [20/self.ntS], [0.,0.],[u0max,u1max])
#         #Note: Lower bounds 0.1,0.1 for differential states required as integrations fails at 0, 0 due to numerical errors causing negative states
#         self.fix_initial_value([x00,x10,x20,x30])
#         # self.fix_time_horizon(0, 6.0)
        
#         x = cs.MX.sym('x', 4)
#         x0,x1,x2,x3 = cs.vertsplit(x)
#         u = cs.MX.sym('u', 2)
#         u0,u1 = cs.vertsplit(u)
#         p = cs.MX.sym('p', 1)
#         dt = p
#         # dt = cs.MX.sym('dt')
        
#         ode_rhs = cs.vertcat(-zeta*x0*cs.log(x0/x1) - F*x0*u1,
#                              b*x0 - mu*x1 - d*x0**(2./3.)*x1 - G*u0*x1 - eta*x1*u1,
#                              u0,
#                              u1
#                              )
#         quad = u0**2
#         self.ODE = {'x':x, 'p':cs.vertcat(dt,u), 'ode': dt*ode_rhs, 'quad':dt*quad}
#         self.multiple_shooting()
#         self.set_objective(20*self.p_tf*self.ntS+self.x_eval[0,-1] + alpha*self.q_tf)
#         self.build_NLP()
#         self.set_stage_param(self.start_point, 0, 4/self.ntS)
#         for i in range(1,self.ntS):
#             self.set_stage_param(self.start_point, i, 6/self.ntS)
#         self.integrate_full(self.start_point)
    
#     def perturbed_start_point(self, ind):
#         s = copy.copy(self.start_point)
#         self.set_stage_control(s, ind, [0.1,0.1])
#         return s
    
#     def plot(self, xi, dpi = None, title = None, it = None):
#         p = self.get_param_arrays(xi)
#         time_grid = np.cumsum(np.concatenate([[0], p]))
#         # time_grid = self.time_grid
#         x0,x1,x2,x3 = self.get_state_arrays(xi)
#         u0,u1 = self.get_control_plot_arrays(xi)
        
#         plt.figure(dpi = dpi)
#         plt.plot(time_grid, x0/100., 'r-', label = 'x0/100')
#         plt.plot(time_grid, x1/100., 'g-', label = 'x1/100')
#         plt.plot(time_grid, x2, 'b-', label = 'x2')
#         plt.plot(time_grid, x3, 'c-', label = 'x3')
#         plt.step(time_grid, u0, 'r-', label = 'u0')
#         plt.step(time_grid, u1*75, 'g-', label = 'u1*75')
#         plt.legend(fontsize='large')
        
#         ttl = None
#         if isinstance(title,str):
#             ttl = title
#         elif title == True:
#             ttl = 'D\'Onofrio chemotherapy problem'
#         if ttl is not None:
#             if isinstance(it, int):
#                 ttl = ttl + f', iteration {it}'
#             plt.title(ttl)
#         else:
#             plt.title('')
            
#         plt.show()  
#         plt.close()
             
        

class Egerstedt_Standard(OCProblem):
    default_params = {
        'x_init': [0.5, 0.5]
    }
    def build_problem(self):
        self.set_OCP_data(2,0,3,1, [-np.inf, 0.4], [np.inf,np.inf], [], [], [0.,0.,0.], [1.,1.,1.])
        self.fix_time_horizon(0.,1.)
        x_init = self.model_params['x_init']
        self.fix_initial_value(x_init)
        self.mark_state_bounds_implicit(True, False)
        
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
        self.mark_state_bounds_implicit(True, False, True)
        
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
        self.mark_state_bounds_implicit()
        
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
        self.mark_state_bounds_implicit([1,2])
        
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
        self.start_point = np.maximum(self.start_point, self.lb_var)
        self.start_point = np.minimum(self.start_point, self.ub_var)
    
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
        
        ax.legend(fontsize='x-large')
        
        ax.set_xlabel('t', fontsize = 17.5)
        ax.xaxis.set_label_coords(1.015,-0.006)
        
        
        ax.tick_params(axis='both', which='major', labelsize='x-large')
        
        plt.show()
        plt.close()
        
# Several local minima, see mintoc.de
class F8_Aircraft(OCProblem):
    def build_problem(self):
        self.set_OCP_data(3,1,1,0,[-np.inf,-np.inf,-np.inf], [np.inf,np.inf,np.inf], [1/self.ntS], [100/self.ntS], [-0.05236], [0.05236])
        self.fix_initial_value([0.4655,0.,0.])
        self.mark_state_bounds_implicit()
        
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
        x0,x1,x2 = self.get_state_arrays_expanded(xi)
        w = self.get_control_plot_arrays(xi)
        p = self.get_param_arrays_expanded(xi)
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
        OCProblem.__init__(self, nt=nt, refine=refine, integrator=integrator, parallel=parallel, N_threads = N_threads, **kwargs)

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
        self.add_constraint(cs.cumsum(self.q_eval, 1), 0., np.inf)
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

# Many local solutions with similar objective value
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
        self.mark_state_bounds_implicit()
        
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
        self.mark_state_bounds_implicit()
        
        a = self.model_params['a']
        x = cs.MX.sym('x',4)
        x1,x2,dx1,dx2 = cs.vertsplit(x)
        u = cs.MX.sym('u')
        dt = cs.MX.sym('dt')
        
        ode_rhs = cs.vertcat(dx1,
                             dx2,
                             a*cs.cos(u),
                             a*cs.sin(u),
                             )
        
        self.ODE = {'x': x, 'p': cs.vertcat(dt,u), 'ode': dt*ode_rhs}
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
        x1,x2,dx1,dx2 = self.get_state_arrays(xi)
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
        self.mark_state_bounds_implicit(0,1,  3,4,5)
        
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
    
#Seems infeasible, maybe verify problem formulation.
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
        OCProblem.__init__(self, nt=nt, refine=refine, integrator=integrator, parallel=parallel, N_threads = N_threads, **kwargs)

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
            self.set_stage_control(self.start_point, i, [1.0,1.0,1.0])
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
    default_params = {'T':12, 
                      'c1':1, 
                      'c2':2, 
                      'c3':0.8, 
                      'k1':2, 
                      'k2':3, 
                      'k3':1, 
                      'k4':3}
    
    param_set_1 = {'k2':3, 
                   'k4':3}
    
    param_set_2 = {'k2':2, 
                   'k4':4}
    
    def build_problem(self):
        self.set_OCP_data(3,0,3,1,[0.,0.,0.], [np.inf,np.inf,np.inf], [],[], [0.,0.,0.], [1.,1.,1.])
        self.fix_time_horizon(0, self.model_params['T'])
        self.fix_initial_value([2.,2.,2.])
        self.mark_state_bounds_implicit()
        
        c1, c2, c3, k1, k2, k3, k4 = (self.model_params[key] for key in ['c1', 'c2', 'c3', 'k1', 'k2', 'k3', 'k4'])
        
        x = cs.MX.sym('x',3)
        x1,x2,x3 = cs.vertsplit(x)
        u = cs.MX.sym('u',3)
        w1,w2,w3 = cs.vertsplit(u)
        dt = cs.MX.sym('dt')
        
        ode_rhs = cs.vertcat(-cs.sqrt(x1) + c1*w1+c2*w2 - w3*cs.sqrt(c3*x1),
                              cs.sqrt(x1) - cs.sqrt(x2),
                              cs.sqrt(x2) - cs.sqrt(x3) + w3*cs.sqrt(c3*x1)
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
        self.mark_state_bounds_implicit()
        
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
    
    param_set_1 = {'vmax': 33.}
    param_set_2 = {'vmax': 15}
    
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

# \dot{x}, \dot{y} = y, u*(1-x**2)*y - x
class Van_der_Pol_Oscillator(OCProblem):
    def __init__(self, nt = 100, refine = 1, integrator = 'cvodes', parallel = False, N_threads = 4, **kwargs):
        OCProblem.__init__(self, nt=nt, refine=refine, integrator=integrator, parallel=parallel, N_threads = N_threads, **kwargs)

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

# \dot{x}, \dot{y} = y, (1-x**2)*y - x + u
class Van_der_Pol_Oscillator_2(OCProblem):
    def __init__(self, nt = 100, refine = 1, integrator = 'cvodes', parallel = False, N_threads = 4, **kwargs):
        OCProblem.__init__(self, nt=nt, refine=refine, integrator=integrator, parallel=parallel, N_threads = N_threads, **kwargs)

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

# \dot{x}, \dot{y} = y, (1-x**2)*y - x + u
class Van_der_Pol_Oscillator_3(OCProblem):
    def __init__(self, nt = 100, refine = 1, integrator = 'cvodes', parallel = False, N_threads = 4, **kwargs):
        OCProblem.__init__(self, nt=nt, refine=refine, integrator=integrator, parallel=parallel, N_threads = N_threads, **kwargs)

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
        OCProblem.__init__(self, nt=nt, refine=refine, integrator=integrator, parallel=parallel, N_threads = N_threads, **kwargs)

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
        OCProblem.__init__(self, nt=nt, refine=refine, integrator=integrator, parallel=parallel, N_threads = N_threads, **kwargs)

    
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
        
        ode_rhs = cs.vertcat(u1 - u2 - gamma*(S - omega*DL), 
                             -u1, 
                             cs.DM(1.))
        
        quad = cs.exp(-rho*t)*(U - A - u1*C - D)
        self.ODE = {'x':x, 'p':cs.vertcat(dt,u), 'ode':dt*ode_rhs, 'quad':dt*quad}
        self.multiple_shooting()
        self.set_objective(-self.q_tf)
        self.build_NLP()
        
        for i in range(self.ntS):
            self.set_stage_control(self.start_point, i, [30.,10.])
            self.set_stage_state(self.start_point, i, self.x_init)
        self.set_stage_state(self.start_point, self.ntS, self.x_init)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        s_ind = self.get_stage_control(s, ind)
        self.set_stage_control(s, ind, s_ind + cs.DM([1.0, 1.0]))
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        S,R,_ = self.get_state_arrays(xi)
        u1,u2 = self.get_control_plot_arrays(xi)
        
        S0, R0 = (self.model_params[key] for key in ('S0', 'R0'))
        
        fig, ax = plt.subplots(dpi = dpi)
        ax.plot(self.time_grid, S-S0, 'tab:green', linestyle = ':', label = 'S-2000')
        ax.plot(self.time_grid, (R-R0)/100, 'tab:blue', linestyle = '-.', label = 'R/1000')
        ax.step(self.time_grid_ref, u1*5, 'tab:red', label = r'$u_1\cdot 10$')
        ax.step(self.time_grid_ref, u2*5, 'tab:grey', linestyle = '-.', label = r'$u_2\cdot 10$')
        ax.legend(fontsize='large')
        
        self.finish_plot(ax, title, it, "Ocean problem")
        # ttl = None
        # if isinstance(title,str):
        #     ttl = title
        # elif title == True:
        #     ttl = 'Ocean problem'
        # if ttl is not None:
        #     if isinstance(it, int):
        #         ttl = ttl + f', iteration {it}'
        #     plt.title(ttl)
        # else:
        #     plt.title('')
            
        # plt.show()


class Lotka_OED(OCProblem):
    default_params = {
        'tf':12, 
        'p1':1,
        'p2':1,
        'p3':1,
        'p4':1,
        'p5':0.4,
        'p6':0.2,
        'x_init':[0.5,0.7],
        'M':4.0,
        'fishing':True,
        'epsilon': 0.0,
        'transform_obj':False
        }
    
    param_set_2 = {
        'tf': 20
        }
    
    param_set_3 = {
        'x_init': [1.0, 0.5]
        }
    
    def build_problem(self):
        self.set_OCP_data(9, 0, 3, 2, [0.,0.]+[-np.inf]*7, [np.inf]*9,[],[],[0.] + [0.]*2, [float(self.model_params['fishing'])] + [1.]*2)
        tf,p1,p2,p3,p4,p5,p6,x_init,M,epsilon, transform_obj= (self.model_params[key] for key in ['tf', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6','x_init', 'M', 'epsilon', 'transform_obj'])
        self.fix_time_horizon(0.,tf)
        self.fix_initial_value(x_init + [0.]*4 + [epsilon, 0., epsilon])
        self.mark_state_bounds_implicit()
        
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
        
        obj_expr = (1/(F11T*F22T - F12T*F12T))*(F22T + F11T)
        if transform_obj:
            self.set_objective(-obj_expr**-2)
        else:
            # self.set_objective((1/(F11T*F22T - F12T*F12T))*(F22T + F11T))
            self.set_objective(obj_expr)
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
        x1, x2, G11, G12, G21, G22, F11, F12, F22 = self.get_state_arrays_expanded(xi)
        
        fig, ax = plt.subplots(dpi=dpi)
        ax.plot(self.time_grid_ref, x1, 'tab:olive', linestyle='-.', label = r'$x_1$')
        ax.plot(self.time_grid_ref, x2, 'tab:cyan', linestyle='-.', label = r'$x_2$')
        ax.step(self.time_grid_ref, u, 'tab:red', linestyle='-', label = r'$u$')
        ax.step(self.time_grid_ref, w1, 'tab:blue', linestyle=':', label = r'$w_1$')
        ax.step(self.time_grid_ref, w2, 'tab:green', linestyle='--', label = r'$w_2$')
        
        # ax.set_ylim(0.,4.)
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
        OCProblem.__init__(self, nt=nt, refine=refine, integrator=integrator, parallel=parallel, N_threads = N_threads, **kwargs)

    def build_problem(self):
        self.set_OCP_data(6, 0, 3, 3, [0.,0.,0.,0.,0.3,0.], [0.1,0.04,0.03,0.1,0.45,0.1], [], [], [0.,0.,0.], [15.,1.,30.])
        mux, mup, gxg, gx1, gp1, gx2, gp2 = (self.model_params[key] for key in ['mux', 'mup', 'gxg', 'gx1', 'gp1', 'gx2', 'gp2'])
        self.fix_time_horizon(0.,1.)
        self.fix_initial_value([0.,0.03,0.03,0.01,0.3,0.1])
        x = cs.MX.sym('x', 6)
        P,S1,S2,E,V,G = cs.vertsplit(x)
        u = cs.MX.sym('u', 3)
        uS1,uS2,uP = cs.vertsplit(u)
        dt = cs.MX.sym('dt', 1)
        
        #In Le, first term in rhs for S1, S2 and G enters with positive sign, negative in Janka and MUSCOD
        #Janka and MUSCOD seem to be correct
        
        Pdot = mup*E*S1*S2 - P*(uS1+uS2)/(25*V)
        ode_rhs = cs.vertcat(
                Pdot,
                -gx1*E*S1*S2*G - gp1*E*S1*S2 + (0.42*uS1 - S1*(uS1 + uS2))/(25*V),
                -gx2*E*S1*S2*G - gp2*E*S1*S2 + (0.333*uS2 - S2*(uS1 + uS2))/(25*V),
                mux*E*S1*S2*G - E*(uS1 + uS2)/(25*V),
                uS1 + uS2 - uP,
                -gxg*E*S1*S2*G - G*(uS1+uS2)/(25*V),
        )
        
        quad = cs.vertcat(uP*P + (uS1 + uS2 - uP)/25 * P + V*Pdot,
                0.0168*uS1,
                0.01332*uS2)
        
        self.ODE = {'x':x, 'p':cs.vertcat(dt, u), 'ode':dt*ode_rhs, 'quad': dt*quad}
        self.multiple_shooting()
        
        P_acc, S1_acc, S2_acc = cs.vertsplit(self.q_tf + cs.DM([0., 0.009, 0.009]))
        
        # self.set_objective(2*(self.x_eval[7,-1]*self.x_eval[8,-1])/self.x_eval[6,-1])
        self.set_objective(2*(S1_acc*S2_acc)/P_acc)
        
        self.add_constraint(cs.cumsum(self.q_eval, 1), np.array([0.,0.,0.]) - np.array([0., 0.009, 0.009]), np.array([0.05,0.2,0.025]) - np.array([0., 0.009, 0.009]))

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
        P,S1,S2,E,V,G = self.get_state_arrays(xi)
        uS1,uS2,uP = self.get_control_plot_arrays(xi)
        
        fig, ax = plt.subplots(dpi=dpi)
        ax.plot(self.time_grid, P*10., 'tab:red', linestyle = '--', label = r'$P\cdot 10$')
        ax.plot(self.time_grid, S1*2, 'tab:green', linestyle = '--', label = r'$S1\cdot 2$')
        ax.plot(self.time_grid, S2*2, 'tab:brown', linestyle = '--', label = r'$S2\cdot 2$')
        ax.plot(self.time_grid, E, 'tab:olive', linestyle = '--', label = 'E')
        ax.plot(self.time_grid, V/3, 'tab:cyan', linestyle = '--', label = 'V/3')
        ax.plot(self.time_grid, G, 'tab:purple', linestyle = '--', label = 'G')
        
        ax.step(self.time_grid_ref, uS1/5., 'tab:red', label = r'$u_{S1}/5$')
        ax.step(self.time_grid_ref, uS2/15., 'tab:green', label = r'$u_{S2}/15$')
        ax.step(self.time_grid_ref, uP/60., 'tab:grey', label = r'$u_{P}/60$')
        
        ax.legend(fontsize='medium', loc = 'upper center')
        ax.set_ylim(0, 0.16)
        
        self.finish_plot(ax, title, it, "Fermenter problem")


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
    M0scale = 1.0
    # M0scale = 1e-2
    MDscale = 1.0
    # MDscale = 1e-2
    xDscale = 1.0
    # xDscale = 1e2
    
    tscale = 1.0
    Rscale = 1.0
    # M0scale = 1.0
    # MDscale = 1.0
    # xDscale = 1.0
    xCscale = 1.0
    x0scale = 1.0
    
    def __init__(self, nt = 100, refine = 1, integrator = 'cvodes', parallel = False, N_threads = 4, **kwargs):
        OCProblem.__init__(self, nt=nt, refine=refine, integrator=integrator, parallel=parallel, N_threads = N_threads, **kwargs)
    
    def build_problem(self):
        M0init, MDinit, x0init, xinit, xCinit, xDinit, alpha, V, m, mC = (self.model_params[key] for key in ['M0init', 'MDinit', 'x0init', 'xinit', 'xCinit', 'xDinit', 'alpha', 'V', 'm', 'mC'])
        # self.set_OCP_data(10,1,1,0, [0.]*9 + [MDinit],[np.inf]*10,[0.5/self.ntS * self.tscale],[10/self.ntS * self.tscale], [0. * self.Rscale], [15. * self.Rscale])
        self.set_OCP_data(10,1,1,0, [0.]*8 + [0.] + [MDinit*self.MDscale],[np.inf] + [self.x0scale] + [1.0]*5 + [self.xCscale, self.xDscale] + [np.inf],[0.5/self.ntS * self.tscale],[10/self.ntS * self.tscale], [0. * self.Rscale], [15. * self.Rscale])
        self.fix_initial_value([M0init*self.M0scale,x0init*self.x0scale] + [xinit]*5+[xCinit*self.xCscale,xDinit*self.xDscale,MDinit*self.MDscale])
        
        X = cs.MX.sym('X',10)
        M0_,x0_,x1,x2,x3,x4,x5,xC_,xD_,MD_ = cs.vertsplit(X)
        M0 = M0_/self.M0scale
        MD = MD_/self.MDscale
        x0 = x0_/self.x0scale
        xC = xC_/self.xCscale
        xD = xD_/self.xDscale
        
        R_ = cs.MX.sym('R')
        R = R_/self.Rscale
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
            self.set_stage_control(self.start_point, j, 1.0*self.Rscale)
        for j in range(math.floor(0.5*self.ntS), self.ntS):
            self.set_stage_control(self.start_point, j, 15.0*self.Rscale)
        self.integrate_full(self.start_point)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        val = self.get_stage_control(s, ind)
        if ind < math.floor(0.5*self.ntS):
            self.set_stage_control(s, ind, val + 1.0*self.Rscale)
        else:
            self.set_stage_control(s, ind, val - 1.0*self.Rscale)
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
        self.set_OCP_data(4,1,1,0, [0.,0.,-np.inf,-np.inf], [np.inf,np.inf,np.inf,np.inf], [75/self.ntS], [1500/self.ntS], [0], [cmax])
        self.fix_initial_value([x0, dxbc, y0, dybc])
        self.mark_state_bounds_implicit(False,False,True,True)
        
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
        x, dx, y, dy = self.get_state_arrays_expanded(xi)
        cL = self.get_control_plot_arrays(xi)
        p = self.get_param_arrays_expanded(xi)
        time_grid = np.cumsum(np.concatenate([[0], p])).reshape(-1)
        
        plt.figure(dpi=dpi)
        plt.step(time_grid, cL, 'tab:red', label = r'$c_L$')
        plt.plot(time_grid, x/500, 'tab:green', linestyle = '-', label = r'$x/500$')
        plt.plot(time_grid, (y-900)/100, 'tab:blue', linestyle = '-', label = r'$(y-900)/100$')
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
        self.mark_state_bounds_implicit()
        x = cs.MX.sym('x', 1)
        u = cs.MX.sym('u', 1)
        dt = cs.MX.sym('dt', 1)
        ode_rhs = -(u + 0.5 * u**2) * x
        quad_expr = u * x
        self.ODE = {'x':x, 
                    'p':cs.vertcat(dt,u), 
                    'ode': dt*ode_rhs, 
                    'quad': dt*quad_expr
                    }
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
        self.mark_state_bounds_implicit()
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
        self.mark_state_bounds_implicit()
        
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
        self.mark_state_bounds_implicit()
        
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
    param_set_3 = {'u_max': 30, 'lambda_u': 0.05} #causes blockSQP2 to run into local optimum, ipopt's central path leads to (likely) global optimum
    
    def build_problem(self):
        M, m, l, g, u_max, lambda_u = (self.model_params[key] for key in ['M','m','l','g','u_max','lambda_u'])
        
        self.set_OCP_data(4,0,1,0, [-2.0, -np.inf, -np.inf, -np.inf], [2.0, np.inf, np.inf, np.inf], [], [], [-u_max], [u_max])
        self.fix_time_horizon(0., 4.0)
        self.fix_initial_value([0., 0., 0., 0.])
        self.mark_state_bounds_implicit(1,2,3)
        
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
        # self.integrate_full(self.start_point)
    
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



class Dielectrophoretic_Particle(OCProblem):
    default_params = {
        'x00': 1.,
        'xf': 2.,
        'alpha':-0.75,
        'c':1.
        }
    
    def build_problem(self):
        self.set_OCP_data(2,1,1,0,[-np.inf,-np.inf],[np.inf, np.inf],[0.01],[np.inf],[-1],[1])
        x00,xf,alpha,c = (self.model_params[key] for key in ('x00', 'xf', 'alpha', 'c'))
        self.fix_initial_value([x00, 0.])
        self.mark_state_bounds_implicit()
        
        x = cs.MX.sym('x', 2)
        u = cs.MX.sym('u', 1)
        x0, x1 = cs.vertsplit(x)
        ode_rhs = cs.vertcat(x1*u + alpha*u**2, -c*x1 + u)
        dt = cs.MX.sym('dt', 1)
        self.ODE = {'x': x, 'p':cs.vertcat(dt, u),'ode': dt*ode_rhs}
        self.multiple_shooting()
        self.set_objective(self.p_tf*self.ntS)
        self.add_constraint(self.x_eval[0,-1], xf, xf)
        self.build_NLP()
        
        self.start_point = np.zeros(self.nVar)
        # for i in range(self.ntS+1):
        #     self.set_stage_state(self.start_point, i, self.x_init)
        for i in range(self.ntS):
            self.set_stage_control(self.start_point, i, [1.0])
            self.set_stage_param(self.start_point, i, 5.0/self.ntS)
        self.integrate_full(self.start_point)
        
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, 0.9)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x0, x1 = self.get_state_arrays_expanded(xi)
        u = self.get_control_plot_arrays(xi)
        p = self.get_param_arrays_expanded(xi)
        time_grid_ref = np.cumsum(np.concatenate([np.array([0]), p]))
        
        # plt.figure(dpi = dpi)
        fig,ax = plt.subplots(dpi=dpi)
        ax.plot(time_grid_ref, x0, 'tab:green', linestyle='-.', label = '$x_0$')
        ax.plot(time_grid_ref, x1, 'tab:blue', linestyle='--', label = '$x_1$')
        ax.step(time_grid_ref, u, 'tab:red', linestyle='-', label = r'$u$')
        ax.legend(fontsize='x-large')
        self.finish_plot(ax, title, it, 'Dielectrophoretic Particle problem')

        
class Double_Oscillator(OCProblem):
    default_params = {
        'm1': 100.,
        'm2': 2.,
        'k1': 100.,
        'k2': 3.,
        'c': 0.5,
        'T': 2*np.pi
        }
    
    def build_problem(self):
        self.set_OCP_data(5,0,1,1,[-np.inf]*5,[np.inf]*5,[],[],[-1],[1])
        m1, m2, k1, k2, c, T = (self.model_params[key] for key in self.default_params.keys())
        self.fix_initial_value([0., 0., None, None, 0.])
        self.fix_time_horizon(0,T)
        self.mark_state_bounds_implicit()
        x = cs.MX.sym('x', 4+1)
        u = cs.MX.sym('u', 1)
        x0, x1, dx0, dx1,t = cs.vertsplit(x)
        ode_rhs = cs.vertcat(dx0, 
                             dx1, 
                             -(k1+k2)/m1 * x0 + k2/m2 * x1 + 1/m1*cs.sin(2*np.pi/T * t),
                             k2/m2 * x0 - k2/m2 * x1 - c*(1-u)/m2 * dx1,
                             cs.DM(1)
                             )
        quad = 0.5*(x0**2 + x1**2 + u**2)
        dt = cs.MX.sym('dt', 1)
        self.ODE = {'x': x, 'p':cs.vertcat(dt, u),'ode': dt*ode_rhs, 'quad': dt*quad}
        self.multiple_shooting()
        self.set_objective(self.q_tf)
        self.build_NLP()
        
        self.start_point = np.zeros(self.nVar)
        self.set_stage_state(self.start_point, 0, [0.,0.])
        # for i in range(self.ntS+1):
        #     self.set_stage_state(self.start_point, i, self.x_init)
        for i in range(self.ntS):
            self.set_stage_control(self.start_point, i, [0.5])
        self.integrate_full(self.start_point)
        
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, 0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x0, x1, dx1, dx2, _ = self.get_state_arrays_expanded(xi)
        u = self.get_control_plot_arrays(xi)
        
        # plt.figure(dpi = dpi)
        fig,ax = plt.subplots(dpi=dpi)
        ax.plot(self.time_grid_ref, x0, 'tab:green', linestyle='-.', label = '$x_0$')
        ax.plot(self.time_grid_ref, x1, 'tab:blue', linestyle='--', label = '$x_1$')
        ax.step(self.time_grid_ref, u*1000, 'tab:red', linestyle='-', label = r'$u\cdot 1000$')
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
    

class Ducted_Fan(OCProblem):
    default_params = {
        'm': 2.2,
        'J': 0.05,
        'r': 0.2,
        'mg': 4.,
        'mu': 4
        }
    
    def build_problem(self):
        self.set_OCP_data(6,1,2,1,[-np.inf]*2 + [-30] + [-np.inf]*3,[np.inf]*2 + [30] + [np.inf]*3,[1.0/self.ntS],[8.0/self.ntS],[-5., 0.],[5., 17.])
        m, J, r, mg, mu = (self.model_params[key] for key in self.default_params.keys())
        self.fix_initial_value([0.]*6)
        self.mark_state_bounds_implicit([i != 2 for i in range(self.nx)])
        
        x = cs.MX.sym('x', 6)
        u = cs.MX.sym('u', 2)
        x1, x2, alpha, dx1, dx2, dalpha = cs.vertsplit(x)
        u1, u2 = cs.vertsplit(u)
        ode_rhs = cs.vertcat(dx1, 
                             dx2,
                             dalpha,
                             1/m*(u1*cs.cos(alpha) - u2*cs.sin(alpha)),
                             1/m * (-mg + u1*cs.sin(alpha) + u2*cs.cos(alpha)),
                             r/J * u1
                             )
        quad = 2*u1**2 + u2**2
        dt = cs.MX.sym('dt', 1)
        self.ODE = {'x': x, 'p':cs.vertcat(dt, u),'ode': dt*ode_rhs, 'quad': dt*quad}
        self.multiple_shooting()
        self.set_objective(1/(self.p_tf*self.ntS) * self.q_tf + mu*self.p_tf*self.ntS)
        self.add_constraint(self.x_eval[:,-1], [1] + [0.]*5, [1.] + [0.]*5)
        self.build_NLP()
        
        self.start_point = np.zeros(self.nVar)
        for i in range(self.ntS):
            self.set_stage_param(self.start_point, i, [5.0/self.ntS])
            self.set_stage_control(self.start_point, i, [1., 1.])
        # self.integrate_full(self.start_point)
        
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        u1,u2 = self.get_stage_control(s, ind)
        self.set_stage_control(s, ind, [u1+0.1,u2+0.1])
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x1, x2, alpha, _, _, _ = self.get_state_arrays_expanded(xi)
        u1,u2 = self.get_control_plot_arrays(xi)
        dt = self.get_param_arrays_expanded(xi)
        time_grid_ref = np.cumsum(np.concatenate([np.array([0]), dt]))
        
        # plt.figure(dpi = dpi)
        fig,ax = plt.subplots(dpi=dpi)
        ax.plot(time_grid_ref, x1*5, 'tab:green', linestyle='-.', label = r'$x_1\cdot 5$')
        ax.plot(time_grid_ref, x2*20, 'tab:blue', linestyle='--', label = r'$x_2\cdot 20$')
        ax.plot(time_grid_ref, alpha, 'tab:olive', linestyle='--', label = r'$\alpha$')
        
        ax.step(time_grid_ref, u1, 'tab:red', linestyle='-', label = r'$u_1$')
        ax.step(time_grid_ref, u2, 'tab:cyan', linestyle='-', label = r'$u_2$')

        ax.legend(fontsize='x-large')
        
        add_title(ax, title, it, 'Ducted fan problem')

        ax.set_xlabel('t', fontsize = 17.5)
        ax.xaxis.set_label_coords(1.015,-0.006)
        
        plt.show()
        plt.close()
        
class Robbins(OCProblem):
    default_params = {
        'alpha': 3.,
        'beta': 0.,
        'gamma': 0.5,
        'T': 10.
        }
    
    def build_problem(self):
        self.set_OCP_data(3,0,1,1,[0.]+[-np.inf]*2,[np.inf]*3,[],[],[-np.inf],[np.inf])
        alpha, beta, gamma, T = (self.model_params[key] for key in self.default_params.keys())
        self.fix_time_horizon(0,T)
        self.fix_initial_value([1.,-2.,0.])
        self.mark_state_bounds_implicit(False, True, True)
        
        X = cs.MX.sym('X', 3)
        u = cs.MX.sym('u', 1)
        x, dx, ddx = cs.vertsplit(X)
        ode_rhs = cs.vertcat(dx, 
                             ddx,
                             u
                             )
        quad = alpha*x + beta*x**2 + gamma*u**2
        dt = cs.MX.sym('dt', 1)
        self.ODE = {'x': X, 'p':cs.vertcat(dt, u),'ode': dt*ode_rhs, 'quad': dt*quad}
        self.multiple_shooting()
        self.set_objective(self.q_tf)
        self.add_constraint(self.x_eval[:,-1], [0.]*3, [0.]*3)
        self.build_NLP()
        
        self.start_point = np.zeros(self.nVar)
        for i in range(self.ntS + 1):
            self.set_stage_state(self.start_point, i, self.x_init)
        # for i in range(self.ntS):
        #     self.set_stage_param(self.start_point, i, [5.0/self.ntS])
        #     self.set_stage_control(self.start_point, i, [1., 1.])
        # # self.integrate_full(self.start_point)
        
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        u = self.get_stage_control(s, ind)
        self.set_stage_control(s, ind, u+0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x,_,_ = self.get_state_arrays_expanded(xi)
        u = self.get_control_plot_arrays(xi)
        
        # plt.figure(dpi = dpi)
        fig,ax = plt.subplots(dpi=dpi)
        ax.plot(self.time_grid_ref, x*20, 'tab:green', linestyle='-.', label = r'$x\cdot 20$')
        ax.step(self.time_grid_ref, u, 'tab:red', linestyle='-', label = r'$u$')

        ax.legend(fontsize='x-large')
        
        add_title(ax, title, it, 'Robbin\'s problem')

        ax.set_xlabel('t', fontsize = 17.5)
        ax.xaxis.set_label_coords(1.015,-0.006)
        
        plt.show()
        plt.close()
        

class Lotka_Volterra_Shared(OCProblem):
    default_params = {'c1':0.1, 
                      'c2':0.4, 
                      't0':0., 
                      'tf':40.0, 
                      'x_init':[1.5,0.5,1.0]}
    
    param_set_1 = {'x_init':[1.5,0.5,1.0]}
    param_set_2 = {'x_init':[1.5,1.0,0.5]}
    def build_problem(self):
        self.set_OCP_data(3,0,1,1,[0.,0.,0.],[np.inf, np.inf, np.inf],[],[],[0.],[1.])
        
        c1, c2 = (self.model_params[key] for key in ['c1', 'c2'])
        self.fix_time_horizon(self.model_params['t0'], self.model_params['tf'])
        self.fix_initial_value(self.model_params['x_init'])
        self.mark_state_bounds_implicit()
        
        x = cs.MX.sym('x', 3)
        u = cs.MX.sym('w', 1)
        x0, x1, x2 = cs.vertsplit(x)
        ode_rhs = cs.vertcat(x0 - x0 * x1 - x0 * x2,
              -x1 + x0 * x1 - c1 * x1 * u, 
              -x2 + 1.2 * x0 * x2 - c2 * x2 * u)
        
        quad_expr = (x0 - 1.7)**2 + (x1 - 1)**2 + (x2 - 1)**2 + 1e-3*u**2
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
        
        fix, ax = plt.subplots(dpi = dpi)
        ax.plot(self.time_grid_ref, x0, 'tab:green', linestyle = '-.', label = '$x_0$')
        ax.plot(self.time_grid_ref, x1, 'tab:blue', linestyle = '--', label = '$x_1$')
        ax.plot(self.time_grid_ref, x2, 'tab:olive', linestyle = ':', label = '$x_2$')
        #'y-.'
        
        ax.step(self.time_grid_ref, u, 'tab:red', label = r'$u$')
        ax.legend(fontsize='x-large')
        
        self.finish_plot(ax, title, it, 'Lotka Volterra shared resource problem')


class Lotka_Volterra_Competitive(OCProblem):
    default_params = {'c1':0.1, 
                      'c2':0.4, 
                      't0':0., 
                      'tf':40.0, 
                      'x_init':[0.5, 1.5]
                      }
    param_set_1 = {'x_init': [0.5, 1.5]}
    param_set_2 = {'x_init': [1.5, 0.5]}
    
    def build_problem(self):
        self.set_OCP_data(2,0,1,1,[0.,0.],[np.inf, np.inf],[],[],[0.],[1.])
        
        c1, c2 = (self.model_params[key] for key in ['c1', 'c2'])
        self.fix_time_horizon(self.model_params['t0'], self.model_params['tf'])
        self.fix_initial_value(self.model_params['x_init'])
        self.mark_state_bounds_implicit()
        
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
        
        fig, ax = plt.subplots(dpi = dpi)
        ax.plot(self.time_grid_ref, x0, 'tab:green', linestyle = '-.', label = '$x_0$')
        ax.plot(self.time_grid_ref, x1, 'tab:blue', linestyle = '--', label = '$x_1$')
        #'y-.'
        
        ax.step(self.time_grid_ref, u, 'tab:red', label = r'$u$')
        ax.legend(fontsize='x-large')
        
        self.finish_plot(ax, title, it, "Lotka Volterra competitive problem")



#Differential state 1 is stiff, approaching its lower bound of 0, so an implicit integrator is required.
#In addition, the bound should be relaxed to something like 1e-[4 -- 6].

#Using one step explicit euler with lower bound 1e-6 also """""""works""""""" as the optimizer can adjust the controls such that the collocation does not overshoot 
#Prone to local optima
class Denbigh_Reaction(OCProblem):
    default_params = {
            'E1':3000.0,
            'E2':6000.0,
            'E3':3000.0,
            'E4':0.,
            }
        
    def __init__(self, nt = 100, refine = 1, integrator = 'cvodes', parallel = False, N_threads = 4, **kwargs):
        OCProblem.__init__(self, nt=nt, refine=refine, integrator=integrator, parallel=parallel, N_threads = N_threads, **kwargs)
        
    def build_problem(self):
        E = [self.model_params[key] for key in ['E1','E2','E3','E4']]

        self.set_OCP_data(2,0,1,1, [-np.inf,-np.inf], [1.0, 1.0], [], [], [273.0], [415.0])
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

#Flat objective
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
        self.mark_state_bounds_implicit(0)
        
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

#Non-unique solution (local minima with very similar objective values)
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
        
        
        'TscalePlt': 1e-6,
        'xscalePlt': 1e-3,
        'zscalePlt': 1e-3,
        'vscalePlt': 1e-2,
        'mscalePlt': 1e-4,
        'tscalePlt': 1.0, #leave at 1.0
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
        TscalePlt, xscalePlt, zscalePlt, vscalePlt, mscalePlt, tscalePlt = (self.model_params[key] for key in ['TscalePlt', 'xscalePlt', 'zscalePlt', 'vscalePlt', 'mscalePlt', 'tscalePlt'])
        
        # x_,z_,vx_,vz_,m_ = self.get_state_arrays_expanded(xi)# / [xscale, zscale, vscale, vscale, mscale]
        # x,z,vx,vz,m = (x_/xscale, z_/zscale, vx_/vscale, vz_/vscale, m_/mscale)
        
        x,z,vx,vz,m = self.get_state_arrays_expanded(xi)
        
        T, theta = self.get_control_plot_arrays(xi)# / [Tscale, 1.0]
        dt_arr = self.get_param_arrays_expanded(xi)
        time_grid_ref = np.cumsum(np.concatenate([[0], dt_arr])).reshape(-1)
        
        Tscale = self.model_params['Tscale']
        
        plt.figure(dpi=dpi)
        plt.plot(time_grid_ref, x*20/Tscale * TscalePlt, 'm-', label = r'x $\cdot$ ' + str(xscale) + r'$\cdot 20$')
        plt.plot(time_grid_ref, z/zscale * zscalePlt, 'g--', label = r'z $\cdot$ ' + str(zscale))
        plt.plot(time_grid_ref, vx/vscale * vscalePlt, 'b:', label = r'vx $\cdot$ ' + str(vscale))
        plt.plot(time_grid_ref, vz/vscale * vscalePlt, 'c-.', label = r'vz $\cdot$ ' + str(vscale))
        plt.plot(time_grid_ref, (m/mscale - self.model_params['mdry'])*mscalePlt*5.0, 'y-.', label = r'(m - mdry)$\cdot$' + str(mscale) + r'$\cdot 5.0$')
        
        plt.step(time_grid_ref, T/Tscale * TscalePlt, 'r', label = r'T$\cdot$ ' + str(Tscale))
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


class Satellite_Deorbiting(OCProblem):
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
            'mdry': 100,
            'omegaE':7.2921e-5,
            'h0': 450000,
            'hreentry': 120000,
            # 'rscale': 1.0e-2,
            # 'vscale': 1.0e-1,
            # 'thetascale': 1.0,
            # 'mscale': 1.0e1,
            # 'TSCALE': 1.0
            'rscale': 1e-4,         #good
            # 'rscale': 1.0,
            'vrscale': 1.0,
            'thetascale': 1.0,
            'vthetascale': 1e-4,    #good
            # 'vthetascale': 1.0,
            'mscale': 1,
            'TSCALE': 1.0
            }
    
    def build_problem(self):
        mu, RE, rho0, H, CD, A, Isp, g0, umax, m0, mdry, omegaE, h0, hreentry, rscale, thetascale, mscale, vrscale, vthetascale, TSCALE = (self.model_params[key] for key in ['mu', 'RE', 'rho0', 'H', 'CD', 'A', 'Isp', 'g0', 'umax', 'm0', 'mdry', 'omegaE', 'h0', 'hreentry', 'rscale', 'thetascale', 'mscale', 'vrscale', 'vthetascale', 'TSCALE'])
        
        r0 = RE + h0
        theta0 = 0.
        vr0 = 0.
        vorb = np.sqrt(mu/r0)
        
        rfinal = RE + hreentry
        
        self.set_OCP_data(5,1,2,0, [(RE+5000. - RE)*rscale, -2*np.pi*thetascale, -10000.*vrscale, 0.*vthetascale, (mdry - 0.1)*mscale], [(r0 + 100000. - RE)*rscale, 2*np.pi*thetascale, 10000.*vrscale, 20000*vthetascale, (m0 + 0.1)*mscale], [300/self.ntS * TSCALE], [21600/self.ntS * TSCALE], [-umax, -umax], [umax, umax])
        self.fix_initial_value([(r0 - RE)*rscale, theta0*thetascale, vr0*vrscale, vorb*vthetascale, m0*mscale])
        
        def safe_sqrt(x):
            return cs.sqrt(cs.fmax(x, 1e-12))
        
        # Atmospheric model
        def atmospheric_density(r_val):
            h = r_val - RE
            h_safe = cs.fmax(h, -100000)
            return rho0 * cs.exp(-h_safe / H)
        
        # # Relative velocity components
        # def relative_velocity(v_r_val, v_theta_val, r_val):
        #     v_rel_r = v_r_val
        #     v_rel_theta = v_theta_val - omegaE * r_val
        #     v_rel_sq = v_rel_r**2 + v_rel_theta**2
        #     v_rel = safe_sqrt(v_rel_sq)
        #     return v_rel_r, v_rel_theta, v_rel
        
        X = cs.MX.sym('X', 5)
        r_, theta_, vr_, vtheta_, m_ = cs.vertsplit(X)
        r = r_/rscale + RE
        theta = theta_/thetascale
        vr = vr_/vrscale
        vtheta = vtheta_/vthetascale
        m = m_/mscale
        
        U = cs.MX.sym('U', 2)
        ur, utheta = cs.vertsplit(U)
        dt_ = cs.MX.sym('dt', 1)
        dt = dt_/TSCALE
        
        rsafe = cs.fmax(r, RE + 10000)
        msafe = cs.fmax(m, mdry)
        
        # rho = atmospheric_density(rsafe)
        hsafe = cs.fmax(rsafe - RE, -100000)
        rho = rho0 * cs.exp(-hsafe/H)
        
        # vrelr, vreltheta, vrel = relative_velocity(vr, vtheta, rsafe)
        
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
        RE, rscale, thetascale, mscale, TSCALE, vrscale, vthetascale, mu = [self.model_params[key] for key in ['RE', 'rscale', 'thetascale', 'mscale', 'TSCALE', 'vrscale', 'vthetascale', 'mu']]
        
        h0 = 450000
        r0 = RE + h0
        vorb = np.sqrt(mu/r0)
        
        r_, theta_, vr_, vtheta_, m_ = self.get_state_arrays(xi)
        r = r_/rscale + RE
        theta = theta_/thetascale
        vr = vr_/vrscale
        vtheta = vtheta_/vthetascale
        m = m_/mscale
        
        ur, utheta = self.get_control_plot_arrays(xi)
        dt = self.get_param_arrays(xi)
        time_grid = np.cumsum(np.concatenate([np.array([0]), dt]))/TSCALE
        dte = self.get_param_arrays_expanded(xi)
        time_grid_ref = np.cumsum(np.concatenate([np.array([0]), dte]))/TSCALE
        
        fix, ax = plt.subplots(dpi=dpi)
        ax.plot(time_grid, (r - RE)/1000, 'tab:cyan', linestyle = '--', label = r'(r - RE)/1000')
        ax.plot(time_grid, theta*50, 'tab:green', linestyle = ':', label = r'$\theta\cdot 50$')
        ax.plot(time_grid, vr, 'tab:blue', linestyle = '--', label = r'$v_r$')
        ax.plot(time_grid, vtheta - vorb, 'tab:olive', linestyle = '-.', label = r'$v_\theta - v_\theta(0)$')
        ax.plot(time_grid, (m - 100.)*4, 'tab:blue', linestyle = ':', label = r'(m - 100)$\cdot 4$')
        
        ax.step(time_grid_ref, ur*20, 'tab:red', label = r'$u_r \cdot 20$')
        ax.step(time_grid_ref, utheta*20, 'tab:green', label = r'$u_\theta \cdot 20$')
        
        ax.legend(fontsize='large')
        
        self.finish_plot(ax, title, it, "Satellite Deorbiting problem")
        

#Doent work well (flat objective due to very low control shifting sensitivity)
# TODO consider adding scaled duration penalty to objective
# Somehow also brings the MUMPS sparse linear solver to its knees inside qpOASES ...

#Ipopt works only with exact Hessian, blockSQP2 almost works with LAPACK factorization inside qpOASES and exact Hessian, getting very close to the optimal solution, but still terminates with an error
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
            'rscale': 1.0e-4, #1.0e-2
            # 'rscale': 1.0,
            'thetascale': 1.0,
            'vrscale': 1.0,
            'vthetascale': 1.0e-4,
            # 'vthetascale': 1.0,
            'mscale': 1.0   #1.0e1
            }
    
    def build_problem(self):
        mu, RE, rho0, H, CD, A, Isp, g0, umax, omegaE, MDTH, rscale, thetascale, vrscale, vthetascale, mscale = (self.model_params[key] for key in ['mu', 'RE', 'rho0', 'H', 'CD', 'A', 'Isp', 'g0', 'umax', 'omegaE', 'MDTH', 'rscale', 'thetascale', 'vrscale', 'vthetascale', 'mscale'])
        
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
        
        self.set_OCP_data(5,0,2,0, [(RE + 5000 - RE)*rscale, -np.ceil(n_orbits)*2*np.pi*thetascale, -8000.*vrscale, 1000.*vthetascale, (mdry - 0.1)*mscale], [(r0 + 50000. - RE)*rscale, np.ceil(n_orbits)*2*np.pi*thetascale, 8000.*vrscale, 15000.*vthetascale, (m0 + 0.1)*mscale], [], [], [-umax, -umax], [umax, umax])
        self.fix_time_horizon(0., T_mission_fixed)
        self.fix_initial_value([(r0 - RE)*rscale, (theta0)*thetascale, vr0*vrscale, vorb*vthetascale, m0*mscale])
        
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
        vr = vr_/vrscale
        vtheta = vtheta_/vthetascale
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
            (centrifugal - gravity + rthrust - drag*vrelr) * vrscale,
            (-vr*vtheta/rsafe + thetathrust - drag*vreltheta) * vthetascale,
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
        vr_init = np.linspace(0., -500., self.ntS + 1) * vrscale
        vtheta_init = np.linspace(vorb, vorb*0.85, self.ntS + 1) * vthetascale
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
        RE, rscale, thetascale, vrscale, vthetascale, mscale = [self.model_params[key] for key in ['RE', 'rscale', 'thetascale', 'vrscale', 'vthetascale', 'mscale']]
        
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
        


class Lotka_Shared_OED(OCProblem):
    default_params = {'alpha0': 1.0,
                      'alpha1': 1.0,
                      'alpha2': 1.2,
                      'c1':0.1, 
                      'c2':0.4, 
                      't0':0., 
                      'tf':20.0, 
                      'x_init':[1.5,0.5,1.0],
                      'M1': 4.0,
                      'M2': 4.0,
                      'M3': 4.0,
                      'reg_init': 0.1
                      }
    param_set_1 = {'x_init':[1.5,0.5,1.0]}
    param_set_2 = {'x_init':[1.5,1.0,0.5]}
    
    def build_problem(self):
        self.set_OCP_data(3+9+6,0,4,3, [0.,0.,0.] + [-np.inf]*15, [np.inf, np.inf, np.inf] + [np.inf]*15,[],[],[0.]*4,[1.]*4)
        
        alpha0, alpha1, alpha2, c1, c2, t0, tf, x_init, M1, M2, M3, reg_init = (self.model_params[key] for key in ['alpha0', 'alpha1', 'alpha2', 'c1', 'c2', 't0', 'tf', 'x_init', 'M1', 'M2', 'M3', 'reg_init'])
        self.fix_time_horizon(self.model_params['t0'], self.model_params['tf'])
        self.fix_initial_value(self.model_params['x_init'] + [0.]*15)
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
        
        #Fix theta in the expressions
        f = cs.Function('f', [x, u, theta], [f_expr])
        f_x = cs.Function('f_x', [x, u, theta], [f_x_expr])
        f_theta = cs.Function('f_p', [x,u,theta], [f_x_expr])
        
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
        
        quad_expr = w
        dt = cs.MX.sym('dt', 1)
        self.ODE = {'x': cs.vertcat(x, cs.vec(G), F), 'p':cs.vertcat(dt, u, w),'ode': dt*ode_rhs, 'quad': dt*quad_expr}
        self.multiple_shooting()
        
        F_rhs_tf = self.x_eval[3+9:3+9+6,-1]
        F_tf = cs.MX.zeros(3,3)
        for j in range(3):
            for i in range(0, j):
                F_tf[i,j] = F_rhs_tf[i + j*3 - (j*(j+1))//2]
            F_tf[j,j] = F_rhs_tf[j*4 - (j*(j+1))//2] + reg_init
            for i in range(j + 1, 3):
                F_tf[i,j] = F_rhs_tf[j + i*3 - (i*(i+1))//2]
        
        self.set_objective(cs.trace(cs.inv(F_tf))/theta.numel())
        self.add_constraint(self.q_tf, [0.,0.,0.], [M1,M2,M3])
        self.build_NLP()
        
        L_t = tf - t0
        for i in range(self.ntS):
            self.set_stage_control(self.start_point, i, [0, M1/L_t, M2/L_t, M3/L_t])
        self.integrate_full(self.start_point)
        
        
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        s_ind = self.get_stage_control(s, ind)
        self.set_stage_control(s, ind, [0.1, *s_ind[1:4]])
        return s
    
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
        
        
class Lotka_OED_new(OCProblem):
    default_params = {
        'tf':12, 
        'p1':1,
        'p2':1,
        'p3':1,
        'p4':1,
        'p5':0.4,
        'p6':0.2,
        'x_init':[0.5,0.7],
        'M':4.0,
        'fishing':True,
        'epsilon': 0.0,
        'transform_obj':False
        }
    def build_problem(self):
        self.set_OCP_data(2 + 4 + 3, 0, 3, 2, [0.,0.]+[-np.inf]*7, [np.inf]*9,[],[],[0.] + [0.]*2, [float(self.model_params['fishing'])] + [1.]*2)
        tf,p1,p2,p3,p4,p5,p6,x_init,M,epsilon, transform_obj= (self.model_params[key] for key in ['tf', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6','x_init', 'M', 'epsilon', 'transform_obj'])
        self.fix_time_horizon(0.,tf)
        self.fix_initial_value(x_init + [0.]*4 + [epsilon, 0., epsilon])
        self.mark_state_bounds_implicit()
        
        x = cs.MX.sym('x', 2)
        x1,x2 = cs.vertsplit(x)
        u = cs.MX.sym('u', 1)
        p = cs.MX.sym('p', 2)
        p2_s, p4_s = cs.vertsplit(p)
        
        f_expr = cs.vertcat(p1*x1 - p2_s*x1*x2 - p5*u*x1,
                           -p3*x2 + p4_s*x1*x2 - p6*u*x2
                            )
        f_x_expr = cs.jacobian(f_expr, x)
        f_p_expr = cs.jacobian(f_expr, p)
        
        f = cs.Function('f', [x,u,p], [f_expr])
        f_x = cs.Function('f', [x,u,p], [f_x_expr])
        f_p = cs.Function('f', [x,u,p], [f_p_expr])
        
        f_expr = f(x,u, cs.DM([p2,p4]))
        f_x_expr = f_x(x,u, cs.DM([p2,p4]))
        f_p_expr = f_p(x,u, cs.DM([p2,p4]))
        
        
        G = cs.MX.sym('G', x.numel(), p.numel())
        dG = f_x_expr@G + f_p_expr
        G_rhs = cs.vec(dG)
        
        w = cs.MX.sym('w', 2)
        w1,w2 = cs.vertsplit(w)
        
        F = cs.MX.sym('F', (p.numel()*(p.numel() + 1))//2)
        dh1, dh2 = cs.DM([1,0]), cs.DM([0,1])
        dF = w1*(dh1.T@G).T @ (dh1.T@G) + w2*(dh2.T@G).T @ (dh2.T@G)
        
        F_rhs = cs.vertcat(dF[0,0], dF[1,0], dF[1,1])
        ode_rhs = cs.vertcat(f_expr, G_rhs, F_rhs)
        
        dt = cs.MX.sym('dt', 1)
        quad_expr = w
        self.ODE = {'x': cs.vertcat(x, cs.vec(G), F), 'p':cs.vertcat(dt, u, w),'ode': dt*ode_rhs, 'quad': dt*quad_expr}
        self.multiple_shooting()
        
        F_rhs_tf = self.x_eval[2 + 2*2:2 + 2*2 + 3, -1]
        F_tf = cs.MX.zeros(2,2)
        for j in range(2):
            for i in range(0, j):
                F_tf[i,j] = F_rhs_tf[i + j*2 - (j*(j+1))//2]
            F_tf[j,j] = F_rhs_tf[j*3 - (j*(j+1))//2]
            for i in range(j + 1, 2):
                F_tf[i,j] = F_rhs_tf[j + i*2 - (i*(i+1))//2]
        
        self.set_objective(cs.trace(cs.inv(F_tf)))
        self.add_constraint(self.q_tf, -np.inf, M)
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
        x1, x2, G11, G12, G21, G22, F11, F12, F22 = self.get_state_arrays_expanded(xi)
        
        fig, ax = plt.subplots(dpi=dpi)
        ax.plot(self.time_grid_ref, x1, 'tab:olive', linestyle='-.', label = r'$x_1$')
        ax.plot(self.time_grid_ref, x2, 'tab:cyan', linestyle='-.', label = r'$x_2$')
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
        

#Require higher accuracy (<= 1e-7) for somewhat smooth control
class Catalyst_Mixing_OED(OCProblem):
    default_params = {
        'p1': 1.0,
        'p2': 10.0,
        'p3': 1.0,
        'M1': 0.2,
        'M2': 0.2,
        'reg_init': 1e-2,
        
        'p1scale': 1.0,
        'p2scale': 1.0
        }
    def build_problem(self):
        self.set_OCP_data(2 + 4 + 3,0,3,2,[-np.inf,-np.inf] + [-np.inf]*7,[np.inf,np.inf] + [np.inf]*7,[],[],[0.] + [0.]*2,[1.] + [1.]*2)
        self.fix_time_horizon(0,1)
        T_l = 1.0
        self.fix_initial_value([1.,0.] + [0.]*7)
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
        ode_rhs = cs.vertcat(f_expr, G_rhs, F_rhs)
        
        dt = cs.MX.sym('dt', 1)
        quad_expr = w
        self.ODE = {'x': cs.vertcat(x, cs.vec(G), F), 'p':cs.vertcat(dt, u, w),'ode': dt*ode_rhs, 'quad': dt*quad_expr}
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
        self.add_constraint(self.q_tf - cs.DM([M1,M2]), -np.inf, 0.)
        self.build_NLP()
        for i in range(self.ntS):
            self.set_stage_control(self.start_point, i, [0.5,M1/T_l,M2/T_l])
        self.integrate_full(self.start_point)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, [0.6, 0., 0.])
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        fig, ax = plt.subplots(dpi=dpi)
        x1,x2, G11, G21, G12, G22, F11, F21, F22 = self.get_state_arrays_expanded(xi)
        u, w1, w2 = self.get_control_plot_arrays(xi)
        ax.plot(self.time_grid_ref, x1, 'tab:green', linestyle='-.', label = r'$x_1$')
        ax.plot(self.time_grid_ref, x2, 'tab:blue', linestyle='--', label = r'$x_2$')
        ax.step(self.time_grid_ref, u, 'tab:red', linestyle='-', label = r'$u$')
        ax.step(self.time_grid_ref, w1, 'tab:green', linestyle='--', label = r'$w_1$')
        ax.step(self.time_grid_ref, w2, 'tab:blue', linestyle='-.', label = r'$w_2$')
        
        ax.legend(fontsize = 'large')
        
        self.finish_plot(ax, title, it, "Catalyst mixing OED")
        

class Dielectrophoretic_Particle_OED(OCProblem):
    default_params = {
        'x0': 1.,
        'xf': 2.,
        'alpha':-0.75,
        'c':1.,
        'M1': 2.0,
        'M2': 2.0,
        'reg_init': 1e-2
        }
    
    def build_problem(self):
        self.set_OCP_data(2 + 4 + 3,0,3,2,[-np.inf,-np.inf] + [-np.inf]*7,[np.inf, np.inf] + [np.inf]*7,[],[],[-1]+[0.,0.],[1] + [1.,1.])
        x0,xf,alpha,c,M1,M2,reg_init = (self.model_params[key] for key in ('x0','xf','alpha','c','M1','M2','reg_init'))
        self.fix_initial_value([x0, 0.] + [0.]*7)
        self.fix_time_horizon(0., 8.0)
        T_l = 8.0
        self.mark_state_bounds_implicit()
        
        x = cs.MX.sym('x', 2)
        x0, x1 = cs.vertsplit(x)
        
        u = cs.MX.sym('u', 1)
        theta = cs.MX.sym('theta', 2)
        alpha_s, c_s = cs.vertsplit(theta)
        
        f_expr = cs.vertcat( x1*u + alpha_s*u**2, 
                            -c_s*x1 + u)
        f_x_expr = cs.jacobian(f_expr, x)
        f_theta_expr = cs.jacobian(f_expr, theta)
        
        f = cs.Function('f', [x,u,theta], [f_expr])
        f_x = cs.Function('f', [x,u,theta], [f_x_expr])
        f_theta = cs.Function('f', [x,u,theta], [f_theta_expr])
        
        f_expr = f(x,u,cs.DM([alpha,c]))
        f_x_expr = f_x(x,u,cs.DM([alpha,c]))
        f_theta_expr = f_theta(x,u,cs.DM([alpha,c]))
        
        G = cs.MX.sym('G', x.numel(), theta.numel())
        dG = f_x_expr@G + f_theta_expr
        G_rhs = cs.vec(dG)
        
        w = cs.MX.sym('w', 2)
        w1,w2 = cs.vertsplit(w)
        
        F = cs.MX.sym('F', (theta.numel()*(theta.numel() + 1))//2)
        dh1, dh2 = cs.DM([1,0]), cs.DM([0,1])
        dF = w1*(dh1.T@G).T @ (dh1.T@G) + w2*(dh2.T@G).T @ (dh2.T@G)
        
        F_rhs = cs.vertcat(dF[0,0], dF[1,0], dF[1,1])
        ode_rhs = cs.vertcat(f_expr, G_rhs, F_rhs)
        
        
        dt = cs.MX.sym('dt', 1)
        quad_expr = w
        
        self.ODE = {'x': cs.vertcat(x, cs.vec(G), F), 'p':cs.vertcat(dt, u, w),'ode': dt*ode_rhs, 'quad':dt*quad_expr}
        self.multiple_shooting()
        
        F_rhs_tf = self.x_eval[2 + 2*2 : 2 + 2*2 + 3, -1]
        
        F_tf = cs.MX.zeros(2,2)
        for j in range(2):
            for i in range(0, j):
                F_tf[i,j] = F_rhs_tf[i + j*2 - (j*(j+1))//2]
            F_tf[j,j] = F_rhs_tf[j*3 - (j*(j+1))//2] + reg_init
            for i in range(j + 1, 2):
                F_tf[i,j] = F_rhs_tf[j + i*2 - (i*(i+1))//2]
        
        self.set_objective(cs.trace(cs.inv(F_tf)))
        self.add_constraint(self.q_tf, [0., 0.], [M1,M2])
        self.add_constraint(self.x_eval[0,-1], xf, xf)
        
        self.build_NLP()
        
        self.start_point = np.zeros(self.nVar)
        for i in range(self.ntS):
            self.set_stage_control(self.start_point, i, [1.0, M1/T_l, M2/T_l])
        self.integrate_full(self.start_point)
        
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, 0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x0, x1 = self.get_state_arrays_expanded(xi)[0:2]
        u, w1, w2 = self.get_control_plot_arrays(xi)
        time_grid_ref = self.time_grid_ref
        
        # plt.figure(dpi = dpi)
        fig, ax = plt.subplots(dpi=dpi)
        ax.plot(time_grid_ref, x0, 'tab:green', linestyle='-.', label = '$x_0$')
        ax.plot(time_grid_ref, x1, 'tab:blue', linestyle='--', label = '$x_1$')
        ax.step(time_grid_ref, u, 'tab:red', linestyle='-', label = r'$u$')
        ax.step(time_grid_ref, w1, 'tab:olive', linestyle=':', label = r'$w_1$')
        ax.step(time_grid_ref, w2, 'tab:cyan', linestyle=':', label = r'$w_2$')

        
        ax.legend(fontsize='x-large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Dielectrophoretic Particle OED'
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
        
# Seems ill posed, neither blockSQP nor ipopt work
# class Three_Tank_OED(OCProblem):
#     default_params = {'T': 12, 
#                       'c1': 1., 
#                       'c2': 2., 
#                       'c3': 0.8, 
#                       'k1': 2, 
#                       'k2': 3, 
#                       'k3': 1, 
#                       'k4': 3,
#                       'M1': 2.0,
#                       'M2': 2.0,
#                       'M3': 2.0,
#                       'reg_init': 1e-1
#                       }
#     def build_problem(self):
#         self.set_OCP_data(3+9+6, 0, 3+3, 4, [0., 0., 0.] + [-np.inf]*15, [np.inf,np.inf,np.inf] + [np.inf]*15, [],[], [0.,0.,0.] + [0.]*3, [1.,1.,1.] + [1.]*3)
#         self.fix_time_horizon(0, self.model_params['T'])
#         self.fix_initial_value([2.,2.,2.] + [0.]*15)
#         self.mark_state_bounds_implicit()
        
#         T, c1, c2, c3, k1, k2, k3, k4, M1, M2, M3, reg_init = (self.model_params[key] for key in ['T', 'c1', 'c2', 'c3', 'k1', 'k2', 'k3', 'k4', 'M1', 'M2', 'M3', 'reg_init'])
        
#         x = cs.MX.sym('x', 3)
#         x1,x2,x3 = cs.vertsplit(x)
#         theta = cs.MX.sym('theta', 3)
#         c1_s, c2_s, c3_s = cs.vertsplit(theta)
        
#         u = cs.MX.sym('u', 3)
#         u1,u2,u3 = cs.vertsplit(u)
#         dt = cs.MX.sym('dt')
        
#         f_expr = cs.vertcat(-cs.sqrt(x1) + c1_s*u1 + c2_s*u2 - u3*cs.sqrt(c3_s*x1),
#                               cs.sqrt(x1) - cs.sqrt(x2),
#                               cs.sqrt(x2) - cs.sqrt(x3) + u3*cs.sqrt(c3_s*x1)
#                               )
#         f_x_expr = cs.jacobian(f_expr, x)
#         f_theta_expr = cs.jacobian(f_expr, theta)
        
#         f = cs.Function('f', [x,u,theta], [f_expr])
#         f_x = cs.Function('f', [x,u,theta], [f_x_expr])
#         f_theta = cs.Function('f', [x,u,theta], [f_theta_expr])
        
#         f_expr = f(x,u,cs.DM([c1, c2, c3]))
#         f_x_expr = f_x(x,u,cs.DM([c1, c2, c3]))
#         f_theta_expr = f_theta(x,u,cs.DM([c1, c2, c3]))
        
        
#         G = cs.MX.sym('G', x.numel(), theta.numel())
#         dG = f_x_expr@G + f_theta_expr
#         G_rhs = cs.vec(dG)
        
#         w = cs.MX.sym('w', 3)
#         w1,w2,w3 = cs.vertsplit(w)
        
#         F = cs.MX.sym('F', (theta.numel()*(theta.numel() + 1))//2)
#         dh1, dh2, dh3 = cs.DM([1,0,0]), cs.DM([0,1,0]), cs.DM([0,0,1])
#         dF = w1*(dh1.T@G).T @ (dh1.T@G) + w2*(dh2.T@G).T @ (dh2.T@G) + w3*(dh3.T@G).T @ (dh3.T@G)
        
#         F_rhs = cs.vertcat(dF[0,0], dF[1,0], dF[2,0], dF[1,1], dF[2,1], dF[2,2])
#         ode_rhs = cs.vertcat(f_expr, G_rhs, F_rhs)
        
#         dt = cs.MX.sym('dt', 1)
#         quad_expr = cs.vertcat(w, k1*(x2-k2)**2 + k3*(x3-k4)**2)
        
#         self.ODE = {'x': cs.vertcat(x, cs.vec(G), F), 'p':cs.vertcat(dt, u, w),'ode': dt*ode_rhs, 'quad':dt*quad_expr}
#         self.multiple_shooting()
        
#         F_rhs_tf = self.x_eval[3 + 3*3 : 3 + 3*3 + 6, -1]
        
#         F_tf = cs.MX.zeros(3,3)
#         for j in range(3):
#             for i in range(0, j):
#                 F_tf[i,j] = F_rhs_tf[i + j*3 - (j*(j+1))//2]
#             F_tf[j,j] = F_rhs_tf[j*4 - (j*(j+1))//2] + reg_init
#             for i in range(j + 1, 3):
#                 F_tf[i,j] = F_rhs_tf[j + i*3 - (i*(i+1))//2]
        
#         self.set_objective(cs.trace(cs.inv(F_tf)))
#         # self.set_objective(self.q_tf[3,-1])
#         self.add_constraint(self.q_tf, [0., 0., 0, -np.inf], [M1, M2, M3, np.inf])
#         self.add_constraint(cs.sum1(self.u_eval), 1., 1.)
        
#         self.build_NLP()
#         for i in range(self.ntS):
#             self.set_stage_control(self.start_point, i, [0.5,0.5,0., M1/T,M2/T,M3/T])
#         self.integrate_full(self.start_point)
        
#     def perturbed_start_point(self, ind):
#         s = copy.copy(self.start_point)
#         self.set_stage_control(s, ind, [0.5, 0.25, 0.25])
#         return s
    
#     def plot(self, xi, dpi = None, title = None, it = None):
#         x1,x2,x3 = self.get_state_arrays(xi)[0:3]
#         u1,u2,u3, w1,w2,w3 = self.get_control_plot_arrays(xi)[0:6]
        
#         fig, ax = plt.subplots(dpi=dpi)
#         ax.plot(self.time_grid, x1, 'tab:olive', linestyle='--', label = r'$x_1$')#, self.time_grid[:,-1], x1, '--', self.time_grid[:,-1], u, 'o')
#         ax.plot(self.time_grid, x2, 'tab:purple', linestyle='-.', label = r'$x_2$')
#         ax.plot(self.time_grid, x3, 'tab:cyan', linestyle=':', label = r'$x_3$')
#         ax.step(self.time_grid_ref, u1, 'tab:olive', linestyle='-', label = r'$u_1$')
#         ax.step(self.time_grid_ref, u2, 'tab:red', linestyle='-', label = r'$u_2$')
#         ax.step(self.time_grid_ref, u3, 'grey', label = r'$u_3$')
        
        
#         ax.step(self.time_grid_ref, w1, 'tab:olive', linestyle='-.', label = r'$w_1$')
#         ax.step(self.time_grid_ref, w2, 'tab:red', linestyle='-.', label = r'$w_2$')
#         ax.step(self.time_grid_ref, w3, 'grey', linestyle = '-.', label = r'$w_3$')
#         ax.legend(prop={'size': 13.4}, loc = 'upper right')
        
#         ax.set_xlabel('t', fontsize = 17.5)
#         ax.xaxis.set_label_coords(1.015,-0.006)
        
#         ttl = None
#         if isinstance(title,str):
#             ttl = title
#         elif title == True:
#             ttl = 'Three tank problem'
#         if ttl is not None:
#             if isinstance(it, int):
#                 ttl = ttl + f', iteration {it}'
#             plt.title(ttl)
#         else:
#             plt.title('')
            
#         plt.show()
#         plt.close()


#Lots of local minima.
class Batch_Reactor_OED(OCProblem):
    default_params = {
        'p1': 4000,
        'p2': 2500,
        'p3': 620000,
        'p4': 5000,
        'M1': 0.4,
        'M2': 0.4,
        'reg_init': 1e-3,
        
        'p1scale': 400/4000,
        'p2scale': 400/2500,
        'p3scale': 400/620000,
        'p4scale': 400/5000,
        
        # 'p1scale': 1.0,
        # 'p2scale': 1.0,
        # 'p3scale': 1.0,
        # 'p4scale': 1.0,
        }
    def build_problem(self):                                                                                                 # Increase lower control bound to force good? local optimum
        self.set_OCP_data(2 + 4*2 + 10, 0, 1 + 2, 2, [-np.inf,-np.inf] + [-np.inf]*18, [np.inf,np.inf] + [np.inf]*18, [], [], [298 + 60] + [0.]*2, [398] + [1.]*2)
        self.mark_state_bounds_implicit()
        
        p1, p2, p3, p4, M1, M2, reg_init, p1scale, p2scale, p3scale, p4scale = (self.model_params[key] for key in ['p1', 'p2', 'p3', 'p4', 'M1', 'M2', 'reg_init', 'p1scale', 'p2scale', 'p3scale', 'p4scale'])
        self.fix_initial_value([1.0,0.0] + [0.]*18)
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
        ode_rhs = cs.vertcat(f_expr, G_rhs, F_rhs)
        
        quad_expr = w
        
        dt = cs.MX.sym('dt', 1)
        quad_expr = w
        self.ODE = {'x': cs.vertcat(x, cs.vec(G), F), 'p':cs.vertcat(dt, T, w),'ode': dt*ode_rhs, 'quad':dt*quad_expr}
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
        self.add_constraint(self.q_tf, [0., 0.], [M1, M2])
        
        self.build_NLP()
        
        self.start_point = np.zeros(self.nVar)
        for i in range(self.ntS):
            self.set_stage_control(self.start_point, i, 398)
            self.set_stage_state(self.start_point, i, self.x_init)
        self.set_stage_state(self.start_point, self.ntS, self.x_init)
        self.integrate_full(self.start_point)

    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, 390)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x1,x2 = self.get_state_arrays_expanded(xi)[0:2]
        T, w1, w2 = self.get_control_plot_arrays(xi)
        
        fix, ax = plt.subplots(dpi = dpi)
        ax.plot(self.time_grid, x1, 'tab:cyan', linestyle = '--', label = r'$x_1$')
        ax.plot(self.time_grid, x2, 'tab:blue', linestyle = '-.', label = r'$x_2$')
        ax.step(self.time_grid_ref, (T-298)*0.05, 'tab:red', label = r'$(u-298)\cdot 0.05$')
        
        ax.step(self.time_grid_ref, w1, 'tab:green', linestyle = '-.', label = r'$w_1$')
        ax.step(self.time_grid_ref, w2, 'tab:orange', linestyle = ':', label = r'$w_2$')
        
        ax.legend(fontsize='large')
        self.finish_plot(ax, title, it, "Batch reactor OED")
        
        

class Bryson_Denham(OCProblem):
    default_params = {}
    def build_problem(self):
        self.set_OCP_data(2, 0, 1, 1, [-np.inf,-np.inf], [1/9, np.inf], [], [], [-np.inf], [np.inf])
        
        self.fix_initial_value([0., 1.])
        self.fix_time_horizon(0,1)
        
        X = cs.MX.sym('X', 2)
        x,v = cs.vertsplit(X)
        
        w = cs.MX.sym('w', 1)
        
        ode_rhs = cs.vertcat(v, w)
        quad_expr = 0.5 * w**2
        
        dt = cs.MX.sym('dt', 1)
        self.ODE = {'x': X, 'p':cs.vertcat(dt, w),'ode': dt*ode_rhs, 'quad':dt*quad_expr}
        self.multiple_shooting()
        
        self.set_objective(self.q_tf)
        self.add_constraint(self.x_eval[:,-1], [0., -1.], [0., -1.])
        
        self.build_NLP()
        
        self.start_point = np.zeros(self.nVar)
        for i in range(self.ntS):
            self.set_stage_state(self.start_point, i, self.x_init)
        self.set_stage_state(self.start_point, self.ntS, self.x_init)

    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, 0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        x, v = self.get_state_arrays_expanded(xi)[0:2]
        w = self.get_control_plot_arrays(xi)
        
        plt.figure(dpi = dpi)
        plt.plot(self.time_grid_ref, x, 'tab:green', linestyle = '--', label = r'$x$')
        plt.plot(self.time_grid_ref, v, 'tab:blue', linestyle = '-.', label = r'$v$')
        plt.step(self.time_grid_ref, w, 'tab:red', label = r'$w$')
        
        
        plt.legend(fontsize='large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Batch reactor OED'
        if ttl is not None:
            if isinstance(it, int):
                ttl = ttl + f', iteration {it}'
            plt.title(ttl)
        else:
            plt.title('')
            
        plt.show()
        plt.close()



class Moon_Landing(OCProblem):
    default_params = {
            'c': 2.349,
            'h0': 1.0,
            'v0': -0.783,
            'm0': 1.0,
            'Tmax': 1.227
            }
    
    def build_problem(self):
        c, h0, v0, m0, Tmax = (self.model_params[key] for key in ('c', 'h0', 'v0', 'm0', 'Tmax'))
        
        self.set_OCP_data(3,1,1,0,[-np.inf, -np.inf, 0.],[np.inf, np.inf, np.inf],[0.1/self.ntS],[100/self.ntS],[0],[Tmax])
        self.fix_initial_value([h0, v0, m0])
        self.mark_state_bounds_implicit(True, True, False)
        
        x = cs.MX.sym('x', 3)
        h,v,m = cs.vertsplit(x)
        
        T = cs.MX.sym('T', 1)
        
        ode_rhs = cs.vertcat(v, -1 + T/m, -T/c)
        dt = cs.MX.sym('dt', 1)
        self.ODE = {'x': x, 'p':cs.vertcat(dt, T),'ode': dt*ode_rhs}
        self.multiple_shooting()
        self.set_objective(-self.x_eval[2,-1])
        self.add_constraint(self.x_eval[:2,-1], [0.,0.], [0.,0.])
        self.build_NLP()
        
        self.start_point = np.zeros(self.nVar)
        for i in range(self.ntS):
            self.set_stage_param(self.start_point, i, 1.0/self.ntS)
            self.set_stage_control(self.start_point, i, Tmax)
            self.set_stage_state(self.start_point, i, self.x_init)
        self.set_stage_state(self.start_point, self.ntS, self.x_init)
        # self.integrate_full(self.start_point)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, 0.1)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        h, v, m = self.get_state_arrays_expanded(xi)
        T = self.get_control_plot_arrays(xi)
        dt_arr = self.get_param_arrays_expanded(xi)
        time_grid_ref = np.cumsum(np.concatenate([[0], dt_arr])).reshape(-1)
        
        fig,ax = plt.subplots(dpi=dpi)
        ax.plot(time_grid_ref, h, 'tab:green', linestyle='-.', label = '$h$')
        ax.plot(time_grid_ref, v, 'tab:blue', linestyle='--', label = '$v$')
        ax.plot(time_grid_ref, m, 'tab:red', linestyle='-.', label = '$m$')
        
        ax.step(time_grid_ref, T, 'tab:red', linestyle='-', label = r'$T$')
        ax.legend(fontsize='x-large')
        
        ttl = None
        if isinstance(title,str):
            ttl = title
        elif title == True:
            ttl = 'Moon Landing problem'
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


class Apollo_Reentry(OCProblem):
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
        
        self.set_OCP_data(3,1,1,1,[0.2, -0.2, 0.006],[0.4, 0.1, 0.02],[220/self.ntS],[240/self.ntS],[-np.pi/2],[np.pi/2])
        self.fix_initial_value([0.36, -8.1*(np.pi/180), 4./R])
        
        x = cs.MX.sym('x', 3)
        v, gamma, xi = cs.vertsplit(x, 1)
        u = cs.MX.sym('u')
        
        C_D = c1 - c2*cs.cos(u)
        C_L = c3*cs.sin(u)
        rho = rho0 * cs.exp(-beta*R*xi)

        vdot = -0.5*Sm*rho*v**2 * C_D - g*cs.sin(gamma)/(1 + xi)**2
        gammadot = 0.5*Sm*rho*v*C_L + v*cs.cos(gamma)/(R*(1 + xi)) - g*cs.cos(gamma)/(v*(1 + xi)**2)
        xidot = v*cs.sin(gamma)/R
        
        ode_rhs = cs.vertcat(vdot, gammadot, xidot)
        quad_rhs = 10 * v**3 * cs.sqrt(rho)
        
        dt = cs.MX.sym('dt', 1)
        self.ODE = {'x': x, 'p':cs.vertcat(dt, u),'ode': dt*ode_rhs, 'quad': dt*quad_rhs}
        self.multiple_shooting()
        self.set_objective(self.q_tf)
        self.add_constraint(self.x_eval[:,-1], [0.27, 0., 2.5/R], [0.27, 0., 2.5/R])
        self.build_NLP()
        
        self.start_point = np.zeros(self.nVar)
        for i in range(self.ntS):
            self.set_stage_param(self.start_point, i, 230.0/self.ntS)
            self.set_stage_control(self.start_point, i, 0.5)
            self.set_stage_state(self.start_point, i, self.x_init)
        self.set_stage_state(self.start_point, self.ntS, self.x_init)
    
    def perturbed_start_point(self, ind):
        s = copy.copy(self.start_point)
        self.set_stage_control(s, ind, 0.6)
        return s
    
    def plot(self, xi, dpi = None, title = None, it = None):
        v, gamma, xivar = self.get_state_arrays_expanded(xi)
        u = self.get_control_plot_arrays(xi)
        dt_arr = self.get_param_arrays_expanded(xi)
        time_grid_ref = np.cumsum(np.concatenate([[0], dt_arr])).reshape(-1)
        
        fig,ax = plt.subplots(dpi=dpi)
        ax.plot(time_grid_ref, v, 'tab:green', linestyle='-.', label = r'$v$')
        ax.plot(time_grid_ref, gamma, 'tab:blue', linestyle='--', label = r'$\gamma$')
        ax.plot(time_grid_ref, xivar*10, 'tab:olive', linestyle='-.', label = r'$\xi\cdot 10$')
        
        ax.step(time_grid_ref, u/5, 'tab:red', linestyle='-', label = r'$u/5$')
        ax.legend(fontsize='x-large', loc = 'upper right')
        
        self.finish_plot(ax, title, it, "Apollo reentry problem")
