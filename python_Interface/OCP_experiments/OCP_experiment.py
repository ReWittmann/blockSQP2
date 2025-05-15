import os
import sys
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
except:
    sys.path.append(os.getcwd() + "/..")

import OCProblems
import py_blockSQP
from blockSQP_pyProblem import blockSQP_pyProblem as Problemspec
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
import casadi as cs


class CountCallback(cs.Callback):
    def __init__(self, name, nx, ng, np):
        cs.Callback.__init__(self)
        self.it = -1
        self.prim_vars = cs.DM([])
        self.nx = nx
        self.ng = ng
        self.np = np
        self.construct(name, {})

    
    def get_n_in(self):
        return cs.nlpsol_n_out()

    def get_n_out(self):
        return 1

    def get_name_in(self, i):
        return cs.nlpsol_out(i)

    def get_name_out(self, i):
        return "ret"

    def get_sparsity_in(self, i):
        n = cs.nlpsol_out(i)
        if n == 'f':
            return cs.Sparsity.scalar()
        elif n in ('x', 'lam_x'):
            return cs.Sparsity.dense(self.nx)
        elif n in ('g', 'lam_g'):
            return cs.Sparsity.dense(self.ng)
        else:
            return cs.Sparsity(0, 0)
    
    def eval(self, arg):
        self.it += 1
        return [0] 
    
    

def perturbStartPoint(OCP : OCProblems.OCProblem, IND : int, SP : np.array):
    if (isinstance(OCP, (OCProblems.Goddard_Rocket, OCProblems.Electric_Car, OCProblems.Hang_Glider, OCProblems.Fullers, OCProblems.Tubular_Reactor))):
        val = OCP.get_stage_control(SP, IND)
        OCP.set_stage_control(SP, IND, val - 0.1)
    
    #For catalyst mixing, Lotka (OED) and cushioned oscillation
    if (isinstance(OCP, (OCProblems.Lotka_Volterra_Fishing, OCProblems.Catalyst_Mixing, OCProblems.Lotka_OED, OCProblems.Cushioned_Oscillation, OCProblems.Egerstedt_Standard, OCProblems.Particle_Steering, OCProblems.Time_Optimal_Car))):
        OCP.set_stage_control(SP, IND, 0.1)
    
    #For hanging chain
    if (isinstance(OCP, (OCProblems.Hanging_Chain, OCProblems.Van_der_Pol_Oscillator_3))):
        val = OCP.get_stage_control(SP, IND)
        OCP.set_stage_control(SP, IND, val + 0.1)
    
        #For batch reactor
    if (isinstance(OCP, OCProblems.Batch_Reactor)):
        OCP.set_stage_control(SP, IND, 300.)

    #For batch distillation
    if (isinstance(OCP, OCProblems.Batch_Distillation)):
        OCP.set_stage_control(SP, IND, 1.5)
    
    if (isinstance(OCP, OCProblems.Three_Tank_Multimode)):
        OCP.set_stage_control(SP, IND, [0.5, 0.25, 0.25])


def create_prob_cond(OCprob : OCProblems.OCProblem):
    vBlocks = py_blockSQP.vblock_array(len(OCprob.vBlock_sizes))
    cBlocks = py_blockSQP.cblock_array(len(OCprob.cBlock_sizes))
    hBlocks = py_blockSQP.int_array(len(OCprob.hessBlock_sizes))
    targets = py_blockSQP.condensing_targets(1)
    for i in range(len(OCprob.vBlock_sizes)):
        vBlocks[i] = py_blockSQP.vblock(OCprob.vBlock_sizes[i], OCprob.vBlock_dependencies[i])
    for i in range(len(OCprob.cBlock_sizes)):
        cBlocks[i] = py_blockSQP.cblock(OCprob.cBlock_sizes[i])
    for i in range(len(OCprob.hessBlock_sizes)):
        hBlocks[i] = OCprob.hessBlock_sizes[i]
    targets[0] = py_blockSQP.condensing_target(*OCprob.ctarget_data)
    HOLD = [vBlocks, cBlocks, hBlocks, targets]
    
    cond = py_blockSQP.Condenser(vBlocks, cBlocks, hBlocks, targets)
    
    
    prob = Problemspec()
    prob.x_start = OCprob.start_point
    
    prob.nVar = OCprob.nVar
    prob.nCon = OCprob.nCon
    prob.f = OCprob.f
    prob.grad_f = OCprob.grad_f
    prob.g = OCprob.g
    prob.make_sparse(OCprob.jac_g_nnz, OCprob.jac_g_row, OCprob.jac_g_colind)
    prob.jac_g_nz = OCprob.jac_g_nz
    prob.hess = OCprob.hess_lag
    prob.set_blockIndex(OCprob.hessBlock_index)
    prob.set_bounds(OCprob.lb_var, OCprob.ub_var, OCprob.lb_con, OCprob.ub_con)
    prob.lam_start = np.zeros(prob.nVar + prob.nCon, dtype = np.float64).reshape(-1)
    prob.vblocks = vBlocks
    return prob, cond, HOLD




def perturbed_starts(OCprob : OCProblems.OCProblem, opts : py_blockSQP.SQPoptions, nPert0, nPertF, COND = False, itMax = 100):
    N_SQP = []
    N_secs = []
    type_sol = []
    for j in range(nPert0,nPertF):
        start_it = copy.copy(OCprob.start_point)
        perturbStartPoint(OCprob, j, start_it)
        prob, cond, HOLD = create_prob_cond(OCprob)
        prob.x_start = start_it
        prob.complete()

        stats = py_blockSQP.SQPstats("./solver_outputs")
        if not COND:
            optimizer = py_blockSQP.SQPmethod(prob, opts, stats)
        else:
            optimizer = py_blockSQP.SCQPmethod(prob, opts, stats, cond)
        optimizer.init()
        t0 = time.time()
        ret = optimizer.run(itMax)
        t1 = time.time()
        
        N_SQP.append(stats.itCount)
        N_secs.append(t1 - t0)
        if int(ret) == 1:
            type_sol.append(0)
        elif int(ret) > 1:
            type_sol.append(1)
        else:
            type_sol.append(-1)
        
        # if int(ret) > 0:
        #     N_SQP.append(stats.itCount)
        #     N_secs.append(t1 - t0)
        #     if int(ret) == 1:
        #         type_sol.append(0)
        #     else:
        #         type_sol.append(1)
        # else:
        #     N_SQP.append(0.00001)
        #     N_secs.append(0.00001)
        #     type_sol.append(-1)
    
    return N_SQP, N_secs, type_sol

def ipopt_perturbed_starts(OCprob : OCProblems.OCProblem, ipopts : dict, nPert0, nPertF, itMax = 100):
    NLP = OCprob.NLP
    counter = CountCallback('counter', OCprob.NLP['x'].size1(), OCprob.NLP['g'].size1(), 0)
    opts = {'iteration_callback': counter}
    opts.update({'ipopt': ipopts})
    S = cs.nlpsol('S', 'ipopt', NLP, opts)
    N_SQP = []
    N_secs = []
    type_sol = []
    for j in range(nPert0, nPertF):
        counter.it = -1
        start_it = copy.copy(OCprob.start_point)
        perturbStartPoint(OCprob, j, start_it)
        t0 = time.time()
        out = S(x0=start_it, lbx=OCprob.lb_var,ubx=OCprob.ub_var, lbg=OCprob.lb_con, ubg=OCprob.ub_con)
        t1 = time.time()
        N_SQP.append(counter.it)
        N_secs.append(t1 - t0)
        type_sol.append(1)
    
    return N_SQP, N_secs, type_sol

def plot_all(n_EXP, nPert0, nPertF, titles, EXP_N_SQP, EXP_N_secs, EXP_type_sol):
    n_xticks = 10
    tdist = round((nPertF - nPert0)/n_xticks)
    tdist += (tdist==0)
    xticks = np.arange(nPert0, nPertF + tdist, tdist)
    ###############################################################################
    EXP_N_SQP_clean = [[EXP_N_SQP[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] >= 0] for i in range(n_EXP)]
    EXP_N_secs_clean = [[EXP_N_secs[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] >= 0] for i in range(n_EXP)]

    EXP_N_SQP_mu = [sum(EXP_N_SQP_clean[i])/len(EXP_N_SQP_clean[i]) for i in range(n_EXP)]
    EXP_N_SQP_sigma = [(sum((np.array(EXP_N_SQP_clean[i]) - EXP_N_SQP_mu[i])**2)/len(EXP_N_SQP_clean[i]))**(0.5) for i in range(n_EXP)]

    EXP_N_secs_mu = [sum(EXP_N_secs_clean[i])/len(EXP_N_secs_clean[i]) for i in range(n_EXP)]
    EXP_N_secs_sigma = [(sum((np.array(EXP_N_secs_clean[i]) - EXP_N_secs_mu[i])**2)/len(EXP_N_secs_clean[i]))**(0.5) for i in range(n_EXP)]

    #Care, doen't work for numbers smaller than 0.0001, representation becomes *e-***
    trunc_float = lambda num, dg: str(float(num))[0:int(np.ceil(abs(np.log(num + (num == 0))/np.log(10)))) + 2 + dg]

    ccodemp = {-1: 'r', 0:'y', 1:'g'}
    cmap = [[ccodemp[v] for v in EXP_type_sol[i]] for i in range(n_EXP)]

    ###############################################################################
    titlesize = 19
    axtitlesize = 15
    labelsize = 13

    fig = plt.figure(constrained_layout=True, dpi = 300, figsize = (14+2*(max(n_EXP - 2, 0)), 3.5 + 3.5*(n_EXP-1)))
    subfigs = fig.subfigures(nrows=n_EXP, ncols=1)
    if n_EXP == 1:
        subfigs = (subfigs,)
        
    for i in range(n_EXP):
        ax_it, ax_time = subfigs[i].subplots(nrows=1,ncols=2)
        subfigs[i].suptitle(titles[i], size = titlesize)
        
        ax_it.scatter(list(range(nPert0,nPertF)), EXP_N_SQP[i], c = cmap[i])
        ax_it.set_ylabel('SQP iterations', size = labelsize)
        ax_it.set_ylim(bottom = 0)
        ax_it.set_xlabel('location of perturbation', size = labelsize)
        ax_it.set_title(r"$\mu = " + trunc_float(EXP_N_SQP_mu[i], 1) + r"\ \sigma = " + trunc_float(EXP_N_SQP_sigma[i], 1) + "$", size = axtitlesize)
        ax_it.set_xticks(xticks)
        
        ax_time.scatter(list(range(nPert0,nPertF)), EXP_N_secs[i], c = cmap[i])
        ax_time.set_ylabel("solution time in seconds", size = labelsize)
        ax_time.set_ylim(bottom = 0)
        ax_time.set_xlabel("location of perturbation", size = labelsize)
        ax_time.set_title(r"$\mu = " + trunc_float(EXP_N_secs_mu[i], 1) + r"\ \sigma = " + trunc_float(EXP_N_secs_sigma[i], 1) + "$", size = axtitlesize)
        ax_time.set_xticks(xticks)

    plt.show()
    
def plot_successful(n_EXP, nPert0, nPertF, titles, EXP_N_SQP, EXP_N_secs, EXP_type_sol):
    n_xticks = 10
    tdist = round((nPertF - nPert0)/n_xticks)
    tdist += (tdist==0)
    xticks = np.arange(nPert0, nPertF + tdist, tdist)
    ###############################################################################
    def F(x,r):
        if r >= 0:
            return x
        else:
            return 0.00001
    
    
    EXP_N_SQP_S = [[F(EXP_N_SQP[i][j], EXP_type_sol[i][j]) for j in range(nPertF - nPert0)] for i in range(n_EXP)]
    EXP_N_secs_S = [[F(EXP_N_secs[i][j], EXP_type_sol[i][j]) for j in range(nPertF - nPert0)] for i in range(n_EXP)]

    # EXP_N_SQP_clean = [[EXP_N_SQP[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] >= 0] for i in range(n_EXP)]
    # EXP_N_secs_clean = [[EXP_N_secs[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] >= 0] for i in range(n_EXP)]
    
    EXP_N_SQP_mu = [sum(EXP_N_SQP[i])/len(EXP_N_SQP[i]) for i in range(n_EXP)]
    EXP_N_SQP_sigma = [(sum((np.array(EXP_N_SQP[i]) - EXP_N_SQP_mu[i])**2)/len(EXP_N_SQP[i]))**(0.5) for i in range(n_EXP)]
    EXP_N_secs_mu = [sum(EXP_N_secs[i])/len(EXP_N_secs[i]) for i in range(n_EXP)]
    EXP_N_secs_sigma = [(sum((np.array(EXP_N_secs[i]) - EXP_N_secs_mu[i])**2)/len(EXP_N_secs[i]))**(0.5) for i in range(n_EXP)]

    #Care, doen't work for numbers smaller than 0.0001, representation becomes *e-***
    trunc_float = lambda num, dg: str(float(num))[0:int(np.ceil(abs(np.log(num + (num == 0))/np.log(10)))) + 2 + dg]

    # ccodemp = {-1: 'r', 0:'y', 1:'g'}
    # cmap = [[ccodemp[v] for v in EXP_type_sol[i]] for i in range(n_EXP)]

    ###############################################################################
    # titlesize = 19
    # axtitlesize = 15
    # labelsize = 13

    titlesize = 23
    axtitlesize = 20
    labelsize = 19
    
    # titlesize = 21
    # axtitlesize = 19
    # labelsize = 16

    fig = plt.figure(constrained_layout=True, dpi = 300, figsize = (14+2*(max(n_EXP - 2, 0)), 3.5 + 3.5*(n_EXP - 1)))
    subfigs = fig.subfigures(nrows=n_EXP, ncols=1)
    
    if n_EXP == 1:
        subfigs = (subfigs,)
    for i in range(n_EXP):
        ax_it, ax_time = subfigs[i].subplots(nrows=1,ncols=2)
        # subfigs[i].suptitle(titles[i], size = titlesize)
        
        ax_it.scatter(list(range(nPert0,nPertF)), EXP_N_SQP_S[i])#, c = cmap[i])
        ax_it.set_ylabel('SQP iterations', size = labelsize)
        ax_it.set_ylim(bottom = 0)
        ax_it.set_xlabel('location of perturbation', size = labelsize)
        ax_it.set_title(r"$\mu = " + trunc_float(EXP_N_SQP_mu[i], 1) + r"\ \sigma = " + trunc_float(EXP_N_SQP_sigma[i], 1) + "$", size = axtitlesize)
        ax_it.set_xticks(xticks)
        ax_it.tick_params(labelsize = labelsize - 1)
        
        ax_time.scatter(list(range(nPert0,nPertF)), EXP_N_secs_S[i])#, c = cmap[i])
        ax_time.set_ylabel("solution time [s]", size = labelsize)
        ax_time.set_ylim(bottom = 0)
        ax_time.set_xlabel("location of perturbation", size = labelsize)
        ax_time.set_title(r"$\mu = " + trunc_float(EXP_N_secs_mu[i], 1) + r"\ \sigma = " + trunc_float(EXP_N_secs_sigma[i], 1) + "$", size = axtitlesize)
        ax_time.set_xticks(xticks)
        ax_time.tick_params(labelsize = labelsize - 1)
        
        subfigs[i].suptitle(titles[i], size = titlesize)
    plt.show()


def plot_varshape(n_EXP, nPert0, nPertF, titles, EXP_N_SQP, EXP_N_secs, EXP_type_sol):
    n_xticks = 10
    tdist = round((nPertF - nPert0)/n_xticks)
    tdist += (tdist==0)
    xticks = np.arange(nPert0, nPertF + tdist, tdist)
    ###############################################################################
    EXP_grid = [list(range(nPert0, nPertF)) for i in range(n_EXP)]
    EXP_grid_sol = [[EXP_grid[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] > 0] for i in range(n_EXP)]
    EXP_grid_part = [[EXP_grid[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] == 0] for i in range(n_EXP)]
    EXP_grid_fail = [[EXP_grid[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] < 0] for i in range(n_EXP)]
    
    EXP_N_SQP_sol = [[EXP_N_SQP[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] > 0] for i in range(n_EXP)]
    EXP_N_SQP_part = [[EXP_N_SQP[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] == 0] for i in range(n_EXP)]
    EXP_N_SQP_fail = [[EXP_N_SQP[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] < 0] for i in range(n_EXP)]
    EXP_N_secs_sol = [[EXP_N_secs[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] > 0] for i in range(n_EXP)]
    EXP_N_secs_part = [[EXP_N_secs[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] == 0] for i in range(n_EXP)]
    EXP_N_secs_fail = [[EXP_N_secs[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] < 0] for i in range(n_EXP)]

    
    EXP_N_SQP_clean = [[EXP_N_SQP[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] >= 0] for i in range(n_EXP)]
    EXP_N_secs_clean = [[EXP_N_secs[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] >= 0] for i in range(n_EXP)]

    EXP_N_SQP_mu = [sum(EXP_N_SQP_clean[i])/len(EXP_N_SQP_clean[i]) for i in range(n_EXP)]
    EXP_N_SQP_sigma = [(sum((np.array(EXP_N_SQP_clean[i]) - EXP_N_SQP_mu[i])**2)/len(EXP_N_SQP_clean[i]))**(0.5) for i in range(n_EXP)]

    EXP_N_secs_mu = [sum(EXP_N_secs_clean[i])/len(EXP_N_secs_clean[i]) for i in range(n_EXP)]
    EXP_N_secs_sigma = [(sum((np.array(EXP_N_secs_clean[i]) - EXP_N_secs_mu[i])**2)/len(EXP_N_secs_clean[i]))**(0.5) for i in range(n_EXP)]

    #Care, doen't work for numbers smaller than 0.0001, representation becomes *e-***
    trunc_float = lambda num, dg: str(float(num))[0:int(np.ceil(abs(np.log(num + (num == 0))/np.log(10)))) + 2 + dg]

    # ccodemp = {-1: 'r', 0:'y', 1:'g'}
    # cmap = [[ccodemp[v] for v in EXP_type_sol[i]] for i in range(n_EXP)]
    
    ###############################################################################
    titlesize = 23
    axtitlesize = 19
    labelsize = 16

    fig = plt.figure(constrained_layout=True, dpi = 300, figsize = (14+2*(max(n_EXP - 2, 0)), 3.5 + 3.5*(n_EXP - 1)))
    subfigs = fig.subfigures(nrows=n_EXP, ncols=1)
    if n_EXP == 1:
        subfigs = (subfigs,)
    
    for i in range(n_EXP):
        ax_it, ax_time = subfigs[i].subplots(nrows=1,ncols=2)
        subfigs[i].suptitle(titles[i], size = titlesize)
        ax_it.scatter(EXP_grid_sol[i], EXP_N_SQP_sol[i], c = 'g', marker = 'o', label = "success")
        ax_it.scatter(EXP_grid_part[i], EXP_N_SQP_part[i], c = 'y', marker = 'v', label = "partial success")
        ax_it.scatter(EXP_grid_fail[i], EXP_N_SQP_fail[i], c = 'r', marker = 'x', label = "failure")

        ax_it.set_ylabel('SQP iterations', size = labelsize)
        ax_it.set_ylim(bottom = 0)
        ax_it.set_xlabel('location of perturbation', size = labelsize)
        ax_it.set_title(r"$\mu = " + trunc_float(EXP_N_SQP_mu[i], 1) + r"\ \sigma = " + trunc_float(EXP_N_SQP_sigma[i], 1) + "$", size = axtitlesize)
        ax_it.set_xticks(xticks)
        ax_it.tick_params(labelsize = labelsize - 1)
        ax_it.legend(fontsize = 'x-large')
        
        ax_time.scatter(EXP_grid_sol[i], EXP_N_secs_sol[i], c = 'g', marker = 'o')
        ax_time.scatter(EXP_grid_part[i], EXP_N_secs_part[i], c = 'y', marker = 'v')
        ax_time.scatter(EXP_grid_fail[i], EXP_N_secs_fail[i], c = 'r', marker = 'x')
        
        ax_time.set_ylabel("solution time [s]", size = labelsize)
        ax_time.set_ylim(bottom = 0)
        ax_time.set_xlabel("location of perturbation", size = labelsize)
        ax_time.set_title(r"$\mu = " + trunc_float(EXP_N_secs_mu[i], 1) + r"\ \sigma = " + trunc_float(EXP_N_secs_sigma[i], 1) + "$", size = axtitlesize)
        ax_time.tick_params(labelsize = labelsize - 1)
        ax_time.set_xticks(xticks)
    plt.show()


    