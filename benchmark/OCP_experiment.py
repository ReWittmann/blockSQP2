# blockSQP2 -- A structure-exploiting nonlinear programming solver based
#              on blockSQP by Dennis Janka.
# Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>

# Licensed under the zlib license. See LICENSE for more details.


# \file OCP_experiment.py
# \author Reinhold Wittmann
# \date 2025
#
# Helper functions for benchmarking blockSQP2 and casadi NLP solvers

import sys
from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()
sys.path += [str(cD.parent/Path("Python"))]

import OCProblems
import blockSQP2
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import shutil
if shutil.which("latex") is not None:
    plt.rcParams["text.usetex"] = True
import casadi as cs


def create_prob_cond(OCprob : OCProblems.OCProblem):
    vBlocks = blockSQP2.vblock_array(len(OCprob.vBlock_sizes))
    cBlocks = blockSQP2.cblock_array(len(OCprob.cBlock_sizes))
    hBlocks = blockSQP2.int_array(len(OCprob.hessBlock_sizes))
    targets = blockSQP2.condensing_targets(1)
    for i in range(len(OCprob.vBlock_sizes)):
        vBlocks[i] = blockSQP2.vblock(OCprob.vBlock_sizes[i], OCprob.vBlock_dependencies[i])
    for i in range(len(OCprob.cBlock_sizes)):
        cBlocks[i] = blockSQP2.cblock(OCprob.cBlock_sizes[i])
    for i in range(len(OCprob.hessBlock_sizes)):
        hBlocks[i] = OCprob.hessBlock_sizes[i]
    targets[0] = blockSQP2.condensing_target(*OCprob.ctarget_data)
    HOLD = [vBlocks, cBlocks, hBlocks, targets]
    
    cond = blockSQP2.Condenser(vBlocks, cBlocks, hBlocks, targets)
    
    
    prob = blockSQP2.Problemspec()
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

def perturbed_starts(OCprob : OCProblems.OCProblem, opts : blockSQP2.SQPoptions, nPert0, nPertF, COND = False, itMax = 100):
    """Run blockSQP on the given problem for start points perturbed at nPert0:nPertF
    Return a vector of the iteration counts, the solution times in seconds and a vector of return codes
    indication the success: < 0 - failure, 0 - max it reached, 1 - partial success, > 1 success"""

    N_SQP = []
    N_secs = []
    type_sol = []
    for j in range(nPert0,nPertF):
        start_it = OCprob.perturbed_start_point(j)
        
        prob, cond, HOLD = create_prob_cond(OCprob)
        prob.x_start = start_it
        if COND:
            prob.cond = cond
        
        prob.complete()
        stats = blockSQP2.SQPstats("./solver_outputs")        
        t0 = time.monotonic()
        optimizer = blockSQP2.SQPmethod(prob, opts, stats)
        optimizer.init()
        ret = optimizer.run(itMax)
        optimizer.finish()
        t1 = time.monotonic()
        
        N_SQP.append(stats.itCount)
        N_secs.append(t1 - t0)
        if ret.value >= 0:
            type_sol.append(ret.value)
        else:
            type_sol.append(-1)    
    return N_SQP, N_secs, type_sol


def casadi_solver_perturbed_starts(plugin : str, OCprob : OCProblems.OCProblem, arg_opts : dict, nPert0, nPertF, itMax = 200):
    """Same as perturbed_starts, but allows specifying a casadi NLP solver as first argument"""
    NLP = OCprob.NLP
    opts = arg_opts
    N_SQP = []
    N_secs = []
    type_sol = []
    for j in range(nPert0, nPertF):
        S = cs.nlpsol('S', plugin, NLP, opts)
        start_it = OCprob.perturbed_start_point(j)
        
        t0 = time.monotonic()
        out = S(x0=start_it, lbx=OCprob.lb_var,ubx=OCprob.ub_var, lbg=OCprob.lb_con, ubg=OCprob.ub_con)
        t1 = time.monotonic()
        stats = S.stats()
        if plugin == 'ipopt':
            N_SQP.append(stats['iter_count'])
        elif plugin == 'worhp':
            N_SQP.append(stats['n_call_nlp_grad_f'] - 1)
        elif plugin == 'blocksqp':
            N_SQP.append(stats['n_call_nlp_grad_f'] - 1)
        elif plugin == 'fatrop':
            N_SQP.append(stats['iterations_count'])
        type_sol.append(int(stats['success']))
        N_secs.append(t1 - t0)
    return N_SQP, N_secs, type_sol


def plot_all(n_EXP, nPert0, nPertF, titles, EXP_N_SQP, EXP_N_secs, EXP_type_sol, suptitle = None):
    """Plot result of multiple runs for perturbed start points for different options. 
    n_EXP - number of different options,
    nPert0, nPertF - start and end index of perturbed start points,
    EXP_... - vector of vectors of iterations counts, solution times and return codes for the pertubed start points.
    """
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
    

    ccodemp = {-1: 'r', 0:'y', 1:'g'}
    cmap = [[ccodemp[v] for v in EXP_type_sol[i]] for i in range(n_EXP)]

    ###############################################################################
    titlesize = 19
    axtitlesize = 15
    labelsize = 13
    
    fig = plt.figure(constrained_layout=True, dpi = 300, figsize = (14+2*(max(n_EXP - 2, 0)), 3.5 + 3.5*(n_EXP-1)))
    fig.suptitle(suptitle, fontsize = 'x-large')
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
        ax_it.set_title(r"$\mu = " + f"{EXP_N_SQP_mu[i]:.2f}" + r"\ \sigma = " + f"{EXP_N_SQP_sigma[i]:.2f}" + "$", size = axtitlesize)
      
        
        ax_it.set_xticks(xticks)
        
        ax_time.scatter(list(range(nPert0,nPertF)), EXP_N_secs[i], c = cmap[i])
        ax_time.set_ylabel("solution time in seconds", size = labelsize)
        ax_time.set_ylim(bottom = 0)
        ax_time.set_xlabel("location of perturbation", size = labelsize)
        ax_time.set_title(r"$\mu = " + f"{EXP_N_secs_mu[i]:.2f}" + r"\ \sigma = " + f"{EXP_N_secs_sigma[i]:.2f}" + "$", size = axtitlesize)
        ax_time.set_xticks(xticks)

    plt.show()


def plot_successful(n_EXP, nPert0, nPertF, titles, EXP_N_SQP, EXP_N_secs, EXP_type_sol, suptitle = None, dirPath : Path = None, savePrefix = None):
    """Plot result of multiple runs for perturbed start points for different options. 
    n_EXP - number of different options,
    nPert0, nPertF - start and end index of perturbed start points,
    EXP_... - vector of vectors of iterations counts, solution times and return codes for the pertubed start points.
    """
    if isinstance(dirPath, str):
        print("\n\nWARNING: Passing a pathstring to plot_successful is not recommended, use pathlib.Path instead\n", flush = True)
        dirPath = Path(dirPath)
    
    n_xticks = 10
    tdist = round((nPertF - nPert0)/n_xticks)
    tdist += (tdist==0)
    xticks = np.arange(nPert0, nPertF + tdist, tdist)
    ###############################################################################
    def F(x,r):
        if r > 0:
            return x
        else:
            return 0.00001    
    EXP_N_SQP_S = [[F(EXP_N_SQP[i][j], EXP_type_sol[i][j]) for j in range(nPertF - nPert0)] for i in range(n_EXP)]
    EXP_N_secs_S = [[F(EXP_N_secs[i][j], EXP_type_sol[i][j]) for j in range(nPertF - nPert0)] for i in range(n_EXP)]

    EXP_N_SQP_mu = [sum(EXP_N_SQP[i])/len(EXP_N_SQP[i]) for i in range(n_EXP)]
    EXP_N_SQP_sigma = [(sum((np.array(EXP_N_SQP[i]) - EXP_N_SQP_mu[i])**2)/len(EXP_N_SQP[i]))**(0.5) for i in range(n_EXP)]
    EXP_N_secs_mu = [sum(EXP_N_secs[i])/len(EXP_N_secs[i]) for i in range(n_EXP)]
    EXP_N_secs_sigma = [(sum((np.array(EXP_N_secs[i]) - EXP_N_secs_mu[i])**2)/len(EXP_N_secs[i]))**(0.5) for i in range(n_EXP)]
    
    ###############################################################################
    titlesize = 23
    axtitlesize = 20
    labelsize = 19
    
    fig = plt.figure(constrained_layout=True, dpi = 300, figsize = (14+2*(max(n_EXP - 2, 0)), 3.5 + 3.5*(n_EXP - 1)))
    if isinstance(suptitle, str):
        if shutil.which("latex") is not None:
            fig.suptitle(r"$\textbf{" + suptitle + "}$", fontsize = 24, fontweight = 'bold')
        else:
            fig.suptitle(suptitle, fontsize = 24, fontweight = 'bold')
    subfigs = fig.subfigures(nrows=n_EXP, ncols=1)
    
    if n_EXP == 1:
        subfigs = (subfigs,)
    for i in range(n_EXP):
        ax_it, ax_time = subfigs[i].subplots(nrows=1,ncols=2)
        
        ax_it.scatter(list(range(nPert0,nPertF)), EXP_N_SQP_S[i])#, c = cmap[i])
        ax_it.set_ylabel('SQP iterations', size = labelsize)
        ax_it.set_ylim(bottom = 0)
        ax_it.set_xlabel('location of perturbation', size = labelsize)
        ax_it.set_title(r"$\mu = " + f"{EXP_N_SQP_mu[i]:.2f}" + r"\ \sigma = " + f"{EXP_N_SQP_sigma[i]:.2f}" + "$", size = axtitlesize)
        ax_it.set_xticks(xticks)
        ax_it.tick_params(labelsize = labelsize - 1)
        
        ax_time.scatter(list(range(nPert0,nPertF)), EXP_N_secs_S[i])#, c = cmap[i])
        ax_time.set_ylabel("solution time [s]", size = labelsize)
        ax_time.set_ylim(bottom = 0)
        ax_time.set_xlabel("location of perturbation", size = labelsize)
        ax_time.set_title(r"$\mu = " + f"{EXP_N_secs_mu[i]:.2f}" + r"\ \sigma = " + f"{EXP_N_secs_sigma[i]:.2f}" + "$", size = axtitlesize)
        
        ax_time.set_xticks(xticks)
        ax_time.tick_params(labelsize = labelsize - 1)
        
        subfigs[i].suptitle(titles[i], size = titlesize)
    if not isinstance(dirPath, Path):
        plt.show()
    else:
        dirPath.mkdir(parents = True, exist_ok = True)
        
        date_app = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_").replace("'", "")
        name_app = "" if suptitle is None else suptitle.replace(" ", "_").replace(":", "_").replace(".", "_").replace("'", "")        
        pref = "" if savePrefix is None else savePrefix
        
        plt.savefig(dirPath / Path(pref + "_it_s_" + name_app + "_" + date_app))
    plt.close()


def plot_varshape(n_EXP, nPert0, nPertF, titles, EXP_N_SQP, EXP_N_secs, EXP_type_sol, suptitle = None, dirPath : Path = None, savePrefix = None):
    """Plot result of multiple runs for perturbed start points for different options. 
    n_EXP - number of different options,
    nPert0, nPertF - start and end index of perturbed start points,
    EXP_... - vector of vectors of iterations counts, solution times and return codes for the pertubed start points.
    """
    if isinstance(dirPath, str):
        print("\n\nWARNING: Passing a pathstring to plot_varshape is not recommended, use pathlib.Path instead\n", flush = True)
        dirPath = Path(dirPath)
    n_xticks = 10
    tdist = round((nPertF - nPert0)/n_xticks)
    tdist += (tdist==0)
    xticks = np.arange(nPert0, nPertF + tdist, tdist)
    ###############################################################################
    EXP_grid = [list(range(nPert0, nPertF)) for i in range(n_EXP)]
    EXP_grid_sol = [[EXP_grid[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] > 1] for i in range(n_EXP)]
    EXP_grid_part = [[EXP_grid[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] == 1] for i in range(n_EXP)]
    EXP_grid_fail = [[EXP_grid[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] < 1] for i in range(n_EXP)]
    
    EXP_N_SQP_sol = [[EXP_N_SQP[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] > 1] for i in range(n_EXP)]
    EXP_N_SQP_part = [[EXP_N_SQP[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] == 1] for i in range(n_EXP)]
    EXP_N_SQP_fail = [[EXP_N_SQP[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] < 1] for i in range(n_EXP)]
    EXP_N_secs_sol = [[EXP_N_secs[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] > 1] for i in range(n_EXP)]
    EXP_N_secs_part = [[EXP_N_secs[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] == 1] for i in range(n_EXP)]
    EXP_N_secs_fail = [[EXP_N_secs[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] < 1] for i in range(n_EXP)]

    EXP_N_SQP_clean = [[EXP_N_SQP[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] >= 1] for i in range(n_EXP)]
    EXP_N_secs_clean = [[EXP_N_secs[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] >= 1] for i in range(n_EXP)]

    EXP_N_SQP_mu = [sum(EXP_N_SQP_clean[i])/len(EXP_N_SQP_clean[i]) for i in range(n_EXP)]
    EXP_N_SQP_sigma = [(sum((np.array(EXP_N_SQP_clean[i]) - EXP_N_SQP_mu[i])**2)/len(EXP_N_SQP_clean[i]))**(0.5) for i in range(n_EXP)]

    EXP_N_secs_mu = [sum(EXP_N_secs_clean[i])/len(EXP_N_secs_clean[i]) for i in range(n_EXP)]
    EXP_N_secs_sigma = [(sum((np.array(EXP_N_secs_clean[i]) - EXP_N_secs_mu[i])**2)/len(EXP_N_secs_clean[i]))**(0.5) for i in range(n_EXP)]
    
    ###############################################################################
    titlesize = 23
    axtitlesize = 19
    labelsize = 16
    
    # titlesize = 23
    # axtitlesize = 20
    # labelsize = 19

    fig = plt.figure(constrained_layout=True, dpi = 300, figsize = (14+2*(max(n_EXP - 2, 0)), 3.5 + 3.5*(n_EXP - 1)))
    if isinstance(suptitle, str):
        if shutil.which("latex") is not None:
            fig.suptitle(r"$\textbf{" + suptitle + "}$", fontsize = 24, fontweight = 'bold')
        else:
            fig.suptitle(suptitle, fontsize = 24, fontweight = 'bold')
    subfigs = fig.subfigures(nrows=n_EXP, ncols=1)
    if n_EXP == 1:
        subfigs = (subfigs,)
    
    for i in range(n_EXP):
        ax_it, ax_time = subfigs[i].subplots(nrows=1,ncols=2)
        subfigs[i].suptitle(titles[i], size = titlesize)
        # ax_it.scatter(EXP_grid_sol[i], EXP_N_SQP_sol[i], c = 'g', marker = 'o', label = "success")
        # ax_it.scatter(EXP_grid_part[i], EXP_N_SQP_part[i], c = 'y', marker = 'v', label = "partial success")
        # ax_it.scatter(EXP_grid_fail[i], EXP_N_SQP_fail[i], c = 'r', marker = 'x', label = "failure")
        ax_it.scatter(EXP_grid_sol[i], EXP_N_SQP_sol[i], c = 'tab:green', marker = 'o', label = "success")
        ax_it.scatter(EXP_grid_part[i], EXP_N_SQP_part[i], c = 'tab:olive', marker = 'v', label = "partial success")
        ax_it.scatter(EXP_grid_fail[i], EXP_N_SQP_fail[i], c = 'tab:red', marker = 'x', label = "failure")


        ax_it.set_ylabel('SQP iterations', size = labelsize)
        ax_it.set_ylim(bottom = 0)
        ax_it.set_xlabel('location of perturbation', size = labelsize)
        ax_it.set_title(r"$\mu = " + f"{EXP_N_SQP_mu[i]:.2f}" + r"\ \sigma = " + f"{EXP_N_SQP_sigma[i]:.2f}" + "$", size = axtitlesize)
        ax_it.set_xticks(xticks)
        ax_it.tick_params(labelsize = labelsize - 1)
        ax_it.legend(fontsize = 'x-large')
        
        ax_time.scatter(EXP_grid_sol[i], EXP_N_secs_sol[i], c = 'g', marker = 'o')
        ax_time.scatter(EXP_grid_part[i], EXP_N_secs_part[i], c = 'y', marker = 'v')
        ax_time.scatter(EXP_grid_fail[i], EXP_N_secs_fail[i], c = 'r', marker = 'x')
        
        ax_time.set_ylabel("solution time [s]", size = labelsize)
        ax_time.set_ylim(bottom = 0)
        ax_time.set_xlabel("location of perturbation", size = labelsize)
        ax_time.set_title(r"$\mu = " + f"{EXP_N_secs_mu[i]:.2f}" + r"\ \sigma = " + f"{EXP_N_secs_sigma[i]:.2f}" + "$", size = axtitlesize)
        ax_time.tick_params(labelsize = labelsize - 1)
        ax_time.set_xticks(xticks)
    if not isinstance(dirPath, Path):
        plt.show()
    else:
        dirPath.mkdir(parents = True, exist_ok = True)
        date_app = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_").replace("'", "")
        name_app = "" if suptitle is None else suptitle.replace(" ", "_").replace(":", "_").replace(".", "_").replace("'", "")        
        # sep = "" if dirPath[-1] == "/" else "/"
        pref = "" if savePrefix is None else savePrefix
        plt.savefig(dirPath / Path(pref + "_it_s_" + name_app + "_" + date_app))
    plt.close()


def plot_successful_small(n_EXP, nPert0, nPertF, titles, EXP_N_SQP, EXP_N_secs, EXP_type_sol, suptitle = None, dirPath : Path = None, savePrefix = None):
    """Plot result of multiple runs for perturbed start points for different options. 
    n_EXP - number of different options,
    nPert0, nPertF - start and end index of perturbed start points,
    EXP_... - vector of vectors of iterations counts, solution times and return codes for the pertubed start points.
    """
    if isinstance(dirPath, str):
        print("\n\nWARNING: Passing a pathstring to plot_successful_small is not recommended, use pathlib.Path instead\n", flush = True)
        dirPath = Path(dirPath)
    n_xticks = 10
    tdist = round((nPertF - nPert0)/n_xticks)
    tdist += (tdist==0)
    xticks = np.arange(nPert0, nPertF + tdist, tdist)
    ###############################################################################
    def F(x,r):
        if r > 0:
            return x
        else:
            return 0.00001    
    EXP_N_SQP_S = [[F(EXP_N_SQP[i][j], EXP_type_sol[i][j]) for j in range(nPertF - nPert0)] for i in range(n_EXP)]
    EXP_N_secs_S = [[F(EXP_N_secs[i][j], EXP_type_sol[i][j]) for j in range(nPertF - nPert0)] for i in range(n_EXP)]

    EXP_N_SQP_mu = [sum(EXP_N_SQP[i])/len(EXP_N_SQP[i]) for i in range(n_EXP)]
    EXP_N_SQP_sigma = [(sum((np.array(EXP_N_SQP[i]) - EXP_N_SQP_mu[i])**2)/len(EXP_N_SQP[i]))**(0.5) for i in range(n_EXP)]
    EXP_N_secs_mu = [sum(EXP_N_secs[i])/len(EXP_N_secs[i]) for i in range(n_EXP)]
    EXP_N_secs_sigma = [(sum((np.array(EXP_N_secs[i]) - EXP_N_secs_mu[i])**2)/len(EXP_N_secs[i]))**(0.5) for i in range(n_EXP)]
    
    ###############################################################################
    titlesize = 24
    axtitlesize = 23
    labelsize = 22
    
    fig, ax = plt.subplots(nrows = n_EXP, ncols = 2, constrained_layout=True, dpi = 300, figsize = (14+2*(max(n_EXP - 2, 0)), 2.5 + 2.5*(n_EXP - 1)))
    
    if isinstance(suptitle, str):
        if shutil.which("latex") is not None:
            fig.suptitle(r"$\textbf{" + suptitle + "}$", fontsize = titlesize, fontweight = 'bold')
        else:
            fig.suptitle(suptitle, fontsize = titlesize, fontweight = 'bold')
    for i in range(n_EXP):
        ax_it, ax_time = ax[i,:]
        ax_it.scatter(list(range(nPert0,nPertF)), EXP_N_SQP_S[i])
        ax_it.set_ylabel(titles[i], size = labelsize)
        ax_it.set_ylim(bottom = 0)
        if i == n_EXP - 1:
            ax_it.set_xlabel('location of perturbation', size = labelsize)
        if i == 0:
            ax_it.set_title('SQP iterations', size = axtitlesize)
            
        
        ax_it.set_xticks(xticks)
        ax_it.tick_params(labelsize = labelsize - 1)
        
        ax_time.scatter(list(range(nPert0,nPertF)), EXP_N_secs_S[i])
        ax_time.set_ylim(bottom = 0)
        if i == n_EXP - 1:
            ax_time.set_xlabel("location of perturbation", size = labelsize)
        if i == 0:
            ax_time.set_title('solution time [s]', size = axtitlesize)
        ax_time.set_xticks(xticks)
        ax_time.tick_params(labelsize = labelsize - 1)
    if not isinstance(dirPath, Path):
        plt.show()
    else:
        dirPath.mkdir(parents = True, exist_ok = True)
        date_app = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_").replace("'", "")
        name_app = "" if suptitle is None else suptitle.replace(" ", "_").replace(":", "_").replace(".", "_").replace("'", "")        
        pref = "" if savePrefix is None else savePrefix
        
        plt.savefig(dirPath / Path(pref + "_it_s_" + name_app + "_" + date_app))
    plt.close()



def print_heading(out, EXP_names : list[str]):
    """Prepare new file for later calling print_iterations on out"""
    out.write(" "*27)
    for EXP_name in EXP_names:
        out.write(EXP_name[0:40].ljust(21 + 5 + 21))
    out.write("\n" + " "*27)
    for i in range(len(EXP_names)):
        out.write("mu_N".ljust(10) + "sigma_N".ljust(11) + "mu_t".ljust(10) + "sigma_t".ljust(11))
        if i < len(EXP_names) - 1:
            out.write("|".ljust(5))
    out.write("\n")
    
def print_iterations(out, name, EXP_N_SQP, EXP_N_secs, EXP_type_sol):
    """Print iteration count and solution time - averages and 
    standard deviations to file, EXP_... being vectors returned by
    the perburbed_starts functions.
    """
    n_EXP = len(EXP_N_SQP)
    EXP_N_SQP_mu = [sum(EXP_N_SQP[i])/len(EXP_N_SQP[i]) for i in range(n_EXP)]
    EXP_N_SQP_sigma = [(sum((np.array(EXP_N_SQP[i]) - EXP_N_SQP_mu[i])**2)/len(EXP_N_SQP[i]))**(0.5) for i in range(n_EXP)]
    EXP_N_secs_mu = [sum(EXP_N_secs[i])/len(EXP_N_secs[i]) for i in range(n_EXP)]
    EXP_N_secs_sigma = [(sum((np.array(EXP_N_secs[i]) - EXP_N_secs_mu[i])**2)/len(EXP_N_secs[i]))**(0.5) for i in range(n_EXP)]
    
    out.write(name[:25].ljust(27))
    for i in range(n_EXP):
        out.write((f"{EXP_N_SQP_mu[i]:.2f}" + ",").ljust(10) + (f"{EXP_N_SQP_sigma[i]:.2f}" + ";").ljust(11) + (f"{EXP_N_secs_mu[i]:.2f}" + "s,").ljust(10) + (f"{EXP_N_secs_sigma[i]:.2f}" + "s").ljust(11))
        if i < n_EXP - 1:
            out.write("|".ljust(5))
    out.write("\n")
    

class out_dummy:
    def __init__(self):
        pass
    def write(self, Str : str):
        pass
    def close(self):
        pass


def run_ipopt_experiments(Examples : list[type[OCProblems.OCProblem]], Experiments : list[tuple[dict, str]], dirPath : Path, nPert0 = 0, nPertF = 40, file_output = True):
    if isinstance(dirPath, str):
        print("\n\nWARNING: Passing a pathstring to run_ipopt_experiments is not recommended, use pathlib.Path instead\n", flush = True)
        dirPath = Path(dirPath)
    dirPath.mkdir(parents = True, exist_ok = True)
    
    if file_output:
        date_app = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_").replace("'", "")
        pref = "ipopt"
        filePath = dirPath / Path(pref + "_it_" + date_app + ".txt")
        out = open(filePath, 'w')
    else:
        out = out_dummy()
    
    titles = [EXP_name for _, EXP_name in Experiments]
    print_heading(out, titles)
    #########
    for OCclass in Examples:        
        OCprob = OCclass(nt=100, integrator='RK4', parallel = True)
        itMax = 200
        # ipopts_base = {'max_iter':itMax}
        EXP_N_SQP = []
        EXP_N_secs = []
        EXP_type_sol = []
        n_EXP = 0
        for EXP_opts, EXP_name in Experiments:
            ret_N_SQP, ret_N_secs, ret_type_sol = casadi_solver_perturbed_starts('ipopt', OCprob, EXP_opts, nPert0, nPertF, itMax = itMax)
            EXP_N_SQP.append(ret_N_SQP)
            EXP_N_secs.append(ret_N_secs)
            EXP_type_sol.append(ret_type_sol)
            n_EXP += 1
        ###############################################################################
        plot_successful(n_EXP, nPert0, nPertF,\
            titles, EXP_N_SQP, EXP_N_secs, EXP_type_sol,\
            suptitle = OCclass.__name__, dirPath = dirPath, savePrefix = "ipopt")
        print_iterations(out, OCclass.__name__, EXP_N_SQP, EXP_N_secs, EXP_type_sol)
    out.close()


def run_blockSQP_experiments(Examples : list[type[OCProblems.OCProblem]], Experiments : list[tuple[blockSQP2.SQPoptions, str]], dirPath : str, nPert0 = 0, nPertF = 40, file_output = True, **kwargs):
    if isinstance(dirPath, str):
        print("\n\nWARNING: Passing a pathstring to run_ipopt_experiments is not recommended, use pathlib.Path instead\n", flush = True)
        dirPath = Path(dirPath)
    dirPath.mkdir(parents = True, exist_ok = True)
    if file_output:
        date_app = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_").replace("'", "")
        pref = "blockSQP"
        filePath = dirPath / Path(pref + "_it_" + date_app + ".txt")
        out = open(filePath, 'w')
    else:
        out = out_dummy()
    titles = [EXP_name for _, EXP_name in Experiments]
    print_heading(out, titles)
    
    for OCclass in Examples:        
        OCprob = OCclass(**kwargs)
        itMax = 200
        titles = []
        EXP_N_SQP = []
        EXP_N_secs = []
        EXP_type_sol = []
        n_EXP = 0
        for EXP_opts, EXP_name in Experiments:
            ret_N_SQP, ret_N_secs, ret_type_sol = perturbed_starts(OCprob, EXP_opts, nPert0, nPertF, itMax = itMax)
            EXP_N_SQP.append(ret_N_SQP)
            EXP_N_secs.append(ret_N_secs)
            EXP_type_sol.append(ret_type_sol)
            titles.append(EXP_name)
            n_EXP += 1
        ###############################################################################
        plot_successful(n_EXP, nPert0, nPertF,\
            titles, EXP_N_SQP, EXP_N_secs, EXP_type_sol,\
            suptitle = OCclass.__name__, dirPath = dirPath, savePrefix = "blockSQP")
        print_iterations(out, OCclass.__name__, EXP_N_SQP, EXP_N_secs, EXP_type_sol)
    out.close()