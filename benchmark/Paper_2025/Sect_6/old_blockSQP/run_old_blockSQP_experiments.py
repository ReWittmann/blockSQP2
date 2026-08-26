# blockSQP_reference_build - build system and Python interface for blockSQP
# Copyright (C) 2025-2026 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
# Licensed under the zlib license. See LICENSE for more details.


import datetime
import py_blockSQP_old

import sys
from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()
sys.path.append(str(cD.parents[2]))
import OCP_experiment
import OCProblems
import numpy as np
import time

Examples = [
            (OCProblems.Apollo_Reentry, dict(), None),            #Doesnt work
            
            (OCProblems.Batch_Distillation, dict(), None),
            # (OCProblems.Batch_Reactor, dict(), None),
            # (OCProblems.Batch_Reactor_OED, dict(), None),
            # (OCProblems.Calcium_Oscillation, dict(), None),
            # (OCProblems.Cart_Pendulum, dict(), None),
            # (OCProblems.Cart_Pendulum, OCProblems.Cart_Pendulum.param_set_2, "Cart_Pendulum_2"),
            # (OCProblems.Catalyst_Mixing, dict(), None),
            # (OCProblems.Catalyst_Mixing_OED, dict(), None),
            # (OCProblems.Cushioned_Oscillation, dict(), None),
            # (OCProblems.Dielectrophoretic_Particle, dict(), None),
            # (OCProblems.D_Onofrio_Chemotherapy, dict(), "D_Onofrio_Chemotherapy"),
            # (OCProblems.D_Onofrio_Chemotherapy, OCProblems.D_Onofrio_Chemotherapy.param_set_2, "D_Onofrio_Chemotherapy_2"),
            # (OCProblems.D_Onofrio_Chemotherapy, OCProblems.D_Onofrio_Chemotherapy.param_set_3, "D_Onofrio_Chemotherapy_3"),
            # (OCProblems.D_Onofrio_Chemotherapy, OCProblems.D_Onofrio_Chemotherapy.param_set_4, "D_Onofrio_Chemotherapy_4"),
            # (OCProblems.Ducted_Fan, dict(), None),
            # (OCProblems.Egerstedt_Standard, dict(), None),
            # (OCProblems.Electric_Car, dict(), None),
            # (OCProblems.Fermenter, dict(), None),
            # (OCProblems.Goddard_Rocket, dict(), None),
            # (OCProblems.Hang_Glider, dict(), None),
            # (OCProblems.Hanging_Chain, dict(), None),
            # (OCProblems.Lotka_Volterra_Fishing, dict(), None),
            # (OCProblems.Lotka_OED, dict(), None),
            # (OCProblems.Lotka_Volterra_Competitive, dict(), None),
            # (OCProblems.Lotka_Volterra_Competitive, OCProblems.Lotka_Volterra_Competitive.param_set_2, "Lotka_Volterra_Competitive_2"),
            # (OCProblems.Lotka_Volterra_Shared, dict(), None),
            # (OCProblems.Lotka_Volterra_Shared, OCProblems.Lotka_Volterra_Shared.param_set_2, "Lotka_Volterra_Shared_2"),
            
            (OCProblems.Lotka_Shared_OED, dict(), None),          #Doesnt work
            
            # (OCProblems.Ocean, dict(), None),
            # (OCProblems.Particle_Steering, dict(), None),
            # (OCProblems.Quadrotor_Helicopter, dict(), None),
            # (OCProblems.Satellite_Deorbiting, dict(), None),
            # (OCProblems.Three_Tank_Multimode, dict(), None),
            # (OCProblems.Time_Optimal_Car, dict(), None),
            # (OCProblems.Tubular_Reactor, dict(), None),
            ]

opt_SR1_BFGS = py_blockSQP_old.SQPoptions()
opt_SR1_BFGS.maxTimeQP = 10.0

opt_conv_comb_4 = py_blockSQP_old.SQPoptions()
opt_conv_comb_4.maxTimeQP = 10.0
opt_conv_comb_4.maxConvQP = 4


Experiments = [
                (opt_SR1_BFGS, "SR1-BFGS"),
                (opt_conv_comb_4, "blockSQP (convex combinations)"),
               ]


file_output = True
plot_folder = cD / Path("out_old_blockSQP_experiments")

nPert0 = 0
nPertF = 3
dirPath = plot_folder
dirPath.mkdir(parents = True, exist_ok = True)
if file_output:
    date_app = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_").replace("'", "")
    pref = "blockSQP"
    filePath = dirPath / Path(pref + "_it_" + date_app + ".txt")
    out = open(filePath, 'w')
else:
    out = OCP_experiment.out_dummy()
titles = [EXP_name for _, EXP_name in Experiments]



def create_prob_old(OCprob : OCProblems.OCProblem):    
    prob = py_blockSQP_old.Problemspec()
    prob.x_start = OCprob.start_point
    
    prob.nVar = OCprob.nVar
    prob.nCon = OCprob.nCon
    prob.f = OCprob.f
    prob.grad_f = OCprob.grad_f
    prob.g = OCprob.g
    prob.make_sparse(OCprob.jac_g_nnz, OCprob.jac_g_row, OCprob.jac_g_colind)
    prob.jac_g_nz = OCprob.jac_g_nz
    prob.hess = OCprob.hess_lag
    prob.set_blockIndex(np.array(OCprob.hessBlock_index, dtype = np.int32))
    prob.set_bounds(OCprob.lb_var, OCprob.ub_var, OCprob.lb_con, OCprob.ub_con)
    prob.lam_start = np.zeros(prob.nVar + prob.nCon, dtype = np.float64).reshape(-1)
    
    return prob

def blockSQP_perturbed_starts(OCprob : OCProblems.OCProblem, opts : py_blockSQP_old.SQPoptions, nPert0, nPertF, COND = False, itMax = 100):
    N_SQP = []
    N_secs = []
    type_sol = []
    for j in range(nPert0,nPertF):
        start_it = OCprob.perturbed_start_point(j)
        
        prob = create_prob_old(OCprob)
        prob.x_start = start_it
        prob.complete()
        stats = py_blockSQP_old.SQPstats("./solver_outputs")        
        t0 = time.monotonic()
        optimizer = py_blockSQP_old.SQPmethod(prob, opts, stats)
        optimizer.init()
        ret = optimizer.run(itMax)
        optimizer.finish()
        t1 = time.monotonic()
        
        N_SQP.append(stats.itCount)
        N_secs.append(t1 - t0)
        if int(ret) >= 0:
            type_sol.append(int(ret))
        else:
            type_sol.append(-1)    
    return N_SQP, N_secs, type_sol


namejust = OCP_experiment.max_example_name_length(Examples) + 2
OCP_experiment.print_heading(out, titles, namejust = namejust)
for OCclass, OCargs, OCname in Examples:    
    OCprob = OCclass(nt = 100, parallel = True, **OCargs)
    itMax = 200
    titles = []
    EXP_N_SQP = []
    EXP_N_secs = []
    EXP_type_sol = []
    n_EXP = 0
    for EXP_opts, EXP_name in Experiments:
        ret_N_SQP, ret_N_secs, ret_type_sol = blockSQP_perturbed_starts(OCprob, EXP_opts, nPert0, nPertF, itMax = itMax)
        EXP_N_SQP.append(ret_N_SQP)
        EXP_N_secs.append(ret_N_secs)
        EXP_type_sol.append(ret_type_sol)
        titles.append(EXP_name)
        n_EXP += 1
    ###############################################################################
    if OCname is None:
        OCname = OCclass.__name__
    
    OCP_experiment.plot_successful(n_EXP, nPert0, nPertF,\
        titles, EXP_N_SQP, EXP_N_secs, EXP_type_sol,\
        suptitle = OCname, dirPath = dirPath, savePrefix = "blockSQP")
    OCP_experiment.print_iterations(out, OCname, EXP_N_SQP, EXP_N_secs, EXP_type_sol)
out.close()

