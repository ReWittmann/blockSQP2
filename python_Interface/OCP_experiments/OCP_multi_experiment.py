import numpy as np
import os
import sys
import os
import sys
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
except:
    sys.path.append(os.getcwd() + "/..")
import py_blockSQP
from blockSQP_pyProblem import blockSQP_pyProblem as Problemspec
import matplotlib.pyplot as plt
import time
import copy
import OCP_experiment

import OCProblems
#Available problems:
# ['Lotka_Volterra_Fishing', 'Lotka_Volterra_multimode', 'Goddard_Rocket', 
#  'Calcium_Oscillation', 'Batch_Reactor', 'Bioreactor', 'Hanging_Chain', 
#  'Hanging_Chain_NQ', 'Catalyst_Mixing', 'Cushioned_Oscillation', 
#  'D_Onofrio_Chemotherapy', 'D_Onofrio_Chemotherapy_VT', 'Egerstedt_Standard', 
#  'Fullers', 'Electric_Car', 'F8_Aircraft', 'Gravity_Turn', 'Oil_Shale_Pyrolysis', 
# 'Particle_Steering', 'Quadrotor_Helicopter', 'Supermarket_Refrigeration', 
#  'Three_Tank_Multimode', 'Time_Optimal_Car', 'Van_der_Pol_Oscillator', 
#  'Van_der_Pol_Oscillator_2', 'Van_der_Pol_Oscillator_3', 'Ocean',
#  'Lotka_OED', 'Fermenter', 'Batch_Distillation', 'Hang_Glider',
#  'Tubular_Reactor]

###############################################################################
OCprob = OCProblems.Goddard_Rocket(nt=100, parallel = False, integrator = 'rk4')

nPert0 = 0
nPertF = 40
EXP = (1,2)

titles = [
    # "SR1-BFGS",
    # "Cushioned oscillation, convexification strategy 2, automatic scaling, stage duration parameters scaled by 100",
    # "Lotka Volterra fishing, convexification strategy 2, automatic scaling",
    "Goddard's rocket, convexification strategy 2, no automatic scaling",
    "Goddard's rocket, convexification strategy 2, automatic scaling"]
itMax = 400
###############################################################################

opts = py_blockSQP.SQPoptions();
opts.maxItQP = 100000000
opts.maxConvQP = 1
opts.convStrategy = 0
opts.restoreFeas = 1
opts.maxTimeQP = 5.0
opts.iniHessDiag = 1.0
opts.hessUpdate = 1
opts.hessScaling = 2
opts.fallbackUpdate = 2
opts.fallbackScaling = 4
opts.hessDampFac = 1/3

opts.whichSecondDerv = 0
opts.hessLimMem = True
opts.hessMemsize = 20
opts.opttol = 1e-6
opts.nlinfeastol = 1e-6


opts.QPsol = 'qpOASES'
QPOPTS = py_blockSQP.qpOASES_options()
QPOPTS.printLevel = 0
QPOPTS.terminationTolerance = 1e-10
opts.QPsol_opts = QPOPTS

opts.autoScaling = False

opts.allow_premature_termination = False
opts.max_local_lenience = 0
opts.max_extra_steps = 0

EXP_N_SQP = []
EXP_N_secs = []
EXP_type_sol = []
n_EXP = 0
if 1 in EXP:
    opts.maxConvQP = 4
    opts.convStrategy = 2
    opts.autoScaling = False
    opts.allow_premature_termination = False
    
    ret_N_SQP, ret_N_secs, ret_type_sol = OCP_experiment.perturbed_starts(OCprob, opts, nPert0, nPertF, itMax = itMax, COND = False)
    EXP_N_SQP.append(ret_N_SQP)
    EXP_N_secs.append(ret_N_secs)
    EXP_type_sol.append(ret_type_sol)
    n_EXP += 1
if 2 in EXP:
    opts.maxConvQP = 4
    opts.convStrategy = 2
    opts.autoScaling = True
    # OCprob = OCProblems.Cushioned_Oscillation_SS(nt = 100, parallel = False)
    
    
    ret_N_SQP, ret_N_secs, ret_type_sol = OCP_experiment.perturbed_starts(OCprob, opts, nPert0, nPertF, itMax = itMax, COND = False)
    EXP_N_SQP.append(ret_N_SQP)
    EXP_N_secs.append(ret_N_secs)
    EXP_type_sol.append(ret_type_sol)
    n_EXP += 1
if 3 in EXP:
    opts.maxConvQP = 4
    opts.convStrategy = 2
    opts.autoScaling = False
    
    
    ret_N_SQP, ret_N_secs, ret_type_sol = OCP_experiment.perturbed_starts(OCprob, opts, nPert0, nPertF, itMax = itMax)
    EXP_N_SQP.append(ret_N_SQP)
    EXP_N_secs.append(ret_N_secs)
    EXP_type_sol.append(ret_type_sol)
    n_EXP += 1
###############################################################################
OCP_experiment.plot_successful(n_EXP, nPert0, nPertF, titles, EXP_N_SQP, EXP_N_secs, EXP_type_sol)
# OCP_experiment.plot_varshape(n_EXP, nPert0, nPertF, titles, EXP_N_SQP, EXP_N_secs, EXP_type_sol)


# n_xticks = 10
# tdist = round((nPertF - nPert0)/n_xticks)
# tdist += (tdist==0)
# xticks = np.arange(nPert0, nPertF + tdist, tdist)
# ###############################################################################
# EXP_grid = [list(range(nPert0, nPertF)) for i in range(n_EXP)]
# EXP_grid_sol = [[EXP_grid[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] > 0] for i in range(n_EXP)]
# EXP_grid_part = [[EXP_grid[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] == 0] for i in range(n_EXP)]
# EXP_grid_fail = [[EXP_grid[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] < 0] for i in range(n_EXP)]

# EXP_N_SQP_sol = [[EXP_N_SQP[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] > 0] for i in range(n_EXP)]
# EXP_N_SQP_part = [[EXP_N_SQP[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] == 0] for i in range(n_EXP)]
# EXP_N_SQP_fail = [[EXP_N_SQP[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] < 0] for i in range(n_EXP)]
# EXP_N_secs_sol = [[EXP_N_secs[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] > 0] for i in range(n_EXP)]
# EXP_N_secs_part = [[EXP_N_secs[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] == 0] for i in range(n_EXP)]
# EXP_N_secs_fail = [[EXP_N_secs[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] < 0] for i in range(n_EXP)]


# EXP_N_SQP_clean = [[EXP_N_SQP[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] >= 0] for i in range(n_EXP)]
# EXP_N_secs_clean = [[EXP_N_secs[i][j] for j in range(nPertF - nPert0) if EXP_type_sol[i][j] >= 0] for i in range(n_EXP)]

# EXP_N_SQP_mu = [sum(EXP_N_SQP_clean[i])/len(EXP_N_SQP_clean[i]) for i in range(n_EXP)]
# EXP_N_SQP_sigma = [(sum((np.array(EXP_N_SQP_clean[i]) - EXP_N_SQP_mu[i])**2)/len(EXP_N_SQP_clean[i]))**(0.5) for i in range(n_EXP)]

# EXP_N_secs_mu = [sum(EXP_N_secs_clean[i])/len(EXP_N_secs_clean[i]) for i in range(n_EXP)]
# EXP_N_secs_sigma = [(sum((np.array(EXP_N_secs_clean[i]) - EXP_N_secs_mu[i])**2)/len(EXP_N_secs_clean[i]))**(0.5) for i in range(n_EXP)]

# #Care, doen't work for numbers smaller than 0.0001, representation becomes *e-***
# trunc_float = lambda num, dg: str(float(num))[0:int(np.ceil(abs(np.log(num + (num == 0))/np.log(10)))) + 2 + dg]

# ccodemp = {-1: 'r', 0:'y', 1:'g'}
# cmap = [[ccodemp[v] for v in EXP_type_sol[i]] for i in range(n_EXP)]

# ###############################################################################
# titlesize = 19
# labelsize = 12

# fig = plt.figure(constrained_layout=True, dpi = 300, figsize = (14+2*(max(n_EXP - 2, 0)),3.5 + 6.5*(n_EXP-1)))
# subfigs = fig.subfigures(nrows=n_EXP, ncols=1)

# for i in range(n_EXP):
#     ax_it, ax_time = subfigs[i].subplots(nrows=1,ncols=2)
#     subfigs[i].suptitle(titles[i], size = titlesize)
    
#     ax_it.scatter(list(range(nPert0,nPertF)), EXP_N_SQP[i], c = cmap[i])
#     ax_it.set_ylabel('SQP iterations', size = labelsize)
#     ax_it.set_ylim(bottom = 0)
#     ax_it.set_xlabel('location of perturbation', size = labelsize)
#     ax_it.set_title(r"$\mu = " + trunc_float(EXP_N_SQP_mu[i], 1) + r"\ \sigma = " + trunc_float(EXP_N_SQP_sigma[i], 1) + "$")
#     ax_it.set_xticks(xticks)
    
#     ax_time.scatter(list(range(nPert0,nPertF)), EXP_N_secs[i], c = cmap[i])
#     ax_time.set_ylabel("solution time in seconds", size = labelsize)
#     ax_time.set_ylim(bottom = 0)
#     ax_time.set_xlabel("location of perturbation", size = labelsize)
#     ax_time.set_title(r"$\mu = " + trunc_float(EXP_N_secs_mu[i], 1) + r"\ \sigma = " + trunc_float(EXP_N_secs_sigma[i], 1) + "$")
#     ax_time.set_xticks(xticks)

# plt.show()


























# OCprob = OCProblems.Lotka_Volterra_Fishing(nt=100, refine = 1, integrator = 'RK4', parallel=False)
# OCprob.integrate_full(OCprob.start_point)

# OCprob = OCProblems.Lotka_Volterra_Fishing(nt=NT, integrator = 'rk4', parallel=False)
# OCprob = OCProblems.Bioreactor(nt=100, integrator = 'rk4', parallel=False)
# OCprob = OCProblems.Goddard_Rocket(nt = NT, integrator = 'RK4', parallel = False)
# OCprob = OCProblems.Electric_Car(nt = NT, integrator = 'RK4', parallel = False)
# OCprob = OCProblems.Catalyst_Mixing(nt = NT, integrator = 'RK4', parallel = False)
# OCprob = OCProblems.Three_Tank_Multimode(nt = NT, integrator = 'RK4', parallel = False)
# OCprob = OCProblems.Egerstedt_Standard(nt = NT, integrator = 'RK4', parallel = False)
# OCprob = OCProblems.Fullers(nt = NT, integrator = 'RK4', parallel=False)
# OCprob = OCProblems.Lotka_OED(nt = NT, integrator = 'RK4', parallel = False)
# OCprob = OCProblems.Hanging_Chain(nt = NT, integrator = 'RK4', parallel = False)
# OCprob = OCProblems.Van_der_Pol_Oscillator_3(nt = NT, integrator = 'RK4', parallel = False)
# OCprob = OCProblems.Batch_Reactor(nt = NT, integration_method = 'rk4', parallel=False)
# OCprob = OCProblems.Hang_Glider(nt = NT, integrator='rk4', parallel=False)
# OCprob = OCProblems.Van_der_Pol_Oscillator_3(nt = NT, integrator='rk4', parallel=False)
# OCprob = OCProblems.Time_Optimal_Car(nt = NT, integrator='rk4', parallel=False)
# OCprob = OCProblems.Cushioned_Oscillation(nt = NT, integrator='rk4', parallel=False)


#Made worse (in SQP iterations) by autoscaling, but better in total time. 
# OCprob = OCProblems.Particle_Steering(nt = NT, integrator = 'RK4', parallel = False)


# OCprob = OCProblems.Lotka_Volterra_Fishing_BSC(nt=100, integrator='RK4', parallel=False, sca1=1.0e1, sca2=1.0e-3, sca3=1.0e-2)
# OCprob = OCProblems.Lotka_Volterra_Fishing_BSC(nt=100, integrator='RK4', parallel=False, sca1=1.0, sca2=1.0, sca3=1.0)

# OCprob = OCProblems.Three_Tank_Multimode_BSC(nt = NT, integrator = 'RK4', parallel = False, sca1 = 1.0e3, sca2 = 1.0, sca3 = 1.0e-3)
# OCprob = OCProblems.Egerstedt_Standard_BSC(nt=100,integrator='rk4',parallel=False,sca1=1e-2,sca2=1e-2,sca3=1e2) #Strong SR1 effect
# OCprob = OCProblems.Batch_Distillation(nt=65, integrator = 'cvodes', parallel = True)
