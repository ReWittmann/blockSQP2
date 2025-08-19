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
OCprob = OCProblems.Electric_Car(nt = 100, parallel = False)

nPert0 = 0
nPertF = 40
EXP = (3,)

titles = [
    # "SR1-BFGS",
    # "Convexification strategy 1",
    # "Convexification strategy 2",
    None
    # "Convexification strategy 2",
    # "Convexification strategy 2, automatic scaling"
    # ""
    ]
itMax = 400
###############################################################################

opts = py_blockSQP.SQPoptions();
opts.max_QP_it = 10000
opts.max_conv_QPs = 1
opts.conv_strategy = 0
opts.enable_rest = True
opts.max_QP_secs = 5.0
opts.initial_hess_scale = 1.0
opts.hess_approx = 1
opts.sizing = 2
opts.fallback_approx = 2
opts.fallback_sizing = 4
opts.BFGS_damping_factor = 1/3

opts.exact_hess = 0
opts.lim_mem = True
opts.mem_size = 20
opts.opt_tol = 1e-6
opts.feas_tol = 1e-6

opts.qpsol = 'qpOASES'
QPOPTS = py_blockSQP.qpOASES_options()
QPOPTS.printLevel = 0
QPOPTS.terminationTolerance = 1e-10
opts.qpsol_options = QPOPTS

opts.automatic_scaling = False
opts.enable_premature_termination = False
opts.max_filter_overrides = 0
opts.max_extra_steps = 0

EXP_N_SQP = []
EXP_N_secs = []
EXP_type_sol = []
n_EXP = 0
if 1 in EXP:
    opts.max_conv_QPs = 1
    opts.conv_strategy = 1
    opts.automatic_scaling = False
    opts.enable_premature_termination = False
    # opts.BFGS_damping_factor = 1/3
    
    ret_N_SQP, ret_N_secs, ret_type_sol = OCP_experiment.perturbed_starts(OCprob, opts, nPert0, nPertF, itMax = itMax, COND = False)
    EXP_N_SQP.append(ret_N_SQP)
    EXP_N_secs.append(ret_N_secs)
    EXP_type_sol.append(ret_type_sol)
    n_EXP += 1
if 2 in EXP:
    opts.max_conv_QPs = 4
    opts.conv_strategy = 1
    opts.automatic_scaling = False
    # opts.BFGS_damping_factor = 1/3
    
    ret_N_SQP, ret_N_secs, ret_type_sol = OCP_experiment.perturbed_starts(OCprob, opts, nPert0, nPertF, itMax = itMax, COND = False)
    EXP_N_SQP.append(ret_N_SQP)
    EXP_N_secs.append(ret_N_secs)
    EXP_type_sol.append(ret_type_sol)
    n_EXP += 1
if 3 in EXP:
    opts.max_conv_QPs = 4
    opts.conv_strategy = 2
    opts.automatic_scaling = False
    opts.par_QPs = False
    # opts.BFGS_damping_factor = 1/3
    opts.enable_premature_termination = True
    
    ret_N_SQP, ret_N_secs, ret_type_sol = OCP_experiment.perturbed_starts(OCprob, opts, nPert0, nPertF, itMax = itMax)
    EXP_N_SQP.append(ret_N_SQP)
    EXP_N_secs.append(ret_N_secs)
    EXP_type_sol.append(ret_type_sol)
    n_EXP += 1
###############################################################################
# OCP_experiment.plot_successful(n_EXP, nPert0, nPertF, titles, EXP_N_SQP, EXP_N_secs, EXP_type_sol)

OCP_experiment.plot_varshape(n_EXP, nPert0, nPertF, titles, EXP_N_SQP, EXP_N_secs, EXP_type_sol, dirPath = "/home/reinhold/PLOT")

