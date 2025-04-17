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
OCprob = OCProblems.Lotka_Volterra_Fishing(nt = 100, parallel = False, integrator = 'rk4')

nPert0 = 0
nPertF = 5
EXP = (1,2,3)

titles = [
    "SR1-BFGS",
    "convexification strategy 1",
    "convexification strategy 2",
    # "Convexification strategy 2",
    # "Convexification strategy 2, automatic scaling"
    # ""
    ]
itMax = 400
###############################################################################

opts = py_blockSQP.SQPoptions();
opts.max_QP_iter = 100000000
opts.max_conv_QPs = 1
opts.conv_strategy = 0
opts.enable_feasibility_restoration = 1
opts.max_QP_seconds = 5.0
opts.initial_hess_scale = 1.0
opts.hess_approximation = 1
opts.sizing_strategy = 2
opts.fallback_approximation = 2
opts.fallback_sizing_strategy = 4
opts.BFGS_damping_factor = 1/3

opts.exact_hess_usage = 0
opts.limited_memory = True
opts.memory_size = 20
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
    
    ret_N_SQP, ret_N_secs, ret_type_sol = OCP_experiment.perturbed_starts(OCprob, opts, nPert0, nPertF, itMax = itMax, COND = False)
    EXP_N_SQP.append(ret_N_SQP)
    EXP_N_secs.append(ret_N_secs)
    EXP_type_sol.append(ret_type_sol)
    n_EXP += 1
if 2 in EXP:
    opts.max_conv_QPs = 4
    opts.conv_strategy = 1
    opts.automatic_scaling = False
    
    ret_N_SQP, ret_N_secs, ret_type_sol = OCP_experiment.perturbed_starts(OCprob, opts, nPert0, nPertF, itMax = itMax, COND = False)
    EXP_N_SQP.append(ret_N_SQP)
    EXP_N_secs.append(ret_N_secs)
    EXP_type_sol.append(ret_type_sol)
    n_EXP += 1
if 3 in EXP:
    opts.max_conv_QPs = 4
    opts.conv_strategy = 2
    opts.automatic_scaling = False
    
    ret_N_SQP, ret_N_secs, ret_type_sol = OCP_experiment.perturbed_starts(OCprob, opts, nPert0, nPertF, itMax = itMax)
    EXP_N_SQP.append(ret_N_SQP)
    EXP_N_secs.append(ret_N_secs)
    EXP_type_sol.append(ret_type_sol)
    n_EXP += 1
###############################################################################
OCP_experiment.plot_successful(n_EXP, nPert0, nPertF, titles, EXP_N_SQP, EXP_N_secs, EXP_type_sol)

# OCP_experiment.plot_varshape(n_EXP, nPert0, nPertF, titles, EXP_N_SQP, EXP_N_secs, EXP_type_sol)

























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
