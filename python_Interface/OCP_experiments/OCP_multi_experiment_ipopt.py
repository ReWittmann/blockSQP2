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
OCprob = OCProblems.Egerstedt_Standard(nt = 100, parallel = False, integrator = 'RK4')

nPert0 = 0
nPertF = 5
EXP = (1,2)

suptitle = 'Egerstedt standard'
titles = [
    "Ipopt, limited-memory BFGS",
    "Ipopt, exact Hessian"
    ]
itMax = 200
###############################################################################

ipopts = {'hessian_approximation': 'limited-memory', 'limited_memory_max_history':20,\
          'constr_viol_tol':1e-5, 'tol':1e-5, 'max_iter':itMax}
EXP_N_SQP = []
EXP_N_secs = []
EXP_type_sol = []
n_EXP = 0
if 1 in EXP:
    
    ret_N_SQP, ret_N_secs, ret_type_sol = OCP_experiment.ipopt_perturbed_starts(OCprob, ipopts, nPert0, nPertF, itMax = itMax)
    EXP_N_SQP.append(ret_N_SQP)
    EXP_N_secs.append(ret_N_secs)
    EXP_type_sol.append(ret_type_sol)
    n_EXP += 1
if 2 in EXP:
    ipopts['hessian_approximation'] = 'exact'
    ret_N_SQP, ret_N_secs, ret_type_sol = OCP_experiment.ipopt_perturbed_starts(OCprob, ipopts, nPert0, nPertF, itMax = itMax)
    EXP_N_SQP.append(ret_N_SQP)
    EXP_N_secs.append(ret_N_secs)
    EXP_type_sol.append(ret_type_sol)
    n_EXP += 1
if 3 in EXP:
    ret_N_SQP, ret_N_secs, ret_type_sol = OCP_experiment.perturbed_starts(OCprob, ipopts, nPert0, nPertF, itMax = itMax)
    EXP_N_SQP.append(ret_N_SQP)
    EXP_N_secs.append(ret_N_secs)
    EXP_type_sol.append(ret_type_sol)
    n_EXP += 1
###############################################################################
OCP_experiment.plot_successful(n_EXP, nPert0, nPertF, titles, EXP_N_SQP, EXP_N_secs, EXP_type_sol, suptitle = suptitle)

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
