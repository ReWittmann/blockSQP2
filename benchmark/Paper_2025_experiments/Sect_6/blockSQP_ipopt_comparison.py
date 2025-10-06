import numpy as np
import os
import sys
import os
import sys
try:
    cD = os.path.dirname(os.path.abspath(__file__))
except:
    cD = os.getcwd()
sys.path += [cD + "/../..", cD + "/../../.."]
import py_blockSQP
import matplotlib.pyplot as plt
import time
import copy
import datetime
import OCP_experiment
import OCProblems

#RK4/collocation/cvodes
ODE_integrator = 'collocation'
dirPath = cD + "/out_blockSQP_ipopt_comparison_collocation"


Examples = [
            # OCProblems.Batch_Reactor,
            # OCProblems.Cart_Pendulum,
            # OCProblems.Catalyst_Mixing,
            # OCProblems.Cushioned_Oscillation,
            # OCProblems.Egerstedt_Standard,
            # OCProblems.Electric_Car,
            # OCProblems.Goddard_Rocket,
            OCProblems.Hang_Glider,
            # OCProblems.Hanging_Chain,
            # OCProblems.Lotka_Volterra_Fishing,
            # OCProblems.Particle_Steering,
            # OCProblems.Quadrotor_Helicopter,
            # OCProblems.Three_Tank_Multimode,
            # OCProblems.Time_Optimal_Car,
            # OCProblems.Tubular_Reactor,
            # OCProblems.Lotka_OED,
            ]
OCProblems.Goddard_Rocket.__name__ = 'Goddard\'s Rocket'

if ODE_integrator == 'collocation' and OCProblems.Hang_Glider in Examples:
    Examples = [x for x in Examples if x != OCProblems.Hang_Glider]



ipopt_Experiments = [
                     ({'ipopt':{'hessian_approximation': 'limited-memory', 'tol': 1e-6, 'constr_viol_tol': 1e-6}}, 'ipopt, limited-memory'),
                     ({'ipopt':{'hessian_approximation': 'exact', 'tol': 1e-6, 'constr_viol_tol': 1e-6}}, 'ipopt, exact Hessian')
                     ]

def opt_conv_str_2_par_scale(max_conv_QPs = 4):
    opts = py_blockSQP.SQPoptions()
    opts.max_conv_QPs = max_conv_QPs
    opts.conv_strategy = 2
    opts.par_QPs = True
    opts.automatic_scaling = True
    return opts

opt1 = opt_conv_str_2_par_scale(max_conv_QPs = 4)
opt2 = opt_conv_str_2_par_scale(max_conv_QPs = 4)
opt2.exact_hess = 2


blockSQP_Experiments = [
                        (opt1, 'blockSQP, SR1-...-BFGS'),
                        (opt2, 'blockSQP, exH-...-BFGS')
                        ]


nPert0 = 0
nPertF = 2

if not os.path.exists(dirPath):
    os.makedirs(dirPath)

date_app = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_").replace("'", "")
sep = "" if dirPath[-1] == "/" else "/"
pref = "ipopt"
filePath = dirPath + sep + pref + "_it_" + date_app + ".txt"

out = open(filePath, 'w')


titles = [EXP_name for _, EXP_name in ipopt_Experiments + blockSQP_Experiments]
OCP_experiment.print_heading(out, titles)
#########
for OCclass in Examples:
    if OCclass == OCProblems.Hang_Glider and ODE_integrator == 'collocation':
        OCprob = OCclass(nt=100, integrator='cvodes', parallel = True)
    else:
        OCprob = OCclass(nt=100, integrator=ODE_integrator, parallel = True)
    itMax = 1000
    ipopts_base = {'max_iter':itMax}
    EXP_N_SQP = []
    EXP_N_secs = []
    EXP_type_sol = []
    n_EXP = 0
    for EXP_opts, EXP_name in ipopt_Experiments:
        ipopts = copy.deepcopy(EXP_opts)
        try:
            ipopts['ipopt']['max_iter'] = itMax
        except KeyError:
            ipopts['ipopt'] = {'max_iter':itMax}
        ret_N_SQP, ret_N_secs, ret_type_sol = OCP_experiment.casadi_solver_perturbed_starts('ipopt', OCprob, ipopts, nPert0, nPertF, itMax = itMax)
        EXP_N_SQP.append(ret_N_SQP)
        EXP_N_secs.append(ret_N_secs)
        EXP_type_sol.append(ret_type_sol)
        n_EXP += 1
    
    for EXP_opts, EXP_name in blockSQP_Experiments:
        ret_N_SQP, ret_N_secs, ret_type_sol = OCP_experiment.perturbed_starts(OCprob, EXP_opts, nPert0, nPertF, itMax = itMax)
        EXP_N_SQP.append(ret_N_SQP)
        EXP_N_secs.append(ret_N_secs)
        EXP_type_sol.append(ret_type_sol)
        titles.append(EXP_name)
        n_EXP += 1
    
    ###############################################################################
    OCP_experiment.plot_successful(n_EXP, nPert0, nPertF,\
        titles, EXP_N_SQP, EXP_N_secs, EXP_type_sol,\
        suptitle = OCclass.__name__, dirPath = dirPath, savePrefix = "blockSQP_ipopt")
    OCP_experiment.print_iterations(out, OCclass.__name__, EXP_N_SQP, EXP_N_secs, EXP_type_sol)
out.close()
