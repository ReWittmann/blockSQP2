# py_blockSQP -- A python interface to blockSQP 2, a nonlinear programming
#                solver based on blockSQP by Dennis Janka.
# Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
#
# Licensed under the zlib license. See LICENSE for more details.


# \file solver_comparison.py
# \author Reinhold Wittmann
# \date 2025
#
# Script to compare casadi NLP solvers and py_blockSQP on 
# several problems for perturbed start points for different options.

import sys
from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()
sys.path += [str(cD.parent), str(cD.parents[1])]

import copy
import py_blockSQP
import datetime
import OCP_experiment
import OCProblems

#RK4/collocation/cvodes
ODE_integrator = 'RK4'
dirPath = cD / Path("out_ipopt_blockSQP")

file_prefix = "ipopt_blockSQP"

nPert0 = 0
nPertF = 40

#Problems
Examples = [
            OCProblems.Batch_Reactor,
            OCProblems.Cart_Pendulum,
            OCProblems.Catalyst_Mixing,
            OCProblems.Cushioned_Oscillation,
            OCProblems.Egerstedt_Standard,
            OCProblems.Electric_Car,
            OCProblems.Goddard_Rocket,
            OCProblems.Hang_Glider,
            OCProblems.Hanging_Chain,
            OCProblems.Lotka_Volterra_Fishing,
            OCProblems.Particle_Steering,
            OCProblems.Quadrotor_Helicopter,
            OCProblems.Three_Tank_Multimode,
            OCProblems.Time_Optimal_Car,
            OCProblems.Tubular_Reactor,
            OCProblems.Lotka_OED,
            ]
OCProblems.Goddard_Rocket.__name__ = 'Goddard\'s Rocket'

#Experiments ~ Options + names

ipopt_Experiments = [
                     # ({'hessian_approximation': 'limited-memory', 'tol': 1e-6}, 'ipopt, limited-memory'),
                     ({'ipopt': {'hessian_approximation': 'exact', 'tol': 1e-6}}, 'ipopt, exact Hessian')
                     ]

casadi_blockSQP_Experiments = [({'block_hess':True, 'linsol':'ma27', 'max_iter': 1000}, 'casadi blockSQP')]

# worhp_Experiments = [({'worhp':{'ScaledKKT':False, 
#                                 'TolOpti':1e-6, 
#                                 'TolFeas':1e-6,
#                                 'UserHM' : False,
#                                 'BFGSmethod' : 100,
#                                 'BFGSmaxblockSize': 20,
#                                 }}, 'WORHP')]


def opt_conv_str_2_par_scale(max_conv_QPs = 4):
    opts = py_blockSQP.SQPoptions()
    opts.max_conv_QPs = max_conv_QPs
    opts.conv_strategy = 2
    opts.par_QPs = True
    opts.automatic_scaling = True
    return opts
opt1 = opt_conv_str_2_par_scale(max_conv_QPs = 4)
opt2 = opt_conv_str_2_par_scale(max_conv_QPs = 4)
opt2.hess_approx = 'exact'
blockSQP_Experiments = [
                        (opt1, 'blockSQP, SR1-...-BFGS'),
                        # (opt2, 'blockSQP, exH-...-BFGS')
                        ]

######################################

dirPath.mkdir(parents = True, exist_ok = True)
date_app = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_").replace("'", "")
filePath = dirPath / Path(file_prefix + "_it_" + date_app + ".txt")
out = open(filePath, 'w')


titles = [EXP_name for _, EXP_name in ipopt_Experiments + blockSQP_Experiments]
# titles = [EXP_name for _, EXP_name in casadi_blockSQP_Experiments]
# titles = [EXP_name for _, EXP_name in worhp_Experiments]


OCP_experiment.print_heading(out, titles)
#########
for OCclass in Examples:
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
        # ipopts = dict(ipopts_base)
        # ipopts.update(EXP_opts)
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
    
    # for EXP_opts, EXP_name in casadi_blockSQP_Experiments:
    #     ret_N_SQP, ret_N_secs, ret_type_sol = OCP_experiment.casadi_solver_perturbed_starts('blocksqp', OCprob, EXP_opts, nPert0, nPertF, itMax = itMax)
    #     EXP_N_SQP.append(ret_N_SQP)
    #     EXP_N_secs.append(ret_N_secs)
    #     EXP_type_sol.append(ret_type_sol)
    #     titles.append(EXP_name)
    #     n_EXP += 1
        
    # for EXP_opts, EXP_name in worhp_Experiments:
    #     ret_N_SQP, ret_N_secs, ret_type_sol = OCP_experiment.casadi_solver_perturbed_starts('worhp', OCprob, EXP_opts, nPert0, nPertF, itMax = itMax)
    #     EXP_N_SQP.append(ret_N_SQP)
    #     EXP_N_secs.append(ret_N_secs)
    #     EXP_type_sol.append(ret_type_sol)
    #     titles.append(EXP_name)
    #     n_EXP += 1
    
    ###############################################################################
    OCP_experiment.plot_successful(n_EXP, nPert0, nPertF,\
        titles, EXP_N_SQP, EXP_N_secs, EXP_type_sol,\
        suptitle = OCclass.__name__, dirPath = dirPath, savePrefix = file_prefix)
    OCP_experiment.print_iterations(out, OCclass.__name__, EXP_N_SQP, EXP_N_secs, EXP_type_sol)
out.close()
