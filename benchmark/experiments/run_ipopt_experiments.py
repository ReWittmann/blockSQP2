# blockSQP2 -- A structure-exploiting nonlinear programming solver based
#              on blockSQP by Dennis Janka.
# Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>

# Licensed under the zlib license. See LICENSE for more details.


# \file run_ipopt_experiments.py
# \author Reinhold Wittmann
# \date 2025
#
# Script to benchmark the NLP solver ipopt several problems 
# for perturbed start points for different options.

import sys
from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()
sys.path += [str(cD.parents[0])]
import OCP_experiment
import OCProblems
import datetime

# Specify problem (class), non-default parameters and plot suptitle (None for default)
Examples = [
            (OCProblems.Apollo_Reentry, dict(), None),
            # (OCProblems.Batch_Distillation, dict(), None),
            (OCProblems.Batch_Reactor, dict(), None),
            (OCProblems.Batch_Reactor_OED, dict(), None),
            # (OCProblems.Calcium_Oscillation, dict(), None),
            (OCProblems.Cart_Pendulum, dict(), None),
            (OCProblems.Cart_Pendulum, OCProblems.Cart_Pendulum.param_set_2, "Cart_Pendulum_2"),
            (OCProblems.Catalyst_Mixing, dict(), None),
            (OCProblems.Catalyst_Mixing_OED, dict(), None),
            (OCProblems.Cushioned_Oscillation, dict(), None),
            (OCProblems.Dielectrophoretic_Particle, dict(), None),
            (OCProblems.D_Onofrio_Chemotherapy, dict(), "D_Onofrio_Chemotherapy"),
            (OCProblems.D_Onofrio_Chemotherapy, OCProblems.D_Onofrio_Chemotherapy.param_set_2, "D_Onofrio_Chemotherapy_2"),
            (OCProblems.D_Onofrio_Chemotherapy, OCProblems.D_Onofrio_Chemotherapy.param_set_3, "D_Onofrio_Chemotherapy_3"),
            (OCProblems.D_Onofrio_Chemotherapy, OCProblems.D_Onofrio_Chemotherapy.param_set_4, "D_Onofrio_Chemotherapy_4"),
            (OCProblems.Ducted_Fan, dict(), None),
            (OCProblems.Egerstedt_Standard, dict(), None),
            (OCProblems.Electric_Car, dict(), None),
            (OCProblems.Fermenter, dict(), None),
            (OCProblems.Goddard_Rocket, dict(), 'Goddard\'s Rocket'),
            (OCProblems.Hang_Glider, dict(), None),
            (OCProblems.Hanging_Chain, dict(), None),
            (OCProblems.Lotka_Volterra_Fishing, dict(), None),
            (OCProblems.Lotka_OED, dict(), None),
            (OCProblems.Lotka_Volterra_Competitive, dict(), None),
            (OCProblems.Lotka_Volterra_Competitive, OCProblems.Lotka_Volterra_Competitive.param_set_2, "Lotka_Volterra_Competitive_2"),
            (OCProblems.Lotka_Volterra_Shared, dict(), None),
            (OCProblems.Lotka_Volterra_Shared, OCProblems.Lotka_Volterra_Shared.param_set_2, "Lotka_Volterra_Shared_2"),
            (OCProblems.Lotka_Shared_OED, dict(), None),
            (OCProblems.Ocean, dict(), None),
            (OCProblems.Particle_Steering, dict(), None),
            (OCProblems.Quadrotor_Helicopter, dict(), None),
            (OCProblems.Satellite_Deorbiting, dict(), None),
            (OCProblems.Three_Tank_Multimode, dict(), None),
            (OCProblems.Time_Optimal_Car, dict(), None),
            (OCProblems.Tubular_Reactor, dict(), None),
            ]
Experiments = [
                # ({'ipopt': {'hessian_approximation': 'limited-memory', 'tol': 1e-6, 'max_iter': 300}}, "Ipopt, limited-memory"),
                ({'ipopt': {'hessian_approximation': "exact", 'tol': 1e-6, 'max_iter': 300}}, "Ipopt, exact Hessian"),
                ]


plot_folder = cD / Path("out_ipopt_experiments")

#Choose perturbed start points to test for,
#modify discretized initial controls u_k in turn for nPert0 <= k < nPertF
nPert0 = 0
nPertF = 10

#Write results to a file?
file_output = True


#Run all example problems for all option sets for perturbed start points
dirPath = plot_folder
dirPath.mkdir(parents = True, exist_ok = True)
if file_output:
    date_app = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_").replace("'", "")
    pref = "ipopt"
    filePath = dirPath / Path(pref + "_it_" + date_app + ".txt")
    out = open(filePath, 'w')
else:
    out = OCP_experiment.out_dummy()

    
titles = [EXP_name for _, EXP_name in Experiments]

namejust = OCP_experiment.max_example_name_length(Examples) + 2
OCP_experiment.print_heading(out, titles, namejust = namejust)
for OCclass, OCargs, OCname in Examples:        
    OCprob = OCclass(nt = 100, parallel = True, **OCargs)
    itMax = 300
    titles = []
    EXP_N_SQP = []
    EXP_N_secs = []
    EXP_type_sol = []
    n_EXP = 0
    
    for EXP_opts, EXP_name in Experiments:
        ret_N_SQP, ret_N_secs, ret_type_sol = OCP_experiment.casadi_solver_perturbed_starts('ipopt', OCprob, EXP_opts, nPert0, nPertF, itMax = itMax)
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
        suptitle = OCname, dirPath = dirPath, savePrefix = "ipopt")
    OCP_experiment.print_iterations(out, OCname, EXP_N_SQP, EXP_N_secs, EXP_type_sol, namejust = namejust)
out.close()


