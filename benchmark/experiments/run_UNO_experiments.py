# blockSQP2 -- A structure-exploiting nonlinear programming solver based
#              on blockSQP by Dennis Janka.
# Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>

# Licensed under the zlib license. See LICENSE for more details.


# \file run_UNO_experiments.py
# \author Reinhold Wittmann
# \date 2025
#
# Script to benchmark the NLP solver UNO on several problems 
# for perturbed start points for different options.

import numpy as np
import time
import datetime
import sys
from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()
sys.path += [str(cD.parent)]
import unopy
import OCProblems
import OCP_experiment




# Specify problem (class), non-default parameters and plot suptitle (None for default)
Examples = [
            (OCProblems.Batch_Reactor, dict(), None),
            (OCProblems.Cart_Pendulum, dict(), None),
            (OCProblems.Cart_Pendulum, OCProblems.Cart_Pendulum.param_set_2, "Cart_Pendulum_2"),
            # (OCProblems.Catalyst_Mixing, dict(), None),
            # (OCProblems.Cushioned_Oscillation, dict(), None),
            # (OCProblems.Ducted_Fan, dict(), None),
            # (OCProblems.Egerstedt_Standard, dict(), None),
            # (OCProblems.Electric_Car, dict(), None),
            # (OCProblems.Goddard_Rocket, dict(), 'Goddard\'s Rocket'),
            # (OCProblems.Hang_Glider, dict(), None),
            # (OCProblems.Hanging_Chain, dict(), None),
            # (OCProblems.Lotka_Volterra_Fishing, dict(), None),
            # (OCProblems.Particle_Steering, dict(), None),
            # (OCProblems.Quadrotor_Helicopter, dict(), None),
            # (OCProblems.Three_Tank_Multimode, dict(), None),
            # (OCProblems.Time_Optimal_Car, dict(), None),
            # (OCProblems.Tubular_Reactor, dict(), None),
            # (OCProblems.Lotka_OED, dict(), None),
            # (OCProblems.Fermenter, dict(), None),
            # (OCProblems.Satellite_Deorbiting_1, dict(), None),
            # (OCProblems.D_Onofrio_Chemotherapy, OCProblems.D_Onofrio_Chemotherapy.param_set_1 | {'integrator': 'RK4'}, "D_Onofrio_Chemotherapy_1"),
            # (OCProblems.D_Onofrio_Chemotherapy, OCProblems.D_Onofrio_Chemotherapy.param_set_2 | {'integrator': 'RK4'}, "D_Onofrio_Chemotherapy_2"),
            # (OCProblems.D_Onofrio_Chemotherapy, OCProblems.D_Onofrio_Chemotherapy.param_set_3 | {'integrator': 'RK4'}, "D_Onofrio_Chemotherapy_3"),
            # (OCProblems.D_Onofrio_Chemotherapy, OCProblems.D_Onofrio_Chemotherapy.param_set_4 | {'integrator': 'RK4'}, "D_Onofrio_Chemotherapy_4"),            
            # (OCProblems.Fermenter, dict(), None),
            ]


#Select option sets to test for

opt_ipopt_LBFGS = {
    'preset': 'ipopt',
    'hessian_model': 'LBFGS'
    }
opt_ipopt_exact = {
    'preset': 'ipopt',
    'hessian_model': 'exact'
    }
opt_filtersqp_LBFGS = {
    'preset': 'filtersqp',
    'hessian_model': 'LBFGS'
    }
opt_filtersqp_exact = {
    'preset': 'filtersqp',
    'hessian_model': 'exact'
    }

Experiments = [
                (opt_ipopt_exact, "UNO (ipopt preset, exact Hessian)"),
                (opt_filtersqp_exact, "UNO (filtersqp preset, exact Hessian)")
                ]

plot_folder = cD / Path("out_UNO_experiments")

#Choose perturbed start points to test for,
#modify discretized initial controls u_k in turn for nPert0 <= k < nPertF
nPert0 = 0
nPertF = 10

#Write results to a file?
file_output = True



def create_UNO_model(OCprob:OCProblems.OCProblem, start_pert = None):
    model = unopy.Model(unopy.PROBLEM_NONLINEAR,
                        OCprob.nVar,
                        unopy.ZERO_BASED_INDEXING
                        )
    model.set_variables_lower_bounds(OCprob.lb_var)
    model.set_variables_upper_bounds(OCprob.ub_var)
    
    model.set_objective(unopy.MINIMIZE, OCprob.f, OCprob.grad_f_inplace)
    model.set_constraints(OCprob.nCon, OCprob.g_inplace, OCprob.lb_con, OCprob.ub_con, OCprob.jac_g_nnz, np.array(OCprob.jac_g_row), OCprob.jac_g_col, OCprob.jac_g_nz_inplace)
    
    model.set_lagrangian_sign_convention(unopy.MULTIPLIER_NEGATIVE)
    model.set_lagrangian_hessian(OCprob.hess_LT_nnz, unopy.LOWER_TRIANGLE, OCprob.hess_LT_row, OCprob.hess_LT_col, OCprob.hess_lag_objmult_inplace)
    
    if start_pert is not None:
        model.set_initial_primal_iterate(OCprob.perturbed_start_point(start_pert))
    else:
        model.set_initial_primal_iterate(OCprob.start_point)
    return model

def set_UNO_options(UNOsol, **kwargs):
    for key in kwargs:
        if key == 'preset':
            continue
        UNOsol.set_option(key, kwargs[key])
    if 'preset' in kwargs:
        UNOsol.set_preset(kwargs['preset'])
    
def UNOsolver_perturbed_starts(OCprob : OCProblems.OCProblem, arg_opts : dict, nPert0, nPertF, itMax = 200):
    UNOmodel = create_UNO_model(OCprob)
    
    N_SQP = []
    N_secs = []
    type_sol = []
    for j in range(nPert0, nPertF):
        UNOmodel.set_initial_primal_iterate(OCprob.perturbed_start_point(j))
        UNOsol = unopy.UnoSolver()
        set_UNO_options(UNOsol, **arg_opts, max_iterations = itMax)
        
        t0 = time.monotonic()
        result = UNOsol.optimize(UNOmodel)
        t1 = time.monotonic()
        
        N_SQP.append(result.number_iterations)
        type_sol.append(int(result.optimization_status == unopy.OptimizationStatus.SUCCESS))
        N_secs.append(t1 - t0)
    return N_SQP, N_secs, type_sol


#Run all example problems for all option sets for perturbed start points
dirPath = plot_folder
dirPath.mkdir(parents = True, exist_ok = True)
if file_output:
    date_app = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_").replace("'", "")
    pref = "UNO"
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
        ret_N_SQP, ret_N_secs, ret_type_sol = UNOsolver_perturbed_starts(OCprob, EXP_opts, nPert0, nPertF, itMax = itMax)
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
        suptitle = OCname, dirPath = dirPath, savePrefix = "UNO")
    OCP_experiment.print_iterations(out, OCname, EXP_N_SQP, EXP_N_secs, EXP_type_sol, namejust = namejust)
out.close()


