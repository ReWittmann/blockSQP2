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

import datetime
import sys
from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()
sys.path += [str(cD.parent)]
import OCP_experiment
import OCProblems
import OCProblems_fatrop

#Fatrop as of writing this does not support formulations involving parameters or quadratures-in-constraints, so use reformulated versions
Examples = [
            # (OCProblems_fatrop.Apollo_Reentry_noParams, dict(), "Apollo_Reentry"),
            # (OCProblems_fatrop.Batch_Distillation_noParams, dict(), "Batch_Distillation"),
            # (OCProblems.Batch_Reactor, dict(), None),
            # (OCProblems_fatrop.Batch_Reactor_OED_noQuads, dict(), "Batch_Reactor_OED"),
            # (OCProblems_fatrop.Calcium_Oscillation_noParams, dict(), "Calcium_Oscillation"),
            # (OCProblems.Cart_Pendulum, dict(), None),
            # (OCProblems.Cart_Pendulum, OCProblems.Cart_Pendulum.param_set_2, "Cart_Pendulum_2"),
            # (OCProblems.Catalyst_Mixing, dict(), None),
            # (OCProblems_fatrop.Catalyst_Mixing_OED_noQuads, dict(), "Catalyst_Mixing_OED"),
            # (OCProblems_fatrop.Cushioned_Oscillation_noParams, dict(), "Cushioned_Oscillation"),
            # (OCProblems_fatrop.Dielectrophoretic_Particle_noParams, dict(), "Dielectrophoretic_Particle"),
            # (OCProblems_fatrop.D_Onofrio_Chemotherapy_noQuads, dict(), "D_Onofrio_Chemotherapy"),
            # (OCProblems_fatrop.D_Onofrio_Chemotherapy_noQuads, OCProblems.D_Onofrio_Chemotherapy.param_set_2, "D_Onofrio_Chemotherapy_2"),
            # (OCProblems_fatrop.D_Onofrio_Chemotherapy_noQuads, OCProblems.D_Onofrio_Chemotherapy.param_set_3, "D_Onofrio_Chemotherapy_3"),
            # (OCProblems_fatrop.D_Onofrio_Chemotherapy_noQuads, OCProblems.D_Onofrio_Chemotherapy.param_set_4, "D_Onofrio_Chemotherapy_4"),
            # (OCProblems_fatrop.Ducted_Fan_noParams, dict(), "Ducted_Fan"),
            # (OCProblems.Egerstedt_Standard, dict(), None),
            # (OCProblems.Electric_Car, dict(), None),
            # (OCProblems_fatrop.Fermenter_noQuads, dict(), "Fermenter"),
            # (OCProblems_fatrop.Goddard_Rocket_noParams, dict(), 'Goddard\'s Rocket'),
            # (OCProblems_fatrop.Hang_Glider_noParams, dict(), "Hang_Glider"),
            # (OCProblems_fatrop.Hanging_Chain_noQuads, dict(), "Hanging_Chain"),
            # (OCProblems.Lotka_Volterra_Fishing, dict(), None),
            # (OCProblems.Lotka_Volterra_Fishing, OCProblems.Lotka_Volterra_Fishing.param_set_2, "Lotka_Volterra_Fishing_2"),
            # (OCProblems.Lotka_Volterra_Fishing, OCProblems.Lotka_Volterra_Fishing.param_set_3, "Lotka_Volterra_Fishing_3"),
            # (OCProblems_fatrop.Lotka_OED_noQuads, dict(), "Lotka_OED"),
            # (OCProblems.Lotka_Volterra_Competitive, dict(), None),
            # (OCProblems.Lotka_Volterra_Competitive, OCProblems.Lotka_Volterra_Competitive.param_set_2, "Lotka_Volterra_Competitive_2"),
            # (OCProblems.Lotka_Volterra_Shared, dict(), None),
            # (OCProblems.Lotka_Volterra_Shared, OCProblems.Lotka_Volterra_Shared.param_set_2, "Lotka_Volterra_Shared_2"),
            # (OCProblems_fatrop.Lotka_Shared_OED_noQuads, dict(), "Lotka_Shared_OED"),
            # (OCProblems.Ocean, dict(), None),
            # (OCProblems_fatrop.Particle_Steering_noParams, dict(), "Particle_Steering"),
            # (OCProblems.Quadrotor_Helicopter, dict(), None),
            # (OCProblems.Satellite_Deorbiting, dict(), None),
            # (OCProblems.Three_Tank_Multimode, dict(), None),
            # (OCProblems_fatrop.Time_Optimal_Car_noParams, dict(), "Time_Optimal_Car"),
            # (OCProblems.Tubular_Reactor, dict(), None),
            ]



plot_folder = cD / Path("out_fatrop_experiments")

#Choose perturbed start points to test for,
#modify discretized initial controls u_k in turn for nPert0 <= k < nPertF
nPert0 = 0
nPertF = 40

#Write results to a file?
file_output = True


Experiments = [
                ({'fatrop': {'tol': 1e-6, 'constr_viol_tol':1e-4, 'max_iter': 300}, 
                  'jit': False, 'convexify_strategy': None}, #Doesnt seem to work with convexification strategies other than "None"
                 'Fatrop (exact Hessian, tol=1e-6)'),
                ]

#Run all example problems for all option sets for perturbed start points
dirPath = plot_folder
dirPath.mkdir(parents = True, exist_ok = True)
if file_output:
    date_app = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_").replace("'", "")
    pref = "fatrop"
    filePath = dirPath / Path(pref + "_it_" + date_app + ".txt")
    out = open(filePath, 'w')
else:
    out = OCP_experiment.out_dummy()

titles = [EXP_name for _, EXP_name in Experiments]
OCP_experiment.print_heading(out, titles)
for OCclass, OCargs, OCname in Examples:        
    OCprob = OCclass(nt = 100, parallel = True, **OCargs)
    itMax = 300
    titles = []
    EXP_N_SQP = []
    EXP_N_secs = []
    EXP_type_sol = []
    n_EXP = 0
    
    for EXP_opts, EXP_name in Experiments:
        ret_N_SQP, ret_N_secs, ret_type_sol = OCP_experiment.fatrop_perturbed_starts(OCprob, *OCProblems_fatrop.get_constr_data(OCprob), EXP_opts, nPert0, nPertF, itMax = itMax)
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
        suptitle = OCname, dirPath = dirPath, savePrefix = "fatrop")
    OCP_experiment.print_iterations(out, OCname, EXP_N_SQP, EXP_N_secs, EXP_type_sol)
out.close()



























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
