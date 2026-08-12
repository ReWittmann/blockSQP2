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
sys.path += [str(cD.parent)]
import OCP_experiment
import OCProblems


# Specify problem (class), non-default parameters and plot suptitle (None for default)
Examples = [
            (OCProblems.Apollo_Reentry, dict(), None),
            # (OCProblems.Batch_Distillation, dict(), None),
            (OCProblems.Batch_Reactor, dict(), None),
            (OCProblems.Batch_Reactor_OED, dict(), None),
            (OCProblems.Calcium_Oscillation, dict(), None),
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
            (OCProblems.Lotka_Volterra_Fishing, OCProblems.Lotka_Volterra_Fishing.param_set_2, "Lotka_Volterra_Fishing_2"),
            (OCProblems.Lotka_Volterra_Fishing, OCProblems.Lotka_Volterra_Fishing.param_set_3, "Lotka_Volterra_Fishing_3"),
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
                ({'ipopt': {'hessian_approximation': 'limited-memory', 'tol': 1e-6}}, "Ipopt, limited-memory, tol 1e-6"),
                ({'ipopt': {'hessian_approximation': "exact", 'tol': 1e-6}}, "Ipopt, exact Hessian"),
                ]


plot_folder = cD / Path("out_ipopt_experiments")
OCP_experiment.run_ipopt_experiments(Examples, 
                                     Experiments, 
                                     plot_folder, 
                                     nPert0 = 0, 
                                     nPertF = 10,
                                     file_output = True
                                     )