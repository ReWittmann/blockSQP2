# blockSQP2 -- A structure-exploiting nonlinear programming solver based
#              on blockSQP by Dennis Janka.
# Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>

# Licensed under the zlib license. See LICENSE for more details.


# \file run_blockSQP_experiments.py
# \author Reinhold Wittmann
# \date 2025
#
# Script to benchmark blockSQP2 on several problems 
# for perturbed start points for different options

import datetime
import sys
from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()
sys.path += [str(cD.parent), str(cD.parents[1]/Path("Python"))]

import blockSQP2
import OCP_experiment
import OCProblems


Examples = [
            OCProblems.Batch_Reactor,
            OCProblems.Cart_Pendulum,
            OCProblems.Catalyst_Mixing,
            OCProblems.Cushioned_Oscillation,
            OCProblems.Ducted_Fan,
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

#SR1_BFGS
opt_SR1_BFGS = blockSQP2.SQPoptions()
opt_SR1_BFGS.max_conv_QPs = 1
opt_SR1_BFGS.max_filter_overrides = 0
opt_SR1_BFGS.BFGS_damping_factor = 0.2

#Convexification strategy 0
opt_CS0 = blockSQP2.SQPoptions()
opt_CS0.max_conv_QPs = 4
opt_CS0.conv_strategy = 0
opt_CS0.max_filter_overrides = 0

#Convexification strategy 1
opt_CS1 = blockSQP2.SQPoptions()
opt_CS1.max_conv_QPs = 4
opt_CS1.conv_strategy = 1
opt_CS1.max_filter_overrides = 0

#Convexification strategy 2
opt_CS2 = blockSQP2.SQPoptions(
    max_conv_QPs = 4,
    conv_strategy = 2,
    max_filter_overrides = 0,
    automatic_scaling = True
)

opt_CS2_new = blockSQP2.SQPoptions(
    max_conv_QPs = 4,
    conv_strategy = 2,
    max_filter_overrides = 0,
    automatic_scaling = True,
    test_opt_1 = True
    )

opt_CS2_noScaling = blockSQP2.SQPoptions(
    max_conv_QPs = 4,
    conv_strategy = 2,
    max_filter_overrides = 0,
    automatic_scaling = False,
    test_opt_1 = True
    )

opt_CS2_newScaling = blockSQP2.SQPoptions(
    max_conv_QPs = 4,
    conv_strategy = 2,
    max_filter_overrides = 0,
    automatic_scaling = True,
    test_opt_1 = True,
    test_opt_2 = True
    )


opt_CS2_S1 = blockSQP2.SQPoptions(
    max_conv_QPs = 4,
    conv_strategy = 2,
    max_filter_overrides = 0,
    automatic_scaling = True,
    test_opt_1 = True,
    test_opt_2 = True,
    )
opt_CS2_S2 = blockSQP2.SQPoptions(
    max_conv_QPs = 4,
    conv_strategy = 2,
    max_filter_overrides = 0,
    automatic_scaling = True,
    test_opt_1 = True,
    test_opt_2 = True,
    scaling_Theta_min = 0.1,
    scaling_Theta_max = 5.0
    )
opt_CS2_S3 = blockSQP2.SQPoptions(
    max_conv_QPs = 4,
    conv_strategy = 2,
    max_filter_overrides = 0,
    automatic_scaling = True,
    test_opt_1 = True,
    test_opt_2 = True,
    scaling_Theta_min = 0.1,
    scaling_Theta_max = 2.0
    )
opt_CS2_S4 = blockSQP2.SQPoptions(
    max_conv_QPs = 4,
    conv_strategy = 2,
    max_filter_overrides = 0,
    automatic_scaling = True,
    test_opt_1 = True,
    test_opt_2 = True,
    scaling_Theta_min = 0.05,
    scaling_Theta_max = 10.0
    )

opt_CS2_S5 = blockSQP2.SQPoptions(
    max_conv_QPs = 4,
    conv_strategy = 2,
    max_filter_overrides = 0,
    automatic_scaling = True,
    test_opt_1 = True,
    test_opt_2 = True,
    scaling_Theta_min = 0.2,
    scaling_Theta_max = 10.0
    )

opt_CS2_S6 = blockSQP2.SQPoptions(
    max_conv_QPs = 4,
    conv_strategy = 2,
    max_filter_overrides = 0,
    automatic_scaling = True,
    test_opt_1 = True,
    test_opt_2 = True,
    scaling_Theta_min = 0.2,
    scaling_Theta_max = 5.0
    )

#Full structure exploitation
opt_full = blockSQP2.SQPoptions()
opt_full.max_conv_QPs = 4
opt_full.conv_strategy = 2
opt_full.automatic_scaling = True
opt_full.par_QPs = True


opt_CS2_par = blockSQP2.SQPoptions(
    max_conv_QPs = 4,
    conv_strategy = 2,
    par_QPs = True,
    max_filter_overrides = 0, 
    automatic_scaling = True, 
    test_opt_1 = False,
    test_opt_2 = True,
    scaling_Theta_min = 0.1,
    scaling_Theta_max = 5.0
    )

opt_CS2_par_new = blockSQP2.SQPoptions(
    max_conv_QPs = 4,
    conv_strategy = 2,
    par_QPs = True,
    max_filter_overrides = 0, 
    automatic_scaling = True, 
    test_opt_1 = True,
    test_opt_2 = True,
    scaling_Theta_min = 0.1,
    scaling_Theta_max = 5.0
    )



#Select option sets to test for
Experiments = [
               # (opt_SR1_BFGS, "SR1-BFGS"),
               # (opt_CS0, "Convexification strategy 0"),
               # (opt_CS1, "conv. str. 1"),
                # (opt_CS2, "conv. str. 2"),
                # (opt_CS2_noScaling, "conv. str. 2 noScaling"),
                # (opt_CS2_new, "conv. str. 2 scaling"),
                # (opt_CS2_newScaling, "conv. str. 2. newScaling"),
               # (opt_full, "opt_full_NTP"),
               
               # (opt_CS2_S1, "scaling_0p1_10"),
               # (opt_CS2_S2, "scaling_0p1_5"),
               # (opt_CS2_S3, "scaling_0p1_2"),
               # (opt_CS2_S4, "scaling_0p05_10"),
               # (opt_CS2_S5, "scaling_0p2_10"),
               # (opt_CS2_S6, "scaling_0p2_5"),
               
               (opt_CS2_par, "par"),
               (opt_CS2_par_new, "par_new")
               ]


plot_folder = cD / Path("out_blockSQP2_experiments")

#Choose perturbed start points to test for,
#modify discretized initial controls u_k in turn for nPert0 <= k < nPertF
nPert0 = 0
nPertF = 40

#Write results to a file?
file_output = True


#Run all example problems for all option sets for perturbed start points
dirPath = plot_folder
dirPath.mkdir(parents = True, exist_ok = True)
if file_output:
    date_app = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_").replace("'", "")
    pref = "blockSQP2"
    filePath = dirPath / Path(pref + "_it_" + date_app + ".txt")
    out = open(filePath, 'w')
else:
    out = OCP_experiment.out_dummy()

titles = [EXP_name for _, EXP_name in Experiments]
OCP_experiment.print_heading(out, titles)
for OCclass in Examples:        
    OCprob = OCclass(nt = 100, integrator = 'RK4', parallel = True)
    itMax = 200
    titles = []
    EXP_N_SQP = []
    EXP_N_secs = []
    EXP_type_sol = []
    n_EXP = 0
    for EXP_opts, EXP_name in Experiments:
        ret_N_SQP, ret_N_secs, ret_type_sol = OCP_experiment.perturbed_starts(OCprob, EXP_opts, nPert0, nPertF, itMax = itMax)
        EXP_N_SQP.append(ret_N_SQP)
        EXP_N_secs.append(ret_N_secs)
        EXP_type_sol.append(ret_type_sol)
        titles.append(EXP_name)
        n_EXP += 1
    ###############################################################################
    OCP_experiment.plot_successful(n_EXP, nPert0, nPertF,\
        titles, EXP_N_SQP, EXP_N_secs, EXP_type_sol,\
        suptitle = OCclass.__name__, dirPath = dirPath, savePrefix = "blockSQP2")
    OCP_experiment.print_iterations(out, OCclass.__name__, EXP_N_SQP, EXP_N_secs, EXP_type_sol)
out.close()
