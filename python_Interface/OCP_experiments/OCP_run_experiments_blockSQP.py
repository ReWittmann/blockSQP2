import os
import sys

try:
    cD = os.path.dirname(os.path.abspath(__file__))
except:
    cD = os.getcwd()
sys.path += [cD + "/..", cD + "/../examples"]

import py_blockSQP
import OCP_experiment
import OCProblems


def opt_SR1_BFGS_seq():
    opts = py_blockSQP.SQPoptions()
    opts.max_conv_QPs = 1
    opts.par_QPs = False
    opts.automatic_scaling = False
    opts.max_filter_overrides = 0
    opts.conv_kappa_max = 8.0
    return opts


def opt_conv_str_1_seq(max_conv_QPs = 4):
    opts = py_blockSQP.SQPoptions()
    opts.max_conv_QPs = max_conv_QPs
    opts.conv_strategy = 1
    opts.par_QPs = False
    opts.automatic_scaling = False
    opts.max_filter_overrides = 0
    opts.conv_kappa_max = 8.0
    return opts


def opt_conv_str_2_seq(max_conv_QPs = 4):
    opts = py_blockSQP.SQPoptions()
    opts.max_conv_QPs = max_conv_QPs
    opts.conv_strategy = 2
    opts.automatic_scaling = False
    opts.max_filter_overrides = 0
    opts.conv_kappa_max = 8.0
    return opts


def opt_SR1_BFGS_par():
    opts = py_blockSQP.SQPoptions()
    opts.max_conv_QPs = 1
    opts.par_QPs = True
    opts.automatic_scaling = False
    opts.max_filter_overrides = 0
    opts.conv_kappa_max = 8.0
    return opts


def opt_conv_str_1_par(max_conv_QPs = 4):
    opts = py_blockSQP.SQPoptions()
    opts.max_conv_QPs = 4
    opts.conv_strategy = 1
    opts.par_QPs = True
    opts.automatic_scaling = False
    opts.max_filter_overrides = 0
    opts.conv_kappa_max = 8.0
    return opts


def opt_conv_str_2_par(max_conv_QPs = 4):
    opts = py_blockSQP.SQPoptions()
    opts.max_conv_QPs = max_conv_QPs
    opts.conv_strategy = 2
    opts.par_QPs = True
    opts.automatic_scaling = False
    opts.max_filter_overrides = 0
    opts.conv_kappa_max = 8.0
    opts.indef_delay = 3
    return opts


def opt_conv_str_2_par_scale(max_conv_QPs = 4):
    opts = py_blockSQP.SQPoptions()
    opts.max_conv_QPs = max_conv_QPs
    opts.conv_strategy = 2
    opts.par_QPs = True
    opts.automatic_scaling = True
    opts.max_filter_overrides = 0
    opts.conv_kappa_max = 8.0
    return opts


Examples = [
            # OCProblems.Batch_Reactor,
            # OCProblems.Goddard_Rocket,
            # OCProblems.Catalyst_Mixing,
            # OCProblems.Lotka_Volterra_Fishing,
            OCProblems.Hanging_Chain,
            # OCProblems.Cushioned_Oscillation,
            # OCProblems.Egerstedt_Standard,
            # OCProblems.Electric_Car,
            # OCProblems.Particle_Steering,
            # OCProblems.Three_Tank_Multimode,
            # OCProblems.Lotka_OED
            ]

opt2 = opt_conv_str_2_par(max_conv_QPs = 6)
opt2.automatic_scaling = False

opt3 = opt_conv_str_2_par(max_conv_QPs = 6)
opt3.automatic_scaling = True


opt4 = opt_conv_str_1_seq(max_conv_QPs = 4)
opt4.indef_delay = 1
opt5 = opt_conv_str_1_seq(max_conv_QPs = 4)
opt5.indef_delay = 1
opt5.automatic_scaling = True

Experiments = [
               (opt4, "Conv str. 1 seq"),
               (opt5, "Conv. str. 1 seq scale")
               ]


plot_folder = "/home/reinhold/PLOT/blockSQP_TEST_NEW_/"
OCP_experiment.run_blockSQP_experiments(Examples, Experiments,\
                                        plot_folder,\
                                        nPert0 = 0, nPertF = 40
                                        )

