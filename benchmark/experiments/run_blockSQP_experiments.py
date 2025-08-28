import os
import sys

try:
    cD = os.path.dirname(os.path.abspath(__file__))
except:
    cD = os.getcwd()
sys.path += [cD + "/..", cD + "/../.."]

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
            OCProblems.Catalyst_Mixing,
            # OCProblems.Cushioned_Oscillation,
            OCProblems.Egerstedt_Standard,
            OCProblems.Electric_Car,
            OCProblems.Goddard_Rocket,
            OCProblems.Hanging_Chain,
            OCProblems.Lotka_Volterra_Fishing,
            OCProblems.Particle_Steering,
            # OCProblems.Three_Tank_Multimode,
            # OCProblems.Lotka_OED,
            ]

opt1 = opt_conv_str_2_par_scale(max_conv_QPs = 6)
opt1.enable_premature_termination = True
opt1.kappaF = 0.8
opt1.kappaSOC = 0.99
opt1.gammaTheta = 1e-2
opt1.eta = 1e-2

# opt2 = opt_conv_str_2_par(max_conv_QPs = 6)
# opt2.automatic_scaling = False

# opt3 = opt_conv_str_2_par(max_conv_QPs = 6)
# opt3.automatic_scaling = True
# opt3.test_opt_1 = False
# opt3.test_opt_2 = 1.5
# opt3.test_opt_3 = 4.0


Experiments = [
                (opt1, "conv2"),
                # (opt2, "conv2_no_scaling"),
                # (opt3, "conv2_scaling_new"),
                # (opt4, "conv2_scale mod2"),
                # (opt5, "conv2_scale mod3"),
                # (opt6, "conv2_scale mod4"),
                # (opt7, "conv2_scale mod5"),
                # (opt8, "conv2_scale mod6"),
                # (opt9, "conv2_scale mod7"),
                # (opt10, "conv2_scale mod8"),
                # (opt11, "conv2_scale mod9"),
                # (opt12, "conv2_scale mod10"),
               # (opt5, "default conv2_scale"),
               # (opt6, "conv2_scale, QPterm mod")
               ]


plot_folder = "/home/reinhold/PLOT/TEST_term_RK4_"
OCP_experiment.run_blockSQP_experiments(Examples, Experiments,\
                                        plot_folder,\
                                        nPert0 = 0, nPertF = 20,
                                        nt = 100,
                                        integrator = 'cvodes',
                                        parallel = True
                                        )

