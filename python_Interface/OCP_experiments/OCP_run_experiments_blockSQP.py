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
            OCProblems.Batch_Reactor,
            OCProblems.Catalyst_Mixing,
            OCProblems.Cushioned_Oscillation,
            OCProblems.Egerstedt_Standard,
            OCProblems.Electric_Car,
            OCProblems.Goddard_Rocket,
            OCProblems.Hanging_Chain,
            OCProblems.Lotka_Volterra_Fishing,
            OCProblems.Particle_Steering,
            OCProblems.Three_Tank_Multimode,
            # OCProblems.Lotka_OED,
            ]

# opt2 = opt_conv_str_2_par(max_conv_QPs = 6)
# opt2.automatic_scaling = True

# opt3 = opt_conv_str_2_par(max_conv_QPs = 6)
# opt3.automatic_scaling = True
# opt3.test_opt_1 = True
# opt3.test_opt_2 = 1.5
# opt3.test_opt_3 = 4.0

# opt4 = opt_conv_str_2_par(max_conv_QPs = 6)
# opt4.automatic_scaling = True
# opt4.test_opt_1 = True
# opt4.test_opt_2 = 1.5
# opt4.test_opt_3 = 8.0

# opt5 = opt_conv_str_2_par(max_conv_QPs = 6)
# opt5.automatic_scaling = True
# opt5.test_opt_1 = True
# opt5.test_opt_2 = 2.0
# opt5.test_opt_3 = 4.0

opt2 = opt_conv_str_2_par(max_conv_QPs = 6)
opt2.automatic_scaling = False

opt3 = opt_conv_str_2_par(max_conv_QPs = 6)
opt3.automatic_scaling = True
opt3.test_opt_1 = False
# opt3.test_opt_2 = 1.5
# opt3.test_opt_3 = 4.0

# opt4 = opt_conv_str_2_par(max_conv_QPs = 6)
# opt4.automatic_scaling = True
# opt4.test_opt_1 = True
# opt4.test_opt_2 = 1.0
# opt4.test_opt_3 = 4.0

# opt5 = opt_conv_str_2_par(max_conv_QPs = 6)
# opt5.automatic_scaling = True
# opt5.test_opt_1 = True
# opt5.test_opt_2 = 1.5
# opt5.test_opt_3 = 4.0


# opt6 = opt_conv_str_2_par(max_conv_QPs = 6)
# opt6.automatic_scaling = True
# opt6.test_opt_1 = True
# opt6.test_opt_2 = 0.5
# opt6.test_opt_3 = 2.0

# opt7 = opt_conv_str_2_par(max_conv_QPs = 6)
# opt7.automatic_scaling = True
# opt7.test_opt_1 = True
# opt7.test_opt_2 = 1.0
# opt7.test_opt_3 = 8.0


# opt8 = opt_conv_str_2_par(max_conv_QPs = 6)
# opt8.automatic_scaling = True
# opt8.test_opt_1 = True
# opt8.test_opt_2 = 1.0
# opt8.test_opt_3 = 5.0

# opt9 = opt_conv_str_2_par(max_conv_QPs = 6)
# opt9.automatic_scaling = True
# opt9.test_opt_1 = True
# opt9.test_opt_2 = 1.0
# opt9.test_opt_3 = 10.0###

# opt10 = opt_conv_str_2_par(max_conv_QPs = 6)
# opt10.automatic_scaling = True
# opt10.test_opt_1 = True
# opt10.test_opt_2 = 0.8
# opt10.test_opt_3 = 6.0

# opt11 = opt_conv_str_2_par(max_conv_QPs = 6)
# opt11.automatic_scaling = True
# opt11.test_opt_1 = True
# opt11.test_opt_2 = 5.0
# opt11.test_opt_3 = 10.0

# opt12 = opt_conv_str_2_par(max_conv_QPs = 6)
# opt12.automatic_scaling = True
# opt12.test_opt_1 = True
# opt12.test_opt_2 = 10.0
# opt12.test_opt_3 = 20.0

# opt4 = opt_conv_str_1_seq(max_conv_QPs = 4)
# opt4.indef_delay = 1
# opt5 = opt_conv_str_1_seq(max_conv_QPs = 4)
# opt5.indef_delay = 1
# opt5.automatic_scaling = True


# opt5 = opt_conv_str_2_par(max_conv_QPs = 6)
# opt5.automatic_scaling = True
# opt6 = opt_conv_str_2_par(max_conv_QPs = 6)
# opt6.automatic_scaling = True
# opt6.test_opt_4 = 1./3.

Experiments = [
                (opt2, "conv2_no_scaling"),
                (opt3, "conv2_scaling_new"),
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


plot_folder = "/home/reinhold/PLOT/TEST_SCALE_9"
OCP_experiment.run_blockSQP_experiments(Examples, Experiments,\
                                        plot_folder,\
                                        nPert0 = 0, nPertF = 40
                                        )

