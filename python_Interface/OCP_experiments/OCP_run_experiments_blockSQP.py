import os
import sys
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
except:
    sys.path.append(os.getcwd() + "/..")
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


Examples = [OCProblems.Batch_Reactor,
            OCProblems.Goddard_Rocket,
            OCProblems.Catalyst_Mixing,
            OCProblems.Lotka_Volterra_Fishing,
            OCProblems.Hanging_Chain,
            OCProblems.Cushioned_Oscillation,
            OCProblems.Egerstedt_Standard,
            # OCProblems.Electric_Car,
            # OCProblems.Particle_Steering,
            # OCProblems.Three_Tank_Multimode,
            # OCProblems.Lotka_OED,
            # OCProblems.Cart_Pendulum,
            ]

# opt2 = opt_conv_str_2_par(max_conv_QPs = 6)
# opt2.test_opt_2 = 3

# opt3 = opt_conv_str_2_par(max_conv_QPs = 6)
# opt3.test_opt_2 = 3
# opt3.automatic_scaling = True

# opt1 = opt_SR1_BFGS_seq()
opt2 = opt_conv_str_2_par(max_conv_QPs = 6)
opt2.automatic_scaling = True
# opt2.exact_hess = 2
opt3 = opt_conv_str_2_par(max_conv_QPs = 6)
opt3.automatic_scaling = True
# opt3.exact_hess = 2
# opt3.conv_kappa_max = 64
QPopts = py_blockSQP.qpOASES_options()
# QPopts.terminationTolerance = 1e-10
opt3.qpsol_options = QPopts

# opt3 = opt_conv_str_2_seq()
# opt4 = opt_conv_str_2_seq()
# opt4.indef_delay = 3

Experiments = [
                # (opt_SR1_BFGS_seq(), "SR1-BFGS"),
                # (opt_conv_str_2_seq(max_conv_QPs = 4), "SEQ"),
                # (opt_conv_str_2_par(max_conv_QPs = 6), "PAR6"),
                # (opt2, "PAR6_test"),
                # (opt3, "PAR6_scale")
                # (opt4, "2SEQ4_TEST")
                # (opt1, "SR1-BFGS"),
                (opt2, "Conv str. 2 par scale"),
                (opt3, "Conv. str. 2 par scale QPOTem10")
                # (opt3, "Conv. Str. 2 par scale ckmax64"),
                # (opt4, "Conv. Str. 2 delay3")
               ]


plot_folder = "/home/reinhold/PLOT/blockSQP_TEST_TT/"
OCP_experiment.run_blockSQP_experiments(Examples, Experiments,\
                                        plot_folder,\
                                        nPert0 = 0, nPertF = 40
                                        )







    

# Experiments = [#(opt_SR1_BFGS_par(), "SR1-BFGS"),
#                #(opt_conv_str_1_par(), "Convexification strategy 1"),
#                (opt_conv_str_2_par(), "Convexification strategy 2")
#                ]
# plot_folder = "/home/reinhold/PLOT/SR1_BFGS_CONV_PAR"
# OCP_experiment.run_blockSQP_experiments(Examples, Experiments,\
#                                         plot_folder,\
#                                         nPert0 = 0, nPertF = 8)