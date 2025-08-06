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


Examples = [# (OCProblems.Batch_Reactor, "Batch reactor"),
            # (OCProblems.Goddard_Rocket, "Goddard's rocket"),
            # (OCProblems.Catalyst_Mixing, "Catalyst mixing"),
            # (OCProblems.Lotka_Volterra_Fishing, "Lotka Volterra fishing"),
            # (OCProblems.Hanging_Chain, "Hanging chain"),
            # (OCProblems.Cushioned_Oscillation_TSCALE, "Cushioned oscillation"),
            # (OCProblems.Egerstedt_Standard, "Egerstedt standard"),
            # (OCProblems.Electric_Car, "Electric car"),
            # (OCProblems.Particle_Steering, "Particle steering"),
            # (OCProblems.Three_Tank_Multimode, "Three tank multimode"),
            # (OCProblems.Lotka_OED, "Lotka_OED"),
            (OCProblems.Cart_Pendulum, "Cart pendulum")
            ]

opt2 = opt_conv_str_2_par(max_conv_QPs = 6)
opt2.test_opt_2 = 3

opt3 = opt_conv_str_2_par(max_conv_QPs = 6)
opt3.test_opt_2 = 3
opt3.automatic_scaling = True


Experiments = [(opt_SR1_BFGS_seq(), "SR1-BFGS"),
               # (opt_conv_str_2_seq(max_conv_QPs = 4), "SEQ"),
                # (opt_conv_str_2_par(max_conv_QPs = 6), "PAR6"),
                (opt2, "PAR6_test"),
                (opt3, "PAR6_scale")
               # (opt4, "2SEQ4_TEST")
               ]


plot_folder = "/home/reinhold/PLOT"
OCP_experiment.run_blockSQP_experiments(Examples, Experiments,\
                                        plot_folder,\
                                        nPert0 = 0, nPertF = 8,
                                        nt = 400, lambda_u = 0.05, u_max = 15
                                        )







    

# Experiments = [#(opt_SR1_BFGS_par(), "SR1-BFGS"),
#                #(opt_conv_str_1_par(), "Convexification strategy 1"),
#                (opt_conv_str_2_par(), "Convexification strategy 2")
#                ]
# plot_folder = "/home/reinhold/PLOT/SR1_BFGS_CONV_PAR"
# OCP_experiment.run_blockSQP_experiments(Examples, Experiments,\
#                                         plot_folder,\
#                                         nPert0 = 0, nPertF = 8)