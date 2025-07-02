import os
import sys
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
except:
    sys.path.append(os.getcwd() + "/..")
import py_blockSQP
import OCP_experiment
import OCProblems

plot_folder = "/home/reinhold/PLOT"


Examples = [(OCProblems.Batch_Reactor, "Batch reactor"),
            (OCProblems.Goddard_Rocket, "Goddard's rocket"),
            (OCProblems.Catalyst_Mixing, "Catalyst mixing"),
            (OCProblems.Lotka_Volterra_Fishing, "Lotka Volterra fishing"),
            (OCProblems.Hanging_Chain, "Hanging chain"),
            (OCProblems.Cushioned_Oscillation, "Cushioned oscillation"),
            (OCProblems.Egerstedt_Standard, "Egerstedt standard"),
            (OCProblems.Electric_Car, "Electric car"),
            (OCProblems.Particle_Steering, "Particle steering"),
            (OCProblems.Three_Tank_Multimode, "Three tank multimode"),
            (OCProblems.Lotka_OED, "Lotka_OED")
            ]
QPopts = py_blockSQP.qpOASES_options()
QPopts.terminationTolerance = 1e-10

EXP_1_opts = py_blockSQP.SQPoptions()
EXP_1_opts.max_conv_QPs = 1
EXP_1_opts.qpsol_options = QPopts
EXP_1_opts.max_filter_overrides = 0
EXP_1_opts.conv_kappa_max = 8.0

EXP_2_opts = py_blockSQP.SQPoptions()
EXP_2_opts.conv_strategy = 1
EXP_2_opts.max_conv_QPs = 4
EXP_2_opts.qpsol_options = QPopts
EXP_2_opts.max_filter_overrides = 0
EXP_2_opts.conv_kappa_max = 8.0

EXP_3_opts = py_blockSQP.SQPoptions()
EXP_3_opts.conv_strategy = 2
EXP_3_opts.max_conv_QPs = 4
EXP_3_opts.qpsol_options = QPopts
EXP_3_opts.max_filter_overrides = 0
EXP_3_opts.conv_kappa_max = 8.0

Experiments = [(EXP_1_opts, "SR1-BFGS"),
               (EXP_2_opts, "Convexification strategy 1"),
               (EXP_3_opts, "Convexification strategy 2")
               ]
# Examples_ = [(OCProblems.Lotka_Volterra_Fishing, "Lotka Volterra fishing"),
#               (OCProblems.Goddard_Rocket, "Goddard's rocket")
#               ]
Examples_ = [(OCProblems.Hanging_Chain, "Hanging chain")]

OCP_experiment.run_blockSQP_experiments(Examples_, Experiments,\
                                        plot_folder,\
                                        nPert0 = 0, nPertF = 40)

