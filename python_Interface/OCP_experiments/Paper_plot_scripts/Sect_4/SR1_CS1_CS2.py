import os
import sys

try:
    cD = os.path.dirname(os.path.abspath(__file__))
except:
    cD = os.getcwd()
sys.path += [cD + "/../../..", cD + "/../..", cD + "/../../../examples"]

import py_blockSQP
import OCP_experiment
import OCProblems



Examples = [
            OCProblems.Batch_Reactor,
            OCProblems.Goddard_Rocket,
            OCProblems.Catalyst_Mixing,
            OCProblems.Lotka_Volterra_Fishing,
            OCProblems.Hanging_Chain,
            OCProblems.Cushioned_Oscillation,
            OCProblems.Egerstedt_Standard,
            OCProblems.Electric_Car,
            OCProblems.Particle_Steering,
            OCProblems.Three_Tank_Multimode,
            OCProblems.Lotka_OED
            ]


#SR1_BFGS
opt_SR1_BFGS = py_blockSQP.SQPoptions()
opt_SR1_BFGS.max_conv_QPs = 1
opt_SR1_BFGS.max_filter_overrides = 0

#Convexification strategy 1
opt_CS1 = py_blockSQP.SQPoptions()
opt_CS1.max_conv_QPs = 4
opt_CS1.conv_strategy = 1
opt_CS1.max_filter_overrides = 0

#Convexification strategy 2
opt_CS2 = py_blockSQP.SQPoptions()
opt_CS2.max_conv_QPs = 4
opt_CS2.conv_strategy = 2
opt_CS2.max_filter_overrides = 0

Experiments = [
               (opt_SR1_BFGS, "SR1-BFGS"),
               (opt_CS1, "Conv str. 1"),
               (opt_CS2, "Conv. str. 2")
               ]

plot_folder = cD + "/out_SR1_CS2_CS2"

OCP_experiment.run_blockSQP_experiments(Examples, Experiments,\
                                        plot_folder,\
                                        nPert0 = 0, nPertF = 40
                                        )

