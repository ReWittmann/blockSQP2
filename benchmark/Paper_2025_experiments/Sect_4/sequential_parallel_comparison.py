import os
import sys

try:
    cD = os.path.dirname(os.path.abspath(__file__))
except:
    cD = os.getcwd()
sys.path += [cD + "/../..", cD + "/../../.."]

import py_blockSQP
import OCP_experiment
import OCProblems


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
            OCProblems.Lotka_OED,
            ]
OCProblems.Goddard_Rocket.__name__ = 'Goddard\'s Rocket'

opt_CS2_seq = py_blockSQP.SQPoptions()
opt_CS2_seq.max_conv_QPs = 4
opt_CS2_seq.conv_strategy = 2
opt_CS2_seq.max_filter_overrides = 0

opt_CS2_par = py_blockSQP.SQPoptions()
opt_CS2_par.max_conv_QPs = 4
opt_CS2_par.conv_strategy = 2
opt_CS2_par.par_QPs = True
opt_CS2_par.enable_QP_cancellation = True
opt_CS2_par.max_filter_overrides = 0

Experiments = [
               (opt_CS2_seq, "Conv. str. 2, sequential"),
               (opt_CS2_par, "Conv. str. 2, parallel")
               ]

plot_folder = cD + "/out_sequential_parallel_comparison"


OCP_experiment.run_blockSQP_experiments(Examples, Experiments,\
                                        plot_folder,\
                                        nPert0 = 0, nPertF = 40
                                        )

