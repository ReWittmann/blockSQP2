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
            # OCProblems.Goddard_Rocket,
            # OCProblems.Catalyst_Mixing,
            # OCProblems.Lotka_Volterra_Fishing,
            # OCProblems.Hanging_Chain,
            # OCProblems.Cushioned_Oscillation,
            # OCProblems.Egerstedt_Standard,
            # OCProblems.Electric_Car,
            # OCProblems.Particle_Steering,
            # OCProblems.Three_Tank_Multimode,
            # OCProblems.Lotka_OED
            ]

opt_CS2 = py_blockSQP.SQPoptions()
opt_CS2.max_conv_QPs = 6
opt_CS2.conv_strategy = 2
opt_CS2.par_QPs = True
opt_CS2.enable_QP_cancellation = True
opt_CS2.automatic_scaling = False

opt_CS2_scale = py_blockSQP.SQPoptions()
opt_CS2_scale.max_conv_QPs = 6
opt_CS2_scale.conv_strategy = 2
opt_CS2_scale.par_QPs = True
opt_CS2_scale.enable_QP_cancellation = True
opt_CS2_scale.automatic_scaling = True


Experiments = [
               (opt_CS2, "Conv. str. 2, no scaling"),
               (opt_CS2_scale, "Conv. str. 2, automatic scaling")
               ]

plot_folder = cD + "/out_scaling_comparison"


OCP_experiment.run_blockSQP_experiments(Examples, Experiments,\
                                        plot_folder,\
                                        nPert0 = 0, nPertF = 40
                                        )

