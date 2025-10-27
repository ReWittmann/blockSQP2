import sys
from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()
sys.path += [str(cD.parents[1]), str(cD.parents[2])]

import py_blockSQP
import OCP_experiment
import OCProblems

Examples = [
            OCProblems.Batch_Reactor,
            OCProblems.Cart_Pendulum,
            OCProblems.Catalyst_Mixing,
            OCProblems.Cushioned_Oscillation,
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

opt_CS2 = py_blockSQP.SQPoptions()
opt_CS2.max_conv_QPs = 4
opt_CS2.conv_strategy = 2
opt_CS2.par_QPs = True
opt_CS2.enable_QP_cancellation = True
opt_CS2.automatic_scaling = False

opt_CS2_scale = py_blockSQP.SQPoptions()
opt_CS2_scale.max_conv_QPs = 4
opt_CS2_scale.conv_strategy = 2
opt_CS2_scale.par_QPs = True
opt_CS2_scale.enable_QP_cancellation = True
opt_CS2_scale.automatic_scaling = True

Experiments = [
               (opt_CS2, "Conv. str. 2, no scaling"),
               (opt_CS2_scale, "Conv. str. 2, automatic scaling")
               ]

plot_folder = cD / Path("out_scaling_comparison")


OCP_experiment.run_blockSQP_experiments(Examples, Experiments,\
                                        plot_folder,\
                                        nPert0 = 0, nPertF = 40,
                                        integrator = 'RK4'
                                        )

