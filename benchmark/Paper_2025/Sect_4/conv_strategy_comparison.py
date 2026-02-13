import sys
from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()
sys.path += [str(cD.parents[1]), str(cD.parents[2]/Path("Python"))]

import blockSQP2
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

#SR1_BFGS
opt_SR1_BFGS = blockSQP2.SQPoptions()
opt_SR1_BFGS.max_conv_QPs = 1
opt_SR1_BFGS.max_filter_overrides = 0

#Convexification strategy 1
opt_CS1 = blockSQP2.SQPoptions()
opt_CS1.max_conv_QPs = 4
opt_CS1.conv_strategy = 1
opt_CS1.max_filter_overrides = 0

#Convexification strategy 2
opt_CS2 = blockSQP2.SQPoptions()
opt_CS2.max_conv_QPs = 4
opt_CS2.conv_strategy = 2
opt_CS2.max_filter_overrides = 0

Experiments = [
               (opt_SR1_BFGS, "SR1-BFGS"),
               (opt_CS1, "Convexification strategy 1"),
               (opt_CS2, "Convexification strategy 2")
               ]

plot_folder = cD / Path("out_conv_strategy_comparison")

OCP_experiment.run_blockSQP2_experiments(Examples, Experiments,\
                                        plot_folder,\
                                        nPert0 = 0, nPertF = 40
                                        )

