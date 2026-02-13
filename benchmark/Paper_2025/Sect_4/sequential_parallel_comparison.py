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

#Note: On Linux, ensure the scaling governor is set to "performance"
#      and that the process actually uses multiple cores - running this
#      script from Spyder resulted in only one CPU core being used.
#      Running from the command line is recommended.

# The scaling governor can be set via
#   sudo cpupower frequency-set -g performance
# and checked via
#   cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

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

opt_CS2_seq = blockSQP2.SQPoptions()
opt_CS2_seq.max_conv_QPs = 4
opt_CS2_seq.conv_strategy = 2
opt_CS2_seq.max_filter_overrides = 0

opt_CS2_par = blockSQP2.SQPoptions()
opt_CS2_par.max_conv_QPs = 4
opt_CS2_par.conv_strategy = 2
opt_CS2_par.par_QPs = True
opt_CS2_par.enable_QP_cancellation = True
opt_CS2_par.max_filter_overrides = 0

Experiments = [
               (opt_CS2_seq, "Conv. str. 2, sequential"),
               (opt_CS2_par, "Conv. str. 2, parallel")
               ]

plot_folder = cD / Path("out_sequential_parallel_comparison")


OCP_experiment.run_blockSQP_experiments(Examples, Experiments,\
                                        plot_folder,\
                                        nPert0 = 0, nPertF = 40
                                        )

