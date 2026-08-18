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
            (OCProblems.Apollo_Reentry, dict(), None),
            # (OCProblems.Batch_Distillation, dict(), None),
            (OCProblems.Batch_Reactor, dict(), None),
            (OCProblems.Batch_Reactor_OED, dict(), None),
            (OCProblems.Calcium_Oscillation, dict(), None),
            (OCProblems.Cart_Pendulum, dict(), None),
            (OCProblems.Cart_Pendulum, OCProblems.Cart_Pendulum.param_set_2, "Cart_Pendulum_2"),
            (OCProblems.Catalyst_Mixing, dict(), None),
            (OCProblems.Catalyst_Mixing_OED, dict(), None),
            (OCProblems.Cushioned_Oscillation, dict(), None),
            (OCProblems.Dielectrophoretic_Particle, dict(), None),
            (OCProblems.D_Onofrio_Chemotherapy, dict(), "D_Onofrio_Chemotherapy"),
            (OCProblems.D_Onofrio_Chemotherapy, OCProblems.D_Onofrio_Chemotherapy.param_set_2, "D_Onofrio_Chemotherapy_2"),
            (OCProblems.D_Onofrio_Chemotherapy, OCProblems.D_Onofrio_Chemotherapy.param_set_3, "D_Onofrio_Chemotherapy_3"),
            (OCProblems.D_Onofrio_Chemotherapy, OCProblems.D_Onofrio_Chemotherapy.param_set_4, "D_Onofrio_Chemotherapy_4"),
            (OCProblems.Ducted_Fan, dict(), None),
            (OCProblems.Egerstedt_Standard, dict(), None),
            (OCProblems.Electric_Car, dict(), None),
            (OCProblems.Fermenter, dict(), None),
            (OCProblems.Goddard_Rocket, dict(), None),
            (OCProblems.Hang_Glider, dict(), None),
            (OCProblems.Hanging_Chain, dict(), None),
            (OCProblems.Lotka_Volterra_Fishing, dict(), None),
            # # (OCProblems.Lotka_Volterra_Fishing, OCProblems.Lotka_Volterra_Fishing.param_set_2, "Lotka_Volterra_Fishing_2"),
            # # (OCProblems.Lotka_Volterra_Fishing, OCProblems.Lotka_Volterra_Fishing.param_set_3, "Lotka_Volterra_Fishing_3"),
            (OCProblems.Lotka_OED, dict(), None),
            (OCProblems.Lotka_Volterra_Competitive, dict(), None),
            (OCProblems.Lotka_Volterra_Competitive, OCProblems.Lotka_Volterra_Competitive.param_set_2, "Lotka_Volterra_Competitive_2"),
            (OCProblems.Lotka_Volterra_Shared, dict(), None),
            (OCProblems.Lotka_Volterra_Shared, OCProblems.Lotka_Volterra_Shared.param_set_2, "Lotka_Volterra_Shared_2"),
            (OCProblems.Lotka_Shared_OED, dict(), None),
            (OCProblems.Ocean, dict(), None),
            (OCProblems.Particle_Steering, dict(), None),
            (OCProblems.Quadrotor_Helicopter, dict(), None),
            (OCProblems.Satellite_Deorbiting, dict(), None),
            (OCProblems.Three_Tank_Multimode, dict(), None),
            (OCProblems.Time_Optimal_Car, dict(), None),
            (OCProblems.Tubular_Reactor, dict(), None),
            ]

opt_rr = blockSQP2.SQPoptions(
    max_conv_QPs = 4,
    conv_strategy = 'reduced_regularization',
    par_QPs = True,
    enable_QP_cancellation = True,
    automatic_scaling = False
)

opt_rr_scale = blockSQP2.SQPoptions(
    max_conv_QPs = 4,
    conv_strategy = 'reduced_regularization',
    par_QPs = True,
    enable_QP_cancellation = True,
    automatic_scaling = True,
)

Experiments = [
               (opt_rr, "Reduced regularization, no scaling"),
               (opt_rr_scale, "rr, automatic scaling 1**-2**-3-4-5-7 final test 2")
               ]

plot_folder = cD / Path("scaling_comparison_test")


OCP_experiment.run_blockSQP2_experiments(Examples, Experiments,\
                                        plot_folder,\
                                        nPert0 = 0, nPertF = 10,
                                        use_condensing = True
                                        )

