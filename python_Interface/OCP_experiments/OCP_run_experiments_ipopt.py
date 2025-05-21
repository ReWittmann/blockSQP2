import os
import sys
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
except:
    sys.path.append(os.getcwd() + "/..")
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
Experiments = [({'hessian_approximation': "limited-memory", 'limited_memory_max_history':12}, "Ipopt, limited-memory BFGS"),
               ({'hessian_approximation': "exact"}, "Ipopt, exact Hessian")
               ]
Examples_ = [(OCProblems.Lotka_Volterra_Fishing, "Lotka Volterra fishing"),
             (OCProblems.Goddard_Rocket, "Goddard's rocket")
             ]
OCP_experiment.run_ipopt_experiments(Examples_, Experiments, plot_folder, nPert0 = 0, nPertF = 5)


























# OCprob = OCProblems.Lotka_Volterra_Fishing(nt=100, refine = 1, integrator = 'RK4', parallel=False)
# OCprob.integrate_full(OCprob.start_point)

# OCprob = OCProblems.Lotka_Volterra_Fishing(nt=NT, integrator = 'rk4', parallel=False)
# OCprob = OCProblems.Bioreactor(nt=100, integrator = 'rk4', parallel=False)
# OCprob = OCProblems.Goddard_Rocket(nt = NT, integrator = 'RK4', parallel = False)
# OCprob = OCProblems.Electric_Car(nt = NT, integrator = 'RK4', parallel = False)
# OCprob = OCProblems.Catalyst_Mixing(nt = NT, integrator = 'RK4', parallel = False)
# OCprob = OCProblems.Three_Tank_Multimode(nt = NT, integrator = 'RK4', parallel = False)
# OCprob = OCProblems.Egerstedt_Standard(nt = NT, integrator = 'RK4', parallel = False)
# OCprob = OCProblems.Fullers(nt = NT, integrator = 'RK4', parallel=False)
# OCprob = OCProblems.Lotka_OED(nt = NT, integrator = 'RK4', parallel = False)
# OCprob = OCProblems.Hanging_Chain(nt = NT, integrator = 'RK4', parallel = False)
# OCprob = OCProblems.Van_der_Pol_Oscillator_3(nt = NT, integrator = 'RK4', parallel = False)
# OCprob = OCProblems.Batch_Reactor(nt = NT, integration_method = 'rk4', parallel=False)
# OCprob = OCProblems.Hang_Glider(nt = NT, integrator='rk4', parallel=False)
# OCprob = OCProblems.Van_der_Pol_Oscillator_3(nt = NT, integrator='rk4', parallel=False)
# OCprob = OCProblems.Time_Optimal_Car(nt = NT, integrator='rk4', parallel=False)
# OCprob = OCProblems.Cushioned_Oscillation(nt = NT, integrator='rk4', parallel=False)


#Made worse (in SQP iterations) by autoscaling, but better in total time. 
# OCprob = OCProblems.Particle_Steering(nt = NT, integrator = 'RK4', parallel = False)


# OCprob = OCProblems.Lotka_Volterra_Fishing_BSC(nt=100, integrator='RK4', parallel=False, sca1=1.0e1, sca2=1.0e-3, sca3=1.0e-2)
# OCprob = OCProblems.Lotka_Volterra_Fishing_BSC(nt=100, integrator='RK4', parallel=False, sca1=1.0, sca2=1.0, sca3=1.0)

# OCprob = OCProblems.Three_Tank_Multimode_BSC(nt = NT, integrator = 'RK4', parallel = False, sca1 = 1.0e3, sca2 = 1.0, sca3 = 1.0e-3)
# OCprob = OCProblems.Egerstedt_Standard_BSC(nt=100,integrator='rk4',parallel=False,sca1=1e-2,sca2=1e-2,sca3=1e2) #Strong SR1 effect
# OCprob = OCProblems.Batch_Distillation(nt=65, integrator = 'cvodes', parallel = True)
