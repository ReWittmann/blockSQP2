import sys
from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()
sys.path += [str(cD.parents[1]), str(cD.parents[2]/Path("Python"))]
import blockSQP2
import copy
import datetime
import OCP_experiment
import OCProblems

#RK4/collocation/cvodes
ODE_integrator = 'RK4'
dirPath = cD / Path("out_blockSQP2_ipopt_comparison_RK4")

#Range for applying perturbations to initial discretized controls
nPert0 = 0
nPertF = 40

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
            (OCProblems.Goddard_Rocket, dict(), 'Goddard\'s Rocket'),
            (OCProblems.Hang_Glider, dict(), None),
            (OCProblems.Hanging_Chain, dict(), None),
            (OCProblems.Lotka_Volterra_Fishing, dict(), None),
            (OCProblems.Lotka_Volterra_Fishing, OCProblems.Lotka_Volterra_Fishing.param_set_2, "Lotka_Volterra_Fishing_2"),
            (OCProblems.Lotka_Volterra_Fishing, OCProblems.Lotka_Volterra_Fishing.param_set_3, "Lotka_Volterra_Fishing_3"),
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

# [(solver options, experiment name)]
ipopt_Experiments = [
                     ({'ipopt':{
                                 'hessian_approximation': 'limited-memory', 
                                 'tol': 1e-6, 
                                 'constr_viol_tol': 1e-6
                                 }}, 
                      'ipopt, limited-memory'),
                     ({'ipopt':{
                                'hessian_approximation': 'exact', 
                                'tol': 1e-6, 
                                'constr_viol_tol': 1e-6}}, 
                     'ipopt, exact Hessian')
                     ]

def opt_conv_str_2_par_scale(max_conv_QPs = 4):
    opts = blockSQP2.SQPoptions()
    opts.max_conv_QPs = max_conv_QPs
    opts.conv_strategy = 2
    opts.par_QPs = True
    opts.automatic_scaling = True
    return opts

opt1 = opt_conv_str_2_par_scale(max_conv_QPs = 4)
opt2 = opt_conv_str_2_par_scale(max_conv_QPs = 4)
opt2.hess_approx = "exact"


blockSQP2_Experiments = [
                        (opt1, 'blockSQP2, SR1-...-BFGS'),
                        (opt2, 'blockSQP2, exH-...-BFGS')
                        ]


#Run the experiments
dirPath.mkdir(parents = True, exist_ok = True)

#Create an open file to write results into
date_app = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_").replace("'", "")
pref = "blockSQP2_ipopt"
filePath = dirPath / Path(pref + "_it_" + date_app + ".txt")
out = open(filePath, 'w')


titles = [EXP_name for _, EXP_name in ipopt_Experiments + blockSQP2_Experiments]
OCP_experiment.print_heading(out, titles)

#Iterate over example problems and experiments
for OCclass, OCargs, OCname in Examples:
    OCprob = OCclass(nt=100, integrator=ODE_integrator, parallel = True, **OCargs)
    itMax = 1000
    ipopts_base = {'max_iter':itMax}
    EXP_N_SQP = []
    EXP_N_secs = []
    EXP_type_sol = []
    n_EXP = 0
    for EXP_opts, EXP_name in ipopt_Experiments:
        ipopts = copy.deepcopy(EXP_opts)
        try:
            ipopts['ipopt']['max_iter'] = itMax
        except KeyError:
            ipopts['ipopt'] = {'max_iter':itMax}
        ret_N_SQP, ret_N_secs, ret_type_sol = OCP_experiment.casadi_solver_perturbed_starts('ipopt', OCprob, ipopts, nPert0, nPertF, itMax = itMax)
        EXP_N_SQP.append(ret_N_SQP)
        EXP_N_secs.append(ret_N_secs)
        EXP_type_sol.append(ret_type_sol)
        n_EXP += 1
    
    for EXP_opts, EXP_name in blockSQP2_Experiments:
        ret_N_SQP, ret_N_secs, ret_type_sol = OCP_experiment.perturbed_starts(OCprob, EXP_opts, nPert0, nPertF, itMax = itMax)
        EXP_N_SQP.append(ret_N_SQP)
        EXP_N_secs.append(ret_N_secs)
        EXP_type_sol.append(ret_type_sol)
        titles.append(EXP_name)
        n_EXP += 1
    
    if OCname is None:
        OCname = OCclass.__name__
    #Create scatter plot of total iterations and runtimes for problem
    OCP_experiment.plot_successful(n_EXP, nPert0, nPertF,\
        titles, EXP_N_SQP, EXP_N_secs, EXP_type_sol,\
        suptitle = OCname, dirPath = dirPath, savePrefix = "blockSQP2_ipopt")
    #Print results (iterations/runtime - mean/stddev) for problem to file
    OCP_experiment.print_iterations(out, OCname, EXP_N_SQP, EXP_N_secs, EXP_type_sol)
out.close()
