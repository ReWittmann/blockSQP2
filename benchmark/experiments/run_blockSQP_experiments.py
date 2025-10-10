import os
import sys
import datetime

try:
    cD = os.path.dirname(os.path.abspath(__file__))
except:
    cD = os.getcwd()
sys.path += [cD + "/..", cD + "/../.."]

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

#SR1_BFGS
opt_SR1_BFGS = py_blockSQP.SQPoptions()
opt_SR1_BFGS.max_conv_QPs = 1
opt_SR1_BFGS.max_filter_overrides = 0
opt_SR1_BFGS.BFGS_damping_factor = 0.2

#Convexification strategy 0
opt_CS0 = py_blockSQP.SQPoptions()
opt_CS0.max_conv_QPs = 4
opt_CS0.conv_strategy = 0
opt_CS0.max_filter_overrides = 0

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

#Select option sets to test for
Experiments = [
               (opt_SR1_BFGS, "SR1-BFGS"),
               # (opt_CS0, "Convexification strategy 0"),
               (opt_CS1, "conv. str. 1"),
               (opt_CS2, "conv. str. 2")
               ]


plot_folder = cD + "/out_blockSQP_experiments"

#Choose perturbed start points to test for,
#modify discretized initial controls u_k in turn for nPert0 <= k < nPertF
nPert0 = 0
nPertF = 40

#Write results to a file?
file_output = True


#Run all example problems for all option sets for perturbed start points
dirPath = plot_folder
if not os.path.exists(dirPath):
    os.makedirs(dirPath)
if file_output:
    date_app = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_").replace("'", "")
    sep = "" if dirPath[-1] == "/" else "/"
    pref = "blockSQP"
    filePath = dirPath + sep + pref + "_it_" + date_app + ".txt"
    out = open(filePath, 'w')
else:
    out = OCP_experiment.out_dummy()

titles = [EXP_name for _, EXP_name in Experiments]
OCP_experiment.print_heading(out, titles)
for OCclass in Examples:        
    OCprob = OCclass(nt = 100, integrator = 'RK4', parallel = True)
    itMax = 200
    titles = []
    EXP_N_SQP = []
    EXP_N_secs = []
    EXP_type_sol = []
    n_EXP = 0
    for EXP_opts, EXP_name in Experiments:
        ret_N_SQP, ret_N_secs, ret_type_sol = OCP_experiment.perturbed_starts(OCprob, EXP_opts, nPert0, nPertF, itMax = itMax)
        EXP_N_SQP.append(ret_N_SQP)
        EXP_N_secs.append(ret_N_secs)
        EXP_type_sol.append(ret_type_sol)
        titles.append(EXP_name)
        n_EXP += 1
    ###############################################################################
    OCP_experiment.plot_successful(n_EXP, nPert0, nPertF,\
        titles, EXP_N_SQP, EXP_N_secs, EXP_type_sol,\
        suptitle = None, dirPath = dirPath, savePrefix = "blockSQP")
    OCP_experiment.print_iterations(out, OCclass.__name__, EXP_N_SQP, EXP_N_secs, EXP_type_sol)
out.close()
