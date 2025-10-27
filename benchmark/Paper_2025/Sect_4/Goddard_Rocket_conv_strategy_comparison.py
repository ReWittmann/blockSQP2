import sys
from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()
sys.path += [str(cD.parents[1]), str(cD.parents[2])]
import datetime

import py_blockSQP
import OCP_experiment
import OCProblems


Examples = [OCProblems.Goddard_Rocket]

opt_SR1_BFGS = py_blockSQP.SQPoptions()
opt_SR1_BFGS.max_conv_QPs = 1
opt_SR1_BFGS.par_QPs = False
opt_SR1_BFGS.automatic_scaling = False
opt_SR1_BFGS.max_filter_overrides = 0
opt_SR1_BFGS.BFGS_damping_factor = 0.2

#Convexification strategy 0
# opt_CS0 = py_blockSQP.SQPoptions()
# opt_CS0.max_conv_QPs = 4
# opt_CS0.conv_strategy = 0
# opt_CS0.max_filter_overrides = 0

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
               # (opt_CS0, "Convexification strategy 0"),
               (opt_CS1, "conv. str. 1"),
               (opt_CS2, "conv. str. 2")
               ]

plot_folder = cD / Path("out_Goddard_conv_strategy")
nPert0 = 0
nPertF = 40
dirPath = plot_folder
dirPath.mkdir(parents = True, exist_ok = True)
print_output = True
if print_output:
    date_app = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_").replace("'", "")
    pref = "blockSQP"
    filePath = dirPath / Path(pref + "_it_" + date_app + ".txt")
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
    OCP_experiment.plot_successful_small(n_EXP, nPert0, nPertF,\
        titles, EXP_N_SQP, EXP_N_secs, EXP_type_sol,\
        suptitle = None, dirPath = dirPath, savePrefix = "blockSQP")
    OCP_experiment.print_iterations(out, OCclass.__name__, EXP_N_SQP, EXP_N_secs, EXP_type_sol)
out.close()
