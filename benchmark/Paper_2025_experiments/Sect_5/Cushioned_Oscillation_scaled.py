import numpy as np
import os
import sys
try:
    cD = os.path.dirname(os.path.abspath(__file__))
except:
    cD = os.getcwd()
sys.path += [cD + "/../..", cD + "/../../.."]
import py_blockSQP
import matplotlib.pyplot as plt
import time
import copy
import OCP_experiment

import OCProblems


###############################################################################
OCprob = OCProblems.Cushioned_Oscillation_TSCALE(
                    nt = 100, 
                    parallel = True, 
                    integrator = 'RK4', 
                    TSCALE = 500.0
                    )
nPert0 = 0
nPertF = 40
itMax = 400
###############################################################################

opts = py_blockSQP.SQPoptions()
opts.max_QP_it = 10000
opts.max_QP_secs = 5.0

opts.max_conv_QPs = 4
opts.conv_strategy = 2
opts.par_QPs = True
opts.enable_QP_cancellation = True
opts.indef_delay = 3

opts.exact_hess = 0
opts.hess_approx = 1
opts.sizing = 2
opts.fallback_approx = 2
opts.fallback_sizing = 4
opts.BFGS_damping_factor = 1/3

opts.lim_mem = True
opts.mem_size = 20
opts.opt_tol = 1e-6
opts.feas_tol = 1e-6

opts.automatic_scaling = True

opts.qpsol = 'qpOASES'
QPopts = py_blockSQP.qpOASES_options()
QPopts.printLevel = 0
QPopts.sparsityLevel = 2
opts.qpsol_options = QPopts


ret_N_SQP, ret_N_secs, ret_type_sol = OCP_experiment.perturbed_starts(OCprob, opts, nPert0, nPertF, itMax = itMax)
EXP_N_SQP = [ret_N_SQP]
EXP_N_secs = [ret_N_secs]
EXP_type_sol = [ret_type_sol]
OCP_experiment.plot_successful(1, nPert0, nPertF, [None], EXP_N_SQP, EXP_N_secs, EXP_type_sol, dirPath = cD + "/out_Cushioned_Oscillation")

