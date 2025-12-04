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

opts.hess_approx = 'SR1'
opts.sizing = 'OL'
opts.fallback_approx = 'BFGS'
opts.fallback_sizing = 'COL'
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
OCP_experiment.plot_successful(1, nPert0, nPertF, [None], EXP_N_SQP, EXP_N_secs, EXP_type_sol, dirPath = cD / Path("out_Cushioned_Oscillation"))

