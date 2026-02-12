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

###############################################################################
OCprob = OCProblems.Electric_Car(
                    nt = 100, 
                    integrator = 'RK4', 
                    parallel = True
                    )
nPert0 = 0
nPertF = 40
itMax = 400
###############################################################################

opts = blockSQP2.SQPoptions()
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

opts.qpsol = 'qpOASES'
QPopts = blockSQP2.qpOASES_options()
QPopts.printLevel = 0
QPopts.sparsityLevel = 2
opts.qpsol_options = QPopts


#New termination features disabled
opts.max_extra_steps = 0
opts.enable_premature_termination = False
opts.max_filter_overrides = 0

ret_N_SQP, ret_N_secs, ret_type_sol = OCP_experiment.perturbed_starts(OCprob, opts, nPert0, nPertF, itMax = itMax)
EXP_N_SQP = [ret_N_SQP]
EXP_N_secs = [ret_N_secs]
EXP_type_sol = [ret_type_sol]
OCP_experiment.plot_varshape(1, nPert0, nPertF, [None], EXP_N_SQP, EXP_N_secs, EXP_type_sol, dirPath = cD / Path("out_Electric_Car"))


#New termination features enabled, no extra steps since we dont need extra accuracy for this experiment.
opts.max_extra_steps = 0
opts.enable_premature_termination = True
opts.max_filter_overrides = 2

ret_N_SQP, ret_N_secs, ret_type_sol = OCP_experiment.perturbed_starts(OCprob, opts, nPert0, nPertF, itMax = itMax)
EXP_N_SQP = [ret_N_SQP]
EXP_N_secs = [ret_N_secs]
EXP_type_sol = [ret_type_sol]
OCP_experiment.plot_varshape(1, nPert0, nPertF, [None], EXP_N_SQP, EXP_N_secs, EXP_type_sol, dirPath = cD / Path("out_Electric_Car"))

