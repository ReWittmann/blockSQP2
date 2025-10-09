import numpy as np
import os
import sys
import time
try:
    cD = os.path.dirname(os.path.abspath(__file__))
except:
    cD = os.getcwd()
sys.path += [cD + "/.."]

import py_blockSQP
import OCProblems
#Available problems:
# ['Lotka_Volterra_Fishing', 'Lotka_Volterra_multimode', 'Goddard_Rocket', 
#  'Calcium_Oscillation', 'Batch_Reactor', 'Bioreactor', 'Hanging_Chain', 
#  'Hanging_Chain_NQ', 'Catalyst_Mixing', 'Cushioned_Oscillation', 
#  'D_Onofrio_Chemotherapy', 'D_Onofrio_Chemotherapy_VT', 'Egerstedt_Standard', 
#  'Fullers', 'Electric_Car', 'F8_Aircraft', 'Gravity_Turn', 'Oil_Shale_Pyrolysis', 
#  'Particle_Steering', 'Quadrotor_Helicopter', 'Supermarket_Refrigeration', 
#  'Three_Tank_Multimode', 'Time_Optimal_Car', 'Van_der_Pol_Oscillator', 
#  'Van_der_Pol_Oscillator_2', 'Van_der_Pol_Oscillator_3',
#  'Lotka_OED', 'Fermenter', 'Batch_Distillation', 'Hang_Glider', 'Cart_Pendulum'
#  'Tubular_Reactor', ...
#  ]

#Note: ImportError: generic_type: ... is an ipython issue with pybind11/boost::python modules, reload the ipython session in case it occurs

OCprob = OCProblems.Lotka_Volterra_Fishing(
                    nt = 100,           #number of shooting intervals
                    refine = 1,         #number of control intervals per shooting interval
                    integrator = 'RK4', #ODE integrator
                    parallel = True,    #run ODE integration in parallel
                    N_threads = 4       #number of threads for parallelization
                    )

itMax = 200                             #max number of steps
step_plots = False                      #plot each iterate?
plot_title = False                      #Put name of problem in plot?


start = OCprob.start_point              #Start point for problem, can use, e.g. OCprob.perturbed_start_point(k)
################################
opts = py_blockSQP.SQPoptions()
opts.max_QP_it = 10000
opts.max_QP_secs = 10.0

opts.max_conv_QPs = 4                   #max number of additional QPs per SQP iteration including fallback Hess QP
opts.conv_strategy = 2                  #Convexification strategy, 2 requires passing vblocks
opts.par_QPs = True                     #Enable parallel solution of QPs
opts.enable_QP_cancellation = True      #Enable cancellation of long running QP threads
opts.indef_delay = 3                    #Only used fallback Hessian in first # iterations

opts.exact_hess = 0                     #0: No second derivatives, 1: Only last Hess. block, 2: Use excact Hessian
opts.hess_approx = 1                    #1: SR1, 2: damped BFGS
opts.sizing = 2                         #2: OL sizing, 4: Selective COL sizing
opts.fallback_approx = 2                # ''   ''
opts.fallback_sizing = 4                # ''   ''
opts.BFGS_damping_factor = 1/3

opts.lim_mem = True
opts.mem_size = 20
opts.opt_tol = 1e-6                     #Tolerances for termination
opts.feas_tol = 1e-6
opts.conv_kappa_max = 8.                #Maximum Hess regularization factor for conv. strategy, default 8.0

opts.automatic_scaling = True

opts.max_extra_steps = 0                    #Extra steps for improved accuracy
opts.enable_premature_termination = True    #Enable early termination at acceptable tolerance
opts.max_filter_overrides = 2

opts.qpsol = 'qpOASES'
QPopts = py_blockSQP.qpOASES_options()
QPopts.printLevel = 0
QPopts.sparsityLevel = 2
opts.qpsol_options = QPopts
################################

#Create condenser, pass as cond attribute of problem specification to enable condensing
#Currently not recommended due to qpOASES only supporting sparse matrices when allowing indefinite Hessians
vBlocks = py_blockSQP.vblock_array(len(OCprob.vBlock_sizes))    # [{size, dependent : bool}] Free-dependent information, required for conv. strategy 2 and automatic scaling  
cBlocks = py_blockSQP.cblock_array(len(OCprob.cBlock_sizes))
hBlocks = py_blockSQP.int_array(len(OCprob.hessBlock_sizes))
targets = py_blockSQP.condensing_targets(1)
for i in range(len(OCprob.vBlock_sizes)):
    vBlocks[i] = py_blockSQP.vblock(OCprob.vBlock_sizes[i], OCprob.vBlock_dependencies[i]) #Create vblock structs {int size; bool dependent}
for i in range(len(OCprob.cBlock_sizes)):
    cBlocks[i] = py_blockSQP.cblock(OCprob.cBlock_sizes[i])
for i in range(len(OCprob.hessBlock_sizes)):
    hBlocks[i] = OCprob.hessBlock_sizes[i]
targets[0] = py_blockSQP.condensing_target(*OCprob.ctarget_data)
cond = py_blockSQP.Condenser(vBlocks, cBlocks, hBlocks, targets, 2)


#Define blockSQP Problemspec
#See class OCProblems.OCProblem and py_blockSQP/blockSQP_Problemspec.py for field specifications
prob = py_blockSQP.Problemspec()
prob.nVar = OCprob.nVar
prob.nCon = OCprob.nCon

prob.f = lambda x: OCprob.f(x)
prob.grad_f = lambda x: OCprob.grad_f(x)
prob.g = lambda x: OCprob.g(x)
prob.make_sparse(OCprob.jac_g_nnz, OCprob.jac_g_row, OCprob.jac_g_colind)
prob.jac_g_nz = lambda x: OCprob.jac_g_nz(x)
prob.hess = OCprob.hess_lag

prob.set_blockIndex(OCprob.hessBlock_index)
prob.set_bounds(OCprob.lb_var, OCprob.ub_var, OCprob.lb_con, OCprob.ub_con)

#Recommended: Dont pass condenser to activate condensing, but pass vBlocks to enable conv. str. 2 and automatic scaling
# prob.cond = cond
prob.vblocks = vBlocks

prob.x_start = start
prob.lam_start = np.zeros(prob.nVar + prob.nCon, dtype = np.float64).reshape(-1)
prob.complete()


stats = py_blockSQP.SQPstats("./solver_outputs")
t0 = time.monotonic()
optimizer = py_blockSQP.SQPmethod(prob, opts, stats)
optimizer.init()


if (step_plots):
    OCprob.plot(OCprob.start_point, dpi = 150, it = 0, title=plot_title)
    ret = int(optimizer.run(1))
    xi = np.array(optimizer.get_xi()).reshape(-1)
    i = 1
    OCprob.plot(xi, dpi = 150, it = i, title=plot_title)
    while ret == 0 and i < itMax:
        ret = int(optimizer.run(1,1))
        xi = np.array(optimizer.get_xi()).reshape(-1)
        i += 1
        OCprob.plot(xi, dpi = 150, it = i, title=plot_title)
else:
    ret = int(optimizer.run(itMax))
t1 = time.monotonic()
xi = np.array(optimizer.get_xi()).reshape(-1)
OCprob.plot(xi, dpi=200, title=plot_title)


time.sleep(0.01)
print(t1 - t0, "s")