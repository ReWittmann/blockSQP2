import numpy as np
import py_blockSQP
import OCProblems

itMax = 400
plot_title = True
######################
##Available problems##
######################
#Easy to medium difficulty for blockSQP
# OCprob = OCProblems.Lotka_Volterra_Fishing(nt=100, refine = 1, integrator = 'RK4',parallel=False)
# OCprob = OCProblems.Lotka_Volterra_multimode(nt=100, refine = 1, integrator = 'RK4',parallel=False)
OCprob = OCProblems.Goddard_Rocket(integrator = 'rk4', nt = 100, parallel=False) #Strong SR1 effect
# OCprob = OCProblems.Calcium_Oscillation(nt = 100, integrator = 'cvodes', parallel= False)
# OCprob = OCProblems.Batch_Reactor(nt=100, integration_method = 'rk4', parallel=False)
# OCprob = OCProblems.Bioreactor(nt=100, integrator = 'rk4', parallel=False)
# OCprob = OCProblems.Hanging_Chain(nt=100,integrator='rk4',parallel=False)
# OCprob = OCProblems.Catalyst_Mixing(nt=100,integrator='rk4',parallel=False)
# OCprob = OCProblems.Cushioned_Oscillation(nt=100,integrator='rk4',parallel=False)#Runs into local infeasibility when setting minimum time horizon to 5 seconds, minimum time is set to 8 to avoid this
# OCprob = OCProblems.Egerstedt_Standard(nt=100,integrator='rk4',parallel=False) #Strong SR1 effect
# OCprob = OCProblems.Fullers(nt=100,integrator='rk4',parallel=False)
# OCprob = OCProblems.Electric_Car(nt=100,integrator='rk4',parallel=False) #Strong SR1 effect
# OCprob = OCProblems.F8_Aircraft(nt=200,integrator='rk4',parallel=False) #nt=200 + perturbed start for best local optimum
# OCprob = OCProblems.Oil_Shale_Pyrolysis(nt=100,integrator='rk4',parallel=False)
# OCprob = OCProblems.Particle_Steering(nt=101,integrator='rk4',parallel=False) #Weak SR1 effect, good BFGS performance
# OCprob = OCProblems.Quadrotor_Helicopter(nt=100,integrator='rk4',parallel=False)
# OCprob = OCProblems.Three_Tank_Multimode(nt=100,integrator='rk4',parallel=False)
# OCprob = OCProblems.Time_Optimal_Car(nt=100,integrator='rk4',parallel=False) #Bang bang
# OCprob = OCProblems.Van_der_Pol_Oscillator(nt=100,integrator='cvodes',parallel=False)
# OCprob = OCProblems.Van_der_Pol_Oscillator_2(nt=100,integrator='cvodes',parallel=False)
# OCprob = OCProblems.Van_der_Pol_Oscillator_3(nt=100,refine=1,integrator='rk4',parallel=False)
# OCprob = OCProblems.Ocean(nt=100,integrator='cvodes',parallel=False)
# OCprob = OCProblems.Lotka_OED(nt=100,integrator='rk4',parallel=False)

#Hard for blockSQP
# OCprob = OCProblems.D_Onofrio_Chemotherapy(nt=100,integrator='cvodes',parallel=False, duration = 6., **OCProblems.D_Onofrio_Chemotherapy.param_set_1)
# OCprob = OCProblems.Gravity_Turn(nt=100,integrator='cvodes',parallel=True) #Exact Hessian recommended
# OCprob = OCProblems.Supermarket_Refrigeration(nt=50,integrator='cvodes',parallel=True)

prob = py_blockSQP.Problemspec()
prob.nVar = OCprob.nVar
prob.nCon = OCprob.nCon

prob.f = OCprob.f
prob.grad_f = OCprob.grad_f
prob.g = OCprob.g
prob.make_sparse(OCprob.jac_g_nnz, OCprob.jac_g_row, OCprob.jac_g_colind)
prob.jac_g_nz = OCprob.jac_g_nz
prob.hess = OCprob.hess_lag

prob.set_blockIndex(OCprob.hessBlock_index)
prob.set_bounds(OCprob.lb_var, OCprob.ub_var, OCprob.lb_con, OCprob.ub_con)

prob.x_start = OCprob.start_point
prob.lam_start = np.zeros(prob.nVar + prob.nCon, dtype = np.float64).reshape(-1)

prob.complete()


opts = py_blockSQP.SQPoptions();
opts.maxItQP = 100000000
opts.maxConvQP = 4
opts.convStrategy = 1
opts.restoreFeas = 1
opts.maxTimeQP = 8
opts.hessMemsize = 20
opts.iniHessDiag = 1.0
opts.hessUpdate = 1
opts.hessScaling = 2
opts.fallbackUpdate = 2
opts.fallbackScaling = 4
opts.hessDampFac = 1./3.

opts.whichSecondDerv = 0
opts.hessLimMem = 1
opts.hessMemsize = 20
opts.opttol = 1e-6
opts.nlinfeastol = 1e-6
opts.skipFirstGlobalization = 0
opts.maxLineSearch = 7
opts.max_bound_refines = 3
opts.max_correction_steps = 6
opts.which_QPsolver = 'qpOASES'
opts.qpOASES_terminationTolerance = 1e-10
opts.qpOASES_printLevel = 0

opts.tau_H = 2./3.

#####################
###qpOASES options###
#####################
opts.qpOASES_terminationTolerance = 1e-10
opts.qpOASES_printLevel = 0

#####################
stats = py_blockSQP.SQPstats("./solver_outputs")
optimizer = py_blockSQP.SQPmethod(prob, opts, stats)
optimizer.init()
#####################
##Standard run

# ret = optimizer.run(itMax)
# xi = np.array(optimizer.vars.xi).reshape(-1)
# OCprob.plot(xi)
#####################
##Plot all steps

OCprob.plot(OCprob.start_point, dpi = 100, it = 0, title=plot_title)
ret = optimizer.run(1)
xi = np.array(optimizer.vars.xi).reshape(-1)
i = 1
OCprob.plot(xi, dpi = 100, it = i, title=plot_title)
while ret != 0 and ret != -1 and i < itMax:
    ret = optimizer.run(1,1)
    xi = np.array(optimizer.vars.xi).reshape(-1)
    i += 1
    OCprob.plot(xi, dpi = 100, it = i, title=plot_title)
#####################


