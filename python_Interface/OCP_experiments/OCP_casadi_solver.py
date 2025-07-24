import casadi as cs
import numpy as np

import numpy as np
import sys
# sys.path.append('/home/reinhold/Test_codes')
sys.path.append('/home/reinhold/blocksqp/python_Interface/OCP_experiments')

class CountCallback(cs.Callback):
    def __init__(self, name, nx, ng, np):
        cs.Callback.__init__(self)
        self.it = -1
        self.prim_vars = cs.DM([])
        self.nx = nx
        self.ng = ng
        self.np = np
        self.construct(name, {})

    
    def get_n_in(self):
        return cs.nlpsol_n_out()

    def get_n_out(self):
        return 1

    def get_name_in(self, i):
        return cs.nlpsol_out(i)

    def get_name_out(self, i):
        return "ret"

    def get_sparsity_in(self, i):
        n = cs.nlpsol_out(i)
        if n == 'f':
            return cs.Sparsity.scalar()
        elif n in ('x', 'lam_x'):
            return cs.Sparsity.dense(self.nx)
        elif n in ('g', 'lam_g'):
            return cs.Sparsity.dense(self.ng)
        else:
            return cs.Sparsity(0, 0)
    
    def eval(self, arg):
        self.it += 1
        return [0]

import OCProblems
# import OCProblem_2
import matplotlib.pyplot as plt
import time
# from tqdm import tqdm #progress bar

itMax = 400
######################
##Available problems##
######################
#Easy to medium for blockSQP
# OCprob = OCProblems.Lotka_Volterra_Fishing(nt=100, refine = 1, integrator = 'RK4', parallel=False, fishing = True)
OCprob = OCProblems.Lotka_OED(nt=100, refine = 1, integrator = 'rk4', parallel=False)
# OCprob = OCProblems.Goddard_Rocket(integrator = 'rk4', nt = 100, refine=1, parallel=False) #Strong SR1 effect
# OCprob = OCProblems.Calcium_Oscillation(nt = 100, integrator = 'cvodes', parallel= False)
# OCprob = OCProblems.Batch_Reactor(nt=100, integration_method = 'rk4', parallel=False)
# OCprob = OCProblems.Bioreactor(nt=100, integrator = 'rk4', parallel=False)
# OCprob = OCProblems.Hanging_Chain(nt=100, integrator='rk4', parallel=True)
# OCprob = OCProblems.Hanging_Chain_NQ(nt=100,integrator='rk4',parallel=False)
# OCprob = OCProblems.Catalyst_Mixing(nt=100,integrator='rk4',parallel=False)
# OCprob = OCProblems.Cushioned_Oscillation(nt=100,integrator='rk4',parallel=False)
# OCprob = OCProblems.Egerstedt_Standard(nt=100,integrator='cvodes',parallel=False) #Strong SR1 effect
# OCprob = OCProblems.Fullers(nt=100,integrator='rk4',parallel=False)
# OCprob = OCProblems.Electric_Car(nt=100,integrator='rk4',parallel=False) #Strong SR1 effect
# OCprob = OCProblems.F8_Aircraft(nt=100,integrator='rk4',parallel=False)
# OCprob = OCProblems.Oil_Shale_Pyrolysis(nt=90,integrator='rk4',parallel=False)
# OCprob = OCProblems.Particle_Steering(nt=100,integrator='rk4',parallel=False)
# OCprob = OCProblems.Quadrotor_Helicopter(nt=100,integrator='rk4',parallel=False)
# OCprob = OCProblems.Three_Tank_Multimode(nt=200,integrator='rk4',parallel=False)
# OCprob = OCProblems.Time_Optimal_Car(nt=200,integrator='cvodes',parallel=False)
# OCprob = OCProblems.Van_der_Pol_Oscillator(nt=400,integrator='cvodes',parallel=False)
# OCprob = OCProblems.Van_der_Pol_Oscillator_2(nt=100,integrator='cvodes',parallel=False)
# OCprob = OCProblems.Ocean(nt=100,integrator='cvodes',parallel=False)

# OCprob = OCProblems.Lotka_Volterra_Fishing(integrator = 'rk4', nt = 92, refine=1, parallel=False)

#Hard for blockSQP
# OCprob = OCProblems.D_Onofrio_Chemotherapy(nt=100,integrator='cvodes',parallel=False, duration = 6., **OCProblems.D_Onofrio_Chemotherapy.param_set_1)
# OCprob = OCProblems.D_Onofrio_Chemotherapy_VT(nt=20,integrator='cvodes',parallel=False)
# OCprob = OCProblems.Gravity_Turn(nt=75,integrator='cvodes',parallel=False)
# OCprob = OCProblems.Supermarket_Refrigeration(nt=50,integrator='rk4',parallel=False)

# OCprob = OCProblems.Lotka_Volterra_Fishing_S(nt=3, refine=25,
                                             # integrator = 'explicit_euler',
                                             # S_u = 1000.0)



counter = CountCallback('counter', OCprob.NLP['x'].size1(), OCprob.NLP['g'].size1(), 0)
ipopts = dict()
ipopts['hessian_approximation'] = 'limited-memory'
# ipopts['limited_memory_max_history'] = 20
ipopts['constr_viol_tol'] = 1e-5
ipopts['tol'] = 1e-5
ipopts['max_iter'] = itMax
# ipopts['hsllib'] = '/home/reinhold/coinhsl-solvers/.libs/libcoinhsl.so.0.0.0'
sp = OCprob.start_point

S = cs.nlpsol('S', 'ipopt', OCprob.NLP, {'ipopt':ipopts, 'iteration_callback':counter})
t0 = time.time()
out = S(x0=sp, lbx=OCprob.lb_var,ubx=OCprob.ub_var, lbg=OCprob.lb_con, ubg=OCprob.ub_con)
t1 = time.time()
xi = out['x']
OCprob.plot(np.array(xi).reshape(-1), dpi=200)

print(t1 - t0, "s")