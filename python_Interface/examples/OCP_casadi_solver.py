import casadi as cs
import numpy as np

import numpy as np
import os
import sys
sys.path.append('/home/reinhold/blocksqp/python_Interface/OCP_experiments')
import OCProblems
import time

itMax = 2000


OCprob = OCProblems.Lotka_Volterra_Fishing(nt = 100,
                    refine = 1,
                    parallel = True,
                    integrator = 'RK4',
                    )

ipopts = dict()
# ipopts['hessian_approximation'] = 'limited-memory'
# ipopts['limited_memory_max_history'] = 20
ipopts['constr_viol_tol'] = 1e-5
ipopts['tol'] = 1e-5
ipopts['max_iter'] = itMax

sp = OCprob.start_point
S = cs.nlpsol('S', 'ipopt', OCprob.NLP, {'ipopt':ipopts})
out = S(x0=sp, lbx=OCprob.lb_var,ubx=OCprob.ub_var, lbg=OCprob.lb_con, ubg=OCprob.ub_con)
xi = out['x']
OCprob.plot(np.array(xi).reshape(-1), dpi=200)