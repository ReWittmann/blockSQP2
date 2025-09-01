import casadi as cs
import numpy as np

import numpy as np
# import os
# import sys
# try:
#     cD = os.path.dirname(os.path.abspath(__file__))
# except:
#     cD = os.getcwd()
# sys.path += [cD]

import OCProblems
import time

itMax = 500


OCprob = OCProblems.Lotka_Volterra_Fishing(nt = 100,
                    refine = 1,
                    parallel = True,
                    integrator = 'RK4',
                    # MDTH = 1.0
                    )

ipopts = dict()
ipopts['hessian_approximation'] = 'exact'
# ipopts['limited_memory_max_history'] = 20
ipopts['constr_viol_tol'] = 1e-5
ipopts['tol'] = 1e-5
ipopts['max_iter'] = itMax

sp = OCprob.start_point
S = cs.nlpsol('S', 'ipopt', OCprob.NLP, {'ipopt':ipopts})
t0 = time.monotonic()
out = S(x0=sp, lbx=OCprob.lb_var,ubx=OCprob.ub_var, lbg=OCprob.lb_con, ubg=OCprob.ub_con)
t1 = time.monotonic()
xi = out['x']
OCprob.plot(np.array(xi).reshape(-1), dpi=200)

time.sleep(0.1)
print(t1 - t0, "s")