import numpy as np
import casadi as cs
import os
import sys
import time
try:
    sys.path += [os.path.dirname(os.path.abspath(__file__)) + "/..", os.path.dirname(os.path.abspath(__file__)) + "/../.."]
except:
    sys.path += [os.getcwd() + "/..", os.getcwd() + "/../.."]


import OCProblems
import matplotlib.pyplot as plt
import time

itMax = 200
OCprob = OCProblems.Lotka_OED(nt=100, fishing = True)

ipopts = dict()
ipopts['hessian_approximation'] = 'limited-memory'
ipopts['limited_memory_max_history'] = 20
ipopts['constr_viol_tol'] = 1e-6
ipopts['tol'] = 1e-6
# ipopts['hsllib'] = '/home/reinhold/coinhsl-solvers/.libs/libcoinhsl.so.0.0.0'
sp = OCprob.start_point

S = cs.nlpsol('S', 'ipopt', OCprob.NLP, {'ipopt':ipopts})
out = S(x0=sp, lbx=OCprob.lb_var,ubx=OCprob.ub_var, lbg=OCprob.lb_con, ubg=OCprob.ub_con)
xi = out['x']
OCprob.plot(np.array(xi).reshape(-1), dpi=200)