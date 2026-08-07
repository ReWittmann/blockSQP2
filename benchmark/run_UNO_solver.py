# blockSQP2 -- A structure-exploiting nonlinear programming solver based
#              on blockSQP by Dennis Janka.
# Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>

# Licensed under the zlib license. See LICENSE for more details.


# \file run_blockSQP2.py
# \author Reinhold Wittmann
# \date 2025
#
# Script to invoke blockSQP2 for an example problem.

import numpy as np
import time
import sys
from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()
sys.path += [str(cD.parent/Path("Python"))]

import unopy
import OCProblems

#Check OCProblems.py for available examples
OCprob = OCProblems.Lotka_Volterra_Fishing(
                    nt = 100,               #number of shooting intervals
                    refine = 1,             #number of control intervals per shooting interval
                    # integrator = 'RK4',     #ODE integrator
                    parallel = True,        #run ODE integration in parallel
                    N_threads = 4,          #number of threads for parallelization
                                            #problem specific keyword parameters, e.g. c0, c1, x_init, t0, tf for Lotka_Volterra_Fishing, see default_params of problems
                    )
itMax = 200                                  #max number of steps


def create_UNO_model(OCprob:OCProblems.OCProblem, start_pert = None):
    model = unopy.Model(unopy.PROBLEM_NONLINEAR,
                        OCprob.nVar,
                        unopy.ZERO_BASED_INDEXING
                        )
    model.set_variables_lower_bounds(OCprob.lb_var)
    model.set_variables_upper_bounds(OCprob.ub_var)
    
    model.set_objective(unopy.MINIMIZE, OCprob.f, OCprob.grad_f_inplace)
    model.set_constraints(OCprob.nCon, OCprob.g_inplace, OCprob.lb_con, OCprob.ub_con, OCprob.jac_g_nnz, np.array(OCprob.jac_g_row), OCprob.jac_g_col, OCprob.jac_g_nz_inplace)
    
    model.set_lagrangian_sign_convention(unopy.MULTIPLIER_NEGATIVE)
    model.set_lagrangian_hessian(OCprob.hess_LT_nnz, unopy.LOWER_TRIANGLE, OCprob.hess_LT_row, OCprob.hess_LT_col, OCprob.hess_lag_objmult_inplace)
    
    if start_pert is not None:
        model.set_initial_primal_iterate(OCprob.perturbed_start_point(start_pert))
    else:
        model.set_initial_primal_iterate(OCprob.start_point)
    return model

ret_f = np.zeros(OCprob.nVar)   
OCprob.grad_f_inplace(OCprob.start_point, ret_f)

ret_g = np.zeros(OCprob.nCon)
OCprob.g_inplace(OCprob.start_point, ret_g)

ret_jac_g = np.zeros(OCprob.jac_g_nnz)
OCprob.jac_g_nz_inplace(OCprob.start_point, ret_jac_g)

ret_hess = np.zeros(OCprob.hess_LT_nnz)
OCprob.hess_lag_objmult_inplace(OCprob.start_point, 1, np.zeros(OCprob.nCon), ret_hess)

UNOmodel = create_UNO_model(OCprob)
UNOsol = unopy.UnoSolver()
UNOsol.set_preset("filtersqp")
UNOsol.set_option("QP_solver", "BQPD")
# UNOsol.set_option("hessian_model", "LBFGS")
UNOsol.set_option("linear_solver", "MUMPS")

t0 = time.time()
result = UNOsol.optimize(UNOmodel)
t1 = time.time()    

OCprob.plot(result.primal_solution)

time.sleep(0.25)
print("\n", t1 - t0, "s")