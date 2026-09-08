#See https://github.com/acados/acados on how to install acados and 
#the python package acados_template


import time



import casadi as cs
import sys
import copy
from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()
sys.path += [str(cD.parents[1]), str(cD.parents[1]/Path('experiments'))]
import OCProblems_fatrop
import OCP_experiment
import blockSQP2
import OCProblems
import numpy as np


#Note: We use the perbured start point no. 1 for catalyst mixing OED,
#      because blockSQP2 shows inconsistent iteration counts between
#      runs for the default start point (either 17 or 24, we suspect due 
#      to randomness in the sparse linear solver MUMPS). fatrop's and 
#      acados' iteration counts seem unaffected, blockSQP2's iteration
#      count is (hopefully) consistently 19, which is close to the average.
use_pert_start_1 = True


#ACADOS
import acados_models as acmo

print("\n###\nSetting up catalyst mixing oed for acados, this should take ~80s on a recent CPU\n###\n")

tm1_acados_catalyst = time.monotonic()
#First perturbed start point somehow makes acados take three times as long despite requiring the same number of iterations, so always use default start point.
acados_solver = acmo.setup_catalyst_mixing_oed_ocp(pert_start_point = None) ###(pert_start_point = 1 if use_pert_start_1 else None)
t0_acados_catalyst = time.monotonic()
acados_solver.solve()
t1_acados_catalyst = time.monotonic()
it_acados_catalyst = acados_solver.get_stats("nlp_iter")

# Somehow acados takes over two times the runtime if acados solve was called before ...
# Still, I see nothing wrong with this setup and will take this value
tm1_acados_D_Onofrio = time.monotonic()
acados_solver = acmo.setup_D_Onofrio_ocp()
t0_acados_D_Onofrio = time.monotonic()
acados_solver.solve()
t1_acados_D_Onofrio = time.monotonic()
it_acados_D_Onofrio = acados_solver.get_stats("nlp_iter")



#FATROP
fatropts = {
    'jit': True,
    'expand': False,
    'jit_options': {'flags': '-O3', 'verbose': False},
    'fatrop':{'tol':1e-6, 'constr_viol_tol':1e-4, 'print_level': 10, 'max_iter': 300},
    'debug': False    
    }

OCprob = OCProblems_fatrop.Catalyst_Mixing_OED_noQuads(
                    nt = 40,
                    refine = 1,
                    parallel = True,
                    N_threads = 4, 
                    )


n_path_constr, n_term_constr, path_constr_0, path_constr_F = OCProblems_fatrop.get_constr_data(OCprob)
g_expr_new, lb_con_new, ub_con_new = OCP_experiment.reorder_constr_for_fatrop(OCprob.NLP['g'], OCprob.lb_con, OCprob.ub_con, OCprob.ntS, OCprob.nx, n_path_constr, n_term_constr, path_constr_0, path_constr_F)


NLP = copy.deepcopy(OCprob.NLP)
NLP['g'] = g_expr_new


tm1_fatrop_catalyst = time.monotonic()

print("\n###\nSetting up catalyst mixing OED for fatrop, this should take ~25s on a recent CPU\n###\n")
S = cs.nlpsol('S', 'fatrop', NLP, 
                  {'structure_detection' : 'manual', 
                    'nx':[len([x for x in OCprob.x_init if x is None])] + [OCprob.nx]*OCprob.ntS, 
                    'nu': [OCprob.nu]*OCprob.ntS + [0], 
                    'ng': [n_path_constr*int(path_constr_0)] + [n_path_constr]*(OCprob.ntS-1) + [n_path_constr*int(path_constr_F) + n_term_constr], 'N':OCprob.ntS, 
                    } | fatropts
              )

sp = OCprob.perturbed_start_point(1) if use_pert_start_1 else OCprob.start_point
t0_fatrop_catalyst = time.monotonic()
out = S(x0=sp, lbx=OCprob.lb_var,ubx=OCprob.ub_var, lbg = lb_con_new, ubg = ub_con_new)
t1_fatrop_catalyst = time.monotonic()

stats = S.stats()
it_fatrop_catalyst = stats["fatrop"]["iterations_count"]

OCprob = OCProblems_fatrop.D_Onofrio_Chemotherapy_noQuads(
                    nt = 100,
                    refine = 1,
                    parallel = True,
                    N_threads = 4, 
                    )

n_path_constr, n_term_constr, path_constr_0, path_constr_F = OCProblems_fatrop.get_constr_data(OCprob)
g_expr_new, lb_con_new, ub_con_new = OCP_experiment.reorder_constr_for_fatrop(OCprob.NLP['g'], OCprob.lb_con, OCprob.ub_con, OCprob.ntS, OCprob.nx, n_path_constr, n_term_constr, path_constr_0, path_constr_F)


NLP = copy.deepcopy(OCprob.NLP)
NLP['g'] = g_expr_new


tm1_fatrop_D_Onofrio = time.monotonic()
S = cs.nlpsol('S', 'fatrop', NLP, 
                  {'structure_detection' : 'manual', 
                    'nx':[len([x for x in OCprob.x_init if x is None])] + [OCprob.nx]*OCprob.ntS, 
                    'nu': [OCprob.nu]*OCprob.ntS + [0], 
                    'ng': [n_path_constr*int(path_constr_0)] + [n_path_constr]*(OCprob.ntS-1) + [n_path_constr*int(path_constr_F) + n_term_constr], 'N':OCprob.ntS, 
                    } | fatropts
              )

t0_fatrop_D_Onofrio = time.monotonic()
out = S(x0=OCprob.start_point, lbx=OCprob.lb_var,ubx=OCprob.ub_var, lbg = lb_con_new, ubg = ub_con_new)
t1_fatrop_D_Onofrio = time.monotonic()
stats = S.stats()
it_fatrop_D_Onofrio = stats["fatrop"]["iterations_count"]



#BLOCKSQP2
OCprob = OCProblems.Catalyst_Mixing_OED(
                    nt = 40,
                    parallel = True,
                    N_threads = 4, 
                    )

tm1_blockSQP2_catalyst = time.monotonic()
print("\n###\nSetting up catalyst mixing OED for blockSQP2, this should take ~12s on a recent CPU\n###\n")
OCprob.jit(jit_hess = False)

opts = blockSQP2.SQPoptions(
    max_QP_it = 10000,
    max_QP_secs = 20.0,
    max_conv_QPs = 4,
    conv_strategy = 'reduced_regularization',
    par_QPs = True,
    automatic_scaling = True,
)

vblocks = [blockSQP2.vblock(size, dep, impl) for size, dep, impl in zip(OCprob.vBlock_sizes, OCprob.vBlock_dependencies, OCprob.vBlock_bounds_implicit)]
cblocks = [blockSQP2.cblock(size) for size in OCprob.cBlock_sizes]
hblocks = [size for size in OCprob.hessBlock_sizes]
targets = [blockSQP2.condensing_target(*OCprob.ctarget_data)]

condenser = blockSQP2.PartialCondenser(vblocks, cblocks, hblocks, targets, 4)

prob = blockSQP2.Problemspec(OCprob.nVar, OCprob.nCon)
prob.f = OCprob.f
prob.grad_f = OCprob.grad_f
prob.g = OCprob.g

prob.make_sparse(OCprob.jac_g_nnz, OCprob.jac_g_row, OCprob.jac_g_colind)
prob.jac_g_nz = OCprob.jac_g_nz

prob.hess = OCprob.hess_lag
prob.blockIdx = OCprob.hessBlock_index
prob.set_bounds(OCprob.lb_var, OCprob.ub_var, OCprob.lb_con, OCprob.ub_con)

prob.vblocks = vblocks
prob.condenser = condenser

sp = OCprob.perturbed_start_point(1) if use_pert_start_1 else OCprob.start_point
prob.x_start = sp
prob.lam_start = np.zeros(prob.nVar + prob.nCon, dtype = np.float64).reshape(-1)

stats = blockSQP2.SQPstats("./solver_outputs")

#Force all dl(m)open and dlsym calls to be loaded, else single run-time is skewed.
#This would not be an issue with a threadsafe sparse linear solver such as HSL MA57,
#but for now, we have to live with MUMPS
T0 = time.time()
optimizer = blockSQP2.SQPmethod(prob, opts, stats)
optimizer.init()
ret = optimizer.run(200)
optimizer.finish()
T1 = time.time()

t0_blockSQP2_catalyst = time.monotonic()
optimizer = blockSQP2.SQPmethod(prob, opts, stats)
optimizer.init()
ret = optimizer.run(200)
optimizer.finish()
t1_blockSQP2_catalyst = time.monotonic()
it_blockSQP2_catalyst = stats.itCount
#########################

OCprob = OCProblems.Catalyst_Mixing_OED(
                    nt = 40,
                    parallel = True,
                    N_threads = 4, 
                    )


tm1_blockSQP2_catalyst_noJIT = time.monotonic()
opts = blockSQP2.SQPoptions(
    max_QP_it = 10000,
    max_QP_secs = 20.0,
    max_conv_QPs = 4,
    conv_strategy = 'reduced_regularization',
    par_QPs = True,
    automatic_scaling = True,
)

vblocks = [blockSQP2.vblock(size, dep, impl) for size, dep, impl in zip(OCprob.vBlock_sizes, OCprob.vBlock_dependencies, OCprob.vBlock_bounds_implicit)]
cblocks = [blockSQP2.cblock(size) for size in OCprob.cBlock_sizes]
hblocks = [size for size in OCprob.hessBlock_sizes]
targets = [blockSQP2.condensing_target(*OCprob.ctarget_data)]

condenser = blockSQP2.PartialCondenser(vblocks, cblocks, hblocks, targets, 4)

prob = blockSQP2.Problemspec(OCprob.nVar, OCprob.nCon)
prob.f = OCprob.f
prob.grad_f = OCprob.grad_f
prob.g = OCprob.g

prob.make_sparse(OCprob.jac_g_nnz, OCprob.jac_g_row, OCprob.jac_g_colind)
prob.jac_g_nz = OCprob.jac_g_nz

prob.hess = OCprob.hess_lag
prob.blockIdx = OCprob.hessBlock_index
prob.set_bounds(OCprob.lb_var, OCprob.ub_var, OCprob.lb_con, OCprob.ub_con)

prob.vblocks = vblocks
prob.condenser = condenser

prob.x_start = OCprob.perturbed_start_point(1)
prob.lam_start = np.zeros(prob.nVar + prob.nCon, dtype = np.float64).reshape(-1)

stats = blockSQP2.SQPstats("./solver_outputs")

time.sleep(0.1)
t0_blockSQP2_catalyst_noJIT = time.monotonic()
optimizer = blockSQP2.SQPmethod(prob, opts, stats)
optimizer.init()
ret = optimizer.run(200)
optimizer.finish()
t1_blockSQP2_catalyst_noJIT = time.monotonic()
#########################

OCprob = OCProblems.D_Onofrio_Chemotherapy(
                    nt = 100,
                    refine = 1,
                    parallel = True,
                    N_threads = 4, 
                    )


tm1_blockSQP2_D_Onofrio = time.monotonic()
OCprob.jit(jit_hess = False)
opts = blockSQP2.SQPoptions(
    max_QP_it = 10000,
    max_QP_secs = 20.0,
    max_conv_QPs = 4,
    conv_strategy = 'reduced_regularization',
    par_QPs = True,
    automatic_scaling = True,
)

vblocks = [blockSQP2.vblock(size, dep, impl) for size, dep, impl in zip(OCprob.vBlock_sizes, OCprob.vBlock_dependencies, OCprob.vBlock_bounds_implicit)]
cblocks = [blockSQP2.cblock(size) for size in OCprob.cBlock_sizes]
hblocks = [size for size in OCprob.hessBlock_sizes]
targets = [blockSQP2.condensing_target(*OCprob.ctarget_data)]

condenser = blockSQP2.PartialCondenser(vblocks, cblocks, hblocks, targets, 4)

prob = blockSQP2.Problemspec(OCprob.nVar, OCprob.nCon)
prob.f = OCprob.f
prob.grad_f = OCprob.grad_f
prob.g = OCprob.g

prob.make_sparse(OCprob.jac_g_nnz, OCprob.jac_g_row, OCprob.jac_g_colind)
prob.jac_g_nz = OCprob.jac_g_nz

prob.hess = OCprob.hess_lag
prob.blockIdx = OCprob.hessBlock_index
prob.set_bounds(OCprob.lb_var, OCprob.ub_var, OCprob.lb_con, OCprob.ub_con)

prob.vblocks = vblocks
prob.condenser = condenser

prob.x_start = OCprob.start_point
prob.lam_start = np.zeros(prob.nVar + prob.nCon, dtype = np.float64).reshape(-1)

stats = blockSQP2.SQPstats("./solver_outputs")


t0_blockSQP2_D_Onofrio = time.monotonic()
optimizer = blockSQP2.SQPmethod(prob, opts, stats)
optimizer.init()
ret = optimizer.run(200)
optimizer.finish()
t1_blockSQP2_D_Onofrio = time.monotonic()
it_blockSQP2_D_Onofrio = stats.itCount
#########################

OCprob = OCProblems.D_Onofrio_Chemotherapy(
                    nt = 100,
                    refine = 1,
                    parallel = True,
                    N_threads = 4, 
                    )


tm1_blockSQP2_D_Onofrio_noJIT = time.monotonic()
opts = blockSQP2.SQPoptions(
    max_QP_it = 10000,
    max_QP_secs = 20.0,
    max_conv_QPs = 4,
    conv_strategy = 'reduced_regularization',
    par_QPs = True,
    automatic_scaling = True,
)

vblocks = [blockSQP2.vblock(size, dep, impl) for size, dep, impl in zip(OCprob.vBlock_sizes, OCprob.vBlock_dependencies, OCprob.vBlock_bounds_implicit)]
cblocks = [blockSQP2.cblock(size) for size in OCprob.cBlock_sizes]
hblocks = [size for size in OCprob.hessBlock_sizes]
targets = [blockSQP2.condensing_target(*OCprob.ctarget_data)]

condenser = blockSQP2.PartialCondenser(vblocks, cblocks, hblocks, targets, 4)

prob = blockSQP2.Problemspec(OCprob.nVar, OCprob.nCon)
prob.f = OCprob.f
prob.grad_f = OCprob.grad_f
prob.g = OCprob.g

prob.make_sparse(OCprob.jac_g_nnz, OCprob.jac_g_row, OCprob.jac_g_colind)
prob.jac_g_nz = OCprob.jac_g_nz

prob.hess = OCprob.hess_lag
prob.blockIdx = OCprob.hessBlock_index
prob.set_bounds(OCprob.lb_var, OCprob.ub_var, OCprob.lb_con, OCprob.ub_con)

prob.vblocks = vblocks
prob.condenser = condenser

prob.x_start = OCprob.start_point
prob.lam_start = np.zeros(prob.nVar + prob.nCon, dtype = np.float64).reshape(-1)

stats = blockSQP2.SQPstats("./solver_outputs")


t0_blockSQP2_D_Onofrio_noJIT = time.monotonic()
optimizer = blockSQP2.SQPmethod(prob, opts, stats)
optimizer.init()
ret = optimizer.run(200)
optimizer.finish()
t1_blockSQP2_D_Onofrio_noJIT = time.monotonic()
#########################


time.sleep(2)

print("\n")
print("acados - setting up catalyst mixing oed took", t0_acados_catalyst - tm1_acados_catalyst, "s")
print("acados - solving catalyst mixing oed took", t1_acados_catalyst - t0_acados_catalyst, "s and", it_acados_catalyst, "it")
print("acados - setting up D\'Onofrio took", t0_acados_D_Onofrio - tm1_acados_D_Onofrio, "s")
print("acados - solving D\'Onofrio took", t1_acados_D_Onofrio - t0_acados_D_Onofrio, "s and", it_acados_D_Onofrio, "it")
print("")
print("fatrop - setting up catalyst mixing oed took", t0_fatrop_catalyst - tm1_fatrop_catalyst, "s")
print("fatrop - solving catalyst mixing oed took", t1_fatrop_catalyst - t0_fatrop_catalyst, "s and", it_fatrop_catalyst, "it")
print("fatrop - setting up D\'Onofrio took", t0_fatrop_D_Onofrio - tm1_fatrop_D_Onofrio, "s")
print("fatrop - solving D\'Onofrio took", t1_fatrop_D_Onofrio - t0_fatrop_D_Onofrio, "s and", it_fatrop_D_Onofrio, "it")
print("")
print("blockSQP2 - setting up catalyst mixing oed took", t0_blockSQP2_catalyst - tm1_blockSQP2_catalyst, "s")
print("blockSQP2 - solving catalyst mixing oed took", t1_blockSQP2_catalyst - t0_blockSQP2_catalyst, "s and", it_blockSQP2_catalyst, "it")
print("blockSQP2 - setting up D\'Onofrio took", t0_blockSQP2_D_Onofrio - tm1_blockSQP2_D_Onofrio, "s")
print("blockSQP2 - solving D\'Onofrio took",  t1_blockSQP2_D_Onofrio - t0_blockSQP2_D_Onofrio, "s and", it_blockSQP2_D_Onofrio, "it")
print("")
print("blockSQP2 (no JIT) - setting up catalyst mixing oed took", t0_blockSQP2_catalyst_noJIT - tm1_blockSQP2_catalyst_noJIT, "s")
print("blockSQP2 (no JIT) - solving catalyst mixing oed took", t1_blockSQP2_catalyst_noJIT - t0_blockSQP2_catalyst_noJIT, "s")
print("blockSQP2 (no JIT) - setting up D\'Onofrio took", t0_blockSQP2_D_Onofrio_noJIT - tm1_blockSQP2_D_Onofrio_noJIT, "s")
print("blockSQP2 (no JIT) - solving D\'Onofrio took", t1_blockSQP2_D_Onofrio_noJIT - t0_blockSQP2_D_Onofrio_noJIT, "s")

