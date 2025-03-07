import os
import sys
import os
import sys
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
except:
    sys.path.append(os.getcwd() + "/..")

import OCProblems
import py_blockSQP
from blockSQP_pyProblem import blockSQP_pyProblem as Problemspec
import numpy as np
import casadi as cs
import copy
import time
import typing

def perturbStartPoint(OCP : OCProblems.OCProblem, IND : int, SP : np.array):
    if (isinstance(OCP, (OCProblems.Goddard_Rocket, OCProblems.Electric_Car, OCProblems.Hang_Glider, OCProblems.Fullers))):
        val = OCP.get_stage_control(SP, IND)
        OCP.set_stage_control(SP, IND, val - 0.1)
    
    #For catalyst mixing, Lotka (OED) and cushioned oscillation
    if (isinstance(OCP, (OCProblems.Lotka_Volterra_Fishing, OCProblems.Catalyst_Mixing, OCProblems.Lotka_OED, OCProblems.Cushioned_Oscillation, OCProblems.Egerstedt_Standard, OCProblems.Particle_Steering, OCProblems.Time_Optimal_Car))):
        OCP.set_stage_control(SP, IND, 0.1)
    
    #For hanging chain
    if (isinstance(OCP, (OCProblems.Hanging_Chain, OCProblems.Van_der_Pol_Oscillator_3))):
        val = OCP.get_stage_control(SP, IND)
        OCP.set_stage_control(SP, IND, val + 0.1)
    
        #For batch reactor
    if (isinstance(OCP, OCProblems.Batch_Reactor)):
        OCP.set_stage_control(SP, IND, 300.)

    #For batch distillation
    if (isinstance(OCP, OCProblems.Batch_Distillation)):
        OCP.set_stage_control(SP, IND, 1.5)
    
    if (isinstance(OCP, OCProblems.Three_Tank_Multimode)):
        OCP.set_stage_control(SP, IND, [0.5, 0.25, 0.25])


def create_prob_cond(OCprob : OCProblems.OCProblem):
    vBlocks = py_blockSQP.vblock_array(len(OCprob.vBlock_sizes))
    cBlocks = py_blockSQP.cblock_array(len(OCprob.cBlock_sizes))
    hBlocks = py_blockSQP.int_array(len(OCprob.hessBlock_sizes))
    targets = py_blockSQP.condensing_targets(1)
    for i in range(len(OCprob.vBlock_sizes)):
        vBlocks[i] = py_blockSQP.vblock(OCprob.vBlock_sizes[i], OCprob.vBlock_dependencies[i])
    for i in range(len(OCprob.cBlock_sizes)):
        cBlocks[i] = py_blockSQP.cblock(OCprob.cBlock_sizes[i])
    for i in range(len(OCprob.hessBlock_sizes)):
        hBlocks[i] = OCprob.hessBlock_sizes[i]
    targets[0] = py_blockSQP.condensing_target(*OCprob.ctarget_data)
    HOLD = [vBlocks, cBlocks, hBlocks, targets]
    
    cond = py_blockSQP.Condenser(vBlocks, cBlocks, hBlocks, targets)
    
    
    prob = Problemspec()
    prob.x_start = OCprob.start_point
    
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
    prob.lam_start = np.zeros(prob.nVar + prob.nCon, dtype = np.float64).reshape(-1)
    prob.vblocks = vBlocks
    return prob, cond, HOLD




def perturbed_starts(OCprob : OCProblems.OCProblem, opts : py_blockSQP.SQPoptions, nPert0, nPertF, COND = False, itMax = 100):
    N_SQP = []
    N_secs = []
    for j in range(nPert0,nPertF):
        start_it = copy.copy(OCprob.start_point)
        perturbStartPoint(OCprob, j, start_it)
        prob, cond, HOLD = create_prob_cond(OCprob)
        prob.x_start = start_it
        prob.complete()

        stats = py_blockSQP.SQPstats("./solver_outputs")
        if not COND:
            optimizer = py_blockSQP.SQPmethod(prob, opts, stats)
        else:
            optimizer = py_blockSQP.SCQPmethod(prob, opts, stats, cond)
        optimizer.init()
        t0 = time.time()
        ret = optimizer.run(itMax)
        t1 = time.time()
        
        if int(ret) > 0:
            N_SQP.append(stats.itCount)
            N_secs.append(t1 - t0)
        else:
            N_SQP.append(0.00001)
            N_secs.append(0.00001)
    return N_SQP, N_secs
    
    