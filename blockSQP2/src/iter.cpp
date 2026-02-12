/*
 * blockSQP -- Sequential quadratic programming for problems with
 *             block-diagonal Hessian matrix.
 * Copyright (C) 2012-2015 by Dennis Janka <dennis.janka@iwr.uni-heidelberg.de>
 *
 * Licensed under the zlib license. See LICENSE for more details.
 */

/*
 * blockSQP 2 -- Condensing, convexification strategies, scaling heuristics and more
 *               for blockSQP, the nonlinear programming solver by Dennis Janka.
 * Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
 * 
 * Licensed under the zlib license. See LICENSE for more details.
 */
 
/**
 * \file blocksqp_iterate.hpp
 * \author Dennis Janka
 * \date 2012-2015
 *
 *  Implementation of SQPiterate class that holds all variables that are
 *  updated during one SQP iteration.
 * 
 * \modifications
 *  \author Reinhold Wittmann
 *  \date 2023-2025
 */

 
#include <blockSQP2/options.hpp>
#include <blockSQP2/stats.hpp>
#include <blockSQP2/iterate.hpp>
#include <stdexcept>

namespace blockSQP2{


SQPiterate::SQPiterate(Problemspec* prob, const SQPoptions* param){
    ///Allocate memory for variables that are updated during optimization:
    // current iterate
    xi.Dimension(prob->nVar).Initialize(0.0);

    // dual variables (for general constraints and variable bounds)
    lambda.Dimension(prob->nVar + prob->nCon).Initialize(0.0);

    // constraint vector with lower and upper bounds
    // (Box constraints are not included in the constraint list)
    constr.Dimension(prob->nCon).Initialize(0.0);

    // gradient of objective
    gradObj.Dimension( prob->nVar ).Initialize( 0.0 );

    // gradient of Lagrangian
    gradLagrange.Dimension( prob->nVar ).Initialize( 0.0 );

    ///Allocate constraint jacobian and hessian approximation, either as dense or sparse matrices
    if (!param->sparse) constrJac.Dimension(prob->nCon, prob->nVar).Initialize(0.0);
    else sparse_constrJac.Dimension(prob->nCon, prob->nVar, prob->nnz);
    
    //Allocate Hessian data
    int maxblocksize;
    // Set nBlocks structure according to if we use block updates or not
    if (param->block_hess == 0 || prob->nBlocks <= 1){
        nBlocks = 1;
        blockIdx = std::make_unique<int[]>(2);
        blockIdx[0] = 0;
        blockIdx[1] = prob->nVar;
        maxblocksize = prob->nVar;
    }
    else if (param->block_hess == 2 && prob->nBlocks > 1){
        // hybrid strategy: 1 block for constraints, 1 for objective
        nBlocks = 2;
        blockIdx = std::make_unique<int[]>(3);
        blockIdx[0] = 0;
        blockIdx[1] = prob->blockIdx[prob->nBlocks-1];
        blockIdx[2] = prob->nVar;
        maxblocksize = std::max(blockIdx[1], blockIdx[2] - blockIdx[1]);
    }
    else{   
        nBlocks = prob->nBlocks;
        blockIdx = std::make_unique<int[]>(nBlocks+1);
        blockIdx[0] = 0;
        maxblocksize = 0;
        for(int iBlock = 0; iBlock < nBlocks; iBlock++){
            blockIdx[iBlock+1] = prob->blockIdx[iBlock+1];
            if (blockIdx[iBlock+1] - blockIdx[iBlock] > maxblocksize) maxblocksize = blockIdx[iBlock+1] - blockIdx[iBlock];
        }
    }
    
    last_block_exact = (param->last_block_approx == Hessians::exact || param->last_block_approx == Hessians::pos_def_exact);
    last_fallback_exact = (param->last_block_approx == Hessians::pos_def_exact);
    
    // Create one Matrix for one diagonal block in the Hessian
    int Bsize;
    hess1 = std::make_unique<SymMatrix[]>(nBlocks);
    for (int iBlock = 0; iBlock < nBlocks; iBlock++){
        Bsize = blockIdx[iBlock+1] - blockIdx[iBlock];
        hess1[iBlock].Dimension(Bsize).Initialize(0.0);
    }
    
    // For SR1 or finite differences, maintain two Hessians
    // if (param->exact_hess > 0 || param->hess_approx == Hessians::SR1 || param->hess_approx == Hessians::finite_diff || param->hess_approx == Hessians::undamped_BFGS){
    if (is_indefinite(param->hess_approx) || is_indefinite(param->last_block_approx)){
        hess2 = std::make_unique<SymMatrix[]>(nBlocks);
        for (int iBlock = 0; iBlock < nBlocks; iBlock++){
            Bsize = blockIdx[iBlock + 1] - blockIdx[iBlock];
            hess2[iBlock].Dimension(Bsize).Initialize(0.0);
        }
    }
    
    hess_conv = std::make_unique<SymMatrix[]>(nBlocks);
    for (int iBlock = 0; iBlock < nBlocks; iBlock++){
        Bsize = blockIdx[iBlock+1] - blockIdx[iBlock];
        hess_conv[iBlock].Dimension(Bsize).Initialize(0.0);
    }
    //Initialize current Hessian pointer to first Hessian
    hess = hess1.get();
    
    ///Allocate additional variables needed by the algorithm
    int nVar = prob->nVar;
    int nCon = prob->nCon;
    
    //Allocate space for one more delta-gamma pair than mem_size so we don't overwrite the oldest pair directly after a successful QP solve.
    //The linesearch may still fall back to the convex QP, which may require calculating the limited-memory fallback Hessian, which starts at the oldest step.
    dg_nsave = std::max(std::max(int(param->lim_mem)*param->mem_size + 1, int(param->automatic_scaling)*5), 1);
    dg_pos = -1;
    
    deltaMat.Dimension(nVar, dg_nsave).Initialize(0.0);
    gammaMat.Dimension(nVar, dg_nsave).Initialize(0.0);
    deltaNormSqMat.Dimension(nBlocks, dg_nsave).Initialize(0.0);
    deltaGammaMat.Dimension(nBlocks, dg_nsave).Initialize(0.0);
    
    deltaXi.Submatrix( deltaMat, nVar, 1, 0, 0 );
    gamma.Submatrix(gammaMat, nVar, 1, 0, 0);
    
    // For selective sizing: for each block save sTs, sTs_, sTy, sTy_
    deltaNormSqOld.Dimension(nBlocks).Initialize(1.0);
    deltaOld.Dimension(prob->nVar).Initialize(0.0);
    deltaGammaOld.Dimension(nBlocks).Initialize(0.0);
    deltaGammaOldFallback.Dimension(nBlocks).Initialize(0.0);
    
    
    AdeltaXi.Dimension( nCon ).Initialize( 0.0 );
    lambdaQP.Dimension( nVar+nCon ).Initialize( 0.0 );
    trialXi.Dimension( nVar, 1, nVar ).Initialize( 0.0 );
    trialLambda.Dimension(nVar + nCon, 1, nVar + nCon).Initialize(0.0);
    trialConstr.Dimension(nCon, 1).Initialize(0.0);
    
    // bounds for step (sub QP)
    delta_lb_var.Dimension(nVar).Initialize(0.0);
    delta_ub_var.Dimension(nVar).Initialize(0.0);
    
    delta_lb_con.Dimension(nCon).Initialize(0.0);
    delta_ub_con.Dimension(nCon).Initialize(0.0);
    
    // Miscellaneous counters
    nquasi = std::unique_ptr<int[]>(new int[nBlocks]());
    noUpdateCounter = std::make_unique<int[]>(nBlocks);
    for (int iBlock = 0; iBlock < nBlocks; iBlock++) noUpdateCounter[iBlock] = -1;
    nRestIt = 0;
    remaining_filter_overrides = param->max_filter_overrides;
    
    // Flags
    conv_qp_only = param->indef_local_only;
    conv_qp_solved = false;
    hess2_updated = true;
    use_homotopy = true;            
    
    KKT_heuristic_enabled = true;
    KKTerror_save = param->inf; 
    nearSol = false;
    milestone = param->inf;
    solution_found = false;
    n_extra = 0;
    
    // Convexification strategy
    hess_num_accepted = 0;
    QP_num_accepted = 0;
    convKappa = param->conv_kappa_0;
    if (param->max_conv_QPs > 1){
        deltaXi_conv.Dimension(prob->nVar);
        lambdaQP_conv.Dimension(prob->nVar + prob->nCon);
    }
    
    // Scaling heuristic
    if (param->automatic_scaling){
        rescaleFactors = std::make_unique<double[]>(prob->nVar);
        vfreeScale = 1.0;
        scaled_prob = static_cast<scaled_Problemspec*>(prob);
        scaleFactors_save = std::make_unique<double[]>(prob->nVar);
    }
    else scaled_prob = nullptr;
    n_scaleIt = 0;
    
    //Derived from parameters
    modified_hess_regularizationFactor = param->reg_factor;
    
    cNormOpt_save = param->inf;
    cNormSOpt_save = param->inf;
    
    N_QP_cancels = 0;
    if (param->par_QPs){
        par_QP_sols_prim = std::make_unique<Matrix[]>(param->max_conv_QPs + 1);
        par_QP_sols_dual = std::make_unique<Matrix[]>(param->max_conv_QPs + 1);
        for (int j = 0; j < param->max_conv_QPs + 1; j++){
            par_QP_sols_prim[j].Dimension(prob->nVar);
            par_QP_sols_dual[j].Dimension(prob->nVar + prob->nCon);
        }
    }
    
    it_saved = false;
}

SQPiterate::SQPiterate(){}

SQPiterate::SQPiterate( const SQPiterate &iter ){

    nBlocks = iter.nBlocks;
    blockIdx = std::make_unique<int[]>(nBlocks + 1);
    for (int i = 0; i < nBlocks + 1; i++)
        blockIdx[i] = iter.blockIdx[i];

    xi = iter.xi;
    lambda = iter.lambda;
    constr = iter.constr;
    gradObj = iter.gradObj;
    gradLagrange = iter.gradLagrange;

    constrJac = iter.constrJac;
    sparse_constrJac = iter.sparse_constrJac;
        
    hess = nullptr;
}


void SQPiterate::initIterate( SQPoptions* param )
{
    alpha = 1.0;
    nSOCS = 0;
    reducedStepCount = 0;
    steptype = 0;
    n_id_hess = 0;

    obj = param->inf;
    cNorm = param->thetaMax;
    cNormS = param->thetaMax;
    gradNorm = param->inf;
    lambdaStepNorm = 0.0;
    tol = param->inf;
}

SQPiterate::~SQPiterate(void){}


void SQPiterate::save_iterate(){
    xiOpt_save = xi;
    lambdaOpt_save = lambda;
    objOpt_save = obj;
    constrOpt_save = constr;
    tolOpt_save = tol;
    cNormOpt_save = cNorm;
    cNormSOpt_save = cNormS;
    if (scaled_prob){
        std::copy(scaled_prob->scaling_factors.get(), scaled_prob->scaling_factors.get() + xi.m, scaleFactors_save.get());
        // for (int i = 0; i < xi.m; i++){
        //     scaleFactors_save[i] = scaled_prob->scaling_factors[i];
        // }
    }
    it_saved = true;
}

void SQPiterate::restore_iterate(){
    xi = xiOpt_save;
    lambda = lambdaOpt_save;
    obj = objOpt_save;
    constr = constrOpt_save;
    tol = tolOpt_save;
    cNorm = cNormOpt_save;
    cNormS = cNormSOpt_save;
    if (scaled_prob){
        scaled_prob->set_scale(scaleFactors_save.get());
    }
}


} // namespace blockSQP2
