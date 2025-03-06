/*
 * blockSQP -- Sequential quadratic programming for problems with
 *             block-diagonal Hessian matrix.
 * Copyright (C) 2012-2015 by Dennis Janka <dennis.janka@iwr.uni-heidelberg.de>
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
 */

#include "blocksqp_options.hpp"
#include "blocksqp_stats.hpp"
#include "blocksqp_iterate.hpp"
#include <stdexcept>

namespace blockSQP{


SQPiterate::SQPiterate(const Problemspec* prob, const SQPoptions* param, bool full){

    int maxblocksize;

    // Set nBlocks structure according to if we use block updates or not
    if (param->blockHess == 0 || prob->nBlocks == 1){
        nBlocks = 1;
        blockIdx = new int[2];
        blockIdx[0] = 0;
        blockIdx[1] = prob->nVar;
        maxblocksize = prob->nVar;
        //param->whichSecondDerv = 0;
    }
    else if (param->blockHess == 2 && prob->nBlocks > 1){
        // hybrid strategy: 1 block for constraints, 1 for objective
        nBlocks = 2;
        blockIdx = new int[3];
        blockIdx[0] = 0;
        blockIdx[1] = prob->blockIdx[prob->nBlocks-1];
        blockIdx[2] = prob->nVar;
        maxblocksize = std::max(blockIdx[1], blockIdx[2] - blockIdx[1]);
    }
    else{   
        nBlocks = prob->nBlocks;
        blockIdx = new int[nBlocks+1];
        blockIdx[0] = 0;
        maxblocksize = 0;
        for(int iBlock = 0; iBlock < nBlocks; iBlock++){
            blockIdx[iBlock+1] = prob->blockIdx[iBlock+1];
            if (blockIdx[iBlock+1] - blockIdx[iBlock] > maxblocksize) maxblocksize = blockIdx[iBlock+1] - blockIdx[iBlock];
        }
    }

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
    if( !param->sparseQP ){
        constrJac.Dimension( prob->nCon, prob->nVar ).Initialize( 0.0 );
        jacNz = nullptr;
        jacIndRow = nullptr;
        jacIndCol = nullptr;
    }
    else{
        jacNz = new double[prob->nnz];
        jacIndRow = new int[prob->nnz];
        jacIndCol = new int[prob->nVar + 1];
    }

    hess = nullptr;
    hess1 = nullptr;
    hess2 = nullptr;
    hess_alt = nullptr;

    noUpdateCounter = nullptr;
    nquasi = nullptr;
    nRestIt = 0;

    modified_hess_regularizationFactor = param->hess_regularizationFactor;
    nearSol = false;
    milestone = std::numeric_limits<double>::infinity();
    convKappa = param->convKappa0;
    conv_qp_only = param->indef_local_only;

    step_heuristic_active = true;
    n_extra = 0;
    sol_found = false;
    tol_save = std::numeric_limits<double>::infinity();
    cNormOpt_save = std::numeric_limits<double>::infinity();
    cNormSOpt_save = std::numeric_limits<double>::infinity();

    if (full){
        ///Allocate block hessian and fallback block hessian if needed
        int iBlock, varDim;

        // Create one Matrix for one diagonal block in the Hessian
        hess1 = new SymMatrix[nBlocks];
        for( iBlock=0; iBlock<nBlocks; iBlock++ )
        {
            varDim = blockIdx[iBlock+1] - blockIdx[iBlock];
            hess1[iBlock].Dimension( varDim ).Initialize( 0.0 );
        }

        // For SR1 or finite differences, maintain two Hessians
        if (param->hessUpdate == 1 || param->hessUpdate == 4 || param->hessUpdate == 6 || param->hessUpdate > 6){
            hess2 = new SymMatrix[nBlocks];
            for (iBlock = 0; iBlock < nBlocks; iBlock++){
                varDim = blockIdx[iBlock + 1] - blockIdx[iBlock];
                hess2[iBlock].Dimension(varDim).Initialize(0.0);
            }
        }

        hess_alt = new SymMatrix[nBlocks];
        for (iBlock = 0; iBlock < nBlocks; iBlock++){
            varDim = blockIdx[iBlock+1] - blockIdx[iBlock];
            hess_alt[iBlock].Dimension( varDim ).Initialize(0.0);
        }

        // Set Hessian pointer
        hess = hess1;

        ///Allocate additional variables needed by the algorithm
        int nVar = prob->nVar;
        int nCon = prob->nCon;

        //Allocate space for one more delta-gamma pair than hessMemsize so we don't overwrite the oldest pair directly after a successful QP solve.
        //The linesearch may still fall back to the convex QP, which may require calculating the limited-memory fallback Hessian, which starts at the oldest step.
        dg_nsave = std::max(std::max(int(param->hessLimMem)*param->hessMemsize + 1, int(param->autoScaling)*5), 1);
        deltaMat.Dimension(nVar, dg_nsave).Initialize(0.0);
        gammaMat.Dimension(nVar, dg_nsave).Initialize(0.0);
        deltaNormSqMat.Dimension(nBlocks, dg_nsave).Initialize(0.0);
        deltaGammaMat.Dimension(nBlocks, dg_nsave).Initialize(0.0);

        deltaXi.Submatrix( deltaMat, nVar, 1, 0, 0 );

        // trial step (temporary variable, for line search)
        trialXi.Dimension( nVar, 1, nVar ).Initialize( 0.0 );

        // Constraint function values at trial point
        trialConstr.Dimension(nCon, 1).Initialize(0.0);

        // bounds for step (QP subproblem)
        delta_lb_var.Dimension(nVar).Initialize(0.0);
        delta_ub_var.Dimension(nVar).Initialize(0.0);

        delta_lb_con.Dimension(nCon).Initialize(0.0);
        delta_ub_con.Dimension(nCon).Initialize(0.0);

        // product of constraint Jacobian with step (deltaXi)
        AdeltaXi.Dimension( nCon ).Initialize( 0.0 );

        // dual variables of QP (simple bounds and general constraints)
        lambdaQP.Dimension( nVar+nCon ).Initialize( 0.0 );

        // filter as a set of pairs
        filter = new std::set<std::pair<double,double>>;

        gamma.Submatrix(gammaMat, nVar, 1, 0, 0);

        // Scalars that are used in various Hessian update procedures
        noUpdateCounter = new int[nBlocks];
        for (iBlock = 0; iBlock < nBlocks; iBlock++)
            noUpdateCounter[iBlock] = -1;

        nquasi = new int[nBlocks]();
        dg_pos = -1;

        // For selective sizing: for each block save sTs, sTs_, sTy, sTy_
        deltaNormSqOld.Dimension(nBlocks).Initialize( 1.0 );
        deltaOld.Dimension(prob->nVar).Initialize(0.0);
        deltaGammaOld.Dimension(nBlocks).Initialize( 0.0 );

        deltaGammaOldFallback.Dimension(nBlocks).Initialize(0.0);

        use_homotopy = true;
        local_lenience = std::max(param->max_local_lenience, 0);

        if (param->autoScaling){
            rescaleFactors = new double[prob->nVar];
            vfreeScale = 1.0;
        }
        else{
            rescaleFactors = nullptr;
        }
        n_scaleIt = 0;
    }
}

SQPiterate::SQPiterate(){}

SQPiterate::SQPiterate( const SQPiterate &iter ){

    nBlocks = iter.nBlocks;
    blockIdx = new int[nBlocks+1];
    for (int i = 0; i < nBlocks + 1; i++)
        blockIdx[i] = iter.blockIdx[i];

    xi = iter.xi;
    lambda = iter.lambda;
    constr = iter.constr;
    gradObj = iter.gradObj;
    gradLagrange = iter.gradLagrange;

    constrJac = iter.constrJac;
    if (iter.jacNz != nullptr){

        int nVar = xi.M();
        int nnz = iter.jacIndCol[nVar];

        jacNz = new double[nnz];
        for(int i = 0; i < nnz; i++)
            jacNz[i] = iter.jacNz[i];

        jacIndRow = new int[nnz];
        for (int i = 0; i < nnz; i++)
            jacIndRow[i] = iter.jacIndRow[i];

        jacIndCol = new int[nVar + 1];
        for (int i = 0; i <= nVar; i++)
            jacIndCol[i] = iter.jacIndCol[i];
    }
    else{
        jacNz = nullptr;
        jacIndRow = nullptr;
        jacIndCol = nullptr;
    }

    noUpdateCounter = nullptr;
    hess = nullptr;
    hess1 = nullptr;
    hess2 = nullptr;
}


void SQPiterate::initIterate( SQPoptions* param )
{
    alpha = 1.0;
    nSOCS = 0;
    reducedStepCount = 0;
    steptype = 0;
    n_id_hess = 0;

    obj = param->inf;
    tol = param->inf;
    cNorm = param->thetaMax;
    gradNorm = param->inf;
    lambdaStepNorm = 0.0;
}

SQPiterate::~SQPiterate(void){
    delete[] blockIdx;
    delete[] noUpdateCounter;
    delete[] jacNz;
    delete[] jacIndRow;
    delete[] jacIndCol;

    delete filter;
    delete[] hess1;
    delete[] hess2;
    delete[] hess_alt;

    delete[] nquasi;
    delete[] rescaleFactors;
}

//TODO: Store scaling factors as well because they may change
void SQPiterate::save_iterate(){
    xiOpt_save = xi;
    lambdaOpt_save = lambda;
    objOpt_save = obj;
    constrOpt_save = constr;
    tolOpt_save = tol;
    cNormOpt_save = cNorm;
    cNormSOpt_save = cNormS;
    return;
}

void SQPiterate::restore_iterate(){
    xi = xiOpt_save;
    lambda = lambdaOpt_save;
    obj = objOpt_save;
    constr = constrOpt_save;
    tol = tolOpt_save;
    cNorm = cNormOpt_save;
    cNormS = cNormSOpt_save;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

SCQPiterate::SCQPiterate(Problemspec* prob, SQPoptions* param, Condenser *cond, bool full):
    SQPiterate(prob, param, full)
{
    //Wrap sparse jacobian array
    Jacobian = Sparse_Matrix(prob->nCon, prob->nVar, prob->nnz, jacNz, jacIndRow, jacIndCol);

    //Allocate solution of condensed QP
    deltaXi_cond.Dimension(cond->condensed_num_vars);
    lambdaQP_cond.Dimension(cond->condensed_num_vars + cond->condensed_num_cons);

    //condensed_hess = nullptr;
    condensed_hess = new SymMatrix[cond->condensed_num_hessblocks];
    condensed_hess_nz = nullptr;
    condensed_hess_row = nullptr;
    condensed_hess_colind = nullptr;
    condensed_hess_loind = nullptr;

    //condensed_hess_2 = nullptr;
    condensed_hess_2 = new SymMatrix[cond->condensed_num_hessblocks];
}


SCQPiterate::~SCQPiterate(){
    Jacobian.nz = nullptr;
    Jacobian.row = nullptr;
    Jacobian.colind = nullptr;

    delete[] condensed_hess;
    delete[] condensed_hess_2;
}




SCQP_correction_iterate::SCQP_correction_iterate(Problemspec* prob, SQPoptions* param, Condenser* cond, bool full): SCQPiterate(prob, param, cond, full){
    corrected_h.Dimension(cond->condensed_num_vars);
    corrected_lb_con.Dimension(cond->condensed_num_vars);
    corrected_ub_con.Dimension(cond->condensed_num_vars);

    deltaXi_save.Dimension(prob->nVar);
    lambdaQP_save.Dimension(prob->nVar + prob->nCon);
}



} // namespace blockSQP
