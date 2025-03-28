/*
 * blockSQP -- Sequential quadratic programming for problems with
 *             block-diagonal Hessian matrix.
 * Copyright (C) 2012-2015 by Dennis Janka <dennis.janka@iwr.uni-heidelberg.de>
 *
 * Licensed under the zlib license. See LICENSE for more details.
 */

/**
 * \file blocksqp_hess.cpp
 * \author Dennis Janka
 * \date 2012-2015
 *
 *  Implementation of methods of SQPmethod class associated with
 *  computation of Hessian approximations.
 *
 */

#include "blocksqp_iterate.hpp"
#include "blocksqp_options.hpp"
#include "blocksqp_stats.hpp"
#include "blocksqp_method.hpp"
#include "blocksqp_general_purpose.hpp"
#include <iostream>
#include <fstream>

namespace blockSQP
{

/**
 * Initial Hessian: Identity matrix
 */
void SQPmethod::calcInitialHessian(SymMatrix *hess){
    for (int iBlock=0; iBlock<vars->nBlocks; iBlock++ )
        //if objective derv is computed exactly, don't set the last block!
        if (!(param->whichSecondDerv == 1 && param->blockHess && iBlock == vars->nBlocks-1)){
            hess[iBlock].Initialize(0.0);
            for (int i=0; i<hess[iBlock].m; i++)
                hess[iBlock]( i, i ) = param->iniHessDiag;
        }
}


/**
 * Initial Hessian for one block: Identity matrix
 */
void SQPmethod::calcInitialHessian(int iBlock, SymMatrix *hess){
    hess[iBlock].Initialize(0.0);
    // Each block is a diagonal matrix
    for (int i = 0; i < hess[iBlock].m; i++)
        hess[iBlock](i, i) = param->iniHessDiag;
}

void SQPmethod::calcInitialHessians(){
    calcInitialHessian(vars->hess1);
    if (vars->hess2 != nullptr) calcInitialHessian(vars->hess2);
}


void SQPmethod::calcScaledInitialHessian(double scale, SymMatrix *hess){
    for (int iBlock = 0; iBlock < vars->nBlocks; iBlock++)
        //if objective derv is computed exactly, don't set the last block!
        if (!(param->whichSecondDerv == 1 && param->blockHess && iBlock == vars->nBlocks-1)){
            hess[iBlock].Initialize(0.0);
            for (int i = 0; i < hess[iBlock].m; i++)
                hess[iBlock]( i, i ) = scale;
        }
}

void SQPmethod::calcScaledInitialHessian(int iBlock, double scale, SymMatrix *hess){
    hess[iBlock].Initialize( 0.0 );
    // Each block is a diagonal matrix
    for (int i = 0; i < hess[iBlock].m; i++)
        hess[iBlock](i, i) = scale;
}




void SQPmethod::resetHessian(SymMatrix *hess){
    for( int iBlock=0; iBlock<vars->nBlocks; iBlock++ )
        //if objective derv is computed exactly, don't set the last block!
        if( !(param->whichSecondDerv == 1 && param->blockHess && iBlock == vars->nBlocks - 1) ){
            vars->noUpdateCounter[iBlock] = -1;
            vars->nquasi[iBlock] = 0;
            calcInitialHessian(iBlock, hess);
        }
    //Don't use homotopy since changing the hessian to (scaled) identity can drastically change the QP
    vars->use_homotopy = false;
}


void SQPmethod::resetHessian(int iBlock, SymMatrix *hess){
    vars->noUpdateCounter[iBlock] = -1;
    vars->nquasi[iBlock] = 0;
    calcInitialHessian(iBlock, hess);

    if (hess == vars->hess1){
        vars->deltaNormSqOld( iBlock ) = 1.0;
        vars->deltaGammaOld( iBlock ) = 0.0;
    }
    else{
        //vars->deltaNormSqOldFallback( iBlock ) = 1.0;
        vars->deltaGammaOldFallback( iBlock ) = 0.0;
    }
}

void SQPmethod::resetHessians(){
    resetHessian(vars->hess1);
    if (vars->hess2 != nullptr) resetHessian(vars->hess2);
    return;
}

/**
 * Approximate Hessian by finite differences
 */
int SQPmethod::calcFiniteDiffHessian(SymMatrix *hess)
{
    int iVar, jVar, k, iBlock, maxBlock, info, idx, idx1, idx2;
    double dummy, lowerVio, upperVio;
    Matrix pert;
    SQPiterate varsP = SQPiterate( *vars );

    const double myDelta = 1.0e-4;
    const double minDelta = 1.0e-6;

    pert.Dimension(prob->nVar);

    info = 0;

    // Find out the largest block
    maxBlock = 0;
    for( iBlock=0; iBlock<vars->nBlocks; iBlock++ )
        if( vars->blockIdx[iBlock+1] - vars->blockIdx[iBlock] > maxBlock )
            maxBlock = vars->blockIdx[iBlock+1] - vars->blockIdx[iBlock];

    // Compute original Lagrange gradient
    calcLagrangeGradient( vars->lambda, vars->gradObj, vars->jacNz, vars->jacIndRow, vars->jacIndCol, vars->gradLagrange, 0 );

    for( iVar = 0; iVar<maxBlock; iVar++ )
    {
        pert.Initialize( 0.0 );

        // Perturb all blocks simultaneously
        for( iBlock=0; iBlock<vars->nBlocks; iBlock++ )
        {
            idx = vars->blockIdx[iBlock] + iVar;
            // Skip blocks that have less than iVar variables
            if( idx < vars->blockIdx[iBlock+1] )
            {
                pert( idx ) = myDelta * fabs( vars->xi( idx ) );
                pert( idx ) = fmax( pert( idx ), minDelta );

                // If perturbation violates upper bound, try to perturb with negative
                upperVio = vars->xi( idx ) + pert( idx ) - prob->ub_var( idx );
                if( upperVio > 0 )
                {
                    lowerVio = prob->lb_var( idx ) -  ( vars->xi( idx ) - pert( idx ) );
                    // If perturbation violates also lower bound, take the largest perturbation possible
                    if( lowerVio > 0 )
                    {
                        if( lowerVio > upperVio )
                            pert( idx ) = -lowerVio;
                        else
                            pert( idx ) = upperVio;
                    }
                    // If perturbation does not violate lower bound, take -computed perturbation
                    else
                    {
                        pert( idx ) = -pert( idx );
                    }
                }
            }
        }

        // Add perturbation
        for( k=0; k<prob->nVar; k++ )
            vars->xi( k ) += pert( k );

        // Compute perturbed Lagrange gradient
        if( param->sparseQP )
        {
            prob->evaluate( vars->xi, vars->lambda, &dummy, varsP.constr, varsP.gradObj,
                            varsP.jacNz, varsP.jacIndRow, varsP.jacIndCol, hess, 1, &info );
            calcLagrangeGradient( vars->lambda, varsP.gradObj, varsP.jacNz, varsP.jacIndRow,
                                  varsP.jacIndCol, varsP.gradLagrange, 0 );
        }
        else
        {
            prob->evaluate( vars->xi, vars->lambda, &dummy, varsP.constr, varsP.gradObj, varsP.constrJac, hess, 1, &info );
            calcLagrangeGradient( vars->lambda, varsP.gradObj, varsP.constrJac, varsP.gradLagrange, 0 );
        }

        // Compute finite difference approximations: one column in every block
        for( iBlock=0; iBlock<vars->nBlocks; iBlock++ )
        {
            idx1 = vars->blockIdx[iBlock] + iVar;
            // Skip blocks that have less than iVar variables
            if( idx1 < vars->blockIdx[iBlock+1] )
            {
                for( jVar=iVar; jVar<vars->blockIdx[iBlock+1]-vars->blockIdx[iBlock]; jVar++ )
                {// Take symmetrized matrices
                    idx2 = vars->blockIdx[iBlock] + jVar;
                    hess[iBlock]( iVar, jVar ) =  ( varsP.gradLagrange( idx1 ) - vars->gradLagrange( idx2 ) );
                    hess[iBlock]( iVar, jVar ) += ( varsP.gradLagrange( idx2 ) - vars->gradLagrange( idx1 ) );
                    hess[iBlock]( iVar, jVar ) *= 0.5 / pert( idx1 );
                }
            }
        }

        // Subtract perturbation
        for( k=0; k<prob->nVar; k++ )
            vars->xi( k ) -= pert( k );
    }

    return info;
}


//void SQPmethod::sizeInitialHessian( const Matrix &gamma, const Matrix &delta, int iBlock, int option, SymMatrix *hess)
void SQPmethod::sizeInitialHessian(int dpos, int iBlock, SymMatrix *hess, int option){
    int i, j;
    double scale;
    double myEps = 1.0e3 * param->eps;
    Matrix gamma;

    //TODO: Consider adding condition l1VectorNorm(delta) > tol

    switch (option){
        case 1: //Shanno-Phua
            //scale = adotb(gamma, gamma) / fmax(adotb(delta, gamma)*param->iniHessDiag, myEps);
            gamma.Submatrix(vars->gammaMat, vars->blockIdx[iBlock+1] - vars->blockIdx[iBlock], 1, vars->blockIdx[iBlock], dpos);
            scale = adotb(gamma, gamma) / fmax(vars->deltaGammaMat(iBlock, dpos)*param->iniHessDiag, myEps);
            break;
        case 2: //Oren-Luenberger
            //scale = adotb(delta, gamma) / fmax(adotb(delta, delta)*param->iniHessDiag, myEps);
            scale = vars->deltaGammaMat(iBlock, dpos) / fmax(vars->deltaNormSqMat(iBlock, dpos)*param->iniHessDiag, myEps);
            if (scale < 0) scale *= -1;
            scale = fmin(scale, 1.0);
            break;
        case 3: //Geometric mean of 1 and 2
            //scale = sqrt(adotb(gamma, gamma)/fmax(adotb(delta, delta)*param->iniHessDiag, myEps));
            gamma.Submatrix(vars->gammaMat, vars->blockIdx[iBlock+1] - vars->blockIdx[iBlock], 1, vars->blockIdx[iBlock], dpos);
            scale = sqrt(adotb(gamma, gamma)/fmax(vars->deltaNormSqMat(iBlock, dpos)*param->iniHessDiag, myEps));
            break;
        case 4: //First COL sizing, = OL sizing with different bounds
            //scale = adotb(delta, gamma) / fmax(adotb(delta, delta)*param->iniHessDiag, myEps);
            scale = vars->deltaGammaMat(iBlock, dpos) / fmax(vars->deltaNormSqMat(iBlock, dpos)*param->iniHessDiag, myEps);
            if (scale < 0) scale *= -1;
            scale = fmax(fmin(scale, 1.0), param->olEps);
            break;
        default:
            return;
    }

    scale = fmax(scale, myEps);

    for (i = 0; i < hess[iBlock].m; i++){
        for (j = i; j < hess[iBlock].m; j++){
            hess[iBlock](i,j) *= scale;
        }
    }
    // statistics: average sizing factor
    stats->averageSizingFactor += scale;
    return;
}


//void SQPmethod::sizeHessianCOL(const Matrix &gamma, const Matrix &delta, const double deltaNormSq, const double deltaNormSqOld, const double deltaGamma, const double deltaGammaOld, int iBlock, SymMatrix *hess)
void SQPmethod::sizeHessianCOL(int dpos, int iBlock, SymMatrix *hess){
    int Bsize = vars->blockIdx[iBlock + 1] - vars->blockIdx[iBlock];
    double theta, scale, myEps = 1.0e3 * param->eps;
    double deltaNormSq, deltaNormSqOld, deltaGamma, deltaGammaOld, deltaBdelta;
    Matrix delta;

    // Get sTs, sTs_, sTy, sTy_ (precalculated) and sTBs
    delta.Submatrix(vars->deltaMat, Bsize, 1, vars->blockIdx[iBlock], dpos);

    deltaNormSq = vars->deltaNormSqMat(iBlock, dpos);
    deltaNormSqOld = vars->deltaNormSqOld(iBlock);
    deltaGamma = vars->deltaGammaMat(iBlock, dpos);
    if (hess == vars->hess1) deltaGammaOld = vars->deltaGammaOld(iBlock);
    else deltaGammaOld = vars->deltaGammaOldFallback(iBlock);

    deltaBdelta = 0.0;
    for (int i = 0; i < Bsize; i++){
        for (int j = 0; j < Bsize; j++){
            deltaBdelta += delta(i) * hess[iBlock](i, j) * delta(j);
        }
    }

    //OL in the first iteration
    theta = fmin(param->colTau1, param->colTau2 * deltaNormSq);
    if (deltaNormSq > myEps && deltaNormSqOld > myEps){
        scale = (1.0 - theta)*deltaGammaOld / deltaNormSqOld + theta*deltaBdelta / deltaNormSq;
        if (scale > param->eps)
            scale = ((1.0 - theta)*deltaGammaOld / deltaNormSqOld + theta*deltaGamma / deltaNormSq) / scale;

        //Don't scale if scaling factor is negative, increase scaling factor to minimum
        if (scale < 0) scale = 1.0;
        scale = fmax(param->colEps, scale);
    }
    else
        scale = 1.0;

    if (scale < 1.0){
        for (int i = 0; i < Bsize; i++){
            for (int j = i; j < Bsize; j++){
                hess[iBlock](i,j) *= scale;
            }
        }
        // statistics: average sizing factor
        stats->averageSizingFactor += scale;
    }
    else
        stats->averageSizingFactor += 1.0;
}

/**
 * Apply BFGS or SR1 update blockwise and size blocks
 */
void SQPmethod::calcHessianUpdate(int updateType, int hessScaling, SymMatrix *hess){
    int iBlock, nBlocks;
    int nVarLocal;
    //Matrix gammai, deltai;
    bool firstIter;

    //if objective derv is computed exactly, don't set the last block!
    if (param->whichSecondDerv == 1 && param->blockHess)
        nBlocks = vars->nBlocks - 1;
    else
        nBlocks = vars->nBlocks;

    // Statistics: how often is damping active, what is the average COL sizing factor?
    stats->hessDamped = 0;
    stats->averageSizingFactor = 0.0;

    for (iBlock = 0; iBlock < nBlocks; iBlock++){
        nVarLocal = hess[iBlock].m;

        // smallGamma and smallDelta are subvectors of gamma and delta, corresponding to partially separability
        //gammai.Submatrix(vars->gammaMat, nVarLocal, 1, vars->blockIdx[iBlock], vars->dg_pos);
        //deltai.Submatrix(vars->deltaMat, nVarLocal, 1, vars->blockIdx[iBlock], vars->dg_pos);

        // Is this the first iteration or the first after a Hessian reset?
        firstIter = (vars->nquasi[iBlock] == 1);

        // Sizing before the update
        if (firstIter)
            sizeInitialHessian( vars->dg_pos, iBlock, hess, hessScaling);
        else if (hessScaling == 4)
            sizeHessianCOL(vars->dg_pos, iBlock, hess);

        // Compute the new update
        // deltaNormOld and deltaGammaOld are set here (damping may be applied)
        if (updateType == 1)
            calcSR1(vars->dg_pos, iBlock, hess);
        else if (updateType == 2)
            calcBFGS(vars->dg_pos, iBlock, hess, true);

        // If an update is skipped to often, reset Hessian block
        if(vars->noUpdateCounter[iBlock] > param->maxConsecSkippedUpdates)
            resetHessian(iBlock, hess);
        
        vars->deltaNormSqOld(iBlock) = vars->deltaNormSqMat(iBlock, vars->dg_pos);
    }

    //Save deltaOld and its sectioned square norms. These are required for COL sizing and may change if the variables are rescaled
    for (int i = 0; i < prob->nVar; i++){
        vars->deltaOld(i) = vars->deltaMat(i, vars->dg_pos);
    }
    //vars->deltaNormSqOld = vars->deltaNormSq;

    // statistics: average sizing factor
    stats->averageSizingFactor /= nBlocks;
}

void SQPmethod::calcHessianUpdateLimitedMemory(int updateType, int hessScaling, SymMatrix *hess){
    int iBlock, nBlocks;
    //Matrix smallGamma, smallDelta;
    //Matrix gammai, deltai;
    int n_updates, pos, posOldest;
    int hessDamped, hessSkipped;
    double averageSizingFactor;

    //if objective derv is computed exactly, don't set the last block!
    if (param->whichSecondDerv == 1 && param->blockHess)
        nBlocks = vars->nBlocks - 1;
    else
        nBlocks = vars->nBlocks;

    // Statistics: how often is damping active, what is the average COL sizing factor?
    stats->hessDamped = 0;
    stats->hessSkipped = 0;
    stats->averageSizingFactor = 0.0;

    for (iBlock = 0; iBlock < nBlocks; iBlock++){
        //nVarLocal = hess[iBlock].m;
        //Bsize = vars->blockIdx[iBlock+1] - vars->blockIdx[iBlock];
        // smallGamma and smallDelta are submatrices of gammaMat, deltaMat,
        // i.e. subvectors of gamma and delta from m prev. iterations
        //smallGamma.Submatrix(vars->gammaMat, nVarLocal, vars->dg_nsave, vars->blockIdx[iBlock], 0);
        //smallDelta.Submatrix(vars->deltaMat, nVarLocal, vars->dg_nsave, vars->blockIdx[iBlock], 0);
        
        // Memory structure
        n_updates = vars->nquasi[iBlock];
        posOldest = (vars->dg_pos - n_updates + 1 + vars->dg_nsave) % vars->dg_nsave;

        // Set B_0 (pretend it's the first step)
        calcInitialHessian(iBlock, hess);
        //vars->deltaNormSqOld( iBlock ) = 1.0;
        //vars->deltaGammaOld( iBlock ) = 0.0;
        vars->noUpdateCounter[iBlock] = -1;
        
        // Size the initial update, but with the most recent delta/gamma-pair
        //gammai.Submatrix(smallGamma, nVarLocal, 1, 0, vars->dg_pos);
        //deltai.Submatrix(smallDelta, nVarLocal, 1, 0, vars->dg_pos);
        //sizeInitialHessian( gammai, deltai, iBlock, hessScaling, hess);
        sizeInitialHessian(vars->dg_pos, iBlock, hess, hessScaling);

        // OL sizing with different bounds on sizing factor and most recent delta/gamma-pair if COL sizing is used
        //if (hessScaling == 4) sizeHessianCOL(gammai, deltai, iBlock, true, hess);

        for (int i = 0; i < n_updates; i++){
            pos = (posOldest + i) % vars->dg_nsave;

            // Get new vector from list
            //gammai.Submatrix( smallGamma, nVarLocal, 1, 0, pos );
            //deltai.Submatrix( smallDelta, nVarLocal, 1, 0, pos );

            //vars->deltaNormSqOld(iBlock) = vars->deltaNormSqMat(iBlock, pos);

            //vars->deltaNormSq.Submatrix(vars->deltaNormSqMat, vars->nBlocks, 1, 0, pos);
            //vars->deltaGamma.Submatrix(vars->deltaGammaMat, vars->nBlocks, 1, 0, pos);

            // Save statistics, we want to record them only for the most recent update
            averageSizingFactor = stats->averageSizingFactor;
            hessDamped = stats->hessDamped;
            hessSkipped = stats->hessSkipped;

            // Selective sizing before the update
            if (hessScaling == 4 && i > 0)
                sizeHessianCOL(pos, iBlock, hess);
            
            // Compute the new update
            if (updateType == 1)
                calcSR1(pos, iBlock, hess);
            else if (updateType == 2)
                calcBFGS(pos, iBlock, hess, true);
            else if (updateType == 7)
                calcBFGS(pos, iBlock, hess, false);

            stats->nTotalUpdates++;

            // Count damping statistics only for the most recent update
            if (pos != vars->dg_pos){
                stats->hessDamped = hessDamped;
                stats->hessSkipped = hessSkipped;
                if (hessScaling == 4)
                    stats->averageSizingFactor = averageSizingFactor;
            }

            //If too many updates are skipped during limited memory update, reset Hessian and restart from next limited memory update
            if (vars->noUpdateCounter[iBlock] > param->maxConsecSkippedUpdates){
                vars->nquasi[iBlock] -= (i+1);
                //If Hessian was reset after the final update, proceed to next block. Sizing of the initial Hessian is still applied in this case
                iBlock -= 1;
                std::cout << "Too many updates skipped, resetting limited memory Hessian block\n";
                goto lim_mem_outer_continue;
            }
            vars->deltaNormSqOld(iBlock) = vars->deltaNormSqMat(iBlock, pos);
        }//inner loop end
        lim_mem_outer_continue:;
    }//out loop end
    stats->averageSizingFactor /= nBlocks;
    return;
}





//void SQPmethod::calcBFGS(const Matrix &gamma, const Matrix &delta, int iBlock, bool damping, SymMatrix *hess){
void SQPmethod::calcBFGS(int dpos, int iBlock, SymMatrix *hess, bool damping){
    Matrix delta, gamma2, Bdelta;
    //NOTE: C stdlib gamma() is included, thats why the variable is called gamma2
    SymMatrix *B;
    double h1 = 0.0;
    double h2 = 0.0;
    double thetaPowell = 0.0;
    int damped;
    int Bsize = vars->blockIdx[iBlock + 1] - vars->blockIdx[iBlock];
    //double myEps = 1.0e4 * param->eps;

    /* Work with a local copy of gamma because damping may need to change gamma.
     * Note that vars->gamma needs to remain unchanged!
     * This may be important in a limited memory context:
     * When information is "forgotten", B_i-1 is different and the
     *  original gamma might lead to an undamped update with the new B_i-1! */
    delta.Submatrix(vars->deltaMat, Bsize, 1, vars->blockIdx[iBlock], dpos);
    gamma2.Dimension(Bsize, 1);
    for (int i = 0; i < Bsize; i++){
        gamma2(i) = vars->gammaMat(vars->blockIdx[iBlock] + i, dpos);
    }
    B = hess + iBlock;

    // Bdelta = B*delta (if sizing is enabled, B is the sized B!)
    // h1 = delta^T * B * delta
    // h2 = delta^T * gamma
    double h2_ = 0.0;
    Bdelta.Dimension(Bsize).Initialize(0.0);
    for (int i = 0; i < Bsize; i++){
        for (int k = 0; k < Bsize; k++){
            Bdelta(i) += (*B)(i,k) * delta(k);
        }
        h1 += delta(i) * Bdelta(i);
        h2_ += delta( i ) * gamma2( i );
    }
    h2 = vars->deltaGammaMat(iBlock, dpos);

    /* Powell's damping strategy to maintain pos. def. (Nocedal/Wright p.537; SNOPT paper)
     * Interpolates between current approximation and unmodified BFGS */
    damped = 0;
    if (damping){
        if (h2 < param->hessDampFac * h1 && fabs( h1 - h2 ) > param->minDampQuot){
            // At the first iteration h1 and h2 are equal due to COL scaling
            thetaPowell = (1.0 - param->hessDampFac)*h1 / ( h1 - h2 );
            // Redefine gamma and h2 = delta^T * gamma
            h2 = 0.0;
            for (int i = 0; i < Bsize; i++){
                gamma2(i) = thetaPowell * gamma2(i) + (1.0 - thetaPowell) * Bdelta(i);
                h2 += delta(i) * gamma2(i);
            }
            damped = 1;
        }
    }

    //Save h2 for COL sizing in next iteration
    if (hess == vars->hess1) vars->deltaGammaOld(iBlock) = h2;
    else vars->deltaGammaOldFallback(iBlock) = h2;

    // For statistics: count number of damped blocks
    stats->hessDamped += damped;

    double myEps = 1.0e2 * param->eps;

    if (fabs(h1) < myEps || fabs(h2) < myEps || (damping && h2 < param->hessDampFac * h1 && fabs( h1 - h2 ) <= param->minDampQuot)){
        // don't perform update because of bad condition, might introduce negative eigenvalues
        vars->noUpdateCounter[iBlock]++;
        stats->hessDamped -= damped;
        stats->hessSkipped++;
        stats->nTotalSkippedUpdates++;
    }
    else{
        for (int i = 0; i < Bsize; i++){
            for (int j = i; j < Bsize; j++){
                (*B)(i,j) = (*B)(i,j) - Bdelta(i) * Bdelta(j) / h1 + gamma2(i) * gamma2(j) / h2;
            }
        }
        vars->noUpdateCounter[iBlock] = 0;
    }
    return;
}

//void SQPmethod::calcSR1( const Matrix &gamma, const Matrix &delta, int iBlock, SymMatrix *hess){
void SQPmethod::calcSR1(int dpos, int iBlock, SymMatrix *hess){
    int Bsize = vars->blockIdx[iBlock + 1] - vars->blockIdx[iBlock];
    Matrix delta, gamma, gmBdelta;
    SymMatrix *B;
    double myEps = 1.0e2 * param->eps;
    double h = 0.0;

    delta.Submatrix(vars->deltaMat, Bsize, 1, vars->blockIdx[iBlock], dpos);
    gamma.Submatrix(vars->gammaMat, Bsize, 1, vars->blockIdx[iBlock], dpos);
    B = &hess[iBlock];

    // gmBdelta = gamma - B*delta

    /*
    gmBdelta.Dimension(Bsize, 1);
    for (int i = 0; i < Bsize; i++){
        gmBdelta(i) = gamma(i);
        for (int j = 0; j < Bsize; j++){
            gmBdelta(i) -= (*B)(i,j)*delta(j);
        }
        h += gmBdelta(i)*delta(i);
    }
    */

    gmBdelta.Dimension(Bsize).Initialize(0.);
    for (int i = 0; i < Bsize; i++){
        for (int j = 0; j < Bsize; j++){
            gmBdelta(i) -= (*B)(i,j)*delta(j);
        }
    }

    // h = (gamma - B*delta)^T * delta
    for (int i = 0; i < Bsize; i++){
        gmBdelta(i) += gamma(i);
        h += gmBdelta(i)*delta(i);
    }

    //Update the scalar products (no damping here!)
    if (hess == vars->hess1) vars->deltaGammaOld(iBlock) = vars->deltaGammaMat(iBlock, dpos);
    else vars->deltaGammaOldFallback(iBlock) = vars->deltaGammaMat(iBlock, dpos);

    // B_k+1 = B_k + gmBdelta * gmBdelta^T / h
    if (fabs(h) < param->SR1_reltol * l2VectorNorm(delta) * l2VectorNorm(gmBdelta) ||
        2*l2VectorNorm(gmBdelta)/(l2VectorNorm(gmBdelta) + l2VectorNorm(gamma)) < param->SR1_reltol ||
        vars->deltaNormSqMat(iBlock, dpos) < param->SR1_abstol ||
        fabs(h) < param->SR1_abstol
        ){
        //Skip update if denominator is too small
        vars->noUpdateCounter[iBlock]++;
        stats->hessSkipped++;
        stats->nTotalSkippedUpdates++;
    }
    else{
        for (int i = 0; i < Bsize; i++)
            for (int j = i; j < Bsize; j++)
                (*B)(i,j) = (*B)(i,j) + gmBdelta(i) * gmBdelta(j) / h;
        vars->noUpdateCounter[iBlock] = 0;
    }
    return;
}


/**
 * Set deltaXi and gamma as a column in the matrix containing
 * the m most recent delta and gamma
 */

 /*
void SQPmethod::updateDeltaGamma(){
    Matrix smallDelta, smallGamma;
    int Bsize;
    //Update position of current calculated delta-gamma pair
    vars->dg_pos = (vars->dg_pos + 1) % vars->dg_nsave;

    //set deltaXi and gamma as storage for the next step
    vars->deltaXi.Submatrix(vars->deltaMat, prob->nVar, 1, 0, (vars->dg_pos + 1)%vars->dg_nsave);
    vars->gamma.Submatrix(vars->gammaMat, prob->nVar, 1, 0, (vars->dg_pos + 1)%vars->dg_nsave);

    //Precalculate some scalar products
    for (int iBlock = 0; iBlock < vars->nBlocks; iBlock++){
        Bsize = vars->blockIdx[iBlock + 1] - vars->blockIdx[iBlock];
        smallDelta.Submatrix(vars->deltaMat, Bsize, 1, vars->blockIdx[iBlock], vars->dg_pos);
        smallGamma.Submatrix(vars->gammaMat, Bsize, 1, vars->blockIdx[iBlock], vars->dg_pos);
        vars->deltaNormSqMat(iBlock, vars->dg_pos) = adotb(smallDelta, smallDelta);
        vars->deltaGammaMat(iBlock, vars->dg_pos) = adotb(smallDelta, smallGamma);
    }
    return;
}
*/

/*
void SQPmethod::updateScalarProducts(){
    Matrix smallDelta, smallGamma;
    int nVarLocal;

    for (int iBlock = 0; iBlock < vars->nBlocks; iBlock++){
        nVarLocal = vars->blockIdx[iBlock + 1] - vars->blockIdx[iBlock];
        smallDelta.Submatrix(vars->deltaMat, nVarLocal, 1, vars->blockIdx[iBlock], vars->dg_pos);
        smallGamma.Submatrix(vars->gammaMat, nVarLocal, 1, vars->blockIdx[iBlock], vars->dg_pos);
        vars->deltaNormSqMat(iBlock, vars->dg_pos) = adotb(smallDelta, smallDelta);
        vars->deltaGammaMat(iBlock, vars->dg_pos) = adotb(smallDelta, smallGamma);
    }
    return;
}

void SQPmethod::updateScalarProductsLimitedMemory(){
    Matrix smallDelta, smallGamma;
    int nVarLocal;

    for (int iBlock = 0; iBlock < vars->nBlocks; iBlock++){
        nVarLocal = vars->blockIdx[iBlock + 1] - vars->blockIdx[iBlock];
        smallDelta.Submatrix(vars->deltaMat, nVarLocal, 1, vars->blockIdx[iBlock], vars->dg_pos);
        smallGamma.Submatrix(vars->gammaMat, nVarLocal, 1, vars->blockIdx[iBlock], vars->dg_pos);
        vars->deltaNormSqMat(iBlock, vars->dg_pos) = adotb(smallDelta, smallDelta);
        vars->deltaGammaMat(iBlock, vars->dg_pos) = adotb(smallDelta, smallGamma);
    }
    return;
}
*/

} // namespace blockSQP

