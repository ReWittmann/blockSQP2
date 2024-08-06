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




void SQPmethod::resetHessian(SymMatrix *hess)
{
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
        vars->deltaNormOld( iBlock ) = 1.0;
        vars->deltaGammaOld( iBlock ) = 0.0;
    }
    else{
        vars->deltaNormOldFallback( iBlock ) = 1.0;
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


void SQPmethod::sizeInitialHessian( const Matrix &gamma, const Matrix &delta, int iBlock, int option, SymMatrix *hess)
{
    int i, j;
    double scale;
    double myEps = 1.0e3 * param->eps;

    //TODO: Consider adding condition l1VectorNorm(delta) > tol

    if( option == 1 )
    {// Shanno-Phua
        scale = adotb( gamma, gamma ) / fmax( adotb( delta, gamma ), myEps );
    }
    else if( option == 2 )
    {// Oren-Luenberger
        scale = adotb( delta, gamma ) / fmax( adotb( delta, delta ), myEps );
        //NEW//
        if (scale < 0) scale *= -1;
        ///////
        scale = fmin( scale, 1.0 );
    }
    else if( option == 3 )
    {// Geometric mean of 1 and 2
        scale = sqrt( adotb( gamma, gamma ) / fmax( adotb( delta, delta ), myEps ) );
    }
    else
    {// Invalid option, ignore
        return;
    }

    if (scale < 0) scale *= -1;

    scale /= fmax(param->iniHessDiag, myEps);
    scale = fmax(scale, myEps);

    for (i = 0; i < hess[iBlock].m; i++){
        for (j = i; j < hess[iBlock].m; j++){
            hess[iBlock](i,j) *= scale;
        }
    }
    // statistics: average sizing factor
    stats->averageSizingFactor += scale;
}


void SQPmethod::sizeHessianCOL( const Matrix &gamma, const Matrix &delta, int iBlock, bool first_sizing, SymMatrix *hess)
{
    int i, j;
    double theta, scale, myEps = 1.0e3 * param->eps;
    double deltaNorm, deltaNormOld, deltaGamma, deltaGammaOld, deltaBdelta;

    // Get sTs, sTs_, sTy, sTy_, sTBs
    deltaNorm = vars->deltaNorm(iBlock);
    deltaGamma = vars->deltaGamma(iBlock);
    if (hess == vars->hess1){
        deltaNormOld = vars->deltaNormOld(iBlock);
        deltaGammaOld = vars->deltaGammaOld(iBlock);
    }
    else{
        deltaNormOld = vars->deltaNormOldFallback(iBlock);
        deltaGammaOld = vars->deltaGammaOldFallback(iBlock);
    }

    deltaBdelta = 0.0;
    for( i=0; i<delta.M(); i++ )
        for( j=0; j<delta.M(); j++ )
            deltaBdelta += delta( i ) * hess[iBlock]( i, j ) * delta( j );

    //OL in the first iteration
    if (first_sizing){
        if (deltaNorm > myEps){
            scale = deltaGamma/fmax(deltaBdelta, myEps);
            //Scale with absolute value of sizing factor the get good initial matrix magnitude
            if (scale < 0) scale *= -1;
            //Increase initial scaling factor to minimum
            scale = fmax(scale, param->olEps);
        }
        else{
            scale = 1.0;
        }
    }
    else{
        theta = fmin(param->colTau1, param->colTau2 * deltaNorm);
        if (deltaNorm > myEps && deltaNormOld > myEps){
            scale = (1.0 - theta)*deltaGammaOld / deltaNormOld + theta*deltaBdelta / deltaNorm;
            if (scale > param->eps)
                scale = ((1.0 - theta)*deltaGammaOld / deltaNormOld + theta*deltaGamma / deltaNorm) / scale;

            //Don't scale if scaling factor is negative, increase scaling factor to minimum
            if (scale < 0) scale = 1.0;
            scale = fmax(param->colEps, scale);
        }
        else
            scale = 1.0;
    }

    if (scale < 1.0){
        for (i = 0; i < hess[iBlock].m; i++)
            for (j = i; j < hess[iBlock].m; j++)
                hess[iBlock](i,j) *= scale;

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
    Matrix smallGamma, smallDelta;
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
        smallGamma.Submatrix(vars->gammaMat, nVarLocal, vars->gammaMat.N(), vars->blockIdx[iBlock], 0);
        smallDelta.Submatrix(vars->deltaMat, nVarLocal, vars->deltaMat.N(), vars->blockIdx[iBlock], 0);

        // Is this the first iteration or the first after a Hessian reset?
        firstIter = (vars->nquasi[iBlock] == 1);

        // Sizing before the update
        if (hessScaling < 4 && firstIter)
            sizeInitialHessian( smallGamma, smallDelta, iBlock, hessScaling, hess);
        else if (hessScaling == 4)
            sizeHessianCOL( smallGamma, smallDelta, iBlock, firstIter, hess);

        // Compute the new update
        // deltaNormOld and deltaGammaOld are set here (damping may be applied)
        if (updateType == 1)
            calcSR1(smallGamma, smallDelta, iBlock, hess);
        else if (updateType == 2)
            calcBFGS(smallGamma, smallDelta, iBlock, true, hess);

        // If an update is skipped to often, reset Hessian block
        if(vars->noUpdateCounter[iBlock] > param->maxConsecSkippedUpdates)
            resetHessian(iBlock, hess);
    }

    // statistics: average sizing factor
    stats->averageSizingFactor /= nBlocks;
}


void SQPmethod::calcHessianUpdateLimitedMemory(int updateType, int hessScaling, SymMatrix *hess){
    int iBlock, nBlocks, nVarLocal;
    Matrix smallGamma, smallDelta;
    Matrix gammai, deltai;
    int m, pos, posOldest, posNewest;
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
        nVarLocal = hess[iBlock].m;
        // smallGamma and smallDelta are submatrices of gammaMat, deltaMat,
        // i.e. subvectors of gamma and delta from m prev. iterations
        smallGamma.Submatrix(vars->gammaMat, nVarLocal, vars->gammaMat.N(), vars->blockIdx[iBlock], 0);
        smallDelta.Submatrix(vars->deltaMat, nVarLocal, vars->deltaMat.N(), vars->blockIdx[iBlock], 0);

        // Memory structure
        posNewest = vars->dg_pos % smallGamma.n;
        posOldest = (vars->dg_pos - vars->nquasi[iBlock] + 1) % smallGamma.n;
        posOldest += (posOldest < 0) * smallGamma.n;

        m = vars->nquasi[iBlock];

        // Set B_0 (pretend it's the first step)
        calcInitialHessian(iBlock, hess);
        vars->deltaNormOld( iBlock ) = 1.0;
        vars->deltaGammaOld( iBlock ) = 0.0;
        vars->noUpdateCounter[iBlock] = -1;

        // Size the initial update, but with the most recent delta/gamma-pair
        gammai.Submatrix(smallGamma, nVarLocal, 1, 0, posNewest);
        deltai.Submatrix(smallDelta, nVarLocal, 1, 0, posNewest);
        sizeInitialHessian( gammai, deltai, iBlock, hessScaling, hess);

        // OL sizing with different bounds on sizing factor and most recent delta/gamma-pair if COL sizing is used
        if (hessScaling == 4) sizeHessianCOL(gammai, deltai, iBlock, true, hess);

        for (int i = 0; i < m; i++){
            pos = (posOldest + i) % smallGamma.n;

            // Get new vector from list
            gammai.Submatrix( smallGamma, nVarLocal, 1, 0, pos );
            deltai.Submatrix( smallDelta, nVarLocal, 1, 0, pos );

            vars->deltaNorm.Submatrix(vars->deltaNormMat, vars->nBlocks, 1, 0, pos);
            vars->deltaGamma.Submatrix(vars->deltaGammaMat, vars->nBlocks, 1, 0, pos);

            // Save statistics, we want to record them only for the most recent update
            averageSizingFactor = stats->averageSizingFactor;
            hessDamped = stats->hessDamped;
            hessSkipped = stats->hessSkipped;

            // Selective sizing before the update
            if (hessScaling == 4 && i > 0)
                sizeHessianCOL(gammai, deltai, iBlock, false, hess);

            // Compute the new update
            if (updateType == 1)
                calcSR1(gammai, deltai, iBlock, hess);
            else if (updateType == 2)
                calcBFGS(gammai, deltai, iBlock, true, hess);

            stats->nTotalUpdates++;

            // Count damping statistics only for the most recent update
            if (pos != posNewest){
                stats->hessDamped = hessDamped;
                stats->hessSkipped = hessSkipped;
                if (hessScaling == 4)
                    stats->averageSizingFactor = averageSizingFactor;
            }
        }

        // If an update is skipped too often, reset Hessian block
        if (vars->noUpdateCounter[iBlock] > param->maxConsecSkippedUpdates)
            resetHessian(iBlock, hess);
    }//blocks
    stats->averageSizingFactor /= nBlocks;
}


void SQPmethod::calcBFGS(const Matrix &gamma, const Matrix &delta, int iBlock, bool damping, SymMatrix *hess){
    int i, j, k, dim = gamma.M();
    Matrix Bdelta;
    SymMatrix *B;
    double h1 = 0.0;
    double h2 = 0.0;
    double thetaPowell = 0.0;
    int damped;
    //double myEps = 1.0e4 * param->eps;

    /* Work with a local copy of gamma because damping may need to change gamma.
     * Note that vars->gamma needs to remain unchanged!
     * This may be important in a limited memory context:
     * When information is "forgotten", B_i-1 is different and the
     *  original gamma might lead to an undamped update with the new B_i-1! */
    Matrix gamma2 = gamma;
    B = &hess[iBlock];

    // Bdelta = B*delta (if sizing is enabled, B is the sized B!)
    // h1 = delta^T * B * delta
    // h2 = delta^T * gamma
    Bdelta.Dimension(dim).Initialize(0.0);
    for (i = 0; i < dim; i++){
        for (k = 0; k < dim; k++)
            Bdelta(i) += (*B)(i,k) * delta(k);

        h1 += delta(i) * Bdelta(i);
        //h2 += delta( i ) * gamma( i );
    }
    h2 = vars->deltaGamma(iBlock);

    /* Powell's damping strategy to maintain pos. def. (Nocedal/Wright p.537; SNOPT paper)
     * Interpolates between current approximation and unmodified BFGS */
    damped = 0;
    if (damping){
        if (h2 < param->hessDampFac * h1 / vars->alpha && fabs( h1 - h2 ) > 1.0e-12){
        //if (h2 < param->hessDampFac * h1 && fabs( h1 - h2 ) > 1.0e-12){
        //if (fabs(h1) >= myEps && fabs(h2) >= myEps && h2 < param->hessDampFac * h1){
            // At the first iteration h1 and h2 are equal due to COL scaling
            thetaPowell = (1.0 - param->hessDampFac)*h1 / ( h1 - h2 );

            // Redefine gamma and h2 = delta^T * gamma
            h2 = 0.0;
            for (i = 0; i < dim; i++){
                gamma2(i) = thetaPowell * gamma2(i) + (1.0 - thetaPowell) * Bdelta(i);
                h2 += delta(i) * gamma2(i);
            }
            damped = 1;
        }
    }

    if (hess == vars->hess1){
        vars->deltaNormOld(iBlock) = vars->deltaNorm(iBlock);
        vars->deltaGammaOld(iBlock) = h2;
    }
    else{
        vars->deltaNormOldFallback(iBlock) = vars->deltaNorm(iBlock);
        vars->deltaGammaOldFallback(iBlock) = h2;
    }

    // For statistics: count number of damped blocks
    stats->hessDamped += damped;

    double myEps = 1.0e2 * param->eps;


    if( fabs( h1 ) < myEps || fabs( h2 ) < myEps || (damping && h2 < param->hessDampFac * h1 / vars->alpha && fabs( h1 - h2 ) <= 1.0e-12)){
    //if( fabs( h1 ) < myEps || fabs( h2 ) < myEps || (damping && h2 < param->hessDampFac * h1 && fabs( h1 - h2 ) <= 1.0e-12)){
    //if(fabs(h1) < myEps || fabs(h2) < myEps){
        // don't perform update because of bad condition, might introduce negative eigenvalues
        vars->noUpdateCounter[iBlock]++;
        stats->hessDamped -= damped;
        stats->hessSkipped++;
        stats->nTotalSkippedUpdates++;
    }
    else{
        for (i = 0; i < dim; i++)
            for (j = i; j < dim; j++)
                (*B)(i,j) = (*B)(i,j) - Bdelta(i) * Bdelta(j) / h1 + gamma2(i) * gamma2(j) / h2;

        vars->noUpdateCounter[iBlock] = 0;
    }
}


void SQPmethod::calcSR1( const Matrix &gamma, const Matrix &delta, int iBlock, SymMatrix *hess){
    int i, j, k, dim = gamma.M();
    Matrix gmBdelta;
    SymMatrix *B;
    double myEps = 1.0e2 * param->eps;
    double r = 1.0e-8;
    double h = 0.0;

    B = &hess[iBlock];

    // gmBdelta = gamma - B*delta
    // h = (gamma - B*delta)^T * delta
    gmBdelta.Dimension(dim);
    for (i = 0; i < dim; i++){
        gmBdelta( i ) = gamma( i );
        for (k = 0; k < dim; k++)
            gmBdelta(i) -= ((*B)( i,k ) * delta(k));

        h += (gmBdelta(i) * delta(i));
    }

    //Update the scalar products (no damping here!)
    if (hess == vars->hess1){
        vars->deltaNormOld(iBlock) = vars->deltaNorm(iBlock);
        vars->deltaGammaOld(iBlock) = vars->deltaGamma(iBlock);
    }
    else{
        vars->deltaNormOldFallback(iBlock) = vars->deltaNorm(iBlock);
        vars->deltaGammaOldFallback(iBlock) = vars->deltaGamma(iBlock);
    }

    // B_k+1 = B_k + gmBdelta * gmBdelta^T / h
    if (fabs(h) < r * l2VectorNorm(delta) * l2VectorNorm(gmBdelta) || fabs(h) < myEps){
        //Skip update if denominator is too small
        vars->noUpdateCounter[iBlock]++;
        stats->hessSkipped++;
        stats->nTotalSkippedUpdates++;
    }
    else{
        for (i = 0; i < dim; i++)
            for (j = i; j < dim; j++)
                (*B)(i,j) = (*B)(i,j) + gmBdelta(i) * gmBdelta(j) / h;
        vars->noUpdateCounter[iBlock] = 0;

    }
}


/**
 * Set deltaXi and gamma as a column in the matrix containing
 * the m most recent delta and gamma
 */
void SQPmethod::updateDeltaGamma(){
    int nVar = vars->gammaMat.M();
    int m = vars->gammaMat.N();
    if( m == 1 )
        return;

    //dg_pos is the position of the current calculated step, set deltaXi and gamma as storage for the next step
    vars->deltaXi.Submatrix(vars->deltaMat, nVar, 1, 0, (vars->dg_pos + 1)%m);
    vars->gamma.Submatrix(vars->gammaMat, nVar, 1, 0, (vars->dg_pos + 1)%m);
}


void SQPmethod::updateScalarProducts(){
    Matrix smallDelta, smallGamma;
    int nVarLocal;

    for (int iBlock = 0; iBlock < vars->nBlocks; iBlock++){
        nVarLocal = vars->blockIdx[iBlock + 1] - vars->blockIdx[iBlock];
        smallDelta.Submatrix(vars->deltaXi, nVarLocal, 1, vars->blockIdx[iBlock], 0);
        smallGamma.Submatrix(vars->gamma, nVarLocal, 1, vars->blockIdx[iBlock], 0);
        vars->deltaNorm(iBlock) = adotb(smallDelta, smallDelta);
        vars->deltaGamma(iBlock) = adotb(smallDelta, smallGamma);
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
        vars->deltaNormMat(iBlock, vars->dg_pos) = adotb(smallDelta, smallDelta);
        vars->deltaGammaMat(iBlock, vars->dg_pos) = adotb(smallDelta, smallGamma);
    }
    return;
}

} // namespace blockSQP

