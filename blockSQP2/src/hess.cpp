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
 * \file blocksqp_hess.cpp
 * \author Dennis Janka
 * \date 2012-2015
 *
 *  Implementation of methods of SQPmethod class associated with
 *  computation of Hessian approximations.
 *  
 * \modifications
 *  \author Reinhold Wittmann
 *  \date 2023-2025
 */


#include <blockSQP2/iterate.hpp>
#include <blockSQP2/options.hpp>
#include <blockSQP2/stats.hpp>
#include <blockSQP2/method.hpp>
#include <blockSQP2/general_purpose.hpp>
#include <iostream>
#include <fstream>
#include <thread>

namespace blockSQP2{


void SQPmethod::calcInitialHessian(SymMatrix *hess){
    for (int iBlock = 0; iBlock < vars->nBlocks - int(skip_last_block(hess)); iBlock++){
        hess[iBlock].Initialize(0.0);
        for (int i = 0; i < hess[iBlock].m; i++) hess[iBlock](i,i) = param->initial_hess_scale;
    }
}


void SQPmethod::calcInitialHessian(int iBlock, SymMatrix *hess){
    hess[iBlock].Initialize(0.0);
    // Each block is a diagonal matrix
    for (int i = 0; i < hess[iBlock].m; i++)
        hess[iBlock](i, i) = param->initial_hess_scale;
}

void SQPmethod::calcInitialHessians(){
    calcInitialHessian(vars->hess1.get());
    if (vars->hess2 != nullptr) calcInitialHessian(vars->hess2.get());
}


void SQPmethod::calcScaledInitialHessian(double scale, SymMatrix *hess){
    for (int iBlock = 0; iBlock < vars->nBlocks - int(skip_last_block(hess)); iBlock++){
        hess[iBlock].Initialize(0.0);
        for (int i = 0; i < hess[iBlock].m; i++)
            hess[iBlock](i, i) = scale;
    }
}

void SQPmethod::calcScaledInitialHessian(int iBlock, double scale, SymMatrix *hess){
    hess[iBlock].Initialize( 0.0 );
    // Each block is a diagonal matrix
    for (int i = 0; i < hess[iBlock].m; i++)
        hess[iBlock](i, i) = scale;
}


void SQPmethod::resetHessian(SymMatrix *hess){
    for (int iBlock = 0; iBlock < vars->nBlocks - int(skip_last_block(hess)); iBlock++){
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

    if (hess == vars->hess1.get()){
        vars->deltaNormSqOld( iBlock ) = 1.0;
        vars->deltaGammaOld( iBlock ) = 0.0;
    }
    else{
        vars->deltaGammaOldFallback( iBlock ) = 0.0;
    }
}

void SQPmethod::resetHessians(){
    resetHessian(vars->hess1.get());
    if (vars->hess2 != nullptr) resetHessian(vars->hess2.get());
    return;
}

int SQPmethod::calcFiniteDiffHessian(SymMatrix *hess){
    int iVar, jVar, k, iBlock, maxBlock, info, idx, idx1, idx2;
    double dummy, lowerVio, upperVio;
    Matrix pert;
    SQPiterate varsP = SQPiterate(*vars);

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
    calcLagrangeGradient( vars->lambda, vars->gradObj, vars->sparse_constrJac.nz.get(), vars->sparse_constrJac.row.get(), vars->sparse_constrJac.colind.get(), vars->gradLagrange, 0 );

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
        if (param->sparse){
            prob->evaluate( vars->xi, vars->lambda, &dummy, varsP.constr, varsP.gradObj,
                            varsP.sparse_constrJac.nz.get(), varsP.sparse_constrJac.row.get(), varsP.sparse_constrJac.colind.get(), hess, 1, &info );
            calcLagrangeGradient( vars->lambda, varsP.gradObj, varsP.sparse_constrJac.nz.get(), varsP.sparse_constrJac.row.get(),
                                  varsP.sparse_constrJac.colind.get(), varsP.gradLagrange, 0 );
        }
        else{
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


void SQPmethod::sizeInitialHessian(Sizings sizingType, int dpos, int iBlock, SymMatrix *hess){
    int i, j;
    double scale;
    double myEps = 1.0e3 * param->eps;
    Matrix gamma;
    
    //TODO: Consider adding condition l1VectorNorm(delta) > tol
    
    switch (sizingType){
        case Sizings::SP: //Shanno-Phua
            gamma.Submatrix(vars->gammaMat, vars->blockIdx[iBlock+1] - vars->blockIdx[iBlock], 1, vars->blockIdx[iBlock], dpos);
            scale = adotb(gamma, gamma) / fmax(vars->deltaGammaMat(iBlock, dpos)*param->initial_hess_scale, myEps);
            break;
        case Sizings::OL: //Oren-Luenberger
            scale = vars->deltaGammaMat(iBlock, dpos) / fmax(vars->deltaNormSqMat(iBlock, dpos)*param->initial_hess_scale, myEps);
            if (scale < 0){scale *= -1;}
            //if (scale < 0){scale = 1.0;}
            scale = fmin(scale, 1.0);
            break;
        case Sizings::GM_SP_OL: //Geometric mean of 1 and 2
            gamma.Submatrix(vars->gammaMat, vars->blockIdx[iBlock+1] - vars->blockIdx[iBlock], 1, vars->blockIdx[iBlock], dpos);
            scale = sqrt(adotb(gamma, gamma)/fmax(vars->deltaNormSqMat(iBlock, dpos)*param->initial_hess_scale, myEps));
            break;
        case Sizings::COL: //First COL sizing, = OL sizing with different bounds
            scale = vars->deltaGammaMat(iBlock, dpos) / fmax(vars->deltaNormSqMat(iBlock, dpos)*param->initial_hess_scale, myEps);
            if (scale < 0){scale *= -1;}
            //if (scale < 0){scale = 1.0;}
            scale = fmax(fmin(scale, 1.0), param->OL_eps);
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
    if (hess == vars->hess1.get()) deltaGammaOld = vars->deltaGammaOld(iBlock);
    else deltaGammaOld = vars->deltaGammaOldFallback(iBlock);
    
    deltaBdelta = 0.0;
    for (int i = 0; i < Bsize; i++){
        for (int j = 0; j < Bsize; j++){
            deltaBdelta += delta(i) * hess[iBlock](i, j) * delta(j);
        }
    }
    
    //OL in the first iteration
    theta = fmin(param->COL_tau_1, param->COL_tau_2 * deltaNormSq);
    if (deltaNormSq > myEps && deltaNormSqOld > myEps){
        scale = (1.0 - theta)*deltaGammaOld / deltaNormSqOld + theta*deltaBdelta / deltaNormSq;
        if (scale > param->eps)
            scale = ((1.0 - theta)*deltaGammaOld / deltaNormSqOld + theta*deltaGamma / deltaNormSq) / scale;

        //Don't scale if scaling factor is negative, increase scaling factor to minimum
        if (scale < 0) scale = 1.0;
        scale = fmax(param->COL_eps, scale);
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



void SQPmethod::calcHessianUpdate(Hessians updateType, Sizings sizingType, SymMatrix *hess){
    int iBlock;
    bool firstIter;
    int nBlocks = vars->nBlocks - int(skip_last_block(hess));
    
    // Statistics: how often is damping active, what is the average COL sizing factor?
    stats->hessDamped = 0;
    stats->averageSizingFactor = 0.0;
    
    for (iBlock = 0; iBlock < nBlocks; iBlock++){
        // Is this the first iteration or the first after a Hessian reset?
        firstIter = (vars->nquasi[iBlock] == 1);
        
        // Sizing before the update
        if (firstIter)
            sizeInitialHessian(sizingType, vars->dg_pos, iBlock, hess);
        else if (sizingType == Sizings::COL)
            sizeHessianCOL(vars->dg_pos, iBlock, hess);
        calcQN(updateType, vars->dg_pos, iBlock, hess);
        
        // If an update is skipped to often, reset Hessian block
        if(vars->noUpdateCounter[iBlock] > param->max_consec_skipped_updates)
            resetHessian(iBlock, hess);
        
        vars->deltaNormSqOld(iBlock) = vars->deltaNormSqMat(iBlock, vars->dg_pos);
    }

    //Save deltaOld and its sectioned square norms. These are required for COL sizing and may change if the variables are rescaled
    for (int i = 0; i < prob->nVar; i++){
        vars->deltaOld(i) = vars->deltaMat(i, vars->dg_pos);
    }
    
    // statistics: average sizing factor
    stats->averageSizingFactor /= nBlocks;
}

void SQPmethod::calcHessianUpdateLimitedMemory(Hessians updateType, Sizings sizingType, SymMatrix *hess){
    int iBlock;
    int n_updates, pos, posOldest;
    // int hessDamped, hessSkipped;
    // double averageSizingFactor;
    int nBlocks = vars->nBlocks - int(skip_last_block(hess));
    // Statistics: how often is damping active, what is the average COL sizing factor?
    stats->hessDamped = 0;
    stats->hessSkipped = 0;
    stats->averageSizingFactor = 0.0;
    
    for (iBlock = 0; iBlock < nBlocks; iBlock++){
        // Memory structure
        n_updates = vars->nquasi[iBlock];
        posOldest = (vars->dg_pos - n_updates + 1 + vars->dg_nsave) % vars->dg_nsave;
        
        // Set B_0 (pretend it's the first step)
        calcInitialHessian(iBlock, hess);
        vars->noUpdateCounter[iBlock] = -1;
        sizeInitialHessian(sizingType, vars->dg_pos, iBlock, hess);
        
        for (int i = 0; i < n_updates; i++){
            pos = (posOldest + i) % vars->dg_nsave;
            // Save statistics, we want to record them only for the most recent update
            // averageSizingFactor = stats->averageSizingFactor;
            // hessDamped = stats->hessDamped;
            // hessSkipped = stats->hessSkipped;
            
            // Selective sizing before the update
            if (sizingType == Sizings::COL && i > 0)
                sizeHessianCOL(pos, iBlock, hess);
            calcQN(updateType, pos, iBlock, hess);
            // stats->nTotalUpdates++;
            // Count damping statistics only for the most recent update
            // if (pos != vars->dg_pos){
            //     stats->hessDamped = hessDamped;
            //     stats->hessSkipped = hessSkipped;
            //     if (sizingType == Sizings::COL)
            //         stats->averageSizingFactor = averageSizingFactor;
            // }
            
            //If too many updates are skipped during limited memory update, reset Hessian and restart from next limited memory update
            if (vars->noUpdateCounter[iBlock] > param->max_consec_skipped_updates){
                vars->nquasi[iBlock] -= (i+1);
                //If Hessian was reset after the final update, proceed to next block. Sizing of the initial Hessian is still applied in this case
                iBlock -= 1;
                std::cout << "Too many updates skipped, resetting limited memory Hessian block\n";
                break;
            }
            vars->deltaNormSqOld(iBlock) = vars->deltaNormSqMat(iBlock, pos);
        }
    }
    stats->averageSizingFactor /= nBlocks;
    return;
}

#define LIM_MEM_N_THREADS 4

void SQPmethod::calcHessianUpdateLimitedMemory_par(Hessians updateType, Sizings sizingType, SymMatrix *hess){
    int nBlocks = vars->nBlocks - int(skip_last_block(hess));
    if (nBlocks < LIM_MEM_N_THREADS){
        calcHessianUpdateLimitedMemory(updateType, sizingType, hess);
        return;
    }
      
    int blockIdxIdx[LIM_MEM_N_THREADS + 1]{0, int(nBlocks/4), int(nBlocks/2), int((3*nBlocks)/4), nBlocks};
    std::jthread upThreads[LIM_MEM_N_THREADS];
    
    for (int j = 0; j < LIM_MEM_N_THREADS; j++){
        upThreads[j] = std::jthread(&SQPmethod::par_inner_update_loop, this, updateType, sizingType, hess, blockIdxIdx[j], blockIdxIdx[j+1]);
    }
    return;
}


void SQPmethod::par_inner_update_loop(Hessians updateType, Sizings sizingType, SymMatrix *hess, int blockIdx_start, int blockIdx_end){
    for (int iBlock = blockIdx_start; iBlock < blockIdx_end; iBlock++){
        // Memory structure
        int n_updates = vars->nquasi[iBlock];
        int posOldest = (vars->dg_pos - n_updates + 1 + vars->dg_nsave) % vars->dg_nsave;
        
        // Set B_0 (pretend it's the first step)
        calcInitialHessian(iBlock, hess);
        vars->noUpdateCounter[iBlock] = -1;
        sizeInitialHessian(sizingType, vars->dg_pos, iBlock, hess);
        
        for (int i = 0; i < n_updates; i++){
            int pos = (posOldest + i) % vars->dg_nsave;
            
            // Selective sizing before the update
            if (sizingType == Sizings::COL && i > 0)
                sizeHessianCOL(pos, iBlock, hess);
            calcQN(updateType, pos, iBlock, hess);
            
            //If too many updates are skipped during limited memory update, reset Hessian and restart from next limited memory update
            if (vars->noUpdateCounter[iBlock] > param->max_consec_skipped_updates){
                vars->nquasi[iBlock] -= (i+1);
                //If Hessian was reset after the final update, proceed to next block. Sizing of the initial Hessian is still applied in this case
                iBlock -= 1;
                std::cout << "Too many updates skipped, resetting limited memory Hessian block\n";
                break;
            }
            vars->deltaNormSqOld(iBlock) = vars->deltaNormSqMat(iBlock, pos);
        }
    }
}


/*
void SQPmethod::calcHessianUpdateLimitedMemory_2(int updateType, int sizing, SymMatrix *hess){
    int iBlock;
    //Matrix smallGamma, smallDelta;
    //Matrix gammai, deltai;
    int  pos, posOldest;
    
    //if objective derv is computed exactly, don't set the last block!
    // if (param->exact_hess == 1 && param->block_hess)
    //     nBlocks = vars->nBlocks - 1;
    // else
    //     nBlocks = vars->nBlocks;
    int nBlocks = vars->nBlocks - int(skip_last_block(hess));
    
    posOldest = (vars->dg_pos - param->mem_size + 1 + vars->dg_nsave) % vars->dg_nsave;
    for (int i = -1; i < param->mem_size; i++){
        // Memory structure
        pos = (posOldest + i) % vars->dg_nsave;
        for (iBlock = 0; iBlock < nBlocks; iBlock++){
            //Set and size initial approximation in first pass
            if (i == param->mem_size - 1 - vars->nquasi[i]){
                calcInitialHessian(iBlock, hess);
                vars->noUpdateCounter[iBlock] = -1;
                sizeInitialHessian(vars->dg_pos, iBlock, hess, sizing);
            }
            else if (i > param->mem_size - 1 - vars->nquasi[i]){
                if ((sizing == 4 && i > param->mem_size - vars->nquasi[i])) 
                    sizeHessianCOL(pos, iBlock, hess);
                
                // Compute the new update
                if (updateType == 1)
                    calcSR1(pos, iBlock, hess);
                else if (updateType == 2)
                    calcBFGS(pos, iBlock, hess, true);
                else if (updateType == 7)
                    calcBFGS(pos, iBlock, hess, false);

                //If too many updates are skipped during limited memory update, reset Hessian and restart from next limited memory update
                if (vars->noUpdateCounter[iBlock] > param->max_consec_skipped_updates){
                    vars->nquasi[iBlock] = param->mem_size - 1 - i;
                    iBlock -= 1;
                    continue;
                }
                vars->deltaNormSqOld(iBlock) = vars->deltaNormSqMat(iBlock, pos);
            }
        }
    }
    stats->averageSizingFactor /= nBlocks;
    return;
}
*/


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
        if (h2 < param->BFGS_damping_factor * h1 && fabs( h1 - h2 ) > param->min_damping_quotient){
            // At the first iteration h1 and h2 are equal due to COL scaling
            thetaPowell = (1.0 - param->BFGS_damping_factor)*h1 / ( h1 - h2 );
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
    if (hess == vars->hess1.get()) vars->deltaGammaOld(iBlock) = h2;
    else vars->deltaGammaOldFallback(iBlock) = h2;

    // For statistics: count number of damped blocks
    stats->hessDamped += damped;

    double myEps = 1.0e2 * param->eps;

    if (fabs(h1) < myEps || fabs(h2) < myEps || (damping && h2 < param->BFGS_damping_factor * h1 && fabs( h1 - h2 ) <= param->min_damping_quotient)){
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


void SQPmethod::calcSR1(int dpos, int iBlock, SymMatrix *hess){
    int Bsize = vars->blockIdx[iBlock + 1] - vars->blockIdx[iBlock];
    Matrix delta, gamma, gmBdelta;
    SymMatrix *B;
    double h = 0.0;

    delta.Submatrix(vars->deltaMat, Bsize, 1, vars->blockIdx[iBlock], dpos);
    gamma.Submatrix(vars->gammaMat, Bsize, 1, vars->blockIdx[iBlock], dpos);
    B = &hess[iBlock];

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
    if (hess == vars->hess1.get()) vars->deltaGammaOld(iBlock) = vars->deltaGammaMat(iBlock, dpos);
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

} // namespace blockSQP2

