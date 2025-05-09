/*
 * blockSQP -- Sequential quadratic programming for problems with
 *             block-diagonal Hessian matrix.
 * Copyright (C) 2012-2015 by Dennis Janka <dennis.janka@iwr.uni-heidelberg.de>
 *
 * Licensed under the zlib license. See LICENSE for more details.
 */

/**
 * \file blocksqp_scaling.cpp
 * \author Reinhold Wittmann
 * \date 2012-2015
 *
 *  Implementation of scaling heuristics
 *
 */

 #include "blocksqp_iterate.hpp"
 #include "blocksqp_options.hpp"
 #include "blocksqp_stats.hpp"
 #include "blocksqp_method.hpp"
 #include "blocksqp_general_purpose.hpp"
 #include "blocksqp_restoration.hpp"
 #include "blocksqp_qpsolver.hpp"
 #include <fstream>
 #include <cmath>
 #include <chrono>


 namespace blockSQP{


//Algorithm 2 from paper, calculate scaling factor for free variables
void SQPmethod::calc_free_variables_scaling(double *arg_SF){
    int nIt, pos, nfree = prob->nVar, ind_1, scfree, scdep, count_delta = 0, count_gamma = 0;
    double bardelta_u, bardelta_x, bargamma_u, bargamma_x, S_u, rgamma = 0., rdelta = 0.;

    if (prob->n_vblocks < 1) return;
    nfree = prob->nVar;
    for (int k = 0; k < prob->n_vblocks; k++){
        nfree -= prob->vblocks[k].size*int(prob->vblocks[k].dependent);
    }

    nIt = std::min(vars->n_scaleIt, 5);
    for (int j = 0; j < nIt; j++){
        bardelta_u = 0.; bardelta_x = 0.; bargamma_u = 0.; bargamma_x = 0.;
        scfree = 0; scdep = 0;
        pos = (vars->dg_pos - nIt + 1 + j + vars->dg_nsave)%vars->dg_nsave;
        ind_1 = 0;
        for (int k = 0; k < prob->n_vblocks; k++){
            for (int i = 0; i < prob->vblocks[k].size; i++){
                if (std::abs(vars->deltaMat(ind_1 + i, pos)) > 1e-8){
                    if (prob->vblocks[k].dependent){
                        bardelta_x += std::abs(vars->deltaMat(ind_1 + i, pos));
                        bargamma_x += std::abs(vars->gammaMat(ind_1 + i, pos));
                        scdep += 1;
                    }
                    else{
                        bardelta_u += std::abs(vars->deltaMat(ind_1 + i, pos));
                        bargamma_u += std::abs(vars->gammaMat(ind_1 + i, pos));
                        scfree += 1;
                    }
                }
            }
            ind_1 += prob->vblocks[k].size;
        }
        
        if (scdep > 0 && scfree > 0){
            bardelta_x /= scdep; bargamma_x /= scdep;
            bardelta_u /= scfree; bargamma_u /= scfree;
        }
        else{
            bardelta_u = 0.; bardelta_x = 1.0;
            bargamma_u = 0.; bargamma_x = 1.0;
        }
        if (bargamma_x > 5e-7 && bargamma_u > 5e-7){
            rgamma += std::log(bargamma_u/bargamma_x);
            count_gamma += 1;
            if (bardelta_x > 5e-7 && bardelta_u > 5e-7){
                rdelta += std::log(bardelta_u/bardelta_x);
                count_delta += 1;
            }
        }
    }
    //If no scaling information was accumulated, rdelta is set to 1.0 => all scaling factors are 1.0
    rdelta = (count_delta > 0) ? std::exp(rdelta/count_delta) : 1.0;
    rgamma = (count_gamma > 0) ? std::exp(rgamma/count_gamma) : 1.0;

    S_u = -1.0;
    if (rgamma > 2.0){
        S_u = rgamma/2.0;
    }
    else if (rgamma < 1.0){
        if (rdelta > 1.0){
            if (rgamma < 0.1) S_u = 10.0*rgamma;
            else S_u = std::min(1.0, rdelta*rgamma);
        }
        else{
            S_u = rgamma;
        }
    }

    if (S_u > 0){
        vars->vfreeScale *= S_u;
        ind_1 = 0;
        for (int k = 0; k < prob->n_vblocks; k++){
            if (!prob->vblocks[k].dependent){
                for (int i = 0; i < prob->vblocks[k].size; i++){
                    arg_SF[ind_1 + i] *= S_u;
                }
            }
            ind_1 += prob->vblocks[k].size;
        }
    }

    return;
}

// Invokation of scaling algorithms. Decide if algorithm should be invoked in this iteration and apply the scaling
void SQPmethod::scaling_heuristic(){
    Matrix deltai, smallDelta, smallGamma;
    int pos, Bsize;
    //Scale after iterations 1, 2, 3, 5, 10, 15, ...
    if (stats->itCount > 3 && stats->itCount%5) return;

    for (int i = 0; i < prob->nVar; i++){
        vars->rescaleFactors[i] = 1.0;
    }
    calc_free_variables_scaling(vars->rescaleFactors.get());
    apply_rescaling(vars->rescaleFactors.get());
    return;
}

// Apply rescaling to the iterate and the scalable problem specification. 
void SQPmethod::apply_rescaling(const double *resfactors){
    Matrix deltai, smallDelta, smallGamma;
    int pos, Bsize, nmem;

    //Rescale the problem
    scaled_prob->rescale(resfactors);

    //Rescale current iteration data
    for (int i = 0; i < prob->nVar; i++){
        //Current iterate and derivatives
        vars->xi(i) *= resfactors[i];
        vars->gradObj(i) /= resfactors[i];
        vars->gradLagrange(i) /= resfactors[i];
    }

    if (param->sparse){
        for (int i = 0; i < prob->nVar; i++){
            for (int k = vars->jacIndCol[i]; k < vars->jacIndCol[i+1]; k++){
                vars->jacNz[k] /= resfactors[i];
            }
        }
    }
    else{
        for (int i = 0; i < prob->nVar; i++){
            for (int k = 0; k < prob->nCon; k++){
                vars->constrJac(k,i) /= resfactors[i];
            }
        }
    }

    //Rescale past iteration data: Hessian(-approximation)s, variable and Lagrange gradient steps, scalar products
    if (!param->lim_mem){
        //For full memory rescale the current Hessians and the last variable/gradient step delta/gamma pair
        for (int iBlock = 0; iBlock < vars->nBlocks; iBlock++){
            for (int i = 0; i < vars->blockIdx[iBlock+1] - vars->blockIdx[iBlock]; i++){
                for (int j = 0; j <= i; j++){
                    vars->hess1[iBlock](i,j) /= resfactors[vars->blockIdx[iBlock] + i]*resfactors[vars->blockIdx[iBlock] + j];
                    if (vars->hess2 != nullptr){
                        vars->hess2[iBlock](i,j) /= resfactors[vars->blockIdx[iBlock] + i]*resfactors[vars->blockIdx[iBlock] + j];
                    }
                }
            }
        }
        for (int iBlock = 0; iBlock < vars->nBlocks; iBlock++){
            deltai.Submatrix(vars->deltaOld, vars->blockIdx[iBlock+1] - vars->blockIdx[iBlock], 1);
            vars->deltaNormSqOld(iBlock) = adotb(deltai, deltai);
        }
    }
    else{
        //For limited memory, rescale only exact Hessian blocks and all variable/gradient step delta/gamma pairs that are still used for updates
        if (param->exact_hess > 0){
            for (int iBlock = (vars->nBlocks - 1)*int(param->exact_hess == 1); iBlock < vars->nBlocks; iBlock++){
                for (int i = 0; i < vars->blockIdx[iBlock+1] - vars->blockIdx[iBlock]; i++){
                    for (int j = 0; j <= i; j++){
                        vars->hess1[iBlock](i,j) /= resfactors[vars->blockIdx[iBlock] + i] * resfactors[vars->blockIdx[iBlock] + j];
                    }
                }
            }
        }
    }
    
    nmem = std::min(stats->itCount, vars->dg_nsave);
    for (int k = 0; k < nmem; k++){
        pos = (vars->dg_pos - nmem + 1 + k + vars->dg_nsave)%vars->dg_nsave;
        for (int i = 0; i < prob->nVar; i++){
            vars->deltaMat(i, pos) *= resfactors[i];
            vars->gammaMat(i, pos) /= resfactors[i];
        }
        for (int iBlock = 0; iBlock < vars->nBlocks; iBlock++){
            deltai.Submatrix(vars->deltaMat, vars->blockIdx[iBlock+1] - vars->blockIdx[iBlock], 1, vars->blockIdx[iBlock], pos);
            vars->deltaNormSqMat(iBlock, pos) = adotb(deltai, deltai);
        }
    }

    return;
}


 }//namespace blockSQP