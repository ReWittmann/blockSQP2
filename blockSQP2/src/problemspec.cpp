/*
 * blockSQP -- Sequential quadratic programming for problems with
 *             block-diagonal Hessian matrix.
 * Copyright (C) 2012-2015 by Dennis Janka <dennis.janka@iwr.uni-heidelberg.de>
 *
 * Licensed under the zlib license. See LICENSE for more details.
 */

/*
 * blockSQP2 -- A structure-exploiting nonlinear programming solver based
 *              on blockSQP by Dennis Janka.
 * Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
 * 
 * Licensed under the zlib license. See LICENSE for more details.
 */

/**
 * \file blocksqp_problemspec.cpp
 * \author Dennis Janka
 * \date 2012-2015
 *
 *  Implementation of problem-independent methods of ProblemSpec class.
 * 
 * \modifications
 *  \author Reinhold Wittmann
 *  \date 2023-2025
 */


#include <blockSQP2/problemspec.hpp>

namespace blockSQP2{
Problemspec::Problemspec(){}
Problemspec::~Problemspec(){}

void Problemspec::evaluate( const Matrix &xi, double *objval, Matrix &constr, int *info ){
    Matrix lambdaDummy, gradObjDummy;
    SymMatrix *hessDummy(nullptr);
    int dmode = 0;

    Matrix constrJacDummy;
    double *jacNzDummy(nullptr);
    int *jacIndRowDummy(nullptr), *jacIndColDummy(nullptr);
    *info = 0;

    // Try sparse version first
    evaluate(xi, lambdaDummy, objval, constr, gradObjDummy, jacNzDummy, jacIndRowDummy, jacIndColDummy, hessDummy, dmode, info);

    // If sparse version is not implemented, try dense version
    if (*info) evaluate(xi, lambdaDummy, objval, constr, gradObjDummy, constrJacDummy, hessDummy, dmode, info);
}


scaled_Problemspec::scaled_Problemspec(Problemspec *UNSCprob): unscaled_prob(UNSCprob), scaling_factors(new double[unscaled_prob->nVar]){
    nVar = unscaled_prob->nVar;
    nCon = unscaled_prob->nCon;
    nnz = unscaled_prob->nnz;
    objLo = unscaled_prob->objLo;
    objUp = unscaled_prob->objUp;
    lb_var = unscaled_prob->lb_var;
    ub_var = unscaled_prob->ub_var;
    lb_con = unscaled_prob->lb_con;
    ub_con = unscaled_prob->ub_con;
    nBlocks = unscaled_prob->nBlocks;
    blockIdx = unscaled_prob->blockIdx;
    vblocks = unscaled_prob->vblocks;
    n_vblocks = unscaled_prob->n_vblocks;
    cond = unscaled_prob->cond;
    
    for (int i = 0; i < nVar; i++){
        scaling_factors[i] = 1.0;
    }
    xi_unscaled.Dimension(nVar);
}

scaled_Problemspec::~scaled_Problemspec(){
    //delete[] scaling_factors;
}

void scaled_Problemspec::initialize(Matrix &xi, Matrix &lambda, Matrix &constrJac){
    unscaled_prob->initialize(xi, lambda, constrJac);
    for (int i = 0; i < nVar; i++){
        xi(i) *= scaling_factors[i];
        for (int j = 0; j < nCon; j++){
            constrJac(j,i) /= scaling_factors[i];
        }
    }
    return;
}

void scaled_Problemspec::initialize(Matrix &xi, Matrix &lambda, double *jacNz, int *jacIndRow, int *jacIndCol){
    unscaled_prob->initialize(xi, lambda, jacNz, jacIndRow, jacIndCol);
    for (int i = 0; i < nVar; i++){
        xi(i) *= scaling_factors[i];
        for (int j = jacIndCol[i]; j < jacIndCol[i+1]; j++){
            jacNz[j] /= scaling_factors[i]; 
        }
    }
    return;
}

void scaled_Problemspec::evaluate(const Matrix &xi, const Matrix &lambda, double *objval, Matrix &constr, Matrix &gradObj, Matrix &constrJac, SymMatrix *hess, int dmode, int *info){
    for (int i = 0; i < nVar; i++){
        xi_unscaled(i) = xi(i)/scaling_factors[i];
    }
    unscaled_prob->evaluate(xi_unscaled, lambda, objval, constr, gradObj, constrJac, hess, dmode, info);
    if (dmode > 0){
        for (int i = 0; i < nVar; i++){
            gradObj(i) /= scaling_factors[i];
            for (int j = 0; j < nCon; j++){
                constrJac(j,i) /= scaling_factors[i];
            }
        }
        if (dmode ==  2){
            for (int i = 0; i < blockIdx[nBlocks] - blockIdx[nBlocks-1]; i++){
                for (int j = 0; j <= i; j++){
                    hess[nBlocks - 1](i, j) /= scaling_factors[blockIdx[nBlocks-1] + i]*scaling_factors[blockIdx[nBlocks-1] + j];
                }
            }
        }
        else if (dmode > 2){
            for (int k = 0; k < nBlocks; k++){
                for (int i = 0; i < blockIdx[k+1] - blockIdx[k]; i++){
                    for (int j = 0; j <= i; j++){
                        hess[k](i, j) /= scaling_factors[blockIdx[k] + i]*scaling_factors[blockIdx[k] + j];
                    }
                }
            }
        }
    }
    return;
}

void scaled_Problemspec::evaluate(const Matrix &xi, const Matrix &lambda, double *objval, Matrix &constr, Matrix &gradObj, double *jacNz, int *jacIndRow, int *jacIndCol, SymMatrix *hess, int dmode, int *info){
    for (int i = 0; i < nVar; i++){
        xi_unscaled(i) = xi(i)/scaling_factors[i];
    }
    unscaled_prob->evaluate(xi_unscaled, lambda, objval, constr, gradObj, jacNz, jacIndRow, jacIndCol, hess, dmode, info);

    if (dmode > 0){
        for (int i = 0; i < nVar; i++){
            gradObj(i) /= scaling_factors[i];
            for (int j = jacIndCol[i]; j < jacIndCol[i+1]; j++){
                jacNz[j] /= scaling_factors[i];
            }
        }
        if (dmode ==  2){
            for (int i = 0; i < blockIdx[nBlocks] - blockIdx[nBlocks-1]; i++){
                for (int j = 0; j <= i; j++){
                    hess[nBlocks - 1](i, j) /= scaling_factors[blockIdx[nBlocks-1] + i]*scaling_factors[blockIdx[nBlocks-1] + j];
                }
            }
        }
        else if (dmode > 2){
            for (int k = 0; k < nBlocks; k++){
                for (int i = 0; i < blockIdx[k+1] - blockIdx[k]; i++){
                    for (int j = 0; j <= i; j++){
                        //hess[k](i, j) /= scaling_factors[blockIdx[k] + i]*scaling_factors[blockIdx[k] + j];
                        hess[k](i, j) = hess[k](i,j)/(scaling_factors[blockIdx[k] + i]*scaling_factors[blockIdx[k] + j]);
                    }
                }
            }
        }
    }
    return;
}

void scaled_Problemspec::evaluate(const Matrix &xi, double *objval, Matrix &constr, int *info){
    for (int i = 0; i < nVar; i++){
        xi_unscaled(i) = xi(i)/scaling_factors[i];
    }
    unscaled_prob->evaluate(xi_unscaled, objval, constr, info);
    return;
}

void scaled_Problemspec::reduceConstrVio(Matrix &xi, int *info){
    for (int i = 0; i < nVar; i++){
        xi_unscaled(i) = xi(i)/scaling_factors[i];
    }
    unscaled_prob->reduceConstrVio(xi_unscaled, info);
    if (!(*info)){
        for (int i = 0; i < nVar; i++){
            xi(i) = xi_unscaled(i)*scaling_factors[i];
        }
    }
}

void scaled_Problemspec::set_scale(const double *const scaleFacs){
    for (int i = 0; i < nVar; i++){
        scaling_factors[i] = scaleFacs[i];
        lb_var(i) = unscaled_prob->lb_var(i)*scaling_factors[i];
        ub_var(i) = unscaled_prob->ub_var(i)*scaling_factors[i];
    }
    return;
}

void scaled_Problemspec::rescale(const double *const scaleFacs){
    for (int i = 0; i < nVar; i++){
        scaling_factors[i] *= scaleFacs[i];
        lb_var(i) *= scaleFacs[i];
        ub_var(i) *= scaleFacs[i];
    }
    return;
}

void scaled_Problemspec::stepModification(Matrix &xi, Matrix &lambda, int *info){
    for (int i = 0; i < nVar; i++){
        xi_unscaled(i) = xi(i)/scaling_factors[i];
    }
    unscaled_prob->stepModification(xi_unscaled, lambda, info);
    if (!(*info)){
        for (int i = 0; i < nVar; i++){
            xi(i) = xi_unscaled(i)*scaling_factors[i];
        }
    }
    return;
}







} // namespace blockSQP2
