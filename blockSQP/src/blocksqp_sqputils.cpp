/*
 * blockSQP -- Sequential quadratic programming for problems with
 *             block-diagonal Hessian matrix.
 * Copyright (C) 2012-2015 by Dennis Janka <dennis.janka@iwr.uni-heidelberg.de>
 *
 * Licensed under the zlib license. See LICENSE for more details.
 */

/**
 * \file blocksqp_SQPutils.cpp
 * \author Reinhold Wittmann
 * \date 2012-2015
 *
 *  Utility methods of the SQPmethod class
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

void SQPmethod::calcLagrangeGradient(const Matrix &lambda, const Matrix &gradObj, double *jacNz, int *jacIndRow, int *jacIndCol,
                                      Matrix &gradLagrange, int flag){
    int iVar, iCon;

    // Objective gradient
    if( flag == 0 )
        for( iVar=0; iVar<prob->nVar; iVar++ )
            gradLagrange( iVar ) = gradObj( iVar );
    else if( flag == 1 )
        for( iVar=0; iVar<prob->nVar; iVar++ )
            gradLagrange( iVar ) = gradObj( iVar ) - gradLagrange( iVar );
    else
        gradLagrange.Initialize( 0.0 );

    // - lambdaT * constrJac
    for( iVar=0; iVar<prob->nVar; iVar++ )
        for( iCon=jacIndCol[iVar]; iCon<jacIndCol[iVar+1]; iCon++ )
            gradLagrange( iVar ) -= lambda( prob->nVar + jacIndRow[iCon] ) * jacNz[iCon];

    // - lambdaT * simpleBounds
    for( iVar=0; iVar<prob->nVar; iVar++ )
        gradLagrange( iVar ) -= lambda( iVar );
}


 
void SQPmethod::calcLagrangeGradient(const Matrix &lambda, const Matrix &gradObj, const Matrix &constrJac,
                                      Matrix &gradLagrange, int flag){
    int iVar, iCon;

    // Objective gradient
    if( flag == 0 )
        for( iVar=0; iVar<prob->nVar; iVar++ )
            gradLagrange( iVar ) = gradObj( iVar );
    else if( flag == 1 )
        for( iVar=0; iVar<prob->nVar; iVar++ )
            gradLagrange( iVar ) = gradObj( iVar ) - gradLagrange( iVar );
    else
        gradLagrange.Initialize( 0.0 );

    // - lambdaT * constrJac
    for( iVar=0; iVar<prob->nVar; iVar++ )
        for( iCon=0; iCon<prob->nCon; iCon++ )
            gradLagrange( iVar ) -= lambda( prob->nVar + iCon ) * constrJac( iCon, iVar );

    // - lambdaT * simpleBounds
    for( iVar=0; iVar<prob->nVar; iVar++ )
        gradLagrange( iVar ) -= lambda( iVar );
}


void SQPmethod::calcLagrangeGradient( Matrix &gradLagrange, int flag )
{
    if( param->sparse )
        calcLagrangeGradient( vars->lambda, vars->gradObj, vars->sparse_constrJac.nz.get(), vars->sparse_constrJac.row.get(), vars->sparse_constrJac.colind.get(), gradLagrange, flag );
    else
        calcLagrangeGradient( vars->lambda, vars->gradObj, vars->constrJac, gradLagrange, flag );
}


void SQPmethod::updateDeltaGammaData(){
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


/**
* Compute optimality conditions:
* ||gradLagrange(xi,lambda)||_infty / (1 + ||lambda||_infty) <= TOL
* and
* ||constrViolation||_infty / (1 + ||xi||_infty) <= TOL
*/
bool SQPmethod::calcOptTol(){

    // scaled norm of Lagrangian gradient

    calcLagrangeGradient( vars->gradLagrange, 0 );

    if (!param->automatic_scaling){
        vars->gradNorm = lInfVectorNorm( vars->gradLagrange );
        vars->tol = vars->gradNorm /( 1.0 + lInfVectorNorm( vars->lambda ) );
    }
    else{
        vars->gradNorm = 0;
        for (int i = 0; i < prob->nVar; i++){
            if (vars->gradNorm < std::abs(vars->gradLagrange(i)*scaled_prob->scaling_factors[i])) vars->gradNorm = std::abs(vars->gradLagrange(i)*scaled_prob->scaling_factors[i]);
        }
        vars->tol = vars->gradNorm /( 1.0 + lInfVectorNorm( vars->lambda ) );
    }

    // norm of constraint violation
    vars->cNorm  = lInfConstraintNorm(vars->xi, vars->constr, prob->lb_var, prob->ub_var, prob->lb_con, prob->ub_con);
    vars->cNormS = vars->cNorm /( 1.0 + lInfVectorNorm( vars->xi ) );

    if (vars->tol <= param->opt_tol && vars->cNormS <= param->feas_tol)
        return true;
    else
        return false;
}


void SQPmethod::set_iterate(const Matrix &xi, const Matrix &lambda, bool resetHessian){
    vars->xi = xi;
    if (param->automatic_scaling){
        for (int i = 0; i < prob->nVar; i++){
            vars->xi(i) *= scaled_prob->scaling_factors[i];
        }
    }
    vars->lambda = lambda;
    int infoEval;
    if (resetHessian) resetHessians();

    if (param->sparse)
        prob->evaluate(vars->xi, vars->lambda, &vars->obj, vars->constr, vars->gradObj,
                        vars->sparse_constrJac.nz.get(), vars->sparse_constrJac.row.get(), vars->sparse_constrJac.colind.get(), vars->hess1.get(), 1+param->exact_hess, &infoEval);
    else
        prob->evaluate(vars->xi, vars->lambda, &vars->obj, vars->constr, vars->gradObj,
                        vars->constrJac, vars->hess1.get(), 1+param->exact_hess, &infoEval);

    //Remove filter entries that dominate the set point
    std::set<std::pair<double,double>>::iterator iter = vars->filter.begin();
    std::set<std::pair<double,double>>::iterator iterToRemove;
    while (iter != vars->filter.end()){
        if (iter->first < vars->cNorm && iter->second < vars->obj){
            iterToRemove = iter;
            iter++;
            vars->filter.erase(iterToRemove);
        }
        else iter++;
    }
    augmentFilter(vars->cNorm, vars->obj);
    return;
}


Matrix SQPmethod::get_xi(){
    Matrix xi_unscaled(vars->xi);
    if (param->automatic_scaling){
        for (int i = 0; i < prob->nVar; i++){
            xi_unscaled(i)/= scaled_prob->scaling_factors[i];
        }
    }
    return xi_unscaled;
}

Matrix SQPmethod::get_lambda(){
    return vars->lambda;
}


void SQPmethod::get_xi(Matrix &xi_hold){
    if (param->automatic_scaling){
        for (int i = 0; i < prob->nVar; i++){
            xi_hold(i) = vars->xi(i)/scaled_prob->scaling_factors[i];
        }
    }
    else{
        for (int i = 0; i < prob->nVar; i++){
            xi_hold(i) = vars->xi(i);
        }
    }
    return;
}

void SQPmethod::get_lambda(Matrix &lambda_hold){
    for (int i = 0; i < prob->nVar + prob->nCon; i++){
        lambda_hold(i) = vars->lambda(i);
    }
    return;
}

void SQPmethod::get_lambdaQP(Matrix &lambdaQP_hold){
    for (int i = 0; i < prob->nVar + prob->nCon; i++){
        lambdaQP_hold(i) = vars->lambdaQP(i);
    }
    return;
}


void SQPmethod::printInfo( int printLevel )
{
    char hessString1[100];
    char hessString2[100];
    char globString[100];
    char qpString[100];

    if( printLevel == 0 )
        return;

    /* QP Solver */
    if( param->sparse == 0 )
        strcpy( qpString, "dense, reduced Hessian factorization" );
    else if( param->sparse == 1 )
        strcpy( qpString, "sparse, reduced Hessian factorization" );
    else if( param->sparse == 2 )
        strcpy( qpString, "sparse, Schur complement approach" );

    /* Globalization */
    if( param->enable_linesearch == 0 )
        strcpy( globString, "none (full step)" );
    else if( param->enable_linesearch == 1 )
        strcpy( globString, "filter line search" );

    /* Hessian approximation */
    if( param->block_hess && (param->hess_approx == 1 || param->hess_approx == 2) )
        strcpy( hessString1, "block " );
    else
        strcpy( hessString1, "" );

    if( param->lim_mem && (param->hess_approx == 1 || param->hess_approx == 2) )
        strcat( hessString1, "L-" );

    /* Fallback Hessian */
    if( param->hess_approx == 1 || param->hess_approx == 4 || (param->hess_approx == 6) )
    {
        strcpy( hessString2, hessString1 );

        /* Fallback Hessian update type */
        if( param->fallback_approx == 0 )
            strcat( hessString2, "Id" );
        else if( param->fallback_approx == 1 )
            strcat( hessString2, "SR1" );
        else if( param->fallback_approx == 2 )
            strcat( hessString2, "BFGS" );
        else if( param->fallback_approx == 4 )
            strcat( hessString2, "Finite differences" );

        /* Fallback Hessian scaling */
        if( param->fallback_sizing == 1 )
            strcat( hessString2, ", SP" );
        else if( param->fallback_sizing == 2 )
            strcat( hessString2, ", OL" );
        else if( param->fallback_sizing == 3 )
            strcat( hessString2, ", mean" );
        else if( param->fallback_sizing == 4 )
            strcat( hessString2, ", selective sizing" );
    }
    else
        strcpy( hessString2, "-" );

    /* First Hessian update type */
    if( param->hess_approx == 0 )
        strcat( hessString1, "Id" );
    else if( param->hess_approx == 1 )
        strcat( hessString1, "SR1" );
    else if( param->hess_approx == 2 )
        strcat( hessString1, "BFGS" );
    else if( param->hess_approx == 4 )
        strcat( hessString1, "Finite differences" );

    /* First Hessian scaling */
    if( param->sizing == 1 )
        strcat( hessString1, ", SP" );
    else if( param->sizing == 2 )
        strcat( hessString1, ", OL" );
    else if( param->sizing == 3 )
        strcat( hessString1, ", mean" );
    else if( param->sizing == 4 )
        strcat( hessString1, ", selective sizing" );

    printf( "\n+---------------------------------------------------------------+\n");
    printf( "| Starting blockSQP with the following algorithmic settings:    |\n");
    printf( "+---------------------------------------------------------------+\n");
    printf( "| qpOASES flavor            | %-34s|\n", qpString );
    printf( "| Globalization             | %-34s|\n", globString );
    printf( "| 1st Hessian approximation | %-34s|\n", hessString1 );
    printf( "| 2nd Hessian approximation | %-34s|\n", hessString2 );
    printf( "+---------------------------------------------------------------+\n\n");
}


 }//namespace blockSQP