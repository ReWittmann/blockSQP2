/*
 * blockSQP -- Sequential quadratic programming for problems with
 *             block-diagonal Hessian matrix.
 * Copyright (C) 2012-2015 by Dennis Janka <dennis.janka@iwr.uni-heidelberg.de>
 *
 * Licensed under the zlib license. See LICENSE for more details.
 */

/**
 * \file blocksqp_qp.cpp
 * \author Dennis Janka
 * \date 2012-2015
 *
 *  Implementation of methods of SQPmethod class associated with
 *  solution of the quadratic subproblems.
 *
 */

#include "blocksqp_iterate.hpp"
#include "blocksqp_options.hpp"
#include "blocksqp_stats.hpp"
#include "blocksqp_method.hpp"
#include "blocksqp_general_purpose.hpp"
#include <iostream>
#include <chrono>
#include <fstream>
#include <cmath>

namespace blockSQP
{

void SQPmethod::computeNextHessian( int idx, int maxQP )
{
    // Compute fallback update only once
    if( idx == 1 )
    {
        // Switch storage
        vars->hess = vars->hess2;

        // If last block contains exact Hessian, we need to copy it
        if( param->whichSecondDerv == 1 )
            for( int i=0; i<vars->hess[prob->nBlocks-1].M(); i++ )
                for( int j=i; j<vars->hess[prob->nBlocks-1].N(); j++ )
                    vars->hess2[prob->nBlocks-1]( i,j ) = vars->hess1[prob->nBlocks-1]( i,j );

        // Limited memory: compute fallback update only when needed
        if( param->hessLimMem )
        {
            stats->itCount--;
            int hessDampSave = param->hessDamp;
            param->hessDamp = 1;
            calcHessianUpdateLimitedMemory( param->fallbackUpdate, param->fallbackScaling );
            param->hessDamp = hessDampSave;
            stats->itCount++;
        }
        /* Full memory: both updates must be computed in every iteration
         * so switching storage is enough */
    }

    // 'Nontrivial' convex combinations
    if( maxQP > 2 )
    {
        /* Convexification parameter: mu_l = l / (maxQP-1).
         * Compute it only in the first iteration, afterwards update
         * by recursion: mu_l/mu_(l-1) */
        double idxF = (double) idx;
        double mu = (idx==1) ? 1.0 / (maxQP-1) : idxF / (idxF - 1.0);
        double mu1 = 1.0 - mu;
        for( int iBlock=0; iBlock<vars->nBlocks; iBlock++ )
            for( int i=0; i<vars->hess[iBlock].M(); i++ )
                for( int j=i; j<vars->hess[iBlock].N(); j++ )
                {
                    vars->hess2[iBlock]( i,j ) *= mu;
                    vars->hess2[iBlock]( i,j ) += mu1 * vars->hess1[iBlock]( i,j );
                }
    }
}


/**
 * Inner loop of SQP algorithm:
 * Solve a sequence of QPs until pos. def. assumption (G3*) is satisfied.
 */
int SQPmethod::solveQP( Matrix &deltaXi, Matrix &lambdaQP, bool matricesChanged )
{

    std::ofstream debug;
    debug.open("/home/reinhold/cond_debug.txt", std::ios_base::app);
    debug << "Inside solveQP\n" << std::flush;

    Matrix jacT;
    int maxQP, l;
    if( param->globalization == 1 &&
        param->hessUpdate == 1 &&
        stats->itCount > 1 )
    {
        maxQP = param->maxConvQP + 1;
    }
    else
        maxQP = 1;


    /*
     * Prepare for qpOASES
     */

    // Setup QProblem data

    delete A_qp;
    if(param->sparseQP){
        A_qp = new qpOASES::SparseMatrix( prob->nCon, prob->nVar,
                    vars->jacIndRow, vars->jacIndCol, vars->jacNz );
    }
    else{
        // transpose Jacobian (qpOASES needs row major arrays)
        Transpose( vars->constrJac, jacT );
        A_qp = new qpOASES::DenseMatrix( prob->nCon, prob->nVar, prob->nVar, jacT.ARRAY() );
    }

    double *g, *lb, *ub, *lbA, *ubA;
    g = vars->gradObj.array;
    lb = vars->delta_lb_var.array;
    ub = vars->delta_ub_var.array;
    lbA = vars->delta_lb_con.array;
    ubA = vars->delta_ub_con.array;


    // qpOASES options
    qpOASES::Options opts;
    if(maxQP > 1)
        opts.enableInertiaCorrection = qpOASES::BT_FALSE;
    opts.enableEqualities = qpOASES::BT_TRUE;
    opts.initialStatusBounds = qpOASES::ST_INACTIVE;

    switch(param->qpOASES_print_level){
        case 0:
            opts.printLevel = qpOASES::PL_NONE;
            break;
        case 1:
            opts.printLevel = qpOASES::PL_LOW;
            break;
        case 2:
            opts.printLevel = qpOASES::PL_MEDIUM;
            break;
        case 3:
            opts.printLevel = qpOASES::PL_HIGH;
            break;
    }

    //opts.printLevel = qpOASES::PL_MEDIUM; //PL_LOW, PL_HIGH, PL_MEDIUM, PL_NONE
    opts.numRefinementSteps = 2;
    opts.epsLITests =  2.2204e-08;
    opts.terminationTolerance = param->qpOASES_terminationTolerance;

    qp->setOptions( opts );

    if( maxQP > 1 )
    {
        // Store last successful QP in temporary storage
        (*qpSave) = *qp;
        /** \todo Storing the active set would be enough but then the QP object
         *        must be properly reset after unsuccessful (SR1-)attempt.
         *        Moreover, passing a guessed active set doesn't yield
         *        exactly the same result as hotstarting a QP. This has
         *        something to do with how qpOASES handles user-given
         *        active sets (->check qpOASES source code). */
    }

    // Other variables for qpOASES
    double cpuTime = param->maxTimeQP;
    int maxIt = param->maxItQP;
    qpOASES::SolutionAnalysis solAna;
    qpOASES::returnValue ret;

    /*
     * QP solving loop for convex combinations (sequential)
     */
    for( l=0; l<maxQP; l++ )
    {
        /*
         * Compute a new Hessian
         */
        if( l > 0 ){// If the solution of the first QP was rejected, consider second Hessian
            stats->qpResolve++;
            *qp = *qpSave;

            computeNextHessian( l, maxQP );

        }

        if( l == maxQP-1 )
        {// Enable inertia correction for supposedly convex QPs, just in case
            opts.enableInertiaCorrection = qpOASES::BT_TRUE;
            qp->setOptions( opts );
        }

        /*
         * Prepare the current Hessian for qpOASES
         */
        delete H_qp;
        if(param->sparseQP){
            // Convert block-Hessian to sparse format
            vars->convertHessian( prob->nVar, prob->nBlocks, param->eps, vars->hess, vars->hessNz,
                              vars->hessIndRow, vars->hessIndCol, vars->hessIndLo );
            H_qp = new qpOASES::SymSparseMat( prob->nVar, prob->nVar,
                                       vars->hessIndRow, vars->hessIndCol, vars->hessNz );
            dynamic_cast<qpOASES::SymSparseMat*>(H_qp)->createDiagInfo();
        }
        else{
            // Convert block-Hessian to double array
            vars->convertHessian( prob, param->eps, vars->hess );
            H_qp = new qpOASES::SymDenseMat( prob->nVar, prob->nVar, prob->nVar, vars->hessNz );
        }

        /*
         * Call qpOASES
         */
        if( param->debugLevel > 2 ) stats->dumpQPCpp( prob, vars, qp, param->sparseQP );

        maxIt = param->maxItQP;
        cpuTime = param->maxTimeQP;
        if( (qp->getStatus() == qpOASES::QPS_HOMOTOPYQPSOLVED ||
            qp->getStatus() == qpOASES::QPS_SOLVED ) && vars->use_homotopy)
        {
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            debug << "Starting hotstart solution of QP\n" << std::flush;

            ret = qp->hotstart( H_qp, g, A_qp, lb, ub, lbA, ubA, maxIt, &cpuTime );

            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << "Solved hotstarted QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms\n";
            std::cout << "ret is " << ret << "\n";
            debug << "Finished hotstart solution of QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() <<" ms\n" << std::flush;
        }
        else
        {
            debug << "Starting solution of initial QP\n" << std::flush;
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            ret = qp->init( H_qp, g, A_qp, lb, ub, lbA, ubA, maxIt, &cpuTime );
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << "Solved initial QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms\n";
            std::cout << "ret is " << ret << "\n";
            debug << "Solved initial QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() <<" ms\n" << std::flush;
            vars->use_homotopy = true;
        }

        /*
         * Check assumption (G3*) if nonconvex QP was solved
         */
        if( l < maxQP-1)
        {
            if( ret == qpOASES::SUCCESSFUL_RETURN )
            {
                if( param->sparseQP == 2 )
                    ret = solAna.checkCurvatureOnStronglyActiveConstraints( dynamic_cast<qpOASES::SQProblemSchur*>(qp) );
                else
                    ret = solAna.checkCurvatureOnStronglyActiveConstraints( qp );
            }

            if( ret == qpOASES::SUCCESSFUL_RETURN )
            {// QP was solved successfully and curvature is positive after removing bounds
                stats->qpIterations = maxIt + 1;
                break; // Success!
            }
            else
            {// QP solution is rejected, save statistics
                if( ret == qpOASES::RET_SETUP_AUXILIARYQP_FAILED )
                    stats->qpIterations2++;
                else
                    stats->qpIterations2 += maxIt + 1;
                stats->rejectedSR1++;
            }
        }
        else // Convex QP was solved, no need to check assumption (G3*)
            stats->qpIterations += maxIt + 1;

    } // End of QP solving loop


    /*
     * Post-processing
     */

    // Get solution from qpOASES
    qp->getPrimalSolution( deltaXi.ARRAY() );
    qp->getDualSolution( lambdaQP.ARRAY() );

    //vars->qpObj = qp->getObjVal();


    // Compute constrJac*deltaXi, need this for second order correction step
    if( param->sparseQP )
        Atimesb( vars->jacNz, vars->jacIndRow, vars->jacIndCol, deltaXi, vars->AdeltaXi );
    else
        Atimesb( vars->constrJac, deltaXi, vars->AdeltaXi );

    // Print qpOASES error code, if any
    if( ret != qpOASES::SUCCESSFUL_RETURN)
        printf( "qpOASES error message: \"%s\"\n",
                qpOASES::getGlobalMessageHandler()->getErrorCodeMessage( ret ) );

    // Point Hessian again to the first Hessian
    vars->hess = vars->hess1;

    /* For full-memory Hessian: Restore fallback Hessian if convex combinations
     * were used during the loop */
    if( !param->hessLimMem && maxQP > 2)
    {
        double mu = 1.0 / ((double) l);
        double mu1 = 1.0 - mu;
        int nBlocks = (param->whichSecondDerv == 1) ? vars->nBlocks-1 : vars->nBlocks;
        for( int iBlock=0; iBlock<nBlocks; iBlock++ )
            for( int i=0; i<vars->hess[iBlock].M(); i++ )
                for( int j=i; j<vars->hess[iBlock].N(); j++ )
                {
                    vars->hess2[iBlock]( i,j ) *= mu;
                    vars->hess2[iBlock]( i,j ) += mu1 * vars->hess1[iBlock]( i,j );
                }
    }

    debug.close();

    /* Return code depending on qpOASES returnvalue
     * 0: Success
     * 1: Maximum number of iterations reached
     * 2: Unbounded
     * 3: Infeasible
     * 4: Other error */
    if( ret == qpOASES::SUCCESSFUL_RETURN )
        return 0;
    else if( ret == qpOASES::RET_MAX_NWSR_REACHED )
        return 1;
    else if( ret == qpOASES::RET_HESSIAN_NOT_SPD ||
             ret == qpOASES::RET_HESSIAN_INDEFINITE ||
             ret == qpOASES::RET_INIT_FAILED_UNBOUNDEDNESS ||
             ret == qpOASES::RET_QP_UNBOUNDED ||
             ret == qpOASES::RET_HOTSTART_STOPPED_UNBOUNDEDNESS ){
        return 2;}
    else if( ret == qpOASES::RET_INIT_FAILED_INFEASIBILITY ||
             ret == qpOASES::RET_QP_INFEASIBLE ||
             ret == qpOASES::RET_HOTSTART_STOPPED_INFEASIBILITY ){
        return 3;}
    else{
	std::cout << "RET is " << ret << "\n";
        return 4;}
}



int SQPmethod::solve_SOC_QP(Matrix &deltaXi, Matrix &lambdaQP){

    std::ofstream debug;
    debug.open("/home/reinhold/cond_debug.txt", std::ios_base::app);
    debug << "Inside solve_SOC_QP\n" << std::flush;

    // Setup QProblem data
    double *g, *lb, *ub, *lbA, *ubA;
    g = vars->gradObj.array;
    lb = vars->delta_lb_var.array;
    ub = vars->delta_ub_var.array;
    lbA = vars->delta_lb_con.array;
    ubA = vars->delta_ub_con.array;

    // qpOASES options
    qpOASES::Options opts;
    opts.enableEqualities = qpOASES::BT_TRUE;
    opts.initialStatusBounds = qpOASES::ST_INACTIVE;

    switch(param->qpOASES_print_level){
        case 0:
            opts.printLevel = qpOASES::PL_NONE;
            break;
        case 1:
            opts.printLevel = qpOASES::PL_LOW;
            break;
        case 2:
            opts.printLevel = qpOASES::PL_MEDIUM;
            break;
        case 3:
            opts.printLevel = qpOASES::PL_HIGH;
            break;
    }

    opts.numRefinementSteps = 2;
    opts.epsLITests =  2.2204e-08;
    opts.terminationTolerance = param->qpOASES_terminationTolerance;
    opts.enableInertiaCorrection = qpOASES::BT_TRUE;

    qp->setOptions( opts );


    // Other variables for qpOASES
    double cpuTime = 0.1*param->maxTimeQP;
    int maxIt = 0.1*param->maxItQP;
    qpOASES::SolutionAnalysis solAna;
    qpOASES::returnValue ret;


    if( param->debugLevel > 2 ) stats->dumpQPCpp( prob, vars, qp, param->sparseQP );

    debug << "Starting solution of SOC-QP\n" << std::flush;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    ret = qp->hotstart( g, lb, ub, lbA, ubA, maxIt, &cpuTime );
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Solved SOC in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms\n";
    debug << "Finished solution of SOC-QP\n" << std::flush;

    /*
     * Check assumption (G3*) if nonconvex QP was solved
     */
    //if( l < maxQP-1 && matricesChanged ){

    if( ret == qpOASES::SUCCESSFUL_RETURN ){
        if( param->sparseQP == 2 )
            ret = solAna.checkCurvatureOnStronglyActiveConstraints( dynamic_cast<qpOASES::SQProblemSchur*>(qp) );
        else
            ret = solAna.checkCurvatureOnStronglyActiveConstraints( qp );
    }
    if( ret == qpOASES::SUCCESSFUL_RETURN ){
        // QP was solved successfully and curvature is positive after removing bounds
        stats->qpIterations = maxIt + 1;
    }

    // Get solution from qpOASES
    qp->getPrimalSolution( deltaXi.ARRAY() );
    qp->getDualSolution( lambdaQP.ARRAY() );

    // Compute constrJac*deltaXi, need this for second order correction step
    if( param->sparseQP )
        Atimesb( vars->jacNz, vars->jacIndRow, vars->jacIndCol, deltaXi, vars->AdeltaXi );
    else
        Atimesb( vars->constrJac, deltaXi, vars->AdeltaXi );

    // Print qpOASES error code, if any
    if( ret != qpOASES::SUCCESSFUL_RETURN)
        printf( "qpOASES error message: \"%s\"\n",
                qpOASES::getGlobalMessageHandler()->getErrorCodeMessage( ret ) );

    // Point Hessian again to the first Hessian
    vars->hess = vars->hess1;

    debug.close();

    if( ret == qpOASES::SUCCESSFUL_RETURN )
        return 0;
    else if( ret == qpOASES::RET_MAX_NWSR_REACHED )
        return 1;
    else if( ret == qpOASES::RET_HESSIAN_NOT_SPD ||
             ret == qpOASES::RET_HESSIAN_INDEFINITE ||
             ret == qpOASES::RET_INIT_FAILED_UNBOUNDEDNESS ||
             ret == qpOASES::RET_QP_UNBOUNDED ||
             ret == qpOASES::RET_HOTSTART_STOPPED_UNBOUNDEDNESS ){
        return 2;}
    else if( ret == qpOASES::RET_INIT_FAILED_INFEASIBILITY ||
             ret == qpOASES::RET_QP_INFEASIBLE ||
             ret == qpOASES::RET_HOTSTART_STOPPED_INFEASIBILITY ){
        return 3;}
    else{
	std::cout << "RET is " << ret << "\n";
        return 4;}
}


/**
 * Set bounds on the step (in the QP), either according
 * to variable bounds in the NLP or according to
 * trust region box radius
 */
void SQPmethod::updateStepBounds( bool soc )
{
    int nVar = prob->nVar;
    int nCon = prob->nCon;

    // Bounds on step
    for(int i = 0; i < nVar; i++){
        if(prob->lb_var(i) != param->inf){
            vars->delta_lb_var(i) = prob->lb_var(i) - vars->xi(i);
        }
        else{
            vars->delta_lb_var(i) = param->inf;
        }

        if(prob->ub_var(i) != param->inf){
            vars->delta_ub_var(i) = prob->ub_var(i) - vars->xi(i);
        }
        else{
            vars->delta_ub_var(i) = param->inf;
        }
    }

    // Bounds on linearized constraints
    for(int i = 0; i < nCon; i++){
        if( prob->lb_con(i) != param->inf ){
            vars->delta_lb_con(i) = prob->lb_con(i) - vars->constr(i);
            if(soc){
                vars->delta_lb_con(i) += vars->AdeltaXi(i);
            }
        }
        else{
            vars->delta_lb_con(i) = param->inf;
        }

        if(prob->ub_con(i) != param->inf){
            vars->delta_ub_con(i) = prob->ub_con(i) - vars->constr(i);
            if(soc){
                vars->delta_ub_con(i) += vars->AdeltaXi(i);
            }
        }
        else{
            vars->delta_ub_con(i) = param->inf;
        }
    }
}


///////////////////////////////////////////////////Subclass methods

int SCQPmethod::solveQP( Matrix &deltaXi, Matrix &lambdaQP, bool matricesChanged ){

    SCQPiterate *c_vars = dynamic_cast<SCQPiterate*>(vars);

    std::ofstream debug;
    debug.open("/home/reinhold/cond_debug.txt", std::ios_base::app);
    debug << "Inside solveQP\n" << std::flush;

    Matrix jacT;
    int maxQP, l;
    if (param->globalization == 1 && param->hessUpdate == 1 && stats->itCount > 1){
        maxQP = param->maxConvQP + 1;
    }
    else
        maxQP = 1;

    //Condense the QP
    cond->full_condense(c_vars->gradObj, c_vars->Jacobian, c_vars->hess,
        c_vars->delta_lb_var, c_vars->delta_ub_var, c_vars->delta_lb_con, c_vars->delta_ub_con,
            c_vars->condensed_h, c_vars->condensed_Jacobian, c_vars->condensed_hess, c_vars->condensed_lb_var, c_vars->condensed_ub_var, c_vars->condensed_lb_con, c_vars->condensed_ub_con);

    //Update qpOASES-jacobian to new condensed jacobian
    delete A_qp;
    A_qp = new qpOASES::SparseMatrix(cond->condensed_num_cons, cond->condensed_num_vars,
            c_vars->condensed_Jacobian.row, c_vars->condensed_Jacobian.colind, c_vars->condensed_Jacobian.nz);


    //qpOASES defines +-1e20 as +-infinity, so set absent bounds accordingly
    for (int i = 0; i < cond->condensed_num_vars; i++){
        if (std::isinf(c_vars->condensed_lb_var(i)))
            c_vars->condensed_lb_var(i) = -1e20;
        if (std::isinf(c_vars->condensed_ub_var(i)))
            c_vars->condensed_ub_var(i) = 1e20;
    }
    for (int i = 0; i < cond->condensed_num_cons; i++){
        if (std::isinf(c_vars->condensed_lb_con(i)))
            c_vars->condensed_lb_con(i) = -1e20;
        if (std::isinf(c_vars->condensed_ub_con(i)))
            c_vars->condensed_ub_con(i) = 1e20;
    }

    double *g, *lb, *ub, *lbA, *ubA;
    g = c_vars->condensed_h.array;
    lb = c_vars->condensed_lb_var.array;
    ub = c_vars->condensed_ub_var.array;
    lbA = c_vars->condensed_lb_con.array;
    ubA = c_vars->condensed_ub_con.array;


    // qpOASES options
    qpOASES::Options opts;
    if(maxQP > 1)
        opts.enableInertiaCorrection = qpOASES::BT_FALSE;
    else
        opts.enableInertiaCorrection = qpOASES::BT_TRUE;
    opts.enableEqualities = qpOASES::BT_TRUE;
    opts.initialStatusBounds = qpOASES::ST_INACTIVE;

    switch(param->qpOASES_print_level){
        case 0:
            opts.printLevel = qpOASES::PL_NONE;
            break;
        case 1:
            opts.printLevel = qpOASES::PL_LOW;
            break;
        case 2:
            opts.printLevel = qpOASES::PL_MEDIUM;
            break;
        case 3:
            opts.printLevel = qpOASES::PL_HIGH;
            break;
    }

    //opts.printLevel = qpOASES::PL_MEDIUM; //PL_LOW, PL_HIGH, PL_MEDIUM, PL_NONE
    opts.numRefinementSteps = 2;
    opts.epsLITests =  2.2204e-08;

    qp->setOptions( opts );

    if( maxQP > 1 )
    {
        // Store last successful QP in temporary storage
        (*qpSave) = *qp;
        /** \todo Storing the active set would be enough but then the QP object
         *        must be properly reset after unsuccessful (SR1-)attempt.
         *        Moreover, passing a guessed active set doesn't yield
         *        exactly the same result as hotstarting a QP. This has
         *        something to do with how qpOASES handles user-given
         *        active sets (->check qpOASES source code). */
    }

    // Other variables for qpOASES
    double cpuTime = param->maxTimeQP;
    int maxIt = param->maxItQP;
    qpOASES::SolutionAnalysis solAna;
    qpOASES::returnValue ret;


    for (l = 0; l < maxQP; l++){
        if( l > 0 ){
            stats->qpResolve++;
            *qp = *qpSave;

            computeNextHessian( l, maxQP );

            cond->new_hessian_condense(c_vars->hess, c_vars->condensed_h, c_vars->condensed_hess);
            g = c_vars->condensed_h.array;
        }

        if( l == maxQP-1 )
        {// Enable inertia correction for supposedly convex QPs, just in case
            opts.enableInertiaCorrection = qpOASES::BT_TRUE;
            qp->setOptions( opts );
        }

        /*
         * Prepare the current Hessian for qpOASES
         */

        c_vars->convertHessian(cond->condensed_num_vars, cond->condensed_num_hessblocks, param->eps, c_vars->condensed_hess,
                        c_vars->condensed_hess_nz, c_vars->condensed_hess_row, c_vars->condensed_hess_colind, c_vars->condensed_hess_loind);

        delete H_qp;
        H_qp = new qpOASES::SymSparseMat(cond->condensed_num_vars, cond->condensed_num_vars,
                                    c_vars->condensed_hess_row, c_vars->condensed_hess_colind, c_vars->condensed_hess_nz);
        dynamic_cast<qpOASES::SymSparseMat*>(H_qp)->createDiagInfo();

        /*
         * Call qpOASES
         */
        if( param->debugLevel > 2 ) stats->dumpQPCpp( prob, vars, qp, param->sparseQP );

        maxIt = param->maxItQP;
        cpuTime = param->maxTimeQP;
        if( (qp->getStatus() == qpOASES::QPS_HOMOTOPYQPSOLVED ||
            qp->getStatus() == qpOASES::QPS_SOLVED ) && c_vars->use_homotopy){

            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            debug << "Starting hotstart solution of QP\n" << std::flush;

            ret = qp->hotstart(H_qp, g, A_qp, lb, ub, lbA, ubA, maxIt, &cpuTime);
            std::cout << "cpuTime is " << cpuTime << "\n";

            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << "Solved hotstarted QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms\n";
            std::cout << "ret is " << ret << "\n";
            debug << "Finished hotstart solution of QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() <<" ms\n" << std::flush;
        }
        else{
            debug << "Starting solution of initial QP\n" << std::flush;
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            ret = qp->init( H_qp, g, A_qp, lb, ub, lbA, ubA, maxIt, &cpuTime );
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << "Solved initial QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms\n";
            std::cout << "ret is " << ret << "\n";
            debug << "Solved initial QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() <<" ms\n" << std::flush;
            c_vars->use_homotopy = true;
        }

        /*
         * Check assumption (G3*) if nonconvex QP was solved
         */
        if( l < maxQP-1){
            if( ret == qpOASES::SUCCESSFUL_RETURN ){
                if( param->sparseQP == 2 )
                    ret = solAna.checkCurvatureOnStronglyActiveConstraints( dynamic_cast<qpOASES::SQProblemSchur*>(qp) );
                else
                    ret = solAna.checkCurvatureOnStronglyActiveConstraints( qp );
                std::cout << "Curvature checked, ret is " << ret << "\n";
            }

            if( ret == qpOASES::SUCCESSFUL_RETURN ){
                // QP was solved successfully and curvature is positive after removing bounds
                stats->qpIterations = maxIt + 1;

                //copy successfully solved QP to initialize QP with added bounds
                (*qp_nb) = *qp;
                (*qp_nb_2) = *dynamic_cast<qpOASES::SQProblemSchur*>(qp);

                //break; // Success!
            }
            else
            {// QP solution is rejected, save statistics
                if( ret == qpOASES::RET_SETUP_AUXILIARYQP_FAILED ){
                    std::cout << "Inertia condition violated, convexifying hessian\n";
                }
                stats->qpIterations2++;
                stats->rejectedSR1++;
            }
        }
        else{ // Convex QP was solved, no need to check assumption (G3*)
            stats->qpIterations += maxIt + 1;

            //copy successfully solved QP to initialize QP with added bounds
            //(*qp_nb) = *dynamic_cast<qpOASES::SQProblemSchur*>(qp);
            (*qp_nb) = *qp;
            (*qp_nb_2) = *dynamic_cast<qpOASES::SQProblemSchur*>(qp);
        }


        //Check if some dependent variable bounds are violated and - if they are - add them to the QP and solve again

        bool solution_found = false;
        if (ret == qpOASES::SUCCESSFUL_RETURN){
            solution_found = true;

            // Get solution from qpOASES
            qp->getPrimalSolution(c_vars->deltaXi_cond.array);
            qp->getDualSolution(c_vars->lambdaQP_cond.array);
            cond->recover_var_mult(c_vars->deltaXi_cond, c_vars->lambdaQP_cond, deltaXi, lambdaQP);

            bool found_direction;
            int ind;
            int ind_1;
            int ind_2;
            int ind_3;
            int num_dep_vars = cond->num_vars - cond->condensed_num_vars;
            int vio_count;

            double cpuTime_ref;

            std::cout << "Checking dependent variable bounds violation\n";
            for (int k = 0; k < param->max_bound_refines; k++){
                //Find dependent variable bounds that are violated at new point, add them to QP and solve again
                found_direction = true;
                ind_1 = 0;
                ind_2 = cond->num_true_cons;
                ind_3 = 0;
                vio_count = 0;

                for (int i = 0; i < cond->num_vblocks; i++){
                    //Iterate over dependent variable blocks
                    if (cond->vblocks[i].dependent){
                        for (int j = 0; j < cond->vblocks[i].size; j++){
                            ind = ind_1 + j;
                            if (c_vars->xi(ind) + deltaXi(ind) < prob->lb_var(ind) - param->dep_bound_tolerance || c_vars->xi(ind) + deltaXi(ind) > prob->ub_var(ind) + param->dep_bound_tolerance){
                                vio_count++;
                                found_direction = false;
                                c_vars->condensed_lb_con(ind_2 + j) = cond->lb_dep_var(ind_3 + j);
                                c_vars->condensed_ub_con(ind_2 + j) = cond->ub_dep_var(ind_3 + j);
                            }
                        }
                        ind_2 += cond->vblocks[i].size;
                        ind_3 += cond->vblocks[i].size;
                    }
                    ind_1 += cond->vblocks[i].size;
                }
                if (found_direction){
                    std::cout << "All dependent variable bounds are respected, exiting bound refinement\n";
                    break;
                }
                std::cout << "Bounds violated by " << vio_count << " dependent variables, adding their bounds to the QP\n";
                debug << "Starting solution of bound refined QP\n" << std::flush;
                maxIt = param->maxItQP;
                cpuTime_ref = std::max(cpuTime, 0.5 * param->maxTimeQP);

                qp_nb->setOptions(opts);
                qp_nb_2->setOptions(opts);

                //Bounds matrices were modified in place, so pointers lb, ub, lbA, ubA need not be updated
                std::chrono::steady_clock::time_point begin_ = std::chrono::steady_clock::now();
                ret = qp_nb->hotstart(g, lb, ub, lbA, ubA, maxIt, &cpuTime_ref);
                std::cout << "ret is " << ret << "\n";
                std::chrono::steady_clock::time_point end_ = std::chrono::steady_clock::now();
                std::cout << "Solved bound refined QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin_).count() << "ms\n";
                debug << "Finished solution of bound refined QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin_).count() << "ms\n" << std::flush;

                /////////////////////////
                maxIt = param->maxItQP;
                cpuTime_ref = std::max(cpuTime, 0.5 * param->maxTimeQP);
                begin_ = std::chrono::steady_clock::now();
                qpOASES::returnValue ret_2 = qp_nb_2->hotstart(g, lb, ub, lbA, ubA, maxIt, &cpuTime_ref);
                end_ = std::chrono::steady_clock::now();
                std::cout << "Solution of qp_nb_2 took " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin_).count() << "ms, ret_2 is" << ret_2 << "\n";
                /////////////////////////

                if (ret == qpOASES::RET_MAX_NWSR_REACHED){
                    std::cout << "Solution of bound refined QP is taking too long, convexify the hessian further";
                    solution_found = false;
                    //Remove added dep bounds
                    for (int j = cond->num_true_cons; j < prob->nCon; j++){
                        c_vars->condensed_lb_con(j) = -1e20;
                        c_vars->condensed_ub_con(j) = 1e20;
                    }
                    break;
                }
                else if (ret != qpOASES::SUCCESSFUL_RETURN){
                    std::cout << "Error in bound refined QP, convexify the hessian further";
                    solution_found = false;
                    //Remove added dep bounds
                    for (int j = cond->num_true_cons; j < prob->nCon; j++){
                        c_vars->condensed_lb_con(j) = -1e20;
                        c_vars->condensed_ub_con(j) = 1e20;
                    }
                    break;
                }

                qp_nb_2->getPrimalSolution(c_vars->deltaXi_cond.array);
                qp_nb_2->getDualSolution(c_vars->lambdaQP_cond.array);
                cond->recover_var_mult(c_vars->deltaXi_cond, c_vars->lambdaQP_cond, deltaXi, lambdaQP);
            }
        }

        if (solution_found){
            //ret = solAna.checkCurvatureOnStronglyActiveConstraints(dynamic_cast<qpOASES::SQProblemSchur*>(qp_nb));
            //if (ret == qpOASES::SUCCESSFUL_RETURN) break;
            break;
        }

    } // End of QP solving loop

    // Compute constrJac*deltaXi, need this for second order correction step
    Atimesb( c_vars->jacNz, c_vars->jacIndRow, c_vars->jacIndCol, deltaXi, c_vars->AdeltaXi );

    // Print qpOASES error code, if any
    if(ret != qpOASES::SUCCESSFUL_RETURN)
        printf( "qpOASES error message: \"%s\"\n",
                qpOASES::getGlobalMessageHandler()->getErrorCodeMessage( ret ) );

    // Point Hessian again to the first Hessian
    c_vars->hess = c_vars->hess1;

    /* For full-memory Hessian: Restore fallback Hessian if convex combinations
     * were used during the loop */
    if( !param->hessLimMem && maxQP > 2)
    {
        double mu = 1.0 / ((double) l);
        double mu1 = 1.0 - mu;
        int nBlocks = (param->whichSecondDerv == 1) ? c_vars->nBlocks-1 : c_vars->nBlocks;
        for( int iBlock=0; iBlock<nBlocks; iBlock++ )
            for( int i=0; i<c_vars->hess[iBlock].M(); i++ )
                for( int j=i; j<c_vars->hess[iBlock].N(); j++ )
                {
                    c_vars->hess2[iBlock]( i,j ) *= mu;
                    c_vars->hess2[iBlock]( i,j ) += mu1 * c_vars->hess1[iBlock]( i,j );
                }
    }

    debug.close();

    /* Return code depending on qpOASES returnvalue
     * 0: Success
     * 1: Maximum number of iterations reached
     * 2: Unbounded
     * 3: Infeasible
     * 4: Other error */
    if( ret == qpOASES::SUCCESSFUL_RETURN )
        return 0;
    else if( ret == qpOASES::RET_MAX_NWSR_REACHED )
        return 1;
    else if( ret == qpOASES::RET_HESSIAN_NOT_SPD ||
             ret == qpOASES::RET_HESSIAN_INDEFINITE ||
             ret == qpOASES::RET_INIT_FAILED_UNBOUNDEDNESS ||
             ret == qpOASES::RET_QP_UNBOUNDED ||
             ret == qpOASES::RET_HOTSTART_STOPPED_UNBOUNDEDNESS ){
        return 2;}
    else if( ret == qpOASES::RET_INIT_FAILED_INFEASIBILITY ||
             ret == qpOASES::RET_QP_INFEASIBLE ||
             ret == qpOASES::RET_HOTSTART_STOPPED_INFEASIBILITY ){
        return 3;}
    else{
        return 4;}
}


int SCQPmethod::solve_SOC_QP( Matrix &deltaXi, Matrix &lambdaQP){

    SCQPiterate *c_vars = dynamic_cast<SCQPiterate*>(vars);

    std::ofstream debug;
    debug.open("/home/reinhold/cond_debug.txt", std::ios_base::app);
    debug << "Inside solveQP\n" << std::flush;

    //Condense QP before invoking QP-solver
    cond->SOC_condense(c_vars->gradObj, c_vars->delta_lb_con, c_vars->delta_ub_con,
            c_vars->condensed_h, c_vars->condensed_lb_con, c_vars->condensed_ub_con);

    //qpOASES defines +-infinity as +-1e20, so set absent bounds accordingly
    for (int i = 0; i < cond->condensed_num_cons; i++){
        if (std::isinf(c_vars->condensed_lb_con(i)))
            c_vars->condensed_lb_con(i) = -1e20;
        if (std::isinf(c_vars->condensed_ub_con(i)))
            c_vars->condensed_ub_con(i) = 1e20;
    }

    double *g, *lb, *ub, *lbA, *ubA;
    g = c_vars->condensed_h.array;
    lb = c_vars->condensed_lb_var.array;
    ub = c_vars->condensed_ub_var.array;
    lbA = c_vars->condensed_lb_con.array;
    ubA = c_vars->condensed_ub_con.array;

    // qpOASES options
    qpOASES::Options opts;
    opts.enableInertiaCorrection = qpOASES::BT_TRUE;
    opts.enableEqualities = qpOASES::BT_TRUE;
    opts.initialStatusBounds = qpOASES::ST_INACTIVE;

    switch(param->qpOASES_print_level){
        case 0:
            opts.printLevel = qpOASES::PL_NONE;
            break;
        case 1:
            opts.printLevel = qpOASES::PL_LOW;
            break;
        case 2:
            opts.printLevel = qpOASES::PL_MEDIUM;
            break;
        case 3:
            opts.printLevel = qpOASES::PL_HIGH;
            break;
    }
    opts.numRefinementSteps = 2;
    opts.epsLITests =  2.2204e-08;
    opts.enableInertiaCorrection = qpOASES::BT_TRUE;

    qp->setOptions( opts );

    // Other variables for qpOASES
    double cpuTime = param->maxTimeQP;
    int maxIt = param->maxItQP;
    qpOASES::SolutionAnalysis solAna;
    qpOASES::returnValue ret;


    if( param->debugLevel > 2 ) stats->dumpQPCpp( prob, vars, qp, param->sparseQP );

    debug << "Starting solution of SOC-QP\n" << std::flush;
    maxIt = 0.1*param->maxItQP;
    cpuTime = 0.1*param->maxTimeQP;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    ret = qp->hotstart( g, lb, ub, lbA, ubA, maxIt, &cpuTime );
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Solved SOC in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms\n";
    debug << "Finished solution of SOC-QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms\n" << std::flush;


    /*
     * Check assumption (G3*) if nonconvex QP was solved
     */
    if( ret == qpOASES::SUCCESSFUL_RETURN )
    {
        if( param->sparseQP == 2 )
            ret = solAna.checkCurvatureOnStronglyActiveConstraints( dynamic_cast<qpOASES::SQProblemSchur*>(qp) );
        else
            ret = solAna.checkCurvatureOnStronglyActiveConstraints( qp );
        std::cout << "Checked curvature, ret is " << ret << "\n";
    }

    if( ret == qpOASES::SUCCESSFUL_RETURN ){
        // QP was solved successfully and curvature is positive after removing bounds
        stats->qpIterations = maxIt + 1;

        //copy successfully solved QP to initialize QP with added bounds
        //(*qp_nb) = *dynamic_cast<qpOASES::SQProblemSchur*>(qp);
        (*qp_nb) = *qp;
        (*qp_nb_2) = *dynamic_cast<qpOASES::SQProblemSchur*>(qp);
        std::cout << "Copied the solved qp to qp_nb\n";
    }
    else{
        // QP solution is rejected, save statistics
        if( ret == qpOASES::RET_SETUP_AUXILIARYQP_FAILED )
            stats->qpIterations2++;
        else
            stats->qpIterations2 += maxIt + 1;
        stats->rejectedSR1++;
    }

    // Get solution from qpOASES
    qp->getPrimalSolution(c_vars->deltaXi_cond.array);
    qp->getDualSolution(c_vars->lambdaQP_cond.array);

    cond->recover_var_mult(c_vars->deltaXi_cond, c_vars->lambdaQP_cond, deltaXi, lambdaQP);


    //Check if some dependent variable bounds are violated and - if they are - add them to the QP and solve again
    if (ret == qpOASES::SUCCESSFUL_RETURN){
        bool found_direction;
        int ind;
        int ind_1;
        int ind_2;
        int ind_3;
        int num_dep_vars = cond->num_vars - cond->condensed_num_vars;
        int vio_count;

        std::cout << "Checking dependent variable bounds violation\n";
        for (int l = 0; l < param->max_bound_refines; l++){
            //Find dependent variable bounds that are violated at new point, add them to QP and solve again
            found_direction = true;
            ind_1 = 0;
            ind_2 = cond->num_true_cons;
            ind_3 = 0;
            vio_count = 0;

            for (int i = 0; i < cond->num_vblocks; i++){
                //Iterate over dependent variable blocks
                if (cond->vblocks[i].dependent){
                    for (int j = 0; j < cond->vblocks[i].size; j++){
                        ind = ind_1 + j;
                        if (c_vars->xi(ind) + deltaXi(ind) < prob->lb_var(ind) - param->dep_bound_tolerance || c_vars->xi(ind) + deltaXi(ind) > prob->ub_var(ind) + param->dep_bound_tolerance){
                            vio_count++;
                            found_direction = false;
                            c_vars->condensed_lb_con(ind_2 + j) = cond->lb_dep_var(ind_3 + j);
                            c_vars->condensed_ub_con(ind_2 + j) = cond->ub_dep_var(ind_3 + j);
                            std::cout << "Variable " << ind_2 + j << " violated bounds by " << std::min(0., c_vars->xi(ind) + deltaXi(ind) - prob->lb_var(ind)) + std::max(0., c_vars->xi(ind) + deltaXi(ind) - prob->ub_var(ind)) << "\n";
                        }
                    }
                    ind_2 += cond->vblocks[i].size;
                    ind_3 += cond->vblocks[i].size;
                }
                ind_1 += cond->vblocks[i].size;
            }
            if (found_direction){
                std::cout << "All dependent variable bounds are respected, exiting bound refinement\n";
                break;
            }
            std::cout << "Bounds violated by " << vio_count << " dependent variables, adding their bounds to the QP\n";
            debug << "Starting solution of bound refined QP\n" << std::flush;
            maxIt = param->maxItQP;
            cpuTime = param->maxTimeQP;

            qp_nb->setOptions(opts);
            qp_nb_2->setOptions(opts);


            qpOASES::SQProblem *qp_nb_init = new qpOASES::SQProblemSchur( prob->nVar, prob->nCon, qpOASES::HST_UNKNOWN, 50 );
            (*qp_nb_init) = *qp_nb;

            std::chrono::steady_clock::time_point begin_ = std::chrono::steady_clock::now();
            ret = qp_nb->hotstart(g, lb, ub, lbA, ubA, maxIt, &cpuTime);
            std::cout << "ret is " << ret << "\n";
            std::chrono::steady_clock::time_point end_ = std::chrono::steady_clock::now();
            std::cout << "Solved bound refined QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin_).count() << "ms\n";
            debug << "Finished solution of bound refined QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin_).count() << "ms\n" << std::flush;

            /////////////////////////
            maxIt = param->maxItQP;
            double cpuTime_ref = std::max(cpuTime, 0.5 * param->maxTimeQP);
            begin_ = std::chrono::steady_clock::now();
            qpOASES::returnValue ret_2 = qp_nb_2->hotstart(g, lb, ub, lbA, ubA, maxIt, &cpuTime_ref);
            end_ = std::chrono::steady_clock::now();
            std::cout << "Solution of qp_nb_2 took " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin_).count() << "ms, ret_2 is" << ret_2 << "\n";
            /////////////////////////


            //////////////////////////////////////////////
            qpOASES::Matrix *A_init = new qpOASES::SparseMatrix(cond->condensed_num_cons, cond->condensed_num_vars,
            c_vars->condensed_Jacobian.row, c_vars->condensed_Jacobian.colind, c_vars->condensed_Jacobian.nz);

            qpOASES::SymmetricMatrix* H_init = new qpOASES::SymSparseMat(cond->condensed_num_vars, cond->condensed_num_vars,
                                    c_vars->condensed_hess_row, c_vars->condensed_hess_colind, c_vars->condensed_hess_nz);
            dynamic_cast<qpOASES::SymSparseMat*>(H_init)->createDiagInfo();

            qpOASES::returnValue ret_init;
            int maxIt_2 = 10000000;
            double cpuTime_2 = 40;
            qp_nb_init->setOptions(opts);

            begin_ = std::chrono::steady_clock::now();
            ret_init = qp_nb_init->init(H_init, g, A_init, lb, ub, lbA, ubA, maxIt_2, &cpuTime_2);
            end_ = std::chrono::steady_clock::now();

            std::cout << "Solving initial bound refined SOC qp took " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin_).count() << "ms\n";
            //////////////////////////////////////////////

            qp_nb_2->getPrimalSolution(c_vars->deltaXi_cond.array);
            qp_nb_2->getDualSolution(c_vars->lambdaQP_cond.array);

            cond->recover_var_mult(c_vars->deltaXi_cond, c_vars->lambdaQP_cond, deltaXi, lambdaQP);
        }
    }

    // Compute constrJac*deltaXi, need this for second order correction step
    Atimesb( c_vars->jacNz, c_vars->jacIndRow, c_vars->jacIndCol, deltaXi, c_vars->AdeltaXi );

    // Print qpOASES error code, if any
    if( ret != qpOASES::SUCCESSFUL_RETURN)
        printf( "qpOASES error message: \"%s\"\n",
                qpOASES::getGlobalMessageHandler()->getErrorCodeMessage( ret ) );

    // Point Hessian again to the first Hessian
    c_vars->hess = c_vars->hess1;

    debug.close();

    /* Return code depending on qpOASES returnvalue
     * 0: Success
     * 1: Maximum number of iterations reached
     * 2: Unbounded
     * 3: Infeasible
     * 4: Other error */
    if( ret == qpOASES::SUCCESSFUL_RETURN )
        return 0;
    else if( ret == qpOASES::RET_MAX_NWSR_REACHED )
        return 1;
    else if( ret == qpOASES::RET_HESSIAN_NOT_SPD ||
             ret == qpOASES::RET_HESSIAN_INDEFINITE ||
             ret == qpOASES::RET_INIT_FAILED_UNBOUNDEDNESS ||
             ret == qpOASES::RET_QP_UNBOUNDED ||
             ret == qpOASES::RET_HOTSTART_STOPPED_UNBOUNDEDNESS ){
        return 2;}
    else if( ret == qpOASES::RET_INIT_FAILED_INFEASIBILITY ||
             ret == qpOASES::RET_QP_INFEASIBLE ||
             ret == qpOASES::RET_HOTSTART_STOPPED_INFEASIBILITY ){
        return 3;}
    else{
        return 4;}
}




int SCQP_rest_method::solveQP(Matrix &deltaXi, Matrix &lambdaQP, bool matricesChanged){

    SCQPiterate *c_vars = dynamic_cast<SCQPiterate*>(vars);

    std::ofstream debug;
    debug.open("/home/reinhold/cond_debug.txt", std::ios_base::app);
    debug << "Inside solveQP\n" << std::flush;

    Matrix jacT;
    int maxQP, l;
    if( param->globalization == 1 &&
        param->hessUpdate == 1 &&
        stats->itCount > 1 )
    {
        maxQP = param->maxConvQP + 1;
    }
    else
        maxQP = 1;

    //Condense the QP before invoking qpOASES
    cond->full_condense(c_vars->gradObj, c_vars->Jacobian, c_vars->hess,
        c_vars->delta_lb_var, c_vars->delta_ub_var, c_vars->delta_lb_con, c_vars->delta_ub_con,
            c_vars->condensed_h, c_vars->condensed_Jacobian, c_vars->condensed_hess, c_vars->condensed_lb_var, c_vars->condensed_ub_var, c_vars->condensed_lb_con, c_vars->condensed_ub_con);

    delete A_qp;
    A_qp = new qpOASES::SparseMatrix(cond->condensed_num_cons, cond->condensed_num_vars,
                c_vars->condensed_Jacobian.row, c_vars->condensed_Jacobian.colind, c_vars->condensed_Jacobian.nz);

    //qpOASES defines +-infinity as +-1e20, so set absent bounds accordingly
    for (int i = 0; i < cond->condensed_num_vars; i++){
        if (std::isinf(c_vars->condensed_lb_var(i)))
            c_vars->condensed_lb_var(i) = -1e20;
        if (std::isinf(c_vars->condensed_ub_var(i)))
            c_vars->condensed_ub_var(i) = 1e20;
    }
    for (int i = 0; i < cond->condensed_num_cons; i++){
        if (std::isinf(c_vars->condensed_lb_con(i)))
            c_vars->condensed_lb_con(i) = -1e20;
        if (std::isinf(c_vars->condensed_ub_con(i)))
            c_vars->condensed_ub_con(i) = 1e20;
    }

    double *g, *lb, *ub, *lbA, *ubA;
    g = c_vars->condensed_h.array;
    lb = c_vars->condensed_lb_var.array;
    ub = c_vars->condensed_ub_var.array;
    lbA = c_vars->condensed_lb_con.array;
    ubA = c_vars->condensed_ub_con.array;


    // qpOASES options
    qpOASES::Options opts;
    if(maxQP > 1)
        opts.enableInertiaCorrection = qpOASES::BT_FALSE;
    else
        opts.enableInertiaCorrection = qpOASES::BT_TRUE;
    opts.enableEqualities = qpOASES::BT_TRUE;
    opts.initialStatusBounds = qpOASES::ST_INACTIVE;

    switch(param->qpOASES_print_level){
        case 0:
            opts.printLevel = qpOASES::PL_NONE;
            break;
        case 1:
            opts.printLevel = qpOASES::PL_LOW;
            break;
        case 2:
            opts.printLevel = qpOASES::PL_MEDIUM;
            break;
        case 3:
            opts.printLevel = qpOASES::PL_HIGH;
            break;
    }

    //opts.printLevel = qpOASES::PL_MEDIUM; //PL_LOW, PL_HIGH, PL_MEDIUM, PL_NONE
    opts.numRefinementSteps = 2;
    opts.epsLITests =  2.2204e-08;

    qp->setOptions( opts );

    if( maxQP > 1 )
    {
        // Store last successful QP in temporary storage
        (*qpSave) = *qp;
        /** \todo Storing the active set would be enough but then the QP object
         *        must be properly reset after unsuccessful (SR1-)attempt.
         *        Moreover, passing a guessed active set doesn't yield
         *        exactly the same result as hotstarting a QP. This has
         *        something to do with how qpOASES handles user-given
         *        active sets (->check qpOASES source code). */
    }

    // Other variables for qpOASES
    double cpuTime = param->maxTimeQP;
    int maxIt = param->maxItQP;
    qpOASES::SolutionAnalysis solAna;
    qpOASES::returnValue ret;


    for( l=0; l<maxQP; l++ ){
        if( l > 0 ){
            // If the solution of the first QP was rejected, consider second Hessian
            stats->qpResolve++;
            *qp = *qpSave;
            computeNextHessian( l, maxQP );

            cond->new_hessian_condense(c_vars->hess, c_vars->condensed_h, c_vars->condensed_hess);
            g = c_vars->condensed_h.array;
        }

        if( l == maxQP-1 )
        {// Enable inertia correction for supposedly convex QPs, just in case
            opts.enableInertiaCorrection = qpOASES::BT_TRUE;
            qp->setOptions( opts );
        }

        // Convert block-Hessian to sparse format
        c_vars->convertHessian(cond->condensed_num_vars, cond->condensed_num_hessblocks, param->eps, c_vars->condensed_hess,
                        c_vars->condensed_hess_nz, c_vars->condensed_hess_row, c_vars->condensed_hess_colind, c_vars->condensed_hess_loind);
        delete H_qp;
        H_qp = new qpOASES::SymSparseMat(cond->condensed_num_vars, cond->condensed_num_vars,
                                    c_vars->condensed_hess_row, c_vars->condensed_hess_colind, c_vars->condensed_hess_nz);
        dynamic_cast<qpOASES::SymSparseMat*>(H_qp)->createDiagInfo();

        /*
         * Call qpOASES
         */
        if( param->debugLevel > 2 ) stats->dumpQPCpp( prob, vars, qp, param->sparseQP );

        maxIt = param->maxItQP;
        cpuTime = param->maxTimeQP;
        if( (qp->getStatus() == qpOASES::QPS_HOMOTOPYQPSOLVED ||
            qp->getStatus() == qpOASES::QPS_SOLVED ) && c_vars->use_homotopy)
        {
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            debug << "Starting hotstart solution of QP\n" << std::flush;

            ret = qp->hotstart(H_qp, g, A_qp, lb, ub, lbA, ubA, maxIt, &cpuTime);

            std::cout << "cpuTime is " << cpuTime << "\n";

            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << "Solved hotstarted QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms\n";
            std::cout << "ret is " << ret << "\n";
            debug << "Finished hotstart solution of QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() <<" ms\n" << std::flush;
        }
        else
        {
            debug << "Starting solution of initial QP\n" << std::flush;
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            ret = qp->init( H_qp, g, A_qp, lb, ub, lbA, ubA, maxIt, &cpuTime );
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << "Solved initial QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms\n";
            std::cout << "ret is " << ret << "\n";
            debug << "Solved initial QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() <<" ms\n" << std::flush;
            c_vars->use_homotopy = true;
        }

        /*
         * Check assumption (G3*) if nonconvex QP was solved
         */
        if(l < maxQP-1){

            if( ret == qpOASES::SUCCESSFUL_RETURN )
            {
                if( param->sparseQP == 2 )
                    ret = solAna.checkCurvatureOnStronglyActiveConstraints( dynamic_cast<qpOASES::SQProblemSchur*>(qp) );
                else
                    ret = solAna.checkCurvatureOnStronglyActiveConstraints( qp );
            }

            if( ret == qpOASES::SUCCESSFUL_RETURN )
            {// QP was solved successfully and curvature is positive after removing bounds
                stats->qpIterations = maxIt + 1;
                break; // Success!
            }
            else
            {// QP solution is rejected, save statistics
                if( ret == qpOASES::RET_SETUP_AUXILIARYQP_FAILED )
                    stats->qpIterations2++;
                else
                    stats->qpIterations2 += maxIt + 1;
                stats->rejectedSR1++;
            }
        }
        else{ // Convex QP was solved, no need to check assumption (G3*)
            stats->qpIterations += maxIt + 1;
        }

    } // End of QP solving loop


    // Get solution from qpOASES
    qp->getPrimalSolution(c_vars->deltaXi_cond.array);
    qp->getDualSolution(c_vars->lambdaQP_cond.array);

    cond->recover_var_mult(c_vars->deltaXi_cond, c_vars->lambdaQP_cond, deltaXi, lambdaQP);

    /*
     * Post-processing
     */

    // Compute constrJac*deltaXi, need this for second order correction step
    Atimesb( c_vars->jacNz, c_vars->jacIndRow, c_vars->jacIndCol, deltaXi, c_vars->AdeltaXi );

    // Print qpOASES error code, if any
    if( ret != qpOASES::SUCCESSFUL_RETURN)
        printf( "qpOASES error message: \"%s\"\n",
                qpOASES::getGlobalMessageHandler()->getErrorCodeMessage( ret ) );

    // Point Hessian again to the first Hessian
    c_vars->hess = c_vars->hess1;

    // For full-memory Hessian: Restore fallback Hessian if convex combinations were used during the loop
    if( !param->hessLimMem && maxQP > 2){

        double mu = 1.0 / ((double) l);
        double mu1 = 1.0 - mu;
        int nBlocks = (param->whichSecondDerv == 1) ? c_vars->nBlocks-1 : c_vars->nBlocks;
        for( int iBlock=0; iBlock<nBlocks; iBlock++ )
            for( int i=0; i<c_vars->hess[iBlock].M(); i++ )
                for( int j=i; j<c_vars->hess[iBlock].N(); j++ )
                {
                    c_vars->hess2[iBlock]( i,j ) *= mu;
                    c_vars->hess2[iBlock]( i,j ) += mu1 * c_vars->hess1[iBlock]( i,j );
                }
    }

    debug.close();
    /* Return code depending on qpOASES returnvalue
     * 0: Success
     * 1: Maximum number of iterations reached
     * 2: Unbounded
     * 3: Infeasible
     * 4: Other error */
    if( ret == qpOASES::SUCCESSFUL_RETURN )
        return 0;
    else if( ret == qpOASES::RET_MAX_NWSR_REACHED )
        return 1;
    else if( ret == qpOASES::RET_HESSIAN_NOT_SPD ||
             ret == qpOASES::RET_HESSIAN_INDEFINITE ||
             ret == qpOASES::RET_INIT_FAILED_UNBOUNDEDNESS ||
             ret == qpOASES::RET_QP_UNBOUNDED ||
             ret == qpOASES::RET_HOTSTART_STOPPED_UNBOUNDEDNESS ){
        return 2;}
    else if( ret == qpOASES::RET_INIT_FAILED_INFEASIBILITY ||
             ret == qpOASES::RET_QP_INFEASIBLE ||
             ret == qpOASES::RET_HOTSTART_STOPPED_INFEASIBILITY ){
        return 3;}
    else{
        return 4;}
}



int SCQP_correction_method::solveQP(Matrix &deltaXi, Matrix &lambdaQP, bool matricesChanged){

    SCQPiterate *c_vars = dynamic_cast<SCQPiterate*>(vars);

    std::ofstream debug;
    debug.open("/home/reinhold/cond_debug.txt", std::ios_base::app);
    debug << "Inside solveQP\n" << std::flush;

    //reset correction vectors
    for (int tnum = 0; tnum < cond->num_targets; tnum++){
        corrections[tnum].Initialize(0.);
    }

    Matrix jacT;
    int maxQP, l;
    if( param->globalization == 1 &&
        param->hessUpdate == 1 &&
        stats->itCount > 1 )
    {
        maxQP = param->maxConvQP + 1;
    }
    else
        maxQP = 1;

    //Condense QP before invoking qpOASES if possible
    cond->full_condense(c_vars->gradObj, c_vars->Jacobian, c_vars->hess,
        c_vars->delta_lb_var, c_vars->delta_ub_var, c_vars->delta_lb_con, c_vars->delta_ub_con,
            c_vars->condensed_h, c_vars->condensed_Jacobian, c_vars->condensed_hess, c_vars->condensed_lb_var, c_vars->condensed_ub_var, c_vars->condensed_lb_con, c_vars->condensed_ub_con);

    delete A_qp;
    A_qp = new qpOASES::SparseMatrix(cond->condensed_num_cons, cond->condensed_num_vars,
                c_vars->condensed_Jacobian.row, c_vars->condensed_Jacobian.colind, c_vars->condensed_Jacobian.nz);

    //qpOASES defines +-infinity as +-1e20, so set absent bounds accordingly
    for (int i = 0; i < cond->condensed_num_vars; i++){
        if (std::isinf(c_vars->condensed_lb_var(i)))
            c_vars->condensed_lb_var(i) = -1e20;
        if (std::isinf(c_vars->condensed_ub_var(i)))
            c_vars->condensed_ub_var(i) = 1e20;
    }
    for (int i = 0; i < cond->condensed_num_cons; i++){
        if (std::isinf(c_vars->condensed_lb_con(i)))
            c_vars->condensed_lb_con(i) = -1e20;
        if (std::isinf(c_vars->condensed_ub_con(i)))
            c_vars->condensed_ub_con(i) = 1e20;
    }

    double *g, *lb, *ub, *lbA, *ubA;
    g = c_vars->condensed_h.array;
    lb = c_vars->condensed_lb_var.array;
    ub = c_vars->condensed_ub_var.array;
    lbA = c_vars->condensed_lb_con.array;
    ubA = c_vars->condensed_ub_con.array;

    // qpOASES options
    qpOASES::Options opts;
    if(maxQP > 1)
        opts.enableInertiaCorrection = qpOASES::BT_FALSE;
    else
        opts.enableInertiaCorrection = qpOASES::BT_TRUE;
    opts.enableEqualities = qpOASES::BT_TRUE;
    opts.initialStatusBounds = qpOASES::ST_INACTIVE;

    switch(param->qpOASES_print_level){
        case 0:
            opts.printLevel = qpOASES::PL_NONE;
            break;
        case 1:
            opts.printLevel = qpOASES::PL_LOW;
            break;
        case 2:
            opts.printLevel = qpOASES::PL_MEDIUM;
            break;
        case 3:
            opts.printLevel = qpOASES::PL_HIGH;
            break;
    }

    //opts.printLevel = qpOASES::PL_MEDIUM; //PL_LOW, PL_HIGH, PL_MEDIUM, PL_NONE
    opts.numRefinementSteps = 2;
    opts.epsLITests =  2.2204e-08;

    qp->setOptions( opts );

    if( maxQP > 1 )
    {
        // Store last successful QP in temporary storage
        (*qpSave) = *qp;
    }

    // Other variables for qpOASES
    double cpuTime = param->maxTimeQP;
    int maxIt = param->maxItQP;
    qpOASES::SolutionAnalysis solAna;
    qpOASES::returnValue ret;

    /*
     * QP solving loop for convex combinations (sequential)
     */
    for( l=0; l<maxQP; l++ )
    {
        //Compute a new Hessian
        if( l > 0 ){
            // If the solution of the first QP was rejected, consider second Hessian
            stats->qpResolve++;
            *qp = *qpSave;

            computeNextHessian( l, maxQP );

            cond->new_hessian_condense(c_vars->hess, c_vars->condensed_h, c_vars->condensed_hess);
            g = c_vars->condensed_h.array;
        }

        if( l == maxQP-1 ){
            //Enable inertia correction for supposedly convex QPs, just in case
            opts.enableInertiaCorrection = qpOASES::BT_TRUE;
            qp->setOptions( opts );
        }

        // Convert block-Hessian to sparse format
        c_vars->convertHessian(cond->condensed_num_vars, cond->condensed_num_hessblocks, param->eps, c_vars->condensed_hess,
                        c_vars->condensed_hess_nz, c_vars->condensed_hess_row, c_vars->condensed_hess_colind, c_vars->condensed_hess_loind);
        delete H_qp;
        H_qp = new qpOASES::SymSparseMat(cond->condensed_num_vars, cond->condensed_num_vars,
                                    c_vars->condensed_hess_row, c_vars->condensed_hess_colind, c_vars->condensed_hess_nz);
        dynamic_cast<qpOASES::SymSparseMat*>(H_qp)->createDiagInfo();

        /*
         * Call qpOASES
         */
        if( param->debugLevel > 2 ) stats->dumpQPCpp( prob, vars, qp, param->sparseQP );

        maxIt = param->maxItQP;
        cpuTime = param->maxTimeQP;
        if( (qp->getStatus() == qpOASES::QPS_HOMOTOPYQPSOLVED || qp->getStatus() == qpOASES::QPS_SOLVED ) && c_vars->use_homotopy){

            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            debug << "Starting hotstart solution of QP\n" << std::flush;

            ret = qp->hotstart(H_qp, g, A_qp, lb, ub, lbA, ubA, maxIt, &cpuTime);

            std::cout << "cpuTime is " << cpuTime << "\n";

            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << "Solved hotstarted QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms\n";
            std::cout << "ret is " << ret << "\n";
            debug << "Finished hotstart solution of QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() <<" ms\n" << std::flush;
        }
        else{
            debug << "Starting solution of initial QP\n" << std::flush;
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            ret = qp->init( H_qp, g, A_qp, lb, ub, lbA, ubA, maxIt, &cpuTime );
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << "Solved initial QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms\n";
            std::cout << "ret is " << ret << "\n";
            debug << "Solved initial QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() <<" ms\n" << std::flush;
            c_vars->use_homotopy = true;
        }


        /*
         * Check assumption (G3*) if nonconvex QP was solved
         */
        if( l < maxQP-1)
        {
            if( ret == qpOASES::SUCCESSFUL_RETURN )
            {
                if( param->sparseQP == 2 )
                    ret = solAna.checkCurvatureOnStronglyActiveConstraints( dynamic_cast<qpOASES::SQProblemSchur*>(qp) );
                else
                    ret = solAna.checkCurvatureOnStronglyActiveConstraints( qp );
            }

            if( ret == qpOASES::SUCCESSFUL_RETURN )
            {// QP was solved successfully and curvature is positive after removing bounds
                stats->qpIterations = maxIt + 1;

                //copy successfully solved QP to initialize SOC condition qp
                (*qp_nb) = *dynamic_cast<qpOASES::SQProblemSchur*>(qp);

                break; // Success!
            }
            else{
                // QP solution is rejected, save statistics
                if( ret == qpOASES::RET_SETUP_AUXILIARYQP_FAILED )
                    stats->qpIterations2++;
                else
                    stats->qpIterations2 += maxIt + 1;
                stats->rejectedSR1++;
            }
        }
        else{
            // Convex QP was solved, no need to check assumption (G3*)
            stats->qpIterations += maxIt + 1;
            (*qp_nb) = *dynamic_cast<qpOASES::SQProblemSchur*>(qp);
        }


        bool solution_found = false;
        if (ret == qpOASES::SUCCESSFUL_RETURN){
            solution_found = true;

            // Get solution from qpOASES
            qp->getPrimalSolution(c_vars->deltaXi_cond.array);
            qp->getDualSolution(c_vars->lambdaQP_cond.array);
            cond->recover_var_mult(c_vars->deltaXi_cond, c_vars->lambdaQP_cond, deltaXi, lambdaQP);


            for (int tnum = 0; tnum < cond->num_targets; tnum++){
                corrections[tnum].Initialize(0.);
            }

            int ind_1, ind_2, ind, vio_count;
            bool found_direction;
            Matrix xi_s(c_vars->xi), lambda_s(c_vars->lambda), deltaXi_s, constr_s, AdeltaXi_s;
            constr_s.Dimension(prob->nCon);
            deltaXi_s.Dimension(prob->nVar);
            AdeltaXi_s.Dimension(prob->nCon);

            double cpuTime_ref;

            for (int l = 0; l < param->max_bound_refines; l++){
                found_direction = true;
                ind_1 = 0;
                vio_count = 0;
                deltaXi_s = deltaXi;

                for (int i = 0; i < cond->num_vblocks; i++){
                    if (cond->vblocks[i].dependent){
                        for (int j = 0; j < cond->vblocks[i].size; j++){
                            ind = ind_1 + j;
                            if (c_vars->xi(ind) + deltaXi(ind) < prob->lb_var(ind) - param->dep_bound_tolerance || c_vars->xi(ind) + deltaXi(ind) > prob->ub_var(ind) + param->dep_bound_tolerance){
                                vio_count++;
                                found_direction = false;
                            }
                        }
                    }
                    ind_1 += cond->vblocks[i].size;
                }
                if (found_direction){
                    std::cout << "All dependent variable bounds are respected, exiting step correction\n";
                    break;
                }
                std::cout << "Bounds violated by " << vio_count << " dependent variables, calculating correction vectors\n";

                xi_s = c_vars->xi + deltaXi;
                lambda_s = lambdaQP;

                for (int tnum = 0; tnum < cond->num_targets; tnum++){

                    //Add difference between dependent state values from QP solution and integration for target tnum
                    ind_1 = 0;
                    ind_2 = cond->vranges[cond->targets[tnum].first_free];

                    for (int i = cond->targets[tnum].first_free; i < cond->targets[tnum].vblock_end; i++){
                        if (cond->vblocks[i].dependent){
                            for (int j = 0; j < cond->vblocks[i].size; j++){
                                if (corrections[tnum](ind_1 + j) > 0 && xi_s(ind_2 + j) > prob->lb_var(ind_2 + j)){
                                    corrections[tnum](ind_1 + j) -= xi_s(ind_2 + j) - prob->lb_var(ind_2 + j);
                                    if (corrections[tnum](ind_1 + j) < 0) corrections[tnum](ind_1 + j) = 0;
                                }
                                else if (corrections[tnum](ind_1 + j) < 0 && xi_s(ind_2 + j) < prob->ub_var(ind_2 + j)){
                                    corrections[tnum](ind_1 + j) -= xi_s(ind_2 + j) - prob->ub_var(ind_2 + j);
                                    if (corrections[tnum](ind_1 + j) > 0) corrections[tnum](ind_1 + j) = 0;
                                }

                                if (xi_s(ind_2 + j) < prob->lb_var(ind_2 + j) - param->dep_bound_tolerance){
                                    corrections[tnum](ind_1 + j) += prob->lb_var(ind_2 + j) - xi_s(ind_2 + j);
                                }
                                else if (xi_s(ind_2 + j) > prob->ub_var(ind_2 + j) + param->dep_bound_tolerance){
                                    corrections[tnum](ind_1 + j) += prob->ub_var(ind_2 + j) - xi_s(ind_2 + j);
                                }
                            }
                            ind_1 += cond->vblocks[i].size;
                        }
                        ind_2 += cond->vblocks[i].size;
                    }
                }

                cond->correction_condense(c_vars->gradObj, c_vars->delta_lb_con, c_vars->delta_ub_con, corrections, c_vars->condensed_h, c_vars->condensed_lb_con, c_vars->condensed_ub_con);

                g = c_vars->condensed_h.array;
                lbA = c_vars->condensed_lb_con.array;
                ubA = c_vars->condensed_ub_con.array;

                for (int i = 0; i < cond->condensed_num_cons; i++){
                    if (std::isinf(c_vars->condensed_lb_con(i)))
                        c_vars->condensed_lb_con(i) = -1e20;
                    if (std::isinf(c_vars->condensed_ub_con(i)))
                        c_vars->condensed_ub_con(i) = 1e20;
                }

                maxIt = param->maxItQP;
                cpuTime_ref = std::max(cpuTime, 0.25 * param->maxTimeQP);
                qp_nb->setOptions(opts);

                std::cout << "Starting solution of correction qp\n";
                debug << "Starting solution of correction qp\n" << std::flush;
                std::chrono::steady_clock::time_point T0 = std::chrono::steady_clock::now();

                ret = qp_nb->hotstart(g, lb, ub, lbA, ubA, maxIt, &cpuTime_ref);

                std::chrono::steady_clock::time_point T1 = std::chrono::steady_clock::now();
                std::cout << "Finished solution of correction qp in " << std::chrono::duration_cast<std::chrono::milliseconds>(T1 - T0).count() << "ms\n";
                debug << "Finished solution of correction qp in " << std::chrono::duration_cast<std::chrono::milliseconds>(T1 - T0).count() << "ms\n" << std::flush;

                if (ret == qpOASES::RET_MAX_NWSR_REACHED){
                    std::cout << "Solution of correction QP is taking too long, convexifying hessian\n";
                    solution_found = false;
                    //Added corrections terms need not be removed, because condensed_h is recalculated by new_hessian_condense and dependent variable bounds are not calculated
                    //Remove added correction terms
                    /*cond->SOC_condense(c_vars->gradObj, c_vars->delta_lb_con, c_vars->delta_ub_con,
                            c_vars->condensed_h, c_vars->condensed_lb_con, c_vars->condensed_ub_con);*/
                    break;
                    /*maxIt = param->maxItQP;
                    cpuTime_ref = std::max(cpuTime, 2.5 * param->maxTimeQP);
                    ret = qp_nb->init(H_qp, g, A_qp, lb, ub, lbA, ubA, maxIt, &cpuTime_ref);*/
                }
                else if (ret != qpOASES::SUCCESSFUL_RETURN){
                    std::cout << "Error in correction QP, convexifying hessian\n";
                    solution_found = false;
                    //Remove added correction terms
                    /*cond->SOC_condense(c_vars->gradObj, c_vars->delta_lb_con, c_vars->delta_ub_con,
                            c_vars->condensed_h, c_vars->condensed_lb_con, c_vars->condensed_ub_con);*/
                    break;
                }
                qp_nb->getPrimalSolution(c_vars->deltaXi_cond.array);
                qp_nb->getDualSolution(c_vars->lambdaQP_cond.array);
                cond->recover_correction_var_mult(c_vars->deltaXi_cond, c_vars->lambdaQP_cond, corrections, deltaXi, lambdaQP);
            }
        }
        if (solution_found) break;


    } // End of QP solving loop

/*
    if (ret == qpOASES::SUCCESSFUL_RETURN){

        // Get solution from qpOASES
        qp->getPrimalSolution(c_vars->deltaXi_cond.array);
        qp->getDualSolution(c_vars->lambdaQP_cond.array);
        cond->recover_var_mult(c_vars->deltaXi_cond, c_vars->lambdaQP_cond, deltaXi, lambdaQP);


        for (int tnum = 0; tnum < cond->num_targets; tnum++){
            corrections[tnum].Initialize(0.);
        }

        int ind_1, ind_2, ind, vio_count;
        bool found_direction;
        Matrix xi_s(c_vars->xi), lambda_s(c_vars->lambda), deltaXi_s, constr_s, AdeltaXi_s;
        constr_s.Dimension(prob->nCon);
        deltaXi_s.Dimension(prob->nVar);
        AdeltaXi_s.Dimension(prob->nCon);

        for (int l = 0; l < param->max_bound_refines; l++){
            found_direction = true;
            ind_1 = 0;
            vio_count = 0;
            deltaXi_s = deltaXi;

            for (int i = 0; i < cond->num_vblocks; i++){
                if (cond->vblocks[i].dependent){
                    for (int j = 0; j < cond->vblocks[i].size; j++){
                        ind = ind_1 + j;
                        if (c_vars->xi(ind) + deltaXi(ind) < prob->lb_var(ind) - param->dep_bound_tolerance || c_vars->xi(ind) + deltaXi(ind) > prob->ub_var(ind) + param->dep_bound_tolerance){
                            vio_count++;
                            found_direction = false;
                        }
                    }
                }
                ind_1 += cond->vblocks[i].size;
            }
            if (found_direction){
                std::cout << "All dependent variable bounds are respected, exiting step correction\n";
                break;
            }
            std::cout << "Bounds violated by " << vio_count << " dependent variables, calculating correction vectors\n";

            xi_s = c_vars->xi + deltaXi;
            lambda_s = lambdaQP;

            for (int tnum = 0; tnum < cond->num_targets; tnum++){

                //Add difference between dependent state values from QP solution and integration for target tnum
                ind_1 = 0;
                ind_2 = cond->vranges[cond->targets[tnum].first_free];

                for (int i = cond->targets[tnum].first_free; i < cond->targets[tnum].vblock_end; i++){
                    if (cond->vblocks[i].dependent){
                        for (int j = 0; j < cond->vblocks[i].size; j++){
                            if (corrections[tnum](ind_1 + j) > 0 && xi_s(ind_2 + j) > prob->lb_var(ind_2 + j)){
                                corrections[tnum](ind_1 + j) -= xi_s(ind_2 + j) - prob->lb_var(ind_2 + j);
                                if (corrections[tnum](ind_1 + j) < 0) corrections[tnum](ind_1 + j) = 0;
                            }
                            else if (corrections[tnum](ind_1 + j) < 0 && xi_s(ind_2 + j) < prob->ub_var(ind_2 + j)){
                                corrections[tnum](ind_1 + j) -= xi_s(ind_2 + j) - prob->ub_var(ind_2 + j);
                                if (corrections[tnum](ind_1 + j) > 0) corrections[tnum](ind_1 + j) = 0;
                            }

                            if (xi_s(ind_2 + j) < prob->lb_var(ind_2 + j) - param->dep_bound_tolerance){
                                corrections[tnum](ind_1 + j) += prob->lb_var(ind_2 + j) - xi_s(ind_2 + j);
                            }
                            else if (xi_s(ind_2 + j) > prob->ub_var(ind_2 + j) + param->dep_bound_tolerance){
                                corrections[tnum](ind_1 + j) += prob->ub_var(ind_2 + j) - xi_s(ind_2 + j);
                            }
                        }
                        ind_1 += cond->vblocks[i].size;
                    }
                    ind_2 += cond->vblocks[i].size;
                }
            }


            cond->correction_condense(c_vars->gradObj, c_vars->delta_lb_con, c_vars->delta_ub_con, corrections, c_vars->condensed_h, c_vars->condensed_lb_con, c_vars->condensed_ub_con);

            g = c_vars->condensed_h.array;
            lbA = c_vars->condensed_lb_con.array;
            ubA = c_vars->condensed_ub_con.array;

            for (int i = 0; i < cond->condensed_num_cons; i++){
                if (std::isinf(c_vars->condensed_lb_con(i)))
                    c_vars->condensed_lb_con(i) = -1e20;
                if (std::isinf(c_vars->condensed_ub_con(i)))
                    c_vars->condensed_ub_con(i) = 1e20;
            }

            maxIt = param->maxItQP;
            //cpuTime = param->maxTimeQP;
            //If correction QP takes longer to solve than original QP, re-initialize the correction QP

            qp_nb->setOptions(opts);

            std::cout << "Starting solution of correction qp\n";
            debug << "Starting solution of correction qp\n" << std::flush;
            std::chrono::steady_clock::time_point T0 = std::chrono::steady_clock::now();
            //if (l > 0)
                ret = qp_nb->hotstart(g, lb, ub, lbA, ubA, maxIt, &cpuTime);
            //else
            //    ret = qp_nb->init(H_qp, g, A_qp, lb, ub, lbA, ubA, maxIt, &cpuTime);
            if (ret == qpOASES::RET_MAX_NWSR_REACHED){
                std::cout << "Solution of correction QP is taking too long, initializing new correction QP\n";
                cpuTime = 2.5*param->maxTimeQP;
                ret = qp_nb->init(H_qp, g, A_qp, lb, ub, lbA, ubA, maxIt, &cpuTime);
            }

            std::chrono::steady_clock::time_point T1 = std::chrono::steady_clock::now();
            std::cout << "Finished solution of correction qp in " << std::chrono::duration_cast<std::chrono::milliseconds>(T1 - T0).count() << "ms\n";
            debug << "Finished solution of correction qp in " << std::chrono::duration_cast<std::chrono::milliseconds>(T1 - T0).count() << "ms\n" << std::flush;

            if (ret == qpOASES::SUCCESSFUL_RETURN){
                qp_nb->getPrimalSolution(c_vars->deltaXi_cond.array);
                qp_nb->getDualSolution(c_vars->lambdaQP_cond.array);
                cond->recover_correction_var_mult(c_vars->deltaXi_cond, c_vars->lambdaQP_cond, corrections, deltaXi, lambdaQP);

            }
            else{break;}

        }
    }
    */

    // Compute constrJac*deltaXi, need this for second order correction step
    Atimesb( c_vars->jacNz, c_vars->jacIndRow, c_vars->jacIndCol, deltaXi, c_vars->AdeltaXi );

    // Print qpOASES error code, if any
    if( ret != qpOASES::SUCCESSFUL_RETURN)
        printf( "qpOASES error message: \"%s\"\n",
                qpOASES::getGlobalMessageHandler()->getErrorCodeMessage( ret ) );

    // Point Hessian again to the first Hessian
    c_vars->hess = c_vars->hess1;

    /* For full-memory Hessian: Restore fallback Hessian if convex combinations
     * were used during the loop */
    if( !param->hessLimMem && maxQP > 2){

        double mu = 1.0 / ((double) l);
        double mu1 = 1.0 - mu;
        int nBlocks = (param->whichSecondDerv == 1) ? c_vars->nBlocks-1 : c_vars->nBlocks;
        for( int iBlock=0; iBlock<nBlocks; iBlock++ )
            for( int i=0; i<c_vars->hess[iBlock].M(); i++ )
                for( int j=i; j<c_vars->hess[iBlock].N(); j++ )
                {
                    c_vars->hess2[iBlock]( i,j ) *= mu;
                    c_vars->hess2[iBlock]( i,j ) += mu1 * c_vars->hess1[iBlock]( i,j );
                }
    }

    debug << "Returning\n" << std::flush;
    debug.close();

    /* Return code depending on qpOASES returnvalue
     * 0: Success
     * 1: Maximum number of iterations reached
     * 2: Unbounded
     * 3: Infeasible
     * 4: Other error */
    if( ret == qpOASES::SUCCESSFUL_RETURN )
        return 0;
    else if( ret == qpOASES::RET_MAX_NWSR_REACHED )
        return 1;
    else if( ret == qpOASES::RET_HESSIAN_NOT_SPD ||
             ret == qpOASES::RET_HESSIAN_INDEFINITE ||
             ret == qpOASES::RET_INIT_FAILED_UNBOUNDEDNESS ||
             ret == qpOASES::RET_QP_UNBOUNDED ||
             ret == qpOASES::RET_HOTSTART_STOPPED_UNBOUNDEDNESS ){
        return 2;}
    else if( ret == qpOASES::RET_INIT_FAILED_INFEASIBILITY ||
             ret == qpOASES::RET_QP_INFEASIBLE ||
             ret == qpOASES::RET_HOTSTART_STOPPED_INFEASIBILITY ){
        return 3;}
    else{
        return 4;}
}





int SCQP_correction_method::solve_SOC_QP(Matrix &deltaXi, Matrix &lambdaQP){

    SCQPiterate *c_vars = dynamic_cast<SCQPiterate*>(vars);

    std::ofstream debug;
    debug.open("/home/reinhold/cond_debug.txt", std::ios_base::app);
    debug << "Inside solve_SOC_QP\n" << std::flush;

    //reset correction vectors (SOC already introduces correction!)
    for (int tnum = 0; tnum < cond->num_targets; tnum++){
        corrections[tnum].Initialize(0.);
    }


    cond->SOC_condense(c_vars->gradObj, c_vars->delta_lb_con, c_vars->delta_ub_con,
            c_vars->condensed_h, c_vars->condensed_lb_con, c_vars->condensed_ub_con);

    delete A_qp;
    A_qp = new qpOASES::SparseMatrix(cond->condensed_num_cons, cond->condensed_num_vars,
                c_vars->Jacobian.row, c_vars->Jacobian.colind, c_vars->Jacobian.nz);
    /*
     * Prepare for qpOASES
     */

    //qpOASES defines +-infinity as +-1e20, so set absent bounds accordingly
    for (int i = 0; i < cond->condensed_num_cons; i++){
        if (std::isinf(c_vars->condensed_lb_con(i)))
            c_vars->condensed_lb_con(i) = -1e20;
        if (std::isinf(c_vars->condensed_ub_con(i)))
            c_vars->condensed_ub_con(i) = 1e20;
    }

    double *g, *lb, *ub, *lbA, *ubA;
    g = c_vars->condensed_h.array;
    lb = c_vars->condensed_lb_var.array;
    ub = c_vars->condensed_ub_var.array;
    lbA = c_vars->condensed_lb_con.array;
    ubA = c_vars->condensed_ub_con.array;


    // qpOASES options
    qpOASES::Options opts;
    opts.enableInertiaCorrection = qpOASES::BT_TRUE;
    opts.enableEqualities = qpOASES::BT_TRUE;
    opts.initialStatusBounds = qpOASES::ST_INACTIVE;

    switch(param->qpOASES_print_level){
        case 0:
            opts.printLevel = qpOASES::PL_NONE;
            break;
        case 1:
            opts.printLevel = qpOASES::PL_LOW;
            break;
        case 2:
            opts.printLevel = qpOASES::PL_MEDIUM;
            break;
        case 3:
            opts.printLevel = qpOASES::PL_HIGH;
            break;
    }

    //opts.printLevel = qpOASES::PL_MEDIUM; //PL_LOW, PL_HIGH, PL_MEDIUM, PL_NONE
    opts.numRefinementSteps = 2;
    opts.epsLITests =  2.2204e-08;

    qp->setOptions( opts );

    // Other variables for qpOASES
    double cpuTime = param->maxTimeQP;
    int maxIt = param->maxItQP;
    qpOASES::SolutionAnalysis solAna;
    qpOASES::returnValue ret;

    //Solve the QP
    if( param->debugLevel > 2 ) stats->dumpQPCpp( prob, vars, qp, param->sparseQP );

    debug << "Starting solution of SOC-QP\n" << std::flush;
    maxIt = 0.1*param->maxItQP;
    cpuTime = 0.1*param->maxTimeQP;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    ret = qp->hotstart( g, lb, ub, lbA, ubA, maxIt, &cpuTime );
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Solved SOC in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms\n";
    debug << "Finished solution of SOC-QP\n" << std::flush;


    if( ret == qpOASES::SUCCESSFUL_RETURN ){
        if( param->sparseQP == 2 )
            ret = solAna.checkCurvatureOnStronglyActiveConstraints( dynamic_cast<qpOASES::SQProblemSchur*>(qp) );
        else
            ret = solAna.checkCurvatureOnStronglyActiveConstraints( qp );
    }

    if( ret == qpOASES::SUCCESSFUL_RETURN ){
        // QP was solved successfully and curvature is positive after removing bounds
        stats->qpIterations = maxIt + 1;

        //copy successfully solved QP to initialize SOC condition qp
        (*qp_nb) = *dynamic_cast<qpOASES::SQProblemSchur*>(qp);
    }

    // Get solution from qpOASES
    qp->getPrimalSolution(c_vars->deltaXi_cond.array);
    qp->getDualSolution(c_vars->lambdaQP_cond.array);

    cond->recover_var_mult(c_vars->deltaXi_cond, c_vars->lambdaQP_cond, deltaXi, lambdaQP);


    //Matrix *corrections = new Matrix[cond->num_targets];
    //for (int tnum = 0; tnum < cond->num_targets; tnum++){
    //    corrections[tnum].Dimension(cond->targets_data[tnum].n_dep).Initialize(0.);
    //}


    debug << "Starting bound correction\n" << std::flush;
    if (ret == qpOASES::SUCCESSFUL_RETURN){
        //H = new qpOASES::SymSparseMat(cond->condensed_num_vars, cond->condensed_num_vars, c_vars->condensed_hess_row, c_vars->condensed_hess_colind, c_vars->condensed_hess_nz);
        //dynamic_cast<qpOASES::SymSparseMat*>(H)->createDiagInfo();

        int ind_1, ind_2, ind, vio_count;
        bool found_direction;
        Matrix xi_s(c_vars->xi), lambda_s(c_vars->lambda), deltaXi_s, constr_s, AdeltaXi_s;
        constr_s.Dimension(prob->nCon);
        deltaXi_s.Dimension(prob->nVar);
        AdeltaXi_s.Dimension(prob->nCon);

        for (int l = 0; l < param->max_bound_refines; l++){
            found_direction = true;
            ind_1 = 0;
            vio_count = 0;
            deltaXi_s = deltaXi;

            for (int i = 0; i < cond->num_vblocks; i++){
                if (cond->vblocks[i].dependent){
                    for (int j = 0; j < cond->vblocks[i].size; j++){
                        ind = ind_1 + j;
                        if (c_vars->xi(ind) + deltaXi(ind) < prob->lb_var(ind) - param->dep_bound_tolerance || c_vars->xi(ind) + deltaXi(ind) > prob->ub_var(ind) + param->dep_bound_tolerance){
                            vio_count++;
                            found_direction = false;
                        }
                    }
                }
                ind_1 += cond->vblocks[i].size;
            }
            if (found_direction){
                std::cout << "All dependent variable bounds are respected, exiting step correction\n";
                break;
            }
            std::cout << "Bounds violated by " << vio_count << " dependent variables, calculating correction vectors\n";

            xi_s = c_vars->xi + deltaXi;
            lambda_s = lambdaQP;

            for (int tnum = 0; tnum < cond->num_targets; tnum++){

                //Add difference between dependent state values from QP solution and integration for target tnum
                ind_1 = 0;
                ind_2 = cond->vranges[cond->targets[tnum].first_free];

                for (int i = cond->targets[tnum].first_free; i < cond->targets[tnum].vblock_end; i++){
                    if (cond->vblocks[i].dependent){
                        for (int j = 0; j < cond->vblocks[i].size; j++){

                            if (corrections[tnum](ind_1 + j) > 0 && xi_s(ind_2 + j) > prob->lb_var(ind_2 + j)){
                                corrections[tnum](ind_1 + j) -= xi_s(ind_2 + j) - prob->lb_var(ind_2 + j);
                                if (corrections[tnum](ind_1 + j) < 0) corrections[tnum](ind_1 + j) = 0;
                            }
                            else if (corrections[tnum](ind_1 + j) < 0 && xi_s(ind_2 + j) < prob->ub_var(ind_2 + j)){
                                corrections[tnum](ind_1 + j) -= xi_s(ind_2 + j) - prob->ub_var(ind_2 + j);
                                if (corrections[tnum](ind_1 + j) > 0) corrections[tnum](ind_1 + j) = 0;
                            }

                            if (xi_s(ind_2 + j) < prob->lb_var(ind_2 + j) - param->dep_bound_tolerance){
                                corrections[tnum](ind_1 + j) += prob->lb_var(ind_2 + j) - xi_s(ind_2 + j);
                            }
                            else if (xi_s(ind_2 + j) > prob->ub_var(ind_2 + j) + param->dep_bound_tolerance){
                                corrections[tnum](ind_1 + j) += prob->ub_var(ind_2 + j) - xi_s(ind_2 + j);
                            }
                        }
                        ind_1 += cond->vblocks[i].size;
                    }
                    ind_2 += cond->vblocks[i].size;
                }
            }


            cond->correction_condense(c_vars->gradObj, c_vars->delta_lb_con, c_vars->delta_ub_con, corrections, c_vars->condensed_h, c_vars->condensed_lb_con, c_vars->condensed_ub_con);

            g = c_vars->condensed_h.array;
            lbA = c_vars->condensed_lb_con.array;
            ubA = c_vars->condensed_ub_con.array;

            for (int i = 0; i < cond->condensed_num_cons; i++){
                if (std::isinf(c_vars->condensed_lb_con(i)))
                    c_vars->condensed_lb_con(i) = -1e20;
                if (std::isinf(c_vars->condensed_ub_con(i)))
                    c_vars->condensed_ub_con(i) = 1e20;
            }

            maxIt = param->maxItQP;
            //cpuTime = param->maxTimeQP;
            //

            qp_nb->setOptions(opts);

            std::cout << "Starting solution of correction qp\n";
            debug << "Starting solution of correction qp\n" << std::flush;
            std::chrono::steady_clock::time_point T0 = std::chrono::steady_clock::now();
            //if (l > 0)
                ret = qp_nb->hotstart(g, lb, ub, lbA, ubA, maxIt, &cpuTime);
            //else
            //    ret = qp_nb->init(H_qp, g, A_qp, lb, ub, lbA, ubA, maxIt, &cpuTime);

            if (ret == qpOASES::RET_MAX_NWSR_REACHED){
                std::cout << "Solution of correction SOC QP is taking too long, initializing new correction QP\n";
                cpuTime = 2.5*param->maxTimeQP;
                ret = qp_nb->init(H_qp, g, A_qp, lb, ub, lbA, ubA, maxIt, &cpuTime);
            }
            std::chrono::steady_clock::time_point T1 = std::chrono::steady_clock::now();
            std::cout << "Finished solution of correction qp in " << std::chrono::duration_cast<std::chrono::milliseconds>(T1 - T0).count() << "ms\n";
            debug << "Finished solution of correction qp in " << std::chrono::duration_cast<std::chrono::milliseconds>(T1 - T0).count() << "ms\n" << std::flush;

            if (ret == qpOASES::SUCCESSFUL_RETURN){
                qp_nb->getPrimalSolution(c_vars->deltaXi_cond.array);
                qp_nb->getDualSolution(c_vars->lambdaQP_cond.array);
                cond->recover_correction_var_mult(c_vars->deltaXi_cond, c_vars->lambdaQP_cond, corrections, deltaXi, lambdaQP);

            }
            else{break;}

        }
    }

    /*
     * Post-processing
     */
     debug << "Post-processing\n" << std::flush;

    // Compute constrJac*deltaXi, need this for second order correction step
    Atimesb( c_vars->jacNz, c_vars->jacIndRow, c_vars->jacIndCol, deltaXi, c_vars->AdeltaXi );

    // Print qpOASES error code, if any
    if(ret != qpOASES::SUCCESSFUL_RETURN)
        printf( "qpOASES error message: \"%s\"\n",
                qpOASES::getGlobalMessageHandler()->getErrorCodeMessage( ret ) );

    // Point Hessian again to the first Hessian
    c_vars->hess = c_vars->hess1;


    debug.close();

    if( ret == qpOASES::SUCCESSFUL_RETURN )
        return 0;
    else if( ret == qpOASES::RET_MAX_NWSR_REACHED )
        return 1;
    else if( ret == qpOASES::RET_HESSIAN_NOT_SPD ||
             ret == qpOASES::RET_HESSIAN_INDEFINITE ||
             ret == qpOASES::RET_INIT_FAILED_UNBOUNDEDNESS ||
             ret == qpOASES::RET_QP_UNBOUNDED ||
             ret == qpOASES::RET_HOTSTART_STOPPED_UNBOUNDEDNESS ){
        return 2;}
    else if( ret == qpOASES::RET_INIT_FAILED_INFEASIBILITY ||
             ret == qpOASES::RET_QP_INFEASIBLE ||
             ret == qpOASES::RET_HOTSTART_STOPPED_INFEASIBILITY ){
        return 3;}
    else{
        return 4;}
}


int SCQP_rest_method::solve_SOC_QP(Matrix&deltaXi, Matrix& lambdaQP){
    throw std::invalid_argument("SCQP_rest_method::solve_SOC_QP: Error, SOC not implemented for projection based linesearch method");
}




} // namespace blockSQP


