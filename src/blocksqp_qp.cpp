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
        vars->hess2_calculated = true;

        /*
        for (int i = 0; i < prob->nBlocks; i++){
            vars->hess_save[i] = vars->hess2[i];
        }
        */
    }

    // 'Nontrivial' convex combinations
    if( maxQP > 2 && idx < maxQP - 1 )
    {
        /* Convexification parameter: mu_l = l / (maxQP-1).
         * Compute it only in the first iteration, afterwards update
         * by recursion: mu_l/mu_(l-1) */

        /*
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
        */


        std::cout << "ComputeNextHessian: maxQP = " << maxQP << ", idx = " << idx << "\n";
        for (int i = 0; i < vars->nBlocks; i++){
            vars->hess_conv[i] = vars->hess1[i] * (1 - static_cast<double>(idx)/static_cast<double>(maxQP - 1)) + vars->hess2[i] * (static_cast<double>(idx)/static_cast<double>(maxQP - 1));
        }
        vars->hess = vars->hess_conv;

        /*
        double maxdiff = 0.;
        double max_hess2, max_hessconv, max_val;
        for (int i = 0; i < prob->nBlocks; i++){
            for (int k = 0; k < vars->hess_conv[i].m; k++){
                for (int j = k; j < vars->hess_conv[i].m; j++){

                    if (std::abs(vars->hess2[i](j,k)) > max_val){
                        max_val = std::abs(vars->hess2[i](j,k));
                    }

                    if (vars->hess2[i](j,k) - vars->hess_conv[i](j,k) > maxdiff){
                        maxdiff = vars->hess2[i](j,k) - vars->hess_conv[i](j,k);
                        max_hess2 = vars->hess2[i](j,k); max_hessconv = vars->hess_conv[i](j,k);
                    }
                    else if (vars->hess_conv[i](j,k) - vars->hess2[i](j,k) > maxdiff){
                        maxdiff = vars->hess_conv[i](j,k) - vars->hess2[i](j,k);
                        max_hess2 = vars->hess2[i](j,k); max_hessconv = vars->hess_conv[i](j,k);
                    }
                }
            }
        }
        std::cout << "maxdiff = " << maxdiff << ", hess2_max = " << max_hess2 << ", hessconv_max = " << max_hessconv << ", max_val = " << max_val << "\n";
        */

    }
    else{
        vars->hess = vars->hess2;
    }
}


/**
 * Inner loop of SQP algorithm:
 * Solve a sequence of QPs until pos. def. assumption (G3*) is satisfied.
 */
int SQPmethod::solveQP( Matrix &deltaXi, Matrix &lambdaQP, bool conv_qp )
{

    Matrix jacT;
    int maxQP, l;
    if( param->globalization == 1 &&
        (param->hessUpdate == 1) &&
        stats->itCount > 1 &&
        !conv_qp
        )
    {
        maxQP = param->maxConvQP + 1;
    }
    else
        maxQP = 1;

    //Solve convex QP using fallback hessian if indefinite approximations are normally tried first.
    if (conv_qp && (param->hessUpdate == 1)){
        vars->hess = vars->hess2;
        if (!vars->hess2_calculated){
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

            vars->hess2_calculated = true;
        }
    }

    vars->conv_qp_solved = false;


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
    for(l = 0; l<maxQP; l++ )
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

        if (l == maxQP - 1)
            cpuTime = param->maxTimeQP;
        else
            cpuTime = std::min(2.5*vars->avg_solution_duration, param->maxTimeQP);

        if( (qp->getStatus() == qpOASES::QPS_HOMOTOPYQPSOLVED ||
            qp->getStatus() == qpOASES::QPS_SOLVED ) && vars->use_homotopy)
        {
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

            ret = qp->hotstart( H_qp, g, A_qp, lb, ub, lbA, ubA, maxIt, &cpuTime );

            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << "Solved hotstarted QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms\n";
            std::cout << "ret is " << ret << "\n";
        }
        else
        {
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            ret = qp->init( H_qp, g, A_qp, lb, ub, lbA, ubA, maxIt, &cpuTime );
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << "Solved initial QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms\n";
            std::cout << "ret is " << ret << "\n";
            if (ret == qpOASES::SUCCESSFUL_RETURN)
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

            if( ret == qpOASES::SUCCESSFUL_RETURN)// && stats->itCount != 4)
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
            vars->conv_qp_solved = true;
        }

    } // End of QP solving loop


    /*
     * Post-processing
     */

    // Get solution from qpOASES
    if (ret == qpOASES::SUCCESSFUL_RETURN){
        qp->getPrimalSolution(deltaXi.ARRAY());
        qp->getDualSolution(lambdaQP.ARRAY());

        vars->avg_solution_duration -= vars->solution_durations[vars->dur_pos]/10.0;
        vars->solution_durations[vars->dur_pos] = cpuTime;
        vars->avg_solution_duration += cpuTime/10;
        vars->dur_pos = (vars->dur_pos + 1)%10;
    }
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
    if( !param->hessLimMem && maxQP > 2 && param->convStrategy == 0)
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
    else if (!param->hessLimMem && maxQP > 2 && param->convStrategy == 1){
        for (int i = 0; i < vars->nBlocks; i++){
            for (int j = 0; j < vars->hess1[i].m; j++){
                vars->hess1[i](j,j) -= vars->conv_identity_scale;
            }
        }
    }

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

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    ret = qp->hotstart( g, lb, ub, lbA, ubA, maxIt, &cpuTime );
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Solved SOC in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms\n";

    /*
     * Check assumption (G3*) if nonconvex QP was solved
     */
    //if( l < maxQP-1 && conv_qp ){

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


void SCQPmethod::compute_condensed_QP(int idx, int maxQP){
    SCQPiterate *c_vars = dynamic_cast<SCQPiterate*>(vars);
    if (idx == 0){
        //Condense the QP
        std::chrono::steady_clock::time_point T0 = std::chrono::steady_clock::now();

        cond->full_condense(c_vars->gradObj, c_vars->Jacobian, c_vars->hess,
            c_vars->delta_lb_var, c_vars->delta_ub_var, c_vars->delta_lb_con, c_vars->delta_ub_con,
                c_vars->condensed_h, c_vars->condensed_Jacobian, c_vars->condensed_hess, c_vars->condensed_lb_var, c_vars->condensed_ub_var, c_vars->condensed_lb_con, c_vars->condensed_ub_con);

        std::chrono::steady_clock::time_point T1 = std::chrono::steady_clock::now();
        std::cout << "Condensing the full qp took " << std::chrono::duration_cast<std::chrono::milliseconds>(T1 - T0).count() << " ms\n";

        //Update qpOASES-jacobian to new condensed jacobian
        delete A_qp;
        A_qp = new qpOASES::SparseMatrix(cond->condensed_num_cons, cond->condensed_num_vars,
                c_vars->condensed_Jacobian.row, c_vars->condensed_Jacobian.colind, c_vars->condensed_Jacobian.nz);

        //Convert block hessian to sparse qpOASES matrix
        c_vars->convertHessian(cond->condensed_num_vars, cond->condensed_num_hessblocks, param->eps, c_vars->condensed_hess,
                        c_vars->condensed_hess_nz, c_vars->condensed_hess_row, c_vars->condensed_hess_colind, c_vars->condensed_hess_loind);

        delete H_qp;
        H_qp = new qpOASES::SymSparseMat(cond->condensed_num_vars, cond->condensed_num_vars,
                                    c_vars->condensed_hess_row, c_vars->condensed_hess_colind, c_vars->condensed_hess_nz);
        dynamic_cast<qpOASES::SymSparseMat*>(H_qp)->createDiagInfo();
    }
    else{
        computeNextHessian(idx, maxQP);
        cond->new_hessian_condense(c_vars->hess, c_vars->condensed_h, c_vars->condensed_hess);

        //Convert block hessian to sparse qpOASES matrix
        c_vars->convertHessian(cond->condensed_num_vars, cond->condensed_num_hessblocks, param->eps, c_vars->condensed_hess,
                        c_vars->condensed_hess_nz, c_vars->condensed_hess_row, c_vars->condensed_hess_colind, c_vars->condensed_hess_loind);

        delete H_qp;
        H_qp = new qpOASES::SymSparseMat(cond->condensed_num_vars, cond->condensed_num_vars,
                                    c_vars->condensed_hess_row, c_vars->condensed_hess_colind, c_vars->condensed_hess_nz);
        dynamic_cast<qpOASES::SymSparseMat*>(H_qp)->createDiagInfo();
    }

    return;
}



int SCQPmethod::solveQP(Matrix &deltaXi, Matrix &lambdaQP, bool conv_qp){

    SCQPiterate *c_vars = dynamic_cast<SCQPiterate*>(vars);

    Matrix jacT;
    int maxQP, l;
    if( param->globalization == 1 &&
        param->hessUpdate == 1 &&
        stats->itCount > 1 &&
        !conv_qp
        )
    {
        maxQP = param->maxConvQP + 1;
    }
    else
        maxQP = 1;



    //Solve convex QP using fallback hessian if indefinite approximations are normally tried first.
    if (conv_qp && (param->hessUpdate == 1)){
        vars->hess = vars->hess2;
        if (!vars->hess2_calculated){
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

            vars->hess2_calculated = true;
        }
    }

    vars->conv_qp_solved = false;
    //Condense the QP before invoking qpOASES
    /*cond->full_condense(c_vars->gradObj, c_vars->Jacobian, c_vars->hess,
        c_vars->delta_lb_var, c_vars->delta_ub_var, c_vars->delta_lb_con, c_vars->delta_ub_con,
            c_vars->condensed_h, c_vars->condensed_Jacobian, c_vars->condensed_hess, c_vars->condensed_lb_var, c_vars->condensed_ub_var, c_vars->condensed_lb_con, c_vars->condensed_ub_con);

    delete A_qp;
    A_qp = new qpOASES::SparseMatrix(cond->condensed_num_cons, cond->condensed_num_vars,
                c_vars->condensed_Jacobian.row, c_vars->condensed_Jacobian.colind, c_vars->condensed_Jacobian.nz);*/
    compute_condensed_QP(0, maxQP);

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


    for(l = 0; l<maxQP; l++){
        if( l > 0 ){
            // If the solution of the first QP was rejected, consider second Hessian
            stats->qpResolve++;
            *qp = *qpSave;
            /*computeNextHessian( l, maxQP );

            cond->new_hessian_condense(c_vars->hess, c_vars->condensed_h, c_vars->condensed_hess);*/
            compute_condensed_QP(l, maxQP);
            g = c_vars->condensed_h.array;
        }

        if( l == maxQP-1 )
        {// Enable inertia correction for supposedly convex QPs, just in case
            opts.enableInertiaCorrection = qpOASES::BT_TRUE;
            qp->setOptions( opts );
        }

        // Convert block-Hessian to sparse format
        /*c_vars->convertHessian(cond->condensed_num_vars, cond->condensed_num_hessblocks, param->eps, c_vars->condensed_hess,
                        c_vars->condensed_hess_nz, c_vars->condensed_hess_row, c_vars->condensed_hess_colind, c_vars->condensed_hess_loind);
        delete H_qp;
        H_qp = new qpOASES::SymSparseMat(cond->condensed_num_vars, cond->condensed_num_vars,
                                    c_vars->condensed_hess_row, c_vars->condensed_hess_colind, c_vars->condensed_hess_nz);
        dynamic_cast<qpOASES::SymSparseMat*>(H_qp)->createDiagInfo();*/

        /*
         * Call qpOASES
         */
        if( param->debugLevel > 2 ) stats->dumpQPCpp( prob, vars, qp, param->sparseQP );

        maxIt = param->maxItQP;
        if (l == maxQP - 1)
            cpuTime = param->maxTimeQP;
        else
            cpuTime = std::min(2.5*c_vars->avg_solution_duration, param->maxTimeQP);

        if( (qp->getStatus() == qpOASES::QPS_HOMOTOPYQPSOLVED ||
            qp->getStatus() == qpOASES::QPS_SOLVED ) && c_vars->use_homotopy)
        {
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

            ret = qp->hotstart(H_qp, g, A_qp, lb, ub, lbA, ubA, maxIt, &cpuTime);

            std::cout << "cpuTime is " << cpuTime << "\n";

            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << "Solved hotstarted QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms\n";
            std::cout << "ret is " << ret << "\n";
        }
        else
        {
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            ret = qp->init( H_qp, g, A_qp, lb, ub, lbA, ubA, maxIt, &cpuTime );
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << "Solved initial QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms\n";
            std::cout << "ret is " << ret << "\n";
            if (ret == qpOASES::SUCCESSFUL_RETURN)
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
            vars->conv_qp_solved = true;
        }
    } // End of QP solving loop


    // Get solution from qpOASES
    if (ret == qpOASES::SUCCESSFUL_RETURN){
        qp->getPrimalSolution(c_vars->deltaXi_cond.array);
        qp->getDualSolution(c_vars->lambdaQP_cond.array);
        cond->recover_var_mult(c_vars->deltaXi_cond, c_vars->lambdaQP_cond, deltaXi, lambdaQP);

        c_vars->avg_solution_duration -= c_vars->solution_durations[c_vars->dur_pos]/10;
        c_vars->solution_durations[c_vars->dur_pos] = cpuTime;
        c_vars->avg_solution_duration += cpuTime/10;
        c_vars->dur_pos = (c_vars->dur_pos + 1)%10;
    }

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
    opts.terminationTolerance = param->qpOASES_terminationTolerance;

    qp->setOptions( opts );

    // Other variables for qpOASES
    double cpuTime = param->maxTimeQP;
    int maxIt = param->maxItQP;
    qpOASES::SolutionAnalysis solAna;
    qpOASES::returnValue ret;


    if( param->debugLevel > 2 ) stats->dumpQPCpp( prob, vars, qp, param->sparseQP );

    maxIt = param->maxItQP;
    cpuTime = param->maxTimeQP;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    ret = qp->hotstart( g, lb, ub, lbA, ubA, maxIt, &cpuTime );
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Solved SOC in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms\n";


    /*
     * Check assumption (G3*) if nonconvex QP was solved
     */
    /*if( ret == qpOASES::SUCCESSFUL_RETURN )
    {
        ret = solAna.checkCurvatureOnStronglyActiveConstraints( dynamic_cast<qpOASES::SQProblemSchur*>(qp) );
        std::cout << "Checked curvature, ret is " << ret << "\n";
    }*/

    if( ret == qpOASES::SUCCESSFUL_RETURN ){
        // QP was solved successfully and curvature is positive after removing bounds
        stats->qpIterations = maxIt + 1;
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

    //Post-processing

    // Compute constrJac*deltaXi, need this for second order correction step
    Atimesb( c_vars->jacNz, c_vars->jacIndRow, c_vars->jacIndCol, deltaXi, c_vars->AdeltaXi );

    // Print qpOASES error code, if any
    if( ret != qpOASES::SUCCESSFUL_RETURN)
        printf( "qpOASES error message: \"%s\"\n",
                qpOASES::getGlobalMessageHandler()->getErrorCodeMessage( ret ) );

    // Point Hessian again to the first Hessian
    c_vars->hess = c_vars->hess1;


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


int SCQP_bound_method::solveQP( Matrix &deltaXi, Matrix &lambdaQP, bool conv_qp ){

    SCQPiterate *c_vars = dynamic_cast<SCQPiterate*>(vars);

    int maxQP, l;
    if (param->globalization == 1 && param->hessUpdate == 1 && stats->itCount > 1 && !conv_qp){
        maxQP = param->maxConvQP + 1;
    }
    else
        maxQP = 1;


    //Solve convex QP using fallback hessian if indefinite approximations are normally tried first.
    if (conv_qp && (param->hessUpdate == 1)){
        vars->hess = vars->hess2;
        if (!vars->hess2_calculated){
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

            vars->hess2_calculated = true;
        }
    }

    vars->conv_qp_solved = false;
    /*
    //Condense the QP
    cond->full_condense(c_vars->gradObj, c_vars->Jacobian, c_vars->hess,
        c_vars->delta_lb_var, c_vars->delta_ub_var, c_vars->delta_lb_con, c_vars->delta_ub_con,
            c_vars->condensed_h, c_vars->condensed_Jacobian, c_vars->condensed_hess, c_vars->condensed_lb_var, c_vars->condensed_ub_var, c_vars->condensed_lb_con, c_vars->condensed_ub_con);

    //Update qpOASES-jacobian to new condensed jacobian
    delete A_qp;
    A_qp = new qpOASES::SparseMatrix(cond->condensed_num_cons, cond->condensed_num_vars,
            c_vars->condensed_Jacobian.row, c_vars->condensed_Jacobian.colind, c_vars->condensed_Jacobian.nz);
    */
    compute_condensed_QP(0, maxQP);

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


    for (l = 0; l < maxQP; l++){
        if( l > 0 ){
            stats->qpResolve++;
            *qp = *qpSave;

            /*computeNextHessian( l, maxQP );

            cond->new_hessian_condense(c_vars->hess, c_vars->condensed_h, c_vars->condensed_hess);*/

            //If maximum scaling factor is reached, skip the remaining identity scaled QPs
            compute_condensed_QP(l, maxQP);
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

        /*c_vars->convertHessian(cond->condensed_num_vars, cond->condensed_num_hessblocks, param->eps, c_vars->condensed_hess,
                        c_vars->condensed_hess_nz, c_vars->condensed_hess_row, c_vars->condensed_hess_colind, c_vars->condensed_hess_loind);

        delete H_qp;
        H_qp = new qpOASES::SymSparseMat(cond->condensed_num_vars, cond->condensed_num_vars,
                                    c_vars->condensed_hess_row, c_vars->condensed_hess_colind, c_vars->condensed_hess_nz);
        dynamic_cast<qpOASES::SymSparseMat*>(H_qp)->createDiagInfo();*/

        /*
         * Call qpOASES
         */
        if( param->debugLevel > 2 ) stats->dumpQPCpp( prob, vars, qp, param->sparseQP );

        maxIt = param->maxItQP;
        if (l == maxQP - 1)
            cpuTime = param->maxTimeQP;
        else
            cpuTime = std::min(2.5*c_vars->avg_solution_duration, param->maxTimeQP);

        if( (qp->getStatus() == qpOASES::QPS_HOMOTOPYQPSOLVED ||
            qp->getStatus() == qpOASES::QPS_SOLVED ) && c_vars->use_homotopy){

            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

            ret = qp->hotstart(H_qp, g, A_qp, lb, ub, lbA, ubA, maxIt, &cpuTime);
            std::cout << "cpuTime is " << cpuTime << "\n";

            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << "Solved hotstarted QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms\n";
            std::cout << "ret is " << ret << "\n";
        }
        else{
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            ret = qp->init( H_qp, g, A_qp, lb, ub, lbA, ubA, maxIt, &cpuTime );
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << "Solved initial QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms\n";
            std::cout << "ret is " << ret << "\n";
            if (ret == qpOASES::SUCCESSFUL_RETURN)
                c_vars->use_homotopy = true;
        }

        /*
         * Check assumption (G3*) if nonconvex QP was solved
         */
        if( l < maxQP-1){
            if( ret == qpOASES::SUCCESSFUL_RETURN ){
                //(*dynamic_cast<qpOASES::SQProblemSchur*>(qpSave)) = *dynamic_cast<qpOASES::SQProblemSchur*>(qp);
                //ret = solAna.checkCurvatureOnStronglyActiveConstraints(dynamic_cast<qpOASES::SQProblemSchur*>(qpSave));

                (*qp_check) = *dynamic_cast<qpOASES::SQProblemSchur*>(qp);
                ret = solAna.checkCurvatureOnStronglyActiveConstraints(qp_check);

                //ret = solAna.checkCurvatureOnStronglyActiveConstraints( dynamic_cast<qpOASES::SQProblemSchur*>(qp) );
                std::cout << "Curvature checked, ret is " << ret << "\n";
            }

            if( ret == qpOASES::SUCCESSFUL_RETURN ){
                // QP was solved successfully and curvature is positive after removing bounds
                stats->qpIterations = maxIt + 1;
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
            vars->conv_qp_solved = true;
        }

        if (ret == qpOASES::SUCCESSFUL_RETURN){
            c_vars->avg_solution_duration -= c_vars->solution_durations[c_vars->dur_pos]/10;
            c_vars->solution_durations[c_vars->dur_pos] = cpuTime;
            c_vars->avg_solution_duration += cpuTime/10;
            c_vars->dur_pos = (c_vars->dur_pos + 1)%10;
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
                maxIt = param->maxItQP;
                cpuTime_ref = std::max(cpuTime, param->maxTimeQP);

                qp->setOptions(opts);

                //Bounds matrices were modified in place, so pointers lb, ub, lbA, ubA need not be updated
                std::chrono::steady_clock::time_point begin_ = std::chrono::steady_clock::now();
                ret = qp->hotstart(g, lb, ub, lbA, ubA, maxIt, &cpuTime_ref);
                std::cout << "ret is " << ret << "\n";
                std::chrono::steady_clock::time_point end_ = std::chrono::steady_clock::now();
                std::cout << "Solved QP with added bounds in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin_).count() << "ms\n";


                if (ret == qpOASES::RET_MAX_NWSR_REACHED){
                    std::cout << "Solution of QP with added bounds is taking too long, initialize new QP\n";
                    maxIt = param->maxItQP;
                    cpuTime_ref = param->maxTimeQP;

                    begin_ = std::chrono::steady_clock::now();
                    ret = qp->init(H_qp, g, A_qp, lb, ub, lbA, ubA, maxIt, &cpuTime_ref);
                    end_ = std::chrono::steady_clock::now();
                    std::cout << "Finished solution initialized SQP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin_).count() << "ms\n";
                }

                /*if (ret == qpOASES::SUCCESSFUL_RETURN){
                    ret = solAna.checkCurvatureOnStronglyActiveConstraints(dynamic_cast<qpOASES::SQProblemSchur*>(qp));
                }*/

                if (ret == qpOASES::SUCCESSFUL_RETURN){
                    qp->getPrimalSolution(c_vars->deltaXi_cond.array);
                    qp->getDualSolution(c_vars->lambdaQP_cond.array);
                    cond->recover_var_mult(c_vars->deltaXi_cond, c_vars->lambdaQP_cond, deltaXi, lambdaQP);
                }
                else{
                    std::cout << "Error in QP with added bounds, convexify the hessian further\n";
                    std::cout << "ret is " << ret << "\n";
                    solution_found = false;
                    //Remove added dep bounds
                    for (int j = cond->num_true_cons; j < prob->nCon; j++){
                        c_vars->condensed_lb_con(j) = -1e20;
                        c_vars->condensed_ub_con(j) = 1e20;
                    }
                    break;
                }
            }
        }

        if (solution_found){
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


int SCQP_bound_method::solve_SOC_QP( Matrix &deltaXi, Matrix &lambdaQP){

    SCQPiterate *c_vars = dynamic_cast<SCQPiterate*>(vars);

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
    opts.terminationTolerance = param->qpOASES_terminationTolerance;

    qp->setOptions( opts );

    // Other variables for qpOASES
    double cpuTime = param->maxTimeQP;
    int maxIt = param->maxItQP;
    qpOASES::SolutionAnalysis solAna;
    qpOASES::returnValue ret;


    if( param->debugLevel > 2 ) stats->dumpQPCpp( prob, vars, qp, param->sparseQP );

    maxIt = param->maxItQP;
    cpuTime = param->maxTimeQP;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    ret = qp->hotstart( g, lb, ub, lbA, ubA, maxIt, &cpuTime );
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Solved SOC in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms\n";

    /*
     * Check assumption (G3*) if nonconvex QP was solved
     */
    /*if( ret == qpOASES::SUCCESSFUL_RETURN )
    {
        ret = solAna.checkCurvatureOnStronglyActiveConstraints( dynamic_cast<qpOASES::SQProblemSchur*>(qp) );
        std::cout << "Checked curvature, ret is " << ret << "\n";
    }*/

    if( ret == qpOASES::SUCCESSFUL_RETURN ){
        // QP was solved successfully and curvature is positive after removing bounds
        stats->qpIterations = maxIt + 1;
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
                            //std::cout << "Variable " << ind << " violated bounds, adding their bounds to condensed QP\n";
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
                if (k >= 1)
                    (*qp) = *qpSave;
                break;
            }
            std::cout << "Bounds violated by " << vio_count << " dependent variables, adding their bounds to the QP\n";
            maxIt = param->maxItQP;
            cpuTime_ref = std::max(cpuTime, param->maxTimeQP);

            qp->setOptions(opts);

            std::chrono::steady_clock::time_point begin_ = std::chrono::steady_clock::now();

            ret = qp->hotstart(g, lb, ub, lbA, ubA, maxIt, &cpuTime_ref);

            std::cout << "ret is " << ret << "\n";
            std::chrono::steady_clock::time_point end_ = std::chrono::steady_clock::now();
            std::cout << "Solved QP with added bounds in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin_).count() << "ms\n";

            if (ret == qpOASES::RET_MAX_NWSR_REACHED){
                maxIt = param->maxItQP;
                cpuTime_ref = param->maxTimeQP;

                std::cout << "QP solution taking too long, initialize new QP\n";
                begin_ = std::chrono::steady_clock::now();
                ret = qp->init(H_qp, g, A_qp, lb, ub, lbA, ubA, maxIt, &cpuTime_ref);
                end_ = std::chrono::steady_clock::now();
                std::cout << "Finished solution of reduced copy QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin_).count() << "ms\n";
            }

            /*if (ret == qpOASES::SUCCESSFUL_RETURN){
                ret = solAna.checkCurvatureOnStronglyActiveConstraints(dynamic_cast<qpOASES::SQProblemSchur*>(qp));
            }*/

            if (ret == qpOASES::SUCCESSFUL_RETURN){
                qp->getPrimalSolution(c_vars->deltaXi_cond.array);
                qp->getDualSolution(c_vars->lambdaQP_cond.array);
                cond->recover_var_mult(c_vars->deltaXi_cond, c_vars->lambdaQP_cond, deltaXi, lambdaQP);
            }
            else{
                break;
            }

        }
    }

    //Post-processing

    // Compute constrJac*deltaXi, need this for second order correction step
    Atimesb( c_vars->jacNz, c_vars->jacIndRow, c_vars->jacIndCol, deltaXi, c_vars->AdeltaXi );

    // Print qpOASES error code, if any
    if( ret != qpOASES::SUCCESSFUL_RETURN)
        printf( "qpOASES error message: \"%s\"\n",
                qpOASES::getGlobalMessageHandler()->getErrorCodeMessage( ret ) );

    // Point Hessian again to the first Hessian
    c_vars->hess = c_vars->hess1;

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



int SCQP_correction_method::solveQP(Matrix &deltaXi, Matrix &lambdaQP, bool conv_qp){

    SCQP_correction_iterate *c_vars = dynamic_cast<SCQP_correction_iterate*>(vars);

    int maxQP, l;
    if( param->globalization == 1 &&
        param->hessUpdate == 1 &&
        stats->itCount > 1 &&
        !conv_qp
        )
    {
        maxQP = param->maxConvQP + 1;
    }
    else
        maxQP = 1;



    //Solve convex QP using fallback hessian if indefinite approximations are normally tried first.
    if (conv_qp && (param->hessUpdate == 1)){
        vars->hess = vars->hess2;
        if (!vars->hess2_calculated){
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

            vars->hess2_calculated = true;
        }
    }

    vars->conv_qp_solved = false;
    //Condense QP before invoking qpOASES if possible
    /*std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    cond->full_condense(c_vars->gradObj, c_vars->Jacobian, c_vars->hess,
        c_vars->delta_lb_var, c_vars->delta_ub_var, c_vars->delta_lb_con, c_vars->delta_ub_con,
            c_vars->condensed_h, c_vars->condensed_Jacobian, c_vars->condensed_hess, c_vars->condensed_lb_var, c_vars->condensed_ub_var, c_vars->condensed_lb_con, c_vars->condensed_ub_con);
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    std::cout << "Condensing of the full QP took " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms\n";

    delete A_qp;
    A_qp = new qpOASES::SparseMatrix(cond->condensed_num_cons, cond->condensed_num_vars,
                c_vars->condensed_Jacobian.row, c_vars->condensed_Jacobian.colind, c_vars->condensed_Jacobian.nz);*/
    compute_condensed_QP(0, maxQP);

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
    opts.terminationTolerance = param->qpOASES_terminationTolerance;

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
    for(l = 0; l<maxQP; l++ )
    {
        //Compute a new Hessian
        if( l > 0 ){
            // If the solution of the first QP was rejected, consider second Hessian
            stats->qpResolve++;
            *qp = *qpSave;

            /*computeNextHessian( l, maxQP );

            cond->new_hessian_condense(c_vars->hess, c_vars->condensed_h, c_vars->condensed_hess);*/
            compute_condensed_QP(l, maxQP);
            g = c_vars->condensed_h.array;
        }

        if( l == maxQP-1 ){
            //Enable inertia correction for supposedly convex QPs, just in case
            opts.enableInertiaCorrection = qpOASES::BT_TRUE;
            qp->setOptions( opts );
        }

        // Convert block-Hessian to sparse format
        /*delete H_qp;

        c_vars->convertHessian(cond->condensed_num_vars, cond->condensed_num_hessblocks, param->eps, c_vars->condensed_hess,
                        c_vars->condensed_hess_nz, c_vars->condensed_hess_row, c_vars->condensed_hess_colind, c_vars->condensed_hess_loind);

        H_qp = new qpOASES::SymSparseMat(cond->condensed_num_vars, cond->condensed_num_vars,
                                    c_vars->condensed_hess_row, c_vars->condensed_hess_colind, c_vars->condensed_hess_nz);
        dynamic_cast<qpOASES::SymSparseMat*>(H_qp)->createDiagInfo();*/

        /*
         * Call qpOASES
         */
        if( param->debugLevel > 2 ) stats->dumpQPCpp( prob, vars, qp, param->sparseQP );

        maxIt = param->maxItQP;

        if (l == maxQP - 1)
            cpuTime = param->maxTimeQP;
        else
            cpuTime = std::min(2.5*c_vars->avg_solution_duration, param->maxTimeQP);

        std::cout << "avg_solution_duration = " << c_vars->avg_solution_duration << "\n";

        if( (qp->getStatus() == qpOASES::QPS_HOMOTOPYQPSOLVED || qp->getStatus() == qpOASES::QPS_SOLVED ) && c_vars->use_homotopy){

            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

            ret = qp->hotstart(H_qp, g, A_qp, lb, ub, lbA, ubA, maxIt, &cpuTime);

            std::cout << "cpuTime is " << cpuTime << "\n";

            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << "Solved hotstarted QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms\n";
            std::cout << "ret is " << ret << "\n";
        }
        else{
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            ret = qp->init( H_qp, g, A_qp, lb, ub, lbA, ubA, maxIt, &cpuTime );
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << "Solved initial QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms\n";
            std::cout << "ret is " << ret << "\n";

            if (ret == qpOASES::SUCCESSFUL_RETURN)
                c_vars->use_homotopy = true;
        }

        /*
         * Check assumption (G3*) if nonconvex QP was solved
         */
        if (l < maxQP-1){
            if (ret == qpOASES::SUCCESSFUL_RETURN){
                //(*dynamic_cast<qpOASES::SQProblemSchur*>(qpSave)) = *dynamic_cast<qpOASES::SQProblemSchur*>(qp);
                //ret = solAna.checkCurvatureOnStronglyActiveConstraints(dynamic_cast<qpOASES::SQProblemSchur*>(qpSave));
                (*qp_check) = *dynamic_cast<qpOASES::SQProblemSchur*>(qp);
                ret = solAna.checkCurvatureOnStronglyActiveConstraints(qp_check);
            }

            /*if (stats->itCount == 4)// && (l < maxQP - 2))
            {
                ret = qpOASES::RET_MAX_NWSR_REACHED;
            }*/

            if( ret == qpOASES::SUCCESSFUL_RETURN )
            {// QP was solved successfully and curvature is positive after removing bounds
                stats->qpIterations = maxIt + 1;
            }
            else{
                // QP solution is rejected, save statistics
                stats->qpIterations2++;
                stats->rejectedSR1++;
            }
        }
        else{
            // Convex QP was solved, no need to check assumption (G3*)
            stats->qpIterations += maxIt + 1;
            vars->conv_qp_solved = true;
        }

        if (ret == qpOASES::SUCCESSFUL_RETURN){
            std::cout << "dur_pos = " << c_vars->dur_pos << ", s_dur[dur_pos] = " << c_vars->solution_durations[c_vars->dur_pos] << ", cpuTime = " << cpuTime << "\n";

            //Save the time required to solve the QP to set maximum solve times in future iterations and prevent wasting solution time on ill posed indefinite QPs
            c_vars->avg_solution_duration -= c_vars->solution_durations[c_vars->dur_pos]/10;
            c_vars->solution_durations[c_vars->dur_pos] = cpuTime;
            c_vars->avg_solution_duration += cpuTime/10;
            c_vars->dur_pos = (c_vars->dur_pos + 1)%10;
        }


        bool solution_found = false;
        if (ret == qpOASES::SUCCESSFUL_RETURN){
            solution_found = true;

            int ind_1, ind_2, ind, vio_count, max_vio_index;
            double max_dep_bound_violation;
            bool found_direction;
            Matrix xi_s(c_vars->xi);

            double cpuTime_ref;

            double *g_corr, *lbA_corr, *ubA_corr;

            //Reset correction vectors
            for (int tnum = 0; tnum < cond->num_targets; tnum++){
                corrections[tnum].Initialize(0.);
            }

            //Get current solution
            qp->getPrimalSolution(c_vars->deltaXi_cond.array);
            qp->getDualSolution(c_vars->lambdaQP_cond.array);
            cond->recover_var_mult(c_vars->deltaXi_cond, c_vars->lambdaQP_cond, deltaXi, lambdaQP);

            //Check how many dependent variables violate bounds, exit if no bounds are violated
            for (int k = 0; k < param->max_correction_steps; k++){
                found_direction = true;
                ind_1 = 0;
                vio_count = 0;
                max_dep_bound_violation = 0;
                xi_s = c_vars->xi + deltaXi;

                for (int i = 0; i < cond->num_vblocks; i++){
                    if (cond->vblocks[i].dependent){
                        for (int j = 0; j < cond->vblocks[i].size; j++){
                            ind = ind_1 + j;
                            if (xi_s(ind) < prob->lb_var(ind) - param->dep_bound_tolerance || xi_s(ind) > prob->ub_var(ind) + param->dep_bound_tolerance){
                                vio_count++;
                                found_direction = false;

                                //Calculate maximum dep bound violation
                                if (prob->lb_var(ind) - xi_s(ind) > max_dep_bound_violation){
                                    max_dep_bound_violation = prob->lb_var(ind) - xi_s(ind);
                                    max_vio_index = ind;
                                }
                                else if (xi_s(ind) - prob->ub_var(ind) > max_dep_bound_violation){
                                    max_dep_bound_violation = xi_s(ind) - prob->ub_var(ind);
                                    max_vio_index = ind;
                                }

                            }
                        }
                    }
                    ind_1 += cond->vblocks[i].size;
                }
                if (found_direction){
                    std::cout << "All dependent variable bounds are respected, exiting step correction\n";
                    /*if (k > 0)
                        (*qp) = *qpSave;*/
                    break;
                }
                std::cout << "Bounds violated by " << vio_count << " dependent variables, calculating correction vectors\n";

                //If a variable is being corrected and not at a bounds, reduce correction
                //If a variable violates a bound, add to its correction term
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

                                /*
                                if (xi_s(ind_2 + j) < prob->lb_var(ind_2 + j) - param->dep_bound_tolerance){
                                    corrections[tnum](ind_1 + j) += prob->lb_var(ind_2 + j) - xi_s(ind_2 + j);
                                }
                                else if (xi_s(ind_2 + j) > prob->ub_var(ind_2 + j) + param->dep_bound_tolerance){
                                    corrections[tnum](ind_1 + j) += prob->ub_var(ind_2 + j) - xi_s(ind_2 + j);
                                }
                                */
                            }
                            ind_1 += cond->vblocks[i].size;
                        }
                        ind_2 += cond->vblocks[i].size;
                    }
                }

                //Condense the QP, adding the correction to g = Gu + g
                cond->correction_condense(c_vars->gradObj, c_vars->delta_lb_con, c_vars->delta_ub_con, corrections, c_vars->corrected_h, c_vars->corrected_lb_con, c_vars->corrected_ub_con);

                for (int i = 0; i < cond->condensed_num_cons; i++){
                    if (std::isinf(c_vars->corrected_lb_con(i)))
                        c_vars->corrected_lb_con(i) = -1e20;
                    if (std::isinf(c_vars->corrected_ub_con(i)))
                        c_vars->corrected_ub_con(i) = 1e20;
                }

                g_corr = c_vars->corrected_h.array;
                lbA_corr = c_vars->corrected_lb_con.array;
                ubA_corr = c_vars->corrected_ub_con.array;

                //Solve the corrected QP
                maxIt = param->maxItQP;
                cpuTime_ref = std::max(0.25*cpuTime, 0.1*param->maxTimeQP);

                qp->setOptions(opts);

                std::cout << "Starting solution of correction qp\n";
                std::cout << "Max dep bound violation is " << max_dep_bound_violation << " at index " << max_vio_index << "\n" << std::flush;

                std::chrono::steady_clock::time_point T0 = std::chrono::steady_clock::now();

                ret = qp->hotstart(g_corr, lb, ub, lbA_corr, ubA_corr, maxIt, &cpuTime_ref);

                std::chrono::steady_clock::time_point T1 = std::chrono::steady_clock::now();
                std::cout << "Finished solution of correction qp in " << std::chrono::duration_cast<std::chrono::milliseconds>(T1 - T0).count() << "ms\n";

                if (ret == qpOASES::RET_MAX_NWSR_REACHED){
                    std::cout << "Solution of correction QP is taking too long, initialize new QP\n";
                    maxIt = param->maxItQP;
                    cpuTime_ref = param->maxTimeQP;

                    T0 = std::chrono::steady_clock::now();
                    ret = qp->init(H_qp, g_corr, A_qp, lb, ub, lbA_corr, ubA_corr, maxIt, &cpuTime_ref);
                    T1 = std::chrono::steady_clock::now();
                    std::cout << "Finished solution of newly initialized correction qp in " << std::chrono::duration_cast<std::chrono::milliseconds>(T1 - T0).count() << "ms\n";
                }

                if (ret == qpOASES::SUCCESSFUL_RETURN){
                    qp->getPrimalSolution(c_vars->deltaXi_cond.array);
                    qp->getDualSolution(c_vars->lambdaQP_cond.array);
                    cond->recover_correction_var_mult(c_vars->deltaXi_cond, c_vars->lambdaQP_cond, corrections, deltaXi, lambdaQP);
                }
                else{
                    std::cout << "Error in correction QP, convexify the hessian further\n";
                    solution_found = false;
                    break;
                }
            }
        }
        if (solution_found) break;
    } // End of QP solving loop

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

    SCQP_correction_iterate *c_vars = dynamic_cast<SCQP_correction_iterate*>(vars);

    //Keep corrections from original QP for SOC QP, add additional correction as needed
    for (int tnum = 0; tnum < cond->num_targets; tnum++){
        SOC_corrections[tnum] = corrections[tnum];
    }

    cond->correction_condense(c_vars->gradObj, c_vars->delta_lb_con, c_vars->delta_ub_con, corrections,
            c_vars->corrected_h, c_vars->corrected_lb_con, c_vars->corrected_ub_con);

    //qpOASES defines +-infinity as +-1e20, so set absent bounds accordingly
    for (int i = 0; i < cond->condensed_num_cons; i++){
        if (std::isinf(c_vars->corrected_lb_con(i)))
            c_vars->corrected_lb_con(i) = -1e20;
        if (std::isinf(c_vars->corrected_ub_con(i)))
            c_vars->corrected_ub_con(i) = 1e20;
    }

    double *g, *lb, *ub, *lbA, *ubA;
    g = c_vars->corrected_h.array;
    lb = c_vars->condensed_lb_var.array;
    ub = c_vars->condensed_ub_var.array;
    lbA = c_vars->corrected_lb_con.array;
    ubA = c_vars->corrected_ub_con.array;


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
    opts.terminationTolerance = param->qpOASES_terminationTolerance;

    qp->setOptions( opts );

    // Other variables for qpOASES
    double cpuTime = param->maxTimeQP;
    int maxIt = param->maxItQP;
    qpOASES::SolutionAnalysis solAna;
    qpOASES::returnValue ret;

    //Solve the QP
    if( param->debugLevel > 2 ) stats->dumpQPCpp( prob, vars, qp, param->sparseQP );

    maxIt = param->maxItQP;
    cpuTime = param->maxTimeQP;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    ret = qp->hotstart( g, lb, ub, lbA, ubA, maxIt, &cpuTime );
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Solved SOC in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms\n";

    if (ret == qpOASES::RET_MAX_NWSR_REACHED){
        std::cout << "Hotstart solution of SOC-QP is taking too long, initialize new QP\n";
        maxIt = param->maxItQP;
        cpuTime = param->maxTimeQP;

        begin = std::chrono::steady_clock::now();
        ret = qp->init(H_qp, g, A_qp, lb, ub, lbA, ubA, maxIt, &cpuTime);
        end = std::chrono::steady_clock::now();
        std::cout << "Solved initialized SOC-QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms\n";
    }

    if( ret == qpOASES::SUCCESSFUL_RETURN ){
        // QP was solved successfully and curvature is positive after removing bounds
        stats->qpIterations = maxIt + 1;
    }

    // Get solution from qpOASES
    qp->getPrimalSolution(c_vars->deltaXi_cond.array);
    qp->getDualSolution(c_vars->lambdaQP_cond.array);

    cond->recover_correction_var_mult(c_vars->deltaXi_cond, c_vars->lambdaQP_cond, corrections, deltaXi, lambdaQP);


    if (ret == qpOASES::SUCCESSFUL_RETURN){

        int ind_1, ind_2, ind, vio_count, max_vio_index;
        double max_dep_bound_violation;
        bool found_direction;
        Matrix xi_s(c_vars->xi);

        double cpuTime_ref;

        for (int k = 0; k < param->max_correction_steps; k++){
            found_direction = true;
            ind_1 = 0;
            vio_count = 0;
            max_dep_bound_violation = 0;
            xi_s = c_vars->xi + deltaXi;

            for (int i = 0; i < cond->num_vblocks; i++){
                if (cond->vblocks[i].dependent){
                    for (int j = 0; j < cond->vblocks[i].size; j++){
                        ind = ind_1 + j;
                        if (xi_s(ind) < prob->lb_var(ind) - param->dep_bound_tolerance || xi_s(ind) > prob->ub_var(ind) + param->dep_bound_tolerance){
                            vio_count++;
                            found_direction = false;

                            //Calculate maximum dep bound violation
                            if (prob->lb_var(ind) - xi_s(ind) > max_dep_bound_violation){
                                max_dep_bound_violation = prob->lb_var(ind) - xi_s(ind);
                                max_vio_index = ind;
                            }
                            else if (xi_s(ind) - prob->ub_var(ind) > max_dep_bound_violation){
                                max_dep_bound_violation = xi_s(ind) - prob->ub_var(ind);
                                max_vio_index = ind;
                            }

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

            for (int tnum = 0; tnum < cond->num_targets; tnum++){

                //Add difference between dependent state values from QP solution and integration for target tnum
                ind_1 = 0;
                ind_2 = cond->vranges[cond->targets[tnum].first_free];

                for (int i = cond->targets[tnum].first_free; i < cond->targets[tnum].vblock_end; i++){
                    if (cond->vblocks[i].dependent){
                        for (int j = 0; j < cond->vblocks[i].size; j++){

                            if (SOC_corrections[tnum](ind_1 + j) > corrections[tnum](ind_1 + j) && xi_s(ind_2 + j) > prob->lb_var(ind_2 + j)){
                                SOC_corrections[tnum](ind_1 + j) -= xi_s(ind_2 + j) - prob->lb_var(ind_2 + j);
                                if (SOC_corrections[tnum](ind_1 + j) < corrections[tnum](ind_1 + j)) SOC_corrections[tnum](ind_1 + j) = corrections[tnum](ind_1 + j);
                            }
                            else if (SOC_corrections[tnum](ind_1 + j) < corrections[tnum](ind_1 + j) && xi_s(ind_2 + j) < prob->ub_var(ind_2 + j)){
                                SOC_corrections[tnum](ind_1 + j) -= xi_s(ind_2 + j) - prob->ub_var(ind_2 + j);
                                if (SOC_corrections[tnum](ind_1 + j) > corrections[tnum](ind_1 + j)) SOC_corrections[tnum](ind_1 + j) = corrections[tnum](ind_1 + j);
                            }

                            if (xi_s(ind_2 + j) < prob->lb_var(ind_2 + j) - param->dep_bound_tolerance){
                                SOC_corrections[tnum](ind_1 + j) += prob->lb_var(ind_2 + j) - xi_s(ind_2 + j);
                            }
                            else if (xi_s(ind_2 + j) > prob->ub_var(ind_2 + j) + param->dep_bound_tolerance){
                                SOC_corrections[tnum](ind_1 + j) += prob->ub_var(ind_2 + j) - xi_s(ind_2 + j);
                            }

                            /*if (xi_s(ind_2 + j) < prob->lb_var(ind_2 + j)){
                                SOC_corrections[tnum](ind_1 + j) += prob->lb_var(ind_2 + j) - xi_s(ind_2 + j);
                            }
                            else if (xi_s(ind_2 + j) > prob->ub_var(ind_2 + j)){
                                SOC_corrections[tnum](ind_1 + j) += prob->ub_var(ind_2 + j) - xi_s(ind_2 + j);
                            }*/

                        }
                        ind_1 += cond->vblocks[i].size;
                    }
                    ind_2 += cond->vblocks[i].size;
                }
            }


            cond->correction_condense(c_vars->gradObj, c_vars->delta_lb_con, c_vars->delta_ub_con, SOC_corrections, c_vars->corrected_h, c_vars->corrected_lb_con, c_vars->corrected_ub_con);

            for (int i = 0; i < cond->condensed_num_cons; i++){
                if (std::isinf(c_vars->corrected_lb_con(i)))
                    c_vars->corrected_lb_con(i) = -1e20;
                if (std::isinf(c_vars->corrected_ub_con(i)))
                    c_vars->corrected_ub_con(i) = 1e20;
            }

            g = c_vars->corrected_h.array;
            lbA = c_vars->corrected_lb_con.array;
            ubA = c_vars->corrected_ub_con.array;

            maxIt = param->maxItQP;
            cpuTime_ref = std::max(cpuTime, 0.25*param->maxTimeQP);


            qp->setOptions(opts);

            std::cout << "Starting solution of correction qp\n";
            std::cout << "Max dep bound violation is " << max_dep_bound_violation << " at index " << max_vio_index << "\n" << std::flush;
            std::chrono::steady_clock::time_point T0 = std::chrono::steady_clock::now();

            ret = qp->hotstart(g, lb, ub, lbA, ubA, maxIt, &cpuTime_ref);

            std::chrono::steady_clock::time_point T1 = std::chrono::steady_clock::now();
            std::cout << "Solved correction QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(T1 - T0).count() << "ms\n";

            if (ret == qpOASES::RET_MAX_NWSR_REACHED){
                std::cout << "Solution of correction SOC QP is taking too long, initializing new correction QP\n";
                maxIt = param->maxItQP;
                cpuTime_ref = param->maxTimeQP;

                T0 = std::chrono::steady_clock::now();
                ret = qp->init(H_qp, g, A_qp, lb, ub, lbA, ubA, maxIt, &cpuTime_ref);
                T1 = std::chrono::steady_clock::now();
                std::cout << "Solved newly initialized correction QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(T1 - T0).count() << "ms\n";
            }

            if (ret == qpOASES::SUCCESSFUL_RETURN){
                qp->getPrimalSolution(c_vars->deltaXi_cond.array);
                qp->getDualSolution(c_vars->lambdaQP_cond.array);
                cond->recover_correction_var_mult(c_vars->deltaXi_cond, c_vars->lambdaQP_cond, SOC_corrections, deltaXi, lambdaQP);
            }
            else{
                break;
            }

        }
    }

    /*
     * Post-processing
     */

    // Compute constrJac*deltaXi, need this for second order correction step
    Atimesb( c_vars->jacNz, c_vars->jacIndRow, c_vars->jacIndCol, deltaXi, c_vars->AdeltaXi );

    // Print qpOASES error code, if any
    if(ret != qpOASES::SUCCESSFUL_RETURN)
        printf( "qpOASES error message: \"%s\"\n",
                qpOASES::getGlobalMessageHandler()->getErrorCodeMessage( ret ) );

    // Point Hessian again to the first Hessian
    c_vars->hess = c_vars->hess1;


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






} // namespace blockSQP


