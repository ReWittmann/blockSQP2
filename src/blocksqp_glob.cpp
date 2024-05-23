/*
 * blockSQP -- Sequential quadratic programming for problems with
 *             block-diagonal Hessian matrix.
 * Copyright (C) 2012-2015 by Dennis Janka <dennis.janka@iwr.uni-heidelberg.de>
 *
 * Licensed under the zlib license. See LICENSE for more details.
 */

/**
 * \file blocksqp_glob.cpp
 * \author Dennis Janka
 * \date 2012-2015
 *
 *  Implementation of methods of SQPmethod class associated with the
 *  globalization strategy.
 *
 */

#include "blocksqp_iterate.hpp"
#include "blocksqp_options.hpp"
#include "blocksqp_stats.hpp"
#include "blocksqp_method.hpp"
#include "blocksqp_restoration.hpp"
#include "blocksqp_general_purpose.hpp"
#include <iostream>
#include <fstream>

namespace blockSQP
{

void SQPmethod::acceptStep( double alpha )
{
    acceptStep( vars->deltaXi, vars->lambdaQP, alpha, 0 );
}

void SQPmethod::acceptStep( const Matrix &deltaXi, const Matrix &lambdaQP, double alpha, int nSOCS )
{
    int k;
    double lStpNorm;

    // Current alpha
    vars->alpha = alpha;
    vars->nSOCS = nSOCS;


    // Set new xi by accepting the current trial step

    for(k=0; k<vars->xi.M(); k++){

        if (alpha * vars->deltaXi(k) < prob->lb_var(k) - vars->xi(k)){
            vars->deltaXi(k) = prob->lb_var(k) - vars->xi(k);
        }
        else if (alpha * vars->deltaXi(k) > prob->ub_var(k) - vars->xi(k)){
            vars->deltaXi(k) = prob->ub_var(k) - vars->xi(k);
        }
        else{
            vars->deltaXi(k) = alpha * deltaXi(k);
        }

        vars->xi(k) = vars->trialXi(k);
    }
    /*
        for( k=0; k<vars->xi.M(); k++ )
        {
            vars->xi( k ) = vars->trialXi( k );
            vars->deltaXi( k ) = alpha * deltaXi( k );
        }
    */


    // Store the infinity norm of the multiplier step
    vars->lambdaStepNorm = 0.0;
    for(k = 0; k < vars->lambda.M(); k++)
        if( (lStpNorm = fabs( alpha*lambdaQP( k ) - alpha*vars->lambda( k ) )) > vars->lambdaStepNorm )
            vars->lambdaStepNorm = lStpNorm;

    // Set new multipliers
    for( k=0; k<vars->lambda.M(); k++ )
        vars->lambda( k ) = (1.0 - alpha)*vars->lambda( k ) + alpha*lambdaQP( k );

    // Count consecutive reduced steps
    if( vars->alpha < 1.0 )
        vars->reducedStepCount++;
    else
        vars->reducedStepCount = 0;
}

void SQPmethod::reduceStepsize( double *alpha )
{
    *alpha = (*alpha) * 0.5;
}


/**
 * Take a full Quasi-Newton step, except when integrator fails:
 * xi = xi + deltaXi
 * lambda = lambdaQP
 */
int SQPmethod::fullstep()
{
    double alpha = 1.0;
    double objTrial, cNormTrial;
    int info;
    int nVar = prob->nVar;

    // Reduce stepsize until dependent variables fit into model bounds
    /*
    int ind_1;
    if (param->condenseQP){
        for ( ; k < param->maxLineSearch; k++){
            // Compute new trial point
            for (int i = 0; i < nVar; i++){
                vars->trialXi( i ) = vars->xi( i ) + alpha * vars->deltaXi( i );
            }
            ind_1 = 0;
            for (int i = 0; i < prob->C->num_vblocks; i++){
                if (prob->C->vblocks[i].dependent){
                    for (int j = ind_1; j < ind_1 + prob->C->vblocks[i].size; j++){
                        if (prob->lb_var(j) - vars->trialXi(j) > 1e-6 || vars->trialXi(j) - prob->ub_var(j) > 1e-6){
                            std::cout << "Step violates dependent variable bounds, reducing stepsize\n";
                            reduceStepsize(&alpha);
                            goto continue_outer_loop;
                        }
                    }
                }
                ind_1 += prob->C->vblocks[i].size;
            }
            break;
            continue_outer_loop:
                ;
        }
    }*/

    // Backtracking line search
    for(int k = 0; k<10; k++){

        // Compute new trial point
        for (int i = 0; i < nVar; i++){
            vars->trialXi(i) = vars->xi(i) + alpha * vars->deltaXi(i);

            if (vars->trialXi(i) < prob->lb_var(i)){
                vars->trialXi(i) = prob->lb_var(i);
            }
            else if (vars->trialXi(i) > prob->ub_var(i)){
                vars->trialXi(i) = prob->ub_var(i);
            }
        }

        /*
            for (int i = 0; i < nVar; i++){
                vars->trialXi(i) = vars->xi(i) + alpha * vars->deltaXi(i);
            }
            */



        // Compute problem functions at trial point
        prob->evaluate( vars->trialXi, &objTrial, vars->constr, &info );
        stats->nFunCalls++;
        cNormTrial = lInfConstraintNorm( vars->trialXi, vars->constr, prob->lb_var, prob->ub_var, prob->lb_con, prob->ub_con );
        // Reduce step if evaluation fails, if lower bound is violated or if objective or a constraint is NaN
        if( info != 0 || objTrial < prob->objLo || objTrial > prob->objUp || !(objTrial == objTrial) || !(cNormTrial == cNormTrial) )
        {
            printf("info=%i, objTrial=%g\n", info, objTrial );
            // evaluation error, reduce stepsize
            reduceStepsize( &alpha );
            continue;
        }
        else
        {
            acceptStep( alpha );
            return 0;
        }
    }// backtracking steps

    return 1;
}


/**
 *
 * Backtracking line search based on a filter
 * as described in Ipopt paper (Waechter 2006)
 *
 */
int SQPmethod::filterLineSearch()
{
    double alpha = 1.0;
    double cNorm, cNormTrial, objTrial, dfTdeltaXi;

    int k, info;
    int nVar = prob->nVar;

    int num_v_bounds;

    bool test = true;

    // Compute ||constr(xi)|| at old point
    //cNorm = l1ConstraintNorm( vars->xi, vars->constr, prob->lb_var, prob->ub_var, prob->lb_con, prob->ub_con );
    cNorm = lInfConstraintNorm( vars->xi, vars->constr, prob->lb_var, prob->ub_var, prob->lb_con, prob->ub_con );

    // Backtracking line search
    for(k = 0; k<param->maxLineSearch; k++){
        // Compute new trial point
        num_v_bounds = 0;

        for (int i = 0; i < nVar; i++){
            vars->trialXi(i) = vars->xi(i) + alpha * vars->deltaXi(i);

            if (vars->trialXi(i) < prob->lb_var(i)){
                if (vars->trialXi(i) - prob->lb_var(i) < -1e-8){
                    //std::cout << "Variable nr " << i << " violated bounds by " << prob->lb_var(i) - vars->trialXi(i) << "\n";
                }

                if (vars->trialXi(i) - prob->lb_var(i) < -1e-8){
                    num_v_bounds++;
                }

                vars->trialXi(i) = prob->lb_var(i);
            }
            else if (vars->trialXi(i) > prob->ub_var(i)){
                if (vars->trialXi(i) - prob->ub_var(i) > 1e-8){
                    //std::cout << "Variable nr " << i << " violated bounds by " << vars->trialXi(i) - prob->ub_var(i) << "\n";
                }

                if (vars->trialXi(i) - prob->ub_var(i) > 1e-8){
                    num_v_bounds++;
                }

                vars->trialXi(i) = prob->ub_var(i);
            }
        }
        //std::cout << "Linesearch iteration " << k << ": " << num_v_bounds << " dependent variable bounds are violated\n";

        //OLD: No bound checking
        /*
        for (int i = 0; i < nVar; i++)
            vars->trialXi(i) = vars->xi(i) + alpha * vars->deltaXi(i);*/


        // Compute grad(f)^T * deltaXi
        dfTdeltaXi = 0.0;
        for(int i = 0; i < nVar; i++)
            dfTdeltaXi += vars->gradObj( i ) * vars->deltaXi( i );

        // Compute objective and at ||constr(trialXi)||_1 at trial point
        prob->evaluate( vars->trialXi, &objTrial, vars->constr, &info );
        stats->nFunCalls++;
        //cNormTrial = l1ConstraintNorm( vars->trialXi, vars->constr, prob->lb_var, prob->ub_var, prob->lb_con, prob->ub_con );
        cNormTrial = lInfConstraintNorm( vars->trialXi, vars->constr, prob->lb_var, prob->ub_var, prob->lb_con, prob->ub_con );

        // Reduce step if evaluation fails, if lower bound is violated or if objective is NaN
        if( info != 0 || objTrial < prob->objLo || objTrial > prob->objUp || !(objTrial == objTrial) || !(cNormTrial == cNormTrial) )
        {
            // evaluation error, reduce stepsize
            reduceStepsize( &alpha );
            continue;
        }

        // Check acceptability to the filter
        if (pairInFilter( cNormTrial, objTrial ))// || (stats->itCount == 4 && test))
        {
            //Try solving again with convex hessian approximation before invoking SOC
            if (k == 0 && !vars->conv_qp_solved){
                if (solveQP(vars->deltaXi, vars->lambdaQP, true) == 0){
                    //Restart filter line search with step from positive definite QP
                    k = -1;
                    test = false;
                    continue;
                }
                //If solution failed, continue with step from nonconvex QP
            }

            // Trial point is in the prohibited region defined by the filter, try second order correction
            if( secondOrderCorrection( cNorm, cNormTrial, dfTdeltaXi, false, k ) )
                break; // SOC yielded suitable alpha, stop
            else
            {
                reduceStepsize( &alpha );
                continue;
            }
        }

        // Check sufficient decrease, case I:
        // If we are (almost) feasible and a "switching condition" is satisfied
        // require sufficient progress in the objective instead of bi-objective condition
        if( cNorm <= param->thetaMin )
        {
            // Switching condition, part 1: grad(f)^T * deltaXi < 0 ?
            if( dfTdeltaXi < 0 )
                // Switching condition, part 2: alpha * ( - grad(f)^T * deltaXi )**sF > delta * cNorm**sTheta ?
                if( alpha * pow( (-dfTdeltaXi), param->sF ) > param->delta * pow( cNorm, param->sTheta ) )
                {
                    // Switching conditions hold: Require satisfaction of Armijo condition for objective
                    if( objTrial > vars->obj + param->eta*alpha*dfTdeltaXi )
                    {
                        // Armijo condition violated, try second order correction
                        if( secondOrderCorrection( cNorm, cNormTrial, dfTdeltaXi, true, k ) )
                            break; // SOC yielded suitable alpha, stop
                        else
                        {
                            reduceStepsize( &alpha );
                            continue;
                        }
                    }
                    else
                    {
                        // found suitable alpha, stop
                        acceptStep( alpha );
                        break;
                    }
                }
        }

        // Check sufficient decrease, case II:
        // Bi-objective (filter) condition
        if( cNormTrial < (1.0 - param->gammaTheta) * cNorm || objTrial < vars->obj - param->gammaF * cNorm )
        {
            // found suitable alpha, stop
            acceptStep( alpha );
            break;
        }
        else
        {
            // Trial point is dominated by current point, try second order correction
            if( secondOrderCorrection( cNorm, cNormTrial, dfTdeltaXi, false, k ) )
                break; // SOC yielded suitable alpha, stop
            else
            {
                reduceStepsize( &alpha );
                continue;
            }
        }
    }// backtracking steps

    // No step could be found by the line search
    if( k == param->maxLineSearch )
        return 1;

    // Augment the filter if switching condition or Armijo condition does not hold
    if( dfTdeltaXi >= 0 )
        augmentFilter( cNormTrial, objTrial );
    else if( alpha * pow( (-dfTdeltaXi), param->sF ) > param->delta * pow( cNorm, param->sTheta ) )// careful with neg. exponents!
        augmentFilter( cNormTrial, objTrial );
    else if( objTrial <= vars->obj + param->eta*alpha*dfTdeltaXi )
        augmentFilter( cNormTrial, objTrial );

    return 0;
}


/**
 *
 * Perform a second order correction step, i.e. solve the QP:
 *
 * min_d d^TBd + d^TgradObj
 * s.t.  bl <= A^Td + constr(xi+alpha*deltaXi) - A^TdeltaXi <= bu
 *
 */
bool SQPmethod::secondOrderCorrection(double cNorm, double cNormTrial, double dfTdeltaXi, bool swCond, int it){

    // Perform SOC only on the first iteration of backtracking line search
    if( it > 0 )
        return false;
    // If constraint violation of the trialstep is lower than the current one skip SOC
    if(cNormTrial < cNorm)// && stats->itCount != 3)
    {
        std::cout << "Constraint violation is lower than current one, skip SOC\n";
        return false;
    }

    int nSOCS = 0;
    double cNormTrialSOC, cNormOld, objTrialSOC;
    int k, info;
    int nVar = prob->nVar;
    Matrix deltaXiSOC, lambdaQPSOC;

    // vars->constr contains result at first trial point: c(xi+deltaXi)
    // vars->constrJac, vars->AdeltaXi and vars->gradObj are unchanged so far.

    // First SOC step
    deltaXiSOC.Dimension( vars->deltaXi.M() ).Initialize( 0.0 );
    lambdaQPSOC.Dimension( vars->lambdaQP.M() ).Initialize( 0.0 );

    // Second order correction loop
    cNormOld = cNorm;
    for( k=0; k<param->maxSOCiter; k++ )
    {
        nSOCS++;

        // Update bounds for SOC QP
        updateStepBounds( 1 );

        // Solve SOC QP to obtain new, corrected deltaXi
        // (store in separate vector to avoid conflict with original deltaXi -> need it in linesearch!)
        info = solve_SOC_QP(deltaXiSOC, lambdaQPSOC);

        if( info != 0 )
            return false; // Could not solve QP, abort SOC

        // Set new SOC trial point
        for (int i = 0; i < nVar; i++){
            vars->trialXi(i) = vars->xi(i) + deltaXiSOC(i);

            if (vars->trialXi(i) < prob->lb_var(i)){
                vars->trialXi(i) = prob->lb_var(i);
            }
            else if (vars->trialXi(i) > prob->ub_var(i)){
                vars->trialXi(i) = prob->ub_var(i);
            }
        }

        // Compute objective and ||constr(trialXiSOC)||_1 at SOC trial point
        prob->evaluate( vars->trialXi, &objTrialSOC, vars->constr, &info );
        stats->nFunCalls++;
        cNormTrialSOC = lInfConstraintNorm( vars->trialXi, vars->constr, prob->lb_var, prob->ub_var, prob->lb_con, prob->ub_con );
        if( info != 0 || objTrialSOC < prob->objLo || objTrialSOC > prob->objUp || !(objTrialSOC == objTrialSOC) || !(cNormTrialSOC == cNormTrialSOC) )
            return false; // evaluation error, abort SOC

        // Check acceptability to the filter (in SOC)
        if( pairInFilter( cNormTrialSOC, objTrialSOC ) ){
            std::cout << "Trial point is in the filter\n";
            return false; // Trial point is in the prohibited region defined by the filter, abort SOC
            }

        // Check sufficient decrease, case I (in SOC)
        // (Almost feasible and switching condition holds for line search alpha)
        if( cNorm <= param->thetaMin && swCond )
        {
            if( objTrialSOC > vars->obj + param->eta*dfTdeltaXi )
            {
                // Armijo condition does not hold for SOC step, next SOC step

                // If constraint violation gets worse by SOC, abort
                if( cNormTrialSOC > param->kappaSOC * cNormOld ){
                    std::cout << "Constraint violation got worse by SOC, abort\n";
                    return false;
                }
                else
                    cNormOld = cNormTrialSOC;
                continue;
            }
            else
            {
                // found suitable alpha during SOC, stop
                acceptStep( deltaXiSOC, lambdaQPSOC, 1.0, nSOCS );
                return true;
            }
        }

        // Check sufficient decrease, case II (in SOC)
        if( cNorm > param->thetaMin || !swCond )
        {
            if( cNormTrialSOC < (1.0 - param->gammaTheta) * cNorm || objTrialSOC < vars->obj - param->gammaF * cNorm )
            {
                // found suitable alpha during SOC, stop
                acceptStep( deltaXiSOC, lambdaQPSOC, 1.0, nSOCS );
                return true;
            }
            else
            {
                // Trial point is dominated by current point, next SOC step

                // If constraint violation gets worse by SOC, abort
                if( cNormTrialSOC > param->kappaSOC * cNormOld )
                    return false;
                else
                    cNormOld = cNormTrialSOC;
                continue;
            }
        }
    }

    return false;
}

/**
 * Minimize constraint violation by solving an NLP with minimum norm objective
 *
 * "The dreaded restoration phase" -- Nick Gould
 */
int SQPmethod::feasibilityRestorationPhase()
{
    // No Feasibility restoration phase
    if( param->restoreFeas == 0 )
        return -1;

    stats->nRestPhaseCalls++;

    int ret, info;
    int maxRestIt = 100;
    int warmStart;
    double cNormTrial, objTrial, lStpNorm, stepsize_sum = 0.;

    // Iterate until a point acceptable to the filter is found
    if (vars->steptype != 3){
        warmStart = 0;
        delete rest_prob;
        delete rest_method;
        delete rest_stats;
        rest_prob = new RestorationProblem(prob, vars->xi);
        rest_stats = new SQPstats(stats->outpath);
        rest_method = new SQPmethod( rest_prob, rest_opts, rest_stats );

        rest_method->init();
    }
    else{
        warmStart = 1;
    }

    for(int it=0; it<maxRestIt; it++){
        // One iteration for minimum norm NLP
        ret = rest_method->run( 1, warmStart );
        warmStart = 1;

        // If restMethod yields error, stop restoration phase
        if( ret == -1 )
            break;

        //Require sufficient progress
        stepsize_sum += rest_method->vars->alpha;
        if (stepsize_sum < 1.0){
            continue;
        }

        // Get new xi from the restoration phase
        for(int i = 0; i < prob->nVar; i++)
            vars->trialXi( i ) = rest_method->vars->xi( i );

        // Compute objective at trial point
        prob->evaluate( vars->trialXi, &objTrial, vars->constr, &info );
        stats->nFunCalls++;
        cNormTrial = lInfConstraintNorm( vars->trialXi, vars->constr, prob->lb_var, prob->ub_var, prob->lb_con, prob->ub_con );
        if( info != 0 || objTrial < prob->objLo || objTrial > prob->objUp || !(objTrial == objTrial) || !(cNormTrial == cNormTrial) )
            continue;

        // Is this iterate acceptable for the filter?
        if( !pairInFilter( cNormTrial, objTrial ) )
        {
            // success
            printf("Found a point acceptable for the filter.\n");
            ret = 0;
            break;
        }

        // If minimum norm NLP has converged, declare local infeasibility
        if( rest_method->vars->tol < param->opttol && rest_method->vars->cNormS < param->nlinfeastol )
        {
            ret = 1;
            break;
        }
    }

    // Success or locally infeasible

    if (ret == 0 || ret == 1){
        for (int i = 0; i < prob->nVar; i++){
            //TODO check sign
            vars->deltaXi(i) = -vars->xi(i);
            vars->xi(i) = vars->trialXi(i);
            vars->deltaXi(i) += vars->xi(i);
        }

        // Store the infinity norm of the multiplier step
        vars->lambdaStepNorm = 0.0;
        for (int i = 0; i < prob->nVar; i++){
            if((lStpNorm = fabs(rest_method->vars->lambda(i) - vars->lambda(i))) > vars->lambdaStepNorm){
                vars->lambdaStepNorm = lStpNorm;
            }
            vars->lambda(i) = rest_method->vars->lambda(i);
            vars->lambdaQP(i) = rest_method->vars->lambdaQP(i);
        }
        //Skip dual variables for the slack variables
        for (int i = prob->nVar; i < prob->nVar + prob->nCon; i++){
            if((lStpNorm = fabs(rest_method->vars->lambda(prob->nCon + i) - vars->lambda(i))) > vars->lambdaStepNorm){
                vars->lambdaStepNorm = lStpNorm;
            }
            vars->lambda(i) = rest_method->vars->lambda(prob->nCon + i);
            vars->lambdaQP(i) = rest_method->vars->lambdaQP(prob->nCon + i);
        }

        vars->alpha = 1.0;
        vars->nSOCS = 0;

        // reset reduced step counter
        vars->reducedStepCount = 0;

        // reset Hessian and limited memory information
        resetHessian();

        // dont use homotopy for next QP since it may differ greatly from previous QP
        //vars->use_homotopy = false;
        //set by resetHessian
    }

    if( ret == 1 )
    {
        stats->printProgress( prob, vars, param, 0 );
        printf("The problem seems to be locally infeasible. Infeasibilities minimized.\n");
    }

    return ret;
}


/**
 * Try to (partly) improve constraint violation by satisfying
 * the (pseudo) continuity constraints, i.e. do a single shooting
 * iteration with the current controls and measurement weights q and w
 */
int SQPmethod::feasibilityRestorationHeuristic()
{
    stats->nRestHeurCalls++;

    int info, k;
    Matrix trial_constr;
    double obj_trial, cNormTrial;

    info = 0;

    // Call problem specific heuristic to reduce constraint violation.
    // For shooting methods that means setting consistent values for shooting nodes by one forward integration.
    for( k=0; k<prob->nVar; k++ ) // input: last successful step
        vars->trialXi( k ) = vars->xi( k );
    prob->reduceConstrVio( vars->trialXi, &info );
    if( info )// If an error occured in restoration heuristics, abort
        return -1;

    // Compute objective and constraints at the new (hopefully feasible) point
    trial_constr.Dimension( prob->nCon ).Initialize(0.0);
    //FIX for weird bug
    //prob->evaluate( vars->trialXi, &vars->obj, vars->constr, &info );
    prob->evaluate(vars->trialXi, &obj_trial, trial_constr, &info);

    stats->nFunCalls++;
    cNormTrial = lInfConstraintNorm( vars->trialXi, trial_constr, prob->lb_var, prob->ub_var, prob->lb_con, prob->ub_con );
    if( info != 0 || obj_trial < prob->objLo || obj_trial > prob->objUp || !(obj_trial == obj_trial) || !(cNormTrial == cNormTrial) )
        return -1;

    // Is the new point acceptable for the filter?
    if( pairInFilter( cNormTrial, obj_trial ) )
    {
        std::cout << "New point is in the filter\n";
        // point is in the taboo region, restoration heuristic not successful!
        return -1;
    }

    // If no error occured in the integration all shooting variables now
    // have the values obtained by a single shooting integration.
    // This is done instead of a Newton-like step in the current SQP iteration

    vars->alpha = 1.0;
    vars->nSOCS = 0;

    // reset reduced step counter
    vars->reducedStepCount = 0;

    // Reset lambda
    vars->lambda.Initialize( 0.0 );
    vars->lambdaQP.Initialize( 0.0 );

    // Compute the "step" taken by closing the continuity conditions
    /// \note deltaXi is reset by resetHessian(), so this doesn't matter
    for( k=0; k<prob->nVar; k++ )
    {
        //vars->deltaXi( k ) = vars->trialXi( k ) - vars->xi( k );
        vars->xi( k ) = vars->trialXi( k );
    }

    // reduce Hessian and limited memory information
    resetHessian();


    return 0;
}


/**
 * If the line search fails, check if the full step reduces the KKT error by a factor kappaF.
 */
int SQPmethod::kktErrorReduction( )
{
    int i, info = 0;
    double objTrial, cNormTrial, trialGradNorm, trialTol;
    Matrix trialConstr, trialGradLagrange;

    // Compute new trial point
    /*
    for( i=0; i<prob->nVar; i++ )
        vars->trialXi( i ) = vars->xi( i ) + vars->deltaXi( i );*/

    for (int i = 0; i < prob->nVar; i++){
        vars->trialXi(i) = vars->xi(i) + vars->deltaXi(i);

        if (vars->trialXi(i) < prob->lb_var(i)){
            vars->trialXi(i) = prob->lb_var(i);
        }
        else if (vars->trialXi(i) > prob->ub_var(i)){
            vars->trialXi(i) = prob->ub_var(i);
        }
    }

    // Compute objective and ||constr(trialXi)|| at trial point
    trialConstr.Dimension( prob->nCon ).Initialize( 0.0 );
    prob->evaluate( vars->trialXi, &objTrial, trialConstr, &info );
    stats->nFunCalls++;
    cNormTrial = lInfConstraintNorm( vars->trialXi, trialConstr, prob->lb_var, prob->ub_var, prob->lb_con, prob->ub_con );
    if( info != 0 || objTrial < prob->objLo || objTrial > prob->objUp || !(objTrial == objTrial) || !(cNormTrial == cNormTrial) )
    {
        // evaluation error
        return 1;
    }

    // Compute KKT error of the new point

    // scaled norm of Lagrangian gradient
    trialGradLagrange.Dimension( prob->nVar ).Initialize( 0.0 );
    if( param->sparseQP )
        calcLagrangeGradient( vars->lambdaQP, vars->gradObj, vars->jacNz,
                              vars->jacIndRow, vars->jacIndCol, trialGradLagrange, 0 );
    else
        calcLagrangeGradient( vars->lambdaQP, vars->gradObj, vars->constrJac,
                              trialGradLagrange, 0 );

    trialGradNorm = lInfVectorNorm( trialGradLagrange );
    trialTol = trialGradNorm /( 1.0 + lInfVectorNorm( vars->lambdaQP ) );

    if( fmax( cNormTrial, trialTol ) < param->kappaF * fmax( vars->cNorm, vars->tol ) )
    {
        acceptStep( 1.0 );
        return 0;
    }
    else
        return 1;
}

/**
 * Check if current entry is accepted to the filter:
 * (cNorm, obj) in F_k
 */
bool SQPmethod::pairInFilter( double cNorm, double obj )
{
    std::set< std::pair<double,double> >::iterator iter;
    std::set< std::pair<double,double> > *filter;
    filter = vars->filter;

    /*
     * A pair is in the filter if:
     * - it increases the objective and
     * - it also increases the constraint violation
     * The second expression in the if-clause states that we exclude
     * entries that are within the feasibility tolerance, e.g.
     * if an entry improves the constraint violation from 1e-16 to 1e-17,
     * but increases the objective considerably we also think of this entry
     * as dominated
     */

    for( iter=filter->begin(); iter!=filter->end(); iter++ )
        if( (cNorm >= (1.0 - param->gammaTheta) * iter->first ||
            (cNorm < 0.01 * param->nlinfeastol && iter->first < 0.01 * param->nlinfeastol ) ) &&
            obj >= iter->second - param->gammaF * iter->first )
        {
            return 1;
        }

    return 0;
}


void SQPmethod::initializeFilter()
{
    std::set< std::pair<double,double> >::iterator iter;
    std::pair<double,double> initPair ( param->thetaMax, prob->objLo );

    // Remove all elements
    iter=vars->filter->begin();
    while (iter != vars->filter->end())
    {
        std::set< std::pair<double,double> >::iterator iterToRemove = iter;
        iter++;
        vars->filter->erase( iterToRemove );
    }

    // Initialize with pair ( maxConstrViolation, objLowerBound );
    vars->filter->insert( initPair );
}


/**
 * Augment the filter:
 * F_k+1 = F_k U { (c,f) | c > (1-gammaTheta)cNorm and f > obj-gammaF*c
 */
void SQPmethod::augmentFilter( double cNorm, double obj )
{
    std::set< std::pair<double,double> >::iterator iter;
    std::pair<double,double> entry ( (1.0 - param->gammaTheta)*cNorm, obj - param->gammaF*cNorm );

    // Augment filter by current element
    vars->filter->insert( entry );

    // Remove dominated elements
    iter=vars->filter->begin();
    while (iter != vars->filter->end())
    {
        //printf(" iter->first=%g, entry.first=%g, iter->second=%g, entry.second=%g\n",iter->first, entry.first, iter->second, entry.second);
        if( iter->first > entry.first && iter->second > entry.second )
        {
            std::set< std::pair<double,double> >::iterator iterToRemove = iter;
            iter++;
            vars->filter->erase( iterToRemove );
        }
        else
            iter++;
    }

}


/////////////////////////////////////////////////////


int SCQPmethod::feasibilityRestorationPhase()
{
    // No Feasibility restoration phase
    if( param->restoreFeas == 0 )
        return -1;


    stats->nRestPhaseCalls++;

    int ret, info;
    int maxRestIt = 100;
    int warmStart;
    double cNormTrial, objTrial, lStpNorm, stepsize_sum = 0.;

    if (vars->steptype != 3){
        warmStart = 0;
        delete rest_prob;
        delete rest_method;
        delete rest_stats;

        rest_prob = new TC_restoration_Problem(prob, cond, vars->xi);
        rest_stats = new SQPstats(stats->outpath);
        rest_method = new SCQPmethod(rest_prob, rest_opts, rest_stats, rest_cond);

        rest_method->init();
    }
    else{
        warmStart = 1;
    }

    for(int it=0; it<maxRestIt; it++){

        // One iteration for minimum norm NLP
        ret = rest_method->run( 1, warmStart );
        warmStart = 1;

        // If restMethod yields error, stop restoration phase
        if( ret == -1 )
            break;

        stepsize_sum += rest_method->vars->alpha;
        if (stepsize_sum < 1.0)
            continue;

        // Get new xi from the restoration phase
        for(int i = 0; i < prob->nVar; i++)
            vars->trialXi( i ) = rest_method->vars->xi( i );

        // Compute objective at trial point
        prob->evaluate( vars->trialXi, &objTrial, vars->constr, &info );
        stats->nFunCalls++;
        cNormTrial = lInfConstraintNorm( vars->trialXi, vars->constr, prob->lb_var, prob->ub_var, prob->lb_con, prob->ub_con );
        if( info != 0 || objTrial < prob->objLo || objTrial > prob->objUp || !(objTrial == objTrial) || !(cNormTrial == cNormTrial) )
            continue;

        // Is this iterate acceptable for the filter?
        if( !pairInFilter( cNormTrial, objTrial ) )
        {
            // success
            printf("Found a point acceptable for the filter.\n");
            ret = 0;
            break;
        }

        // If minimum norm NLP has converged, declare local infeasibility
        if( rest_method->vars->tol < param->opttol && rest_method->vars->cNormS < param->nlinfeastol )
        {
            ret = 1;
            break;
        }
    }

    // Success or locally infeasible
    if (ret == 0 || ret == 1){
        for (int i = 0; i < prob->nVar; i++){
            //TODO check sign
            vars->deltaXi(i) = -vars->xi(i);
            vars->xi(i) = vars->trialXi(i);
            vars->deltaXi(i) += vars->xi(i);
        }

        // Store the infinity norm of the multiplier step
        vars->lambdaStepNorm = 0.0;

        dynamic_cast<TC_restoration_Problem*>(rest_prob)->recover_multipliers(rest_method->vars->lambda, vars->lambda, vars->lambdaStepNorm);
        dynamic_cast<TC_restoration_Problem*>(rest_prob)->recover_multipliers(rest_method->vars->lambdaQP, vars->lambdaQP);

        vars->alpha = 1.0;
        vars->nSOCS = 0;

        // reset reduced step counter
        vars->reducedStepCount = 0;

        // reset Hessian and limited memory information
        resetHessian();
    }

    if( ret == 1 )
    {
        stats->printProgress( prob, vars, param, 0 );
        printf("The problem seems to be locally infeasible. Infeasibilities minimized.\n");
    }

    return ret;
}




int SCQP_correction_method::feasibilityRestorationPhase(){
    // No Feasibility restoration phase
    if( param->restoreFeas == 0 )
        return -1;


    stats->nRestPhaseCalls++;

    int ret, info;
    int maxRestIt = 100;
    int warmStart;
    double cNormTrial, objTrial, lStpNorm, stepsize_sum;

    // Iterate until a point acceptable to the filter is found
    if (vars->steptype != 3){
        warmStart = 0;
        delete rest_prob;
        delete rest_method;
        delete rest_stats;

        rest_prob = new TC_restoration_Problem(prob, cond, vars->xi);
        rest_stats = new SQPstats(stats->outpath);
        rest_method = new SCQP_correction_method(rest_prob, rest_opts, rest_stats, rest_cond);

        rest_method->init();
    }
    else{
        warmStart = 1;
    }

    for(int it=0; it<maxRestIt; it++){

        // One iteration for minimum norm NLP
        ret = rest_method->run( 1, warmStart );
        warmStart = 1;

        // If restMethod yields error, stop restoration phase
        if( ret == -1 )
            break;

        stepsize_sum += rest_method->vars->alpha;
        if (stepsize_sum < 1.0){
            continue;
        }

        // Get new xi from the restoration phase
        for(int i = 0; i < prob->nVar; i++)
            vars->trialXi( i ) = rest_method->vars->xi( i );

        // Compute objective at trial point
        prob->evaluate( vars->trialXi, &objTrial, vars->constr, &info );
        stats->nFunCalls++;
        cNormTrial = lInfConstraintNorm( vars->trialXi, vars->constr, prob->lb_var, prob->ub_var, prob->lb_con, prob->ub_con );
        if( info != 0 || objTrial < prob->objLo || objTrial > prob->objUp || !(objTrial == objTrial) || !(cNormTrial == cNormTrial) )
            continue;

        // Is this iterate acceptable for the filter?
        if( !pairInFilter( cNormTrial, objTrial ) )
        {
            // success
            printf("Found a point acceptable for the filter.\n");
            ret = 0;
            break;
        }

        // If minimum norm NLP has converged, declare local infeasibility
        if( rest_method->vars->tol < param->opttol && rest_method->vars->cNormS < param->nlinfeastol )
        {
            ret = 1;
            break;
        }
    }

    // Success or locally infeasible

    if (ret == 0 || ret == 1){
        for (int i = 0; i < prob->nVar; i++){
            //TODO check sign
            vars->deltaXi(i) = -vars->xi(i);
            vars->xi(i) = vars->trialXi(i);
            vars->deltaXi(i) += vars->xi(i);
        }

        // Store the infinity norm of the multiplier step
        vars->lambdaStepNorm = 0.0;

        dynamic_cast<TC_restoration_Problem*>(rest_prob)->recover_multipliers(rest_method->vars->lambda, vars->lambda, vars->lambdaStepNorm);
        dynamic_cast<TC_restoration_Problem*>(rest_prob)->recover_multipliers(rest_method->vars->lambdaQP, vars->lambdaQP);

        vars->alpha = 1.0;
        vars->nSOCS = 0;

        // reset reduced step counter
        vars->reducedStepCount = 0;

        // reset Hessian and limited memory information
        resetHessian();

        // dont use homotopy for next QP since it may differ greatly from previous QP
        //vars->use_homotopy = false;
    }

    if( ret == 1 )
    {
        stats->printProgress( prob, vars, param, 0 );
        printf("The problem seems to be locally infeasible. Infeasibilities minimized.\n");
    }

    return ret;
}




} // namespace blockSQP
