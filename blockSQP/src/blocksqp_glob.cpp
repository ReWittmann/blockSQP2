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
 * \file blocksqp_glob.cpp
 * \author Dennis Janka
 * \date 2012-2015
 *
 *  Implementation of methods of SQPmethod class associated with the
 *  enable_linesearch strategy.
 * 
 * \modifications
 *  \author Reinhold Wittmann
 *  \date 2023-2025
 */


#include "blocksqp_iterate.hpp"
#include "blocksqp_options.hpp"
#include "blocksqp_stats.hpp"
#include "blocksqp_method.hpp"
#include "blocksqp_restoration.hpp"
#include "blocksqp_general_purpose.hpp"
#include <iostream>
#include <fstream>

namespace blockSQP{

void SQPmethod::acceptStep(double alpha){
    acceptStep(vars->deltaXi, vars->lambdaQP, alpha, 0);
}

void SQPmethod::acceptStep(const Matrix &deltaXi, const Matrix &lambdaQP, double alpha, int nSOCS){
    int k;
    double lStpNorm;
    
    // Current alpha
    vars->alpha = alpha;
    vars->nSOCS = nSOCS;
    
    // Set new xi by accepting the current trial step
    for(k=0; k<vars->xi.M(); k++){
        
        //TrialXi was already set in bounds, set the step in bounds as well
        // if (alpha * vars->deltaXi(k) < prob->lb_var(k) - vars->xi(k)){
        if (alpha * deltaXi(k) < prob->lb_var(k) - vars->xi(k)){
            vars->deltaXi(k) = prob->lb_var(k) - vars->xi(k);
        }
        // else if (alpha * vars->deltaXi(k) > prob->ub_var(k) - vars->xi(k)){
        else if (alpha * deltaXi(k) > prob->ub_var(k) - vars->xi(k)){
            vars->deltaXi(k) = prob->ub_var(k) - vars->xi(k);
        }
        else{
            vars->deltaXi(k) = alpha * deltaXi(k);
        }
        
        //Trial iterate becomes new iterate
        vars->xi(k) = vars->trialXi(k);
    }
    
    // Store the infinity norm of the multiplier step
    vars->lambdaStepNorm = 0.0;
    for(k = 0; k < vars->lambda.M(); k++)
        if( (lStpNorm = fabs( alpha*lambdaQP( k ) - alpha*vars->lambda( k ) )) > vars->lambdaStepNorm )
            vars->lambdaStepNorm = lStpNorm;
    
    // Set new multipliers
    for (k = 0; k < vars->lambda.M(); k++)
        vars->lambda( k ) = (1.0 - alpha)*vars->lambda( k ) + alpha*lambdaQP( k );
    
    // Count consecutive reduced steps
    /*
    if( vars->alpha < 1.0 )
        vars->reducedStepCount++;
    else
        vars->reducedStepCount = 0;
    */
    if (vars->alpha < 0.25)
        vars->reducedStepCount++;
    else if (vars->alpha > 0.99)
        vars->reducedStepCount = 0;
}

void SQPmethod::reduceStepsize(double *alpha){
    *alpha = (*alpha) * 0.5;
}

void SQPmethod::force_accept(double alpha){
    force_accept(vars->deltaXi, vars->lambdaQP, alpha, 0);
}

//Force a step to be accepted, ignoring the filter and remove dominating entries
void SQPmethod::force_accept(const Matrix &deltaXi, const Matrix &lambdaQP, double alpha, int nSOCS){
    int infoEval;
    
    acceptStep(deltaXi, lambdaQP, alpha, nSOCS);
    
    prob->evaluate(vars->xi, &vars->obj, vars->constr, &infoEval);
    vars->cNorm = lInfConstraintNorm(vars->xi, vars->constr, prob->lb_var, prob->ub_var, prob->lb_con, prob->ub_con);
    
    //Remove filter entries that dominate the set point
    std::set<std::pair<double,double>>::iterator iter;
    std::set<std::pair<double,double>>::iterator iterToRemove;
    iter = vars->filter.begin();
    
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



/**
 * Take a full Quasi-Newton step, except when integrator fails:
 * xi = xi + deltaXi
 * lambda = lambdaQP
 */
int SQPmethod::fullstep(){
    double alpha = 1.0;
    double objTrial, cNormTrial;
    int info;
    int nVar = prob->nVar;
    
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
bool SQPmethod::filterLineSearch(){
    double alpha = 1.0;
    double cNorm, cNormTrial(0), objTrial, dfTdeltaXi(0);   //cNormTrial and dfTdeltaXi are initialized to prevent compiler warnings
    // bool armijo_accepted = false;
    int k, info;
    int nVar = prob->nVar;
    
    // Compute ||constr(xi)|| at old point
    cNorm = lInfConstraintNorm(vars->xi, vars->constr, prob->lb_var, prob->ub_var, prob->lb_con, prob->ub_con);
    
    // Backtracking line search
    for (k = 0; k<param->max_linesearch_steps; k++){        
        // Compute new trial point and set it in bounds
        for (int i = 0; i < nVar; i++){
            vars->trialXi(i) = vars->xi(i) + alpha * vars->deltaXi(i);

            if (vars->trialXi(i) < prob->lb_var(i)){
                vars->trialXi(i) = prob->lb_var(i);
            }
            else if (vars->trialXi(i) > prob->ub_var(i)){
                vars->trialXi(i) = prob->ub_var(i);
            }
        }
        
        // Compute grad(f)^T * deltaXi
        dfTdeltaXi = 0.0;
        for (int i = 0; i < nVar; i++)
            dfTdeltaXi += vars->gradObj(i) * vars->deltaXi(i);
        
        // Compute objective and at ||constr(trialXi)||_1 at trial point
        prob->evaluate(vars->trialXi, &objTrial, vars->trialConstr, &info);
        stats->nFunCalls++;
        
        //cNormTrial = l1ConstraintNorm( vars->trialXi, vars->constr, prob->lb_var, prob->ub_var, prob->lb_con, prob->ub_con );
        cNormTrial = lInfConstraintNorm(vars->trialXi, vars->trialConstr, prob->lb_var, prob->ub_var, prob->lb_con, prob->ub_con);
        
        // Reduce step if evaluation fails, if lower bound is violated or if objective is NaN
        if (info != 0 || objTrial < prob->objLo || objTrial > prob->objUp || !(objTrial == objTrial) || !(cNormTrial == cNormTrial)){
            // evaluation error, reduce stepsize
            reduceStepsize(&alpha);
            continue;
        }
        
        // Check acceptability to the filter
        if (pairInFilter(cNormTrial, objTrial)){
            // Trial point is in the prohibited region defined by the filter, try second order correction
            if (k == 0 && secondOrderCorrection(cNorm, cNormTrial, dfTdeltaXi, true))
                break;
            else{
                reduceStepsize(&alpha);
                continue;
            }
        }
        
        // Check sufficient decrease, case I:
        // If we are (almost) feasible and a "switching condition" is satisfied
        // require sufficient progress in the objective instead of bi-objective condition
        if (cNorm <= param->thetaMin){
            // Switching condition, part 1: grad(f)^T * deltaXi < 0 ?
            if (dfTdeltaXi < 0){
                // Switching condition, part 2: alpha * ( - grad(f)^T * deltaXi )**sF > delta * cNorm**sTheta ?
                if (alpha*pow((-dfTdeltaXi), param->sF) > param->delta*pow(cNorm, param->sTheta)){
                    // Switching conditions holds: Require satisfaction of Armijo condition for objective
                    if (objTrial <= vars->obj + param->eta*alpha*dfTdeltaXi){
                        // found suitable alpha, stop
                        acceptStep( alpha );
                        // armijo_accepted = true;
                        break;
                    }
                    else{
                        //Armijo condition violated, try second order correction
                        if (k == 0 && secondOrderCorrection(cNorm, cNormTrial, dfTdeltaXi, true))
                            break;
                        else{
                            reduceStepsize(&alpha);
                            continue;
                        }
                    }
                }
            }
        }
        
        // Check sufficient decrease, case II:
        // Bi-objective (filter) condition
        if (cNormTrial < (1.0 - param->gammaTheta)*cNorm || objTrial < vars->obj - param->gammaF*cNorm){
            // found suitable alpha, stop
            acceptStep(alpha);
            break;
        }
        else{
            // Trial point is dominated by current point, try second order correction
            if (k == 0 && secondOrderCorrection(cNorm, cNormTrial, dfTdeltaXi, false)) break; // SOC yielded suitable alpha, stop
            else{
                reduceStepsize(&alpha);
                continue;
            }
        }
    }// backtracking steps
    
    // No step could be found by the line search
    if (k == param->max_linesearch_steps) return true;
    
    // Augment the filter if switching condition or Armijo condition does not hold
    if (dfTdeltaXi >= 0){
        augmentFilter(cNorm, vars->obj);
    }
    else if (alpha * pow((-dfTdeltaXi), param->sF) <= param->delta*pow(cNorm, param->sTheta)){// careful with neg. exponents!
        augmentFilter(cNorm, vars->obj);
    }
    else if (objTrial > vars->obj + param->eta*alpha*dfTdeltaXi){
        augmentFilter(cNorm, vars->obj);
    }
    
    return false;
}


/**
 *
 * Perform a second order correction step, i.e. solve the QP:
 *
 * min_d d^TBd + d^TgradObj
 * s.t.  bl <= A^Td + constr(xi+alpha*deltaXi) - A^TdeltaXi <= bu
 *
 */
bool SQPmethod::secondOrderCorrection(double cNorm, double cNormTrial, double dfTdeltaXi, bool swCond){

    // If constraint violation of the trialstep is lower than the current one skip SOC
    if(cNormTrial < cNorm || cNormTrial < 1e-2*param->feas_tol)// && stats->itCount != 4)
    {
        std::cout << "Constraint violation is low, skip SOC\n";
        return false;
    }

    int nSOCS = 0;
    double cNormTrialSOC, cNormOld, objTrialSOC;
    int k, info;
    int nVar = prob->nVar;
    Matrix deltaXiSOC, lambdaQPSOC;

    // vars->trialConstr contains result at first trial point: c(xi+deltaXi)
    // vars->constrJac, vars->AdeltaXi and vars->gradObj are unchanged so far.

    // First SOC step
    deltaXiSOC.Dimension( vars->deltaXi.M() ).Initialize( 0.0 );
    lambdaQPSOC.Dimension( vars->lambdaQP.M() ).Initialize( 0.0 );

    // Second order correction loop
    cNormOld = cNorm;
    for (k = 0; k<param->max_SOC; k++){
        nSOCS++;

        //Update AdeltaXi, where we use the original step in the first iteration and the previous SOC step in the following iterations (thats why we don't do it in the solve_SOC_QP method)
        if (k == 0){
            if (param->sparse)
                Atimesb(vars->sparse_constrJac.nz.get(), vars->sparse_constrJac.row.get(), vars->sparse_constrJac.colind.get(), vars->deltaXi, vars->AdeltaXi);
            else
                Atimesb(vars->constrJac, vars->deltaXi, vars->AdeltaXi);
        }
        else{
            if (param->sparse)
                Atimesb(vars->sparse_constrJac.nz.get(), vars->sparse_constrJac.row.get(), vars->sparse_constrJac.colind.get(), deltaXiSOC, vars->AdeltaXi);
            else
                Atimesb(vars->constrJac, deltaXiSOC, vars->AdeltaXi);
        }

        // Solve SOC QP to obtain new, corrected deltaXi
        // (store in separate vector to avoid conflict with original deltaXi -> need it in linesearch!)
        QPresults infoQP = solve_SOC_QP(deltaXiSOC, lambdaQPSOC);
            
        if (infoQP != QPresults::success)
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
        prob->evaluate( vars->trialXi, &objTrialSOC, vars->trialConstr, &info );
        stats->nFunCalls++;
        cNormTrialSOC = lInfConstraintNorm( vars->trialXi, vars->trialConstr, prob->lb_var, prob->ub_var, prob->lb_con, prob->ub_con );
        if (info != 0 || objTrialSOC < prob->objLo || objTrialSOC > prob->objUp || !(objTrialSOC == objTrialSOC) || !(cNormTrialSOC == cNormTrialSOC))
            return false; // evaluation error, abort SOC

        // Check acceptability to the filter (in SOC)
        if (pairInFilter(cNormTrialSOC, objTrialSOC)){
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
        if( cNorm > param->thetaMin || !swCond ){
            if( cNormTrialSOC < (1.0 - param->gammaTheta) * cNorm || objTrialSOC < vars->obj - param->gammaF * cNorm ){
                // found suitable alpha during SOC, stop
                acceptStep( deltaXiSOC, lambdaQPSOC, 1.0, nSOCS );
                return true;
            }
            else{
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
int SQPmethod::feasibilityRestorationPhase(){
    // No Feasibility restoration phase
    if (param->enable_rest == 0) throw std::logic_error("feasibility restoration called when enable_rest == 0, this should not happen");

    //Set up the restoration problem and restoration method
    stats->nRestPhaseCalls++;
    bool warmStart = (vars->steptype == 3);
    
    if (!warmStart){
        vars->nRestIt = 0;
        rest_stats = std::make_unique<SQPstats>(stats->outpath);
        //NEW
        rest_prob = std::make_unique<RestorationProblem>(prob, vars->xi, param->rest_rho, param->rest_zeta);
        //rest_prob->update_xi_ref(vars->xi);
        
        rest_method = std::make_unique<SQPmethod>(rest_prob.get(), rest_param.get(), rest_stats.get());
        rest_method->init();
    }
    //Invoke the restoration loop with setup problem and method
    return innerRestorationPhase(rest_prob.get(), rest_method.get(), warmStart);
}


int SQPmethod::innerRestorationPhase(RestorationProblemBase *Rprob, SQPmethod *Rmeth, bool RwarmStart, double min_stepsize_sum){
    int info;
    int feas_result = 1; //0: Success, 1: max_rest_IT reached, 2: converged/locally infeasible, 3: Some error occurred
    SQPresults ret;
    int maxRestIt = 20;
    double cNormTrial, objTrial, stepsize_sum = 0.;
    
    for (; vars->nRestIt < maxRestIt; vars->nRestIt++){
        // One iteration for minimum norm NLP
        ret = rest_method->run(1, RwarmStart);
        RwarmStart = 1;
        
        // If restMethod yields error, stop restoration phase
        if (int(ret) < 0){
            feas_result = 3;
            break;
        }
        
        // Require sufficient progress
        stepsize_sum += rest_method->vars->alpha;
        if (stepsize_sum < min_stepsize_sum){
            continue;
        }
        // Get new xi from the restoration phase
        rest_method->get_xi(rest_xi);
        rest_prob->recover_xi(rest_xi, vars->trialXi);
        
        // Compute objective at trial point
        prob->evaluate(vars->trialXi, &objTrial, vars->constr, &info);
        stats->nFunCalls++;
        cNormTrial = lInfConstraintNorm(vars->trialXi, vars->constr, prob->lb_var, prob->ub_var, prob->lb_con, prob->ub_con);
        if (info != 0 || objTrial < prob->objLo || objTrial > prob->objUp || !(objTrial == objTrial) || !(cNormTrial == cNormTrial))
            continue;
        
        // Is this iterate acceptable for the filter?
        if (!pairInFilter(cNormTrial, objTrial)){
            // success
            std::cout << "Found a point acceptable for the filter.\n";
            feas_result = 0;
            break;
        }
        
        // If minimum norm NLP has converged, declare local infeasibility
        /*
        if (rest_method->vars->tol < param->opt_tol && rest_method->vars->cNormS < param->feas_tol){
            std::cout << "feas_ret = " << static_cast<int>(ret) << "\n";
            feas_result = 2;
            break;
        }
        */
        
        if (static_cast<int>(ret) > 0){
            feas_result = 2;
            break;
        }
    }
    
    // Success, locally infeasible or maximum restoration iterations reached
    if (feas_result == 0 || feas_result == 2){        
        for (int i = 0; i < prob->nVar; i++){
            vars->deltaXi(i) = -vars->xi(i);
            vars->deltaXi(i) += vars->trialXi(i);
        }
        rest_method->get_lambda(rest_lambda); 
        rest_method->get_lambdaQP(rest_lambdaQP);
        rest_prob->recover_lambda(rest_lambda, vars->trialLambda);
        rest_prob->recover_lambda(rest_lambdaQP, vars->lambdaQP);
        
        if (feas_result == 0){
            acceptStep(vars->deltaXi, vars->trialLambda, 1.0, 0);
            // Original blockSQP does not add restoration result to filter, though this is done in the Biegler/Waechter filter line search paper
            //augmentFilter(cNormTrial, objTrial);
        }
        else if (feas_result == 2){
            //If restoration method converged and remaining constraint violation is small, try overriding the filter
            if (cNormTrial < 1e-4 && vars->remaining_filter_overrides > 0){
                force_accept(vars->deltaXi, vars->trialLambda, 1.0, 0);
                vars->remaining_filter_overrides -= 1;
                feas_result = 0;
            }
            else{
                acceptStep(vars->deltaXi, vars->trialLambda, 1.0, 0);
            }
        }
        
        vars->obj = objTrial;
        vars->cNorm = cNormTrial;
        
        // reset reduced step counter
        vars->reducedStepCount = 0;
        
        // reset Hessian and limited memory information
        resetHessians();
        vars->n_scaleIt = 0;
    }
    
    
    if (feas_result == 2){
        stats->printProgress( prob, vars.get(), param, 0 );
        printf("The problem seems to be locally infeasible. Infeasibilities minimized.\n");
    }
    
    return feas_result;
}


/**
 * Try to (partly) improve constraint violation by satisfying
 * the (pseudo) continuity constraints, i.e. do a single shooting
 * iteration with the current controls and measurement weights q and w
 */

int SQPmethod::feasibilityRestorationHeuristic(){
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
        return 1;
    
    // Compute objective and constraints at the new (hopefully feasible) point
    trial_constr.Dimension(prob->nCon).Initialize(0.0);
    prob->evaluate(vars->trialXi, &obj_trial, trial_constr, &info);
    
    stats->nFunCalls++;
    cNormTrial = lInfConstraintNorm(vars->trialXi, trial_constr, prob->lb_var, prob->ub_var, prob->lb_con, prob->ub_con);
    if (info != 0 || obj_trial < prob->objLo || obj_trial > prob->objUp || !(obj_trial == obj_trial) || !(cNormTrial == cNormTrial))
        return 1;
    
    // Is the new point acceptable for the filter?
    if (pairInFilter(cNormTrial, obj_trial)){
        std::cout << "New point is in the filter\n";
        // point is in the taboo region, restoration heuristic not successful!
        return 1;
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
    for (k = 0; k < prob->nVar; k++){
        vars->deltaXi(k) = vars->trialXi(k) - vars->xi(k);
        vars->xi(k) = vars->trialXi(k);
    }
    
    return 0;
}


/**
 * If the line search fails, check if the full step reduces the KKT error by a factor kappaF.
 */
int SQPmethod::kktErrorReduction(){
    int info = 0;
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
    trialConstr.Dimension(prob->nCon).Initialize(0.0);
    prob->evaluate(vars->trialXi, &objTrial, trialConstr, &info);
    stats->nFunCalls++;
    cNormTrial = lInfConstraintNorm(vars->trialXi, trialConstr, prob->lb_var, prob->ub_var, prob->lb_con, prob->ub_con);
    if (info != 0 || objTrial < prob->objLo || objTrial > prob->objUp || !(objTrial == objTrial) || !(cNormTrial == cNormTrial)){
        // evaluation error
        return 1;
    }
    
    // Compute KKT error of the new point

    // scaled norm of Lagrangian gradient
    trialGradLagrange.Dimension( prob->nVar ).Initialize( 0.0 );
    if( param->sparse )
        calcLagrangeGradient( vars->lambdaQP, vars->gradObj, vars->sparse_constrJac.nz.get(),
                              vars->sparse_constrJac.row.get(), vars->sparse_constrJac.colind.get(), trialGradLagrange, 0 );
    else
        calcLagrangeGradient( vars->lambdaQP, vars->gradObj, vars->constrJac,
                              trialGradLagrange, 0 );
    
    trialGradNorm = lInfVectorNorm( trialGradLagrange );
    trialTol = trialGradNorm /( 1.0 + lInfVectorNorm( vars->lambdaQP ) );
    
    if (std::max(cNormTrial, trialTol) < param->kappaF*std::max(vars->cNorm, vars->tol)){
        acceptStep(1.0);
        return 0;
    }
    return 1;
}

/**
 * Check if current entry is accepted to the filter:
 * (cNorm, obj) in F_k
 */
bool SQPmethod::pairInFilter( double cNorm, double obj )
{
    std::set< std::pair<double,double> >::iterator iter;
    //std::set< std::pair<double,double> > *filter;
    //filter = vars->filter;

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

    for (iter = vars->filter.begin(); iter != vars->filter.end(); iter++){
        //Note: Changing the tolerance away from 0.01 is not recommended, leads to a lot more linesearch failures and unsuccessful terminations
        if ((cNorm >= (1.0 - param->gammaTheta) * iter->first || (cNorm < 0.01 * param->feas_tol && iter->first < 0.01 * param->feas_tol)) &&
             obj >= iter->second - param->gammaF * iter->first)
            return 1;
    }
    return 0;
}


void SQPmethod::initializeFilter()
{
    std::set< std::pair<double,double> >::iterator iter;
    std::pair<double,double> initPair(param->thetaMax, prob->objLo);

    // Remove all elements
    iter = vars->filter.begin();
    while (iter != vars->filter.end()){
        std::set< std::pair<double,double> >::iterator iterToRemove = iter;
        iter++;
        vars->filter.erase(iterToRemove);
    }

    // Initialize with pair ( maxConstrViolation, objLowerBound );
    vars->filter.insert(initPair);
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
    vars->filter.insert( entry );

    // Remove dominated elements
    iter=vars->filter.begin();
    while (iter != vars->filter.end())
    {
        //printf(" iter->first=%g, entry.first=%g, iter->second=%g, entry.second=%g\n",iter->first, entry.first, iter->second, entry.second);
        if( iter->first > entry.first && iter->second > entry.second )
        {
            std::set< std::pair<double,double> >::iterator iterToRemove = iter;
            iter++;
            vars->filter.erase( iterToRemove );
        }
        else
            iter++;
    }
    return;
}

/*
int SCQPmethod::feasibilityRestorationPhase(){
    // No Feasibility restoration phase
    if (param->enable_rest == 0) throw std::logic_error("feasibility restoration called when enable_rest == 0, this should not happen");

    //Set up the restoration problem and restoration method
    stats->nRestPhaseCalls++;
    int warmStart;
    // Iterate until a point acceptable to the filter is found
    if (vars->steptype != 3){
        rest_prob->update_xi_ref(vars->xi);
        //TODO scaling in restoration phase
        //rest_prob->n_vblocks = prob->n_vblocks;
        //rest_prob->vblocks = rest_vblocks;

        warmStart = 0;
        vars->nRestIt = 0;
        rest_method = std::make_unique<SCQPmethod>(rest_prob.get(), rest_param.get(), rest_stats.get(), rest_cond.get());
        rest_method->init();
    }
    else warmStart = 1;
    
    //Invoke the restoration phase with setup problem and method
    return innerRestorationPhase(rest_prob.get(), rest_method.get(), warmStart);
}

*/

bool bound_correction_method::filterLineSearch(){

    double alpha = 1.0;
    double cNorm, cNormTrial, objTrial, dfTdeltaXi;

    int k, info;
    int nVar = prob->nVar;

    //int ind_1, ind_2, ind;
    //double max_dep_bound_violation, xi_s;
    
    Matrix deltaXi_save, lambdaQP_save;

    // Compute ||constr(xi)|| at old point
    cNorm = lInfConstraintNorm( vars->xi, vars->constr, prob->lb_var, prob->ub_var, prob->lb_con, prob->ub_con );

    // Backtracking line search
    for (k = 0; k < param->max_linesearch_steps; k++){
        //If indefinite hessian yielded step with small stepsize, retry with step from fallback hessian
        /*
        if (k > 3 && !vars->conv_qp_solved){
            if (solveQP(vars->deltaXi, vars->lambdaQP, 1)) return 1;
            else{k = 0; alpha = 1.0;}
        }
        */

        // Compute grad(f)^T * deltaXi (deltaXi being the original step without (second order) corrections)
        dfTdeltaXi = 0.0;
        for (int i = 0; i < nVar; i++)
            dfTdeltaXi += vars->gradObj( i ) * vars->deltaXi( i );
        
        //Since the original step vars->deltaXi, vars->lambdaQP may get modified by correction,
        //work with a different variable 
        if (k == 0){
            deltaXi_save = vars->deltaXi;
            lambdaQP_save = vars->lambdaQP;
            
            QPresults infoQP = bound_correction(vars->deltaXi, vars->lambdaQP);
            
            //If model bound correction failed for indefinite Hessian, resolve with convex Hessian and try again
            if (infoQP != QPresults::success && !vars->conv_qp_solved){
                if (solveQP(vars->deltaXi, vars->lambdaQP, 1) != QPresults::success) return 1;
                else{k = -1; alpha = 1.0; continue;}
            }
        }
        else if (k == 1){
            //Both bound and second order corrections are only tried in the first iteration, restore original step for backtracking line search
            vars->deltaXi = deltaXi_save;
            vars->lambdaQP = lambdaQP_save;
        }
        
        // Compute new trial point and truncate any bound violation
        for (int i = 0; i < nVar; i++){
            vars->trialXi(i) = vars->xi(i) + alpha * vars->deltaXi(i);
            
            if (vars->trialXi(i) < prob->lb_var(i)){
                vars->trialXi(i) = prob->lb_var(i);
            }
            else if (vars->trialXi(i) > prob->ub_var(i)){
                vars->trialXi(i) = prob->ub_var(i);
            }
        }
        
        // Compute objective and at ||constr(trialXi)||_1 at trial point
        prob->evaluate( vars->trialXi, &objTrial, vars->trialConstr, &info );
        stats->nFunCalls++;
        
        //cNormTrial = l1ConstraintNorm( vars->trialXi, vars->constr, prob->lb_var, prob->ub_var, prob->lb_con, prob->ub_con );
        cNormTrial = lInfConstraintNorm( vars->trialXi, vars->trialConstr, prob->lb_var, prob->ub_var, prob->lb_con, prob->ub_con );
        
        // Reduce step if evaluation fails, if lower bound is violated or if objective is NaN
        if(info != 0 || objTrial < prob->objLo || objTrial > prob->objUp || !(objTrial == objTrial) || !(cNormTrial == cNormTrial)){
            // evaluation error, reduce stepsize
            reduceStepsize(&alpha);
            continue;
        }

        // Check acceptability to the filter
        if (pairInFilter(cNormTrial, objTrial)){
            // Trial point is in the prohibited region defined by the filter, try second order correction
            if (k == 0) std::cout << "Point is in the filter, try SOC\n";

            if (k == 0 && secondOrderCorrection(cNorm, cNormTrial, dfTdeltaXi, true))
                break;
            else{
                reduceStepsize(&alpha);
                continue;
            }
        }
        
        // Check sufficient decrease, case I:
        // If we are (almost) feasible and a "switching condition" is satisfied
        // require sufficient progress in the objective instead of bi-objective condition
        if( cNorm <= param->thetaMin ){
            // Switching condition, part 1: grad(f)^T * deltaXi < 0 ?
            if( dfTdeltaXi < 0 ){
                // Switching condition, part 2: alpha * ( - grad(f)^T * deltaXi )**sF > delta * cNorm**sTheta ?
                if( alpha * pow( (-dfTdeltaXi), param->sF ) > param->delta * pow( cNorm, param->sTheta ) ){
                    // Switching conditions hold: Require satisfaction of Armijo condition for objective
                    if( objTrial <= vars->obj + param->eta*alpha*dfTdeltaXi ){
                        // found suitable alpha, stop
                        acceptStep( alpha );
                        break;
                    }
                    else{
                        //Armijo condition violated for convex QP, try second order correction
                        if (k == 0 && secondOrderCorrection(cNorm, cNormTrial, dfTdeltaXi, true))
                            break;
                        else{
                            reduceStepsize(&alpha);
                            continue;
                        }
                    }
                }
            }
        }
        
        // Check sufficient decrease, case II:
        // Bi-objective (filter) condition
        if (cNormTrial < (1.0 - param->gammaTheta) * cNorm || objTrial < vars->obj - param->gammaF * cNorm){
            // found suitable alpha, stop
            acceptStep(alpha);
            break;
        }
        else{
            std::cout << "Filter condition violated, try SOC\n";
            // Trial point is dominated by current point, try second order correction
            if (k == 0 && secondOrderCorrection(cNorm, cNormTrial, dfTdeltaXi, false))
                break; // SOC yielded suitable alpha, stop
            else{
                reduceStepsize(&alpha);
                continue;
            }
        }
    }// backtracking steps
    
    // No step could be found by the line search
    if( k == param->max_linesearch_steps )
        return true;
    
    // Augment the filter if switching condition or Armijo condition does not hold
    // if( dfTdeltaXi >= 0 )
    //     augmentFilter( cNormTrial, objTrial );
    // else if( alpha * pow( (-dfTdeltaXi), param->sF ) > param->delta * pow( cNorm, param->sTheta ) )// careful with neg. exponents!
    //     augmentFilter( cNormTrial, objTrial );
    // else if( objTrial <= vars->obj + param->eta*alpha*dfTdeltaXi )
    //     augmentFilter( cNormTrial, objTrial );
    
    if (dfTdeltaXi >= 0){
        std::cout << "Step is not a descent direction, augment filter\n";
        // augmentFilter(cNormTrial, objTrial);
        augmentFilter(cNorm, vars->obj);
    }
    else if (alpha * pow((-dfTdeltaXi), param->sF) <= param->delta*pow(cNorm, param->sTheta)){// careful with neg. exponents!
        std::cout << "Switching condition violated, augment filter\n";
        // augmentFilter(cNormTrial, objTrial);
        augmentFilter(cNorm, vars->obj);
    }
    else if (objTrial > vars->obj + param->eta*alpha*dfTdeltaXi){
        std::cout << "Armijo condition violated, augment filter\n";
        // augmentFilter(cNormTrial, objTrial);
        augmentFilter(cNorm, vars->obj);
    }

    return false;
}


int bound_correction_method::feasibilityRestorationPhase(){
    // No Feasibility restoration phase
    if (param->enable_rest == 0) [[unlikely]] throw std::logic_error("feasibility restoration called when enable_rest == 0, this should not happen");
    
    //Set up the restoration problem and restoration method
    stats->nRestPhaseCalls++;
    bool warmStart = (vars->steptype == 3);
    // Iterate until a point acceptable to the filter is found
    if (!warmStart){
        vars->nRestIt = 0;
        rest_stats = std::make_unique<SQPstats>(stats->outpath);
        
        rest_prob = std::make_unique<TC_restoration_Problem>(prob, vars->xi, param->rest_rho, param->rest_zeta);
        
        warmStart = 0;
        vars->nRestIt = 0;
        rest_method = std::make_unique<bound_correction_method>(rest_prob.get(), rest_param.get(), rest_stats.get());
        rest_method->init();
    }
    else warmStart = 1;
    
    //Invoke the restoration phase with setup problem and method
    return innerRestorationPhase(rest_prob.get(), rest_method.get(), warmStart);
}



} // namespace blockSQP
