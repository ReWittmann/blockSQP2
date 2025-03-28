/*
 * blockSQP -- Sequential quadratic programming for problems with
 *             block-diagonal Hessian matrix.
 * Copyright (C) 2012-2015 by Dennis Janka <dennis.janka@iwr.uni-heidelberg.de>
 *
 * Licensed under the zlib license. See LICENSE for more details.
 */

/**
 * \file blocksqp_main.cpp
 * \author Dennis Janka
 * \date 2012-2015
 *
 *  Implementation of SQPmethod class.
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

SQPmethod::SQPmethod( Problemspec *problem, SQPoptions *parameters, SQPstats *statistics ): prob(problem), param(parameters), stats(statistics){
    // Check if there are options that are infeasible and set defaults accordingly
    param->optionsConsistency(problem);
    
    if (param->autoScaling){
        scaled_prob = new scaled_Problemspec(problem);
        prob = scaled_prob;
    }
    else{
        scaled_prob = nullptr;
        prob = problem;
    }
    vars = new SQPiterate(prob, param, true);

    // Create a solver object for quadratic subproblems.
    sub_QP = create_QPsolver(prob->nVar, prob->nCon, vars->nBlocks, param);
    
    initCalled = false;
    
    //Setup the feasibility restoration problem
    if (param->restoreFeas){
        rest_opts = new SQPoptions();
        // Set options for the SQP method for this problem
        rest_opts->globalization = 1;
        rest_opts->whichSecondDerv = 0;
        rest_opts->restoreFeas = 0;
        rest_opts->hessUpdate = 2;
        rest_opts->hessLimMem = 1;
        rest_opts->hessScaling = 2;
        rest_opts->opttol = param->opttol;
        rest_opts->nlinfeastol = param->nlinfeastol;
        rest_opts->QPsol = param->QPsol;
        rest_opts->QPsol_opts = param->QPsol_opts;
        rest_opts->hessDampFac = 0.2;
        
        rest_opts->printRes = false;

        //rest_opts->autoScaling = param->autoScaling;
        
        rest_prob = nullptr;
        rest_stats = nullptr;
        rest_method = nullptr;
    }
    else{
        rest_prob = nullptr;
        rest_opts = nullptr;
        rest_stats = nullptr;
        rest_method = nullptr;
    }
}

SQPmethod::SQPmethod(): prob(nullptr), param(nullptr), stats(nullptr), vars(nullptr), sub_QP(nullptr),
    rest_prob(nullptr), rest_opts(nullptr), rest_stats(nullptr), rest_method(nullptr), scaled_prob(nullptr), initCalled(false){}

SQPmethod::~SQPmethod(){
    delete vars;
    delete sub_QP;

    delete scaled_prob;
    delete rest_prob;
    delete rest_opts;
    delete rest_stats;
    delete rest_method;
}

void SQPmethod::init(){
    // Print header and information about the algorithmic parameters
    printInfo( param->printLevel );

    // Open output files
    stats->initStats( param );
    vars->initIterate( param );

    // Initialize filter with pair ( maxConstrViolation, objLowerBound )
    initializeFilter();

    // Set initial values for all xi and set the Jacobian for linear constraints
    if( param->sparseQP )
        prob->initialize( vars->xi, vars->lambda, vars->jacNz, vars->jacIndRow, vars->jacIndCol );
    else
        prob->initialize( vars->xi, vars->lambda, vars->constrJac );

    initCalled = true;
}


SQPresult SQPmethod::run(int maxIt, int warmStart){
    int it = 0, infoQP = 0, infoEval = 0;
    bool skipLineSearch = false;
    bool hasConverged = false;
    int whichDerv = param->whichSecondDerv;
    int n_convShift;

    if (!initCalled){
        printf("init() must be called before run(). Aborting.\n");
        //return -1;
        return loud_SQPresult(SQPresult::misc_error, param->printRes);
    }
    
    if (warmStart == 0 || stats->itCount == 0){
        // SQP iteration 0
        if (param->sparseQP)
            prob->evaluate(vars->xi, vars->lambda, &vars->obj, vars->constr, vars->gradObj,
                            vars->jacNz, vars->jacIndRow, vars->jacIndCol, vars->hess1, 1+whichDerv, &infoEval);
        else
            prob->evaluate(vars->xi, vars->lambda, &vars->obj, vars->constr, vars->gradObj,
                            vars->constrJac, vars->hess1, 1+whichDerv, &infoEval);
        stats->nDerCalls++;

        /// Check if converged
        hasConverged = calcOptTol();
        stats->printProgress( prob, vars, param, hasConverged );
        if (hasConverged) return loud_SQPresult(SQPresult::success, param->printRes);

        /// Set initial Hessian approximation
        //Consider implementing strategy for the initial hessian, see e.g. Leineweber 1995 Theory of MUSCOD S. 72

        calcInitialHessians();
        vars->hess2_updated = true;
    }

    /*
     * Main SQP Loop
     */

    for (; it<maxIt; it++){
        //Enter new iteration
        stats->itCount++;

        /////////////////////////////////////////////
        ///PHASE 1: Solve the quadratic subproblem///
        /////////////////////////////////////////////

        /// Solve QP subproblem with qpOASES or QPOPT
        infoQP = solveQP(vars->deltaXi, vars->lambdaQP, int(vars->conv_qp_only));

        //if (infoQP == 0) printf("***QP solution successful***");
        if (infoQP == 0);
        else if (infoQP == 1){
            bool qpError = true;

            std::cout << "QP solution is taking too long, solve again with identity matrix.\n";
            infoQP = solveQP(vars->deltaXi, vars->lambdaQP, 2);
            if (infoQP){
                std::cout << "QP solution failed again, try to reduce constraint violation\n";
                skipLineSearch = true;

                if (vars->steptype < 2){
                    qpError = feasibilityRestorationHeuristic();
                    if (!qpError){
                        vars->steptype = 2;
                        std::cout << "Success\n";
                    }
                    else
                        std::cout << "Failed\n";
                }

                if (qpError && param->restoreFeas && vars->cNorm > 0.01 * param->nlinfeastol){
                    std::cout << "Start feasibility restoration phase\n";
                    qpError = feasibilityRestorationPhase();
                    vars->steptype = 3;
                }

                if (qpError){
                    std::cout << "QP error, stop\n";
                    return loud_SQPresult(SQPresult::qp_failure, param->printRes);
                }
            }
            else vars->steptype = 1;
        }
        else if (infoQP == 2 || infoQP > 3){
            std::cout << "***QP error. Solve again with identity matrix.***\n";
            infoQP = solveQP(vars->deltaXi, vars->lambdaQP, 2);
            if (infoQP){
                // If there is still an error, terminate.
                printf( "***QP error. Stop.***\n" );
                printf("InfoQP is %d\n", infoQP);
                return loud_SQPresult(SQPresult::qp_failure, param->printRes);
            }
            else vars->steptype = 1;
        }
        else if (infoQP == 3){
            // 3.) QP infeasible, try to restore feasibility
            int feasError = 1;
            skipLineSearch = true; // don't do line search with restoration step

            // Try to reduce constraint violation by heuristic
            if (vars->steptype < 2){
                printf("***QP infeasible. Trying to reduce constraint violation...");
                feasError = feasibilityRestorationHeuristic();
                if (!feasError){
                    vars->steptype = 2;
                    printf("Success.***\n");
                }
                else printf("Failed.***\n");
            }

            // Invoke feasibility restoration phase
            if (feasError && param->restoreFeas && vars->cNorm > 0.01 * param->nlinfeastol){
                printf("***Start feasibility restoration phase.***\n");
                feasError = feasibilityRestorationPhase();
                vars->steptype = 3;
            }
            
            // If everything failed, abort.
            if (feasError == 1 || feasError > 2) return loud_SQPresult(SQPresult::restoration_failure, param->printRes);
            else if (feasError == 2) return loud_SQPresult(SQPresult::local_infeasibility, param->printRes);
        }

        /////////////////////////////////////////////////////////////
        ///PHASE 2: Do the filter line search (+ failure handling)///
        /////////////////////////////////////////////////////////////

        /// Determine steplength alpha
        if (param->globalization == 0 || (param->skipFirstGlobalization && stats->itCount == 1)){
            // No globalization strategy, but reduce step if function cannot be evaluated
            if (fullstep()){
                printf( "***Constraint or objective could not be evaluated at new point. Stop.***\n" );
                return loud_SQPresult(SQPresult::eval_failure, param->printRes);
            }
            vars->steptype = 0;
        }
        else if (param->globalization == 1 && !skipLineSearch){
            // Filter line search based on Waechter et al., 2006 (Ipopt paper)
            if (filterLineSearch() || vars->reducedStepCount > param->maxConsecReducedSteps){
                // Filter line search did not produce a step. Now there are a few things we can try ...
                bool lsError = true;
                
                std::cout << "Filter line search failed, begin handling\n";

                //If we already found a solution and steps are only for improving accuracy, terminate.
                if (vars->sol_found){
                    vars->restore_iterate();
                    if (vars->tol <= 1e-2*param->opttol && vars->cNormS <= 1e-2*param->nlinfeastol) return loud_SQPresult(SQPresult::super_success, param->printRes);
                    else return loud_SQPresult(SQPresult::success, param->printRes);
                }
                

                if (vars->KKT_heuristic_active){
                    std::cout << "filterLineSearch failed, try to reduce kktError\n" << std::flush;
                    vars->tol_save = vars->tol;
                    lsError = kktErrorReduction();
                    if (!lsError)
                        vars->steptype = -1;
                }

                //Heuristic 2: If possibly indefinite Hessian was used, retry with step from fallback Hessian
                if (lsError && !vars->conv_qp_solved){
                    std::cout << "filterLineSearch failed, try again with fallback Hessian\n";
                    infoQP = solveQP(vars->deltaXi, vars->lambdaQP, 1);
                    if (infoQP == 0) lsError = bool(filterLineSearch());
                    if (!lsError) vars->steptype = 0;
                }

                //Heuristic 3: Ignore acceptance criteria up to a limited number of times if we are close to a solution and feasible
                //Remove entries from filter that dominate the new point.

                //if (lsError && vars->tol <= 1e2*param->opttol && vars->cNormS <= param->nlinfeastol && vars->local_lenience > 0){
                if (lsError && vars->tol <= std::pow(param->opttol, 2./3.) && vars->cNormS <= param->nlinfeastol && vars->local_lenience > 0){
                    force_accept(1.0);
                    vars->local_lenience--;
                    lsError = false;
                    std::cout << "Filter line search failed close to a local solution, ignore filter. We can only do this " << vars->local_lenience << " more times\n";
                    vars->steptype = -2;
                }

                ///If filter line search and first set of heuristics failed, check for feasibility and low KKT error. Declare partial success and terminate if true.
                if (param->allow_premature_termination && lsError && vars->cNormS <= param->nlinfeastol && vars->tol <= std::pow(param->opttol, 0.75))
                    return loud_SQPresult(SQPresult::partial_success, param->printRes);

                // Heuristic 4: Try to reduce constraint violation by closing continuity gaps to produce an admissable iterate
                if (lsError && vars->cNorm > 0.01 * param->nlinfeastol && vars->steptype < 2){
                    // Don't do this twice in a row!
                    printf("***Warning! Steplength too short. Trying to reduce constraint violation...");
                    // Integration over whole time interval
                    lsError = bool(feasibilityRestorationHeuristic());
                    if (!lsError){
                        vars->steptype = 2;
                        printf("Success.***\n");
                    }
                    else printf("Failed.***\n");
                }

                if (lsError && vars->steptype != 1){
                    std::cout << "***Warning! Steplength too short. Trying to find a new step with identity Hessian.***\n";
                    infoQP = solveQP(vars->deltaXi, vars->lambdaQP, 2);
                    if (infoQP == 0) lsError = bool(filterLineSearch());
                    vars->steptype = 1;
                }

                // If this does not yield a successful step, start restoration phase
                if (lsError && vars->cNorm > 0.01 * param->nlinfeastol && param->restoreFeas){
                    printf("***Warning! Steplength too short. Start feasibility restoration phase.***\n");
                    // Solve NLP with minimum norm objective
                    lsError = bool(feasibilityRestorationPhase());
                    vars->steptype = 3;
                }

                // If everything failed, abort.
                if (lsError){
                    printf( "***Line search error. Stop.***\n" );
                    return loud_SQPresult(SQPresult::linesearch_failure, param->printRes);
                }
            }
            else{
                vars->steptype = 0;
                vars->KKT_heuristic_active = true;
                /*
                    if (vars->reducedStepCount > 3){}
                */

            }
        }
        
        ////////////////////////////////////
        ///PHASE 3: Update iteration data///
        ////////////////////////////////////

        /// Calculate "old" Lagrange gradient: gamma = dL(xi_k, lambda_k+1)
        calcLagrangeGradient( vars->gamma, 0 );

        /// Evaluate functions and gradients at the new xi
        if (param->sparseQP){
            prob->evaluate(vars->xi, vars->lambda, &vars->obj, vars->constr, vars->gradObj,
                            vars->jacNz, vars->jacIndRow, vars->jacIndCol, vars->hess1, 1+whichDerv, &infoEval);
        }
        else
            prob->evaluate(vars->xi, vars->lambda, &vars->obj, vars->constr, vars->gradObj,
                            vars->constrJac, vars->hess1, 1+whichDerv, &infoEval);
        stats->nDerCalls++;

        /// Check if converged
        hasConverged = calcOptTol();

        /// Calculate difference of old and new Lagrange gradient: gamma = -gamma + dL(xi_k+1, lambda_k+1)
        calcLagrangeGradient(vars->gamma, 1);
        
        stats->printProgress(prob, vars, param, false);
        

        ///Decide wether it is time to terminate///

        //1. Check if termination criteria are satisfied. Either terminate or enter extra step phase
        if (hasConverged && vars->steptype < 2){
            if (param->max_extra_steps > 0){
                if (!vars->sol_found){
                    vars->save_iterate();
                    vars->sol_found = true;
                }
            }
            else return loud_SQPresult(SQPresult::success, param->printRes);
            //return RES::SUCCESS; //Convergence achieved!
        }

        //Handle extra steps for improved accuracy if requested
        if (vars->sol_found && param->max_extra_steps > 0){
            if (vars->n_extra >= param->max_extra_steps){
                vars->restore_iterate();
                if (vars->tol < 1e-2*param->opttol && vars->cNormS < 1e-2*param->nlinfeastol) return loud_SQPresult(SQPresult::super_success, param->printRes);
                else return loud_SQPresult(SQPresult::success, param->printRes);
            }
            //Save current point if it is better in terms of constraint violation and KKT error
            if (std::max(vars->tol/param->opttol, vars->cNormS/param->nlinfeastol) < std::max(vars->tolOpt_save/param->opttol, vars->cNormSOpt_save/param->nlinfeastol))
                vars->save_iterate();
            vars->n_extra++;
        }
        ///No termination at this point, proceed///

        // Check if KKT error was indeed reduced, if not, disable linesearch heuristic until next successful linesearch
        if (vars->steptype == -1){
            if (!(vars->tol < param->kappaF*vars->tol_save)){
                std::cout << "KKT error was not sufficiently reduced, disable step heuristic\n";
                vars->KKT_heuristic_active = false;
            }
            else std::cout << "Step heuristic successful\n";
        }

        //If identity hessian was used three consecutive times, reset Hessian
        if (vars->steptype == 1)
            vars->n_id_hess += 1;
        else
            vars->n_id_hess = 0;
        if (vars->n_id_hess > 2){
            resetHessians();
            vars->n_id_hess = 0;
            //continue;
        }

        //Update position of current calculated delta - gamma pair, set vars->deltaXi, vars->gamma to next (empty) position, precalculate scalar products for Hessian update and sizing
        updateDeltaGammaData();

        //If we appear to be reasonably close to a local optimum, enable SR1 updates for faster local convergence if only convex QPs were enabled before
        if (vars->tol <= 1e-4 && vars->cNormS <= 1e-4 && stats->itCount >= 8){
        //if (vars->tol <= std::sqrt(param->opttol) && vars->cNormS <= std::sqrt(param->nlinfeastol) && it >= 8){
            vars->nearSol = true;
            vars->conv_qp_only = false;
        }
        if (vars->milestone > std::max(vars->tol, vars->cNormS)) vars->milestone = std::max(vars->tol, vars->cNormS);


        //Increment memory counter of each block and scaling memory counter unless step is restoration step.
        if (vars->steptype < 3){
            for (int ind = 0; ind < vars->nBlocks; ind++){
                vars->nquasi[ind] += int(vars->nquasi[ind] < param->hessMemsize);
            }
            vars->n_scaleIt += int(vars->n_scaleIt < vars->dg_nsave);
        }

        //Rescale variables if automatic scaling is enabled. This has to be done before limited memory quasi newton updates are applied.
        if (param->autoScaling) scaling_heuristic();

        ///
        ///PHASE 3.5: Update the Hessian 'approximations' and related data///
        ///

        if (param->hessLimMem){
            //Subvectors deltaNorm and deltaGamma will be updated as needed when calculating the hessian approximation
            //Skip update for the indefinite hessian when we only solve convex QPs. Delay update for convex hessian when we try indefinite Hessian first
            if (vars->conv_qp_only && vars->hess2 != nullptr){
                if (param->fallbackUpdate <= 2)
                    calcHessianUpdateLimitedMemory(param->fallbackUpdate, param->fallbackScaling, vars->hess2);
                vars->hess2_updated = true;
            }
            else{
                if (param->whichSecondDerv < 2){
                    //Calculate/Update first (pos. indefinite) hessian
                    if (param->hessUpdate <= 2 || param->hessUpdate > 6)
                        calcHessianUpdateLimitedMemory(param->hessUpdate, param->hessScaling, vars->hess1);
                    else if (param->hessUpdate == 4)
                        calcFiniteDiffHessian(vars->hess1);
                    vars->hess2_updated = false;
                }
                vars->hess2_updated = false;
            }
        }
        else{
            //Vectors deltaXi and gamma need not be updated when previous steps are not stored and can be overwritten. We also don't need to store their current position.
            if (param->whichSecondDerv < 2){
                if (param->hessUpdate <= 2)
                    calcHessianUpdate(param->hessUpdate, param->hessScaling, vars->hess1);
                else if (param->hessUpdate == 4)
                    calcFiniteDiffHessian(vars->hess1);
            }

            //Also update the fallback hessian as we need to update it in every iteration regardless of whether it is needed
            if (vars->hess2 != nullptr){
                if (param->fallbackUpdate <= 2)
                    calcHessianUpdate(param->fallbackUpdate, param->fallbackScaling, vars->hess2);
                vars->hess2_updated = true;
            }
        }

        //Adjust scaling factor if indefinite hessians are attempted to be convexified by adding scaled identities
        if (param->convStrategy >= 1 && param->maxConvQP > 1 && vars->steptype == 0 && stats->itCount > 1 && !vars->conv_qp_only){
            if (param->maxConvQP > 2){
                //If more than one convexified indefinite QP is tried, shift convexification factor of the successful QP to the last attempted convexified QP.
                //If more than two convexified indefinite QPs are tried and none were accepted, shift last factor to first factor.
                n_convShift = vars->hess_num_accepted - param->maxConvQP + 1 + (param->maxConvQP - 3) * int(vars->hess_num_accepted == param->maxConvQP);
                vars->convKappa *= std::pow(2, n_convShift);
            }
            else{
                //If only one convexified indefinite QP is tried, increase convexification factor if it was rejected, decrease it if it was accepted.
                if (vars->hess_num_accepted == param->maxConvQP) vars->convKappa *= 2;
                else vars->convKappa *= 0.5;
            }
            if (vars->convKappa > 2.0) vars->convKappa = 2.0;
        }
        //The scaling factor adjustment in one line of code
        //vars->convKappa = std::min(1.0e2, vars->convKappa*std::pow(2, (vars->hess_num_accepted - param->maxConvQP + 1 + (param->maxConvQP - 3) * (vars->hess_num_accepted == param->maxConvQP))*(param->maxConvQP > 2) + (1 - 2*(vars->hess_num_accepted < param->maxConvQP))*(param->maxConvQP <= 2))) * (param->convStrategy >= 1 && vars->hess_num_accepted > 0 && vars->steptype == 0) + vars->convKappa * (param->convStrategy < 1 || vars->hess_num_accepted == 0 || vars->steptype != 0);
        vars->hess = vars->hess1;

        //stats->itCount++;
        skipLineSearch = false;
    }

    return SQPresult::it_finished;
}


void SQPmethod::finish()
{
    if( initCalled )
        initCalled = false;
    else
    {
        printf("init() must be called before finish().\n");
        return;
    }

    stats->finish( param );
}


/*

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
    if( param->sparseQP )
        calcLagrangeGradient( vars->lambda, vars->gradObj, vars->jacNz, vars->jacIndRow, vars->jacIndCol, gradLagrange, flag );
    else
        calcLagrangeGradient( vars->lambda, vars->gradObj, vars->constrJac, gradLagrange, flag );
}


bool SQPmethod::calcOptTol(){

    // scaled norm of Lagrangian gradient

    calcLagrangeGradient( vars->gradLagrange, 0 );

    if (!param->autoScaling){
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

    if( vars->tol <= param->opttol && vars->cNormS <= param->nlinfeastol )
        return true;
    else
        return false;

}

*/

/*
void SQPmethod::calc_free_variables_scaling(double *SF){
    int nIt, pos, nfree = prob->nVar, ind_1, scfree, scdep, count_delta = 0, count_gamma = 0;
    double bardelta_u, bardelta_x, bargamma_u, bargamma_x, resF, rgamma = 0., rdelta = 0.;

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
                if (std::abs(vars->deltaMat(ind_1 + i, pos)) > 1e-8){ //Maybe set lower, e.g.1e-10
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

    resF = -1.0;
    if (rgamma > 2.0){
        resF = rgamma/2.0;
    }
    else if (rgamma < 1.0){
        if (rdelta > 1.0){
            if (rgamma < 0.1) resF = 10.0*rgamma;
            else resF = std::min(1.0, rdelta*rgamma);
        }
        else{
            resF = rgamma;
        }
    }

    if (resF > 0){
        vars->vfreeScale *= resF;
        ind_1 = 0;
        for (int k = 0; k < prob->n_vblocks; k++){
            if (!prob->vblocks[k].dependent){
                for (int i = 0; i < prob->vblocks[k].size; i++){
                    SF[ind_1 + i] *= resF;
                }
            }
            ind_1 += prob->vblocks[k].size;
        }
    }

    return;
}

/// Try to rescale variables such that less weight falls on the Hessian approximation and sizing strategy
void SQPmethod::scaling_heuristic(){
    Matrix deltai, smallDelta, smallGamma;
    int pos, Bsize;
    //Scale after iterations 1, 2, 3, 5, 10, 15, ...
    if (stats->itCount > 3 && stats->itCount%5) return;

    for (int i = 0; i < prob->nVar; i++){
        vars->rescaleFactors[i] = 1.0;
    }
    calc_free_variables_scaling(vars->rescaleFactors);
    apply_rescaling(vars->rescaleFactors);
    return;
}

void SQPmethod::apply_rescaling(double *resfactors){
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

    if (param->sparseQP){
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
    if (!param->hessLimMem){
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
        if (param->whichSecondDerv > 0){
            for (int iBlock = (vars->nBlocks - 1)*int(param->whichSecondDerv == 1); iBlock < vars->nBlocks; iBlock++){
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

*/

//Set a new iterate, ignoring the filter and removing dominating entries
/*
void SQPmethod::set_iterate(const Matrix &xi, const Matrix &lambda, bool resetHessian){
    vars->xi = xi;
    if (param->autoScaling){
        for (int i = 0; i < prob->nVar; i++){
            vars->xi(i) *= scaled_prob->scaling_factors[i];
        }
    }
    vars->lambda = lambda;
    int infoEval;
    if (resetHessian) resetHessians();

    if (param->sparseQP)
        prob->evaluate(vars->xi, vars->lambda, &vars->obj, vars->constr, vars->gradObj,
                        vars->jacNz, vars->jacIndRow, vars->jacIndCol, vars->hess1, 1+param->whichSecondDerv, &infoEval);
    else
        prob->evaluate(vars->xi, vars->lambda, &vars->obj, vars->constr, vars->gradObj,
                        vars->constrJac, vars->hess1, 1+param->whichSecondDerv, &infoEval);

    //Remove filter entries that dominate the set point
    std::set<std::pair<double,double>>::iterator iter = vars->filter->begin();
    std::set<std::pair<double,double>>::iterator iterToRemove;
    while (iter != vars->filter->end()){
        if (iter->first < vars->cNorm && iter->second < vars->obj){
            iterToRemove = iter;
            iter++;
            vars->filter->erase(iterToRemove);
        }
        else iter++;
    }
    augmentFilter(vars->cNorm, vars->obj);
    return;
}


Matrix SQPmethod::get_xi(){
    Matrix xi_unscaled(vars->xi);
    if (param->autoScaling){
        for (int i = 0; i < prob->nVar; i++){
            xi_unscaled(i)/= scaled_prob->scaling_factors[i];
        }
    }
    return xi_unscaled;
}



Matrix SQPmethod::get_lambda(){
    return vars->lambda;
}

void SQPmethod::printInfo( int printLevel )
{
    char hessString1[100];
    char hessString2[100];
    char globString[100];
    char qpString[100];

    if( printLevel == 0 )
        return;

    if( param->sparseQP == 0 )
        strcpy( qpString, "dense, reduced Hessian factorization" );
    else if( param->sparseQP == 1 )
        strcpy( qpString, "sparse, reduced Hessian factorization" );
    else if( param->sparseQP == 2 )
        strcpy( qpString, "sparse, Schur complement approach" );

    if( param->globalization == 0 )
        strcpy( globString, "none (full step)" );
    else if( param->globalization == 1 )
        strcpy( globString, "filter line search" );

    if( param->blockHess && (param->hessUpdate == 1 || param->hessUpdate == 2) )
        strcpy( hessString1, "block " );
    else
        strcpy( hessString1, "" );

    if( param->hessLimMem && (param->hessUpdate == 1 || param->hessUpdate == 2) )
        strcat( hessString1, "L-" );

    if( param->hessUpdate == 1 || param->hessUpdate == 4 || (param->hessUpdate == 6) )
    {
        strcpy( hessString2, hessString1 );

        if( param->fallbackUpdate == 0 )
            strcat( hessString2, "Id" );
        else if( param->fallbackUpdate == 1 )
            strcat( hessString2, "SR1" );
        else if( param->fallbackUpdate == 2 )
            strcat( hessString2, "BFGS" );
        else if( param->fallbackUpdate == 4 )
            strcat( hessString2, "Finite differences" );

        if( param->fallbackScaling == 1 )
            strcat( hessString2, ", SP" );
        else if( param->fallbackScaling == 2 )
            strcat( hessString2, ", OL" );
        else if( param->fallbackScaling == 3 )
            strcat( hessString2, ", mean" );
        else if( param->fallbackScaling == 4 )
            strcat( hessString2, ", selective sizing" );
    }
    else
        strcpy( hessString2, "-" );

    if( param->hessUpdate == 0 )
        strcat( hessString1, "Id" );
    else if( param->hessUpdate == 1 )
        strcat( hessString1, "SR1" );
    else if( param->hessUpdate == 2 )
        strcat( hessString1, "BFGS" );
    else if( param->hessUpdate == 4 )
        strcat( hessString1, "Finite differences" );

    if( param->hessScaling == 1 )
        strcat( hessString1, ", SP" );
    else if( param->hessScaling == 2 )
        strcat( hessString1, ", OL" );
    else if( param->hessScaling == 3 )
        strcat( hessString1, ", mean" );
    else if( param->hessScaling == 4 )
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
*/

//////////////////////////////////////////////////////////////////////


SCQPmethod::SCQPmethod( Problemspec *problem, SQPoptions *parameters, SQPstats *statistics, Condenser *CND){

    prob = problem;
    param = parameters; param->optionsConsistency();
    stats = statistics;
    cond = CND;

    if (param->autoScaling){
        scaled_prob = new scaled_Problemspec(problem);
        prob = scaled_prob;
    }
    else{
        scaled_prob = nullptr;
        prob = problem;
    }
    vars = new SCQPiterate(prob, param, cond, true);

    // Check if there are options that are infeasible and set defaults accordingly
    if (param->sparseQP == 0){
        throw std::invalid_argument("SCQPmethod: Error, condensing only works with sparse QPs");
    }
    if (param->blockHess != 1){
        throw std::invalid_argument("SCQPmethod: Error, condensing requires block diagonal hessian for efficient linear algebra");
    }


    sub_QP = create_QPsolver(cond->condensed_num_vars, cond->condensed_num_cons, cond->condensed_num_hessblocks, param);

    initCalled = false;

    if (param->restoreFeas){
        //Setup condenser for the restoration problem
        int N_vblocks = cond->num_vblocks + cond->num_true_cons;
        int N_cblocks = cond->num_cblocks;
        int N_hessblocks = cond->num_hessblocks + cond->num_true_cons;
        int N_targets = cond->num_targets;

        rest_vblocks = new vblock[N_vblocks];
        rest_cblocks = new cblock[N_cblocks];
        rest_h_sizes = new int[N_hessblocks];
        rest_targets = new condensing_target[N_targets];

        for (int i = 0; i<cond->num_vblocks; i++){
            rest_vblocks[i] = cond->vblocks[i];
        }
        for (int i = cond->num_vblocks; i < N_vblocks; i++){
            rest_vblocks[i] = vblock(1, false);
        }

        for (int i = 0; i<cond->num_cblocks; i++){
            rest_cblocks[i] = cond->cblocks[i];
        }

        for (int i = 0; i<cond->num_hessblocks; i++){
            rest_h_sizes[i] = cond->hess_block_sizes[i];
        }
        for (int i = cond->num_hessblocks; i<N_hessblocks; i++){
            rest_h_sizes[i] = 1;
        }

        for (int i = 0; i<cond->num_targets; i++){
            rest_targets[i] = cond->targets[i];
        }
        rest_cond = new Condenser(rest_vblocks, N_vblocks, rest_cblocks, N_cblocks, rest_h_sizes, N_hessblocks, rest_targets, N_targets, 0);

        //Setup options for the restoration problem
        rest_opts = new SQPoptions();
        rest_opts->globalization = 1;
        rest_opts->whichSecondDerv = 0;
        rest_opts->restoreFeas = 0;
        //rest_opts->hessUpdate = param->hessUpdate;
        rest_opts->hessLimMem = 1;
        rest_opts->hessUpdate = 2;
        rest_opts->hessScaling = 4;
        rest_opts->maxConvQP = param->maxConvQP;
        rest_opts->opttol = param->opttol;
        rest_opts->nlinfeastol = param->nlinfeastol;
        rest_opts->QPsol = param->QPsol;
        rest_opts->QPsol_opts = param->QPsol_opts;
        
        //rest_opts->autoScaling = param->autoScaling;
        
        rest_prob = nullptr;
        rest_stats = nullptr;
        rest_method = nullptr;
    }
    else{
        rest_vblocks = nullptr;
        rest_cblocks = nullptr;
        rest_h_sizes = nullptr;
        rest_targets = nullptr;
        rest_cond = nullptr;
        rest_opts = nullptr;

        rest_prob = nullptr;
        rest_stats = nullptr;
        rest_method = nullptr;
    }
}

SCQPmethod::SCQPmethod(): cond(nullptr), rest_cond(nullptr), rest_vblocks(nullptr), rest_cblocks(nullptr), rest_h_sizes(nullptr), rest_targets(nullptr)
{};

SCQPmethod::~SCQPmethod(){
    delete[] rest_vblocks;
    delete[] rest_cblocks;
    delete[] rest_h_sizes;
    delete[] rest_targets;
    delete rest_cond;
}


SCQP_bound_method::SCQP_bound_method(Problemspec *problem, SQPoptions *parameters, SQPstats *statistics, Condenser *CND){
    cond = CND;
    if (cond->add_dep_bounds != 1){
        std::cout << "SCQP_bound_method: Condenser needs to add inactive dependent variable bounds, changing condenser add_dep_bound option to 1\n";
        cond->set_dep_bound_handling(1);
    }

    prob = problem;
    param = parameters; param->optionsConsistency();
    stats = statistics;

    if (param->autoScaling){
        scaled_prob = new scaled_Problemspec(problem);
        prob = scaled_prob;
    }
    else{
        scaled_prob = nullptr;
        prob = problem;
    }
    vars = new SCQPiterate(prob, param, cond, true);

    // Check if there are options that are infeasible and set defaults accordingly
    if (param->sparseQP == 0){
        throw std::invalid_argument("SCQPmethod: Error, condensing only works with sparse QPs");
    }
    if (param->blockHess != 1){
        throw std::invalid_argument("SCQPmethod: Error, condensing requires block diagonal hessian for efficient linear algebra");
    }

    sub_QP = create_QPsolver(cond->condensed_num_vars, cond->condensed_num_cons, cond->condensed_num_hessblocks, param);

    initCalled = false;

    if (param->restoreFeas){
        //Setup condenser for the restoration problem
        int N_vblocks = cond->num_vblocks + cond->num_true_cons;
        int N_cblocks = cond->num_cblocks;
        int N_hessblocks = cond->num_hessblocks + cond->num_true_cons;
        int N_targets = cond->num_targets;

        rest_vblocks = new vblock[N_vblocks];
        rest_cblocks = new cblock[N_cblocks];
        rest_h_sizes = new int[N_hessblocks];
        rest_targets = new condensing_target[N_targets];

        for (int i = 0; i<cond->num_vblocks; i++){
            rest_vblocks[i] = cond->vblocks[i];
        }
        for (int i = cond->num_vblocks; i < N_vblocks; i++){
            rest_vblocks[i] = vblock(1, false);
        }

        for (int i = 0; i<cond->num_cblocks; i++){
            rest_cblocks[i] = cond->cblocks[i];
        }

        for (int i = 0; i<cond->num_hessblocks; i++){
            rest_h_sizes[i] = cond->hess_block_sizes[i];
        }
        for (int i = cond->num_hessblocks; i<N_hessblocks; i++){
            rest_h_sizes[i] = 1;
        }

        for (int i = 0; i<cond->num_targets; i++){
            rest_targets[i] = cond->targets[i];
        }
        rest_cond = new Condenser(rest_vblocks, N_vblocks, rest_cblocks, N_cblocks, rest_h_sizes, N_hessblocks, rest_targets, N_targets, 0);

        //Setup options for the restoration problem
        rest_opts = new SQPoptions();
        rest_opts->globalization = 1;
        rest_opts->whichSecondDerv = 0;
        rest_opts->restoreFeas = 0;
        //rest_opts->hessUpdate = param->hessUpdate;
        rest_opts->hessLimMem = 1;
        rest_opts->hessUpdate = 2;
        rest_opts->hessScaling = 4;
        rest_opts->maxConvQP = param->maxConvQP;
        rest_opts->opttol = param->opttol;
        rest_opts->nlinfeastol = param->nlinfeastol;
        rest_opts->QPsol = param->QPsol;

        //rest_opts->autoScaling = param->autoScaling;

        rest_prob = nullptr;
        rest_stats = nullptr;
        rest_method = nullptr;
    }
    else{
        rest_vblocks = nullptr;
        rest_cblocks = nullptr;
        rest_h_sizes = nullptr;
        rest_targets = nullptr;
        rest_cond = nullptr;
        rest_opts = nullptr;

        rest_prob = nullptr;
        rest_stats = nullptr;
        rest_method = nullptr;
    }
}


SCQP_correction_method::SCQP_correction_method(Problemspec *problem, SQPoptions *parameters, SQPstats *statistics, Condenser *CND){
    cond = CND;
    if (cond->add_dep_bounds > 0){
        std::cout << "Warning: Condenser adds dependent variable bounds to constraint matrix, performance may be impeded\n";
    }

    prob = problem;
    param = parameters; param->optionsConsistency();
    stats = statistics;

    if (param->autoScaling){
        scaled_prob = new scaled_Problemspec(problem);
        prob = scaled_prob;
    }
    else{
        scaled_prob = nullptr;
        prob = problem;
    }
    vars = new SCQP_correction_iterate(prob, param, cond, true);

    // Check if there are options that are infeasible and set defaults accordingly
    if (param->sparseQP == 0){
        throw std::invalid_argument("SCQPmethod: Error, condensing only works with sparse QPs");
    }
    if (param->blockHess != 1){
        throw std::invalid_argument("SCQPmethod: Error, condensing requires block diagonal hessian for efficient linear algebra");
    }

    sub_QP = create_QPsolver(cond->condensed_num_vars, cond->condensed_num_cons, cond->condensed_num_hessblocks, param);

    initCalled = false;

    corrections = new Matrix[cond->num_targets];
    SOC_corrections = new Matrix[cond->num_targets];
    for (int tnum = 0; tnum < cond->num_targets; tnum++){
        corrections[tnum].Dimension(cond->targets_data[tnum].n_dep).Initialize(0.);
        SOC_corrections[tnum].Dimension(cond->targets_data[tnum].n_dep).Initialize(0.);
    }

    if (param->restoreFeas){
        //Setup condenser for the restoration problem
        int N_vblocks = cond->num_vblocks + cond->num_true_cons;
        int N_cblocks = cond->num_cblocks;
        int N_hessblocks = cond->num_hessblocks + cond->num_true_cons;
        int N_targets = cond->num_targets;

        rest_vblocks = new vblock[N_vblocks];
        rest_cblocks = new cblock[N_cblocks];
        rest_h_sizes = new int[N_hessblocks];
        rest_targets = new condensing_target[N_targets];

        for (int i = 0; i<cond->num_vblocks; i++){
            rest_vblocks[i] = cond->vblocks[i];
        }
        for (int i = cond->num_vblocks; i < N_vblocks; i++){
            rest_vblocks[i] = vblock(1, false);
        }

        for (int i = 0; i<cond->num_cblocks; i++){
            rest_cblocks[i] = cond->cblocks[i];
        }

        for (int i = 0; i<cond->num_hessblocks; i++){
            rest_h_sizes[i] = cond->hess_block_sizes[i];
        }
        for (int i = cond->num_hessblocks; i<N_hessblocks; i++){
            rest_h_sizes[i] = 1;
        }

        for (int i = 0; i<cond->num_targets; i++){
            rest_targets[i] = cond->targets[i];
        }
        rest_cond = new Condenser(rest_vblocks, N_vblocks, rest_cblocks, N_cblocks, rest_h_sizes, N_hessblocks, rest_targets, N_targets, 0);

        //Setup options for the restoration problem
        rest_opts = new SQPoptions();
        rest_opts->globalization = 1;
        rest_opts->whichSecondDerv = 0;
        rest_opts->restoreFeas = 0;
        rest_opts->hessLimMem = 1;
        rest_opts->hessUpdate = 2;
        rest_opts->hessScaling = 4;
        rest_opts->maxConvQP = param->maxConvQP;
        rest_opts->opttol = param->opttol;
        rest_opts->nlinfeastol = param->nlinfeastol;
        rest_opts->QPsol = param->QPsol;
        rest_opts->QPsol_opts = param->QPsol_opts;
        rest_opts->max_correction_steps = param->max_correction_steps;
        
        //rest_opts->autoScaling = param->autoScaling;

        rest_prob = nullptr;
        rest_stats = nullptr;
        rest_method = nullptr;
    }
    else{
        rest_vblocks = nullptr;
        rest_cblocks = nullptr;
        rest_h_sizes = nullptr;
        rest_targets = nullptr;
        rest_cond = nullptr;
        rest_opts = nullptr;

        rest_prob = nullptr;
        rest_stats = nullptr;
        rest_method = nullptr;
    }
}


SCQP_correction_method::~SCQP_correction_method(){
    delete[] corrections;
    delete[] SOC_corrections;
}





} // namespace blockSQP
