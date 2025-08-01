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
using namespace std::chrono;

namespace blockSQP{

void SQPmethod::init(){
    // Print header and information about the algorithmic parameters
    printInfo( param->print_level );

    // Open output files
    stats->initStats( param );
    vars->initIterate( param );

    // Initialize filter with pair ( maxConstrViolation, objLowerBound )
    initializeFilter();

    // Set initial values for all xi and set the Jacobian for linear constraints
    if( param->sparse )
        prob->initialize(vars->xi, vars->lambda, vars->sparse_constrJac.nz.get(), vars->sparse_constrJac.row.get(), vars->sparse_constrJac.colind.get());
    else
        prob->initialize(vars->xi, vars->lambda, vars->constrJac);

    initCalled = true;
}


SQPresult SQPmethod::run(int maxIt, int warmStart){
    int it = 0, infoQP = 0, infoEval = 0;
    bool skipLineSearch = false;
    bool hasConverged = false;
    int whichDerv = param->exact_hess;
    int n_convShift;

    if (!initCalled){
        printf("init() must be called before run(). Aborting.\n");
        //return -1;
        return print_SQPresult(SQPresult::misc_error, param->result_print_color);
    }
    
    if (warmStart == 0 || stats->itCount == 0){
        // SQP iteration 0
        if (param->sparse)
            prob->evaluate(vars->xi, vars->lambda, &vars->obj, vars->constr, vars->gradObj,
                            vars->sparse_constrJac.nz.get(), vars->sparse_constrJac.row.get(), vars->sparse_constrJac.colind.get(), vars->hess1.get(), 1+whichDerv, &infoEval);
        else
            prob->evaluate(vars->xi, vars->lambda, &vars->obj, vars->constr, vars->gradObj,
                            vars->constrJac, vars->hess1.get(), 1+whichDerv, &infoEval);
        stats->nDerCalls++;
        
        /// Check if converged
        hasConverged = calcOptTol();
        stats->printProgress( prob, vars.get(), param, hasConverged );
        if (hasConverged) return print_SQPresult(SQPresult::success, param->result_print_color);

        /// Set initial Hessian approximation
        //Consider implementing strategy for the initial hessian, see e.g. Leineweber 1995 Theory of MUSCOD S. 72

        calcInitialHessians();
        vars->hess2_updated = true;
    }


    for (; it<maxIt; it++){
        //Enter new iteration
        stats->itCount++;

        /////////////////////////////////////////////
        ///PHASE 1: Solve the quadratic subproblem///
        /////////////////////////////////////////////
        
        /// Solve QP subproblem with qpOASES or QPOPT
        if (!param->par_QPs){
            infoQP = solveQP(vars->deltaXi, vars->lambdaQP, int(vars->conv_qp_only));
        }
        else{
            //if (stats->itCount > 1) infoQP = solveQP_par(vars->deltaXi, vars->lambdaQP);
            if (stats->itCount > param->test_opt_2) infoQP = solveQP_par(vars->deltaXi, vars->lambdaQP);
            //else infoQP = solve_initial_QP_par(vars->deltaXi, vars->lambdaQP);
            else{
                infoQP = solve_convex_QP_par(vars->deltaXi, vars->lambdaQP);
            }
        }
        
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

                if (qpError && param->enable_rest && vars->cNorm > 0.01 * param->feas_tol){
                    std::cout << "Start feasibility restoration phase\n";
                    qpError = feasibilityRestorationPhase();
                    vars->steptype = 3;
                }

                if (qpError){
                    std::cout << "QP error, stop\n";
                    return print_SQPresult(SQPresult::qp_failure, param->result_print_color);
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
                return print_SQPresult(SQPresult::qp_failure, param->result_print_color);
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
            if (feasError && param->enable_rest && vars->cNorm > 0.01 * param->feas_tol){
                printf("***Start feasibility restoration phase.***\n");
                feasError = feasibilityRestorationPhase();
                vars->steptype = 3;
            }
            
            // If everything failed, abort.
            if (feasError == 1 || feasError > 2) return print_SQPresult(SQPresult::restoration_failure, param->result_print_color);
            else if (feasError == 2) return print_SQPresult(SQPresult::local_infeasibility, param->result_print_color);
        }

        /////////////////////////////////////////////////////////////
        ///PHASE 2: Do the filter line search (+ failure handling)///
        /////////////////////////////////////////////////////////////

        /// Determine steplength alpha
        if (param->enable_linesearch == 0 || (param->skip_first_linesearch && stats->itCount == 1)){
            // No enable_linesearch strategy, but reduce step if function cannot be evaluated
            if (fullstep()){
                printf( "***Constraint or objective could not be evaluated at new point. Stop.***\n" );
                return print_SQPresult(SQPresult::eval_failure, param->result_print_color);
            }
            vars->steptype = 0;
        }
        else if (param->enable_linesearch == 1 && !skipLineSearch){
            // Filter line search based on Waechter et al., 2006 (Ipopt paper)
            if (filterLineSearch() || vars->reducedStepCount > param->max_consec_reduced_steps){
                // Filter line search did not produce a step. Now there are a few things we can try ...
                bool lsError = true;
                
                std::cout << "Filter line search failed, begin handling\n";

                //If we already found a solution and steps are only for improving accuracy, terminate.
                if (vars->solution_found){
                    vars->restore_iterate();
                    if (vars->tol <= 1e-2*param->opt_tol && vars->cNormS <= 1e-2*param->feas_tol) return print_SQPresult(SQPresult::super_success, param->result_print_color);
                    else return print_SQPresult(SQPresult::success, param->result_print_color);
                }
                
                if (vars->KKT_heuristic_enabled){
                    std::cout << "filterLineSearch failed, try to reduce kktError\n";
                    vars->KKTerror_save = vars->tol;
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

                //if (lsError && vars->tol <= 1e2*param->opttol && vars->cNormS <= param->feas_tol && vars->remaining_filter_overrides > 0){
                if (lsError && vars->tol <= std::pow(param->opt_tol, 2./3.) && vars->cNormS <= param->feas_tol && vars->remaining_filter_overrides > 0){
                    force_accept(1.0);
                    vars->remaining_filter_overrides--;
                    lsError = false;
                    std::cout << "Filter line search failed close to a local solution, ignore filter. We can only do this " << vars->remaining_filter_overrides << " more times\n";
                    vars->steptype = -2;
                }

                ///If filter line search and first set of heuristics failed, check for feasibility and low KKT error. Declare partial success and terminate if true.
                if (param->enable_premature_termination && lsError && vars->cNormS <= param->feas_tol && vars->tol <= std::pow(param->opt_tol, 0.75))
                    return print_SQPresult(SQPresult::partial_success, param->result_print_color);

                // Heuristic 4: Try to reduce constraint violation by closing continuity gaps to produce an admissable iterate
                if (lsError && vars->cNorm > 0.01 * param->feas_tol && vars->steptype < 2){
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
                if (lsError && vars->cNorm > 0.01 * param->feas_tol && param->enable_rest){
                    printf("***Warning! Steplength too short. Start feasibility restoration phase.***\n");
                    // Solve NLP with minimum norm objective
                    lsError = bool(feasibilityRestorationPhase());
                    vars->steptype = 3;
                }

                // If everything failed, abort.
                if (lsError){
                    printf( "***Line search error. Stop.***\n" );
                    return print_SQPresult(SQPresult::linesearch_failure, param->result_print_color);
                }
            }
            else{
                vars->steptype = 0;
                vars->KKT_heuristic_enabled = true;
            }
        }
        
        ////////////////////////////////////
        ///PHASE 3: Update iteration data///
        ////////////////////////////////////

        /// Calculate "old" Lagrange gradient: gamma = dL(xi_k, lambda_k+1)
        calcLagrangeGradient( vars->gamma, 0 );

        /// Evaluate functions and gradients at the new xi
        if (param->sparse){
            prob->evaluate(vars->xi, vars->lambda, &vars->obj, vars->constr, vars->gradObj,
                            vars->sparse_constrJac.nz.get(), vars->sparse_constrJac.row.get(), vars->sparse_constrJac.colind.get(), vars->hess1.get(), 1+whichDerv, &infoEval);
        }
        else
            prob->evaluate(vars->xi, vars->lambda, &vars->obj, vars->constr, vars->gradObj,
                            vars->constrJac, vars->hess1.get(), 1+whichDerv, &infoEval);
        stats->nDerCalls++;

        /// Check if converged
        hasConverged = calcOptTol();

        /// Calculate difference of old and new Lagrange gradient: gamma = -gamma + dL(xi_k+1, lambda_k+1)
        calcLagrangeGradient(vars->gamma, 1);
        
        stats->printProgress(prob, vars.get(), param, false);
        

        ///Decide wether it is time to terminate///
        //1. Check if termination criteria are satisfied. Either terminate or enter extra step phase
        if (hasConverged && vars->steptype < 2){
            if (param->max_extra_steps > 0){
                if (!vars->solution_found){
                    vars->save_iterate();
                    vars->solution_found = true;
                }
            }
            else return print_SQPresult(SQPresult::success, param->result_print_color);
            //return RES::SUCCESS; //Convergence achieved!
        }

        //Handle extra steps for improved accuracy if requested
        if (vars->solution_found && param->max_extra_steps > 0){
            if (vars->n_extra >= param->max_extra_steps){
                vars->restore_iterate();
                if (vars->tol < 1e-2*param->opt_tol && vars->cNormS < 1e-2*param->feas_tol) return print_SQPresult(SQPresult::super_success, param->result_print_color);
                else return print_SQPresult(SQPresult::success, param->result_print_color);
            }
            //Save current point if it is better in terms of constraint violation and KKT error
            if (std::max(vars->tol/param->opt_tol, vars->cNormS/param->feas_tol) < std::max(vars->tolOpt_save/param->opt_tol, vars->cNormSOpt_save/param->feas_tol))
                vars->save_iterate();
            vars->n_extra++;
        }
        ///No termination at this point, proceed///

        // Check if KKT error was indeed reduced, if not, disable KKT heuristic until next successful linesearch
        if (vars->steptype == -1){
            if (!(vars->tol < param->kappaF*vars->KKTerror_save)){
                std::cout << "KKT error was not sufficiently reduced, disable KKT heuristic\n";
                vars->KKT_heuristic_enabled = false;
            }
            else std::cout << "KKT heuristic successful\n";
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
                vars->nquasi[ind] += int(vars->nquasi[ind] < param->mem_size);
            }
            vars->n_scaleIt += int(vars->n_scaleIt < vars->dg_nsave);
        }

        //Rescale variables if automatic scaling is enabled. This has to be done before limited memory quasi newton updates are applied.
        if (param->automatic_scaling) scaling_heuristic();

        ///
        ///PHASE 3.5: Update the Hessian 'approximations' and related data///
        ///

        if (param->lim_mem){
            //Subvectors deltaNorm and deltaGamma will be updated as needed when calculating the hessian approximation
            //Skip update for the indefinite hessian when we only solve convex QPs. Delay update for convex hessian when we try indefinite Hessian first
            if (vars->conv_qp_only && vars->hess2 != nullptr){
                if (param->fallback_approx <= 2)
                    calcHessianUpdateLimitedMemory_par(param->fallback_approx, param->fallback_sizing, vars->hess2.get());
                vars->hess2_updated = true;
            }
            else{
                if (param->exact_hess < 2){
                    //Calculate/Update first (pos. indefinite) hessian
                    if (param->hess_approx <= 2 || param->hess_approx > 6)
                        calcHessianUpdateLimitedMemory_par(param->hess_approx, param->sizing, vars->hess1.get());
                    else if (param->hess_approx == 4)
                        calcFiniteDiffHessian(vars->hess1.get());
                    vars->hess2_updated = false;
                }
                vars->hess2_updated = false;
            }
        }
        else{
            //Vectors deltaXi and gamma need not be updated when previous steps are not stored and can be overwritten. We also don't need to store their current position.
            if (param->exact_hess < 2){
                if (param->hess_approx <= 2)
                    calcHessianUpdate(param->hess_approx, param->sizing, vars->hess1.get());
                else if (param->hess_approx == 4)
                    calcFiniteDiffHessian(vars->hess1.get());
            }

            //Also update the fallback hessian as we need to update it in every iteration regardless of whether it is needed
            if (vars->hess2 != nullptr){
                if (param->fallback_approx <= 2)
                    calcHessianUpdate(param->fallback_approx, param->fallback_sizing, vars->hess2.get());
                vars->hess2_updated = true;
            }
        }

        //Adjust scaling factor if indefinite hessians are attempted to be convexified by adding scaled identities
        //if (param->conv_strategy >= 1 && param->max_conv_QPs > 1 && vars->steptype == 0 && stats->itCount > 1 && !vars->conv_qp_only){
        if (param->conv_strategy >= 1 && param->max_conv_QPs > 1 && vars->steptype == 0 && stats->itCount > param->test_opt_2 && !vars->conv_qp_only){
            if (param->max_conv_QPs > 2){
                //If more than one convexified indefinite QP is tried, shift convexification factor of the successful QP to the last attempted convexified QP.
                //If more than two convexified indefinite QPs are tried and none were accepted, shift last factor to first factor.
                n_convShift = vars->hess_num_accepted - param->max_conv_QPs + 1 + (param->max_conv_QPs - 3) * int(vars->hess_num_accepted == param->max_conv_QPs);
                vars->convKappa *= std::pow(2, n_convShift);
            }
            else{
                //If only one convexified indefinite QP is tried, increase convexification factor if it was rejected, decrease it if it was accepted.
                if (vars->hess_num_accepted == param->max_conv_QPs) vars->convKappa *= 2;
                else vars->convKappa *= 0.5;
            }
            if (vars->convKappa > param->conv_kappa_max) vars->convKappa = param->conv_kappa_max;
        }
        //The scaling factor adjustment in one line of code
        //vars->convKappa = std::min(1.0e2, vars->convKappa*std::pow(2, (vars->hess_num_accepted - param->max_conv_QPs + 1 + (param->max_conv_QPs - 3) * (vars->hess_num_accepted == param->max_conv_QPs))*(param->max_conv_QPs > 2) + (1 - 2*(vars->hess_num_accepted < param->max_conv_QPs))*(param->max_conv_QPs <= 2))) * (param->conv_strategy >= 1 && vars->hess_num_accepted > 0 && vars->steptype == 0) + vars->convKappa * (param->conv_strategy < 1 || vars->hess_num_accepted == 0 || vars->steptype != 0);
        vars->hess = vars->hess1.get();

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



} // namespace blockSQP
