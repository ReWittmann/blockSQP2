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

void SQPmethod::computeNextHessian(int idx, int maxQP){
    // Compute fallback update only once
    if (idx == 1){
        // Switch storage
        vars->hess = vars->hess2;

        // If last block contains exact Hessian, we need to copy it
        if (param->whichSecondDerv == 1)
            for (int i=0; i<vars->hess[vars->nBlocks-1].m; i++)
                for (int j=i; j<vars->hess[vars->nBlocks-1].m; j++)
                    vars->hess2[vars->nBlocks-1]( i,j ) = vars->hess1[vars->nBlocks-1]( i,j );

        // Limited memory: compute fallback update only when needed
        if (param->hessLimMem && !vars->hess2_updated){
            if (param->fallbackUpdate <= 2)
                calcHessianUpdateLimitedMemory( param->fallbackUpdate, param->fallbackScaling, vars->hess2);
            vars->hess2_updated = true;
        }
        /* Full memory: both updates must be computed in every iteration
         * so switching storage is enough */
    }

    // 'Nontrivial' convex combinations
    if (maxQP > 2 && idx < maxQP - 1){
        //Store convex combination in vars->hess_alt, to avoid having to restore the second hessian if full memory updates are used
        for (int i = 0; i < vars->nBlocks; i++){
            vars->hess_alt[i] = vars->hess1[i] * (1 - static_cast<double>(idx)/static_cast<double>(maxQP - 1)) + vars->hess2[i] * (static_cast<double>(idx)/static_cast<double>(maxQP - 1));
        }
        vars->hess = vars->hess_alt;
    }
    else{
        vars->hess = vars->hess2;
    }
}


void SQPmethod::computeConvexHessian(){
    vars->hess = vars->hess2;

    // If last block contains exact Hessian, we need to copy it
    if (param->whichSecondDerv == 1)
        for (int i = 0; i < vars->hess[vars->nBlocks-1].m; i++)
            for (int j = i; j < vars->hess[vars->nBlocks-1].m; j++)
                vars->hess2[vars->nBlocks-1](i,j) = vars->hess1[vars->nBlocks - 1](i,j);

    if (!vars->hess2_updated){
        // Limited memory: compute fallback update only when needed
        if (param->hessLimMem){
            calcHessianUpdateLimitedMemory(param->fallbackUpdate, param->fallbackScaling, vars->hess2);
        }
        vars->hess2_updated = true;
    }
    return;
}


void SQPmethod::setIdentityHessian(){
    calcInitialHessian(vars->hess_alt);
    vars->hess = vars->hess_alt;
}



/**
 * Inner loop of SQP algorithm:
 * Solve a sequence of QPs until pos. def. assumption (G3*) is satisfied.
 */

int SQPmethod::solveQP(Matrix &deltaXi, Matrix &lambdaQP, int hess_type){

    int l, maxQP;
    if (param->globalization == 1 && (param->hessUpdate == 1 || param->hessUpdate == 4 || param->hessUpdate == 6) && stats->itCount > 1 && hess_type == 0)
        maxQP = param->maxConvQP + 1;
    else
        maxQP = 1;

    //Solve convex QP using fallback hessian if indefinite approximations are normally tried first.
    if (hess_type == 1 && (param->hessUpdate == 1 || param->hessUpdate == 4 || param->hessUpdate == 6))
        computeConvexHessian();

    if (hess_type >= 2)
        setIdentityHessian();

    vars->conv_qp_solved = false;

    if (param->sparseQP)
        sub_QP->set_constr(vars->jacNz, vars->jacIndRow, vars->jacIndCol);
    else
        sub_QP->set_constr(vars->constrJac);

    updateStepBounds();
    sub_QP->set_bounds(vars->delta_lb_var, vars->delta_ub_var, vars->delta_lb_con, vars->delta_ub_con);
    sub_QP->set_lin(vars->gradObj);

    sub_QP->use_hotstart = vars->use_homotopy;

    int QP_result;
    for (l = 0; l < maxQP; l++){
        //Compute a new Hessian
        if (l > 0){
            //If the solution of the first QP was rejected, consider the second Hessian
            stats->qpResolve++;
            computeNextHessian(l, maxQP);
        }

        if (l == maxQP - 1){
            //Pass hessian and inform QP solver of supposed convexity
            sub_QP->set_hess(vars->hess, true, vars->modified_hess_regularizationFactor);
            sub_QP->time_limit_type = 1;
        }
        else{
            sub_QP->time_limit_type = 0;
            sub_QP->set_hess(vars->hess, false, 0);
        }

        //Solve the QP
        QP_result = sub_QP->solve(deltaXi, lambdaQP);

        if (QP_result == 0){
            if (l == maxQP - 1)
                vars->conv_qp_solved = true;
            stats->qpIterations += sub_QP->get_QP_it();
            break; // Success!
        }
        stats->qpIterations2 += sub_QP->get_QP_it();
        stats->rejectedSR1++;
    } // End of QP solving loop

    // Point Hessian again to the first Hessian
    vars->hess = vars->hess1;

    return QP_result;
}


int SQPmethod::solve_SOC_QP(Matrix &deltaXi, Matrix &lambdaQP){

    updateStepBoundsSOC();

    sub_QP->set_bounds(vars->delta_lb_var, vars->delta_ub_var, vars->delta_lb_con, vars->delta_ub_con);
    //sub_QP->convex_QP = vars->conv_qp_solved;

    int QP_result;
    QP_result = sub_QP->solve(deltaXi, lambdaQP);

    // Point Hessian again to the first Hessian
    vars->hess = vars->hess1;

    stats->qpIterations += sub_QP->get_QP_it();
    return QP_result;
}


void SQPmethod::updateStepBounds(){
    int nVar = prob->nVar;
    int nCon = prob->nCon;

    // Bounds on step
    for (int i = 0; i < nVar; i++){
        if (prob->lb_var(i) > -param->inf)
            vars->delta_lb_var(i) = prob->lb_var(i) - vars->xi(i);
        else
            vars->delta_lb_var(i) = -param->inf;

        if (prob->ub_var(i) < param->inf)
            vars->delta_ub_var(i) = prob->ub_var(i) - vars->xi(i);
        else
            vars->delta_ub_var(i) = param->inf;
    }

    // Bounds on linearized constraints
    for (int i = 0; i < nCon; i++){
        if (prob->lb_con(i) > -param->inf)
            vars->delta_lb_con(i) = prob->lb_con(i) - vars->constr(i);
        else
            vars->delta_lb_con(i) = -param->inf;

        if (prob->ub_con(i) < param->inf)
            vars->delta_ub_con(i) = prob->ub_con(i) - vars->constr(i);
        else
            vars->delta_ub_con(i) = param->inf;
    }
}

void SQPmethod::updateStepBoundsSOC(){
    //  Constraint was evaluated at new potential iterate in filterLineSearch and saved in trialConstr
    //  Constraint jacobian times potential step was calculated in SOC loop method and saved in AdeltaXi
    //  This method is to be called right before solving/condensing the SOC QP
    for (int i = 0; i < prob->nCon; i++){
        if (prob->lb_con(i) > -param->inf)
            vars->delta_lb_con(i) = prob->lb_con(i) - vars->trialConstr(i) + vars->AdeltaXi(i);
        else
            vars->delta_lb_con(i) = -param->inf;

        if (prob->ub_con(i) < param->inf)
            vars->delta_ub_con(i) =  prob->ub_con(i) - vars->trialConstr(i) + vars->AdeltaXi(i);
        else
            vars->delta_ub_con(i) = param->inf;
    }
}


///////////////////////////////////////////////////Subclass methods


int SCQPmethod::solveQP(Matrix &deltaXi, Matrix &lambdaQP, int hess_type){

    SCQPiterate *c_vars = dynamic_cast<SCQPiterate*>(vars);

    int maxQP, l;
    if (param->globalization == 1 && (param->hessUpdate == 1 || param->hessUpdate == 4 || param->hessUpdate == 6) && stats->itCount > 1 && hess_type == 0)
        maxQP = param->maxConvQP + 1;
    else
        maxQP = 1;

    //hess_type 1: Solve convex QP using fallback hessian if indefinite approximations are normally tried first.
    if (hess_type == 1 && (param->hessUpdate == 1 || param->hessUpdate == 4 || param->hessUpdate == 6)){
        computeConvexHessian();
    }

    if (hess_type >= 2){
        setIdentityHessian();
    }

    vars->conv_qp_solved = false;

    updateStepBounds();
    cond->full_condense(c_vars->gradObj, c_vars->Jacobian, c_vars->hess,
            c_vars->delta_lb_var, c_vars->delta_ub_var, c_vars->delta_lb_con, c_vars->delta_ub_con,
                c_vars->condensed_h, c_vars->condensed_Jacobian, c_vars->condensed_hess, c_vars->condensed_lb_var, c_vars->condensed_ub_var, c_vars->condensed_lb_con, c_vars->condensed_ub_con);

    if (param->sparseQP)
        sub_QP->set_constr(c_vars->condensed_Jacobian.nz, c_vars->condensed_Jacobian.row, c_vars->condensed_Jacobian.colind);
    else
        sub_QP->set_constr(c_vars->constrJac);

    sub_QP->set_bounds(c_vars->condensed_lb_var, c_vars->condensed_ub_var, c_vars->condensed_lb_con, c_vars->condensed_ub_con);
    sub_QP->set_lin(c_vars->condensed_h);
    //sub_QP->set_hess(c_vars->condensed_hess);
    sub_QP->use_hotstart = vars->use_homotopy;

    int QP_result;
    for (l = 0; l < maxQP; l++){
        if (l > 0){
            // If the solution of the first QP was rejected, consider second Hessian
            stats->qpResolve++;
            computeNextHessian(l, maxQP);
            cond->new_hessian_condense(c_vars->hess, c_vars->condensed_h, c_vars->condensed_hess);
            sub_QP->set_lin(c_vars->condensed_h);
            //sub_QP->set_hess(c_vars->condensed_hess);
        }

        if (l == maxQP-1){
            //Inform QP solver about convexity
            //sub_QP->convex_QP = true;
            sub_QP->set_hess(c_vars->condensed_hess, true, vars->modified_hess_regularizationFactor);
            sub_QP->time_limit_type = 1;
        }
        else{
            sub_QP->set_hess(c_vars->condensed_hess, false, 0);
            sub_QP->time_limit_type = 0;
            }

        //Solve the QP
        QP_result = sub_QP->solve(c_vars->deltaXi_cond, c_vars->lambdaQP_cond);
        std::cout << "sub_QP->solve returned, QP_result is " << QP_result << "\n" << std::flush;

        if (QP_result == 0){
            if (l == maxQP - 1)
                vars->conv_qp_solved = true;
            cond->recover_var_mult(c_vars->deltaXi_cond, c_vars->lambdaQP_cond, deltaXi, lambdaQP);
            stats->qpIterations += sub_QP->get_QP_it();
            break; // Success!
        }
        stats->qpIterations2 += sub_QP->get_QP_it();
        stats->rejectedSR1++;
    } // End of QP solving loop

    // Point Hessian again to the first Hessian
    c_vars->hess = c_vars->hess1;
    return QP_result;
}


int SCQPmethod::solve_SOC_QP( Matrix &deltaXi, Matrix &lambdaQP){

    SCQPiterate *c_vars = dynamic_cast<SCQPiterate*>(vars);

    updateStepBoundsSOC();

    //Condense QP before invoking QP-solver
    cond->SOC_condense(c_vars->gradObj, c_vars->delta_lb_con, c_vars->delta_ub_con,
            c_vars->condensed_h, c_vars->condensed_lb_con, c_vars->condensed_ub_con);

    sub_QP->set_lin(c_vars->condensed_h);
    sub_QP->set_bounds(c_vars->condensed_lb_var, c_vars->condensed_ub_var, c_vars->condensed_lb_con, c_vars->condensed_ub_con);
    //sub_QP->convex_QP = c_vars->conv_qp_solved;

    int QP_result;
    QP_result = sub_QP->solve(c_vars->deltaXi_cond, c_vars->lambdaQP_cond);

    if (QP_result == 0){
        cond->recover_var_mult(c_vars->deltaXi_cond, c_vars->lambdaQP_cond, deltaXi, lambdaQP);
    }
    // Point Hessian again to the first Hessian
    c_vars->hess = c_vars->hess1;

    return QP_result;
}


int SCQP_bound_method::solveQP(Matrix &deltaXi, Matrix &lambdaQP, int hess_type){

    SCQPiterate *c_vars = dynamic_cast<SCQPiterate*>(vars);

    int maxQP, l;
    if (param->globalization == 1 && (param->hessUpdate == 1 || param->hessUpdate == 4 || param->hessUpdate == 6) && stats->itCount > 1 && hess_type == 0){
        maxQP = param->maxConvQP + 1;
    }
    else
        maxQP = 1;

    //Solve convex QP using fallback hessian if indefinite approximations are normally tried first.
    if (hess_type == 1 && (param->hessUpdate == 1 || param->hessUpdate == 4 || param->hessUpdate == 6)){
        computeConvexHessian();
    }

    if (hess_type >= 2){
        setIdentityHessian();
    }

    vars->conv_qp_solved = false;

    updateStepBounds();
    cond->full_condense(c_vars->gradObj, c_vars->Jacobian, c_vars->hess,
        c_vars->delta_lb_var, c_vars->delta_ub_var, c_vars->delta_lb_con, c_vars->delta_ub_con,
            c_vars->condensed_h, c_vars->condensed_Jacobian, c_vars->condensed_hess, c_vars->condensed_lb_var, c_vars->condensed_ub_var, c_vars->condensed_lb_con, c_vars->condensed_ub_con);

    if (param->sparseQP)
        sub_QP->set_constr(c_vars->condensed_Jacobian.nz, c_vars->condensed_Jacobian.row, c_vars->condensed_Jacobian.colind);
    else
        sub_QP->set_constr(c_vars->constrJac);

    sub_QP->set_bounds(c_vars->condensed_lb_var, c_vars->condensed_ub_var, c_vars->condensed_lb_con, c_vars->condensed_ub_con);
    sub_QP->set_lin(c_vars->condensed_h);
    //sub_QP->set_hess(c_vars->condensed_hess);
    sub_QP->use_hotstart = vars->use_homotopy;

    int QP_result;
    for (l = 0; l < maxQP; l++){
        if (l > 0){
            stats->qpResolve++;
            computeNextHessian(l, maxQP);
            cond->new_hessian_condense(c_vars->hess, c_vars->condensed_h, c_vars->condensed_hess);
            sub_QP->set_lin(c_vars->condensed_h);
            //sub_QP->set_hess(c_vars->condensed_hess);
        }

        if( l == maxQP-1 ){
            //Inform QP solver about convexity
            //sub_QP->convex_QP = true;
            sub_QP->set_hess(c_vars->condensed_hess, true, vars->modified_hess_regularizationFactor);
            sub_QP->time_limit_type = 1;
        }
        else{
            sub_QP->set_hess(c_vars->condensed_hess, false, 0);
            sub_QP->time_limit_type = 0;
        }

        QP_result = sub_QP->solve(c_vars->deltaXi_cond, c_vars->lambdaQP_cond);

        //Check if some dependent variable bounds are violated and - if they are - add them to the QP and solve again
        bool solution_found = false;
        if (QP_result == 0){
            solution_found = true;

            cond->recover_var_mult(c_vars->deltaXi_cond, c_vars->lambdaQP_cond, deltaXi, lambdaQP);

            stats->qpIterations += sub_QP->get_QP_it();

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

                sub_QP->set_bounds(c_vars->condensed_lb_var, c_vars->condensed_ub_var, c_vars->condensed_lb_con, c_vars->condensed_ub_con);
                sub_QP->record_time = false;
                sub_QP->time_limit_type = 0;

                std::chrono::steady_clock::time_point begin_ = std::chrono::steady_clock::now();
                QP_result = sub_QP->solve(c_vars->deltaXi_cond, c_vars->lambdaQP_cond);
                std::cout << "QP_result is " << QP_result << "\n";
                std::chrono::steady_clock::time_point end_ = std::chrono::steady_clock::now();
                std::cout << "Solved QP with added bounds in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin_).count() << "ms\n";

                if (QP_result == 1){
                    std::cout << "Solution of QP with added bounds is taking too long, initialize new QP\n";
                    sub_QP->use_hotstart = false;
                    sub_QP->time_limit_type = 1;
                    sub_QP->record_time = false;

                    begin_ = std::chrono::steady_clock::now();
                    QP_result = sub_QP->solve(c_vars->deltaXi_cond, c_vars->lambdaQP_cond);
                    end_ = std::chrono::steady_clock::now();
                    std::cout << "Finished solution initialized SQP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin_).count() << "ms\n";
                }

                if (QP_result == 0){
                    cond->recover_var_mult(c_vars->deltaXi_cond, c_vars->lambdaQP_cond, deltaXi, lambdaQP);
                }
                else{
                    std::cout << "Error in QP with added bounds, convexify the hessian further\n";
                    std::cout << "QP_result is " << QP_result << "\n";
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

        if (solution_found)
            break;

        stats->qpIterations2 += sub_QP->get_QP_it();
        stats->rejectedSR1++;
    } // End of QP solving loop

    if (QP_result == 0){
        if (l == maxQP - 1)
            vars->conv_qp_solved = true;
    }
    // Point Hessian again to the first Hessian
    c_vars->hess = c_vars->hess1;

    return QP_result;
}



int SCQP_bound_method::solve_SOC_QP( Matrix &deltaXi, Matrix &lambdaQP){

    SCQPiterate *c_vars = dynamic_cast<SCQPiterate*>(vars);

    updateStepBoundsSOC();

    cond->SOC_condense(c_vars->gradObj, c_vars->delta_lb_con, c_vars->delta_ub_con,
            c_vars->condensed_h, c_vars->condensed_lb_con, c_vars->condensed_ub_con);

    sub_QP->set_lin(c_vars->condensed_h);
    sub_QP->set_bounds(c_vars->condensed_lb_var, c_vars->condensed_ub_var, c_vars->condensed_lb_con, c_vars->condensed_ub_con);
    //sub_QP->convex_QP = c_vars->conv_qp_solved;

    int QP_result;
    QP_result = sub_QP->solve(c_vars->deltaXi_cond, c_vars->lambdaQP_cond);

    bool solution_found = false;
    if (QP_result == 0){
        solution_found = true;

        cond->recover_var_mult(c_vars->deltaXi_cond, c_vars->lambdaQP_cond, deltaXi, lambdaQP);

        bool found_direction;
        int ind;
        int ind_1;
        int ind_2;
        int ind_3;
        int num_dep_vars = cond->num_vars - cond->condensed_num_vars;
        int vio_count;

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

            sub_QP->set_bounds(c_vars->condensed_lb_var, c_vars->condensed_ub_var, c_vars->condensed_lb_con, c_vars->condensed_ub_con);

            std::chrono::steady_clock::time_point begin_ = std::chrono::steady_clock::now();
            QP_result = sub_QP->solve(c_vars->deltaXi_cond, c_vars->lambdaQP_cond);
            std::cout << "QP_result is " << QP_result << "\n";
            std::chrono::steady_clock::time_point end_ = std::chrono::steady_clock::now();
            std::cout << "Solved QP with added bounds in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin_).count() << "ms\n";

            if (QP_result == 1){
                std::cout << "Solution of QP with added bounds is taking too long, initialize new QP\n";
                sub_QP->use_hotstart = false;

                begin_ = std::chrono::steady_clock::now();
                QP_result = sub_QP->solve(c_vars->deltaXi_cond, c_vars->lambdaQP_cond);
                end_ = std::chrono::steady_clock::now();
                std::cout << "Finished solution initialized SQP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin_).count() << "ms\n";
            }
            if (QP_result == 0){
                cond->recover_var_mult(c_vars->deltaXi_cond, c_vars->lambdaQP_cond, deltaXi, lambdaQP);
            }
            else{
                std::cout << "Error in QP with added bounds, convexify the hessian further\n";
                std::cout << "QP_result is " << QP_result << "\n";
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

    // Point Hessian again to the first Hessian
    c_vars->hess = c_vars->hess1;

    return QP_result;
}



int SCQP_correction_method::solveQP(Matrix &deltaXi, Matrix &lambdaQP, int hess_type){

    SCQP_correction_iterate *c_vars = dynamic_cast<SCQP_correction_iterate*>(vars);

    int maxQP, l;
    if (param->globalization == 1 && (param->hessUpdate == 1 || param->hessUpdate == 4 || param->hessUpdate == 6) && stats->itCount > 1 && hess_type == 0){
        maxQP = param->maxConvQP + 1;
    }
    else
        maxQP = 1;

    //Solve convex QP using fallback hessian if indefinite approximations are normally tried first.
    if (hess_type == 1 && (param->hessUpdate == 1 || param->hessUpdate == 4 || param->hessUpdate == 6)){
        computeConvexHessian();
    }

    if (hess_type >= 2){
        setIdentityHessian();
    }

    vars->conv_qp_solved = false;

    updateStepBounds();
    cond->full_condense(c_vars->gradObj, c_vars->Jacobian, c_vars->hess,
        c_vars->delta_lb_var, c_vars->delta_ub_var, c_vars->delta_lb_con, c_vars->delta_ub_con,
            c_vars->condensed_h, c_vars->condensed_Jacobian, c_vars->condensed_hess, c_vars->condensed_lb_var, c_vars->condensed_ub_var, c_vars->condensed_lb_con, c_vars->condensed_ub_con);

    if (param->sparseQP)
        sub_QP->set_constr(c_vars->condensed_Jacobian.nz, c_vars->condensed_Jacobian.row, c_vars->condensed_Jacobian.colind);
    else
        sub_QP->set_constr(c_vars->constrJac);

    sub_QP->set_bounds(c_vars->condensed_lb_var, c_vars->condensed_ub_var, c_vars->condensed_lb_con, c_vars->condensed_ub_con);
    sub_QP->set_lin(c_vars->condensed_h);
    //sub_QP->set_hess(c_vars->condensed_hess);
    sub_QP->use_hotstart = vars->use_homotopy;

    int QP_result;
    for (l = 0; l < maxQP; l++){
        if (l > 0){
            stats->qpResolve++;
            computeNextHessian(l, maxQP);
            cond->new_hessian_condense(c_vars->hess, c_vars->condensed_h, c_vars->condensed_hess);
            sub_QP->set_lin(c_vars->condensed_h);
            //sub_QP->set_hess(c_vars->condensed_hess);
        }

        if (l == maxQP - 1){
            //Inform QP solver about convexity
            //sub_QP->convex_QP = true;
            sub_QP->set_hess(c_vars->condensed_hess, true, vars->modified_hess_regularizationFactor);
            sub_QP->time_limit_type = 1;
        }
        else{
            sub_QP->set_hess(c_vars->condensed_hess, false, 0);
            sub_QP->time_limit_type = 0;
        }

        QP_result = sub_QP->solve(c_vars->deltaXi_cond, c_vars->lambdaQP_cond);

        std::cout << "Solved QP in " << sub_QP->get_solutionTime() << "s\n";

        //Check if some dependent variable bounds are violated and - if they are - correct them in the QP and solve again
        bool solution_found = false;
        if (QP_result == 0){
            solution_found = true;

            cond->recover_var_mult(c_vars->deltaXi_cond, c_vars->lambdaQP_cond, deltaXi, lambdaQP);

            stats->qpIterations += sub_QP->get_QP_it();

            int ind_1, ind_2, ind, vio_count, max_vio_index;
            double max_dep_bound_violation;
            bool found_direction;
            Matrix xi_s(c_vars->xi);

            //Reset correction vectors
            for (int tnum = 0; tnum < cond->num_targets; tnum++){
                corrections[tnum].Initialize(0.);
            }

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
                    break;
                }
                std::cout << "Bounds violated by " << vio_count << " dependent variables, calculating correction vectors\n";
                std::cout << "Max dep bound violation is " << max_dep_bound_violation << "\n";

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
                            }
                            ind_1 += cond->vblocks[i].size;
                        }
                        ind_2 += cond->vblocks[i].size;
                    }
                }

                //Condense the QP, adding the correction to g = Gu + g
                cond->correction_condense(c_vars->gradObj, c_vars->delta_lb_con, c_vars->delta_ub_con, corrections, c_vars->corrected_h, c_vars->corrected_lb_con, c_vars->corrected_ub_con);

                sub_QP->set_bounds(c_vars->condensed_lb_var, c_vars->condensed_ub_var, c_vars->corrected_lb_con, c_vars->corrected_ub_con);
                sub_QP->set_lin(c_vars->corrected_h);

                sub_QP->record_time = false;
                sub_QP->time_limit_type = 0;

                std::chrono::steady_clock::time_point begin_ = std::chrono::steady_clock::now();
                QP_result = sub_QP->solve(c_vars->deltaXi_cond, c_vars->lambdaQP_cond);
                std::cout << "QP_result is " << QP_result << "\n";
                std::chrono::steady_clock::time_point end_ = std::chrono::steady_clock::now();
                std::cout << "Solved QP with added corrections in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin_).count() << "ms\n";

                if (QP_result == 1){
                    std::cout << "Solution of QP with corrections is taking too long, initialize new QP\n";
                    sub_QP->use_hotstart = false;
                    sub_QP->time_limit_type = 1;
                    sub_QP->record_time = false;

                    begin_ = std::chrono::steady_clock::now();
                    QP_result = sub_QP->solve(c_vars->deltaXi_cond, c_vars->lambdaQP_cond);
                    end_ = std::chrono::steady_clock::now();
                    std::cout << "Finished solution initialized SQP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin_).count() << "ms\n";
                }

                if (QP_result == 0){
                    cond->recover_correction_var_mult(c_vars->deltaXi_cond, c_vars->lambdaQP_cond, corrections, deltaXi, lambdaQP);
                }
                else{
                    std::cout << "Error in QP with corrections, convexify the hessian further\n";
                    std::cout << "QP_result is " << QP_result << "\n";
                    solution_found = false;
                    break;
                }
            }
        }
        if (solution_found) break;

        stats->qpIterations2 += sub_QP->get_QP_it();
        stats->rejectedSR1++;
    } // End of QP solving loop

    if (QP_result == 0){
        if (l == maxQP - 1)
            vars->conv_qp_solved = true;
    }
    // Point Hessian again to the first Hessian
    c_vars->hess = c_vars->hess1;

    return QP_result;
}


int SCQP_correction_method::solve_SOC_QP( Matrix &deltaXi, Matrix &lambdaQP){

    SCQP_correction_iterate *c_vars = dynamic_cast<SCQP_correction_iterate*>(vars);

    //Keep corrections from original QP for SOC QP, add additional correction as needed
    for (int tnum = 0; tnum < cond->num_targets; tnum++){
        SOC_corrections[tnum] = corrections[tnum];
    }

    updateStepBoundsSOC();

    //Condense QP before invoking QP-solver
    cond->correction_condense(c_vars->gradObj, c_vars->delta_lb_con, c_vars->delta_ub_con, corrections,
            c_vars->condensed_h, c_vars->condensed_lb_con, c_vars->condensed_ub_con);

    sub_QP->set_lin(c_vars->condensed_h);
    sub_QP->set_bounds(c_vars->condensed_lb_var, c_vars->condensed_ub_var, c_vars->condensed_lb_con, c_vars->condensed_ub_con);
    //sub_QP->convex_QP = c_vars->conv_qp_solved;

    int QP_result;
    QP_result = sub_QP->solve(c_vars->deltaXi_cond, c_vars->lambdaQP_cond);

    if (QP_result == 0){

        cond->recover_correction_var_mult(c_vars->deltaXi_cond, c_vars->lambdaQP_cond, corrections, deltaXi, lambdaQP);

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
            std::cout << "Max dep bound violation is " << max_dep_bound_violation << "\n";

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
                        }
                        ind_1 += cond->vblocks[i].size;
                    }
                    ind_2 += cond->vblocks[i].size;
                }
            }

            cond->correction_condense(c_vars->gradObj, c_vars->delta_lb_con, c_vars->delta_ub_con, SOC_corrections, c_vars->corrected_h, c_vars->corrected_lb_con, c_vars->corrected_ub_con);

            sub_QP->set_bounds(c_vars->condensed_lb_var, c_vars->condensed_ub_var, c_vars->corrected_lb_con, c_vars->corrected_ub_con);
            sub_QP->set_lin(c_vars->corrected_h);

            std::chrono::steady_clock::time_point begin_ = std::chrono::steady_clock::now();
            QP_result = sub_QP->solve(c_vars->deltaXi_cond, c_vars->lambdaQP_cond);
            std::cout << "QP_result is " << QP_result << "\n";
            std::chrono::steady_clock::time_point end_ = std::chrono::steady_clock::now();
            std::cout << "Solved SOC QP with added corrections in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin_).count() << "ms\n";

            if (QP_result == 1){
                std::cout << "Solution of QP with added corrections is taking too long, initialize new QP\n";
                sub_QP->use_hotstart = false;

                begin_ = std::chrono::steady_clock::now();
                QP_result = sub_QP->solve(c_vars->deltaXi_cond, c_vars->lambdaQP_cond);
                end_ = std::chrono::steady_clock::now();
                std::cout << "Finished solution initialized SQP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin_).count() << "ms\n";
            }
            if (QP_result == 0){
                cond->recover_correction_var_mult(c_vars->deltaXi_cond, c_vars->lambdaQP_cond, SOC_corrections, deltaXi, lambdaQP);
            }
            else break;
        }
    }
    // Point Hessian again to the first Hessian
    c_vars->hess = c_vars->hess1;
    return QP_result;
}






} // namespace blockSQP


