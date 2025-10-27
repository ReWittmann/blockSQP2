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
 * \file blocksqp_qp.cpp
 * \author Dennis Janka
 * \date 2012-2015
 *
 *  Implementation of methods of SQPmethod class associated with
 *  solution of the quadratic subproblems.
 * 
 * \modifications
 *  \author Reinhold Wittmann
 *  \date 2023-2025
 */

#include "blocksqp_iterate.hpp"
#include "blocksqp_options.hpp"
#include "blocksqp_stats.hpp"
#include "blocksqp_method.hpp"
#include "blocksqp_general_purpose.hpp"
#include "blocksqp_defs.hpp"
#include <iostream>
#include <chrono>
#include <fstream>
#include <cmath>
#include <thread>
#include <functional>

using namespace std::chrono;

namespace blockSQP
{

void SQPmethod::computeNextHessian(int idx, int maxQP){
    double idScale;
    // Compute fallback update only once
    if ((idx == 1 && param->conv_strategy == 0) || (idx == maxQP - 1 && param->conv_strategy > 0)){
    //if ((idx == 1 && (param->conv_strategy == 0 || param->conv_strategy == 3)) || (idx == maxQP - 1 && param->conv_strategy > 0)){
        // If last block contains exact Hessian, we need to copy it
        if (param->exact_hess == 1)
            for (int i=0; i<vars->hess[vars->nBlocks-1].m; i++)
                for (int j=i; j<vars->hess[vars->nBlocks-1].m; j++)
                    vars->hess2[vars->nBlocks-1]( i,j ) = vars->hess1[vars->nBlocks-1]( i,j );

        // Limited memory: compute fallback update only when needed
        if (param->lim_mem && !vars->hess2_updated){
            if (param->fallback_approx <= 2){
                calcHessianUpdateLimitedMemory_par( param->fallback_approx, param->fallback_sizing, vars->hess2.get());
            }
            vars->hess2_updated = true;
        }
    }
    
    // 'Nontrivial' convex combinations
    if (maxQP > 2 && idx < maxQP - 1){
        //Store convex combination in vars->hess_conv, to avoid having to restore the second Hessian if full memory updates are used
        if (param->conv_strategy == 0){
            for (int i = 0; i < vars->nBlocks; i++){
                vars->hess_conv[i] = vars->hess1[i] * (1 - static_cast<double>(idx)/static_cast<double>(maxQP - 1)) + vars->hess2[i] * (static_cast<double>(idx)/static_cast<double>(maxQP - 1));
            }
        }
        else if (param->conv_strategy == 1){
            if (idx == 1){
                //Copy the first Hessian to reserved space
                for (int i = 0; i < vars->nBlocks; i++){
                    vars->hess_conv[i] = vars->hess1[i];
                }
            }

            //Regularize intermediate Hessian by adding scaled identity
            //idScale = vars->convKappa * std::pow(2, idx - maxQP + 2) * (1.0 - 0.5*(idx == 1));
            idScale = vars->convKappa * std::pow(2, idx - maxQP + 2) * (1.0 - 0.5*(idx > 1));
            for (int i = 0; i < vars->nBlocks; i++){
                for (int j = 0; j < vars->blockIdx[i+1] - vars->blockIdx[i]; j++){
                    vars->hess_conv[i](j,j) += idScale;
                }
            }
        }
        else if (param->conv_strategy == 2){
            if (idx == 1){
                //Copy the first Hessian to reserved space
                for (int i = 0; i < vars->nBlocks; i++){
                    vars->hess_conv[i] = vars->hess1[i];
                }
            }

            //Regularize intermediate Hessian by adding scaled identity to free components
            idScale = vars->convKappa * std::pow(2, idx - maxQP + 2) * (1.0 - 0.5*(idx > 1));
            int ind_b = 0, offset = 0, ind_1 = 0;
            for (int k = 0; k < prob->n_vblocks; k++){
                for (int i = 0; i < prob->vblocks[k].size; i++){
                    if (ind_1 + i == vars->blockIdx[ind_b + 1]){
                        ind_b += 1;
                        offset = ind_1 + i;
                    }
                    if (!prob->vblocks[k].dependent){
                        vars->hess_conv[ind_b](ind_1 + i - offset, ind_1 + i - offset) += idScale;
                    }
                }
                ind_1 += prob->vblocks[k].size;
            }
        }/*
        else if (param->conv_strategy == 3){
            idScale = vars->convKappa * std::pow(2, idx - maxQP + 2) * (1.0 - 0.5*(idx > 1));
            for (int k = 0; k < vars->nBlocks; k++){
                vars->hess_conv[k] = vars->hess1[k] + vars->hess2[k] * idScale;
            }
        }*/
        
        vars->hess = vars->hess_conv.get();
    }
    else{
        vars->hess = vars->hess2.get();
    }
}


void SQPmethod::computeConvexHessian(){
    if (vars->hess2 == nullptr){
        vars->hess = vars->hess1.get();
        return;
    }
    
    vars->hess = vars->hess2.get();
    // If last block contains exact Hessian block, we need to copy it
    if (param->exact_hess == 1)
        for (int i = 0; i < vars->hess[vars->nBlocks-1].m; i++)
            for (int j = i; j < vars->hess[vars->nBlocks-1].m; j++)
                vars->hess2[vars->nBlocks-1](i,j) = vars->hess1[vars->nBlocks - 1](i,j);

    if (!vars->hess2_updated){
        // Limited memory: compute fallback update only when needed
        if (param->lim_mem){
            calcHessianUpdateLimitedMemory_par(param->fallback_approx, param->fallback_sizing, vars->hess2.get());
        }
        vars->hess2_updated = true;
    }
    return;
}



void SQPmethod::setIdentityHessian(){
    calcInitialHessian(vars->hess_conv.get());
    vars->hess = vars->hess_conv.get();
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


QPresult SQPmethod::solveQP(Matrix &deltaXi, Matrix &lambdaQP, int hess_type){
    bool QP_loop_active = (
                          (param->exact_hess > 0 || param->hess_approx == 1 || param->hess_approx == 4 || param->hess_approx == 6 || param->hess_approx > 6)
                        && stats->itCount > param->indef_delay 
                        && !vars->conv_qp_only
                        && hess_type == 0)
                        ;
    if (QP_loop_active){
        if (param->par_QPs) return solveQP_par(deltaXi, lambdaQP);
        return solveQP_seq(deltaXi, lambdaQP);
    }
    
    return solve_convex_QP(deltaXi, lambdaQP, hess_type == 2, param->par_QPs ? sub_QPs_par[param->max_conv_QPs].get() : sub_QP.get());
}


QPresult SQPmethod::solve_convex_QP(Matrix &deltaXi, Matrix &lambdaQP, bool id_hess, QPsolverBase *QPS){
    if (id_hess) setIdentityHessian(); 
    else computeConvexHessian();
    
    QPS->set_hess(vars->hess, true);
    if (param->sparse)
        QPS->set_constr(vars->sparse_constrJac.nz.get(), vars->sparse_constrJac.row.get(), vars->sparse_constrJac.colind.get());
    else
        QPS->set_constr(vars->constrJac);
    
    updateStepBounds();
    QPS->set_bounds(vars->delta_lb_var, vars->delta_ub_var, vars->delta_lb_con, vars->delta_ub_con);
    QPS->set_lin(vars->gradObj);
    
    QPS->set_timeLimit(1);
    QPS->set_use_hotstart(vars->use_homotopy);
    
    QPresult QP_result = QPS->solve(deltaXi, lambdaQP);
    
    if (QP_result == QPresult::success){
        stats->qpIterations = QPS->get_QP_it();
        vars->conv_qp_solved = true;
        vars->hess_num_accepted = param->max_conv_QPs;
        vars->QP_num_accepted = param->max_conv_QPs;
        stats->qpResolve = 0;
    }
    return QP_result;
}


QPresult SQPmethod::solveQP_seq(Matrix &deltaXi, Matrix &lambdaQP){
    double s_indf_N, s_conv_N;
    
    int maxQP = param->max_conv_QPs + 1;    
    vars->conv_qp_solved = false;

    if (param->sparse)
        sub_QP->set_constr(vars->sparse_constrJac.nz.get(), vars->sparse_constrJac.row.get(), vars->sparse_constrJac.colind.get());
    else
        sub_QP->set_constr(vars->constrJac);
    
    updateStepBounds();
    sub_QP->set_bounds(vars->delta_lb_var, vars->delta_ub_var, vars->delta_lb_con, vars->delta_ub_con);
    sub_QP->set_lin(vars->gradObj);
    sub_QP->set_use_hotstart(vars->use_homotopy);
    
    QPresult QP_result, QP_result_conv;
    for (int l = 0; l < maxQP; l++){
        //Compute a new Hessian
        if (l > 0){
            //If the solution of the first QP was rejected, consider the second Hessian
            stats->qpResolve++;
            computeNextHessian(l, maxQP);
        }
        if (l == maxQP - 1){
            //Pass hessian and inform QP solver of supposed convexity
            sub_QP->set_hess(vars->hess, true, vars->modified_hess_regularizationFactor);
            sub_QP->set_timeLimit(1);
        }
        else{
            sub_QP->set_timeLimit(0);
            sub_QP->set_hess(vars->hess, false, 0);
        }
        
        QP_result = sub_QP->solve(deltaXi, lambdaQP);
        
        if (QP_result == QPresult::success){
            if (l == maxQP - 1)
                vars->conv_qp_solved = true;
            stats->qpIterations += sub_QP->get_QP_it();
            
            //Save the number of the first hessian for which the QP solved (even though the step may still be replaced by the step from the convex Hessian)
            vars->hess_num_accepted = l;
            vars->QP_num_accepted = l;
            
            //For regularized indefinite hessians, compare steplength to fallback hessian to avoid over-regularized hessians leading to small steps.
            //Skip this for the first regularization as this tends to help lock iterates down to a region of fast convergence.
            if (param->conv_strategy > 0 && l > 1 && l < maxQP - 1){
                computeConvexHessian();
                sub_QP->set_hess(vars->hess, true, vars->modified_hess_regularizationFactor);
                sub_QP->set_timeLimit(1);
                QP_result_conv = sub_QP->solve(vars->deltaXi_conv, vars->lambdaQP_conv);
                if (QP_result_conv == QPresult::success){
                    s_indf_N = l2VectorNorm(deltaXi);
                    s_conv_N = l2VectorNorm(vars->deltaXi_conv);
                    if (s_indf_N < param->conv_tau_H*s_conv_N){
                        deltaXi = vars->deltaXi_conv;
                        lambdaQP = vars->lambdaQP_conv;
                        vars->conv_qp_solved = true;
                        stats->qpResolve = maxQP - 1;
                        vars->QP_num_accepted = maxQP - 1;
                    }
                }
            }
            break; // Success!
        }
        stats->qpIterations2 += sub_QP->get_QP_it();
        stats->rejectedSR1++;
    } // End of QP solving loop

    return QP_result;
}


QPresult SQPmethod::solveQP_par(Matrix &deltaXi, Matrix &lambdaQP){
    int maxQP = param->max_conv_QPs + 1;
    steady_clock::time_point T0, T1;
    
    std::promise<QPresult> QP_results_p[PAR_QP_MAX];
    std::future<QPresult> QP_results_f[PAR_QP_MAX];
    for (int j = 0; j < maxQP - 1; j++) QP_results_f[j] = QP_results_p[j].get_future();
    
    std::future_status QP_results_fs[PAR_QP_MAX];
    QPresult QP_results[PAR_QP_MAX];
    for (int j = 0; j < maxQP; j++) QP_results[j] = QPresult::undef;
    
    // VERY important if using qpOASES solver, else SR1-QP always fails for some reason    
    for (int j = 0; j < vars->hess_num_accepted; j++){
        sub_QPs_par[j]->set_hotstart_point(sub_QPs_par[vars->hess_num_accepted].get());
    }
    
    updateStepBounds();
    for (int j = 0; j < maxQP; j++){
        
        if (j > 0) computeNextHessian(j, maxQP);
        
        if (param->sparse)
            sub_QPs_par[j]->set_constr(vars->sparse_constrJac.nz.get(), vars->sparse_constrJac.row.get(), vars->sparse_constrJac.colind.get());
        else
            sub_QPs_par[j]->set_constr(vars->constrJac);
        sub_QPs_par[j]->set_bounds(vars->delta_lb_var, vars->delta_ub_var, vars->delta_lb_con, vars->delta_ub_con);
        sub_QPs_par[j]->set_lin(vars->gradObj);
        sub_QPs_par[j]->set_use_hotstart(vars->use_homotopy);
        sub_QPs_par[j]->set_timeLimit(int(j == (maxQP - 1)));
        sub_QPs_par[j]->set_hess(vars->hess, j == (maxQP - 1));
        
        if (j < maxQP - 1){
            QP_threads[j] = std::jthread(
                [](std::stop_token stp, QPsolverBase *arg_QPS, std::promise<QPresult> arg_PRM, Matrix &arg_1, Matrix &arg_2){
                    arg_QPS->solve(stp, std::move(arg_PRM), arg_1, arg_2);
                },
                sub_QPs_par[j].get(), std::move(QP_results_p[j]), std::ref(vars->par_QP_sols_prim[j]), std::ref(vars->par_QP_sols_dual[j])
            );
        }
        else{
            //Record BFGS QP solution time to determine termination times for the other QPs
            T0 = steady_clock::now();
            QP_results[maxQP - 1] = sub_QPs_par[j]->solve(vars->par_QP_sols_prim[j], vars->par_QP_sols_dual[j]);
            T1 = steady_clock::now();
        }
    }
    
    // Set time at which still running QPs get terminated. 
    // To prevent too many "good" QPs from being terminated, increase allowed time with successive terminations
    //steady_clock::time_point TF = T1 + microseconds(duration_cast<microseconds>((T1 - T0)*(1 + vars->N_QP_cancels)).count() + 2000);
    
    steady_clock::time_point TF = T1 + microseconds(duration_cast<microseconds>((T1 - T0)*(1.0 + 0.5*vars->N_QP_cancels)).count() + 2000);
    
    bool QP_cancelled = false;
    if (param->enable_QP_cancellation){
        //SR1 QP almost always succeeds/fails fast, so join it. Wait + stop/join QPs until one successfully solved
        QP_threads[0].join();
        QP_results[0] = QP_results_f[0].get();
        if (QP_results[0] == QPresult::success){
            for (int k = 1; k < maxQP - 1; k++){
                QP_threads[k].request_stop();
            }
        }
        // Wait for convexified QPs from most convexified to least convexified. 
        // More convexified QPs almost always solve faster than less convexified QPs, so no significant time waste occurs
        else{
            for (int j = 1; j < maxQP - 1; j++){
                QP_results_fs[j] = QP_results_f[j].wait_until(TF);
                if (QP_results_fs[j] != std::future_status::ready){
                    QP_threads[j].request_stop();
                    QP_cancelled = true;
                }
                else{
                    QP_results[j] = QP_results_f[j].get();
                    if (QP_results[j] == QPresult::success){
                        for (int k = j + 1; k < maxQP - 1; k++){
                            QP_threads[k].request_stop();
                        }
                        break;
                    }
                }
            }
        }
        
        // Wait several times, check if QPs are solved, terminate all that are more convexified.
        // The simpler approach above seemed sufficient, so we use it instead
        /*
        else{
            int Nt = 4, k_QP_acc = maxQP - 1;
            for (int tk = 1; tk < Nt + 1; tk++){
                steady_clock::time_point TFk = T1 + QP_wait_time/Nt * tk;
                for (int j = 1; j < k_QP_acc; j++){
                    if (QP_results[j] > 0) continue;
                    
                    QP_results_fs[j] = QP_results_f[j].wait_until(TFk);
                    if (QP_results_fs[j] == std::future_status::ready){
                        QP_results[j] = QP_results_f[j].get();
                        if (QP_results[j] == 0){
                            k_QP_acc = j;
                            for (int k = j + 1; k < maxQP - 1; k++){
                                QP_threads[k].request_stop();
                            }
                            break;
                        }
                    }
                    else if (tk == Nt) QP_threads[j].request_stop();
                }
                if (k_QP_acc == 1) break;
            }
        }
        */
        
        for (int j = 1; j < maxQP - 1; j++)
            QP_threads[j].join();
    }
    else{
        for (int j = maxQP - 2; j>= 0; j--){
            QP_threads[j].join();
            QP_results[j] = QP_results_f[j].get();
        }
    }
    
    vars->hess_num_accepted = -1;
    stats->qpResolve = -1;
    for (int j = 0; j < maxQP; j++){
        if (QP_results[j] == QPresult::success){
            stats->qpResolve = j;
            vars->hess_num_accepted = j;
            break;
        }
    }
    
    // If no QP was accepted, return error of convex fallback QP
    if (vars->hess_num_accepted == -1)
        return QP_results[maxQP - 1];
    
    // Increment/reset counter of successive QP terminations
    vars->N_QP_cancels = (1 + vars->N_QP_cancels) * int(QP_cancelled);
    
    // If second or later convexified QP was accepted, compare step with that of the BFGS QP
    double s_indf_N = 1.0, s_conv_N = 0.0;
    if (vars->hess_num_accepted > 1 && vars->hess_num_accepted < maxQP - 1 && QP_results[maxQP - 1] == QPresult::success){
        s_indf_N = l2VectorNorm(vars->par_QP_sols_prim[vars->hess_num_accepted]);
        s_conv_N = l2VectorNorm(vars->par_QP_sols_prim[maxQP - 1]);
    }
    //Always true if above conditions are false
    if (s_indf_N >= param->conv_tau_H*s_conv_N){
        deltaXi = vars->par_QP_sols_prim[vars->hess_num_accepted];
        lambdaQP = vars->par_QP_sols_dual[vars->hess_num_accepted];
        vars->QP_num_accepted = vars->hess_num_accepted;
    }
    else{
        deltaXi = vars->par_QP_sols_prim[maxQP - 1];
        lambdaQP = vars->par_QP_sols_dual[maxQP - 1];
        stats->qpResolve = maxQP - 1;
        vars->QP_num_accepted = maxQP - 1;
    }
    
    vars->conv_qp_solved = (vars->QP_num_accepted == (maxQP - 1));
    stats->qpIterations = sub_QPs_par[vars->QP_num_accepted]->get_QP_it();
    stats->rejectedSR1 += vars->hess_num_accepted;
    return QPresult::success;
}


QPresult SQPmethod::solve_SOC_QP(Matrix &deltaXi, Matrix &lambdaQP){
    QPsolverBase *SOC_QP = param->par_QPs ? sub_QPs_par[vars->QP_num_accepted].get() : sub_QP.get();
    
    updateStepBoundsSOC();
    SOC_QP->set_bounds(vars->delta_lb_var, vars->delta_ub_var, vars->delta_lb_con, vars->delta_ub_con);

    QPresult QP_result = SOC_QP->solve(deltaXi, lambdaQP);
    stats->qpIterations += SOC_QP->get_QP_it();
    return QP_result;
}


///////////////////
//Sublass methods//
///////////////////

QPresult bound_correction_method::bound_correction(Matrix &deltaXi_corr, Matrix &lambdaQP_corr){
    QPsolverBase *correction_QP = param->par_QPs ? sub_QPs_par[vars->QP_num_accepted].get() : sub_QP.get();
    return static_cast<CQPsolver*>(correction_QP)->bound_correction(vars->xi, prob->lb_var, prob->ub_var, deltaXi_corr, lambdaQP_corr);
}


QPresult bound_correction_method::solve_SOC_QP(Matrix &deltaXi, Matrix &lambdaQP){
    QPsolverBase *SOC_QP = param->par_QPs ? sub_QPs_par[vars->QP_num_accepted].get() : sub_QP.get();
    
    updateStepBoundsSOC();
    SOC_QP->set_bounds(vars->delta_lb_var, vars->delta_ub_var, vars->delta_lb_con, vars->delta_ub_con);

    QPresult QP_result = static_cast<CQPsolver*>(SOC_QP)->correction_solve(deltaXi, lambdaQP);
    stats->qpIterations += SOC_QP->get_QP_it();
    return QP_result;
}

} // namespace blockSQP


