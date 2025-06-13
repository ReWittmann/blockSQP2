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
#include <thread>
#include <functional>

using namespace std::chrono;

namespace blockSQP
{

void SQPmethod::computeNextHessian(int idx, int maxQP){
    double idScale;
    // Compute fallback update only once
    if ((idx == 1 && param->conv_strategy == 0) || (idx == maxQP - 1 && param->conv_strategy > 0)){
        // If last block contains exact Hessian, we need to copy it
        if (param->exact_hess == 1)
            for (int i=0; i<vars->hess[vars->nBlocks-1].m; i++)
                for (int j=i; j<vars->hess[vars->nBlocks-1].m; j++)
                    vars->hess2[vars->nBlocks-1]( i,j ) = vars->hess1[vars->nBlocks-1]( i,j );

        // Limited memory: compute fallback update only when needed
        if (param->lim_mem && !vars->hess2_updated){
            if (param->fallback_approx <= 2){
                //steady_clock::time_point T0 = steady_clock::now();
                //calcHessianUpdateLimitedMemory( param->fallback_approx, param->fallback_sizing, vars->hess2.get());
                //steady_clock::time_point T1 = steady_clock::now();
                calcHessianUpdateLimitedMemory_par(param->fallback_approx, param->fallback_sizing, vars->hess2.get());
                //steady_clock::time_point T2 = steady_clock::now();
                //std::cout << "Lim mem update took " << duration_cast<microseconds>(T1 - T0).count() << "mus\n";
                //std::cout << "Lim mem par update took " << duration_cast<microseconds>(T2 - T1).count() << "mus\n";
            }
            vars->hess2_updated = true;
        }
    }

    // 'Nontrivial' convex combinations
    if (maxQP > 2 && idx < maxQP - 1){
        //Store convex combination in vars->hess_conv, to avoid having to restore the second hessian if full memory updates are used
        if (param->conv_strategy == 0){
            for (int i = 0; i < vars->nBlocks; i++){
                vars->hess_conv[i] = vars->hess1[i] * (1 - static_cast<double>(idx)/static_cast<double>(maxQP - 1)) + vars->hess2[i] * (static_cast<double>(idx)/static_cast<double>(maxQP - 1));
            }
        }
        else if (param->conv_strategy == 1){
            if (idx == 1){
                //Copy the first hessian to reserved space
                for (int i = 0; i < vars->nBlocks; i++){
                    vars->hess_conv[i] = vars->hess1[i];
                }
            }

            //Regularize intermediate hessian by adding scaled identity
            //idScale = vars->convKappa * std::pow(2, idx - maxQP + 2) * (1.0 - 0.5*(idx == 1));
            idScale = vars->convKappa * std::pow(2, idx - maxQP + 2) * (1.0 - 0.5*(idx > 1));
            //std::cout << "H idScale = " << idScale << ", convKappa = " << vars->convKappa << "\n";
            for (int i = 0; i < vars->nBlocks; i++){
                for (int j = 0; j < vars->blockIdx[i+1] - vars->blockIdx[i]; j++){
                    vars->hess_conv[i](j,j) += idScale;
                }
            }
        }
        else if (param->conv_strategy == 2){
            if (idx == 1){
                //Copy the first hessian to reserved space
                for (int i = 0; i < vars->nBlocks; i++){
                    vars->hess_conv[i] = vars->hess1[i];
                }
            }

            //Regularize intermediate hessian by adding scaled identity to free components
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
        }
        vars->hess = vars->hess_conv.get();
    }
    else{
        vars->hess = vars->hess2.get();
    }
}


void SQPmethod::computeConvexHessian(){
    vars->hess = vars->hess2.get();

    // If last block contains exact Hessian block, we need to copy it
    if (param->exact_hess == 1)
        for (int i = 0; i < vars->hess[vars->nBlocks-1].m; i++)
            for (int j = i; j < vars->hess[vars->nBlocks-1].m; j++)
                vars->hess2[vars->nBlocks-1](i,j) = vars->hess1[vars->nBlocks - 1](i,j);

    if (!vars->hess2_updated){
        // Limited memory: compute fallback update only when needed
        if (param->lim_mem){
            calcHessianUpdateLimitedMemory(param->fallback_approx, param->fallback_sizing, vars->hess2.get());
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



int SQPmethod::solve_initial_QP(Matrix &deltaXi, Matrix &lambdaQP){
    sub_QP->set_hess(vars->hess, true);
    if (param->sparse)
        sub_QP->set_constr(vars->sparse_constrJac.nz.get(), vars->sparse_constrJac.row.get(), vars->sparse_constrJac.colind.get());
    else
        sub_QP->set_constr(vars->constrJac);
    updateStepBounds();
    sub_QP->set_bounds(vars->delta_lb_var, vars->delta_ub_var, vars->delta_lb_con, vars->delta_ub_con);
    sub_QP->set_lin(vars->gradObj);
    int QP_result = sub_QP->solve(deltaXi, lambdaQP);
    stats->qpIterations2 += sub_QP->get_QP_it();
    vars->conv_qp_solved = true;
    return QP_result;
}


int SQPmethod::solve_initial_QP_par(Matrix &deltaXi, Matrix &lambdaQP){
    sub_QPs_par[0]->set_hess(vars->hess, true);
    if (param->sparse)
        sub_QPs_par[0]->set_constr(vars->sparse_constrJac.nz.get(), vars->sparse_constrJac.row.get(), vars->sparse_constrJac.colind.get());
    else
        sub_QPs_par[0]->set_constr(vars->constrJac);
    updateStepBounds();
    sub_QPs_par[0]->set_bounds(vars->delta_lb_var, vars->delta_ub_var, vars->delta_lb_con, vars->delta_ub_con);
    sub_QPs_par[0]->set_lin(vars->gradObj);
    int QP_result = sub_QPs_par[0]->solve(deltaXi, lambdaQP);
    stats->qpIterations2 += sub_QPs_par[0]->get_QP_it();
    vars->conv_qp_solved = true;
    vars->hess_num_accepted = 0;
    stats->qpResolve = 0;
    return QP_result;
}
 

int SQPmethod::solveQP(Matrix &deltaXi, Matrix &lambdaQP, int hess_type){
    //Matrix deltaXi_conv, lambdaQP_conv;
    //deltaXi_conv.Dimension(prob->nVar);
    //lambdaQP_conv.Dimension(prob->nVar + prob->nCon);
    double s_indf_N, s_conv_N;
    
    int l, maxQP;
    if (param->enable_linesearch == 1 && (param->exact_hess > 0 || param->hess_approx == 1 || param->hess_approx == 4 || param->hess_approx == 6 || param->hess_approx > 6) && stats->itCount > 1 && hess_type == 0)
        maxQP = param->max_conv_QPs + 1;
    else
        maxQP = 1;
    
    //Solve convex QP using fallback hessian if indefinite approximations are normally tried first.
    if (hess_type == 1 && (param->hess_approx == 1 || param->hess_approx == 4 || param->hess_approx == 6))
        computeConvexHessian();
    if (hess_type >= 2)
        setIdentityHessian();
    vars->conv_qp_solved = false;

    if (param->sparse)
        sub_QP->set_constr(vars->sparse_constrJac.nz.get(), vars->sparse_constrJac.row.get(), vars->sparse_constrJac.colind.get());
    else
        sub_QP->set_constr(vars->constrJac);
    
    updateStepBounds();
    sub_QP->set_bounds(vars->delta_lb_var, vars->delta_ub_var, vars->delta_lb_con, vars->delta_ub_con);
    sub_QP->set_lin(vars->gradObj);
    sub_QP->set_use_hotstart(vars->use_homotopy);
    
    int QP_result, QP_result_conv;
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
            sub_QP->set_timeLimit(1);
        }
        else{
            //sub_QP->time_limit_type = 0;
            sub_QP->set_timeLimit(0);
            sub_QP->set_hess(vars->hess, false, 0);
        }
        
        //Solve the QP
        QP_result = sub_QP->solve(deltaXi, lambdaQP);
        if (QP_result == 0){
            if (l == maxQP - 1)
                vars->conv_qp_solved = true;
            stats->qpIterations += sub_QP->get_QP_it();
            
            //Save the number of the first hessian for which the QP solved (even though the step may still be replaced by the step from the convex Hessian)
            if (hess_type == 0) vars->hess_num_accepted = l;
            
            //For regularized indefinite hessians, compare steplength to fallback hessian to avoid over-regularized hessians leading to small steps.
            //Skip this for the first regularization as this tends to help lock iterates down to a region of fast convergence.
            if (param->conv_strategy > 0 && l > 1 && l < maxQP - 1){
                computeConvexHessian();
                sub_QP->set_hess(vars->hess, true, vars->modified_hess_regularizationFactor);
                sub_QP->set_timeLimit(0);
                //sub_QP->skip_timeRecord = true;
                QP_result_conv = sub_QP->solve(vars->deltaXi_conv, vars->lambdaQP_conv);
                if (QP_result_conv == 0){
                    s_indf_N = l2VectorNorm(deltaXi);
                    s_conv_N = l2VectorNorm(vars->deltaXi_conv);
                    if (s_indf_N < param->conv_tau_H*s_conv_N){
                        deltaXi = vars->deltaXi_conv;
                        lambdaQP = vars->lambdaQP_conv;
                        vars->conv_qp_solved = true;
                        stats->qpResolve = maxQP - 1;
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


int SQPmethod::solve_SOC_QP(Matrix &deltaXi, Matrix &lambdaQP){

    updateStepBoundsSOC();

    sub_QP->set_bounds(vars->delta_lb_var, vars->delta_ub_var, vars->delta_lb_con, vars->delta_ub_con);
    //sub_QP->convex_QP = vars->conv_qp_solved;

    int QP_result;
    QP_result = sub_QP->solve(deltaXi, lambdaQP);

    stats->qpIterations += sub_QP->get_QP_it();
    return QP_result;
}


int SQPmethod::solveQP_par(Matrix &deltaXi, Matrix &lambdaQP){
    int maxQP = param->max_conv_QPs + 1;
    
    steady_clock::time_point t_0 = steady_clock::now();
    
    
    updateStepBounds();
    
    std::unique_ptr<std::jthread[]> QP_threads = std::make_unique<std::jthread[]>(maxQP);
    
    std::unique_ptr<Matrix[]> QP_sols_prim = std::make_unique<Matrix[]>(maxQP);
    std::unique_ptr<Matrix[]> QP_sols_dual = std::make_unique<Matrix[]>(maxQP);
    
    std::unique_ptr<std::promise<int>[]> QP_results_p = std::make_unique<std::promise<int>[]>(maxQP);
    std::unique_ptr<std::future<int>[]> QP_results_f = std::make_unique<std::future<int>[]>(maxQP);
    for (int j = 0; j < maxQP; j++) QP_results_f[j] = QP_results_p[j].get_future();
    
    std::unique_ptr<std::future_status[]> QP_results_fs = std::make_unique<std::future_status[]>(maxQP-1);
    
    std::unique_ptr<int[]> QP_results = std::make_unique<int[]>(maxQP);
    for (int j = 0; j < maxQP; j++) QP_results[j] = -1;
    
    //VERY important if using qpOASES solver, else SR1-QP always fails for some reason
<<<<<<< HEAD
    /*
    if (vars->hess_num_accepted > 0){
        for (int j = 0; j < maxQP; j++){
            sub_QPs_par[j]->set_hotstart_point(sub_QPs_par[vars->hess_num_accepted].get());
        }
    }
    */
    
    if (param->test_qp_hotstart == 0){
        if (vars->hess_num_accepted > 0){
            for (int j = 0; j < vars->hess_num_accepted; j++){
                sub_QPs_par[j]->set_hotstart_point(sub_QPs_par[vars->hess_num_accepted].get());
            }
        }
    }
    else if (param->test_qp_hotstart == 1){
        if (vars->hess_num_accepted > 0){
            for (int j = 0; j < vars->hess_num_accepted; j++){
                sub_QPs_par[j]->set_hotstart_point(sub_QPs_par[maxQP - 1].get());
            }
        }
    }
    else if (param->test_qp_hotstart == 2){
        for (int j = 0; j < maxQP - 2; j++){
            sub_QPs_par[j]->set_hotstart_point(sub_QPs_par[maxQP - 1].get());
        }
    }
    
   steady_clock::time_point t_1 = steady_clock::now();
   
   std::cout << "QP allocations and preparations took " << duration_cast<microseconds>(t_1 - t_0).count() << "mus\n";
    
    if (param->test_opt_2){
    
        for (int j = 0; j < maxQP; j++){
            steady_clock::time_point T0 = steady_clock::now();
            
            if (j > 0) computeNextHessian(j, maxQP);
            
            if (param->sparse)
                sub_QPs_par[j]->set_constr(vars->sparse_constrJac.nz.get(), vars->sparse_constrJac.row.get(), vars->sparse_constrJac.colind.get());
            else
                sub_QPs_par[j]->set_constr(vars->constrJac);
            sub_QPs_par[j]->set_bounds(vars->delta_lb_var, vars->delta_ub_var, vars->delta_lb_con, vars->delta_ub_con);
            sub_QPs_par[j]->set_lin(vars->gradObj);
            //sub_QPs_par[j]->set_use_hotstart(vars->use_homotopy);
            sub_QPs_par[j]->set_use_hotstart(vars->use_homotopy);
            sub_QPs_par[j]->set_timeLimit(int(j == (maxQP - 1)));
            sub_QPs_par[j]->set_hess(vars->hess, j == maxQP - 1);
            
            QP_sols_prim[j].Dimension(prob->nVar);
            QP_sols_dual[j].Dimension(prob->nVar + prob->nCon);
            
            QP_threads[j] = std::jthread(
                [](std::stop_token stp, QPsolverBase *arg_QPs, std::promise<int> arg_PRM, Matrix &arg_1, Matrix &arg_2){
                    arg_QPs->solve(stp, std::move(arg_PRM), arg_1, arg_2);
                },
                sub_QPs_par[j].get(), std::move(QP_results_p[j]), std::ref(QP_sols_prim[j]), std::ref(QP_sols_dual[j])
            );
            steady_clock::time_point T1 = steady_clock::now();
            std::cout << "Setting up QP " << j << " took " << duration_cast<microseconds>(T1 - T0).count() << "mus\n";
        }
        //Wait for all
        /*
        vars->hess_num_accepted = -1;
        for (int j = 0; j < maxQP; j++){
            QP_results[j] = QP_results_f[j].get();
            if (QP_results[j] == 0 && vars->hess_num_accepted < 0){
                vars->hess_num_accepted = j;
                stats->qpResolve = j;
            }
            std::cout << "QP " << j << " took " << sub_QPs_par[j]->get_solutionTime() << "s\n";
        }
        */
       
        //Terminate long running
        steady_clock::time_point T0 = steady_clock::now();
        QP_results[maxQP - 1] = QP_results_f[maxQP - 1].get();
        QP_threads[maxQP - 1].join();
        steady_clock::time_point T1 = steady_clock::now();
        steady_clock::time_point TF(T1 + microseconds(int(duration_cast<microseconds>(T1 - T0).count()*1.25)) + microseconds(1000));
        
        for (int j = maxQP - 2; j >= 0; j--){
            QP_results_fs[j] = QP_results_f[j].wait_until(TF);
            if (QP_results_fs[j] != std::future_status::ready)  QP_threads[j].request_stop();
            else                                                QP_results[j] = QP_results_f[j].get();
            QP_threads[j].join();
        }
        
        vars->hess_num_accepted = -1;
        stats->qpResolve = -1;
        for (int j = 0; j < maxQP; j++){
            if (QP_results[j] == 0){
                stats->qpResolve = j;
                vars->hess_num_accepted = j;
                break;
            }
        }
        std::cout << "BFGS QP took " << duration_cast<microseconds>(T1 - T0).count() << "mus\n";
    }
    else{
        //TEST SEQ
        stats->qpResolve = -1;
        vars->hess_num_accepted = -1;
        for (int j = 0; j < maxQP; j++){
            if (j > 0) computeNextHessian(j, maxQP);
            
            if (param->sparse)
                sub_QPs_par[j]->set_constr(vars->sparse_constrJac.nz.get(), vars->sparse_constrJac.row.get(), vars->sparse_constrJac.colind.get());
            else
                sub_QPs_par[j]->set_constr(vars->constrJac);
            sub_QPs_par[j]->set_bounds(vars->delta_lb_var, vars->delta_ub_var, vars->delta_lb_con, vars->delta_ub_con);
            sub_QPs_par[j]->set_lin(vars->gradObj);
            //sub_QPs_par[j]->set_use_hotstart(vars->use_homotopy);
            sub_QPs_par[j]->set_use_hotstart(vars->use_homotopy);
            sub_QPs_par[j]->set_timeLimit(int(j == (maxQP - 1)));
            sub_QPs_par[j]->set_hess(vars->hess, j == maxQP - 1);
            
            QP_sols_prim[j].Dimension(prob->nVar);
            QP_sols_dual[j].Dimension(prob->nVar + prob->nCon);
            
            steady_clock::time_point Tseq0 = steady_clock::now();
            QP_results[j] = sub_QPs_par[j]->solve(QP_sols_prim[j], QP_sols_dual[j]);
            if (QP_results[j] == 0 && stats->qpResolve == -1){stats->qpResolve = j; vars->hess_num_accepted = j;}
            steady_clock::time_point Tseq1 = steady_clock::now();
            std::cout << "QP " << j << " took " << duration_cast<microseconds>(Tseq1 - Tseq0).count() << "mus\n";
        }
    }
=======
    /*
    if (vars->hess_num_accepted > 0){
        for (int j = 0; j < maxQP; j++){
            sub_QPs_par[j]->set_hotstart_point(sub_QPs_par[vars->hess_num_accepted].get());
        }
    }
    */
    
    if (vars->hess_num_accepted > 0){
        for (int j = 0; j < vars->hess_num_accepted; j++){
            sub_QPs_par[j]->set_hotstart_point(sub_QPs_par[vars->hess_num_accepted].get());
        }
    }
    
    
    if (param->test_opt_2){
    
    for (int j = 0; j < maxQP; j++){
        if (j > 0) computeNextHessian(j, maxQP);
        
        if (param->sparse)
            sub_QPs_par[j]->set_constr(vars->sparse_constrJac.nz.get(), vars->sparse_constrJac.row.get(), vars->sparse_constrJac.colind.get());
        else
            sub_QPs_par[j]->set_constr(vars->constrJac);
        sub_QPs_par[j]->set_bounds(vars->delta_lb_var, vars->delta_ub_var, vars->delta_lb_con, vars->delta_ub_con);
        sub_QPs_par[j]->set_lin(vars->gradObj);
        //sub_QPs_par[j]->set_use_hotstart(vars->use_homotopy);
        sub_QPs_par[j]->set_use_hotstart(vars->use_homotopy);
        sub_QPs_par[j]->set_timeLimit(int(j == (maxQP - 1)));
        sub_QPs_par[j]->set_hess(vars->hess, j == maxQP - 1);
        
        QP_sols_prim[j].Dimension(prob->nVar);
        QP_sols_dual[j].Dimension(prob->nVar + prob->nCon);
        
        QP_threads[j] = std::jthread(
            [](std::stop_token stp, QPsolverBase *arg_QPs, std::promise<int> arg_PRM, Matrix &arg_1, Matrix &arg_2){
                arg_QPs->solve(stp, std::move(arg_PRM), arg_1, arg_2);
            },
            sub_QPs_par[j].get(), std::move(QP_results_p[j]), std::ref(QP_sols_prim[j]), std::ref(QP_sols_dual[j])
        );
    }
    
    
    //Wait for all
    /*
    vars->hess_num_accepted = -1;
    for (int j = 0; j < maxQP; j++){
        QP_results[j] = QP_results_f[j].get();
        if (QP_results[j] == 0 && vars->hess_num_accepted < 0){
            vars->hess_num_accepted = j;
            stats->qpResolve = j;
        }
        std::cout << "QP " << j << " took " << sub_QPs_par[j]->get_solutionTime() << "s\n";
    }
    */
    
>>>>>>> c8c058613bc0706bac6970558c599c4836ad8bac
    
    //Terminate long running
    
    std::chrono::steady_clock::time_point T0 = std::chrono::steady_clock::now();
    QP_results[maxQP - 1] = QP_results_f[maxQP - 1].get();
    QP_threads[maxQP - 1].join();
    std::chrono::steady_clock::time_point T1 = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point TF(T1 + std::chrono::duration_cast<std::chrono::microseconds>(T1 - T0) + std::chrono::microseconds(10000));
    
    //std::cout << "TF - T1 is " << std::chrono::duration_cast<std::chrono::microseconds>(TF - T1).count() << " mu s\n";
    for (int j = maxQP - 2; j >= 0; j--){
        //QP_results_fs[j] = std::future_status::ready;
        QP_results_fs[j] = QP_results_f[j].wait_until(TF);
        if (QP_results_fs[j] != std::future_status::ready)  QP_threads[j].request_stop();
        else                                                QP_results[j] = QP_results_f[j].get();
        QP_threads[j].join();
    }
    
    vars->hess_num_accepted = -1;
    stats->qpResolve = -1;
    for (int j = 0; j < maxQP; j++){
        stats->qpResolve += (1 + j)*(QP_results[j] == 0)*(stats->qpResolve == -1);
        vars->hess_num_accepted += (1 + j)*(QP_results[j] == 0) * (vars->hess_num_accepted == -1);
    }
    
    
    //for (int j = 0; j < maxQP; j++) std::cout << "QP " << j << " took " << sub_QPs_par[j]->get_solutionTime() << "s\n";
    
    
    }
    else{
    //TEST SEQ
    
    stats->qpResolve = -1;
    vars->hess_num_accepted = -1;
    for (int j = 0; j < maxQP; j++){
        if (j > 0) computeNextHessian(j, maxQP);
        
        if (param->sparse)
            sub_QPs_par[j]->set_constr(vars->sparse_constrJac.nz.get(), vars->sparse_constrJac.row.get(), vars->sparse_constrJac.colind.get());
        else
            sub_QPs_par[j]->set_constr(vars->constrJac);
        sub_QPs_par[j]->set_bounds(vars->delta_lb_var, vars->delta_ub_var, vars->delta_lb_con, vars->delta_ub_con);
        sub_QPs_par[j]->set_lin(vars->gradObj);
        //sub_QPs_par[j]->set_use_hotstart(vars->use_homotopy);
        sub_QPs_par[j]->set_use_hotstart(vars->use_homotopy);
        sub_QPs_par[j]->set_timeLimit(int(j == (maxQP - 1)));
        sub_QPs_par[j]->set_hess(vars->hess, j == maxQP - 1);
        
        QP_sols_prim[j].Dimension(prob->nVar);
        QP_sols_dual[j].Dimension(prob->nVar + prob->nCon);
        
        QP_results[j] = sub_QPs_par[j]->solve(QP_sols_prim[j], QP_sols_dual[j]);
        if (QP_results[j] == 0 && stats->qpResolve == -1){stats->qpResolve = j; vars->hess_num_accepted = j;}
    }
    
    
    }
    
    double s_indf_N = 1.0, s_conv_N = 0.0;
    if (vars->hess_num_accepted > 1 && vars->hess_num_accepted < maxQP - 1 && QP_results[maxQP - 1] == 0){
        s_indf_N = l2VectorNorm(QP_sols_prim[vars->hess_num_accepted]);
        s_conv_N = l2VectorNorm(QP_sols_prim[maxQP - 1]);
    }
    //Always false if above condition false, always false if fallback QP failed (QP_results[maxQP - 1] != 0)
    if (s_indf_N >= param->conv_tau_H*s_conv_N){
        deltaXi = QP_sols_prim[vars->hess_num_accepted];
        lambdaQP = QP_sols_dual[vars->hess_num_accepted];
    }
    else{
        deltaXi = QP_sols_prim[maxQP - 1];
        lambdaQP = QP_sols_dual[maxQP - 1];
        stats->qpResolve = maxQP - 1;
    }
    
    stats->qpIterations2 += sub_QPs_par[vars->hess_num_accepted]->get_QP_it();
    stats->rejectedSR1 += vars->hess_num_accepted;
    
        
    return QP_results[maxQP - 1];
}


int SQPmethod::solve_SOC_QP_par(Matrix &deltaXi, Matrix &lambdaQP){

    updateStepBoundsSOC();

    sub_QPs_par[vars->hess_num_accepted]->set_bounds(vars->delta_lb_var, vars->delta_ub_var, vars->delta_lb_con, vars->delta_ub_con);
    //sub_QP->convex_QP = vars->conv_qp_solved;
    
    int QP_result = sub_QPs_par[vars->hess_num_accepted]->solve(deltaXi, lambdaQP);

    stats->qpIterations += sub_QPs_par[vars->hess_num_accepted]->get_QP_it();
    return QP_result;
}



//Specialized method that only condenses the Hessian and fallback Hessian once each
/*
int SQPmethod::solveQP_par_cond_conv_2(Matrix &deltaXi, Matrix &lambdaQP){
    
}
*/




///////////////////////////////////////////////////Subclass methods

/*
void SCQPmethod::convexify_condensed(SymMatrix *condensed_hess, int idx, int maxQP){
    double idScale = vars->convKappa * std::pow(2, idx - maxQP + 2) * (1.0 - 0.5*(idx > 1));
    //std::cout << "CH idScale = " << idScale << ", convKappa = " << vars->convKappa << "\n";
    for (int i = 0; i < cond->condensed_num_hessblocks; i++){
        for (int j = 0; j < cond->condensed_hess_block_sizes[i]; j++){
            condensed_hess[i](j,j) += idScale;
        }
    }
    return;
}


int SCQPmethod::solveQP(Matrix &deltaXi, Matrix &lambdaQP, int hess_type){
    Matrix deltaXi_conv, lambdaQP_conv;
    SCQPiterate *c_vars = dynamic_cast<SCQPiterate*>(vars.get());
    
    int maxQP, l;
    if (param->enable_linesearch == 1 && (param->hess_approx == 1 || param->hess_approx == 4 || param->hess_approx == 6) && stats->itCount > 1 && hess_type == 0)
        maxQP = param->max_conv_QPs + 1;
    else
        maxQP = 1;

    //hess_type 1: Solve convex QP using fallback hessian if indefinite approximations are normally tried first.
    if (hess_type == 1 && (param->hess_approx == 1 || param->hess_approx == 4 || param->hess_approx == 6)){
        computeConvexHessian();
    }

    if (hess_type >= 2){
        setIdentityHessian();
    }

    vars->conv_qp_solved = false;

    updateStepBounds();
    
    cond->full_condense(c_vars->gradObj, c_vars->Jacobian, c_vars->hess,
                        c_vars->delta_lb_var, c_vars->delta_ub_var, c_vars->delta_lb_con, c_vars->delta_ub_con,
                    c_vars->condensed_h, c_vars->condensed_Jacobian, c_vars->condensed_hess.get(), 
                    c_vars->condensed_lb_var, c_vars->condensed_ub_var, c_vars->condensed_lb_con, c_vars->condensed_ub_con);
    
    if (param->sparse)
        sub_QP->set_constr(c_vars->condensed_Jacobian.nz.get(), c_vars->condensed_Jacobian.row.get(), c_vars->condensed_Jacobian.colind.get());
    else
        sub_QP->set_constr(c_vars->constrJac);
    
    sub_QP->set_bounds(c_vars->condensed_lb_var, c_vars->condensed_ub_var, c_vars->condensed_lb_con, c_vars->condensed_ub_con);
    sub_QP->set_lin(c_vars->condensed_h);
    //sub_QP->set_hess(c_vars->condensed_hess);
    sub_QP->set_use_hotstart(vars->use_homotopy);
    
    int QP_result, QP_result_conv;
    double s_indf_N, s_conv_N;
    for (l = 0; l < maxQP; l++){
        if (l > 0){
            // If the solution of the first QP was rejected, consider second Hessian
            stats->qpResolve++;
            if (param->conv_strategy < 2 || l == maxQP - 1){
                //stats->qpResolve++;
                computeNextHessian(l, maxQP);
                cond->new_hessian_condense(c_vars->hess, c_vars->condensed_h, c_vars->condensed_hess.get());
                sub_QP->set_lin(c_vars->condensed_h);
                //sub_QP->set_hess(c_vars->condensed_hess);
            }
            else convexify_condensed(c_vars->condensed_hess.get(), l, maxQP);
        }

        if (l == maxQP - 1){
            //Inform QP solver about convexity
            sub_QP->set_hess(c_vars->condensed_hess.get(), true, vars->modified_hess_regularizationFactor);
            sub_QP->time_limit_type = 1;
        }
        else{
            sub_QP->set_hess(c_vars->condensed_hess.get(), false, 0);
            sub_QP->time_limit_type = 0;
        }

        //Solve the QP
        QP_result = sub_QP->solve(c_vars->deltaXi_cond, c_vars->lambdaQP_cond);

        if (QP_result == 0){
            if (l == maxQP - 1)
                vars->conv_qp_solved = true;
            cond->recover_var_mult(c_vars->deltaXi_cond, c_vars->lambdaQP_cond, deltaXi, lambdaQP);
            stats->qpIterations += sub_QP->get_QP_it();

            //Save the number of the first hessian for which the QP solved
            if (hess_type == 0)
                vars->hess_num_accepted = l;

            //For regularized indefinite hessians, compare steplength to fallback hessian to avoid over-regularized hessians leading to small steps
            if (param->conv_strategy > 0 && l > 1 && l < maxQP - 1){
                computeConvexHessian();
                cond->new_hessian_condense(c_vars->hess, c_vars->condensed_h, c_vars->condensed_hess.get());

                sub_QP->set_hess(c_vars->condensed_hess.get(), true, vars->modified_hess_regularizationFactor);
                sub_QP->set_lin(c_vars->condensed_h);
                sub_QP->time_limit_type = 0;
                QP_result_conv = sub_QP->solve(c_vars->deltaXi_cond, c_vars->lambdaQP_cond);
                if (QP_result_conv == 0){
                    cond->recover_var_mult(c_vars->deltaXi_cond, c_vars->lambdaQP_cond, deltaXi_conv, lambdaQP_conv);
                    s_indf_N = l2VectorNorm(deltaXi);
                    s_conv_N = l2VectorNorm(deltaXi_conv);
                    if (param->conv_strategy == 1 && s_indf_N < 0.8*s_conv_N){
                        deltaXi = deltaXi_conv;
                        lambdaQP = lambdaQP_conv;
                        vars->conv_qp_solved = true;
                        stats->qpResolve = maxQP - 1;
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


int SCQPmethod::solve_SOC_QP( Matrix &deltaXi, Matrix &lambdaQP){

    SCQPiterate *c_vars = dynamic_cast<SCQPiterate*>(vars.get());

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

    return QP_result;
}


int SCQP_bound_method::solveQP(Matrix &deltaXi, Matrix &lambdaQP, int hess_type){

    SCQPiterate *c_vars = dynamic_cast<SCQPiterate*>(vars.get());

    int maxQP, l;
    if (param->enable_linesearch == 1 && (param->hess_approx == 1 || param->hess_approx == 4 || param->hess_approx == 6) && stats->itCount > 1 && hess_type == 0){
        maxQP = param->max_conv_QPs + 1;
    }
    else
        maxQP = 1;

    //Solve convex QP using fallback hessian if indefinite approximations are normally tried first.
    if (hess_type == 1 && (param->hess_approx == 1 || param->hess_approx == 4 || param->hess_approx == 6)){
        computeConvexHessian();
    }

    if (hess_type >= 2){
        setIdentityHessian();
    }

    vars->conv_qp_solved = false;

    updateStepBounds();
    cond->full_condense(c_vars->gradObj, c_vars->Jacobian, c_vars->hess,
        c_vars->delta_lb_var, c_vars->delta_ub_var, c_vars->delta_lb_con, c_vars->delta_ub_con,
            c_vars->condensed_h, c_vars->condensed_Jacobian, c_vars->condensed_hess.get(), c_vars->condensed_lb_var, c_vars->condensed_ub_var, c_vars->condensed_lb_con, c_vars->condensed_ub_con);
    
    if (param->sparse)
        sub_QP->set_constr(c_vars->condensed_Jacobian.nz.get(), c_vars->condensed_Jacobian.row.get(), c_vars->condensed_Jacobian.colind.get());
    else
        sub_QP->set_constr(c_vars->constrJac);
    
    sub_QP->set_bounds(c_vars->condensed_lb_var, c_vars->condensed_ub_var, c_vars->condensed_lb_con, c_vars->condensed_ub_con);
    sub_QP->set_lin(c_vars->condensed_h);
    //sub_QP->set_hess(c_vars->condensed_hess);
    sub_QP->set_use_hotstart(vars->use_homotopy);
    
    int QP_result;
    for (l = 0; l < maxQP; l++){
        if (l > 0){
            stats->qpResolve++;
            computeNextHessian(l, maxQP);
            cond->new_hessian_condense(c_vars->hess, c_vars->condensed_h, c_vars->condensed_hess.get());
            sub_QP->set_lin(c_vars->condensed_h);
            //sub_QP->set_hess(c_vars->condensed_hess);
        }

        if( l == maxQP-1 ){
            //Inform QP solver about convexity
            //sub_QP->convex_QP = true;
            sub_QP->set_hess(c_vars->condensed_hess.get(), true, vars->modified_hess_regularizationFactor);
            sub_QP->time_limit_type = 1;
        }
        else{
            sub_QP->set_hess(c_vars->condensed_hess.get(), false, 0);
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
                sub_QP->skip_timeRecord = true;
                sub_QP->time_limit_type = 0;

                std::chrono::steady_clock::time_point begin_ = std::chrono::steady_clock::now();
                QP_result = sub_QP->solve(c_vars->deltaXi_cond, c_vars->lambdaQP_cond);
                std::cout << "QP_result is " << QP_result << "\n";
                std::chrono::steady_clock::time_point end_ = std::chrono::steady_clock::now();
                std::cout << "Solved QP with added bounds in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin_).count() << "ms\n";

                if (QP_result == 1){
                    std::cout << "Solution of QP with added bounds is taking too long, initialize new QP\n";
                    sub_QP->set_use_hotstart(false);
                    sub_QP->time_limit_type = 1;
                    sub_QP->skip_timeRecord = true;

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

    return QP_result;
}



int SCQP_bound_method::solve_SOC_QP( Matrix &deltaXi, Matrix &lambdaQP){

    SCQPiterate *c_vars = dynamic_cast<SCQPiterate*>(vars.get());

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
                sub_QP->set_use_hotstart(false);

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

    return QP_result;
}

*/


/*
int SCQP_correction_method::solveQP(Matrix &deltaXi, Matrix &lambdaQP, int hess_type){

    SCQP_correction_iterate *c_vars = dynamic_cast<SCQP_correction_iterate*>(vars);

    int maxQP, l;
    if (param->enable_linesearch == 1 && (param->hessUpdate == 1 || param->hessUpdate == 4 || param->hessUpdate == 6) && stats->itCount > 1 && hess_type == 0){
        maxQP = param->max_conv_QPs + 1;
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

    if (param->sparse)
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

    return QP_result;
}
*/





/*

int SCQP_correction_method::solve_SOC_QP( Matrix &deltaXi, Matrix &lambdaQP){

    SCQP_correction_iterate *c_vars = dynamic_cast<SCQP_correction_iterate*>(vars.get());

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
        std::cout << "Solved QP in " << sub_QP->get_solutionTime() << "s\n";
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
            std::cout << "Solved correction QP in " << sub_QP->get_solutionTime() << "s\n";
            std::cout << "QP_result is " << QP_result << "\n";
            std::chrono::steady_clock::time_point end_ = std::chrono::steady_clock::now();
            std::cout << "Solved SOC QP with added corrections in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin_).count() << "ms\n";

            if (QP_result == 1){
                std::cout << "Solution of QP with added corrections is taking too long, initialize new QP\n";
                sub_QP->set_use_hotstart(false);

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
    return QP_result;
}


int SCQP_correction_method::bound_correction(Matrix &deltaXi_corr, Matrix &lambdaQP_corr){

    int ind_1, ind_2, ind, vio_count, QP_result;
    double xi_s, max_dep_bound_violation;

    SCQP_correction_iterate *c_vars = static_cast<SCQP_correction_iterate*>(vars.get());

    deltaXi_corr = vars->deltaXi;
    lambdaQP_corr = vars->lambdaQP;

    //Reset correction vectors
    for (int tnum = 0; tnum < cond->num_targets; tnum++){
        corrections[tnum].Initialize(0.);
    }
    
        //If a variable is being corrected and not at a bounds, reduce correction
        //If a variable violates a bound, add to its correction term
    for (int k = 0; k < param->max_correction_steps; k++){
        ind_1 = 0;
        vio_count = 0;
        max_dep_bound_violation = 0;

        for (int i = 0; i < cond->num_vblocks; i++){
            if (cond->vblocks[i].dependent){
                for (int j = 0; j < cond->vblocks[i].size; j++){
                    ind = ind_1 + j;
                    xi_s = vars->xi(ind) + deltaXi_corr(ind);
                    if (xi_s < prob->lb_var(ind) - param->dep_bound_tolerance || xi_s > prob->ub_var(ind) + param->dep_bound_tolerance){
                        vio_count++;

                        //Optional: Calculate maximum dep bound violation
                        if (prob->lb_var(ind) - xi_s > max_dep_bound_violation)
                            max_dep_bound_violation = prob->lb_var(ind) - xi_s;
                        else if (xi_s - prob->ub_var(ind) > max_dep_bound_violation)
                            max_dep_bound_violation = xi_s - prob->ub_var(ind);
                        //
                    }
                }
            }
            ind_1 += cond->vblocks[i].size;
        }

        if (vio_count == 0)
            return 0;

        std::cout << "Bounds violated by " << vio_count << " dependent variables, calculating correction vectors\n";
        std::cout << "Max dep bound violation is " << max_dep_bound_violation << "\n";

        for (int tnum = 0; tnum < cond->num_targets; tnum++){

            //Add difference between dependent state values from QP solution and integration for target tnum
            ind_1 = 0;
            ind_2 = cond->vranges[cond->targets[tnum].first_free];

            for (int i = cond->targets[tnum].first_free; i < cond->targets[tnum].vblock_end; i++){
                if (cond->vblocks[i].dependent){
                    for (int j = 0; j < cond->vblocks[i].size; j++){
                        xi_s = vars->xi(ind_2 + j) + deltaXi_corr(ind_2 + j);

                        //Optional: Reduce corrections if is strictly within bounds
                        if (corrections[tnum](ind_1 + j) > 0 && xi_s > prob->lb_var(ind_2 + j)){
                            corrections[tnum](ind_1 + j) -= xi_s - prob->lb_var(ind_2 + j);
                            if (corrections[tnum](ind_1 + j) < 0) corrections[tnum](ind_1 + j) = 0;
                        }
                        else if (corrections[tnum](ind_1 + j) < 0 && xi_s < prob->ub_var(ind_2 + j)){
                            corrections[tnum](ind_1 + j) -= xi_s - prob->ub_var(ind_2 + j);
                            if (corrections[tnum](ind_1 + j) > 0) corrections[tnum](ind_1 + j) = 0;
                        }


                        if (xi_s < prob->lb_var(ind_2 + j) - param->dep_bound_tolerance){
                            corrections[tnum](ind_1 + j) += prob->lb_var(ind_2 + j) - xi_s;
                        }
                        else if (xi_s > prob->ub_var(ind_2 + j) + param->dep_bound_tolerance){
                            corrections[tnum](ind_1 + j) += prob->ub_var(ind_2 + j) - xi_s;
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

        sub_QP->skip_timeRecord = true;
        sub_QP->time_limit_type = 0;

        std::chrono::steady_clock::time_point begin_ = std::chrono::steady_clock::now();
        QP_result = sub_QP->solve(c_vars->deltaXi_cond, c_vars->lambdaQP_cond);
        std::cout << "QP_result is " << QP_result << "\n";
        std::chrono::steady_clock::time_point end_ = std::chrono::steady_clock::now();
        std::cout << "Solved QP with added corrections in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin_).count() << "ms\n";

        if (!QP_result)
            cond->recover_correction_var_mult(c_vars->deltaXi_cond, c_vars->lambdaQP_cond, corrections, deltaXi_corr, lambdaQP_corr);
        else
            return 1;
    }
    return 0;
}
    
*/


} // namespace blockSQP


