/*
 * blockSQP2 -- A structure-exploiting nonlinear programming solver based
 *              on blockSQP by Dennis Janka.
 * Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
 * 
 * Licensed under the zlib license. See LICENSE for more details.
 */

/**
 * \file blocksqp_qpsolver.cpp
 * \author Reinhold Wittmann
 * \date 2023-2025
 *
 *  Implementation of abstract interface base classes and subclasses
 *  for calling quadratic programming solvers
 *  
 */


#include <blockSQP2/qpsolver.hpp>
#include <blockSQP2/general_purpose.hpp>
#include <blockSQP2/problemspec.hpp>
#include <blockSQP2/defs.hpp>
#include <blockSQP2/iterate.hpp>
#include <cmath>
#include <chrono>
using namespace std::chrono;


namespace blockSQP2{

std::string to_string(QPresults qpres){
    switch(qpres){
        case QPresults::undef:
            return "undefined";
        case QPresults::success:
            return "success";
        case QPresults::time_it_limit_reached:
            return "time/it_limit_exceeded";
        case QPresults::indef_unbounded:
            return "definiteness condition violated or unbounded";
        case QPresults::infeasible:
            return "infeasibility";
        default:;
    }
    return "unspecified error";
}
std::ostream& operator<<(std::ostream &os, QPresults qpres){
    return os << "\"" << to_string(qpres) << "\"";
}

BasicQPsolver::~BasicQPsolver(){}

void BasicQPsolver::set_hotstart_point(BasicQPsolver *hot_QP){return;};
void BasicQPsolver::solve(std::stop_token stopRequest, std::promise<QPresults> QP_result, Matrix &deltaXi, Matrix &lambdaQP){QP_result.set_value(solve(deltaXi, lambdaQP));}

void BasicQPsolver::set_constr(const Matrix &constr_jac){
    throw NotImplementedError("QPsolver::set_constr(const Matrix &constr_jac)");
}

void BasicQPsolver::set_constr(double *const jac_nz, int *const jac_row, int *const jac_colind){
    throw NotImplementedError("QPsolver::set_constr(const Sparse_Matrix &constr_jac)");
}

//QPsolver base class implemented methods
QPsolver::QPsolver(int n_QP_var, int n_QP_con, int n_QP_hessblocks, const QPsolver_options *QPopts): nVar(n_QP_var), nCon(n_QP_con), nHess(n_QP_hessblocks), Qparam(QPopts){
    //For managing QP solution times
    default_time_limit = Qparam->max_QP_secs;
    custom_time_limit = Qparam->max_QP_secs;
    timeLimitType = TimeLimitTypes::past_avg;
    skip_timeRecord = false;

    dur_pos = 9; dur_count = 0;
    QPtime_avg = default_time_limit/2.5;
    for (int i = 0; i < 10; i++){
        solution_durations[i] = default_time_limit/2.5;
    }
    //Problem information
    convex_QP = false;
    convex_regF = QPopts->reg_factor;
    regF = 0.;
    regInd_l = -1;
    
    //Flags
    use_hotstart = false;
};

QPsolver::~QPsolver(){}


void QPsolver::set_timeLimit(TimeLimitTypes limitType, double custom_limit_secs){
    timeLimitType = limitType;
    if (timeLimitType == TimeLimitTypes::custom){
        if (custom_limit_secs <= 0.0){
            std::cout << "WARNING: Custom time limit selected but no valid custom limit passed!\n";
            custom_limit_secs = default_time_limit;
        }
        custom_time_limit = custom_limit_secs;
    }
    return;
}

void QPsolver::set_use_hotstart(bool use_hom){
    use_hotstart = use_hom;
}

void QPsolver::set_reg_indices(int *arg_regInd, int arg_regInd_l, bool regInd_own){
    if (regInd_own){
        regInd_l = arg_regInd_l;
        regInd = std::unique_ptr<int[]>(arg_regInd);
        return;
    }
    if (regInd_l < arg_regInd_l){
        regInd_l = arg_regInd_l;
        regInd = std::make_unique<int[]>(arg_regInd_l);
    }
    std::copy(arg_regInd, arg_regInd + regInd_l, regInd.get());
}

double QPsolver::get_reg(){
    return regF;
}

int QPsolver::get_QP_it(){return 0;}
double QPsolver::get_solutionTime(){return solution_durations[dur_pos];}


void QPsolver::record_time(double solTime){
    dur_pos = (dur_pos + 1)%10;
    solution_durations[dur_pos] = solTime;
    dur_count += int(dur_count < 10);
    QPtime_avg = 0.0;
    for (int i = 0; i < dur_count; i++){
        QPtime_avg += solution_durations[(dur_pos - i + 10)%10];
    }
    QPtime_avg /= dur_count;
    return;
}

void QPsolver::reset_timeRecord(){
    dur_pos = 0;
    dur_count = 0;
    QPtime_avg = default_time_limit/2.5;
}


BasicCQPsolver::BasicCQPsolver(BasicQPsolver *arg_inner_QPsol, Condenser *arg_cond, bool arg_QPsol_own, bool arg_cond_own):
        inner_QPsol(arg_inner_QPsol), condenser(arg_cond), QPsol_own(arg_QPsol_own), condenser_own(arg_cond_own),
        xi_cond(condenser->condensed_num_vars), lambda_cond(condenser->condensed_num_vars + condenser->condensed_num_cons),
        regF(0.), reg_updated(false),
        h_cond_updated(true), hess_cond_updated(true), A_cond_updated(true), bounds_cond_updated(true){
    int inner_regInd_l = 0;
    for (int i = 0; i < condenser->num_vblocks; i++){
        inner_regInd_l += condenser->vblocks[i].size*int(!condenser->vblocks[i].dependent && !condenser->vblocks[i].removed);
    }
    std::unique_ptr<int[]> inner_regInd = std::make_unique<int[]>(inner_regInd_l);
    int offset = 0, ind = 0;
    for (int i = 0; i < condenser->num_vblocks; i++){
        const vblock &VB = condenser->vblocks[i];
        if (VB.removed) continue;
        if (!VB.dependent){
            for (int j = 0; j < VB.size; j++){
                inner_regInd[ind + j] = offset + j;
            }
            ind += VB.size;
        }
        offset += VB.size;
    }
    inner_QPsol->set_reg_indices(std::move(inner_regInd), inner_regInd_l);
}

BasicCQPsolver::~BasicCQPsolver(){}

void BasicCQPsolver::set_cond_update_flags(bool up_h, bool up_hess, bool up_A, bool up_bounds){
    h_cond_updated = up_h;
    hess_cond_updated = up_hess;
    A_cond_updated = up_A;
    bounds_cond_updated = up_bounds;
}


void BasicCQPsolver::set_timeLimit(TimeLimitTypes limitType, double custom_limit_secs){inner_QPsol->set_timeLimit(limitType, custom_limit_secs);}
void BasicCQPsolver::set_use_hotstart(bool use_hom){inner_QPsol->set_use_hotstart(use_hom);}

void BasicCQPsolver::set_hotstart_point(BasicQPsolver *hot_QP){
    if (dynamic_cast<BasicCQPsolver*>(hot_QP) != nullptr){
        set_hotstart_point(static_cast<BasicCQPsolver*>(hot_QP));
        return;
    }
    throw std::invalid_argument("BasicCQPsolver::set_hotstart_point: Argument error, expected BasicCQPsolver");
}

void BasicCQPsolver::set_hotstart_point(BasicCQPsolver *hot_QP){
    inner_QPsol->set_hotstart_point(hot_QP->inner_QPsol);
    return;
}


void BasicCQPsolver::set_reg(double arg){
    if (regF < 0.0) throw std::invalid_argument("BasicCQPsolver::set_reg - regularization factor < 0 invalid");
    regF = arg;
    reg_updated = true;
}

void BasicCQPsolver::set_reg_indices(int *arg_regInd, int arg_l, bool regInd_own){
    int inner_regInd_l = 0, j = 0, ind = 0;
    for (int i = 0; i < condenser->num_vblocks; i++){
        for (; arg_regInd[j++] < ind + condenser->vblocks[i].size;){
            inner_regInd_l += int(!condenser->vblocks[i].removed);
            if (j == arg_l) goto sri_outer_loop_1_break;
        }
        ind += condenser->vblocks[i].size;
    } sri_outer_loop_1_break:
    std::unique_ptr<int[]> inner_regInd = std::make_unique<int[]>(inner_regInd_l);
    
    int offset = 0; j = 0; ind = 0;
    for (int i = 0; i < condenser->num_vblocks; i++){
        for (; arg_regInd[j++] < ind + condenser->vblocks[i].size;){
            inner_regInd[j-1] = arg_regInd[j-1] - offset;
            if (j == arg_l) goto sri_outer_loop_2_break;
        }
        ind += condenser->vblocks[i].size;
        offset += condenser->vblocks[i].size * int(condenser->vblocks[i].removed);
    } sri_outer_loop_2_break:
    inner_QPsol->set_reg_indices(std::move(inner_regInd), inner_regInd_l);
    if (regInd_own) delete[] arg_regInd;
    return;
}

double BasicCQPsolver::get_reg(){
    return regF;
}


QPresults BasicCQPsolver::solve(Matrix &deltaXi, Matrix &lambdaQP){
    // steady_clock::time_point T0 = steady_clock::now();
    setup_inner_QPsol(deltaXi, lambdaQP);
    // steady_clock::time_point T1 = steady_clock::now();
    // std::cout << "Setting up the QP took " << duration_cast<microseconds>(T1 - T0) << "\n";
    
    QPresults QPret = inner_QPsol->solve(xi_cond, lambda_cond);
    // steady_clock::time_point T2 = steady_clock::now();
    // std::cout << "Setting up the QP took " << duration_cast<microseconds>(T1 - T0) << ", solving the QP took " << duration_cast<microseconds>(T2 - T1) << "\n";
    if (QPret == QPresults::success)
        condenser->recover_var_mult(xi_cond, lambda_cond, deltaXi, lambdaQP);
    return QPret;
}

void BasicCQPsolver::solve(std::stop_token stopRequest, std::promise<QPresults> QP_result, Matrix &deltaXi, Matrix &lambdaQP){
    // steady_clock::time_point T0 = steady_clock::now();
    setup_inner_QPsol(deltaXi, lambdaQP);
    // steady_clock::time_point T1 = steady_clock::now();
    
    std::promise<QPresults> QP_result_cond_p;
    std::future<QPresults> QP_result_cond_f = QP_result_cond_p.get_future();
    QPresults QP_result_cond;
    
    inner_QPsol->solve(stopRequest, std::move(QP_result_cond_p), xi_cond, lambda_cond);
    // steady_clock::time_point T2 = steady_clock::now();
    // std::cout << "Setting up the QP took " << duration_cast<microseconds>(T1 - T0) << ", solving the QP took " << duration_cast<microseconds>(T2 - T1) << "\n";
    
    QP_result_cond = QP_result_cond_f.get();
    
    if (QP_result_cond == QPresults::success)
        condenser->recover_var_mult(xi_cond, lambda_cond, deltaXi, lambdaQP);
    QP_result.set_value(QP_result_cond);
    return;
}

int BasicCQPsolver::get_QP_it(){return inner_QPsol->get_QP_it();}
double BasicCQPsolver::get_solutionTime(){return inner_QPsol->get_solutionTime();}



CQPsolver::CQPsolver(BasicQPsolver *arg_inner_QPsol, Condenser *arg_cond, bool arg_QPsol_own, bool arg_cond_own):
        BasicCQPsolver(arg_inner_QPsol, arg_cond, arg_QPsol_own, arg_cond_own),
        // inner_QPsol(arg_CQPsol), condenser(arg_cond), QPsol_own(arg_QPsol_own), condenser_own(arg_cond_own),
        h_qp(condenser->num_vars),
        hess_qp(new SymMatrix[condenser->num_hessblocks]),
        lb_x(condenser->num_vars), ub_x(condenser->num_vars), lb_A(condenser->num_cons), ub_A(condenser->num_cons),
        //TODO: Pass sparsity pattern of Jacobian to condenser and precompute sparsity pattern of condensed Jacobian
        h_cond(condenser->condensed_num_hessblocks),
        hess_cond(new SymMatrix[condenser->condensed_num_hessblocks]),
        lb_x_cond(condenser->condensed_num_vars), ub_x_cond(condenser->condensed_num_vars), 
        lb_A_cond(condenser->condensed_num_cons), ub_A_cond(condenser->condensed_num_cons),
        convex_QP(false), 
        //TODO: ' '   ' ' 
        // xi_cond(condenser->condensed_num_vars), lambda_cond(condenser->condensed_num_vars + condenser->condensed_num_cons),
        h_updated(true), hess_updated(true), A_updated(true), bounds_updated(true),
        // h_cond_updated(true), hess_cond_updated(true), A_cond_updated(true), bounds_cond_updated(true),
        // regF(0.0), reg_updated(false)
        shared_CQPsols(0){
    for (int k = 0; k < condenser->num_hessblocks; k++){
        hess_qp[k].Dimension(condenser->hess_block_sizes[k]);
    }
    
    for (int k = 0; k < condenser->condensed_num_hessblocks; k++){
        hess_cond[k].Dimension(condenser->condensed_hess_block_sizes[k]);
    }
}

CQPsolver::~CQPsolver(){
    if (QPsol_own) delete inner_QPsol;
    if (condenser_own) delete condenser;
}


// void CQPsolver::set_reg(double arg){
//     regF = arg;
//     reg_updated = true;
// }

// double CQPsolver::get_reg(){
//     return regF;
// }


void CQPsolver::set_lin(const Matrix &grad_obj){
    h_qp = grad_obj;
    h_updated = true;
}

void CQPsolver::set_hess(SymMatrix *const hess, bool pos_def){
    convex_QP = pos_def;
    for (int k = 0; k < condenser->num_hessblocks; k++){
        hess_qp[k] = hess[k];
    }
    hess_updated = true;
    regF = 0.0;
    reg_updated = false;
}

void CQPsolver::set_constr(const Matrix &constr_jac){
    throw NotImplementedError("CQPsolver::set_constr(const Matrix &constr_jac)");
}
void CQPsolver::set_constr(double *const jac_nz, int *const jac_row, int *const jac_colind){
    if (sparse_A_qp.m == 0){
        sparse_A_qp.Dimension(condenser->num_cons, condenser->num_vars, jac_colind[condenser->num_vars]);
    }
    std::copy(jac_nz, jac_nz + sparse_A_qp.colind[sparse_A_qp.n], sparse_A_qp.nz.get());
    std::copy(jac_row, jac_row + sparse_A_qp.colind[sparse_A_qp.n], sparse_A_qp.row.get());
    std::copy(jac_colind, jac_colind + sparse_A_qp.n + 1, sparse_A_qp.colind.get());
    A_updated = true;
}

void CQPsolver::set_bounds(const Matrix &arg_lb_x, const Matrix &arg_ub_x, const Matrix &arg_lb_A, const Matrix &arg_ub_A){
    lb_x = arg_lb_x; ub_x = arg_ub_x;
    lb_A = arg_lb_A; ub_A = arg_ub_A;
    bounds_updated = true;
}

void CQPsolver::cond_update_notify(bool up_h, bool up_hess, bool up_A, bool up_bounds){
    set_cond_update_flags(up_h, up_hess, up_A, up_bounds);
    for (SharedCQPsolver* sCQP : shared_CQPsols){
        sCQP->set_cond_update_flags(up_h, up_hess, up_A, up_bounds);
    }
}

void CQPsolver::invoke_condensing(){
    if (!hess_updated && !h_updated && !A_updated && !bounds_updated){
        return;
    }
    if (!hess_updated && !h_updated && !A_updated && bounds_updated){
        condenser->SOC_condense(h_qp, lb_A, ub_A, h_cond, lb_A_cond, ub_A_cond);
        cond_update_notify(true, false, false, true);
        // h_cond_updated = true; 
        // bounds_cond_updated = true;
        bounds_updated = false;
    }
    else if (hess_updated && !h_updated && !A_updated && !bounds_updated){
        condenser->new_hessian_condense(hess_qp.get(), h_cond, hess_cond.get());
        cond_update_notify(true, true, false, false);
        // h_cond_updated = true;
        // hess_cond_updated = true;
        hess_updated = false;
    }
    else{
        condenser->full_condense(h_qp, sparse_A_qp, hess_qp.get(), lb_x, ub_x, lb_A, ub_A, 
            h_cond, sparse_A_cond, hess_cond.get(), lb_x_cond, ub_x_cond, lb_A_cond, ub_A_cond);
        cond_update_notify(true, true, true, true);
        // h_cond_updated = true;
        // hess_cond_updated = true;
        // A_cond_updated = true;
        // bounds_cond_updated = true;
        h_updated = false; hess_updated = false; A_updated = false; bounds_updated = false;
    }
}

// void CQPsolver::setup_inner_QPsol(Matrix &deltaXi, Matrix &lambdaQP){
//     if (!hess_updated && !h_updated && !A_updated && bounds_updated){
//         condenser->SOC_condense(h_qp, lb_A, ub_A, h_cond, lb_A_cond, ub_A_cond);
//         h_cond_updated = true; 
//         bounds_cond_updated = true;
//         inner_QPsol->set_lin(h_cond);
//         inner_QPsol->set_bounds(lb_x_cond, ub_x_cond, lb_A_cond, ub_A_cond);
//     }
//     else if (hess_updated && !h_updated && !A_updated && !bounds_updated){
//         condenser->new_hessian_condense(hess_qp.get(), h_cond, hess_cond.get());
//         inner_QPsol->set_lin(h_cond);
//         inner_QPsol->set_hess(hess_cond.get(), convex_QP);
//     }
//     else if (!hess_updated && !h_updated && !A_updated && !bounds_updated && reg_updated){
//         inner_QPsol->set_reg(regF);
//     }
//     else{
//         condenser->full_condense(h_qp, sparse_A_qp, hess_qp.get(), lb_x, ub_x, lb_A, ub_A, 
//             h_cond, sparse_A_cond, hess_cond.get(), lb_x_cond, ub_x_cond, lb_A_cond, ub_A_cond);
//         inner_QPsol->set_lin(h_cond);
//         inner_QPsol->set_constr(sparse_A_cond.nz.get(), sparse_A_cond.row.get(), sparse_A_cond.colind.get());
//         inner_QPsol->set_hess(hess_cond.get(), convex_QP);
//         inner_QPsol->set_bounds(lb_x_cond, ub_x_cond, lb_A_cond, ub_A_cond);
//     }
    
//     h_updated = false; hess_updated = false; A_updated = false; bounds_updated = false; reg_updated = false;
//     return;
// }

void CQPsolver::setup_inner_QPsol(Matrix &deltaXi, Matrix &lambdaQP){
    invoke_condensing();
    if (h_cond_updated) inner_QPsol->set_lin(h_cond);
    if (hess_cond_updated) inner_QPsol->set_hess(hess_cond.get(), convex_QP);
    if (A_cond_updated) inner_QPsol->set_constr(sparse_A_cond.nz.get(), sparse_A_cond.row.get(), sparse_A_cond.colind.get());
    if (bounds_cond_updated) inner_QPsol->set_bounds(lb_x_cond, ub_x_cond, lb_A_cond, ub_A_cond);
    
    if (reg_updated) inner_QPsol->set_reg(regF);
    reg_updated = false;
}

// QPresults CQPsolver::solve(Matrix &deltaXi, Matrix &lambdaQP){
//     // steady_clock::time_point T0 = steady_clock::now();
//     setup_inner_QPsol(deltaXi, lambdaQP);
//     // steady_clock::time_point T1   = steady_clock::now();
//     QPresults QPret = inner_QPsol->solve(xi_cond, lambda_cond);
//     // std::cout << "Setting up condensed QP took " << duration_cast<microseconds>(T1-T0) << "\n";
    
//     if (QPret != QPresults::success) return QPret;
    
//     condenser->recover_var_mult(xi_cond, lambda_cond, deltaXi, lambdaQP);
//     return QPret;
// }

// void CQPsolver::solve(std::stop_token stopRequest, std::promise<QPresults> QP_result, Matrix &deltaXi, Matrix &lambdaQP){
//     setup_inner_QPsol(deltaXi, lambdaQP);
    
//     std::promise<QPresults> QP_result_cond_p;
//     std::future<QPresults> QP_result_cond_f = QP_result_cond_p.get_future();
//     QPresults QP_result_cond;
//     inner_QPsol->solve(stopRequest, std::move(QP_result_cond_p), xi_cond, lambda_cond);
//     QP_result_cond = QP_result_cond_f.get();
//     if (QP_result_cond != QPresults::success){
//         QP_result.set_value(QP_result_cond); 
//         return;
//     }
//     condenser->recover_var_mult(xi_cond, lambda_cond, deltaXi, lambdaQP);
//     QP_result.set_value(QP_result_cond);
// }


// void CQPsolver::set_timeLimit(TimeLimitTypes limitType, double custom_limit_secs){inner_QPsol->set_timeLimit(limitType, custom_limit_secs);}
// void CQPsolver::set_use_hotstart(bool use_hom){inner_QPsol->set_use_hotstart(use_hom);}

// void CQPsolver::set_hotstart_point(BasicQPsolver *hot_QP){
//     if (dynamic_cast<CQPsolver*>(hot_QP) != nullptr){
//         set_hotstart_point(static_cast<CQPsolver*>(hot_QP));
//     }
//     return;
// }

// void CQPsolver::set_hotstart_point(CQPsolver *hot_QP){
//     inner_QPsol->set_hotstart_point(hot_QP->inner_QPsol);
//     return;
// }


// int CQPsolver::get_QP_it(){return inner_QPsol->get_QP_it();}
// double CQPsolver::get_solutionTime(){return inner_QPsol->get_solutionTime();}

void CQPsolver::add_shared(SharedCQPsolver *arg_shared){
    shared_CQPsols.push_back(arg_shared);
}

void CQPsolver::remove_shared(SharedCQPsolver *arg_shared){
    for (std::vector<SharedCQPsolver*>::iterator it = shared_CQPsols.begin(); it != shared_CQPsols.end(); it++)
        if (*it == arg_shared){
            shared_CQPsols.erase(it);
            break;
        }
}


QPresults CQPsolver::bound_correction(const Matrix &xi, const Matrix &lb_var, const Matrix &ub_var, Matrix &deltaXi_corr, Matrix &lambdaQP_corr){
    int max_correction_steps = 6;
    double dep_bound_tolerance = 1e-7;
    
    int ind_1, ind_2, ind, vio_count;
    QPresults QP_result;
    double xi_s, max_dep_bound_violation;
    
    corrections = std::make_unique<Matrix[]>(condenser->num_targets); 
    for (int i = 0; i < condenser->num_targets; i++){
        corrections[i].Dimension(condenser->targets_data[i].n_dep).Initialize(0.);
    }
    h_corr.Dimension(condenser->condensed_num_vars);
    lb_A_corr.Dimension(condenser->condensed_num_cons);
    ub_A_corr.Dimension(condenser->condensed_num_cons);
    
    //If a variable is being corrected and not at a bounds, reduce correction
    //If a variable violates a bound, add to its correction term
    for (int k = 0; k < max_correction_steps; k++){
        ind_1 = 0;
        vio_count = 0;
        max_dep_bound_violation = 0;

        for (int i = 0; i < condenser->num_vblocks; i++){
            if (condenser->vblocks[i].dependent){
                for (int j = 0; j < condenser->vblocks[i].size; j++){
                    ind = ind_1 + j;
                    xi_s = xi(ind) + deltaXi_corr(ind);
                    if (xi_s < lb_var(ind) - dep_bound_tolerance || xi_s > ub_var(ind) + dep_bound_tolerance){
                        vio_count++;
                        
                        //Optional: Calculate maximum dep bound violation
                        if (lb_var(ind) - xi_s > max_dep_bound_violation)
                            max_dep_bound_violation = lb_var(ind) - xi_s;
                        else if (xi_s - ub_var(ind) > max_dep_bound_violation)
                            max_dep_bound_violation = xi_s - ub_var(ind);
                    }
                }
            }
            ind_1 += condenser->vblocks[i].size;
        }

        if (vio_count == 0){
            std::cout << "All dependent variable bounds are fulfilled\n" << std::flush;
            return QPresults::success;
        }
        
        std::cout << "Bounds violated by " << vio_count << " dependent variables, calculating correction vectors\n";
        std::cout << "Max dep bound violation is " << max_dep_bound_violation << "\n";
        
        for (int tnum = 0; tnum < condenser->num_targets; tnum++){
            
            //Add difference between dependent state values from QP solution and integration for target tnum
            ind_1 = 0;
            ind_2 = condenser->vranges[condenser->targets[tnum].vblock_start];
            
            for (int i = condenser->targets[tnum].vblock_start; i < condenser->targets[tnum].vblock_end; i++){
                if (condenser->vblocks[i].dependent){
                    for (int j = 0; j < condenser->vblocks[i].size; j++){
                        xi_s = xi(ind_2 + j) + deltaXi_corr(ind_2 + j);
                        
                        //Optional: Reduce corrections if is strictly within bounds
                        if (corrections[tnum](ind_1 + j) > 0 && xi_s > lb_var(ind_2 + j)){
                            corrections[tnum](ind_1 + j) -= xi_s - lb_var(ind_2 + j);
                            if (corrections[tnum](ind_1 + j) < 0) corrections[tnum](ind_1 + j) = 0;
                        }
                        else if (corrections[tnum](ind_1 + j) < 0 && xi_s < ub_var(ind_2 + j)){
                            corrections[tnum](ind_1 + j) -= xi_s - ub_var(ind_2 + j);
                            if (corrections[tnum](ind_1 + j) > 0) corrections[tnum](ind_1 + j) = 0;
                        }
                        
                        if (xi_s < lb_var(ind_2 + j) - dep_bound_tolerance){
                            corrections[tnum](ind_1 + j) += lb_var(ind_2 + j) - xi_s;
                        }
                        else if (xi_s > ub_var(ind_2 + j) + dep_bound_tolerance){
                            corrections[tnum](ind_1 + j) += ub_var(ind_2 + j) - xi_s;
                        }
                    }
                    ind_1 += condenser->vblocks[i].size;
                }
                ind_2 += condenser->vblocks[i].size;
            }
        }

        //Condense the QP, adding the correction to g = Gu + g
        //cond->correction_condense(c_vars->gradObj, c_vars->delta_lb_con, c_vars->delta_ub_con, corrections, c_vars->corrected_h, c_vars->corrected_lb_con, c_vars->corrected_ub_con);
        condenser->correction_condense(h_qp, lb_A, ub_A, corrections.get(), h_corr, lb_A_corr, ub_A_corr);
        
        inner_QPsol->set_bounds(lb_x_cond, ub_x_cond, lb_A_corr, ub_A_corr);
        inner_QPsol->set_lin(h_corr);
        
        inner_QPsol->set_timeLimit(TimeLimitTypes::past_avg);
        static_cast<QPsolver*>(inner_QPsol)->record_time(false);
        
        std::chrono::steady_clock::time_point T0 = std::chrono::steady_clock::now();
        QP_result = inner_QPsol->solve(xi_cond, lambda_cond);
        std::cout << "QP_result is " << QP_result << "\n";
        std::chrono::steady_clock::time_point T1 = std::chrono::steady_clock::now();
        std::cout << "Solved QP with added corrections in " << duration_cast<milliseconds>(T1 - T0).count() << "ms\n"; //gcc-10 compatibility: Dont use operator<< duration overload
        
        if (QP_result == QPresults::success)
            condenser->recover_correction_var_mult(xi_cond, lambda_cond, corrections.get(), deltaXi_corr, lambdaQP_corr);
        else
            return QPresults::other_error;
    }
    return QPresults::success;
}

QPresults CQPsolver::correction_solve(Matrix &deltaXi, Matrix &lambdaQP){
    if (corrections == nullptr) throw std::runtime_error("CQPsolver correction_solve called before bound_correction");
    if (!hess_updated && !h_updated && !A_updated){
        condenser->correction_condense(h_qp, lb_A, ub_A, corrections.get(), h_corr, lb_A_corr, ub_A_corr);
        inner_QPsol->set_lin(h_corr);
        inner_QPsol->set_bounds(lb_x_cond, ub_x_cond, lb_A_corr, ub_A_corr);
    }
    else throw std::runtime_error("CQPsolver: correction_solve called after hess, linear term or constraint matrix was changed");
    
    QPresults QPret = inner_QPsol->solve(xi_cond, lambda_cond);
    if (QPret != QPresults::success) return QPret;
    
    condenser->recover_correction_var_mult(xi_cond, lambda_cond, corrections.get(), deltaXi, lambdaQP);
    return QPret;
}

//int CQPsolver::SOC_bound_correction(const Matrix &xi, const Matrix &lb_var, const Matrix &ub_var, Matrix &deltaXi_corr, Matrix &lambdaQP_corr){
    //TODO
//}


SharedCQPsolver::SharedCQPsolver(CQPsolver *arg_CQPsol, BasicQPsolver *arg_inner_QPsol, bool arg_QPsol_own):
        BasicCQPsolver(arg_inner_QPsol, arg_CQPsol->condenser, arg_QPsol_own, false),
        CQPsol(arg_CQPsol){
    CQPsol->add_shared(this);
}

SharedCQPsolver::~SharedCQPsolver(){
    CQPsol->remove_shared(this);
}


// void SharedCQPsolver::set_lin(const Matrix &grad_obj){CQPsol->set_lin(grad_obj);}
// void SharedCQPsolver::set_hess(SymMatrix *const hess, bool pos_def){CQPsol->set_hess(hess, pos_def);}

// void SharedCQPsolver::set_constr(const Matrix &constr_jac){CQPsol->set_constr(constr_jac);}
// void SharedCQPsolver::set_constr(double *const jac_nz, int *const jac_row, int *const jac_colind){
//     CQPsol->set_constr(jac_nz, jac_row, jac_colind);
// }
// void SharedCQPsolver::set_bounds(const Matrix &lb_x, const Matrix &ub_x, const Matrix &lb_A, const Matrix &ub_A){
//     CQPsol->set_bounds(lb_x, ub_x, lb_A, ub_A);
// }

void SharedCQPsolver::set_lin(const Matrix &grad_obj){}
void SharedCQPsolver::set_hess(SymMatrix *const hess, bool pos_def){}

void SharedCQPsolver::set_constr(const Matrix &constr_jac){}
void SharedCQPsolver::set_constr(double *const jac_nz, int *const jac_row, int *const jac_colind){}
void SharedCQPsolver::set_bounds(const Matrix &lb_x, const Matrix &ub_x, const Matrix &lb_A, const Matrix &ub_A){}
    

void SharedCQPsolver::invoke_condensing(){
    CQPsol->invoke_condensing();
}

void SharedCQPsolver::setup_inner_QPsol(Matrix &deltaXi, Matrix &lambdaQP){
    invoke_condensing();
    if (h_cond_updated) inner_QPsol->set_lin(CQPsol->h_cond);
    if (hess_cond_updated) inner_QPsol->set_hess(CQPsol->hess_cond.get(), CQPsol->convex_QP);
    if (A_cond_updated) inner_QPsol->set_constr(CQPsol->sparse_A_cond);
    if (bounds_cond_updated) inner_QPsol->set_bounds(CQPsol->lb_x_cond, CQPsol->ub_x_cond, CQPsol->lb_A_cond, CQPsol->ub_A_cond);
    
    if (reg_updated) inner_QPsol->set_reg(regF);
    reg_updated = false;
}


QPresults SharedCQPsolver::bound_correction(const Matrix &xi, const Matrix &lb_var, const Matrix &ub_var, Matrix &deltaXi_corr, Matrix &lambdaQP_corr){
    int max_correction_steps = 6;
    double dep_bound_tolerance = 1e-7;
    
    int ind_1, ind_2, ind, vio_count;
    QPresults QP_result;
    double xi_s, max_dep_bound_violation;
    
    corrections = std::make_unique<Matrix[]>(condenser->num_targets); 
    for (int i = 0; i < condenser->num_targets; i++){
        corrections[i].Dimension(condenser->targets_data[i].n_dep).Initialize(0.);
    }
    h_corr.Dimension(condenser->condensed_num_vars);
    lb_A_corr.Dimension(condenser->condensed_num_cons);
    ub_A_corr.Dimension(condenser->condensed_num_cons);
    
    //If a variable is being corrected and not at a bounds, reduce correction
    //If a variable violates a bound, add to its correction term
    for (int k = 0; k < max_correction_steps; k++){
        ind_1 = 0;
        vio_count = 0;
        max_dep_bound_violation = 0;

        for (int i = 0; i < condenser->num_vblocks; i++){
            if (condenser->vblocks[i].dependent){
                for (int j = 0; j < condenser->vblocks[i].size; j++){
                    ind = ind_1 + j;
                    xi_s = xi(ind) + deltaXi_corr(ind);
                    if (xi_s < lb_var(ind) - dep_bound_tolerance || xi_s > ub_var(ind) + dep_bound_tolerance){
                        vio_count++;
                        
                        //Optional: Calculate maximum dep bound violation
                        if (lb_var(ind) - xi_s > max_dep_bound_violation)
                            max_dep_bound_violation = lb_var(ind) - xi_s;
                        else if (xi_s - ub_var(ind) > max_dep_bound_violation)
                            max_dep_bound_violation = xi_s - ub_var(ind);
                    }
                }
            }
            ind_1 += condenser->vblocks[i].size;
        }

        if (vio_count == 0){
            std::cout << "All dependent variable bounds are fulfilled\n" << std::flush;
            return QPresults::success;
        }
        
        std::cout << "Bounds violated by " << vio_count << " dependent variables, calculating correction vectors\n";
        std::cout << "Max dep bound violation is " << max_dep_bound_violation << "\n";
        
        for (int tnum = 0; tnum < condenser->num_targets; tnum++){
            
            //Add difference between dependent state values from QP solution and integration for target tnum
            ind_1 = 0;
            ind_2 = condenser->vranges[condenser->targets[tnum].vblock_start];
            
            for (int i = condenser->targets[tnum].vblock_start; i < condenser->targets[tnum].vblock_end; i++){
                if (condenser->vblocks[i].dependent){
                    for (int j = 0; j < condenser->vblocks[i].size; j++){
                        xi_s = xi(ind_2 + j) + deltaXi_corr(ind_2 + j);
                        
                        //Optional: Reduce corrections if is strictly within bounds
                        if (corrections[tnum](ind_1 + j) > 0 && xi_s > lb_var(ind_2 + j)){
                            corrections[tnum](ind_1 + j) -= xi_s - lb_var(ind_2 + j);
                            if (corrections[tnum](ind_1 + j) < 0) corrections[tnum](ind_1 + j) = 0;
                        }
                        else if (corrections[tnum](ind_1 + j) < 0 && xi_s < ub_var(ind_2 + j)){
                            corrections[tnum](ind_1 + j) -= xi_s - ub_var(ind_2 + j);
                            if (corrections[tnum](ind_1 + j) > 0) corrections[tnum](ind_1 + j) = 0;
                        }
                        
                        if (xi_s < lb_var(ind_2 + j) - dep_bound_tolerance){
                            corrections[tnum](ind_1 + j) += lb_var(ind_2 + j) - xi_s;
                        }
                        else if (xi_s > ub_var(ind_2 + j) + dep_bound_tolerance){
                            corrections[tnum](ind_1 + j) += ub_var(ind_2 + j) - xi_s;
                        }
                    }
                    ind_1 += condenser->vblocks[i].size;
                }
                ind_2 += condenser->vblocks[i].size;
            }
        }

        //Condense the QP, adding the correction to g = Gu + g
        //cond->correction_condense(c_vars->gradObj, c_vars->delta_lb_con, c_vars->delta_ub_con, corrections, c_vars->corrected_h, c_vars->corrected_lb_con, c_vars->corrected_ub_con);
        condenser->correction_condense(CQPsol->h_qp, CQPsol->lb_A, CQPsol->ub_A, corrections.get(), h_corr, lb_A_corr, ub_A_corr);
        
        inner_QPsol->set_bounds(CQPsol->lb_x_cond, CQPsol->ub_x_cond, lb_A_corr, ub_A_corr);
        inner_QPsol->set_lin(h_corr);
        
        inner_QPsol->set_timeLimit(TimeLimitTypes::past_avg);
        static_cast<QPsolver*>(inner_QPsol)->record_time(false);
        
        std::chrono::steady_clock::time_point T0 = std::chrono::steady_clock::now();
        QP_result = inner_QPsol->solve(xi_cond, lambda_cond);
        std::cout << "QP_result is " << QP_result << "\n";
        std::chrono::steady_clock::time_point T1 = std::chrono::steady_clock::now();
        std::cout << "Solved QP with added corrections in " << duration_cast<milliseconds>(T1 - T0).count() << "ms\n"; //gcc-10 compatibility: Dont use operator<< duration overload
        
        if (QP_result == QPresults::success)
            condenser->recover_correction_var_mult(xi_cond, lambda_cond, corrections.get(), deltaXi_corr, lambdaQP_corr);
        else
            return QPresults::other_error;
    }
    return QPresults::success;
}

QPresults SharedCQPsolver::correction_solve(Matrix &deltaXi, Matrix &lambdaQP){
    if (corrections == nullptr) throw std::runtime_error("CQPsolver correction_solve called before bound_correction");
    if (!CQPsol->hess_updated && !CQPsol->h_updated && !CQPsol->A_updated){
        condenser->correction_condense(CQPsol->h_qp, CQPsol->lb_A, CQPsol->ub_A, corrections.get(), h_corr, lb_A_corr, ub_A_corr);
        inner_QPsol->set_lin(h_corr);
        inner_QPsol->set_bounds(CQPsol->lb_x_cond, CQPsol->ub_x_cond, lb_A_corr, ub_A_corr);
    }
    else throw std::runtime_error("CQPsolver: correction_solve called after hess, linear term or constraint matrix was changed");
    
    QPresults QPret = inner_QPsol->solve(xi_cond, lambda_cond);
    if (QPret != QPresults::success) return QPret;
    
    condenser->recover_correction_var_mult(xi_cond, lambda_cond, corrections.get(), deltaXi, lambdaQP);
    return QPret;
}






BasicQPsolver *create_QPsolver(int nVarQP, int nConQP, int nHessQP, int *blockIdx, const QPsolver_options *Qparam, int par_ind){
    // BasicQPsolver *QPsol = nullptr;
    #ifdef QPSOLVER_QPOASES
    if (Qparam->sol == QPsolvers::qpOASES){
        #ifdef SOLVER_MUMPS
            if (par_ind > - 1){
                load_mumps_libs(par_ind);
                return new qpOASES_MUMPS_solver(nVarQP, nConQP, nHessQP, blockIdx, Qparam, get_fptr_dmumps_c(par_ind));
            }
            else return new qpOASES_solver(nVarQP, nConQP, nHessQP, blockIdx, Qparam);
        #else
            return new qpOASES_solver(nVarQP, nConQP, nHessQP, blockIdx, Qparam);
        #endif
    }
    #elif defined(QPSOLVER_GUROBI)
    if (Qparam->sol == QPsolvers::gurobi){
        return new gurobi_solver(nVarQP, nConQP, nHessQP, Qparam);
    }
    #elif defined(QPSOLVER_QPALM)
    if (Qparam->sol == QPsolvers::qpalm){
        return new qpalm_solver(nVarQP, nConQP, nHessQP,  Qparam);
    }
    #endif
    throw ParameterError("Selected QPsolver is not available");
}


BasicQPsolver *create_QPsolver(const Problemspec *prob, const SQPiterate *vars, const QPsolver_options *Qparam){
    int nVarQP, nConQP, nHessQP, *blockIdx;
    BasicQPsolver *QPsol = nullptr;
    if (prob->condenser == nullptr){
        nConQP = prob->nCon;
        nVarQP = prob->nVar;
        nHessQP = vars->nBlocks;
        blockIdx = vars->blockIdx.get();
    }
    else{
        nConQP = prob->condenser->condensed_num_cons;
        nVarQP = prob->condenser->condensed_num_vars;
        nHessQP = prob->condenser->condensed_num_hessblocks;
        blockIdx = prob->condenser->condensed_blockIdx.get();
    }
    QPsol = create_QPsolver(nVarQP, nConQP, nHessQP, blockIdx, Qparam);
    
    //Wrap condensing step over external QP solver
    if (prob->condenser != nullptr) QPsol = new CQPsolver(QPsol, prob->condenser->layout_copy(), true, true);
    return QPsol;
}


std::unique_ptr<std::unique_ptr<BasicQPsolver>[]> create_QPsolvers_par(const Problemspec *prob, const SQPiterate *vars, const SQPoptions *param, int arg_N_QP){
    if (prob->condenser != nullptr && param->conv_strategy == 2) return create_QPsolvers_par_cond(prob, vars, param, arg_N_QP);
    int N_QP = arg_N_QP == -1 ? param->max_conv_QPs + 1 : arg_N_QP;
    std::unique_ptr<std::unique_ptr<BasicQPsolver>[]> QPsols_par = std::make_unique<std::unique_ptr<BasicQPsolver>[]>(N_QP);
    
    int nVarQP, nConQP, nHessQP, *blockIdx;
    if (prob->condenser == nullptr){
        nConQP = prob->nCon;
        nVarQP = prob->nVar;
        nHessQP = vars->nBlocks;
        blockIdx = vars->blockIdx.get();
    }
    else{
        nConQP = prob->condenser->condensed_num_cons;
        nVarQP = prob->condenser->condensed_num_vars;
        nHessQP = prob->condenser->condensed_num_hessblocks;
        blockIdx = prob->condenser->condensed_blockIdx.get();
    }
    
    BasicQPsolver *QPsol = nullptr;
    for (int i = 0; i < N_QP - 1; i++){
        QPsol = create_QPsolver(nVarQP, nConQP, nHessQP, blockIdx, param->qpsol_options, i);
        if (prob->condenser != nullptr) QPsol = new CQPsolver(QPsol, prob->condenser->layout_copy(), true, true);
        QPsols_par[i] = std::unique_ptr<BasicQPsolver>(QPsol);
    }
    QPsol = create_QPsolver(nVarQP, nConQP, nHessQP, blockIdx, param->qpsol_options, -1);
    if (prob->condenser != nullptr) QPsol = new CQPsolver(QPsol, prob->condenser, true, true);
    QPsols_par[N_QP - 1] = std::unique_ptr<BasicQPsolver>(QPsol);
    return QPsols_par;
}



// std::unique_ptr<std::unique_ptr<BasicQPsolver>[]> create_QPsolvers_par(const Problemspec *prob, const SQPiterate *vars, const SQPoptions *param, int arg_N_QP){
//     if (prob->condenser != nullptr && param->conv_strategy == 2 && param->qpsol == QPsolvers::qpOASES) return create_QPsolvers_par_cond(prob, vars, param, arg_N_QP);
//     int N_QP = arg_N_QP == -1 ? param->max_conv_QPs + 1 : arg_N_QP;
//     std::unique_ptr<std::unique_ptr<BasicQPsolver>[]> QPsols_par = std::make_unique<std::unique_ptr<BasicQPsolver>[]>(N_QP);
    
//     #ifdef SOLVER_MUMPS
//     if (param->qpsol != QPsolvers::qpOASES || param->qpsol_options->condensed){
//     #endif
    
//         for (int i = 0; i < N_QP; i++){
//             QPsols_par[i] = std::unique_ptr<BasicQPsolver>(create_QPsolver(prob, vars, param->qpsol_options));
//         }
//         return QPsols_par;
    
//     #ifdef SOLVER_MUMPS
//     }
//     //Work around the MUMPS sparse solver not being thread safe (Currently only possible on linux and windows)
//     int n_QP, m_QP, n_hess_QP, *blockIdx;
//     BasicQPsolver *QPsol = nullptr;
//     if (prob->condenser == nullptr){
//         m_QP = prob->nCon;
//         n_QP = prob->nVar;
//         n_hess_QP = vars->nBlocks;
//         blockIdx = vars->blockIdx.get();
//     }
//     else{
//         m_QP = prob->condenser->condensed_num_cons;
//         n_QP = prob->condenser->condensed_num_vars;
//         n_hess_QP = prob->condenser->condensed_num_hessblocks;
//         blockIdx = prob->condenser->condensed_blockIdx.get();
//     }
    
//     load_mumps_libs(N_QP - 1);
//     for (int i = 0; i < N_QP - 1; i++){
//         QPsol = new qpOASES_MUMPS_solver(n_QP, m_QP, n_hess_QP, blockIdx, static_cast<const qpOASES_options*>(param->qpsol_options), get_fptr_dmumps_c(i));
//         if (prob->condenser != nullptr) QPsol = new CQPsolver(QPsol, prob->condenser->layout_copy(), true, true);
//         QPsols_par[i] = std::unique_ptr<BasicQPsolver>(QPsol);
//     }
    
//     QPsol = new qpOASES_solver(n_QP, m_QP, n_hess_QP, blockIdx, static_cast<const qpOASES_options*>(param->qpsol_options));
//     if (prob->condenser != nullptr) QPsol = new CQPsolver(QPsol, prob->condenser->layout_copy(), true, true);
//     QPsols_par[N_QP - 1] = std::unique_ptr<BasicQPsolver>(QPsol);
    
//     return QPsols_par;
//     #endif
// }



std::unique_ptr<std::unique_ptr<BasicQPsolver>[]> create_QPsolvers_par_cond(const Problemspec *prob, const SQPiterate *vars, const SQPoptions *param, int arg_N_QP){
    int N_QP = arg_N_QP == -1 ? param->max_conv_QPs + 1 : arg_N_QP;
    std::unique_ptr<std::unique_ptr<BasicQPsolver>[]> QPsols_par = std::make_unique<std::unique_ptr<BasicQPsolver>[]>(N_QP);
    
    int nVarQP, nConQP, nHessQP, *blockIdx;
    BasicQPsolver *QPsol = nullptr;
    nConQP = prob->condenser->condensed_num_cons;
    nVarQP = prob->condenser->condensed_num_vars;
    nHessQP = prob->condenser->condensed_num_hessblocks;
    blockIdx = prob->condenser->condensed_blockIdx.get();
    
    QPsol = create_QPsolver(nVarQP, nConQP, nHessQP, blockIdx, static_cast<const qpOASES_options*>(param->qpsol_options), 0);
    QPsol = new CQPsolver(QPsol, prob->condenser->layout_copy(), true, true);
    QPsols_par[0] = std::unique_ptr<BasicQPsolver>(QPsol);
    for (int i = 1; i < N_QP - 1; i++){
        QPsol = create_QPsolver(nVarQP, nConQP, nHessQP, blockIdx, static_cast<const qpOASES_options*>(param->qpsol_options), i);
        QPsol = new SharedCQPsolver(dynamic_cast<CQPsolver*>(QPsols_par[0].get()), QPsol, true);
        QPsols_par[i] = std::unique_ptr<BasicQPsolver>(QPsol);
    }
    QPsol = create_QPsolver(nVarQP, nConQP, nHessQP, blockIdx, static_cast<const qpOASES_options*>(param->qpsol_options), -1);
    QPsol = new CQPsolver(QPsol, prob->condenser->layout_copy(), true, true);
    QPsols_par[N_QP - 1] = std::unique_ptr<BasicQPsolver>(QPsol);
    return QPsols_par;
}



// std::unique_ptr<std::unique_ptr<BasicQPsolver>[]> create_QPsolvers_par_cond(const Problemspec *prob, const SQPiterate *vars, const SQPoptions *param, int arg_N_QP){
//     #ifdef QPSOLVER_QPOASES
    
//     int N_QP = arg_N_QP == -1 ? param->max_conv_QPs + 1 : arg_N_QP;
//     std::unique_ptr<std::unique_ptr<BasicQPsolver>[]> QPsols_par = std::make_unique<std::unique_ptr<BasicQPsolver>[]>(N_QP);
    
//     int n_QP, m_QP, n_hess_QP, *blockIdx;
//     BasicQPsolver *QPsol = nullptr;
//     m_QP = prob->condenser->condensed_num_cons;
//     n_QP = prob->condenser->condensed_num_vars;
//     n_hess_QP = prob->condenser->condensed_num_hessblocks;
//     blockIdx = prob->condenser->condensed_blockIdx.get();
    
//     QPsol = new qpOASES_solver(n_QP, m_QP, n_hess_QP, blockIdx, static_cast<const qpOASES_options*>(param->qpsol_options));
//     QPsol = new CQPsolver(QPsol, prob->condenser->layout_copy(), true, true);
//     QPsols_par[0] = std::unique_ptr<BasicQPsolver>(QPsol);
//     for (int i = 1; i < N_QP - 1; i++){
//         QPsol = new qpOASES_solver(n_QP, m_QP, n_hess_QP, blockIdx, static_cast<const qpOASES_options*>(param->qpsol_options));
//         QPsol = new SharedCQPsolver(dynamic_cast<CQPsolver*>(QPsols_par[0].get()), QPsol, true);
//         QPsols_par[i] = std::unique_ptr<BasicQPsolver>(QPsol);
//     }
//     QPsol = new qpOASES_solver(n_QP, m_QP, n_hess_QP, blockIdx, static_cast<const qpOASES_options*>(param->qpsol_options));
//     QPsol = new CQPsolver(QPsol, prob->condenser->layout_copy(), true, true);
//     QPsols_par[N_QP - 1] = std::unique_ptr<BasicQPsolver>(QPsol);
//     return QPsols_par;
    
//     #endif
//     throw std::logic_error("create_QPsolvers_par_cond: Error, currently only supports qpOASES");
// }



////////////////////////////////////////////////////////////////
/////////////Interfaces to (third party) QP solvers/////////////
////////////////////////////////////////////////////////////////



///////////////////////
///qpOASES interface///
///////////////////////

#ifdef QPSOLVER_QPOASES


qpOASES_solver::qpOASES_solver(int n_QP_var, int n_QP_con, int n_QP_hessblocks, int *blockIdx, const QPsolver_options *QPopts):
                    QPsolver(n_QP_var, n_QP_con, n_QP_hessblocks, QPopts){
        
    // if (static_cast<const qpOASES_options*>(Qparam)->sparsityLevel < 2){
    //     qp = std::unique_ptr<qpOASES::SQProblem>(new qpOASES::SQProblem(nVar, nCon));
    //     qpSave = std::unique_ptr<qpOASES::SQProblem>(new qpOASES::SQProblem(nVar, nCon));
    //     qpCheck = std::unique_ptr<qpOASES::SQProblem>(new qpOASES::SQProblem(nVar, nCon));
    // }
    // else{
        // qpOASES::LinearSolverType LST = Qparam->condensed ? qpOASES::LST_LAPACK : qpOASES::LST_ANY;
    
    // switch (static_cast<const qpOASES_options*>(Qparam)->matrixSparsity){
    //     case 0: sparseMatrices = false; break;
    //     case 1: sparseMatrices = true; break;
    //     case -1: sparseMatrices = n_QP_hessblocks > 10; break;
    //     default: throw ParameterError(std::string("qpOASES_options::matrixSparsity has an invalid value of ") + std::to_string(sparseMatrices) + std::string(", should be -1 for auto, 0 for dense, 1 for sparse"));
    // }
    if (Qparam->sol != QPsolvers::qpOASES) throw ParameterError("Wrong options type given for QPsolver qpOASES");
    
    if (static_cast<const qpOASES_options*>(Qparam)->matrixSparsity != -1)
        throw ParameterError("Values of qpOASES_options::matrixSparsity other that -1 are currently unsupported");
    // sparseMatrices = !Qparam->condensed;
    
    sparseMatrices = true;
    qpOASES::LinearSolverType LST = sparseMatrices ? qpOASES::LST_ANY : qpOASES::LST_LAPACK; //n_QP_hessblocks < 11 ? qpOASES::LST_LAPACK : qpOASES::LST_ANY;

    // int maxSchur = 15;//15 + 35*int(LST != qpOASES::LST_LAPACK);
    int maxSchur = 15;
    
    // qpOASES::LinearSolverType LST = Qparam->condensed ? qpOASES::LST_ANY : qpOASES::LST_LAPACK;
    qp = std::unique_ptr<qpOASES::SQProblemSchur>(new qpOASES::SQProblemSchur(nVar, nCon, qpOASES::HST_UNKNOWN, maxSchur, LST));
    qpSave = std::unique_ptr<qpOASES::SQProblemSchur>(new qpOASES::SQProblemSchur(nVar, nCon, qpOASES::HST_UNKNOWN, maxSchur, LST));
    qpCheck = std::unique_ptr<qpOASES::SQProblemSchur>(new qpOASES::SQProblemSchur(nVar, nCon, qpOASES::HST_UNKNOWN, maxSchur, LST));
    
    init_QP_common(blockIdx);
}

void qpOASES_solver::init_QP_common(int *blockIdx){
    A_qp = nullptr;
    H_qp = nullptr;
    
    lb = std::make_unique<double[]>(nVar);
    ub = std::make_unique<double[]>(nVar);
    lbA = std::make_unique<double[]>(nCon);
    ubA = std::make_unique<double[]>(nCon);
    
    h_qp = std::make_unique<double[]>(nVar);
    A_qp_nz = nullptr;
    A_qp_row = nullptr;
    A_qp_colind = nullptr;
    
    if (sparseMatrices){
        int hess_nzCount = 0;
        for (int i = 0; i < nHess; i++){
            hess_nzCount += (blockIdx[i+1] - blockIdx[i])*(blockIdx[i+1] - blockIdx[i]);
        }
        //Allocate enough memory to support all structurally nonzero elements being nonzero.
        hess_nz = std::make_unique<double[]>(hess_nzCount);
        hess_row = std::make_unique<int[]>(hess_nzCount);
        hess_colind = std::make_unique<int[]>(nVar + 1);
        hess_loind = std::make_unique<int[]>(nVar + 1);
    }
    else{
        hess_nz = std::make_unique<double[]>(nVar*nVar);
        jacT.Dimension(nVar, nCon);
    }
    
    //Options
    opts.enableEqualities = qpOASES::BT_TRUE;
    // opts.boundTolerance = 1.0e-8;
    opts.boundTolerance = 1e4*qpOASES::EPS;
    opts.initialStatusBounds = qpOASES::ST_INACTIVE;
    switch(static_cast<const qpOASES_options*>(Qparam)->printLevel){
        case 0: opts.printLevel = qpOASES::PL_NONE;     break;
        case 1: opts.printLevel = qpOASES::PL_LOW;      break;
        case 2: opts.printLevel = qpOASES::PL_MEDIUM;   break;
        default: opts.printLevel = qpOASES::PL_HIGH;    break;
    }
    opts.numRefinementSteps = 2;
    opts.epsLITests =  2.2204e-08;
    opts.terminationTolerance = static_cast<const qpOASES_options*>(Qparam)->terminationTolerance;
}


qpOASES_solver::qpOASES_solver(int n_QP_var, int n_QP_con, int n_QP_hessblocks, const QPsolver_options *QPopts):
                                        QPsolver(n_QP_var, n_QP_con, n_QP_hessblocks, QPopts){
    if (Qparam->sol != QPsolvers::qpOASES) throw ParameterError("Wrong options type given for QPsolver qpOASES");
}
                                        


//This flag is added in the modified qpOASES version, a runtime error is thrown if this class is attempted to be constructed with an unmodified version.
#ifdef SOLVER_MUMPS
    #ifdef SQPROBLEMSCHUR_ENABLE_PASSTHROUGH
        qpOASES_MUMPS_solver::qpOASES_MUMPS_solver(int n_QP_var, int n_QP_con, int n_QP_hessblocks, 
                                                                        int *blockIdx, const QPsolver_options *QPopts, void *fptr_dmumps_c):
                                                qpOASES_solver(n_QP_var, n_QP_con, n_QP_hessblocks, QPopts){
            // sparseMatrices = !Qparam->condensed;
            sparseMatrices = true;
            qpOASES::LinearSolverType LST = sparseMatrices ? qpOASES::LST_MUMPS : qpOASES::LST_LAPACK;
            int maxSchur = 15;
            
            qp = std::unique_ptr<qpOASES::SQProblem>(new qpOASES::SQProblemSchur(nVar, nCon, qpOASES::HST_UNKNOWN, maxSchur, LST, fptr_dmumps_c));
            qpSave = std::unique_ptr<qpOASES::SQProblem>(new qpOASES::SQProblemSchur(nVar, nCon, qpOASES::HST_UNKNOWN, maxSchur, LST, fptr_dmumps_c));
            qpCheck = std::unique_ptr<qpOASES::SQProblem>(new qpOASES::SQProblemSchur(nVar, nCon, qpOASES::HST_UNKNOWN, maxSchur, LST, fptr_dmumps_c));
            // sparseMatrices = !Qparam->condensed;
            init_QP_common(blockIdx);    
        }
    #else
        qpOASES_MUMPS_solver::qpOASES_MUMPS_solver(int n_QP_var, int n_QP_con, int n_QP_hessblocks, 
                                                                        int *blockIdx, const QPsolver_options *QPopts, void *fptr_dmumps_c):
                                                qpOASES_solver(n_QP_var, n_QP_con, n_QP_hessblocks, QPopts){
            throw NotImplementedError("Using qpOASES with MUMPS in parallel requires the patched version");
        }
    #endif
#endif

void qpOASES_solver::set_reg(double arg){
    double delta_regF = arg - regF;
    regF += delta_regF;
    if (H_qp == nullptr) throw std::logic_error("qpOASES_solver::set_reg - error, a Hessian must be set before a regularization");
    qpOASES::returnValue ret = (regInd == nullptr) ? H_qp->addToDiag(delta_regF) :  H_qp->addToDiagIndices(delta_regF, regInd.get(), regInd_l);
    // if (regInd == nullptr) H_qp->addToDiag(delta_regF); else H_qp->addToDiagIndices(delta_regF, regInd.get(), regInd_l);
    if (ret != qpOASES::SUCCESSFUL_RETURN) throw std::logic_error("addToDiag failed\n");
    matrices_changed = true;
}


void qpOASES_solver::set_lin(const Matrix &grad_obj){
    std::copy(grad_obj.array, grad_obj.array + grad_obj.m, h_qp.get());
    return;
}

void qpOASES_solver::set_hess(SymMatrix *const hess, bool pos_def){
    convex_QP = pos_def;
    double regFactor = convex_QP ? convex_regF : 0.0;
    
    // if (static_cast<const qpOASES_options*>(Qparam)->sparsityLevel > 0){
    if (sparseMatrices){
        // convertHessian_noalloc(Qparam->eps, hess, nHess, nVar, regFactor, hess_nz.get(), hess_row.get(), hess_colind.get(), hess_loind.get());
        convertHessian_noalloc(Qparam->eps, hess, nHess, nVar, hess_nz.get(), hess_row.get(), hess_colind.get(), hess_loind.get());
        H_qp = std::make_unique<qpOASES::SymSparseMat>(nVar, nVar, hess_row.get(), hess_colind.get(), hess_nz.get());
        dynamic_cast<qpOASES::SymSparseMat*>(H_qp.get())->createDiagInfo();
    }
    else{
        // convertHessian_noalloc(hess, nHess, nVar, regFactor, hess_nz.get());
        convertHessian_noalloc(hess, nHess, nVar, hess_nz.get());
        H_qp = std::make_unique<qpOASES::SymDenseMat>(nVar, nVar, nVar, hess_nz.get());
    }
    // if (regInd == nullptr) H_qp->addToDiag(regFactor); else H_qp->addToDiagIndices(regFactor, regInd.get(), regInd_l);
    qpOASES::returnValue ret = (regInd == nullptr) ? H_qp->addToDiag(regFactor) :  H_qp->addToDiagIndices(regFactor, regInd.get(), regInd_l);
    if (ret != qpOASES::SUCCESSFUL_RETURN) throw std::logic_error("addToDiag failed\n");
    matrices_changed = true;
    regF = 0.0;
    return;
}

// void qpOASES_solver::set_constr(const Matrix &constr_jac){
//     Transpose(constr_jac, jacT);
//     A_qp = std::make_unique<qpOASES::DenseMatrix>(nCon, nVar, nVar, jacT.array);
//     return;
// }
void qpOASES_solver::set_constr(const Matrix &constr_jac){
    Transpose(constr_jac, jacT);
    A_qp = std::make_unique<qpOASES::DenseMatrix>(nCon, nVar, nVar, jacT.array);
    return;
}

// void qpOASES_solver::set_constr(double *const jac_nz, int *const jac_row, int *const jac_colind){
//     if (A_qp_nz == nullptr){
//         A_qp_nz = std::make_unique<double[]>(jac_colind[nVar]);
//         A_qp_row = std::make_unique<int[]>(jac_colind[nVar]);
//         A_qp_colind = std::make_unique<int[]>(nVar + 1);
//     }
//     std::copy(jac_nz, jac_nz + jac_colind[nVar], A_qp_nz.get());
//     std::copy(jac_row, jac_row + jac_colind[nVar], A_qp_row.get());
//     std::copy(jac_colind, jac_colind + nVar + 1, A_qp_colind.get());
    
//     A_qp = std::make_unique<qpOASES::SparseMatrix>(nCon, nVar, A_qp_row.get(), A_qp_colind.get(), A_qp_nz.get());
    
//     matrices_changed = true;
//     return;
// }

void qpOASES_solver::set_constr(double *const jac_nz, int *const jac_row, int *const jac_colind){
    if (sparseMatrices){
        if (A_qp_nz == nullptr){
            A_qp_nz = std::make_unique<double[]>(jac_colind[nVar]);
            A_qp_row = std::make_unique<int[]>(jac_colind[nVar]);
            A_qp_colind = std::make_unique<int[]>(nVar + 1);
        }
        std::copy(jac_nz, jac_nz + jac_colind[nVar], A_qp_nz.get());
        std::copy(jac_row, jac_row + jac_colind[nVar], A_qp_row.get());
        std::copy(jac_colind, jac_colind + nVar + 1, A_qp_colind.get());
        
        //A_qp = std::make_unique<qpOASES::SparseMatrix>(nCon, nVar, jac_row, jac_colind, jac_nz);
        A_qp = std::make_unique<qpOASES::SparseMatrix>(nCon, nVar, A_qp_row.get(), A_qp_colind.get(), A_qp_nz.get());
    }
    else{
        // steady_clock::time_point T0 = steady_clock::now();
        jacT.Initialize(0.);
        for (int j = 0; j < nVar; j++){
            for (int i = jac_colind[j]; i < jac_colind[j+1]; i++){
                jacT.array[j + jac_row[i]*jacT.m] = jac_nz[i];
            }
        }
        A_qp = std::make_unique<qpOASES::DenseMatrix>(nCon, nVar, nVar, jacT.array);
        // steady_clock::time_point T1 = steady_clock::now();
        // std::cout << "Converting Jacobian to dense took " << duration_cast<microseconds>(T1 - T0) << "\n";
    }
    
    matrices_changed = true;
    return;
}

void qpOASES_solver::set_bounds(const Matrix &lb_x, const Matrix &ub_x, const Matrix &lb_A, const Matrix &ub_A){
    //by default, qpOASES defines +-inifinity as +-1e20 (see qpOASES Constants.hpp), set bounds accordingly
    for (int i = 0; i < nVar; i++){
        if (lb_x(i) > -Qparam->inf)
            lb[i] = lb_x(i);
        else
            lb[i] = -1e20;

        if (ub_x(i) < Qparam->inf)
            ub[i] = ub_x(i);
        else
            ub[i] = 1e20;
    }
    for (int i = 0; i < nCon; i++){
        if (lb_A(i) > -Qparam->inf)
            lbA[i] = lb_A(i);
        else
            lbA[i] = -1e20;

        if (ub_A(i) < Qparam->inf)
            ubA[i] = ub_A(i);
        else
            ubA[i] = 1e20;
    }
    return;
}


void qpOASES_solver::set_hotstart_point(BasicQPsolver *hot_QP){
    if (dynamic_cast<qpOASES_solver*>(hot_QP) != nullptr){
        set_hotstart_point(static_cast<qpOASES_solver*>(hot_QP));
    }
    else throw std::invalid_argument("qpOASES_solver::set_hotstart_point: Expected qpOASES_solver");
}

void qpOASES_solver::set_hotstart_point(qpOASES_solver *hot_QP){
    if (nVar != hot_QP->nVar || nCon != hot_QP->nCon)
        throw std::invalid_argument("Error setting hotstart point: QPs have different dimensions");
    if (this == hot_QP) return;
    *qp = *(hot_QP->qp);
    return;
}

QPresults qpOASES_solver::solve(Matrix &deltaXi, Matrix &lambdaQP){
    double QPtime;
    // steady_clock::time_point T0 = steady_clock::now();
    if (convex_QP)  opts.enableInertiaCorrection = qpOASES::BT_TRUE;
    else            opts.enableInertiaCorrection = qpOASES::BT_FALSE;
    
    qp->setOptions(opts);
    
    // Other variables for qpOASES
    
    //Set time limit to prevent wasting time on ill conditioned QPs:
    // 0 - limit by 2.5*(average solution time), 2 - limit by custom time, else - limit by maximum time set in options
    if (timeLimitType == TimeLimitTypes::past_avg)
        QPtime = std::min(2.5*QPtime_avg, default_time_limit);
    else if (timeLimitType == TimeLimitTypes::custom)
        QPtime = custom_time_limit;
    else
        QPtime = default_time_limit;
    
    QP_it = Qparam->max_QP_it;
    qpOASES::SolutionAnalysis solAna;
    qpOASES::returnValue ret;
    
    if ((qp->getStatus() == qpOASES::QPS_HOMOTOPYQPSOLVED ||
         qp->getStatus() == qpOASES::QPS_SOLVED) && use_hotstart){
        if (matrices_changed)
            ret = qp->hotstart(H_qp.get(), h_qp.get(), A_qp.get(), lb.get(), ub.get(), lbA.get(), ubA.get(), QP_it, &QPtime);
        else
            ret = qp->hotstart(h_qp.get(), lb.get(), ub.get(), lbA.get(), ubA.get(), QP_it, &QPtime);
    }
    else
        ret = qp->init(H_qp.get(), h_qp.get(), A_qp.get(), lb.get(), ub.get(), lbA.get(), ubA.get(), QP_it, &QPtime);
    
    if (!convex_QP && ret == qpOASES::SUCCESSFUL_RETURN){
            *dynamic_cast<qpOASES::SQProblemSchur*>(qpCheck.get()) = *dynamic_cast<qpOASES::SQProblemSchur*>(qp.get());
            ret = solAna.checkCurvatureOnStronglyActiveConstraints(dynamic_cast<qpOASES::SQProblemSchur*>(qpCheck.get()));
    }
    
    if (deltaXi.m != nVar) throw std::invalid_argument("QPsolver.solve: Error in argument deltaXi, wrong matrix size");
    if (lambdaQP.m != nVar + nCon) throw std::invalid_argument("QPsolver.solve: Error in argument lambdaQP, wrong matrix size");
    
    
    // Return codes: 0 - success, 1 - took too long/too many steps, 2 definiteness condition violated or QP unbounded, 3 - QP was infeasible, 4 - other error
    if (ret == qpOASES::SUCCESSFUL_RETURN){
        use_hotstart = true;
        matrices_changed = false;
        
        qp->getPrimalSolution(deltaXi.array);
        qp->getDualSolution(lambdaQP.array);
        if (!skip_timeRecord) record_time(QPtime);
        else skip_timeRecord = false;

        QP_it += 1;
        *qpSave = *qp;
        // steady_clock::time_point T1 = steady_clock::now();
        // std::cout << "Solving the QP took " << duration_cast<microseconds>(T1 - T0) << "\n";
        return QPresults::success;
    }
    
    *qp = *qpSave;
    //if (ret == qpOASES::RET_SETUP_AUXILIARYQP_FAILED)
        QP_it = 1;
    
    if (ret == qpOASES::RET_MAX_NWSR_REACHED)
        return QPresults::time_it_limit_reached;
    else if (ret == qpOASES::RET_HESSIAN_NOT_SPD ||
             ret == qpOASES::RET_HESSIAN_INDEFINITE ||
             ret == qpOASES::RET_INIT_FAILED_UNBOUNDEDNESS ||
             ret == qpOASES::RET_QP_UNBOUNDED ||
             ret == qpOASES::RET_HOTSTART_STOPPED_UNBOUNDEDNESS){
        return QPresults::indef_unbounded;}
    else if (ret == qpOASES::RET_INIT_FAILED_INFEASIBILITY ||
             ret == qpOASES::RET_QP_INFEASIBLE ||
             ret == qpOASES::RET_HOTSTART_STOPPED_INFEASIBILITY){
        return QPresults::infeasible;}
    else if (ret == qpOASES::RET_INIT_FAILED){
        //Hacky workaround for qpOASES sometimes failing of a variable has equal upper and lower bounds with equalities enabled
        int eq_count = 0;
        for (int i = 0; i < nVar; i++){
            if (opts.enableEqualities && std::abs(ub[i] - lb[i]) < opts.boundTolerance){
                double bound_shift = opts.boundTolerance - (ub[i] - lb[i]) + 2*qpOASES::EPS;
                lb[i] -= (std::max)(0.55*bound_shift, std::abs(lb[i])*1e-14); 
                ub[i] += (std::max)(0.55*bound_shift, std::abs(ub[i])*1e-14);
                eq_count++;
            }
        }
        if (eq_count > 0){
            std::cout << "Note: QP solution failed and variables with equal lower and upper bound were detected. This hints at a qpOASES bug. Attempting workaround...\n";
            return solve(deltaXi, lambdaQP);
        }
    }
    return QPresults::other_error;
}

void qpOASES_solver::solve(std::stop_token stopRequest, std::promise<QPresults> QP_result, Matrix &deltaXi, Matrix &lambdaQP){    
    #ifdef SQPROBLEMSCHUR_ENABLE_PASSTHROUGH
        qp->set_stop_token(std::move(stopRequest));
    #endif
    QPresults inner_QP_result = solve(deltaXi, lambdaQP);
    QP_result.set_value(inner_QP_result);
}

int qpOASES_solver::get_QP_it(){return QP_it;}

#endif




//////////////////////
///gurobi interface///
//////////////////////

#ifdef QPSOLVER_GUROBI
gurobi_solver::gurobi_solver(int n_QP_var, int n_QP_con, int n_QP_hessblocks, QPsolver_options *QPopts): QPsolver(n_QP_var, n_QP_con, n_QP_hessblocks, QPopts), obj_lin(0), obj_quad(0){
    if (Qparam->sol != QPsolvers::gurobi) throw ParameterError("Wrong options type given for QPsolver gurobi");
    
    //Check for inconsistent options before construction
    env = new GRBEnv();
    model = new GRBModel(env);
    
    model->set(GRB_IntParam_OutputFlag, static_cast<gurobi_options*>(Qparam)->OutputFlag);
    model->set(GRB_IntParam_Method, static_cast<gurobi_options*>(Qparam)->Method);
    model->set(GRB_IntParam_NumericFocus, static_cast<gurobi_options*>(Qparam)->NumericFocus);
    model->set(GRB_IntParam_Presolve, static_cast<gurobi_options*>(Qparam)->Presolve);
    model->set(GRB_IntParam_Aggregate, static_cast<gurobi_options*>(Qparam)->Aggregate);

    model->set(GRB_DoubleParam_OptimalityTol, static_cast<gurobi_options*>(Qparam)->OptimalityTol);
    model->set(GRB_DoubleParam_FeasibilityTol, static_cast<gurobi_options*>(Qparam)->FeasibilityTol);
    model->set(GRB_DoubleParam_PSDTol, static_cast<gurobi_options*>(Qparam)->PSDTol);

    model->set(GRB_IntParam_NonConvex, 0);

    QP_vars = model->addVars(nVar, GRB_CONTINUOUS);
    QP_cons_lb = model->addConstrs(nCon);
    QP_cons_ub = model->addConstrs(nCon);
    for (int i = 0; i < nCon; i++){
        QP_cons_lb[i].set(GRB_CharAttr_Sense, GRB_GREATER_EQUAL);
        QP_cons_ub[i].set(GRB_CharAttr_Sense, GRB_LESS_EQUAL);
    }
}

gurobi_solver::~gurobi_solver(){
    delete[] QP_vars;
    delete[] QP_cons_lb;
    delete[] QP_cons_ub;
    delete model;
    delete env;
}

void gurobi_solver::set_reg(double arg){
    regF = arg;
    int offset = 0;
    for (int k = 0; k < nHess; k++){
        for (int i = 0; i < hess[k].m; i++){
            obj_quad += 0.5 * QP_vars[offset + i] * QP_vars[offset + i] * regF;
        }
        offset += hess[k].m;
    }
}

void gurobi_solver::set_lin(const Matrix &grad_obj){
    obj_lin = 0;
    obj_lin.addTerms(grad_obj.array, QP_vars, nVar);
    return;
}

void gurobi_solver::set_hess(SymMatrix *const hess, bool pos_def){
    convex_QP = pos_def;
    double regFactor = convex_QP ? convex_regF : 0.0;

    obj_quad = 0;

    int offset = 0;
    for (int k = 0; k < nHess; k++){
        for (int i = 0; i < hess[k].m; i++){
            for (int j = 0; j < i; j++){
                obj_quad += QP_vars[offset + i] * QP_vars[offset + j] * hess[k](i,j);
            }
            obj_quad += 0.5 * QP_vars[offset + i] * QP_vars[offset + i] * (hess[k](i,i) + regFactor);
        }
        offset += hess[k].m;
    }
    
    //Unnecessary right now as gurobi only supplies lagrange multipliers for convex QPs
    //model->set(GRB_IntParam_NonConvex, int(!pos_def));
    return;
}

void gurobi_solver::set_constr(const Matrix &constr_jac){
    for (int i = 0; i < nCon; i++){
        for (int j = 0; j < nVar; j++){
            model->chgCoeff(QP_cons_lb[i], QP_vars[j], constr_jac(i,j));
            model->chgCoeff(QP_cons_ub[i], QP_vars[j], constr_jac(i,j));
        }
    }
    return;
}
void gurobi_solver::set_constr(double *const jac_nz, int *const jac_row, int *const jac_colind){
    for (int j = 0; j < nVar; j++){
        for (int i = jac_colind[j]; i < jac_colind[j+1]; i++){
            model->chgCoeff(QP_cons_lb[jac_row[i]], QP_vars[j], jac_nz[i]);
            model->chgCoeff(QP_cons_ub[jac_row[i]], QP_vars[j], jac_nz[i]);
        }
    }
    return;
}

void gurobi_solver::set_bounds(const Matrix &lb_x, const Matrix &ub_x, const Matrix &lb_A, const Matrix &ub_A){
    for (int i = 0; i < nVar; i++){
        if (lb_x(i) > -Qparam->inf)
            QP_vars[i].set(GRB_DoubleAttr_LB, lb_x(i));
        else
            QP_vars[i].set(GRB_DoubleAttr_LB, -GRB_INFINITY);

        if (ub_x(i) < Qparam->inf)
            QP_vars[i].set(GRB_DoubleAttr_UB, ub_x(i));
        else
            QP_vars[i].set(GRB_DoubleAttr_UB, GRB_INFINITY);
    }
    for (int i = 0; i < nCon; i++){
        if (lb_A(i) > -Qparam->inf)
            QP_cons_lb[i].set(GRB_DoubleAttr_RHS, lb_A(i));
        else
            QP_cons_lb[i].set(GRB_DoubleAttr_RHS, -GRB_INFINITY);

        if (ub_A(i) < Qparam->inf)
            QP_cons_ub[i].set(GRB_DoubleAttr_RHS, ub_A(i));
        else
            QP_cons_ub[i].set(GRB_DoubleAttr_RHS, GRB_INFINITY);
    }
    return;
}


int gurobi_solver::solve(Matrix &deltaXi, Matrix &lambdaQP){
    model->setObjective(obj_quad + obj_lin, GRB_MINIMIZE);

    //Set time limit to prevent wasting time on ill conditioned QPs:
    // 0 - limit by 2.5*(average solution time), 2 - limit by custom time, else - limit by maximum time set in options
    if (timeLimitType == TimeLimitTypes::past_avg)
        model->set(GRB_DoubleParam_TimeLimit, std::min(2.5*QPtime_avg, default_time_limit));
    else if (timeLimitType == TimeLimitTypes::custom)
        model->set(GRB_DoubleParam_TimeLimit, custom_time_limit);
    else
        model->set(GRB_DoubleParam_TimeLimit, default_time_limit);


    try{
        model->optimize();
    }
    catch (GRBException &e){
        return 4;
    }

    int ret = model->get(GRB_IntAttr_Status);
    if (ret == 2){
        for (int i = 0; i < nVar; i++){
            deltaXi(i) = QP_vars[i].get(GRB_DoubleAttr_X);
            lambdaQP(i) = QP_vars[i].get(GRB_DoubleAttr_RC);
        }
        for (int i = 0; i < nCon; i++){
            lambdaQP(nVar + i) = QP_cons_lb[i].get(GRB_DoubleAttr_Pi);
            lambdaQP(nVar + i) += QP_cons_ub[i].get(GRB_DoubleAttr_Pi);
        }

        if (!skip_timeRecord) record_time(model->get(GRB_DoubleAttr_Runtime));
        else skip_timeRecord = false;

        return QPresults::success;
    }
    else if (ret == 3)
        return QPresults::infeasible;
    else if (ret == 4)
        return QPresults::indef_unbounded;
    else if (ret == 7 || ret == 9 || ret == 16)
        return QPresults::time_it_limit_reached;
    return QPresults::other_error;
    /*
    else if (ret == 3)
        return 3;
    else if (ret == 4)
        return 2;
    else if (ret == 7 || ret == 9 || ret == 16)
        return 1;
    return 4;
    */
}


int gurobi_solver::get_QP_it(){
    if (model->get(GRB_IntParam_Method) == 2)
        return model->get(GRB_IntAttr_BarIterCount);
    else if (model->get(GRB_IntParam_Method) == 0 || model->get(GRB_IntParam_Method) == 1)
        return int(model->get(GRB_DoubleAttr_IterCount));
    return model->get(GRB_IntAttr_BarIterCount) + int(model->get(GRB_DoubleAttr_IterCount));
}

#endif



/////////////////////
///qpalm interface///
/////////////////////

#ifdef QPSOLVER_QPALM
qpalm_solver::qpalm_solver(int n_QP_var, int n_QP_con, int n_QP_hessblocks, QPsolver_options *QPopts): QPsolver(n_QP_var, n_QP_con, n_QP_hessblocks, QPopts),
data(qpalm::index_t(n_QP_var), qpalm::index_t(n_QP_var + n_QP_con)), Q(n_QP_var, n_QP_var), q(n_QP_var), A(n_QP_con + n_QP_var, n_QP_var), lb(n_QP_con + n_QP_var), ub(n_QP_con + n_QP_var){
    if (Qparam->sol != QPsolvers::qpalm) throw ParameterError("Wrong options type given for QPsolver qpalm");
    
    settings.eps_abs     = 1e-9;
    settings.eps_abs_in = 1.0e-4;
    settings.eps_rel     = 1e-9;
    settings.eps_rel_in = 1.0e-4;
    settings.eps_prim_inf = 1e-8;
    settings.eps_dual_inf = 1e-6;
    
    settings.max_iter    = 1000;
    settings.inner_max_iter = 100;
    
    settings.verbose = 1;
}
qpalm_solver::~qpalm_solver(){};

void qpalm_solver::set_reg(double arg){
    regF = arg;
    for (int i = 0; i < nVar; i++){
        Q.coeffRef(i,i) += regF;
    }
}

void qpalm_solver::set_lin(const Matrix &grad_obj){
    for (int i = 0; i < nVar; i++){
        q(i) = grad_obj(i);
    }
    return;
}

void qpalm_solver::set_hess(SymMatrix *const hess, bool pos_def){
    convex_QP = pos_def;
    triplets.resize(0);
    int offset = 0;
    double regFactor = convex_QP ? convex_regF : 0.0;
    for (int iBlock = 0; iBlock < nHess; iBlock++){
        for (int i = 0; i < hess[iBlock].m; i++){
            for (int j = 0; j < hess[iBlock].m; j++){
                triplets.push_back(qpalm::triplet_t(offset + i, offset + j, hess[iBlock](i,j) + regFactor*int(i == j)));
            }
        }
        offset += hess[iBlock].m;
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());
    settings.nonconvex = !pos_def;
    return;
}

void qpalm_solver::set_constr(const Matrix &constr_jac){
    triplets.reserve((nVar+nCon)*nCon);
    triplets.resize(0);
    for (int i = 0; i < nCon; i++){
        for (int j = 0; j < nVar; j++){
            triplets.push_back(qpalm::triplet_t(i, j, constr_jac(i,j)));
        }
    }
    for (int i = 0; i < nVar; i++){
        triplets.push_back(qpalm::triplet_t(nCon + i, i, 1.0));
    }
    A.setFromTriplets(triplets.begin(), triplets.end());
    return;
}
void qpalm_solver::set_constr(double *const jac_nz, int *const jac_row, int *const jac_colind){
    triplets.reserve(jac_colind[nVar] + nVar);
    triplets.resize(0);
    for (int j = 0; j < nVar; j++){
        for (int i = jac_colind[j]; i < jac_colind[j+1]; i++){
            triplets.push_back(qpalm::triplet_t(jac_row[i], j, jac_nz[i]));
        }
    }
    for (int i = 0; i < nVar; i++){
        triplets.push_back(qpalm::triplet_t(nCon + i, i, 1.0));
    }
    A.setFromTriplets(triplets.begin(), triplets.end());
    return;
}

void qpalm_solver::set_bounds(const Matrix &lb_x, const Matrix &ub_x, const Matrix &lb_A, const Matrix &ub_A){
    for (int i = 0; i < nCon; i++){
        lb(i) = lb_A(i);
        ub(i) = ub_A(i);
    }
    for (int i = 0; i < nVar; i++){
        lb(nCon + i) = lb_x(i);
        ub(nCon + i) = ub_x(i);
    }
    return;
}


QPresults qpalm_solver::solve(Matrix &deltaXi, Matrix &lambdaQP){
    data.set_Q(Q);
    data.q = q;
    data.set_A(A);
    data.c = 0;
    data.bmin = lb;
    data.bmax = ub;

    //Set time limit to prevent wasting time on ill conditioned QPs:
    // 0 - limit by 2.5*(average solution time), 2 - limit by custom time, else - limit by maximum time set in options
    if (timeLimitType == TimeLimitTypes::past_avg)
        settings.time_limit = std::min(2.5*QPtime_avg, default_time_limit);
    else if (timeLimitType == TimeLimitTypes::custom)
        settings.time_limit = custom_time_limit;
    else
        settings.time_limit = default_time_limit;

    qpalm::Solver solver = {data, settings};
    solver.solve();

    qpalm::SolutionView sol = solver.get_solution();
    info = solver.get_info();
    std::cout << "qpalm returned, info is " << info.status << "\n";
    
    if (!strcmp(info.status, "solved")){ //strcmp is zero if strings are equal
        for (int i = 0; i < nCon; i++){
            //qpalm defines Lagrangian as f + lambda^T g, we have f - lambda^T g. Change sign of Lagrange multipliers.
            lambdaQP(nVar + i) = -sol.y(i);
        }
        for (int i = 0; i < nVar; i++){
            deltaXi(i) = sol.x(i);
            lambdaQP(i) = -sol.y(nCon + i);
        }

        if (!skip_timeRecord) record_time(info.run_time);
        else skip_timeRecord = false;

        return QPresults::success;
    }
    return QPresults::other_error;
}

int qpalm_solver::get_QP_it(){return info.iter;};

#endif



} // namespace blockSQP2



