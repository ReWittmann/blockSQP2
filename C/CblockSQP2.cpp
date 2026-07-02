/*
 * blockSQP2 -- A structure-exploiting nonlinear programming solver based
 *              on blockSQP by Dennis Janka.
 * Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
 * 
 * Licensed under the zlib license. See LICENSE for more details.
 */

/**
 * \file CblockSQP2.cpp
 * \author Reinhold Wittmann
 * \date 2025
 *
 * C interface to the blockSQP2 nonlinear programming solver. 
 * For now intended for dynamic loading, e.g. from blockSQP.jl
 */


#include <blockSQP2/method.hpp>
#include <blockSQP2/condensing.hpp>
#include <blockSQP2/options.hpp>
#include <blockSQP2/problemspec.hpp>
#include <blockSQP2/matrix.hpp>
#include <iostream>
#include <string>
#include <stdexcept>
#include <cstring>
#include "stdlib.h"

using namespace blockSQP2;

#ifdef _MSC_VER
    #define CDLEXP extern "C" __declspec(dllexport)
#else
    #define CDLEXP extern "C" __attribute__((visibility("default")))
#endif

#define MAXLEN_CBLOCKSQP_ERROR_MESSAGE 1000
char CblockSQP_error_message[MAXLEN_CBLOCKSQP_ERROR_MESSAGE + 1];

CDLEXP char *get_error_message(){
    CblockSQP_error_message[MAXLEN_CBLOCKSQP_ERROR_MESSAGE] = '\0';
    return CblockSQP_error_message;
}

class CProblemspec : public Problemspec{
public:
    CProblemspec(int NVARS, int NCONS){
        nVar = NVARS;
        nCon = NCONS;
    };
    
    virtual ~CProblemspec(){
        delete[] blockIdx;
        delete[] vblocks;
    };
    
    // Allocate callbacks (function pointers to global julia functions)
    void (*initialize_dense)(void *closure_pass, double *xi, double *lambda, double *constrJac) = nullptr;
    void (*evaluate_dense)(void *closure_pass, const double *xi, const double *lambda, double *objval, double *constr, double *gradObj, double *constrJac, double **hess, int dmode, int *info) = nullptr;
    void (*evaluate_simple)(void *closure_pass, const double *xi, double *objval, double *constr, int *info) = nullptr;
    
    void (*initialize_sparse)(void *closure_pass, double *xi, double *lambda, double *jacNz, int *jacIndRow, int *jacIndCol) = nullptr;
    void (*evaluate_sparse)(void *closure_pass, const double *xi, const double *lambda, double *objval, double *constr, double *gradObj, double *jacNz, int *jacIndRow, int *jacIndCol, double **hess, int dmode, int *info) = nullptr;
    
    void (*reduce_constr_vio)(void *closure_pass, double *xi, int *info) = nullptr;
    void (*modify_step)(void *closure_pass, double *xi, double *lambda, int *info) = nullptr;
    
    // Pass-through pointer to a closure of the caller, passed to callbacks.
    void *closure = nullptr;
    
    // Invoke callbacks in overridden methods
    virtual void initialize(Matrix &xi, Matrix &lambda, Matrix &constrJac){
        (*initialize_dense)(closure, xi.array, lambda.array, constrJac.array);
    }

    virtual void initialize(Matrix &xi, Matrix &lambda, double *jacNz, int *jacIndRow, int *jacIndCol){
        (*initialize_sparse)(closure, xi.array, lambda.array, jacNz, jacIndRow, jacIndCol);
    }
    
    virtual void evaluate(const Matrix &xi, const Matrix &lambda, double *objval, Matrix &constr, Matrix &gradObj, Matrix &constrJac, SymMatrix *hess, int dmode, int *info){
        double **hessNz = nullptr;
        if (dmode == 3){
            hessNz = new double *[nBlocks];
            for (int i = 0; i < nBlocks; i++){
                hessNz[i] = hess[i].array;
            }
        }
        else if (dmode == 2){
            hessNz = new double *[nBlocks];
            hessNz[nBlocks - 1] = hess[nBlocks - 1].array;
        }
        
        (*evaluate_dense)(closure, xi.array, lambda.array, objval, constr.array, gradObj.array, constrJac.array, hessNz, dmode, info);
        delete[] hessNz;
    }
    
    virtual void evaluate(const Matrix &xi, const Matrix &lambda, double *objval, Matrix &constr, Matrix &gradObj, double *jacNz, int *jacIndRow, int *jacIndCol, SymMatrix *hess, int dmode, int *info){
        double **hessNz = nullptr;
        if (dmode == 3){
            hessNz = new double *[nBlocks];
            for (int i = 0; i < nBlocks; i++){
                hessNz[i] = hess[i].array;
            }
        }
        else if (dmode == 2){
            hessNz = new double *[nBlocks];
            hessNz[nBlocks - 1] = hess[nBlocks - 1].array;
        }
        
        (*evaluate_sparse)(closure, xi.array, lambda.array, objval, constr.array, gradObj.array, jacNz, jacIndRow, jacIndCol, hessNz, dmode, info);
        delete[] hessNz;
    }

    virtual void evaluate(const Matrix &xi, double *objval, Matrix &constr, int *info){
        (*evaluate_simple)(closure, xi.array, objval, constr.array, info);
    }
    
    // Optional Methods
    virtual void reduceConstrVio(Matrix &xi, int *info){
        if (reduce_constr_vio != nullptr){
            (*reduce_constr_vio)(closure, xi.array, info);
        }
        *info = 1;
    };
};

// vblock[size]
CDLEXP void *create_vblock_array(int size){
    return static_cast<void *>(new vblock[size]);
}

CDLEXP void delete_vblock_array(void *ptr){
    delete[] static_cast<vblock *>(ptr);
}

CDLEXP void vblock_array_set(void *ptr, int index, int size, char dependent){
    static_cast<vblock *>(ptr)[index].size = size;
    static_cast<vblock *>(ptr)[index].dependent = bool(dependent);
}

// QPsolver
// No Constructor, only subclass constructors
// Virtual destructor
CDLEXP void delete_QPsolver_options(void *ptr_QPsolver_options){
    delete static_cast<QPsolver_options *>(ptr_QPsolver_options);
}

// qpOASES_options
CDLEXP void *create_qpOASES_options(){
    return static_cast<void *>(new qpOASES_options());
}

CDLEXP void qpOASES_options_set_matrixSparsity(void *opts, int val){
    static_cast<qpOASES_options *>(opts)->matrixSparsity = val;
}

CDLEXP void qpOASES_options_set_printLevel(void *opts, int val){
    static_cast<qpOASES_options *>(opts)->printLevel = val;
}

CDLEXP void qpOASES_options_set_terminationTolerance(void *opts, double val){
    static_cast<qpOASES_options *>(opts)->terminationTolerance = val;
}

// SQPoptions

inline SQPoptions *castOPT(void *ptr){
    return static_cast<SQPoptions *>(ptr);
}

CDLEXP void *create_SQPoptions(){
    return static_cast<void *>(new SQPoptions);
}

CDLEXP void delete_SQPoptions(void *ptr_SQPoptions){
    delete castOPT(ptr_SQPoptions);
}

CDLEXP void SQPoptions_pass_qpsol_options(void *obj, void *qpsol_options_obj){
    static_cast<SQPoptions*>(obj)->pass_qpsol_options(std::unique_ptr<QPsolver_options>(static_cast<QPsolver_options*>(qpsol_options_obj)));
}


CDLEXP void SQPoptions_set_print_level(void *ptr_SQPoptions, int val){
    castOPT(ptr_SQPoptions)->print_level = val;
}

CDLEXP void SQPoptions_set_result_print_color(void *ptr, int val){
    castOPT(ptr)->result_print_color = val;
}
CDLEXP void SQPoptions_set_debug_level(void *ptr, int val){
    castOPT(ptr)->debug_level = val;
}
CDLEXP void SQPoptions_set_eps(void *ptr, double val){
    castOPT(ptr)->eps = val;
}
CDLEXP void SQPoptions_set_inf(void *ptr, double val){
    castOPT(ptr)->inf = val;
}
CDLEXP void SQPoptions_set_opt_tol(void *ptr, double val){
    castOPT(ptr)->opt_tol = val;
}
CDLEXP void SQPoptions_set_feas_tol(void *ptr, double val){
    castOPT(ptr)->feas_tol = val;
}
CDLEXP void SQPoptions_set_sparse(void *ptr, char val){
    castOPT(ptr)->sparse = bool(val);
}
CDLEXP void SQPoptions_set_enable_linesearch(void *ptr, char val){
    castOPT(ptr)->enable_linesearch = bool(val);
}
CDLEXP void SQPoptions_set_enable_rest(void *ptr, char val){
    castOPT(ptr)->enable_rest = bool(val);
}
CDLEXP void SQPoptions_set_rest_rho(void *ptr, double val){
    castOPT(ptr)->rest_rho = val;
}
CDLEXP void SQPoptions_set_rest_zeta(void *ptr, double val){
    castOPT(ptr)->rest_zeta = val;
}
CDLEXP void SQPoptions_set_max_linesearch_steps(void *ptr, int val){
    castOPT(ptr)->max_linesearch_steps = val;
}
CDLEXP void SQPoptions_set_max_consec_reduced_steps(void *ptr, int val){
    castOPT(ptr)->max_consec_reduced_steps = val;
}
CDLEXP void SQPoptions_set_max_consec_skipped_updates(void *ptr, int val){
    castOPT(ptr)->max_consec_skipped_updates = val;
}
CDLEXP void SQPoptions_set_max_QP_it(void *ptr, int val){
    castOPT(ptr)->max_QP_it = val;
}
CDLEXP void SQPoptions_set_block_hess(void *ptr, int val){
    castOPT(ptr)->block_hess = val;
}
// CDLEXP void SQPoptions_set_sizing(void *ptr, int val){
//     castOPT(ptr)->sizing = Sizings(val);
// }
// CDLEXP void SQPoptions_set_fallback_sizing(void *ptr, int val){
//     castOPT(ptr)->fallback_sizing = Sizings(val);
// }
CDLEXP int SQPoptions_set_sizing(void *ptr, char* val){
    try{
        castOPT(ptr)->sizing = Sizings_from_string(std::string(val));
        return 0;
    }
    catch (std::exception &E){
        strncpy(CblockSQP_error_message, E.what(), MAXLEN_CBLOCKSQP_ERROR_MESSAGE);
    }
    CblockSQP_error_message[MAXLEN_CBLOCKSQP_ERROR_MESSAGE] = '\0';
    return 1;
}
CDLEXP int SQPoptions_set_fallback_sizing(void *ptr, char* val){
    try{
        castOPT(ptr)->fallback_sizing = Sizings_from_string(std::string(val));
        return 0;
    }
    catch (std::exception &E){
        strncpy(CblockSQP_error_message, E.what(), MAXLEN_CBLOCKSQP_ERROR_MESSAGE);
    }
    CblockSQP_error_message[MAXLEN_CBLOCKSQP_ERROR_MESSAGE] = '\0';
    return 1;
}

CDLEXP void SQPoptions_set_max_QP_secs(void *ptr, double val){
    castOPT(ptr)->max_QP_secs = val;
}
CDLEXP void SQPoptions_set_initial_hess_scale(void *ptr, double val){
    castOPT(ptr)->initial_hess_scale = val;
}
CDLEXP void SQPoptions_set_COL_eps(void *ptr, double val){
    castOPT(ptr)->COL_eps = val;
}
CDLEXP void SQPoptions_set_OL_eps(void *ptr, double val){
    castOPT(ptr)->OL_eps = val;
}
CDLEXP void SQPoptions_set_COL_tau_1(void *ptr, double val){
    castOPT(ptr)->COL_tau_1 = val;
}
CDLEXP void SQPoptions_set_COL_tau_2(void *ptr, double val){
    castOPT(ptr)->COL_tau_2 = val;
}
CDLEXP void SQPoptions_set_BFGS_damping_factor(void *ptr, double val){
    castOPT(ptr)->BFGS_damping_factor = val;
}
CDLEXP void SQPoptions_set_min_damping_quotient(void *ptr, double val){
    castOPT(ptr)->min_damping_quotient = val;
}
// CDLEXP void SQPoptions_set_hess_approx(void *ptr, int val){
//     castOPT(ptr)->hess_approx = Hessians(val);
// }
// CDLEXP void SQPoptions_set_fallback_approx(void *ptr, int val){
//     castOPT(ptr)->fallback_approx = Hessians(val);
// }

CDLEXP int SQPoptions_set_hess_approx(void *ptr, char* val){
    try{
        castOPT(ptr)->hess_approx = Hessians_from_string(std::string(val));
        return 0;
    }
    catch (std::exception &E){
        strncpy(CblockSQP_error_message, E.what(), MAXLEN_CBLOCKSQP_ERROR_MESSAGE);
    }
    CblockSQP_error_message[MAXLEN_CBLOCKSQP_ERROR_MESSAGE] = '\0';
    return 1;
}
CDLEXP int SQPoptions_set_fallback_approx(void *ptr, char* val){
    try{
        castOPT(ptr)->fallback_approx = Hessians_from_string(std::string(val));
        return 0;
    }
    catch (std::exception &E){
        strncpy(CblockSQP_error_message, E.what(), MAXLEN_CBLOCKSQP_ERROR_MESSAGE);
    }
    CblockSQP_error_message[MAXLEN_CBLOCKSQP_ERROR_MESSAGE] = '\0';
    return 1;
}
CDLEXP void SQPoptions_set_indef_local_only(void *ptr, char val){
    castOPT(ptr)->indef_local_only = bool(val);
}
CDLEXP void SQPoptions_set_lim_mem(void *ptr, char val){
    castOPT(ptr)->lim_mem = bool(val);
}
CDLEXP void SQPoptions_set_mem_size(void *ptr, int val){
    castOPT(ptr)->mem_size = val;
}
// CDLEXP void SQPoptions_set_exact_hess(void *ptr, int val){
//     castOPT(ptr)->exact_hess = val;
// }
CDLEXP void SQPoptions_set_skip_first_linesearch(void *ptr, int val){
    castOPT(ptr)->skip_first_linesearch = val;
}
CDLEXP void SQPoptions_set_conv_strategy(void *ptr, int val){
    castOPT(ptr)->conv_strategy = val;
}
CDLEXP void SQPoptions_set_max_conv_QPs(void *ptr, int val){
    castOPT(ptr)->max_conv_QPs = val;
}
CDLEXP void SQPoptions_set_conv_kappa_0(void *ptr, double val){
    castOPT(ptr)->conv_kappa_0 = val;
}
CDLEXP void SQPoptions_set_conv_kappa_max(void *ptr, double val){
    castOPT(ptr)->conv_kappa_max = val;
}
CDLEXP void SQPoptions_set_hess_regularization_factor(void *ptr, double val){
    castOPT(ptr)->reg_factor = val;
}
CDLEXP void SQPoptions_set_max_SOC(void *ptr, int val){
    castOPT(ptr)->max_SOC = val;
}
CDLEXP void SQPoptions_set_qpsol_options(void *ptr, QPsolver_options *QPopts){
    castOPT(ptr)->qpsol_options = QPopts;
}
CDLEXP void SQPoptions_set_automatic_scaling(void *ptr, char val){
    castOPT(ptr)->automatic_scaling = bool(val);
}
CDLEXP void SQPoptions_set_scaling_Theta_min(void *ptr, double val){
    castOPT(ptr)->scaling_Theta_min = val;
}
CDLEXP void SQPoptions_set_scaling_Theta_max(void *ptr, double val){
    castOPT(ptr)->scaling_Theta_max = val;
}
CDLEXP void SQPoptions_set_max_filter_overrides(void *ptr, int val){
    castOPT(ptr)->max_filter_overrides = val;
}
CDLEXP void SQPoptions_set_max_extra_steps(void *ptr, int val){
    castOPT(ptr)->max_extra_steps = val;
}
CDLEXP void SQPoptions_set_par_QPs(void *ptr, char val){
    castOPT(ptr)->par_QPs = bool(val);
}
CDLEXP void SQPoptions_set_enable_QP_cancellation(void *ptr, char val){
    castOPT(ptr)->enable_QP_cancellation = bool(val);
}
CDLEXP void SQPoptions_set_enable_premature_termination(void *ptr, char val){
    castOPT(ptr)->enable_premature_termination = bool(val);
}
CDLEXP void SQPoptions_set_qpsol(void *ptr, int val){
    QPsolvers QPS;
    if (val == 0)
        QPS = QPsolvers::qpOASES;
    else if (val == 1)
        QPS = QPsolvers::gurobi;
    else
        QPS = QPsolvers::unset;
    castOPT(ptr)->qpsol = QPS;
}
CDLEXP void SQPoptions_set_indef_delay(void *ptr, int val){
    castOPT(ptr)->indef_delay = val;
}

CDLEXP void SQPoptions_set_test_opt_1(void *ptr, char val){
    castOPT(ptr)->test_opt_1 = bool(val);
}

CDLEXP void SQPoptions_set_test_opt_2(void *ptr, char val){
    castOPT(ptr)->test_opt_2 = bool(val);
}

CDLEXP void SQPoptions_set_test_opt_3(void *ptr, char val){
    castOPT(ptr)->test_opt_3 = bool(val);
}

CDLEXP void SQPoptions_set_test_val_1(void *ptr, double val){
    castOPT(ptr)->test_val_1 = double(val);
}

CDLEXP void SQPoptions_set_test_val_2(void *ptr, double val){
    castOPT(ptr)->test_val_2 = double(val);
}

// SQPstats
CDLEXP void *create_SQPstats(char *pathstr){
    return static_cast<void *>(new SQPstats(pathstr));
}

CDLEXP void delete_SQPstats(void *ptr){
    delete static_cast<SQPstats *>(ptr);
}

CDLEXP int SQPstats_get_itCount(void *ptr){
    return static_cast<SQPstats *>(ptr)->itCount;
}

// Problemspec (C callback subclass)
inline CProblemspec *castCP(void *ptr){
    return static_cast<CProblemspec *>(ptr);
}

CDLEXP void *create_Problemspec(int nVar, int nCon){
    return static_cast<void *>(new CProblemspec(nVar, nCon));
}

CDLEXP void delete_Problemspec(void *ptr){
    delete castCP(ptr);
}

CDLEXP void Problemspec_print_info(void *ptr){
    Problemspec *P = castCP(ptr);
    std::cout << "\nnVar: " << P->nVar << "\nnCon: " << P->nCon << "\nnBlocks: " << P->nBlocks << "\nnnz: " << P->nnz << "\nblockIdx: ";
    for (int i = 0; i <= P->nBlocks; i++)
    {
        std::cout << P->blockIdx[i] << ", ";
    }
    std::cout << "\n";
}

CDLEXP void Problemspec_set_nnz(void *ptr, int nnz){
    castCP(ptr)->nnz = nnz;
}

CDLEXP int Problemspec_get_nVar(void *ptr){
    return static_cast<Problemspec*>(ptr)->nVar;
}
CDLEXP int Problemspec_get_nCon(void *ptr){
    return static_cast<Problemspec*>(ptr)->nCon;
}
CDLEXP int Problemspec_get_nnz(void *ptr){
    return static_cast<Problemspec*>(ptr)->nnz;
}
CDLEXP int Problemspec_get_nBlocks(void *ptr){
    return static_cast<Problemspec*>(ptr)->nBlocks;
}
CDLEXP int *Problemspec_get_blockIdx(void *ptr){
    return static_cast<Problemspec*>(ptr)->blockIdx;
}


CDLEXP void Problemspec_set_bounds(void *ptr, double *arg_lb_var, double *arg_ub_var, double *arg_lb_con, double *arg_ub_con, double arg_lb_obj, double arg_ub_obj){
    castCP(ptr)->objLo = arg_lb_obj;
    castCP(ptr)->objUp = arg_ub_obj;

    castCP(ptr)->lb_var.Dimension(castCP(ptr)->nVar);
    castCP(ptr)->ub_var.Dimension(castCP(ptr)->nVar);
    castCP(ptr)->lb_con.Dimension(castCP(ptr)->nCon);
    castCP(ptr)->ub_con.Dimension(castCP(ptr)->nCon);

    std::copy(arg_lb_var, arg_lb_var + castCP(ptr)->nVar, castCP(ptr)->lb_var.array);
    std::copy(arg_ub_var, arg_ub_var + castCP(ptr)->nVar, castCP(ptr)->ub_var.array);

    std::copy(arg_lb_con, arg_lb_con + castCP(ptr)->nCon, castCP(ptr)->lb_con.array);
    std::copy(arg_ub_con, arg_ub_con + castCP(ptr)->nCon, castCP(ptr)->ub_con.array);
    return;
}

CDLEXP void Problemspec_set_blockIdx(void *ptr, int *arg_blockIdx, int arg_nBlocks){
    castCP(ptr)->nBlocks = arg_nBlocks;
    delete[] castCP(ptr)->blockIdx;
    castCP(ptr)->blockIdx = new int[arg_nBlocks + 1];
    std::copy(arg_blockIdx, arg_blockIdx + arg_nBlocks + 1, castCP(ptr)->blockIdx);
}

CDLEXP void Problemspec_set_vblocks(void *ptr, void *arg_vblocks, int arg_n_vblocks){
    delete[] castCP(ptr)->vblocks;
    castCP(ptr)->n_vblocks = arg_n_vblocks;
    castCP(ptr)->vblocks = new vblock[arg_n_vblocks];
    std::copy(static_cast<vblock *>(arg_vblocks), static_cast<vblock *>(arg_vblocks) + arg_n_vblocks, castCP(ptr)->vblocks);
}

CDLEXP void Problemspec_pass_vblocks(void *ptr, void *arg_vblocks, int arg_n_vblocks){
    delete[] castCP(ptr)->vblocks;
    castCP(ptr)->n_vblocks = arg_n_vblocks;
    castCP(ptr)->vblocks = static_cast<vblock *>(arg_vblocks);
}

CDLEXP void Problemspec_set_condenser(void *ptr, void *Condenser_obj){
    castCP(ptr)->condenser = static_cast<Condenser*>(Condenser_obj);
}

CDLEXP void Problemspec_set_closure(void *ptr, void *arg_closure){
    castCP(ptr)->closure = arg_closure;
}

CDLEXP void Problemspec_set_dense_init(void *ptr, void (*fp_init_dense)(void *closure_pass, double *xi, double *lambda, double *constrJac)){
    castCP(ptr)->initialize_dense = fp_init_dense;
}

CDLEXP void Problemspec_set_dense_eval(void *ptr, void (*fp_eval_dense)(void *closure_pass, const double *xi, const double *lambda, double *objval, double *constr, double *gradObj, double *constrJac, double **hess, int dmode, int *info)){
    castCP(ptr)->evaluate_dense = fp_eval_dense;
}

CDLEXP void Problemspec_set_simple_eval(void *ptr, void (*fp_eval_simple)(void *closure_pass, const double *xi, double *objval, double *constr, int *info)){
    castCP(ptr)->evaluate_simple = fp_eval_simple;
}

CDLEXP void Problemspec_set_sparse_init(void *ptr, void (*fp_init_sparse)(void *closure_pass, double *xi, double *lambda, double *jacNz, int *jacIndRow, int *jacIndCol)){
    castCP(ptr)->initialize_sparse = fp_init_sparse;
}

CDLEXP void Problemspec_set_sparse_eval(void *ptr, void (*fp_eval_sparse)(void *closure_pass, const double *xi, const double *lambda, double *objval, double *constr, double *gradObj, double *jacNz, int *jacIndRow, int *jacIndCol, double **hess, int dmode, int *info)){
    castCP(ptr)->evaluate_sparse = fp_eval_sparse;
}

CDLEXP void Problemspec_set_reduce_constr_vio(void *ptr, void (*fp_red_vio)(void *closure_pass, double *xi, int *info)){
    castCP(ptr)->reduce_constr_vio = fp_red_vio;
}

CDLEXP void Problemspec_set_modify_step(void *ptr, void (*fp_mod_step)(void *closure_pass, double *xi, double *lambda, int *info)){
    castCP(ptr)->modify_step = fp_mod_step;
}


CDLEXP void *create_scaled_Problemspec(void *parent){
    return static_cast<void *>(new scaled_Problemspec(castCP(parent)));
}

CDLEXP void scaled_Problemspec_set_scale(void *ptr, double *scaleFactors){
    static_cast<scaled_Problemspec*>(ptr)->set_scale(scaleFactors);
}

// CDLEXP void delete_Scaled_Problemspec(void *ptr){
//     delete_Problemspec(ptr);
// }


// SQPmethod
CDLEXP void *create_SQPmethod(void *Problemspec_prob, void *SQPoptions_opts, void *SQPstats_stats){
    try{
        return static_cast<void *>(new SQPmethod(static_cast<Problemspec *>(Problemspec_prob), castOPT(SQPoptions_opts), static_cast<SQPstats *>(SQPstats_stats)));
    }
    catch (std::exception &E){
        strncpy(CblockSQP_error_message, E.what(), MAXLEN_CBLOCKSQP_ERROR_MESSAGE);
    }
    CblockSQP_error_message[MAXLEN_CBLOCKSQP_ERROR_MESSAGE] = '\0';
    return nullptr;
}

CDLEXP void delete_SQPmethod(void *ptr){
    delete static_cast<SQPmethod *>(ptr);
}

CDLEXP void SQPmethod_init(void *ptr){
    static_cast<SQPmethod *>(ptr)->init();
}

CDLEXP int SQPmethod_run(void *ptr, int maxIt, int warmStart){
    try{
        return static_cast<int>(static_cast<SQPmethod *>(ptr)->run(maxIt, warmStart));
    }
    catch (std::exception &E){
        strncpy(CblockSQP_error_message, E.what(), MAXLEN_CBLOCKSQP_ERROR_MESSAGE);
    }
    CblockSQP_error_message[MAXLEN_CBLOCKSQP_ERROR_MESSAGE] = '\0';
    return -1000;
}

CDLEXP void SQPmethod_finish(void *ptr){
    static_cast<SQPmethod *>(ptr)->finish();
}

CDLEXP void SQPmethod_get_xi(void *ptr, double *ret_xi){
    Matrix xi(static_cast<SQPmethod *>(ptr)->get_xi());
    std::copy(xi.array, xi.array + xi.m, ret_xi);
}

CDLEXP void SQPmethod_get_lambda(void *ptr, double *ret_lambda){
    Matrix lambda(static_cast<SQPmethod *>(ptr)->get_lambda());
    std::copy(lambda.array, lambda.array + lambda.m, ret_lambda);
}

CDLEXP double *SQPmethod_get_hess1_block(void *ptr, int ind){
    return static_cast<SQPmethod *>(ptr)->vars->hess1[ind].array;
}
CDLEXP double *SQPmethod_get_hess2_block(void *ptr, int ind){
    return static_cast<SQPmethod *>(ptr)->vars->hess2[ind].array;
}


// cblock[size]
CDLEXP void *create_cblock_array(int size){
    return static_cast<void *>(new cblock[size]);
}

CDLEXP void delete_cblock_array(void *ptr){
    delete[] static_cast<cblock *>(ptr);
}

CDLEXP void cblock_array_set(void *ptr, int index, int size){
    static_cast<cblock *>(ptr)[index].size = size;
}

// hsize[size] ~ int[size]
CDLEXP void *create_hsize_array(int size){
    return static_cast<void *>(new int[size]);
}

CDLEXP void delete_hsize_array(void *ptr){
    delete[] static_cast<int *>(ptr);
}

CDLEXP void hsize_array_set(void *ptr, int index, int size){
    static_cast<int *>(ptr)[index] = size;
}

// condensing_target[]
CDLEXP void *create_target_array(int size){
    return static_cast<void *>(new condensing_target[size]);
}

CDLEXP void delete_target_array(void *ptr){
    delete[] static_cast<condensing_target *>(ptr);
}

CDLEXP void target_array_set(void *ptr, int index, int n_stages, int vblock_start, int vblock_end, int cblock_start, int cblock_end){
    static_cast<condensing_target *>(ptr)[index].n_stages = n_stages;
    static_cast<condensing_target *>(ptr)[index].vblock_start = vblock_start;
    static_cast<condensing_target *>(ptr)[index].vblock_end = vblock_end;
    static_cast<condensing_target *>(ptr)[index].cblock_start = cblock_start;
    static_cast<condensing_target *>(ptr)[index].cblock_end = cblock_end;
}

// Condenser
inline Condenser *castCND(void *ptr){
    return static_cast<Condenser *>(ptr);
}

CDLEXP void *create_Condenser(void *arg_vblocks, int N_vblocks, void *arg_cblocks, int N_cblocks, void *arg_hsizes, int N_hsizes, void *arg_targets, int N_targets, int arg_dep_bounds){
    // return new Condenser(static_cast<vblock *>(arg_vblocks), N_vblocks, static_cast<cblock *>(arg_cblocks), N_cblocks, static_cast<int *>(arg_hsizes), N_hsizes, static_cast<condensing_target *>(arg_targets), N_targets, arg_dep_bounds);
    try{
        return static_cast<void*>(new Condenser(static_cast<vblock *>(arg_vblocks), N_vblocks, static_cast<cblock *>(arg_cblocks), N_cblocks, static_cast<int *>(arg_hsizes), N_hsizes, static_cast<condensing_target *>(arg_targets), N_targets, arg_dep_bounds));
        // return static_cast<int>(static_cast<SQPmethod *>(ptr)->run(maxIt, warmStart));
    }
    catch (std::exception &E){
        strncpy(CblockSQP_error_message, E.what(), MAXLEN_CBLOCKSQP_ERROR_MESSAGE);
    }
    CblockSQP_error_message[MAXLEN_CBLOCKSQP_ERROR_MESSAGE] = '\0';
    return nullptr;
}

CDLEXP void delete_Condenser(void *ptr){
    delete castCND(ptr);
}

CDLEXP void Condenser_print_info(void *ptr){
    castCND(ptr)->print_info();
}

CDLEXP int Condenser_full_condense(void *ptr, void *Matrix_grad_obj, void *Sparse_Matrix_constr_jac, void *SymMatrix_array_hess, void *Matrix_lb_var, void *Matrix_ub_var, void *Matrix_lb_con, void *Matrix_ub_con, void *Matrix_condensed_grad_obj, void *Sparse_Matrix_condensed_constr_jac, void *SymMatrix_array_condensed_hess, void *Matrix_condensed_lb_var, void *Matrix_condensed_ub_var, void *Matrix_condensed_lb_con, void *Matrix_condensed_ub_con){
    try{
        castCND(ptr)->full_condense(*static_cast<Matrix *>(Matrix_grad_obj), *static_cast<Sparse_Matrix *>(Sparse_Matrix_constr_jac), static_cast<SymMatrix *>(SymMatrix_array_hess), *static_cast<Matrix *>(Matrix_lb_var), *static_cast<Matrix *>(Matrix_ub_var), *static_cast<Matrix *>(Matrix_lb_con), *static_cast<Matrix *>(Matrix_ub_con),
                                    *static_cast<Matrix *>(Matrix_condensed_grad_obj), *static_cast<Sparse_Matrix *>(Sparse_Matrix_condensed_constr_jac), static_cast<SymMatrix *>(SymMatrix_array_condensed_hess), *static_cast<Matrix *>(Matrix_condensed_lb_var), *static_cast<Matrix *>(Matrix_condensed_ub_var), *static_cast<Matrix *>(Matrix_condensed_lb_con), *static_cast<Matrix *>(Matrix_condensed_ub_con));
        return 0;
    }
    catch (std::exception &E){
        strncpy(CblockSQP_error_message, E.what(), MAXLEN_CBLOCKSQP_ERROR_MESSAGE);
    }
    CblockSQP_error_message[MAXLEN_CBLOCKSQP_ERROR_MESSAGE] = '\0';
    return 1;
}

CDLEXP void Condenser_recover_var_mult(void *ptr, void *xi_cond, void *lambda_cond, void *xi_rest, void *lambda_rest){
    castCND(ptr)->recover_var_mult(*static_cast<Matrix *>(xi_cond), *static_cast<Matrix *>(lambda_cond), *static_cast<Matrix *>(xi_rest), *static_cast<Matrix *>(lambda_rest));
}

// Member access
CDLEXP int Condenser_nVar(void *ptr){
    return castCND(ptr)->num_vars;
}

CDLEXP int Condenser_nCon(void *ptr){
    return castCND(ptr)->num_cons;
}

CDLEXP int Condenser_num_true_cons(void *ptr){
    return castCND(ptr)->num_true_cons;
}


CDLEXP int Condenser_nBlocks(void *ptr){
    return castCND(ptr)->num_hessblocks;
}

CDLEXP int *Condenser_hsizes(void *ptr){
    return castCND(ptr)->hess_block_sizes;
}

CDLEXP int Condenser_condensed_nVar(void *ptr){
    return castCND(ptr)->condensed_num_vars;
}

CDLEXP int Condenser_condensed_nCon(void *ptr){
    return castCND(ptr)->condensed_num_cons;
}

CDLEXP int Condenser_condensed_nBlocks(void *ptr){
    return castCND(ptr)->condensed_num_hessblocks;
}

CDLEXP int *Condenser_condensed_hsizes(void *ptr){
    return castCND(ptr)->condensed_hess_block_sizes.get();
}


CDLEXP void *create_PartialCondenser(void *arg_vblocks, int N_vblocks, void *arg_cblocks, int N_cblocks, void *arg_hsizes, int N_hsizes, void *arg_targets, int N_targets, int n_split, int arg_dep_bounds){
    try{
        return static_cast<void*>(new PartialCondenser(static_cast<vblock *>(arg_vblocks), N_vblocks, static_cast<cblock *>(arg_cblocks), N_cblocks, static_cast<int *>(arg_hsizes), N_hsizes, static_cast<condensing_target *>(arg_targets), N_targets, n_split, arg_dep_bounds));
    }
    catch (std::exception &E){
        strncpy(CblockSQP_error_message, E.what(), MAXLEN_CBLOCKSQP_ERROR_MESSAGE);
    }
    CblockSQP_error_message[MAXLEN_CBLOCKSQP_ERROR_MESSAGE] = '\0';
    return nullptr;
}


// Matrix
CDLEXP void *create_Matrix(int m, int n){
    return static_cast<void *>(new Matrix(m, n));
}

CDLEXP void *create_Matrix_default(){
    return static_cast<void *>(new Matrix());
}

CDLEXP void delete_Matrix(void *ptr){
    delete static_cast<Matrix *>(ptr);
}

CDLEXP double *Matrix_array(void *ptr){
    return static_cast<Matrix *>(ptr)->array;
}

// SymMatrix
CDLEXP void *create_SymMatrix(int m){
    return static_cast<void *>(new SymMatrix(m));
}

CDLEXP void delete_SymMatrix(void *ptr){
    delete static_cast<SymMatrix *>(ptr);
}

CDLEXP double *SymMatrix_show_array(void *ptr){
    return static_cast<Matrix *>(ptr)->array;
}

// SymMatrix[]
CDLEXP void *create_SymMatrix_array(int size){
    return static_cast<void *>(new SymMatrix[size]);
}

CDLEXP void delete_SymMatrix_array(void *ptr){
    delete[] static_cast<SymMatrix *>(ptr);
}

CDLEXP void SymMatrix_array_index_resize(void *ptr, int index, int m){
    static_cast<SymMatrix *>(ptr)[index].Dimension(m);
}

CDLEXP double *SymMatrix_array_index_array(void *ptr, int index){
    return static_cast<SymMatrix *>(ptr)[index].array;
}

// Sparse_Matrix
CDLEXP void *create_Sparse_Matrix(int m, int n, int nnz){
    return static_cast<void *>(new Sparse_Matrix(m, n, nnz));
}

CDLEXP void *create_Sparse_Matrix_default(){
    return static_cast<void *>(new Sparse_Matrix());
}

CDLEXP void Sparse_Matrix_set_structure(void *ptr, int m, int n, int nnz){
    if (static_cast<Sparse_Matrix *>(ptr)->m != m || static_cast<Sparse_Matrix *>(ptr)->n != n || static_cast<Sparse_Matrix *>(ptr)->colind[static_cast<Sparse_Matrix *>(ptr)->n] != nnz)
        *static_cast<Sparse_Matrix *>(ptr) = Sparse_Matrix(m, n, nnz);
}

CDLEXP void delete_Sparse_Matrix(void *ptr){
    delete static_cast<Sparse_Matrix *>(ptr);
}

CDLEXP int Sparse_Matrix_nnz(void *ptr){
    return static_cast<Sparse_Matrix *>(ptr)->colind[static_cast<Sparse_Matrix *>(ptr)->n];
}

CDLEXP double *Sparse_Matrix_nz(void *ptr){
    return static_cast<Sparse_Matrix *>(ptr)->nz.get();
}

CDLEXP int *Sparse_Matrix_row(void *ptr){
    return static_cast<Sparse_Matrix *>(ptr)->row.get();
}

CDLEXP int *Sparse_Matrix_colind(void *ptr){
    return static_cast<Sparse_Matrix *>(ptr)->colind.get();
}



CDLEXP void *create_BoundCorrectionSolver(void *Problemspec_prob, void *SQPoptions_opts, void *SQPstats_stats){
    try{
        return static_cast<void *>(new bound_correction_method(static_cast<Problemspec *>(Problemspec_prob), castOPT(SQPoptions_opts), static_cast<SQPstats *>(SQPstats_stats)));
    }
    catch (std::exception &E){
        strncpy(CblockSQP_error_message, E.what(), MAXLEN_CBLOCKSQP_ERROR_MESSAGE);
    }
    CblockSQP_error_message[MAXLEN_CBLOCKSQP_ERROR_MESSAGE] = '\0';
    return nullptr;
}

CDLEXP void *create_TCfeasibilityProblem(void *parent){
    return static_cast<void*>(new TC_feasibility_Problem(static_cast<Problemspec*>(parent)));
}


//Some utility functions
CDLEXP void lower_to_full(double *full, double const *lower, int n){
    for (int i = 0; i < n; i++){
        for (int j = 0; j < i; j++){
            full[i + j*n] = lower[i + j*n - (j*(j+1))/2];
        }
        for (int j = i; j < n; j++){
            full[i + j*n] = lower[j + i*n - (i*(i+1))/2];
        }
    }   
}

CDLEXP void full_to_lower(double *lower, double const *full, int n){
    for (int i = 0; i < n; i++){
        for (int j = 0; j <= i; j++){
            lower[i + j*n - (j*(j+1))/2] = full[i + j*n];
        }
    }
}