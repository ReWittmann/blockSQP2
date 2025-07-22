/**
 * \file blockSQP_julbind.cpp
 * \author Reinhold Wittmann
 * \date 2024-
 *
 * C interface for the blockSQP nonlinear programming solver
 */

#include "blocksqp_method.hpp"
#include "blocksqp_condensing.hpp"
#include "blocksqp_options.hpp"
#include "blocksqp_problemspec.hpp"
#include "blocksqp_matrix.hpp"
#include <iostream>
#include <string>
#include <stdexcept>
#include <cstring>
#include "stdlib.h"

using namespace blockSQP;

#ifdef _MSC_VER
#define CDLEXP extern "C" __declspec(dllexport)
#else
#define CDLEXP extern "C" __attribute__((visibility("default")))
#endif

#define MAXLEN_CBLOCKSQP_ERROR_MESSAGE 1000
char CblockSQP_error_message[MAXLEN_CBLOCKSQP_ERROR_MESSAGE];

CDLEXP char *get_error_message(){
    return CblockSQP_error_message;
}

class CProblemspec : public blockSQP::Problemspec{
public:
    CProblemspec(int NVARS, int NCONS)
    {
        nVar = NVARS;
        nCon = NCONS;
    };

    virtual ~CProblemspec()
    {
        delete[] blockIdx;
        delete[] vblocks;
    };

    // Allocate callbacks (function pointers to global julia functions)
    void (*initialize_dense)(void *closure_pass, double *xi, double *lambda, double *constrJac);
    void (*evaluate_dense)(void *closure_pass, const double *xi, const double *lambda, double *objval, double *constr, double *gradObj, double *constrJac, double **hess, int dmode, int *info);
    void (*evaluate_simple)(void *closure_pass, const double *xi, double *objval, double *constr, int *info);

    void (*initialize_sparse)(void *closure_pass, double *xi, double *lambda, double *jacNz, int *jacIndRow, int *jacIndCol);
    void (*evaluate_sparse)(void *closure_pass, const double *xi, const double *lambda, double *objval, double *constr, double *gradObj, double *jacNz, int *jacIndRow, int *jacIndCol, double **hess, int dmode, int *info);

    void (*restore_continuity)(void *closure_pass, double *xi, int *info);

    // Pass-through pointer to a closure of the caller, currently for all callbacks.
    void *closure;

    // Invoke callbacks in overridden methods
    virtual void initialize(blockSQP::Matrix &xi, blockSQP::Matrix &lambda, blockSQP::Matrix &constrJac)
    {
        (*initialize_dense)(closure, xi.array, lambda.array, constrJac.array);
    }

    virtual void initialize(blockSQP::Matrix &xi, blockSQP::Matrix &lambda, double *jacNz, int *jacIndRow, int *jacIndCol)
    {
        (*initialize_sparse)(closure, xi.array, lambda.array, jacNz, jacIndRow, jacIndCol);
    }

    virtual void evaluate(const blockSQP::Matrix &xi, const blockSQP::Matrix &lambda, double *objval, blockSQP::Matrix &constr, blockSQP::Matrix &gradObj, blockSQP::Matrix &constrJac, blockSQP::SymMatrix *hess, int dmode, int *info)
    {
        double **hessNz = nullptr;
        if (dmode == 3)
        {
            hessNz = new double *[nBlocks];
            for (int i = 0; i < nBlocks; i++)
            {
                hessNz[i] = hess[i].array;
            }
        }
        else if (dmode == 2)
        {
            hessNz = new double *[nBlocks];
            hessNz[nBlocks - 1] = hess[nBlocks - 1].array;
        }

        (*evaluate_dense)(closure, xi.array, lambda.array, objval, constr.array, gradObj.array, constrJac.array, hessNz, dmode, info);
        delete[] hessNz;
    }

    virtual void evaluate(const blockSQP::Matrix &xi, const blockSQP::Matrix &lambda, double *objval, blockSQP::Matrix &constr, blockSQP::Matrix &gradObj, double *jacNz, int *jacIndRow, int *jacIndCol, blockSQP::SymMatrix *hess, int dmode, int *info)
    {
        double **hessNz = nullptr;
        if (dmode == 3)
        {
            hessNz = new double *[nBlocks];
            for (int i = 0; i < nBlocks; i++)
            {
                hessNz[i] = hess[i].array;
            }
        }
        else if (dmode == 2)
        {
            hessNz = new double *[nBlocks];
            hessNz[nBlocks - 1] = hess[nBlocks - 1].array;
        }

        (*evaluate_sparse)(closure, xi.array, lambda.array, objval, constr.array, gradObj.array, jacNz, jacIndRow, jacIndCol, hessNz, dmode, info);

        delete[] hessNz;
    }

    virtual void evaluate(const blockSQP::Matrix &xi, double *objval, blockSQP::Matrix &constr, int *info)
    {
        (*evaluate_simple)(closure, xi.array, objval, constr.array, info);
    }

    // Optional Methods
    virtual void reduceConstrVio(blockSQP::Matrix &xi, int *info)
    {
        if (restore_continuity != nullptr)
        {
            (*restore_continuity)(closure, xi.array, info);
        }
    };
};

CDLEXP void *create_vblock_array(int size){
    return (void *)new vblock[size];
}

CDLEXP void delete_vblock_array(void *ptr){
    delete[] static_cast<vblock *>(ptr);
}

CDLEXP void vblock_array_set(void *ptr, int index, int size, char dependent){
    static_cast<vblock *>(ptr)[index].size = size;
    static_cast<vblock *>(ptr)[index].dependent = bool(dependent);
}

CDLEXP void delete_QPsolver_options(void *ptr_QPsolver_options){
    delete static_cast<QPsolver_options *>(ptr_QPsolver_options);
}

CDLEXP void *create_qpOASES_options(){
    return static_cast<void *>(new blockSQP::qpOASES_options());
}

CDLEXP void qpOASES_options_set_sparsityLevel(void *opts, int val){
    static_cast<blockSQP::qpOASES_options *>(opts)->sparsityLevel = val;
}

CDLEXP void qpOASES_options_set_printLevel(void *opts, int val){
    static_cast<blockSQP::qpOASES_options *>(opts)->printLevel = val;
}

CDLEXP void qpOASES_options_set_terminationTolerance(void *opts, double val){
    static_cast<blockSQP::qpOASES_options *>(opts)->terminationTolerance = val;
}

CDLEXP void *create_SQPoptions(){
    return static_cast<void *>(new blockSQP::SQPoptions);
}

CDLEXP void delete_SQPoptions(void *ptr_SQPoptions){
    delete static_cast<SQPoptions *>(ptr_SQPoptions);
}

CDLEXP void SQPoptions_set_print_level(void *ptr_SQPoptions, int val){
    static_cast<SQPoptions *>(ptr_SQPoptions)->print_level = val;
}

CDLEXP void SQPoptions_set_result_print_color(void *ptr, int val){
    static_cast<SQPoptions *>(ptr)->result_print_color = val;
}
CDLEXP void SQPoptions_set_debug_level(void *ptr, int val){
    static_cast<SQPoptions *>(ptr)->debug_level = val;
}
CDLEXP void SQPoptions_set_eps(void *ptr, double val){
    static_cast<SQPoptions *>(ptr)->eps = val;
}
CDLEXP void SQPoptions_set_inf(void *ptr, double val){
    static_cast<SQPoptions *>(ptr)->inf = val;
}
CDLEXP void SQPoptions_set_opt_tol(void *ptr, double val){
    static_cast<SQPoptions *>(ptr)->opt_tol = val;
}
CDLEXP void SQPoptions_set_feas_tol(void *ptr, double val){
    static_cast<SQPoptions *>(ptr)->feas_tol = val;
}
CDLEXP void SQPoptions_set_sparse(void *ptr, char val){
    static_cast<SQPoptions *>(ptr)->sparse = bool(val);
}
CDLEXP void SQPoptions_set_enable_linesearch(void *ptr, char val){
    static_cast<SQPoptions *>(ptr)->enable_linesearch = bool(val);
}
CDLEXP void SQPoptions_set_enable_rest(void *ptr, char val){
    static_cast<SQPoptions *>(ptr)->enable_rest = bool(val);
}
CDLEXP void SQPoptions_set_rest_rho(void *ptr, double val){
    static_cast<SQPoptions *>(ptr)->rest_rho = val;
}
CDLEXP void SQPoptions_set_rest_zeta(void *ptr, double val){
    static_cast<SQPoptions *>(ptr)->rest_zeta = val;
}
CDLEXP void SQPoptions_set_max_linesearch_steps(void *ptr, int val){
    static_cast<SQPoptions *>(ptr)->max_linesearch_steps = val;
}
CDLEXP void SQPoptions_set_max_consec_reduced_steps(void *ptr, int val){
    static_cast<SQPoptions *>(ptr)->max_consec_reduced_steps = val;
}
CDLEXP void SQPoptions_set_max_consec_skipped_updates(void *ptr, int val){
    static_cast<SQPoptions *>(ptr)->max_consec_skipped_updates = val;
}
CDLEXP void SQPoptions_set_max_QP_it(void *ptr, int val){
    static_cast<SQPoptions *>(ptr)->max_QP_it = val;
}
CDLEXP void SQPoptions_set_block_hess(void *ptr, int val){
    static_cast<SQPoptions *>(ptr)->block_hess = val;
}
CDLEXP void SQPoptions_set_sizing(void *ptr, int val){
    static_cast<SQPoptions *>(ptr)->sizing = val;
}
CDLEXP void SQPoptions_set_fallback_sizing(void *ptr, int val){
    static_cast<SQPoptions *>(ptr)->fallback_sizing = val;
}
CDLEXP void SQPoptions_set_max_QP_secs(void *ptr, double val){
    static_cast<SQPoptions *>(ptr)->max_QP_secs = val;
}
CDLEXP void SQPoptions_set_initial_hess_scale(void *ptr, double val){
    static_cast<SQPoptions *>(ptr)->initial_hess_scale = val;
}
CDLEXP void SQPoptions_set_COL_eps(void *ptr, double val){
    static_cast<SQPoptions *>(ptr)->COL_eps = val;
}
CDLEXP void SQPoptions_set_OL_eps(void *ptr, double val){
    static_cast<SQPoptions *>(ptr)->OL_eps = val;
}
CDLEXP void SQPoptions_set_COL_tau_1(void *ptr, double val){
    static_cast<SQPoptions *>(ptr)->COL_tau_1 = val;
}
CDLEXP void SQPoptions_set_COL_tau_2(void *ptr, double val){
    static_cast<SQPoptions *>(ptr)->COL_tau_2 = val;
}
CDLEXP void SQPoptions_set_BFGS_damping_factor(void *ptr, double val){
    static_cast<SQPoptions *>(ptr)->BFGS_damping_factor = val;
}
CDLEXP void SQPoptions_set_min_damping_quotient(void *ptr, double val){
    static_cast<SQPoptions *>(ptr)->min_damping_quotient = val;
}
CDLEXP void SQPoptions_set_hess_approx(void *ptr, int val){
    static_cast<SQPoptions *>(ptr)->hess_approx = val;
}
CDLEXP void SQPoptions_set_fallback_approx(void *ptr, int val){
    static_cast<SQPoptions *>(ptr)->fallback_approx = val;
}
CDLEXP void SQPoptions_set_indef_local_only(void *ptr, char val){
    static_cast<SQPoptions *>(ptr)->indef_local_only = bool(val);
}
CDLEXP void SQPoptions_set_lim_mem(void *ptr, char val){
    static_cast<SQPoptions *>(ptr)->lim_mem = bool(val);
}
CDLEXP void SQPoptions_set_mem_size(void *ptr, int val){
    static_cast<SQPoptions *>(ptr)->mem_size = val;
}
CDLEXP void SQPoptions_set_exact_hess(void *ptr, int val){
    static_cast<SQPoptions *>(ptr)->exact_hess = val;
}
CDLEXP void SQPoptions_set_skip_first_linesearch(void *ptr, int val){
    static_cast<SQPoptions *>(ptr)->skip_first_linesearch = val;
}
CDLEXP void SQPoptions_set_conv_strategy(void *ptr, int val){
    static_cast<SQPoptions *>(ptr)->conv_strategy = val;
}
CDLEXP void SQPoptions_set_max_conv_QPs(void *ptr, int val){
    static_cast<SQPoptions *>(ptr)->max_conv_QPs = val;
}
CDLEXP void SQPoptions_set_hess_regularization_factor(void *ptr, double val){
    static_cast<SQPoptions *>(ptr)->reg_factor = val;
}
CDLEXP void SQPoptions_set_max_SOC(void *ptr, int val){
    static_cast<SQPoptions *>(ptr)->max_SOC = val;
}
CDLEXP void SQPoptions_set_max_bound_refines(void *ptr, int val){
    static_cast<SQPoptions *>(ptr)->max_bound_refines = val;
}
CDLEXP void SQPoptions_set_max_correction_steps(void *ptr, int val){
    static_cast<SQPoptions *>(ptr)->max_correction_steps = val;
}
CDLEXP void SQPoptions_set_dep_bound_tolerance(void *ptr, double val){
    static_cast<SQPoptions *>(ptr)->dep_bound_tolerance = val;
}
CDLEXP void SQPoptions_set_qpsol_options(void *ptr, QPsolver_options *QPopts){
    static_cast<SQPoptions *>(ptr)->qpsol_options = QPopts;
}
CDLEXP void SQPoptions_set_automatic_scaling(void *ptr, char val){
    static_cast<SQPoptions *>(ptr)->automatic_scaling = bool(val);
}
CDLEXP void SQPoptions_set_max_local_lenience(void *ptr, int val){
    static_cast<SQPoptions *>(ptr)->max_filter_overrides = val;
}
CDLEXP void SQPoptions_set_max_extra_steps(void *ptr, int val){
    static_cast<SQPoptions *>(ptr)->max_extra_steps = val;
}
CDLEXP void SQPoptions_set_par_QPs(void *ptr, char val){
    static_cast<SQPoptions *>(ptr)->par_QPs = bool(val);
}
CDLEXP void SQPoptions_set_qpsol(void *ptr, int val){
    QPsolvers QPS;
    if (val == 0)
        QPS = QPsolvers::qpOASES;
    else if (val == 1)
        QPS = QPsolvers::gurobi;
    else
        QPS = QPsolvers::unset;
    static_cast<SQPoptions *>(ptr)->qpsol = QPS;
}

CDLEXP void *create_SQPstats(char *pathstr){
    return static_cast<void *>(new SQPstats(pathstr));
}

CDLEXP void delete_SQPstats(void *ptr){
    delete static_cast<SQPstats *>(ptr);
}

CDLEXP void *create_Problemspec(int nVar, int nCon){
    return static_cast<void *>(new CProblemspec(nVar, nCon));
}

CDLEXP void delete_Problemspec(void *ptr){
    delete static_cast<CProblemspec *>(ptr);
}

CDLEXP void Problemspec_print_info(void *ptr){
    Problemspec *P = static_cast<CProblemspec *>(ptr);
    std::cout << "\nnVar: " << P->nVar << "\nnCon: " << P->nCon << "\nnBlocks: " << P->nBlocks << "\nnnz: " << P->nnz << "\nblockIdx: ";
    for (int i = 0; i <= P->nBlocks; i++)
    {
        std::cout << P->blockIdx[i] << ", ";
    }
    std::cout << "\n";
}

CDLEXP void Problemspec_set_nnz(void *ptr, int nnz){
    static_cast<CProblemspec *>(ptr)->nnz = nnz;
}

CDLEXP void Problemspec_set_bounds(void *ptr, double *arg_lb_var, double *arg_ub_var, double *arg_lb_con, double *arg_ub_con, double arg_lb_obj, double arg_ub_obj){
    static_cast<CProblemspec *>(ptr)->objLo = arg_lb_obj;
    static_cast<CProblemspec *>(ptr)->objUp = arg_ub_obj;

    static_cast<CProblemspec *>(ptr)->lb_var.Dimension(static_cast<CProblemspec *>(ptr)->nVar);
    static_cast<CProblemspec *>(ptr)->ub_var.Dimension(static_cast<CProblemspec *>(ptr)->nVar);
    static_cast<CProblemspec *>(ptr)->lb_con.Dimension(static_cast<CProblemspec *>(ptr)->nCon);
    static_cast<CProblemspec *>(ptr)->ub_con.Dimension(static_cast<CProblemspec *>(ptr)->nCon);

    std::copy(arg_lb_var, arg_lb_var + static_cast<CProblemspec *>(ptr)->nVar, static_cast<CProblemspec *>(ptr)->lb_var.array);
    std::copy(arg_ub_var, arg_ub_var + static_cast<CProblemspec *>(ptr)->nVar, static_cast<CProblemspec *>(ptr)->ub_var.array);

    std::copy(arg_lb_con, arg_lb_con + static_cast<CProblemspec *>(ptr)->nCon, static_cast<CProblemspec *>(ptr)->lb_con.array);
    std::copy(arg_ub_con, arg_ub_con + static_cast<CProblemspec *>(ptr)->nCon, static_cast<CProblemspec *>(ptr)->ub_con.array);

    return;
}

CDLEXP void Problemspec_set_blockIdx(void *ptr, int *arg_blockIdx, int arg_nBlocks){
    static_cast<CProblemspec *>(ptr)->nBlocks = arg_nBlocks;
    delete[] static_cast<CProblemspec *>(ptr)->blockIdx;
    static_cast<CProblemspec *>(ptr)->blockIdx = new int[arg_nBlocks + 1];
    std::copy(arg_blockIdx, arg_blockIdx + arg_nBlocks + 1, static_cast<CProblemspec *>(ptr)->blockIdx);
}

CDLEXP void Problemspec_set_vblocks(void *ptr, void *arg_vblocks, int arg_n_vblocks){
    delete[] static_cast<CProblemspec *>(ptr)->vblocks;
    static_cast<CProblemspec *>(ptr)->n_vblocks = arg_n_vblocks;
    static_cast<CProblemspec *>(ptr)->vblocks = new vblock[arg_n_vblocks];
    std::copy(static_cast<vblock *>(arg_vblocks), static_cast<vblock *>(arg_vblocks) + arg_n_vblocks, static_cast<CProblemspec *>(ptr)->vblocks);
}

CDLEXP void Problemspec_pass_vblocks(void *ptr, void *arg_vblocks, int arg_n_vblocks){
    delete[] static_cast<CProblemspec *>(ptr)->vblocks;
    static_cast<CProblemspec *>(ptr)->n_vblocks = arg_n_vblocks;
    static_cast<CProblemspec *>(ptr)->vblocks = static_cast<vblock *>(arg_vblocks);
}

CDLEXP void Problemspec_set_closure(void *ptr, void *arg_closure){
    static_cast<CProblemspec *>(ptr)->closure = arg_closure;
}

CDLEXP void Problemspec_set_dense_init(void *ptr, void (*fp_init_dense)(void *closure_pass, double *xi, double *lambda, double *constrJac)){
    static_cast<CProblemspec *>(ptr)->initialize_dense = fp_init_dense;
}

CDLEXP void Problemspec_set_dense_eval(void *ptr, void (*fp_eval_dense)(void *closure_pass, const double *xi, const double *lambda, double *objval, double *constr, double *gradObj, double *constrJac, double **hess, int dmode, int *info)){
    static_cast<CProblemspec *>(ptr)->evaluate_dense = fp_eval_dense;
}

CDLEXP void Problemspec_set_simple_eval(void *ptr, void (*fp_eval_simple)(void *closure_pass, const double *xi, double *objval, double *constr, int *info)){
    static_cast<CProblemspec *>(ptr)->evaluate_simple = fp_eval_simple;
}

CDLEXP void Problemspec_set_sparse_init(void *ptr, void (*fp_init_sparse)(void *closure_pass, double *xi, double *lambda, double *jacNz, int *jacIndRow, int *jacIndCol)){
    static_cast<CProblemspec *>(ptr)->initialize_sparse = fp_init_sparse;
}

CDLEXP void Problemspec_set_sparse_eval(void *ptr, void (*fp_eval_sparse)(void *closure_pass, const double *xi, const double *lambda, double *objval, double *constr, double *gradObj, double *jacNz, int *jacIndRow, int *jacIndCol, double **hess, int dmode, int *info)){
    static_cast<CProblemspec *>(ptr)->evaluate_sparse = fp_eval_sparse;
}

CDLEXP void Problemspec_set_continuity_restoration(void *ptr, void (*fp_rest_cont)(void *closure_pass, double *xi, int *info)){
    static_cast<CProblemspec *>(ptr)->restore_continuity = fp_rest_cont;
}

CDLEXP void *create_SQPmethod(void *Problemspec_prob, void *SQPoptions_opts, void *SQPstats_stats){
    try
    {
        return static_cast<void *>(new SQPmethod(static_cast<Problemspec *>(Problemspec_prob), static_cast<SQPoptions *>(SQPoptions_opts), static_cast<SQPstats *>(SQPstats_stats)));
    }
    catch (std::exception &E)
    {
        strncpy(CblockSQP_error_message, E.what(), MAXLEN_CBLOCKSQP_ERROR_MESSAGE);
    }
    CblockSQP_error_message[MAXLEN_CBLOCKSQP_ERROR_MESSAGE - 1] = '\0';
    return nullptr;
}

CDLEXP void delete_SQPmethod(void *ptr){
    delete static_cast<SQPmethod *>(ptr);
}

CDLEXP void SQPmethod_init(void *ptr){
    static_cast<SQPmethod *>(ptr)->init();
}

CDLEXP int SQPmethod_run(void *ptr, int maxIt, int warmStart){
    try
    {
        return static_cast<int>(static_cast<SQPmethod *>(ptr)->run(maxIt, warmStart));
    }
    catch (std::exception &E)
    {
        strncpy(CblockSQP_error_message, E.what(), MAXLEN_CBLOCKSQP_ERROR_MESSAGE);
    }
    CblockSQP_error_message[MAXLEN_CBLOCKSQP_ERROR_MESSAGE - 1] = '\0';
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


CDLEXP void *create_cblock_array(int size){
    return static_cast<void *>(new cblock[size]);
}

CDLEXP void delete_cblock_array(void *ptr){
    delete[] static_cast<cblock *>(ptr);
}

CDLEXP void cblock_array_set(void *ptr, int index, int size){
    static_cast<cblock *>(ptr)[index].size = size;
}

CDLEXP void *create_hsize_array(int size){
    return static_cast<void *>(new int[size]);
}

CDLEXP void delete_hsize_array(void *ptr){
    delete[] static_cast<int *>(ptr);
}

CDLEXP void hsize_array_set(void *ptr, int index, int size){
    static_cast<int *>(ptr)[index] = size;
}

CDLEXP void *create_target_array(int size){
    return static_cast<void *>(new condensing_target[size]);
}

CDLEXP void delete_target_array(void *ptr){
    delete[] static_cast<condensing_target *>(ptr);
}

CDLEXP void target_array_set(void *ptr, int index, int n_stages, int first_free, int vblock_end, int first_cond, int cblock_end){
    static_cast<condensing_target *>(ptr)[index].n_stages = n_stages;
    static_cast<condensing_target *>(ptr)[index].first_free = first_free;
    static_cast<condensing_target *>(ptr)[index].vblock_end = vblock_end;
    static_cast<condensing_target *>(ptr)[index].first_cond = first_cond;
    static_cast<condensing_target *>(ptr)[index].cblock_end = cblock_end;
}

CDLEXP void *create_Condenser(void *arg_vblocks, int N_vblocks, void *arg_cblocks, int N_cblocks, void *arg_hsizes, int N_hsizes, void *arg_targets, int N_targets, int arg_dep_bounds){
    return new Condenser(static_cast<vblock *>(arg_vblocks), N_vblocks, static_cast<cblock *>(arg_cblocks), N_cblocks, static_cast<int *>(arg_hsizes), N_hsizes, static_cast<condensing_target *>(arg_targets), N_targets, arg_dep_bounds);
}

CDLEXP void delete_Condenser(void *ptr){
    delete static_cast<Condenser *>(ptr);
}

CDLEXP void Condenser_print_debug(void *ptr){
    static_cast<Condenser*>(ptr)->print_debug();
}

CDLEXP void Condenser_full_condense(void *ptr, void *Matrix_grad_obj, void *Sparse_Matrix_constr_jac, void *SymMatrix_array_hess, void *Matrix_lb_var, void *Matrix_ub_var, void *Matrix_lb_con, void *Matrix_ub_con, void *Matrix_condensed_grad_obj, void *Sparse_Matrix_condensed_constr_jac, void *SymMatrix_array_condensed_hess, void *Matrix_condensed_lb_var, void *Matrix_condensed_ub_var, void *Matrix_condensed_lb_con, void *Matrix_condensed_ub_con){
    static_cast<Condenser*>(ptr)->full_condense(*static_cast<Matrix *>(Matrix_grad_obj), *static_cast<Sparse_Matrix *>(Sparse_Matrix_constr_jac), static_cast<SymMatrix *>(SymMatrix_array_hess), *static_cast<Matrix *>(Matrix_lb_var), *static_cast<Matrix *>(Matrix_ub_var), *static_cast<Matrix *>(Matrix_lb_con), *static_cast<Matrix *>(Matrix_ub_con),
                                                 *static_cast<Matrix *>(Matrix_condensed_grad_obj), *static_cast<Sparse_Matrix *>(Sparse_Matrix_condensed_constr_jac), static_cast<SymMatrix *>(SymMatrix_array_condensed_hess), *static_cast<Matrix *>(Matrix_condensed_lb_var), *static_cast<Matrix *>(Matrix_condensed_ub_var), *static_cast<Matrix *>(Matrix_condensed_lb_con), *static_cast<Matrix *>(Matrix_condensed_ub_con));
}

CDLEXP void Condenser_recover_var_mult(void *ptr, void *xi_cond, void *lambda_cond, void *xi_rest, void *lambda_rest){
    static_cast<Condenser*>(ptr)->recover_var_mult(*static_cast<Matrix *>(xi_cond), *static_cast<Matrix *>(lambda_cond), *static_cast<Matrix *>(xi_rest), *static_cast<Matrix *>(lambda_rest));
}

CDLEXP int Condenser_nVar(void *ptr){
    return static_cast<Condenser*>(ptr)->num_vars;
}

CDLEXP int Condenser_nCon(void *ptr){
    return static_cast<Condenser*>(ptr)->num_cons;
}

CDLEXP int Condenser_nBlocks(void *ptr){
    return static_cast<Condenser*>(ptr)->num_hessblocks;
}

CDLEXP int *Condenser_hsizes(void *ptr){
    return static_cast<Condenser*>(ptr)->hess_block_sizes;
}



CDLEXP int Condenser_condensed_nVar(void *ptr){
    return static_cast<Condenser*>(ptr)->condensed_num_vars;
}

CDLEXP int Condenser_condensed_nCon(void *ptr){
    return static_cast<Condenser*>(ptr)->condensed_num_cons;
}

CDLEXP int Condenser_condensed_nBlocks(void *ptr){
    return static_cast<Condenser*>(ptr)->condensed_num_hessblocks;
}

CDLEXP int *Condenser_condensed_hsizes(void *ptr){
    return static_cast<Condenser*>(ptr)->condensed_hess_block_sizes;
}


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
/*
mod.add_type<blockSQP::Matrix>("BSQP_Matrix")
    .constructor<>()
    .constructor<int, int>()
    .method("size_1", [](blockSQP::Matrix &M){return M.m;})
    .method("size_2", [](blockSQP::Matrix &M){return M.n;})
    .method("release!", [](blockSQP::Matrix &M){
        double *ptr = M.array;
        M.array = nullptr;
        M.m = 0;
        M.n = 0;
        return ptr;
    })
    .method("show_ptr", [](blockSQP::Matrix &M){return M.array;})
    ;
*/

CDLEXP void *create_SymMatrix(int m){
    return static_cast<void *>(new SymMatrix(m));
}

CDLEXP void delete_SymMatrix(void *ptr){
    delete static_cast<SymMatrix *>(ptr);
}

CDLEXP double *SymMatrix_show_array(void *ptr){
    return static_cast<Matrix *>(ptr)->array;
}

/*
mod.add_type<blockSQP::SymMatrix>("SymMatrix")//, jlcxx::julia_base_type<blockSQP::Matrix>())
    .constructor<>()
    .method("size_1", [](blockSQP::SymMatrix &M){return M.m;})
    .method("release!", [](blockSQP::SymMatrix &M){
        double *ptr = M.array;
        M.array = nullptr;
        M.m = 0;
        return ptr;
    })
    .method("show_ptr", [](blockSQP::SymMatrix &M){return M.array;})
    .method("set_size!", [](blockSQP::SymMatrix &M, int dim){M.Dimension(dim);})
    ;

mod.method("show_ptr", [](blockSQP::SymMatrix *M){return M->array;});
mod.method("set_size!", [](blockSQP::SymMatrix *M, int dim){M->Dimension(dim);});
mod.method("size_1", [](blockSQP::SymMatrix *M){return M->m;});
*/

CDLEXP void *create_Sparse_Matrix(int m, int n, int nnz){
    return static_cast<void *>(new Sparse_Matrix(m, n, nnz));
}

CDLEXP void *create_Sparse_Matrix_default(){
    return static_cast<void *>(new Sparse_Matrix());
}

CDLEXP void Sparse_Matrix_set_structure(void *ptr, int m, int n, int nnz){
    if (static_cast<Sparse_Matrix*>(ptr)->m != m || static_cast<Sparse_Matrix*>(ptr)->n != n || static_cast<Sparse_Matrix*>(ptr)->colind[static_cast<Sparse_Matrix*>(ptr)->n] != nnz)
        *static_cast<Sparse_Matrix*>(ptr) = Sparse_Matrix(m, n, nnz);
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

/*
mod.add_type<blockSQP::Sparse_Matrix>("Cxx_Sparse_Matrix")
    .constructor<>()
    .method("disown!", [](blockSQP::Sparse_Matrix &M){
        M.m = 0;
        M.n = 0;
        M.nz.release();
        M.row.release();
        M.colind.release();
    })
    .method("get_nnz", [](blockSQP::Sparse_Matrix &M){return M.colind[M.n];})
    .method("show_nz", [](blockSQP::Sparse_Matrix &M){return M.nz.get();})
    .method("show_row", [](blockSQP::Sparse_Matrix &M){return M.row.get();})
    .method("show_colind", [](blockSQP::Sparse_Matrix &M){return M.colind.get();})
    ;

mod.method("alloc_Cxx_Sparse_Matrix", [](int m, int n, int nnz){
    return blockSQP::Sparse_Matrix(m, n, std::make_unique<double[]>(nnz), std::make_unique<int[]>(nnz), std::make_unique<int[]>(n+1));
});
*/

CDLEXP void *create_SymMatrix_array(int size){
    return static_cast<void*>(new SymMatrix[size]);
}

CDLEXP void delete_SymMatrix_array(void *ptr){
    delete[] static_cast<SymMatrix*>(ptr);
}

CDLEXP void SymMatrix_array_index_resize(void *ptr, int index, int m){
    static_cast<SymMatrix*>(ptr)[index].Dimension(m);
}

CDLEXP double *SymMatrix_array_index_array(void *ptr, int index){
    return static_cast<SymMatrix*>(ptr)[index].array;
}


/*
mod.add_type<SymMat_array>("SymMat_array")
    .constructor<int>()
    .method("array_size", [](SymMat_array &ARR){return ARR.size;})
    .method("array_set!", [](SymMat_array &ARR, int index, blockSQP::SymMatrix V){ARR.ptr[index - 1] = V;})
    .method("array_get", [](SymMat_array &ARR, int index){return ARR.ptr[index - 1];})
    .method("array_get_ptr", [](SymMat_array &ARR, int index){return ARR.ptr + index - 1;})
    ;
*/

/*
mod.add_type<blockSQP::Condenser>("Cxx_Condenser")
    .method("print_debug", &blockSQP::Condenser::print_debug)
    .method("get_num_vars", [](blockSQP::Condenser &C){return C.num_vars;})
    .method("get_num_cons", [](blockSQP::Condenser &C){return C.num_cons;})
    .method("get_num_hessblocks", [](blockSQP::Condenser &C){return C.num_hessblocks;})
    .method("get_condensed_num_vars", [](blockSQP::Condenser &C){return C.condensed_num_vars;})
    .method("get_num_true_cons", [](blockSQP::Condenser &C){return C.num_true_cons;})
    .method("get_condensed_num_cons", [](blockSQP::Condenser &C){return C.condensed_num_cons;})
    .method("get_condensed_num_hessblocks", [](blockSQP::Condenser &C){return C.condensed_num_hessblocks;})
    .method("get_hess_block_sizes", [](blockSQP::Condenser &C){return C.hess_block_sizes;})
    .method("Cxx_full_condense!", [](blockSQP::Condenser &C, const blockSQP::Matrix &grad_obj, const blockSQP::Sparse_Matrix &con_jac, const SymMat_array &hess, const blockSQP::Matrix &lb_var, const blockSQP::Matrix &ub_var, const blockSQP::Matrix &lb_con, const blockSQP::Matrix &ub_con,
                blockSQP::Matrix &condensed_h, blockSQP::Sparse_Matrix &condensed_jacobian, SymMat_array &condensed_hess, blockSQP::Matrix &condensed_lb_var, blockSQP::Matrix &condensed_ub_var, blockSQP::Matrix &condensed_lb_con, blockSQP::Matrix &condensed_ub_con){
                C.full_condense(grad_obj, con_jac, hess.ptr, lb_var, ub_var, lb_con, ub_con, condensed_h, condensed_jacobian, condensed_hess.ptr, condensed_lb_var, condensed_ub_var, condensed_lb_con, condensed_ub_con);
            }
        )
    .method("Cxx_recover_var_mult!", [](blockSQP::Condenser &C, const blockSQP::Matrix &xi_cond, const blockSQP::Matrix &lambda_cond, blockSQP::Matrix &xi_rest, blockSQP::Matrix &lambda_rest){

            C.recover_var_mult(xi_cond, lambda_cond, xi_rest, lambda_rest);
        }
    )
    ;

mod.method("construct_Condenser", [](vblock_array *VBLOCKS, cblock_array &CBLOCKS, int_array &HSIZES, condensing_targets &TARGETS, int DEP_BOUNDS){
    return blockSQP::Condenser(VBLOCKS->ptr, VBLOCKS->size, CBLOCKS.ptr, CBLOCKS.size, HSIZES.ptr, HSIZES.size, TARGETS.ptr, TARGETS.size, DEP_BOUNDS);}
    );
*/
