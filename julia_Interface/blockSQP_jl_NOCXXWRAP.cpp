/**
 * \file blockSQP_julbind.cpp
 * \author Reinhold Wittmann
 * \date 2024-
 *
 * CxxWrap based julia interface for the blockSQP nonlinear programming solver
 */
/*
#include "jlcxx/jlcxx.hpp"
#include "jlcxx/array.hpp"
#include "jlcxx/functions.hpp"
*/
/*
#include <jlcxx/jlcxx.hpp>
#include <jlcxx/array.hpp>
#include <jlcxx/functions.hpp>
*/

#include "blocksqp_method.hpp"
#include "blocksqp_condensing.hpp"
#include "blocksqp_options.hpp"
#include "blocksqp_problemspec.hpp"
#include "blocksqp_matrix.hpp"
#include <limits>
#include <iostream>
#include <string>
#include <stdexcept>
#include <cstring>
#include "stdlib.h"

using namespace blockSQP;

#ifdef _MSC_VER
    #define CDLEXP extern "C" __declspec(dllexport)
#else
    #define CDLEXP extern "C" __attribute__((visibility ("default")))
#endif


#define MAXLEN_CBLOCKSQP_ERROR_MESSAGE 1000
char CblockSQP_error_message[MAXLEN_CBLOCKSQP_ERROR_MESSAGE];

CDLEXP char *get_error_message(){
    return CblockSQP_error_message;
}

class CProblemspec : public blockSQP::Problemspec{
public:
    CProblemspec(int NVARS, int NCONS){
        nVar = NVARS;
        nCon = NCONS;
    };

    virtual ~CProblemspec(){
        delete[] blockIdx;
        delete[] vblocks;
    };

    //Allocate callbacks (function pointers to global julia functions)
    void (*initialize_dense)(void *closure_pass, double *xi, double *lambda, double *constrJac);
    void (*evaluate_dense)(void *closure_pass, const double *xi, const double* lambda, double *objval, double *constr, double *gradObj, double *constrJac, double **hess, int dmode, int *info);
    void (*evaluate_simple)(void *closure_pass, const double *xi, double *objval, double *constr, int *info);

    void (*initialize_sparse)(void *closure_pass, double *xi, double *lambda, double *jacNz, int *jacIndRow, int *jacIndCol);
    void (*evaluate_sparse)(void *closure_pass, const double *xi, const double *lambda, double *objval, double *constr, double *gradObj, double *jacNz, int *jacIndRow, int *jacIndCol, double **hess, int dmode, int *info);

    void (*restore_continuity)(void *closure_pass, double *xi, int *info);

    //Pass-through pointer to a closure of the caller, currently for all callbacks. 
    void *closure;

    //Invoke callbacks in overridden methods
    virtual void initialize(blockSQP::Matrix &xi, blockSQP::Matrix &lambda, blockSQP::Matrix &constrJac){
        (*initialize_dense)(closure, xi.array, lambda.array, constrJac.array);
    }


    virtual void initialize(blockSQP::Matrix &xi, blockSQP::Matrix &lambda, double *jacNz, int *jacIndRow, int *jacIndCol){
        (*initialize_sparse)(closure, xi.array, lambda.array, jacNz, jacIndRow, jacIndCol);
    }

    virtual void evaluate(const blockSQP::Matrix &xi, const blockSQP::Matrix &lambda, double *objval, blockSQP::Matrix &constr, blockSQP::Matrix &gradObj, blockSQP::Matrix &constrJac, blockSQP::SymMatrix *hess, int dmode, int *info){
        double **hessNz = nullptr;
        if (dmode == 3){
            hessNz = new double*[nBlocks];
            for (int i = 0; i < nBlocks; i++){
                hessNz[i] = hess[i].array;
            }
        }
        else if (dmode == 2){
            hessNz = new double*[nBlocks];
            hessNz[nBlocks - 1] = hess[nBlocks - 1].array;
        }

        (*evaluate_dense)(closure, xi.array, lambda.array, objval, constr.array, gradObj.array, constrJac.array, hessNz, dmode, info);
        delete[] hessNz;
    }

    virtual void evaluate(const blockSQP::Matrix &xi, const blockSQP::Matrix &lambda, double *objval, blockSQP::Matrix &constr, blockSQP::Matrix &gradObj, double *jacNz, int *jacIndRow, int *jacIndCol, blockSQP::SymMatrix *hess, int dmode, int *info){
        double **hessNz = nullptr;
        if (dmode == 3){
            hessNz = new double*[nBlocks];
            for (int i = 0; i < nBlocks; i++){
                hessNz[i] = hess[i].array;
            }
        }
        else if (dmode == 2){
            hessNz = new double*[nBlocks];
            hessNz[nBlocks - 1] = hess[nBlocks - 1].array;
        }

        (*evaluate_sparse)(closure, xi.array, lambda.array, objval, constr.array, gradObj.array, jacNz, jacIndRow, jacIndCol, hessNz, dmode, info);
        
        delete[] hessNz;
    }

    virtual void evaluate(const blockSQP::Matrix &xi, double *objval, blockSQP::Matrix &constr, int *info){
        (*evaluate_simple)(closure, xi.array, objval, constr.array, info);
    }

    //Optional Methods
    virtual void reduceConstrVio(blockSQP::Matrix &xi, int *info){
        if (restore_continuity != nullptr){
            (*restore_continuity)(closure, xi.array, info);
        }
    };
};


/*
class JL_Condenser{
    std::unique_ptr<blockSQP::Condenser> Cxx_Condenser;
    std::unique_ptr<blockSQP::vblock[]> vblocks;
    std::unique_ptr<blockSQP::cblock[]> cblocks;
    std::unique_ptr<int[]> hsizes;
    std::unique_ptr<blockSQP::condensing_target[]> targets;

    JL_Condenser(jlcxx::ArrayRef<blockSQP::vblock, 1> &VBLOCKS, jlcxx::ArrayRef<blockSQP::cblock, 1> &CBLOCKS, jlcxx::ArrayRef<int, 1> &HSIZES, jlcxx::ArrayRef<blockSQP::condensing_target, 1> &TARGETS, int DEP_BOUNDS){
        vblocks = std::make_unique<blockSQP::vblock[]>(VBLOCKS.size());
        for (std::size_t i = 0; i < VBLOCKS.size(); i++){
            vblocks[i] = VBLOCKS[i];
        }

        cblocks = std::make_unique<blockSQP::cblock[]>(CBLOCKS.size());
        for (std::size_t i = 0; i < CBLOCKS.size(); i++){
            cblocks[i] = CBLOCKS[i];
        }
        
        hsizes = std::make_unique<int[]>(HSIZES.size());
        for (std::size_t i = 0; i < HSIZES.size(); i++){
            hsizes[i] = HSIZES[i];
        }

        targets = std::make_unique<blockSQP::condensing_target[]>(TARGETS.size());
        for (std::size_t i = 0; i < TARGETS.size(); i++){
            targets[i] = TARGETS[i];
        }
        
        Cxx_Condenser = std::make_unique<blockSQP::Condenser>(vblocks.get(), VBLOCKS.size(), cblocks.get(), CBLOCKS.size(), hsizes.get(), HSIZES.size(), targets.get(), TARGETS.size(), DEP_BOUNDS);
    }
};
*/
/*
class NULL_QPsolver_options : public blockSQP::QPsolver_options{
    public:
    NULL_QPsolver_options() : QPsolver_options(blockSQP::QPsolvers::unset){}
};
*/


CDLEXP void *create_vblock_array(int size){
    return (void*) new vblock[size];
}

CDLEXP void delete_vblock_array(void* arr){
    delete[] static_cast<vblock*>(arr);
}

CDLEXP void vblock_array_set(void *arr, int index, int size, char dependent){
    static_cast<vblock*>(arr)[index].size = size;
    static_cast<vblock*>(arr)[index].dependent = bool(dependent);
}


CDLEXP void delete_QPsolver_options(void *ptr_QPsolver_options){
    delete static_cast<QPsolver_options*>(ptr_QPsolver_options);
}

CDLEXP void* create_qpOASES_options(){
    return static_cast<void*>(new blockSQP::qpOASES_options());
}

CDLEXP void qpOASES_options_set_sparsityLevel(void *opts, int val){
    static_cast<blockSQP::qpOASES_options*>(opts)->sparsityLevel = val;
}

CDLEXP void qpOASES_options_set_printLevel(void *opts, int val){
    static_cast<blockSQP::qpOASES_options*>(opts)->printLevel = val;
}

CDLEXP void qpOASES_options_set_terminationTolerance(void *opts, double val){
    static_cast<blockSQP::qpOASES_options*>(opts)->terminationTolerance = val;
}



CDLEXP void *create_SQPoptions(){
    return static_cast<void*>(new blockSQP::SQPoptions);
}

CDLEXP void delete_SQPoptions(void *ptr_SQPoptions){
    delete static_cast<SQPoptions*>(ptr_SQPoptions);
}

CDLEXP void SQPoptions_set_print_level(void *ptr_SQPoptions, int val){
    static_cast<SQPoptions*>(ptr_SQPoptions)->print_level = val;
}

//Thx ChatGPT
CDLEXP void SQPoptions_set_result_print_color(void* ptr, int val) {
    static_cast<SQPoptions*>(ptr)->result_print_color = val;
}
CDLEXP void SQPoptions_set_debug_level(void* ptr, int val) {
    static_cast<SQPoptions*>(ptr)->debug_level = val;
}
CDLEXP void SQPoptions_set_eps(void* ptr, double val) {
    static_cast<SQPoptions*>(ptr)->eps = val;
}
CDLEXP void SQPoptions_set_inf(void* ptr, double val) {
    static_cast<SQPoptions*>(ptr)->inf = val;
}
CDLEXP void SQPoptions_set_opt_tol(void* ptr, double val) {
    static_cast<SQPoptions*>(ptr)->opt_tol = val;
}
CDLEXP void SQPoptions_set_feas_tol(void* ptr, double val) {
    static_cast<SQPoptions*>(ptr)->feas_tol = val;
}
CDLEXP void SQPoptions_set_sparse(void* ptr, char val) {
    static_cast<SQPoptions*>(ptr)->sparse = bool(val);
}
CDLEXP void SQPoptions_set_enable_linesearch(void* ptr, char val) {
    static_cast<SQPoptions*>(ptr)->enable_linesearch = bool(val);
}
CDLEXP void SQPoptions_set_enable_rest(void* ptr, char val) {
    static_cast<SQPoptions*>(ptr)->enable_rest = bool(val);
}
CDLEXP void SQPoptions_set_rest_rho(void* ptr, double val) {
    static_cast<SQPoptions*>(ptr)->rest_rho = val;
}
CDLEXP void SQPoptions_set_rest_zeta(void* ptr, double val) {
    static_cast<SQPoptions*>(ptr)->rest_zeta = val;
}
CDLEXP void SQPoptions_set_max_linesearch_steps(void* ptr, int val) {
    static_cast<SQPoptions*>(ptr)->max_linesearch_steps = val;
}
CDLEXP void SQPoptions_set_max_consec_reduced_steps(void* ptr, int val) {
    static_cast<SQPoptions*>(ptr)->max_consec_reduced_steps = val;
}
CDLEXP void SQPoptions_set_max_consec_skipped_updates(void* ptr, int val) {
    static_cast<SQPoptions*>(ptr)->max_consec_skipped_updates = val;
}
CDLEXP void SQPoptions_set_max_QP_it(void* ptr, int val) {
    static_cast<SQPoptions*>(ptr)->max_QP_it = val;
}
CDLEXP void SQPoptions_set_block_hess(void* ptr, int val) {
    static_cast<SQPoptions*>(ptr)->block_hess = val;
}
CDLEXP void SQPoptions_set_sizing(void* ptr, int val) {
    static_cast<SQPoptions*>(ptr)->sizing = val;
}
CDLEXP void SQPoptions_set_fallback_sizing(void* ptr, int val) {
    static_cast<SQPoptions*>(ptr)->fallback_sizing = val;
}
CDLEXP void SQPoptions_set_max_QP_secs(void* ptr, double val) {
    static_cast<SQPoptions*>(ptr)->max_QP_secs = val;
}
CDLEXP void SQPoptions_set_initial_hess_scale(void* ptr, double val) {
    static_cast<SQPoptions*>(ptr)->initial_hess_scale = val;
}
CDLEXP void SQPoptions_set_COL_eps(void* ptr, double val) {
    static_cast<SQPoptions*>(ptr)->COL_eps = val;
}
CDLEXP void SQPoptions_set_OL_eps(void* ptr, double val) {
    static_cast<SQPoptions*>(ptr)->OL_eps = val;
}
CDLEXP void SQPoptions_set_COL_tau_1(void* ptr, double val) {
    static_cast<SQPoptions*>(ptr)->COL_tau_1 = val;
}
CDLEXP void SQPoptions_set_COL_tau_2(void* ptr, double val) {
    static_cast<SQPoptions*>(ptr)->COL_tau_2 = val;
}
CDLEXP void SQPoptions_set_BFGS_damping_factor(void* ptr, double val) {
    static_cast<SQPoptions*>(ptr)->BFGS_damping_factor = val;
}
CDLEXP void SQPoptions_set_min_damping_quotient(void* ptr, double val) {
    static_cast<SQPoptions*>(ptr)->min_damping_quotient = val;
}
CDLEXP void SQPoptions_set_hess_approx(void* ptr, int val) {
    static_cast<SQPoptions*>(ptr)->hess_approx = val;
}
CDLEXP void SQPoptions_set_fallback_approx(void* ptr, int val) {
    static_cast<SQPoptions*>(ptr)->fallback_approx = val;
}
CDLEXP void SQPoptions_set_indef_local_only(void* ptr, char val) {
    static_cast<SQPoptions*>(ptr)->indef_local_only = bool(val);
}
CDLEXP void SQPoptions_set_lim_mem(void* ptr, char val) {
    static_cast<SQPoptions*>(ptr)->lim_mem = bool(val);
}
CDLEXP void SQPoptions_set_mem_size(void* ptr, int val) {
    static_cast<SQPoptions*>(ptr)->mem_size = val;
}
CDLEXP void SQPoptions_set_exact_hess(void* ptr, int val) {
    static_cast<SQPoptions*>(ptr)->exact_hess = val;
}
CDLEXP void SQPoptions_set_skip_first_linesearch(void* ptr, int val) {
    static_cast<SQPoptions*>(ptr)->skip_first_linesearch = val;
}
CDLEXP void SQPoptions_set_conv_strategy(void* ptr, int val) {
    static_cast<SQPoptions*>(ptr)->conv_strategy = val;
}
CDLEXP void SQPoptions_set_max_conv_QPs(void* ptr, int val) {
    static_cast<SQPoptions*>(ptr)->max_conv_QPs = val;
}
CDLEXP void SQPoptions_set_hess_regularization_factor(void* ptr, double val) {
    static_cast<SQPoptions*>(ptr)->reg_factor = val;
}
CDLEXP void SQPoptions_set_max_SOC(void* ptr, int val) {
    static_cast<SQPoptions*>(ptr)->max_SOC = val;
}
CDLEXP void SQPoptions_set_max_bound_refines(void* ptr, int val) {
    static_cast<SQPoptions*>(ptr)->max_bound_refines = val;
}
CDLEXP void SQPoptions_set_max_correction_steps(void* ptr, int val) {
    static_cast<SQPoptions*>(ptr)->max_correction_steps = val;
}
CDLEXP void SQPoptions_set_dep_bound_tolerance(void* ptr, double val) {
    static_cast<SQPoptions*>(ptr)->dep_bound_tolerance = val;
}
CDLEXP void SQPoptions_set_qpsol_options(void* ptr, QPsolver_options* QPopts) {
    static_cast<SQPoptions*>(ptr)->qpsol_options = QPopts;
}
CDLEXP void SQPoptions_set_automatic_scaling(void* ptr, char val) {
    static_cast<SQPoptions*>(ptr)->automatic_scaling = bool(val);
}
CDLEXP void SQPoptions_set_max_local_lenience(void* ptr, int val) {
    static_cast<SQPoptions*>(ptr)->max_filter_overrides = val;
}
CDLEXP void SQPoptions_set_max_extra_steps(void* ptr, int val) {
    static_cast<SQPoptions*>(ptr)->max_extra_steps = val;
}
CDLEXP void SQPoptions_set_par_QPs(void* ptr, char val) {
    static_cast<SQPoptions*>(ptr)->par_QPs = bool(val);
}
CDLEXP void SQPoptions_set_qpsol(void* ptr, int val){
    QPsolvers QPS;
    if (val == 0) QPS = QPsolvers::qpOASES;
    else if (val == 1) QPS = QPsolvers::gurobi;
    else QPS = QPsolvers::unset;
    static_cast<SQPoptions*>(ptr)->qpsol = QPS;
}


/*
mod.add_type<blockSQP::SQPoptions>("Cxx_SQPoptions")
    .constructor<>()
    //.method("get_max_QP_it", [](blockSQP::SQPoptions &opts){return opts.max_QP_it;})
    .method("set_print_level", [](blockSQP::SQPoptions &opts, int val){opts.print_level = val;})
    .method("set_result_print_color", [](blockSQP::SQPoptions &opts, int val){opts.result_print_color = val;})
    .method("set_debug_level", [](blockSQP::SQPoptions &opts, int val){opts.debug_level = val;})
    .method("set_eps", [](blockSQP::SQPoptions &opts, double val){opts.eps = val;})
    .method("set_inf", [](blockSQP::SQPoptions &opts, double val){opts.inf = val;})
    .method("set_opt_tol", [](blockSQP::SQPoptions &opts, double val){opts.opt_tol = val;})
    .method("set_feas_tol", [](blockSQP::SQPoptions &opts, double val){opts.feas_tol = val;})
    .method("set_sparse", [](blockSQP::SQPoptions &opts, int val){opts.sparse = val;})
    .method("set_enable_linesearch", [](blockSQP::SQPoptions &opts, int val){opts.enable_linesearch = val;})
    .method("set_enable_rest", [](blockSQP::SQPoptions &opts, int val){opts.enable_rest = val;})
    .method("set_rest_rho", [](blockSQP::SQPoptions &opts, double val){opts.rest_rho = val;})
    .method("set_rest_zeta", [](blockSQP::SQPoptions &opts, double val){opts.rest_zeta = val;})
    .method("set_max_linesearch_steps", [](blockSQP::SQPoptions &opts, int val){opts.max_linesearch_steps = val;})
    .method("set_max_consec_reduced_steps", [](blockSQP::SQPoptions &opts, int val){opts.max_consec_reduced_steps = val;})
    .method("set_max_consec_skipped_updates", [](blockSQP::SQPoptions &opts, int val){opts.max_consec_skipped_updates = val;})
    .method("set_max_QP_it", [](blockSQP::SQPoptions &opts, int val){opts.max_QP_it = val;})
    .method("set_block_hess", [](blockSQP::SQPoptions &opts, int val){opts.block_hess = val;})
    .method("set_sizing", [](blockSQP::SQPoptions &opts, int val){opts.sizing = val;})
    .method("set_fallback_sizvoid* ptring", [](blockSQP::SQPoptions &opts, int val){opts.fallback_sizing = val;})
    .method("set_max_QP_secs", [](blockSQP::SQPoptions &opts, double val){opts.max_QP_secs = val;})
    .method("set_initial_hess_scale", [](blockSQP::SQPoptions &opts, double val){opts.initial_hess_scale = val;})
    .method("set_COL_eps", [](blockSQP::SQPoptions &opts, double val){opts.COL_eps = val;})
    .method("set_OL_eps", [](blockSQP::SQPoptions &opts, double val){opts.OL_eps = val;})
    .method("set_COL_tau_1", [](blockSQP::SQPoptions &opts, double val){opts.COL_tau_1 = val;})
    .method("set_COL_tau_2", [](blockSQP::SQPoptions &opts, double val){opts.COL_tau_2 = val;})
    .method("set_BFGS_damping_factor", [](blockSQP::SQPoptions &opts, double val){opts.BFGS_damping_factor = val;})
    .method("set_min_damping_quotient", [](blockSQP::SQPoptions &opts, double val){opts.min_damping_quotient = val;})
    .method("set_hess_approx", [](blockSQP::SQPoptions &opts, int val){opts.hess_approx = val;})
    .method("set_fallback_approx", [](blockSQP::SQPoptions &opts, int val){opts.fallback_approx = val;})
    .method("set_indef_local_only", [](blockSQP::SQPoptions &opts, bool val){opts.indef_local_only = val;})
    .method("set_lim_mem", [](blockSQP::SQPoptions &opts, int val){opts.lim_mem = val;})
    .method("set_mem_size", [](blockSQP::SQPoptions &opts, int val){opts.mem_size = val;})
    .method("set_exact_hess", [](blockSQP::SQPoptions &opts, int val){opts.exact_hess = val;})
    .method("set_skip_first_linesearch", [](blockSQP::SQPoptions &opts, int val){opts.skip_first_linesearch = val;})
    .method("set_conv_strategy", [](blockSQP::SQPoptions &opts, int val){opts.conv_strategy = val;})
    .method("set_max_conv_QPs", [](blockSQP::SQPoptions &opts, int val){opts.max_conv_QPs = val;})
    .method("set_hess_regularization_factor", [](blockSQP::SQPoptions &opts, double val){opts.reg_factor = val;})
    .method("set_max_SOC", [](blockSQP::SQPoptions &opts, int val){opts.max_SOC = val;})
    .method("set_max_bound_refines", [](blockSQP::SQPoptions &opts, int val){opts.max_bound_refines = val;})
    .method("set_max_correction_steps", [](blockSQP::SQPoptions &opts, int val){opts.max_correction_steps = val;})
    .method("set_dep_bound_tolerance", [](blockSQP::SQPoptions &opts, double val){opts.dep_bound_tolerance = val;})
    .method("set_qpsol_options", [](blockSQP::SQPoptions &opts, blockSQP::QPsolver_options *QPopts){opts.qpsol_options = QPopts;})
    .method("set_automatic_scaling", [](blockSQP::SQPoptions &opts, bool val){opts.automatic_scaling = val;})
	.method("set_max_local_lenience", [](blockSQP::SQPoptions &opts, int val){opts.max_filter_overrides = val;})
    .method("set_max_extra_steps", [](blockSQP::SQPoptions &opts, int val){opts.max_extra_steps = val;})
    .method("set_par_QPs", [](blockSQP::SQPoptions &opts, bool val){opts.par_QPs = val;})
    ;
    */
    
    
    /*
    .method("set_qpsol", [](blockSQP::SQPoptions &opts, std::string &QPsolver_name){
        if (QPsolver_name == "qpOASES") opts.qpsol = blockSQP::QPsolvers::qpOASES;
        else if (QPsolver_name == "gurobi") opts.qpsol = blockSQP::QPsolvers::gurobi;
        else throw blockSQP::ParameterError("Unknown QP solver options, known are blockSQP::qpOASES_options, blockSQP::gurobi_options");
    })
    .method("get_qpsol", [](blockSQP::SQPoptions &opts){
        if (opts.qpsol == blockSQP::QPsolvers::qpOASES) return std::string("qpOASES");
        else if (opts.qpsol == blockSQP::QPsolvers::gurobi) return std::string("gurobi");
        else throw blockSQP::ParameterError("Unknown QP solver name, known (no neccessarily linked) are qpOASES, gurobi");
    })
    */

CDLEXP void *create_SQPstats(char *pathstr){
    return static_cast<void*>(new SQPstats(pathstr));
}

CDLEXP void delete_SQPstats(void *ptr){
    delete static_cast<SQPstats*>(ptr);
}

/*
mod.add_type<blockSQP::SQPstats>("SQPstats")
    .constructor<char*>()
    .method("get_pathstr", [](blockSQP::SQPstats S){return std::string(S.outpath);})
    ;
*/

/*
mod.add_bits<blockSQP::SQPresult>("SQPresult", jlcxx::julia_type("CppEnum"));
mod.set_const("it_finished", blockSQP::SQPresult::it_finished);
mod.set_const("partial_success", blockSQP::SQPresult::partial_success);
mod.set_const("success", blockSQP::SQPresult::success);
mod.set_const("super_success", blockSQP::SQPresult::super_success);
mod.set_const("local_infeasibility", blockSQP::SQPresult::local_infeasibility);
mod.set_const("restoration_failure", blockSQP::SQPresult::restoration_failure);
mod.set_const("linesearch_failure", blockSQP::SQPresult::linesearch_failure);
mod.set_const("qp_failure", blockSQP::SQPresult::qp_failure);
mod.set_const("eval_failure", blockSQP::SQPresult::eval_failure);
mod.set_const("misc_error", blockSQP::SQPresult::misc_error);
*/


//mod.add_type<blockSQP::Problemspec>("Problemspec");


CDLEXP void *create_Problemspec(int nVar, int nCon){
    return static_cast<void*>(new CProblemspec(nVar, nCon));
}

CDLEXP void delete_Problemspec(void *ptr){
    delete static_cast<CProblemspec*>(ptr);
}

CDLEXP void Problemspec_print_info(void *ptr){
    Problemspec *P = static_cast<CProblemspec*>(ptr);
    std::cout << "\nnVar: " << P->nVar << "\nnCon: " << P->nCon << "\nnBlocks: " << P->nBlocks << "\nnnz: " << P->nnz << "\nblockIdx: ";
    for (int i = 0; i <= P->nBlocks; i++){
        std::cout << P->blockIdx[i] << ", ";
    }
    std::cout << "\n";
}

CDLEXP void Problemspec_set_nnz(void *ptr, int nnz){
    static_cast<CProblemspec*>(ptr)->nnz = nnz;
}

CDLEXP void Problemspec_set_bounds(void *ptr, double *arg_lb_var, double *arg_ub_var, double *arg_lb_con, double *arg_ub_con, double arg_lb_obj, double arg_ub_obj){
    static_cast<CProblemspec*>(ptr)->objLo = arg_lb_obj;
    static_cast<CProblemspec*>(ptr)->objUp = arg_ub_obj;

    static_cast<CProblemspec*>(ptr)->lb_var.Dimension(static_cast<CProblemspec*>(ptr)->nVar);
    static_cast<CProblemspec*>(ptr)->ub_var.Dimension(static_cast<CProblemspec*>(ptr)->nVar);
    static_cast<CProblemspec*>(ptr)->lb_con.Dimension(static_cast<CProblemspec*>(ptr)->nCon);
    static_cast<CProblemspec*>(ptr)->ub_con.Dimension(static_cast<CProblemspec*>(ptr)->nCon);
    
    std::copy(arg_lb_var, arg_lb_var + static_cast<CProblemspec*>(ptr)->nVar, static_cast<CProblemspec*>(ptr)->lb_var.array);
    std::copy(arg_ub_var, arg_ub_var + static_cast<CProblemspec*>(ptr)->nVar, static_cast<CProblemspec*>(ptr)->ub_var.array);
    
    std::copy(arg_lb_con, arg_lb_con + static_cast<CProblemspec*>(ptr)->nCon, static_cast<CProblemspec*>(ptr)->lb_con.array);
    std::copy(arg_ub_con, arg_ub_con + static_cast<CProblemspec*>(ptr)->nCon, static_cast<CProblemspec*>(ptr)->ub_con.array);
    
    return;
}

CDLEXP void Problemspec_set_blockIdx(void *ptr, int *arg_blockIdx, int arg_nBlocks){
    static_cast<CProblemspec*>(ptr)->nBlocks = arg_nBlocks;
    delete[] static_cast<CProblemspec*>(ptr)->blockIdx;
    static_cast<CProblemspec*>(ptr)->blockIdx = new int[arg_nBlocks + 1];
    std::copy(arg_blockIdx, arg_blockIdx + arg_nBlocks + 1, static_cast<CProblemspec*>(ptr)->blockIdx);
}

CDLEXP void Problemspec_set_vblocks(void *ptr, void *arg_vblocks, int arg_n_vblocks){
    delete[] static_cast<CProblemspec*>(ptr)->vblocks;
    static_cast<CProblemspec*>(ptr)->n_vblocks = arg_n_vblocks;
    static_cast<CProblemspec*>(ptr)->vblocks = new vblock[arg_n_vblocks];
    std::copy(static_cast<vblock*>(arg_vblocks), static_cast<vblock*>(arg_vblocks) + arg_n_vblocks, static_cast<CProblemspec*>(ptr)->vblocks);
}

CDLEXP void Problemspec_pass_vblocks(void *ptr, void *arg_vblocks, int arg_n_vblocks){
    delete[] static_cast<CProblemspec*>(ptr)->vblocks;
    static_cast<CProblemspec*>(ptr)->n_vblocks = arg_n_vblocks;
    static_cast<CProblemspec*>(ptr)->vblocks = static_cast<vblock*>(arg_vblocks);
}

CDLEXP void Problemspec_set_closure(void *ptr, void *arg_closure) {
    static_cast<CProblemspec*>(ptr)->closure = arg_closure;
}

CDLEXP void Problemspec_set_dense_init(void *ptr, void (*fp_init_dense)(void* closure_pass, double* xi, double* lambda, double* constrJac)) {
    static_cast<CProblemspec*>(ptr)->initialize_dense = fp_init_dense;
}

CDLEXP void Problemspec_set_dense_eval(void *ptr, void (*fp_eval_dense)(void *closure_pass, const double *xi, const double* lambda, double *objval, double *constr, double *gradObj, double *constrJac, double **hess, int dmode, int *info)) {
    static_cast<CProblemspec*>(ptr)->evaluate_dense = fp_eval_dense;
}

CDLEXP void Problemspec_set_simple_eval(void *ptr, void (*fp_eval_simple)(void *closure_pass, const double *xi, double *objval, double *constr, int *info)) {
    static_cast<CProblemspec*>(ptr)->evaluate_simple = fp_eval_simple;
}

CDLEXP void Problemspec_set_sparse_init(void *ptr, void (*fp_init_sparse)(void *closure_pass, double *xi, double *lambda, double *jacNz, int *jacIndRow, int *jacIndCol)) {
    static_cast<CProblemspec*>(ptr)->initialize_sparse = fp_init_sparse;
}

CDLEXP void Problemspec_set_sparse_eval(void *ptr, void (*fp_eval_sparse)(void *closure_pass, const double *xi, const double *lambda, double *objval, double *constr, double *gradObj, double *jacNz, int *jacIndRow, int *jacIndCol, double **hess, int dmode, int *info)) {
    static_cast<CProblemspec*>(ptr)->evaluate_sparse = fp_eval_sparse;
}

CDLEXP void Problemspec_set_continuity_restoration(void *ptr, void (*fp_rest_cont)(void *closure_pass, double *xi, int *info)) {
    static_cast<CProblemspec*>(ptr)->restore_continuity = fp_rest_cont;
}

CDLEXP void *create_SQPmethod(void *Problemspec_prob, void *SQPoptions_opts, void *SQPstats_stats){
    try{
        return static_cast<void*>(new SQPmethod(static_cast<Problemspec*>(Problemspec_prob), static_cast<SQPoptions*>(SQPoptions_opts), static_cast<SQPstats*>(SQPstats_stats)));
    }
    catch (std::exception &E){
        strncpy(CblockSQP_error_message, E.what(), MAXLEN_CBLOCKSQP_ERROR_MESSAGE);
    }
    CblockSQP_error_message[MAXLEN_CBLOCKSQP_ERROR_MESSAGE - 1] = '\0';
    return nullptr;
}

CDLEXP void delete_SQPmethod(void *ptr){
    delete static_cast<SQPmethod*>(ptr);
}

CDLEXP void SQPmethod_init(void *ptr){
    static_cast<SQPmethod*>(ptr)->init();
}

CDLEXP int SQPmethod_run(void *ptr, int maxIt, int warmStart){
    
    try{
        return static_cast<int>(static_cast<SQPmethod*>(ptr)->run(maxIt, warmStart));
    }
    catch (std::exception &E){
        strncpy(CblockSQP_error_message, E.what(), MAXLEN_CBLOCKSQP_ERROR_MESSAGE);
    }
    CblockSQP_error_message[MAXLEN_CBLOCKSQP_ERROR_MESSAGE - 1] = '\0';
    return -1000;
}

CDLEXP void SQPmethod_finish(void *ptr){
    static_cast<SQPmethod*>(ptr)->finish();
}

CDLEXP double *SQPmethod_release_xi(void *ptr){
    Matrix xi(static_cast<SQPmethod*>(ptr)->get_xi());
    double *xi_array = (double *) malloc(xi.m*sizeof(double));
    memcpy(xi_array, xi.array, xi.m*sizeof(double));
    return xi_array;
    //return static_cast<SQPmethod*>(ptr)->get_xi().release();
}

CDLEXP double *SQPmethod_release_lambda(void *ptr){
    Matrix lambda(static_cast<SQPmethod*>(ptr)->get_lambda());
    double *lambda_array = (double *) malloc(lambda.m*sizeof(double));
    memcpy(lambda_array, lambda.array, lambda.m*sizeof(double));
    return lambda_array;
    //return static_cast<SQPmethod*>(ptr)->get_lambda().release();
}

/*
mod.add_type<blockSQP::SQPmethod>("Cxx_SQPmethod")
    .constructor<blockSQP::Problemspec*, blockSQP::SQPoptions*, blockSQP::SQPstats*>()
    .method("cpp_init", &blockSQP::SQPmethod::init)
    .method("cpp_run", &blockSQP::SQPmethod::run)
    .method("cpp_finish", &blockSQP::SQPmethod::finish)
    .method("get_primal", [](blockSQP::SQPmethod &optimizer){
            double *ret_xi = new double[optimizer.prob->nVar];
            if (optimizer.param->automatic_scaling){
                for (int i = 0; i < optimizer.prob->nVar; i++){
                    ret_xi[i] = optimizer.vars->xi(i) / optimizer.scaled_prob->scaling_factors[i];
                }
            }
            else{
                for (int i = 0; i < optimizer.prob->nVar; i++){
                    ret_xi[i] = optimizer.vars->xi(i);
                }
            }
            return ret_xi;
        }
    )
    .method("get_dual", [](blockSQP::SQPmethod &optimizer){
            double *ret_lam = new double[optimizer.prob->nVar + optimizer.prob->nCon];
            for (int i = 0; i < optimizer.prob->nVar + optimizer.prob->nCon; i++){
                ret_lam[i] = optimizer.vars->lambda(i);
            }
            return ret_lam;
        }
    )
    ;
*/

//////////////////////////////////////
//Classes and methods for condensing//
//////////////////////////////////////

/*
mod.add_type<blockSQP::cblock>("cblock")
    .constructor<int>()
    ;

mod.add_type<blockSQP::condensing_target>("condensing_target")
    .constructor<int, int, int, int, int>()
    .method("get_first_free", [](blockSQP::condensing_target &T){return T.first_free;})
    .method("get_vblock_end", [](blockSQP::condensing_target &T){return T.vblock_end;})
    ;

mod.add_type<cblock_array>("cblock_array")
    .constructor<int>()
    .method("array_set", [](cblock_array &ARR, int index, blockSQP::cblock V){ARR.ptr[index - 1] = V;})
    .method("array_get", [](cblock_array &ARR, int index){return ARR.ptr[index - 1];})
    ;

mod.add_type<int_array>("int_array")
    .constructor<int>()
    .method("array_set", [](int_array &ARR, int index, int V){ARR.ptr[index - 1] = V;})
    .method("array_get", [](int_array &ARR, int index){return ARR.ptr[index - 1] ;})
    ;

mod.add_type<condensing_targets>("condensing_targets")
    .constructor<int>()
    .method("get_first_first_free", [](condensing_targets &arr){return arr.ptr[0].first_free;})
    .method("get_first_vblock_end", [](condensing_targets &arr){return arr.ptr[0].vblock_end;})
    .method("array_set", [](condensing_targets &ARR, int index, blockSQP::condensing_target V){ARR.ptr[index - 1] = V;})
    .method("array_get", [](condensing_targets &ARR, int index){return ARR.ptr[index - 1];})
    ;

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


mod.add_type<SymMat_array>("SymMat_array")
    .constructor<int>()
    .method("array_size", [](SymMat_array &ARR){return ARR.size;})
    .method("array_set!", [](SymMat_array &ARR, int index, blockSQP::SymMatrix V){ARR.ptr[index - 1] = V;})
    .method("array_get", [](SymMat_array &ARR, int index){return ARR.ptr[index - 1];})
    .method("array_get_ptr", [](SymMat_array &ARR, int index){return ARR.ptr + index - 1;})
    ;

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

//SCQP (sequential condensed quadratic programming) classes jlcxx::julia_base_type<blockSQP::Problemspec>()

/*
mod.add_type<blockSQP::SCQPmethod>("Cxx_SCQPmethod", jlcxx::julia_base_type<blockSQP::SQPmethod>())
    .constructor<blockSQP::Problemspec*, blockSQP::SQPoptions*, blockSQP::SQPstats*, blockSQP::Condenser*>()
    ;

mod.add_type<blockSQP::SCQP_bound_method>("Cxx_SCQP_bound_method", jlcxx::julia_base_type<blockSQP::SCQPmethod>())
    .constructor<blockSQP::Problemspec*, blockSQP::SQPoptions*, blockSQP::SQPstats*, blockSQP::Condenser*>()
    ;

mod.add_type<blockSQP::SCQP_correction_method>("Cxx_SCQP_correction_method", jlcxx::julia_base_type<blockSQP::SCQPmethod>())
    .constructor<blockSQP::Problemspec*, blockSQP::SQPoptions*, blockSQP::SQPstats*, blockSQP::Condenser*>()
    ;
*/


