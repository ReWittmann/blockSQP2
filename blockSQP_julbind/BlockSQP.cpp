#include "jlcxx/jlcxx.hpp"
#include "jlcxx/array.hpp"
#include "jlcxx/functions.hpp"
#include "blocksqp_method.hpp"
#include "blocksqp_condensing.hpp"
#include "blocksqp_options.hpp"
#include "blocksqp_problemspec.hpp"
#include "blocksqp_matrix.hpp"
#include <limits>
#include <iostream>
#include <string>

template <typename T> class T_array{
    public:
    int size = 0;
    T *ptr;

    T_array(int size_): size(size_){
        ptr = new T[size];
    }
    T_array(): size(0), ptr(nullptr){}

    ~T_array(){
        delete[] ptr;
    }
};

typedef T_array<blockSQP::vblock> vblock_array;
typedef T_array<blockSQP::cblock> cblock_array;
typedef T_array<blockSQP::SymMatrix> SymMat_array;
typedef T_array<int> int_array;
typedef T_array<blockSQP::condensing_target> condensing_targets;



class Problemform : public blockSQP::Problemspec{
public:
    Problemform(int NVARS, int NCONS){
        nVar = NVARS;
        nCon = NCONS;
    };

    virtual ~Problemform(){
        delete[] blockIdx;
        delete[] vblocks;
    };


    //Allocate callbacks (function pointers to global julia functions)
    void (*initialize_dense)(void *jscope, double *xi, double *lambda, double *constrJac);
    void (*evaluate_dense)(void *jscope, const double *xi, const double* lambda, double *objval, double *constr, double *gradObj, double *constrJac, double **hess, int dmode, int *info);
    void (*evaluate_simple)(void *jscope, const double *xi, double *objval, double *constr, int *info);

    void (*initialize_sparse)(void *jscope, double *xi, double *lambda, double *jacNz, int *jacIndRow, int *jacIndCol);
    void (*evaluate_sparse)(void *jscope, const double *xi, const double *lambda, double *objval, double *constr, double *gradObj, double *jacNz, int *jacIndRow, int *jacIndCol, double **hess, int dmode, int *info);

    void (*restore_continuity)(void *jscope, double *xi, int *info);


    //Pass-through pointer to julia object wrapper
    void *Julia_Scope;


    //Invoke callbacks in overridden methods
    virtual void initialize(blockSQP::Matrix &xi, blockSQP::Matrix &lambda, blockSQP::Matrix &constrJac){
        (*initialize_dense)(Julia_Scope, xi.array, lambda.array, constrJac.array);
    }


    virtual void initialize(blockSQP::Matrix &xi, blockSQP::Matrix &lambda, double *&jacNz, int *&jacIndRow, int *&jacIndCol){
        (*initialize_sparse)(Julia_Scope, xi.array, lambda.array, jacNz, jacIndRow, jacIndCol);
    }

    virtual void evaluate(const blockSQP::Matrix &xi, const blockSQP::Matrix &lambda, double *objval, blockSQP::Matrix &constr, blockSQP::Matrix &gradObj, blockSQP::Matrix &constrJac, blockSQP::SymMatrix *&hess, int dmode, int *info){
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

        (*evaluate_dense)(Julia_Scope, xi.array, lambda.array, objval, constr.array, gradObj.array, constrJac.array, hessNz, dmode, info);

        delete[] hessNz;
    }

    virtual void evaluate(const blockSQP::Matrix &xi, const blockSQP::Matrix &lambda, double *objval, blockSQP::Matrix &constr, blockSQP::Matrix &gradObj, double *&jacNz, int *&jacIndRow, int *&jacIndCol, blockSQP::SymMatrix *&hess, int dmode, int *info){
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

        (*evaluate_sparse)(Julia_Scope, xi.array, lambda.array, objval, constr.array, gradObj.array, jacNz, jacIndRow, jacIndCol, hessNz, dmode, info);

        delete[] hessNz;
    }

    virtual void evaluate(const blockSQP::Matrix &xi, double *objval, blockSQP::Matrix &constr, int *info){
        (*evaluate_simple)(Julia_Scope, xi.array, objval, constr.array, info);
    }


    //Optional Methods
    virtual void reduceConstrVio(blockSQP::Matrix &xi, int *info){
        if (restore_continuity != nullptr){
            (*restore_continuity)(Julia_Scope, xi.array, info);
        }
    };


    //Interface methods
    void set_bounds(jlcxx::ArrayRef<double, 1> LBV, jlcxx::ArrayRef<double, 1> UBV,
                    jlcxx::ArrayRef<double, 1> LBC, jlcxx::ArrayRef<double, 1> UBC,
                    double LBO, double UBO){
        objLo = LBO;
        objUp = UBO;

        lb_var.Dimension(nVar);
        ub_var.Dimension(nVar);
        lb_con.Dimension(nCon);
        ub_con.Dimension(nCon);

        for (int i = 0; i < nVar; i++){
            lb_var(i) = LBV[i];
            ub_var(i) = UBV[i];
        }

        for (int i = 0; i < nCon; i++){
            lb_con(i) = LBC[i];
            ub_con(i) = UBC[i];
        }
        return;
    }
    
    void set_blockIdx(jlcxx::ArrayRef<int,1> BIDX){
        nBlocks = BIDX.size() - 1;
        blockIdx = new int[nBlocks + 1];

        for (int i = 0; i < nBlocks + 1; i++){
            blockIdx[i] = BIDX[i];
        }
    }

    //void set_vblocks(jlcxx::ArrayRef<blockSQP::vblock,1> VB){
    void set_vblocks(vblock_array &VB){
        n_vblocks = VB.size;
        vblocks = new blockSQP::vblock[n_vblocks];
        for (int i = 0; i < n_vblocks; i++){
            vblocks[i] = VB.ptr[i];
        }
    }


    void set_scope(void *JSCOPE){
        Julia_Scope = JSCOPE;
    }

    //Callback setters
    void set_dense_init(void (*INIT_DENSE)(void* jscope, double* xi, double* lambda, double* constrJac)){
        initialize_dense = INIT_DENSE;
    }

    void set_dense_eval(void (*EVAL_DENSE)(void *jscope, const double *xi, const double* lambda, double *objval, double *constr, double *gradObj, double *constrJac, double **hess, int dmode, int *info)){
        evaluate_dense = EVAL_DENSE;
    }

    void set_simple_eval(void (*EVAL_SIMPLE)(void *jscope, const double *xi, double *objval, double *constr, int *info)){
        evaluate_simple = EVAL_SIMPLE;
    }

    void set_sparse_init(void (*INIT_SPARSE)(void *jscope, double *xi, double *lambda, double *jacNz, int *jacIndRow, int *jacIndCol)){
        initialize_sparse = INIT_SPARSE;
    }

    void set_sparse_eval(void (*EVAL_SPARSE)(void *jscope, const double *xi, const double *lambda, double *objval, double *constr, double *gradObj, double *jacNz, int *jacIndRow, int *jacIndCol, double **hess, int dmode, int *info)){
        evaluate_sparse = EVAL_SPARSE;
    }

    void set_continuity_restoration(void (*REST_CONT)(void *jscope, double *xi, int *info)){
        restore_continuity = REST_CONT;
    }

};

class JL_Condenser{
    blockSQP::Condenser *Cxx_Condenser;
    blockSQP::vblock *vblocks;
    blockSQP::cblock *cblocks;
    int *hsizes;
    blockSQP::condensing_target *targets;

    JL_Condenser(jlcxx::ArrayRef<blockSQP::vblock, 1> &VBLOCKS, jlcxx::ArrayRef<blockSQP::cblock, 1> &CBLOCKS, jlcxx::ArrayRef<int, 1> &HSIZES, jlcxx::ArrayRef<blockSQP::condensing_target, 1> &TARGETS, int DEP_BOUNDS){
        vblocks = new blockSQP::vblock[VBLOCKS.size()];
        for (int i = 0; i < VBLOCKS.size(); i++){
            vblocks[i] = VBLOCKS[i];
        }

        cblocks = new blockSQP::cblock[CBLOCKS.size()];
        for (int i = 0; i < CBLOCKS.size(); i++){
            cblocks[i] = CBLOCKS[i];
        }

        hsizes = new int[HSIZES.size()];
        for (int i = 0; i < HSIZES.size(); i++){
            hsizes[i] = HSIZES[i];
        }

        targets = new blockSQP::condensing_target[TARGETS.size()];
        for (int i = 0; i < TARGETS.size(); i++){
            targets[i] = TARGETS[i];
        }

        Cxx_Condenser = new blockSQP::Condenser(vblocks, VBLOCKS.size(), cblocks, CBLOCKS.size(), hsizes, HSIZES.size(), targets, TARGETS.size(), DEP_BOUNDS);
    }

    ~JL_Condenser(){
        delete Cxx_Condenser;
        delete[] vblocks, cblocks, hsizes, targets;
    }
};



class NULL_QPSOLVER_options : public blockSQP::QPSOLVER_options{
    public:
    NULL_QPSOLVER_options() : QPSOLVER_options(blockSQP::QPSOLVER::unset){}
};


namespace jlcxx{
    template<> struct SuperType<Problemform>{typedef blockSQP::Problemspec type;};
    template<> struct SuperType<blockSQP::SCQPmethod>{typedef blockSQP::SQPmethod type;};
    template<> struct SuperType<blockSQP::SCQP_bound_method>{typedef blockSQP::SCQPmethod type;};
    template<> struct SuperType<blockSQP::SCQP_correction_method>{typedef blockSQP::SCQPmethod type;};

    template<> struct SuperType<NULL_QPSOLVER_options>{typedef blockSQP::QPSOLVER_options type;};
    template<> struct SuperType<blockSQP::qpOASES_options>{typedef blockSQP::QPSOLVER_options type;};
    template<> struct SuperType<blockSQP::gurobi_options>{typedef blockSQP::QPSOLVER_options type;};
}


JLCXX_MODULE define_julia_module(jlcxx::Module& mod){
mod.add_type<blockSQP::vblock>("vblock")
    .constructor<int, bool>()
    ;

mod.add_type<vblock_array>("vblock_array")
    .constructor<int>()
    .method("array_set", [](vblock_array &ARR, int index, blockSQP::vblock V){ARR.ptr[index - 1] = V;})
    .method("array_get", [](vblock_array &ARR, int index){return ARR.ptr[index - 1];})
    ;

mod.add_type<blockSQP::QPSOLVER_options>("QPSOLVER_options")
    ;

mod.add_type<NULL_QPSOLVER_options>("NULL_QPSOLVER_options", jlcxx::julia_base_type<blockSQP::QPSOLVER_options>())
    .constructor<>()
    ;

mod.add_type<blockSQP::qpOASES_options>("qpOASES_options", jlcxx::julia_base_type<blockSQP::QPSOLVER_options>())
    .constructor<>()
    .method("set_printLevel", [](blockSQP::qpOASES_options &QPopts, int val){QPopts.printLevel = val;})
    .method("set_terminationTolerance", [](blockSQP::qpOASES_options &QPopts, double val){QPopts.terminationTolerance = val;})
    ;

mod.add_type<blockSQP::gurobi_options>("gurobi_options", jlcxx::julia_base_type<blockSQP::QPSOLVER_options>())
    .constructor<>()
    .method("set_Method", [](blockSQP::gurobi_options &GRBopts, int val){GRBopts.Method = val;})
    .method("set_NumericFocus", [](blockSQP::gurobi_options &GRBopts, int val){std::cout << "Set NumericFocus to " << val << "\n"; GRBopts.NumericFocus = val;})
    .method("set_OutputFlag", [](blockSQP::gurobi_options &GRBopts, int val){GRBopts.OutputFlag = val;})
    .method("set_Presolve", [](blockSQP::gurobi_options &GRBopts, int val){GRBopts.Presolve = val;})
    .method("set_Aggregate", [](blockSQP::gurobi_options &GRBopts, int val){GRBopts.Aggregate = val;})
    .method("set_BarHomogeneous", [](blockSQP::gurobi_options &GRBopts, int val){GRBopts.BarHomogeneous = val;})
    .method("set_OptimalityTol", [](blockSQP::gurobi_options &GRBopts, double val){GRBopts.OptimalityTol = val;})
    .method("set_FeasibilityTol", [](blockSQP::gurobi_options &GRBopts, double val){GRBopts.FeasibilityTol = val;})
    .method("set_PSDTol", [](blockSQP::gurobi_options &GRBopts, double val){GRBopts.PSDTol = val;})
    ;

mod.add_type<blockSQP::SQPoptions>("SQPoptions")
    .constructor<>()
    .method("get_maxItQP", [](blockSQP::SQPoptions &opts){return opts.maxItQP;})
    .method("set_printLevel", [](blockSQP::SQPoptions &opts, int val){opts.printLevel = val;})
    .method("set_printColor", [](blockSQP::SQPoptions &opts, int val){opts.printColor = val;})
    .method("set_debugLevel", [](blockSQP::SQPoptions &opts, int val){opts.debugLevel = val;})
    .method("set_eps", [](blockSQP::SQPoptions &opts, double val){opts.eps = val;})
    .method("set_inf", [](blockSQP::SQPoptions &opts, double val){opts.inf = val;})
    .method("set_opttol", [](blockSQP::SQPoptions &opts, double val){opts.opttol = val;})
    .method("set_nlinfeastol", [](blockSQP::SQPoptions &opts, double val){opts.nlinfeastol = val;})
    .method("set_sparseQP", [](blockSQP::SQPoptions &opts, int val){opts.sparseQP = val;})
    .method("set_globalization", [](blockSQP::SQPoptions &opts, int val){opts.globalization = val;})
    .method("set_restoreFeas", [](blockSQP::SQPoptions &opts, int val){opts.restoreFeas = val;})
    .method("set_maxLineSearch", [](blockSQP::SQPoptions &opts, int val){opts.maxLineSearch = val;})
    .method("set_maxConsecReducedSteps", [](blockSQP::SQPoptions &opts, int val){opts.maxConsecReducedSteps = val;})
    .method("set_maxConsecSkippedUpdates", [](blockSQP::SQPoptions &opts, int val){opts.maxConsecSkippedUpdates = val;})
    .method("set_maxItQP", [](blockSQP::SQPoptions &opts, int val){opts.maxItQP = val;})
    .method("set_blockHess", [](blockSQP::SQPoptions &opts, int val){opts.blockHess = val;})
    .method("set_hessScaling", [](blockSQP::SQPoptions &opts, int val){opts.hessScaling = val;})
    .method("set_fallbackScaling", [](blockSQP::SQPoptions &opts, int val){opts.fallbackScaling = val;})
    .method("set_maxTimeQP", [](blockSQP::SQPoptions &opts, double val){opts.maxTimeQP = val;})
    .method("set_iniHessDiag", [](blockSQP::SQPoptions &opts, double val){opts.iniHessDiag = val;})
    .method("set_colEps", [](blockSQP::SQPoptions &opts, double val){opts.colEps = val;})
    .method("set_olEps", [](blockSQP::SQPoptions &opts, double val){opts.olEps = val;})
    .method("set_colTau1", [](blockSQP::SQPoptions &opts, double val){opts.colTau1 = val;})
    .method("set_colTau2", [](blockSQP::SQPoptions &opts, double val){opts.colTau2 = val;})
    .method("set_hessDampFac", [](blockSQP::SQPoptions &opts, double val){opts.hessDampFac = val;})
    .method("set_minDampQuot", [](blockSQP::SQPoptions &opts, double val){opts.minDampQuot = val;})
    .method("set_hessUpdate", [](blockSQP::SQPoptions &opts, int val){opts.hessUpdate = val;})
    .method("set_fallbackUpdate", [](blockSQP::SQPoptions &opts, int val){opts.fallbackUpdate = val;})
    .method("set_indef_local_only", [](blockSQP::SQPoptions &opts, bool val){opts.indef_local_only = val;})
    .method("set_hessLimMem", [](blockSQP::SQPoptions &opts, int val){opts.hessLimMem = val;})
    .method("set_hessMemsize", [](blockSQP::SQPoptions &opts, int val){opts.hessMemsize = val;})
    .method("set_whichSecondDerv", [](blockSQP::SQPoptions &opts, int val){opts.whichSecondDerv = val;})
    .method("set_skipFirstGlobalization", [](blockSQP::SQPoptions &opts, int val){opts.skipFirstGlobalization = val;})
    .method("set_convStrategy", [](blockSQP::SQPoptions &opts, int val){opts.convStrategy = val;})
    .method("set_maxConvQP", [](blockSQP::SQPoptions &opts, int val){opts.maxConvQP = val;})
    .method("set_hess_regularizationFactor", [](blockSQP::SQPoptions &opts, double val){opts.hess_regularizationFactor = val;})
    .method("set_maxSOCiter", [](blockSQP::SQPoptions &opts, int val){opts.maxSOCiter = val;})
    .method("set_max_bound_refines", [](blockSQP::SQPoptions &opts, int val){opts.max_bound_refines = val;})
    .method("set_max_correction_steps", [](blockSQP::SQPoptions &opts, int val){opts.max_correction_steps = val;})
    .method("set_dep_bound_tolerance", [](blockSQP::SQPoptions &opts, double val){opts.dep_bound_tolerance = val;})
    .method("set_QPsol", [](blockSQP::SQPoptions &opts, std::string &QPsolver_name){
        if (QPsolver_name == "qpOASES") opts.QPsol = blockSQP::QPSOLVER::qpOASES;
        else if (QPsolver_name == "gurobi") opts.QPsol = blockSQP::QPSOLVER::gurobi;
        else throw blockSQP::ParameterError("Unknown QP solver options, known are blockSQP::qpOASES_options, blockSQP::gurobi_options");
    })
    .method("get_QPsol", [](blockSQP::SQPoptions &opts){
        if (opts.QPsol == blockSQP::QPSOLVER::qpOASES) return std::string("qpOASES");
        else if (opts.QPsol == blockSQP::QPSOLVER::gurobi) return std::string("gurobi");
        else throw blockSQP::ParameterError("Unknown QP solver name, known (no neccessarily linked) are qpOASES, gurobi");
    })
    
    //.method("get_QPsol", [](blockSQP::SQPoptions &opts){return 1;})
    //.method("set_QPsol_opts", [](blockSQP::SQPoptions &opts, blockSQP::qpOASES_options QPopts){opts.QPsol_opts = &QPopts;})
    .method("set_QPsol_opts", [](blockSQP::SQPoptions &opts, blockSQP::QPSOLVER_options *QPopts){opts.QPsol_opts = QPopts;})
    .method("set_autoScaling", [](blockSQP::SQPoptions &opts, bool val){opts.autoScaling = val;})
	.method("set_max_local_lenience", [](blockSQP::SQPoptions &opts, int val){opts.max_local_lenience = val;})
    .method("set_max_extra_steps", [](blockSQP::SQPoptions &opts, int val){opts.max_extra_steps = val;})
    ;
    

mod.add_type<blockSQP::SQPstats>("SQPstats")
    .constructor<char*>()
    .method("get_pathstr", [](blockSQP::SQPstats S){return std::string(S.outpath);})
    ;


mod.add_bits<blockSQP::RES>("RES", jlcxx::julia_type("CppEnum"));
mod.set_const("IT_FINISHED", blockSQP::RES::IT_FINISHED);
mod.set_const("FEAS_SUCCESS", blockSQP::RES::FEAS_SUCCESS);
mod.set_const("SUCCESS", blockSQP::RES::SUCCESS);
mod.set_const("SUPER_SUCCESS", blockSQP::RES::SUPER_SUCCESS);
mod.set_const("LOCAL_INFEASIBILITY", blockSQP::RES::LOCAL_INFEASIBILITY);
mod.set_const("RESTORATION_FAILURE", blockSQP::RES::RESTORATION_FAILURE);
mod.set_const("LINESEARCH_FAILURE", blockSQP::RES::LINESEARCH_FAILURE);
mod.set_const("QP_FAILURE", blockSQP::RES::QP_FAILURE);
mod.set_const("EVAL_FAILURE", blockSQP::RES::EVAL_FAILURE);
mod.set_const("MISC_ERROR", blockSQP::RES::MISC_ERROR);


mod.add_type<blockSQP::Problemspec>("Problemspec");

mod.add_type<Problemform>("Problemform", jlcxx::julia_base_type<blockSQP::Problemspec>())
    .constructor<int, int>()
    .method("set_bounds", &Problemform::set_bounds)
    .method("set_blockIdx", &Problemform::set_blockIdx)
    .method("set_nnz", [](Problemform &P, int NNZ){P.nnz = NNZ; return;})
    .method("set_dense_init", &Problemform::set_dense_init)
    .method("set_dense_eval", &Problemform::set_dense_eval)
    .method("set_simple_eval", &Problemform::set_simple_eval)
    .method("set_sparse_init", &Problemform::set_sparse_init)
    .method("set_sparse_eval", &Problemform::set_sparse_eval)
    .method("set_continuity_restoration", &Problemform::set_continuity_restoration)
    .method("set_scope", &Problemform::set_scope)
    .method("set_vblocks", &Problemform::set_vblocks)
    ;

mod.add_type<blockSQP::SQPiterate>("SQPiterate");

mod.add_type<blockSQP::SQPmethod>("SQPmethod")
    .constructor<blockSQP::Problemspec*, blockSQP::SQPoptions*, blockSQP::SQPstats*>()
    .method("cpp_init", &blockSQP::SQPmethod::init)
    .method("cpp_run", &blockSQP::SQPmethod::run)
    .method("cpp_finish", &blockSQP::SQPmethod::finish)
    .method("get_primal", [](blockSQP::SQPmethod &optimizer){
            double *ret_xi = new double[optimizer.prob->nVar];
            if (optimizer.param->autoScaling){
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


//////////////////////////////////////
//Classes and methods for condensing//
//////////////////////////////////////
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



mod.add_type<blockSQP::Sparse_Matrix>("Sparse_Matrix")
    .constructor<>()
    .method("disown!", [](blockSQP::Sparse_Matrix &M){
        M.m = 0;
        M.n = 0;
        M.nnz = 0;
        M.nz = nullptr;
        M.row = nullptr;
        M.colind = nullptr;
    })
    .method("get_nnz", [](blockSQP::Sparse_Matrix &M){return M.nnz;})
    .method("show_nz", [](blockSQP::Sparse_Matrix &M){return M.nz;})
    .method("show_row", [](blockSQP::Sparse_Matrix &M){return M.row;})
    .method("show_colind", [](blockSQP::Sparse_Matrix &M){return M.colind;})
    ;

mod.method("alloc_Sparse_Matrix", [](int m, int n, int nnz){
    return blockSQP::Sparse_Matrix(m, n, nnz, new double[nnz], new int[nnz], new int[n+1]);
});


mod.add_type<SymMat_array>("SymMat_array")
    .constructor<int>()
    .method("array_size", [](SymMat_array &ARR){return ARR.size;})
    .method("array_set!", [](SymMat_array &ARR, int index, blockSQP::SymMatrix V){ARR.ptr[index - 1] = V;})
    .method("array_get", [](SymMat_array &ARR, int index){return ARR.ptr[index - 1];})
    .method("array_get_ptr", [](SymMat_array &ARR, int index){return ARR.ptr + index - 1;})
    ;

mod.add_type<blockSQP::Condenser>("Cpp_Condenser")
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
                //std::cout << "grad_obj\n" << grad_obj << "\nlb_var\n" << lb_var << "\nub_var\n" << ub_var << "\nlb_con\n" << lb_con << "\nub_con\n" << ub_con << "\n";
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


//SCQP (sequential condensed quadratic programming) classes jlcxx::julia_base_type<blockSQP::Problemspec>()
mod.add_type<blockSQP::SCQPmethod>("SCQPmethod", jlcxx::julia_base_type<blockSQP::SQPmethod>())
    .constructor<blockSQP::Problemspec*, blockSQP::SQPoptions*, blockSQP::SQPstats*, blockSQP::Condenser*>()
    ;

mod.add_type<blockSQP::SCQP_bound_method>("SCQP_bound_method", jlcxx::julia_base_type<blockSQP::SCQPmethod>())
    .constructor<blockSQP::Problemspec*, blockSQP::SQPoptions*, blockSQP::SQPstats*, blockSQP::Condenser*>()
    ;

mod.add_type<blockSQP::SCQP_correction_method>("SCQP_correction_method", jlcxx::julia_base_type<blockSQP::SCQPmethod>())
    .constructor<blockSQP::Problemspec*, blockSQP::SQPoptions*, blockSQP::SQPstats*, blockSQP::Condenser*>()
    ;




}

