/*
 * blockSQP -- Sequential quadratic programming for problems with
 *             block-diagonal Hessian matrix.
 * Copyright (C) 2012-2015 by Dennis Janka <dennis.janka@iwr.uni-heidelberg.de>
 *
 * Licensed under the zlib license. See LICENSE for more details.
 */

/*
 * blockSQP2 -- A structure-exploiting nonlinear programming solver based
 *              on blockSQP by Dennis Janka.
 * Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
 * 
 * Licensed under the zlib license. See LICENSE for more details.
 */

/**
 * \file blocksqp_options.cpp
 * \author Dennis Janka
 * \date 2012-2015
 *
 *  Implementation of SQPoptions class that holds all algorithmic options.
 * 
 * \modifications
 *  \author Reinhold Wittmann
 *  \date 2023-2025
 */

 
#include <blockSQP2/options.hpp>
#include <blockSQP2/qpsolver.hpp>
#include <blockSQP2/defs.hpp>
#include <iostream>
#include <limits>

namespace blockSQP2{


SQPoptions::SQPoptions(){
    #ifdef QPSOLVER_QPOASES
        qpsol = QPsolvers::qpOASES;
    #elif defined(QPSOLVER_GUROBI)
        qpsol = QPsolvers::gurobi;
    #elif defined(QPSOLVER_QPALM)
        qpsol = QPsolvers::qpalm;
    #endif
}

SQPoptions::~SQPoptions(){}


void SQPoptions::optionsConsistency(Problemspec *problem){
    if ((automatic_scaling || (conv_strategy == 2 && max_conv_QPs > 1)) && (problem->vblocks == nullptr || problem->n_vblocks < 1))
        throw ParameterError("automatic_scaling or convexification strategy 2 activated, but no structure information (vblocks) provided to problem specification");
    if (sparse && problem->nnz < 0)
        throw ParameterError("Sparse mode enabled, but number of jacobian non-zero elements not set");
    
    if (problem->cond != nullptr && !block_hess)
        throw ParameterError("Condenser passed, but block updates not enabled");
    
    if (block_hess == 2)
        throw ParameterError("Passing only the last exact Hessian block is disabled in this version");
    if ((problem->nBlocks < 1 || block_hess == 0) && (is_exact(hess_approx) || is_exact(last_block_approx)))
        throw ParameterError("Using exact Hessian for now requires passing blockIdx to Problemspec and enabling block updates through SQPoptions");
    
    optionsConsistency();
}

void SQPoptions::optionsConsistency(){
    //Check if selected QP solver was properly linked
    #ifndef QPSOLVER_QPOASES
        if (qpsol == QPsolvers::qpOASES)
            throw ParameterError("qpOASES specified as QP solver, but not (properly) linked");
    #endif
    #ifndef QPSOLVER_GUROBI
        if (qpsol == QPsolvers::gurobi)
            throw ParameterError("gurobi specified as QP solver, but not (properly) linked");
    #endif
    #ifndef QPSOLVER_QPALM
    if (qpsol == QPsolvers::qpalm)
        throw ParameterError("qpalm specified as QP solver, but not (properly) linked");
    #endif

    //Check for wrong QP options
    if (qpsol_options != nullptr && qpsol_options->sol != qpsol)
        throw ParameterError("Incorrect QP solver options type given for specified QP solver");
    
    if (last_block_approx != Hessians::last_block_default && last_block_approx != Hessians::exact && last_block_approx != Hessians::pos_def_exact)
        throw ParameterError("last_block_approx can only be last_block_default, exact or pos_def_exact");
    
    //Currently, indefinite Hessian approximations are only supported by qpOASES with Schur-complement approach
    if (is_indefinite(hess_approx) || is_indefinite(last_block_approx)){
        if (qpsol == QPsolvers::qpOASES){
            if (qpsol_options == nullptr || static_cast<qpOASES_options*>(qpsol_options)->sparsityLevel == -1){
                if (!sparse) throw ParameterError("Indefinite Hessians not supported for dense qpOASES (derived from SQPoptions::sparse = 0, as no or default qpOASES_options::sparsityLevel was given)");
            }
            else if (static_cast<qpOASES_options*>(qpsol_options)->sparsityLevel != 2)
                throw ParameterError("Indefinite Hessians only supported for qpOASES with Schur-complement approach (qpOASES_options::sparsityLevel = 2)");
        }
        else throw ParameterError("Only qpOASES with option sparsityLevel = 2 currently supports indefinite Hessians");
    }
    
    if (par_QPs){
        #ifdef BLOCKSQP_PAR_QPS_DISABLED
            throw ParameterError("Option par_QPs = true --> error, parallel solution of QPs is disabled in this build.");
        #endif
        if(qpsol == QPsolvers::qpOASES){
            #ifdef SOLVER_MUMPS
                #ifndef SQPROBLEMSCHUR_ENABLE_PASSTHROUGH
                    throw ParameterError("SQPoptions::par_QPs = true --> error, a modified version of qpOASES must be linked to enable parallel solution of QPs with dynamically loaded MUMPS linear solver");
                #elif !defined(LINUX) && !defined(WINDOWS)
                    throw ParameterError("SQPoptions::par_QPs = true --> error, parallel solution of quadratic subproblems is not available on this build or platform");
                #elif !defined(DMUMPS_C_DYN)
                    throw ParameterError("SQPoptions::par_QPs = true --> error, parallel QP solution must be enabled via DMUMPS_C_DYN preprocessor flag if using qpOASES with mumps sparse solver, in addition to providing dynamically loadable mumps shared libraries");
                #endif
            #endif
        }
        if (max_conv_QPs > PAR_QP_MAX - 1)
            throw ParameterError("Only up to SQPoptions::max_conv_QPs == 7 convexified QPs are supported for parallel solution");
    }
    
    if ((is_indefinite(hess_approx) || is_indefinite(last_block_approx)) && is_indefinite(fallback_approx))
        throw ParameterError("Positive definite fallback Hessian is needed when Hessian is not positive definite");
    
    if (enable_linesearch == 1 && is_indefinite(hess_approx) && max_conv_QPs < 1){
        throw ParameterError("Fallback Hessian QP attempts (max_conv_QPs > 1) are required when using SR1.");
    }
    
    if (lim_mem && mem_size == 0) 
        throw ParameterError("hessMemsize must be greater zero for limited memory quasi newton");
    
    
    if (lim_mem && mem_size > 200){
        std::cout << "WARNING: Large value of mem_size (> 200). Performance may be impeded\n";
    }   
    
    complete_QP_options();
}

void SQPoptions::complete_QP_options(){
    //Create default options if no options have been passed
    if (qpsol_options == nullptr){
        if (qpsol == QPsolvers::qpOASES) default_qpsol_options = std::make_unique<qpOASES_options>();
        else if (qpsol == QPsolvers::gurobi) default_qpsol_options = std::make_unique<gurobi_options>();
        else if (qpsol == QPsolvers::qpalm) default_qpsol_options = std::make_unique<qpalm_options>();
        else throw ParameterError("No valid option for QP solver");

        qpsol_options = default_qpsol_options.get();
    }
    //Some values can also be set directly in the options, copy them over default QP solver options.
    if (qpsol_options->eps == 1e-16) qpsol_options->eps = eps;
    if (qpsol_options->inf == std::numeric_limits<double>::infinity()) qpsol_options->inf = inf;  
    if (qpsol_options->max_QP_secs == 10.) qpsol_options->max_QP_secs = max_QP_secs;
    if (qpsol_options->max_QP_it == std::numeric_limits<int>::max()) qpsol_options->max_QP_it = max_QP_it;

    //Infer solver specific options from SQPoptions
    //  Infer qpOASES sparsityLevel
    if (qpsol == QPsolvers::qpOASES && static_cast<qpOASES_options*>(qpsol_options)->sparsityLevel == -1){
        if (!sparse) static_cast<qpOASES_options*>(qpsol_options)->sparsityLevel = 0;
        else         static_cast<qpOASES_options*>(qpsol_options)->sparsityLevel = 2;
    }
}


QPsolver_options::QPsolver_options(QPsolvers SOL): sol(SOL){
    eps = 1e-16;
    inf = std::numeric_limits<double>::infinity();
    max_QP_secs = 10.;
    max_QP_it = std::numeric_limits<int>::max();
}
QPsolver_options::~QPsolver_options(){}

qpOASES_options::qpOASES_options(): QPsolver_options(QPsolvers::qpOASES){
    sparsityLevel = -1;
    printLevel = 0;
    terminationTolerance = 5.0e6*2.221e-16;
}

gurobi_options::gurobi_options(): QPsolver_options(QPsolvers::gurobi){
        Method = 1;
        NumericFocus = 3;
        OutputFlag = 0;
        Presolve = -1;
        Aggregate = 1;
        OptimalityTol = 1e-9;
        FeasibilityTol = 1e-9;
        BarHomogeneous = 0;
        PSDTol = 1e-6;

        //regularization_factor = 1e-8;
}

qpalm_options::qpalm_options(): QPsolver_options(QPsolvers::qpalm){}


} // namespace blockSQP2
