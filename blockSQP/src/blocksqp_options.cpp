/*
 * blockSQP -- Sequential quadratic programming for problems with
 *             block-diagonal Hessian matrix.
 * Copyright (C) 2012-2015 by Dennis Janka <dennis.janka@iwr.uni-heidelberg.de>
 *
 * Licensed under the zlib license. See LICENSE for more details.
 */

/**
 * \file blocksqp_options.cpp
 * \author Dennis Janka
 * \date 2012-2015
 *
 *  Implementation of SQPoptions class that holds all algorithmic options.
 */

#include "blocksqp_options.hpp"
#include "blocksqp_qpsolver.hpp"
#include <iostream>
#include <limits>

namespace blockSQP
{

/**
 * Standard Constructor:
 * Default settings
 */

SQPoptions::SQPoptions(){
    #ifdef QPSOLVER_QPOASES
        qpsol = QPsolvers::qpOASES;
    #elif defined(QPSOLVER_GUROBI)
        qpsol = QPsolvers::gurobi;
    #elif defined(QPSOLVER_QPALM)
        qpsol = QPsolvers::qpalm;
    #endif
}

void SQPoptions::reset(){
    /* qpOASES: dense (0), sparse (1), or Schur (2)
     * Choice of qpOASES method:
     * 0: dense Hessian and Jacobian, dense factorization of reduced Hessian
     * 1: sparse Hessian and Jacobian, dense factorization of reduced Hessian
     * 2: sparse Hessian and Jacobian, Schur complement approach (recommended) */
    sparse = 2;

    // 0: no output, 1: normal output, 2: verbose output
    print_level = 2;
    // 1: (some) colorful output
    result_print_color = 1;

    /* 0: no debug output, 1: print one line per iteration to file,
       2: extensive debug output to files (impairs performance) */
    debug_level = 0;

    //eps = 2.2204e-16;
    eps = 1.0e-16;
    //inf = 1.0e20;
    inf = std::numeric_limits<double>::infinity();
    opt_tol = 1.0e-6;
    feas_tol = 1.0e-6;

    #if defined(QPSOLVER_QPOASES)
        qpsol = QPsolvers::qpOASES;
    #elif defined(QPSOLVER_GUROBI)
        qpsol = QPsolvers::gurobi;
    #else
        qpsol = QPsolvers::unset;
    #endif


    // 0: no enable_linesearch, 1: filter line search
    enable_linesearch = 1;

    // 0: no feasibility restoration phase 1: if line search fails, start feasibility restoration phase
    enable_rest = 1;

    rest_zeta = 1e-6;
    rest_rho = 1.0;

    // 0: enable_linesearch is always active, 1: take a full step at first SQP iteration, no matter what
    skip_first_linesearch = false;

    // 0: one update for large Hessian, 1: apply updates blockwise, 2: 2 blocks: 1 block updates, 1 block Hessian of obj.
    block_hess = 1;

    // after too many consecutive skipped updates, Hessian block is reset to (scaled) identity
    max_consec_skipped_updates = 100;

    // for which blocks should second derivatives be provided by the user:
    // 0: none, 1: for the last block, 2: for all blocks
    exact_hess = 0;

    // 0: initial Hessian is diagonal matrix, 1: scale initial Hessian according to Nocedal p.143,
    // 2: scale initial Hessian with Oren-Luenberger factor 3: geometric mean of 1 and 2
    // 4: centered Oren-Luenberger sizing according to Tapia paper
    sizing = 2;
    fallback_sizing = 4;
    initial_hess_scale = 1.0;
    //HessDiag2 = 1.0;

    // Activate damping strategy for BFGS (if deactivated, BFGS might yield indefinite updates!)
    //hessDamp = 1;

    // Damping factor for Powell modification of BFGS updates ( between 0.0 and 1.0 )
    //Originally: BFGS_damping_factor = 0.2;
    BFGS_damping_factor = 1./3.;

    //
    min_damping_quotient = 1e-12;

    //Originally: 1e2*eps, 1e-8
    SR1_abstol = 1e-18;
    SR1_reltol = 1e-5;

    // 0: (sized) identity, 1: SR1, 2: BFGS (damped), 3: [not used] , 4: finiteDiff, 5: Gauss-Newton, 6: BFGS (undamped), ...
    hess_approx = 1;
    fallback_approx = 2;

    indef_local_only = false;

    max_filter_overrides = 2;
    //size_hessian_first_step = false;
    conv_tau_H = 2./3.;
    conv_kappa_0 = 1./16.;
    conv_kappa_max = 8.;

    //0: Convex combinations between indefinite hessian and convex fallback hessian
    //1: First try adding decreasing scaled identities, then use convex fallback hessian
    conv_strategy = 1;

    // How many ADDITIONAL (convexified) QPs may be solved per iteration?
    max_conv_QPs = 4;

    //The identity scaled with this factor is added to convex hessian approximations.
    //Useful if the selected qp solver requires/performs better with convex hessians with eigenvalues uniformly > m_H for some m_H > 0
    reg_factor = 0.0;

    //Options for solving additional QPs to enforce bounds on dependent variables
    max_bound_refines = 3;
    max_correction_steps = 5;


    // 0: full memory updates 1: limited memory
    lim_mem = 1;

    // memory size for L-BFGS/L-SR1 updates
    mem_size = 20;

    // maximum number of line search iterations
    max_linesearch_steps = 10;

    // if step has to be reduced in too many consecutive iterations, step/restoration heuristics are invoked
    max_consec_reduced_steps = 12;

    // maximum number of second-order correction steps
    max_SOC = 3;

    // maximum number of QP iterations per QP solve
    max_QP_it = 5000;
    // maximum time (in seconds) for one QP solve
    max_QP_secs = 10000.0;

    // Oren-Luenberger scaling parameters
    COL_eps = 0.1;
    // Minimum sizing factor in first iteration (OL sizing)
    OL_eps = 1.0e-4;
    COL_tau_1 = 0.5;
    COL_tau_2 = 1.0e4;

    // Filter line search parameters
    gammaTheta = 1.0e-5;
    gammaF = 1.0e-5;
    kappaSOC = 0.99;
    //kappaF = 0.999;
    kappaF = 0.8;
    thetaMax = 1.0e7;       // reject steps if constr viol. is larger than thetaMax
    thetaMin = 1.0e-5;      // if constr viol. is smaller than thetaMin require Armijo cond. for obj.
    delta = 1.0;
    sTheta = 1.1;
    sF = 2.3;
    eta = 1.0e-4;


    automatic_scaling = false;

    //For SCQPmethod subclasses
    dep_bound_tolerance = 1e-7;

    qpsol = QPsolvers::unset;
    qpsol_options = nullptr;

    default_qpsol_options = nullptr;

    enable_premature_termination = true;
    max_extra_steps = 0;
}

SQPoptions::~SQPoptions(){}


void SQPoptions::optionsConsistency(Problemspec *problem){
    if ((automatic_scaling || (conv_strategy == 2 && max_conv_QPs > 1)) && (problem->vblocks == nullptr || problem->n_vblocks < 1))
        throw ParameterError("automatic_scaling or convexification strategy 2 activated, but no structure information (vblocks) provided problem specification");
    if (sparse && problem->nnz < 0)
        throw ParameterError("Sparse mode enabled, but number of jacobian non-zero elements not set");
    
    if (problem->cond != nullptr && !block_hess)
        throw ParameterError("Condenser passed, but block updates not enabled");
        
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
    
    //Currently, indefinite Hessian approximations are only supported by qpOASES with Schur-complement approach
    if (hess_approx == 1 || hess_approx == 4 || exact_hess > 0){
        if (qpsol == QPsolvers::qpOASES){
            if (qpsol_options == nullptr || static_cast<qpOASES_options*>(qpsol_options)->sparsityLevel == -1){
                if (!sparse) throw ParameterError("Indefinite Hessians not supported for dense qpOASES (derived from SQPoptions::sparse = 0, as no or default qpOASES_options::sparsityLevel was given)");
            }
            else if (static_cast<qpOASES_options*>(qpsol_options)->sparsityLevel != 2)
                throw ParameterError("Indefinite Hessians only supported for qpOASES with Schur-complement approach (qpOASES_options::sparsityLevel = 2)");
        }
        else throw ParameterError("Only qpOASES with option sparsityLevel = 2 currently supports indefinite Hessians");
    }
    
    #ifndef ENABLE_PAR_QPS
        if (par_QPs) throw ParameterError("Parallel solution of QPs activated, but solver not built for parallel QPs");
    #endif
    
    // If we compute second constraints derivatives then no update or sizing is needed for the first hessian
    if (exact_hess == 2){
        std::cout << "Exact Hessian is available, hessUpdate and sizing are ignored\n";
    }
    
    //Ensure a positive definite fallback hessian is available if first hessian approximation is not guaranteed to be positive definite
    if ((exact_hess > 0 || hess_approx == 1 || hess_approx == 4 || hess_approx == 6) && max_conv_QPs < 1 && !(fallback_approx == 0 || fallback_approx == 2 || fallback_approx == 5)) 
        throw ParameterError("Positive definite fallback hessian is needed when Hessian is not positive definite");


    if (enable_linesearch == 1 && hess_approx == 1 && max_conv_QPs < 1){
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
        else              static_cast<qpOASES_options*>(qpsol_options)->sparsityLevel = 2;
    }

    return;
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


} // namespace blockSQP
