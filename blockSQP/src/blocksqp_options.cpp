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
        QPsol = QPSOLVER::qpOASES;
    #elifdef QPSOLVER_GUROBI
        QPsol = QPSOLVER::gurobi;
    #elifdef QPSOLVER_QPALM
        QPSOL = QPSOLVER::qpalm;
    #endif
}

void SQPoptions::reset(){
    /* qpOASES: dense (0), sparse (1), or Schur (2)
     * Choice of qpOASES method:
     * 0: dense Hessian and Jacobian, dense factorization of reduced Hessian
     * 1: sparse Hessian and Jacobian, dense factorization of reduced Hessian
     * 2: sparse Hessian and Jacobian, Schur complement approach (recommended) */
    sparseQP = 2;

    // 0: no output, 1: normal output, 2: verbose output
    printLevel = 2;
    // 1: (some) colorful output
    printColor = 1;

    /* 0: no debug output, 1: print one line per iteration to file,
       2: extensive debug output to files (impairs performance) */
    debugLevel = 0;

    //eps = 2.2204e-16;
    eps = 1.0e-16;
    //inf = 1.0e20;
    inf = std::numeric_limits<double>::infinity();
    opttol = 1.0e-6;
    nlinfeastol = 1.0e-6;

    #if defined(QPSOLVER_QPOASES)
        QPsol = QPSOLVER::qpOASES;
    #elif defined(QPSOLVER_GUROBI)
        QPsol = QPSOLVER::gurobi;
    #else
        QPsol = QPSOLVER::unset;
    #endif


    // 0: no globalization, 1: filter line search
    globalization = 1;

    // 0: no feasibility restoration phase 1: if line search fails, start feasibility restoration phase
    restoreFeas = 1;

    restZeta = 1e-6;
    restRho = 1.0;

    // 0: globalization is always active, 1: take a full step at first SQP iteration, no matter what
    skipFirstGlobalization = false;

    // 0: one update for large Hessian, 1: apply updates blockwise, 2: 2 blocks: 1 block updates, 1 block Hessian of obj.
    blockHess = 1;

    // after too many consecutive skipped updates, Hessian block is reset to (scaled) identity
    maxConsecSkippedUpdates = 100;

    // for which blocks should second derivatives be provided by the user:
    // 0: none, 1: for the last block, 2: for all blocks
    whichSecondDerv = 0;

    // 0: initial Hessian is diagonal matrix, 1: scale initial Hessian according to Nocedal p.143,
    // 2: scale initial Hessian with Oren-Luenberger factor 3: geometric mean of 1 and 2
    // 4: centered Oren-Luenberger sizing according to Tapia paper
    hessScaling = 2;
    fallbackScaling = 4;
    iniHessDiag = 1.0;
    //HessDiag2 = 1.0;

    // Activate damping strategy for BFGS (if deactivated, BFGS might yield indefinite updates!)
    //hessDamp = 1;

    // Damping factor for Powell modification of BFGS updates ( between 0.0 and 1.0 )
    //Originally: hessDampFac = 0.2;
    hessDampFac = 1./3.;

    //
    minDampQuot = 1e-12;

    //Originally: 1e2*eps, 1e-8
    SR1_abstol = 1e-18;
    SR1_reltol = 1e-5;

    // 0: (sized) identity, 1: SR1, 2: BFGS (damped), 3: [not used] , 4: finiteDiff, 5: Gauss-Newton, 6: BFGS (undamped), ...
    hessUpdate = 1;
    fallbackUpdate = 2;

    indef_local_only = false;

    max_local_lenience = 2;
    //size_hessian_first_step = false;
    tau_H = 2./3.;
    convKappa0 = 1./16.;
    convKappaMax = 8.;

    //0: Convex combinations between indefinite hessian and convex fallback hessian
    //1: First try adding decreasing scaled identities, then use convex fallback hessian
    convStrategy = 1;

    // How many ADDITIONAL (convexified) QPs may be solved per iteration?
    maxConvQP = 4;

    //The identity scaled with this factor is added to convex hessian approximations.
    //Useful if the selected qp solver requires/performs better with convex hessians with eigenvalues uniformly > m_H for some m_H > 0
    hess_regularizationFactor = 0.0;

    //Options for solving additional QPs to enforce bounds on dependent variables
    max_bound_refines = 3;
    max_correction_steps = 5;


    // 0: full memory updates 1: limited memory
    hessLimMem = 1;

    // memory size for L-BFGS/L-SR1 updates
    hessMemsize = 20;

    // maximum number of line search iterations
    maxLineSearch = 10;

    // if step has to be reduced in too many consecutive iterations, step/restoration heuristics are invoked
    maxConsecReducedSteps = 12;

    // maximum number of second-order correction steps
    maxSOCiter = 3;

    // maximum number of QP iterations per QP solve
    maxItQP = 5000;
    // maximum time (in seconds) for one QP solve
    maxTimeQP = 10000.0;

    // Oren-Luenberger scaling parameters
    colEps = 0.1;
    // Minimum sizing factor in first iteration (OL sizing)
    olEps = 1.0e-4;
    colTau1 = 0.5;
    colTau2 = 1.0e4;

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


    autoScaling = false;

    //For SCQPmethod subclasses
    dep_bound_tolerance = 1e-7;

    QPsol = QPSOLVER::unset;
    QPsol_opts = nullptr;

    default_QPsol_opts = nullptr;

    allow_premature_termination = true;
    max_extra_steps = 0;
}

SQPoptions::~SQPoptions(){
    //delete default_QPsol_opts;
}


/**
 * Some options cannot be set together, check for invalid combinations here
 */

void SQPoptions::optionsConsistency(){ 
    #ifndef QPSOLVER_QPOASES
        if (QPsol == QPSOLVER::qpOASES)
            throw ParameterError("qpOASES specified as QP solver, but not (properly) linked");
    #endif
    #ifndef QPSOLVER_GUROBI
        if (QPsol == QPSOLVER::gurobi)
            throw ParameterError("gurobi specified as QP solver, but not (properly) linked");
    #endif
    #ifndef QPSOLVER_QPALM
    if (QPsol == QPSOLVER::qpalm)
        throw ParameterError("qpalm specified as QP solver, but not (properly) linked");
    #endif

    if (QPsol == QPSOLVER::qpOASES){
        if (QPsol_opts != nullptr && QPsol_opts->sol != QPSOLVER::qpOASES)
            throw ParameterError("qpOASES specified as QP solver, but options given for different QP solver");
        if (hessUpdate == 1 && sparseQP < 2)
            throw ParameterError("qpOASES supports inertia checks for indefinite Hessians only in schur-complement approach (sparseQP == 2)");
    }
    if (QPsol == QPSOLVER::gurobi){
        if (QPsol_opts != nullptr && QPsol_opts->sol != QPSOLVER::gurobi)
            throw ParameterError("gurobi specified as QP solver, but options given for different QP solver");
        if (hessUpdate == 1 || hessUpdate == 4)
            throw ParameterError("gurobi provides needed lagrange multipliers only for convex QPs, given Hessian options not possible");
    }
    if (QPsol == QPSOLVER::qpalm){
        if (QPsol_opts != nullptr && QPsol_opts->sol != QPSOLVER::qpalm)
            throw ParameterError("qpalm specified as QP solver, but options given for different QP solver");
    }

    // If we compute second constraints derivatives then no update or sizing is needed for the first hessian
    if (whichSecondDerv == 2){
        std::cout << "Exact hessian is available, hessUpdate and hessScaling are ignored\n";
    }
    
    //Ensure a positive definite fallback hessian is available if first hessian approximation is not guaranteed to be positive definite
    if ((hessUpdate == 1 || hessUpdate == 4 || hessUpdate == 6) && maxConvQP < 1 && !(fallbackUpdate == 0 || fallbackUpdate == 2 || fallbackUpdate == 5)) 
        throw ParameterError("Positive definite fallback hessian is needed when hessian is not positive definite");


    if (globalization == 1 && hessUpdate == 1 && maxConvQP < 1){
        throw ParameterError("Fallback Hessian QP attempts (maxConvQP > 1) are required when using SR1.");
    }

    if (hessLimMem && hessMemsize == 0) 
        throw ParameterError("hessMemsize must be greater zero for limited memory quasi newton");

    if (hessLimMem && hessMemsize > 200){
        std::cout << "WARNING: Large value of hessMemsize (> 200). Performance may be impeded\n";
    }

    complete_QPsol_opts();
}

void SQPoptions::optionsConsistency(Problemspec *problem){
    if ((autoScaling || convStrategy == 2) && (problem->vblocks == nullptr || problem->n_vblocks < 1))
        throw ParameterError("autoScaling or convexification strategy 2 activated, but no structure information (vblocks) provided problem specification");
    if (sparseQP && problem->nnz < 0)
        throw ParameterError("Sparse mode enabled, but number of jacobian non-zero elements not set");
    
    optionsConsistency();
}


QPSOLVER_options::QPSOLVER_options(QPSOLVER SOL): sol(SOL){
    eps = 1e-16;
    inf = std::numeric_limits<double>::infinity();
    maxTimeQP = 10.;
    maxItQP = std::numeric_limits<int>::max();
}

void SQPoptions::complete_QPsol_opts(){
    //Create default options if no options have been passed
    if (QPsol_opts == nullptr){
        if (QPsol == QPSOLVER::qpOASES) default_QPsol_opts = std::make_unique<qpOASES_options>();
        else if (QPsol == QPSOLVER::gurobi) default_QPsol_opts = std::make_unique<gurobi_options>();
        else if (QPsol == QPSOLVER::qpalm) default_QPsol_opts = std::make_unique<qpalm_options>();
        else throw ParameterError("No valid option for QP solver");

        QPsol_opts = default_QPsol_opts.get();
    }
    //Some values can also be set directly in the options, copy them over default QP solver options.
    if (QPsol_opts->eps == 1e-16) QPsol_opts->eps = eps;
    if (QPsol_opts->inf == std::numeric_limits<double>::infinity()) QPsol_opts->inf = inf;  
    if (QPsol_opts->maxTimeQP == 10.) QPsol_opts->maxTimeQP = maxTimeQP;
    if (QPsol_opts->maxItQP == std::numeric_limits<int>::max()) QPsol_opts->maxItQP = maxItQP;
    return;
}

qpOASES_options::qpOASES_options(): QPSOLVER_options(QPSOLVER::qpOASES){
    printLevel = 0;
    terminationTolerance = 5.0e6*2.221e-16;
}

gurobi_options::gurobi_options(): QPSOLVER_options(QPSOLVER::gurobi){
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

qpalm_options::qpalm_options(): QPSOLVER_options(QPSOLVER::qpalm){}


} // namespace blockSQP
