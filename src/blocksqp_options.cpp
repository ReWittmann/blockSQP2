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
SQPoptions::SQPoptions()
{
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

    which_QPsolver = QPSOLVER::ANY;

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
    hessDamp = 1;

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
    maxLineSearch = 20;

    // if step has to be reduced in too many consecutive iterations, feasibility restoration phase is invoked
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
    kappaF = 0.999;
    thetaMax = 1.0e7;       // reject steps if constr viol. is larger than thetaMax
    thetaMin = 1.0e-5;      // if constr viol. is smaller than thetaMin require Armijo cond. for obj.
    delta = 1.0;
    sTheta = 1.1;
    sF = 2.3;
    eta = 1.0e-4;

    //cNormFilterTol = 1e-8;
    // Inertia correction for filter line search and indefinite Hessians
    kappaMinus = 0.333;
    kappaPlus = 8.0;
    kappaPlusMax = 100.0;
    deltaH0 = 1.0e-4;

    //For SCQPmethod subclasses
    dep_bound_tolerance = 1e-7;

    //Options for linked QP solvers
    #ifdef QPSOLVER_QPOASES
        qpOASES_printLevel = 0;
        qpOASES_terminationTolerance = 5.0e6*2.221e-16;
    #endif

    #ifdef QPSOLVER_GUROBI
        gurobi_Method = 1;
        gurobi_NumericFocus = 3;
        gurobi_OutputFlag = 0;
        gurobi_Presolve = -1;
        gurobi_Aggregate = 1;
        gurobi_OptimalityTol = 1e-9;
        gurobi_FeasibilityTol = 1e-9;
        gurobi_BarHomogeneous = 0;
        gurobi_PSDTol = 1e-6;

        //gurobi_solver_regularization_factor = 1e-8;
    #endif

}


/**
 * Some options cannot be set together, resolve here
 */
void SQPoptions::optionsConsistency()
{
    // If we compute second constraints derivatives then no update or sizing is needed for the first hessian
    if (whichSecondDerv == 2){
        std::cout << "Exact hessian is available, overwrite hessUpdate and hessScaling\n";
        hessUpdate = 6;
        hessScaling = 0;
        blockHess = 1;
    }

    // If we don't use limited memory BFGS we need to store only one vector.
    if (!hessLimMem)
        hessMemsize = std::numeric_limits<int>::max();

    //Ensure a positive definite fallback hessian is available if first hessian approximation is not guaranteed to be positive definite
    if ((hessUpdate == 1 || hessUpdate == 4 || hessUpdate == 6) && maxConvQP < 1 && !(fallbackUpdate == 0 || fallbackUpdate == 2 || fallbackUpdate == 5))
        throw ParameterError("Error, positive definite fallback hessian is needed when hessian is not positive definite");

    /*
    if (convStrategy == 1 && maxConvQP > 2){
        std::cout << "Only one indefinite hessian with added scaled identity can be tried, setting maxConvQP to 2";
        maxConvQP = 2;
    }*/

    //If no convexified hessians are tried, set convStrategy to 0 to avoid some unnecessary computation
    if (maxConvQP == 1){
        std::cout << "maxConvQP = 1, no Hessian regularization used, set convStrategy to default 0\n";
        convStrategy = 0;
    }

    if (globalization == 1 && hessUpdate == 1 && (maxConvQP < 1)){
        std::cout << "Fallback update is needed for SR1, setting maxConvQP to 1\n";
        maxConvQP = 1;
    }

}

} // namespace blockSQP
