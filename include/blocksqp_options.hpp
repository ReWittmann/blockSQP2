/*
 * blockSQP -- Sequential quadratic programming for problems with
 *             block-diagonal Hessian matrix.
 * Copyright (C) 2012-2015 by Dennis Janka <dennis.janka@iwr.uni-heidelberg.de>
 *
 * Licensed under the zlib license. See LICENSE for more details.
 */

/**
 * \file blocksqp_options.hpp
 * \author Dennis Janka
 * \date 2012-2015
 *
 *  Declaration of SQPoptions class that holds all algorithmic options.
 */

#ifndef BLOCKSQP_OPTIONS_HPP
#define BLOCKSQP_OPTIONS_HPP

#include "blocksqp_defs.hpp"
#include <string>

namespace blockSQP
{

/**
 * \brief Contains algorithmic options and parameters for SQPmethod.
 * \author Dennis Janka
 * \date 2012-2015
 */

 enum class QPSOLVER{
    ANY,
    QPOASES,
    GUROBI
 };


class SQPoptions
{
    /*
     * Variables
     */
    public:
        int printLevel;                     ///< information about the current iteration
        int printColor;                     ///< use colored terminal output
        int debugLevel;                     ///< amount of debug information that is printed during every iteration
        double eps;                         ///< values smaller than this are regarded as numerically zero
        double inf;                         ///< values larger than this are regarded as numerically infinity
        double opttol;                      ///< optimality tolerance
        double nlinfeastol;                 ///< nonlinear feasibility tolerance
        QPSOLVER which_QPsolver;            ///< which linked QP solver (qpOASES, gurobi, ...) should be used

        /* Algorithmic options */
        int sparseQP;                       ///< which qpOASES variant is used (dense/sparse/Schur)
        int globalization;                  ///< Globalization strategy
        int restoreFeas;                    ///< Use feasibility restoration phase
        double restZeta;
        double restRho;
        int maxLineSearch;                  ///< Maximum number of steps in line search
        int maxConsecReducedSteps;          ///< Maximum number of consecutive reduced steps
        int maxConsecSkippedUpdates;        ///< Maximum number of consecutive skipped updates
        int maxItQP;                        ///< Maximum number of QP iterations per SQP iteration
        int blockHess;                      ///< Blockwise Hessian approximation?
        int hessScaling;                    ///< Scaling strategy for Hessian approximation
        int fallbackScaling;                ///< If indefinite update is used, the type of fallback strategy
        double maxTimeQP;                   ///< Maximum number of time in seconds per QP solve per SQP iteration
        double iniHessDiag;                 ///< Initial Hessian guess: diagonal matrix diag(iniHessDiag)
        //double HessDiag2;
        //bool size_hessian_first_step;       ///< Size hessian to get a better initial stepsize, currently only for (almost) feasible starting points

        double colEps;                      ///< epsilon for COL scaling strategy
        double olEps;                       ///< epsilon for first sizing in COL scaling strategy (OL scaling)
        double colTau1;                     ///< tau1 for COL scaling strategy
        double colTau2;                     ///< tau2 for COL scaling strategy
        int hessDamp;                       ///< activate Powell damping for BFGS
        double minDampQuot;               ///< Minimum allowed value of quotient in damping strategy
        double hessDampFac;                 ///< damping factor for BFGS Powell modification
        int hessUpdate;                     ///< Type of Hessian approximation
        int fallbackUpdate;                 ///< If indefinite update is used, the type of fallback strategy
        bool indef_local_only;              ///< If set to true, only use positive definite fallback update until we are "close" to a local optimum (vars->tol <= 1e-4, vars->cNormS <= 1e-4, it >= 10)
        int max_local_lenience;             ///< If filter line search fails close to a solution (e.g. through rounding errors), allow ignoring the acceptance criteria a limited number of times
        double tau_H;                       ///< If ||d_{SR1 + kappa*I}|| <= tau_H*||d_{BFGS}||, then d_{BFGS} is used instead
        double convKappa0;

        //Conditioning tolerances for SR1 update. Update is skipped if denominator goes below a bound determined by this
        double SR1_abstol;
        double SR1_reltol;

        int hessLimMem;                     ///< Full or limited memory
        int hessMemsize;                    ///< Memory size for L-BFGS updates
        int whichSecondDerv;                ///< For which block should second derivatives be provided by the user
        bool skipFirstGlobalization;        ///< If set to true, no globalization strategy in first iteration is applied
        int convStrategy;                   ///< Convexification strategy
        int maxConvQP;                      ///< How many additional QPs may be solved for convexification per iteration?
        double hess_regularizationFactor;   ///< The identity matrix times this regulrization factor is added to supposedly convex hessian approximations


        /* Options for qpOASES with condensing*/
        int max_bound_refines;              ///< Options for condensed QPs. Up to how many additional QPs with added violated dependent variable should be solved
        int max_correction_steps;           ///< How many additional QPs with bound correction added to dependent variables should be solved
        double dep_bound_tolerance;          ///< Maximum dependent variable bound violation, before they are added to the QP

        /* Filter line search parameters */
        int maxSOCiter;                     ///< Maximum number of SOC line search iterations
        double gammaTheta;                  ///< see IPOPT paper
        double gammaF;                      ///< see IPOPT paper
        double kappaSOC;                    ///< see IPOPT paper
        double kappaF;                      ///< see IPOPT paper
        double thetaMax;                    ///< see IPOPT paper
        double thetaMin;                    ///< see IPOPT paper
        double delta;                       ///< see IPOPT paper
        double sTheta;                      ///< see IPOPT paper
        double sF;                          ///< see IPOPT paper
        double kappaMinus;                  ///< see IPOPT paper
        double kappaPlus;                   ///< see IPOPT paper
        double kappaPlusMax;                ///< see IPOPT paper
        double deltaH0;                     ///< see IPOPT paper
        double eta;                         ///< see IPOPT paper

        //double cNormFilterTol;

        //Options for linked QP solvers
        #ifdef QPSOLVER_QPOASES
            int qpOASES_printLevel;                 ///< print level of qpOASES sub-qp solver, 0 = PL_NONE, 1 = PL_LOW, 2 = PL_MEDIUM, 3 = PL_HIGH
            double qpOASES_terminationTolerance;    ///< Termination tolerance of qp-subproblem solver qpOASES
        #endif

        #ifdef QPSOLVER_GUROBI
            //Gurobi options
            int gurobi_Method;
            int gurobi_NumericFocus;
            int gurobi_OutputFlag;
            int gurobi_Presolve;
            int gurobi_Aggregate;
            int gurobi_BarHomogeneous;
            double gurobi_OptimalityTol;
            double gurobi_FeasibilityTol;
            double gurobi_PSDTol;

            //Scaling factor for identity added to hessian in gurobi model
            //double gurobi_solver_regularization_factor; ///< Scaling factor for identity added to hessian when invoking gurobi
        #endif

    /*
     * Methods
     */
    public:
        SQPoptions();
        /// Some options cannot be used together. In this case set defaults
        void optionsConsistency();
};

} // namespace blockSQP

#endif
