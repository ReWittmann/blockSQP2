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
#include "blocksqp_problemspec.hpp"
#include <string>

namespace blockSQP
{

/**
 * \brief Contains algorithmic options and parameters for SQPmethod.
 * \author Dennis Janka
 * \date 2012-2015
 */


 enum class QPSOLVER{
    unset = 0,
    qpOASES,
    gurobi,
    qpalm
 };

class QPSOLVER_options;

/*
class SQPoptions{

    public:
        int printLevel;                     ///< information about the current iteration
        int printColor;                     ///< use colored terminal output
        int debugLevel;                     ///< amount of debug information that is printed during every iteration
        double eps;                         ///< values smaller than this are regarded as numerically zero
        double inf;                         ///< values larger than this are regarded as numerically infinity
        double opttol;                      ///< optimality tolerance
        double nlinfeastol;                 ///< nonlinear feasibility tolerance
        bool allow_premature_termination;   ///< Terminate with partial success if linesearch fails but we are feasible and KKT error is low enough
        int max_extra_steps;                ///< Allow additional steps after (partial) termination criterion has been reached to further improve accuracy

        int sparseQP;                       ///< which qpOASES variant is used (dense/sparse/Schur)
        int globalization;                  ///< Globalization strategy
        int restoreFeas;                    ///< Use feasibility restoration phase
        double restZeta;
        double restRho;
        int maxLineSearch;                  ///< Maximum number of steps in line search
        int maxConsecReducedSteps;          ///< Maximum number of consecutive reduced steps
        int maxConsecSkippedUpdates;        ///< Maximum number of consecutive skipped updates
        int blockHess;                      ///< Blockwise Hessian approximation?
        int hessScaling;                    ///< Scaling strategy for Hessian approximation
        int fallbackScaling;                ///< If indefinite update is used, the type of fallback strategy
        double iniHessDiag;                 ///< Initial Hessian guess: diagonal matrix diag(iniHessDiag)

        double colEps;                      ///< epsilon for COL scaling strategy
        double olEps;                       ///< epsilon for first sizing in COL scaling strategy (OL scaling)
        double colTau1;                     ///< tau1 for COL scaling strategy
        double colTau2;                     ///< tau2 for COL scaling strategy
        int hessDamp;                       ///< activate Powell damping for BFGS
        double minDampQuot;                 ///< Minimum allowed value of quotient in damping strategy
        double hessDampFac;                 ///< damping factor for BFGS Powell modification
        int hessUpdate;                     ///< Type of Hessian approximation
        int fallbackUpdate;                 ///< If indefinite update is used, the type of fallback strategy
        bool indef_local_only;              ///< If set to true, only use positive definite fallback update until we are "close" to a local optimum (vars->tol <= 1e-4, vars->cNormS <= 1e-4, it >= 10)
        int max_local_lenience;             ///< If filter line search fails close to a solution (e.g. through rounding errors), allow ignoring the acceptance criteria a limited number of times
        double tau_H;                       ///< If ||d_{SR1 + kappa*I}|| <= tau_H*||d_{BFGS}||, then d_{BFGS} is used instead
        double convKappa0;                  ///< Initial identity scaling factors for corresponding convexification strategy
        double convKappaMax;                ///< Maximum ''   ''   ''

        int autoScaling;                    ///< activate automatic scaling heuristic for variables

        

        //Conditioning tolerances for SR1 update. Update is skipped if denominator goes below a bound determined by this
        double SR1_abstol;
        double SR1_reltol;

        bool hessLimMem;                    ///< Full or limited memory
        int hessMemsize;                    ///< Memory size for L-BFGS updates
        int whichSecondDerv;                ///< For which block should second derivatives be provided by the user
        bool skipFirstGlobalization;        ///< If set to true, no globalization strategy in first iteration is applied
        int convStrategy;                   ///< Convexification strategy
        int maxConvQP;                      ///< How many additional QPs may be solved for convexification per iteration?
        double hess_regularizationFactor;   ///< The identity matrix times this regulrization factor is added to supposedly convex hessian approximations

        int max_bound_refines;              ///< Options for condensed QPs. Up to how many additional QPs with added violated dependent variable should be solved
        int max_correction_steps;           ///< How many additional QPs with bound correction added to dependent variables should be solved
        double dep_bound_tolerance;         ///< Maximum dependent variable bound violation, before they are added to the QP

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

        QPSOLVER QPsol;                     ///< which linked QP solver (qpOASES, gurobi, ...) should be used
        QPSOLVER_options *QPsol_opts;       ///< options to be passed to the specific qp solver. The generic options eps, inf, maxTimeQP and maxItQP, maxTimeQP will automatically be added to this. 
        int maxItQP;                        ///< Maximum number of QP iterations per SQP iteration
        double maxTimeQP;                   ///< Maximum number of time in seconds per QP solve per SQP iteration

        bool new_step_heuristic;
        
    private:
        QPSOLVER_options *default_QPsol_opts;

    public:
        SQPoptions();
        ~SQPoptions();
        /// Some options cannot be used together. Throw ParameterError if hard inconsistencies are detected. Print warnings for soft inconsistencies.
        void optionsConsistency();
        /// Checks for options that are incompatible with the given problem specification, calls optionsConsistency()
        void optionsConsistency(Problemspec *problem);
        /// Create default QPsol_opts if none is given, then copy relevant options into QPsol_opts. Called in optionsConsistency().
        void complete_QPsol_opts();
};
*/

class SQPoptions{
    /*
     * Variables
     */
    public:
        int printLevel = 2;                     ///< information about the current iteration
        int printColor = 1;                     ///< use colored terminal output
        int debugLevel = 0;                     ///< amount of debug information that is printed during every iteration
        double eps = 1.0e-16;                   ///< values smaller than this are regarded as numerically zero
        double inf = std::numeric_limits<double>::infinity(); ///< values larger than this are regarded as numerically infinity
        double opttol = 1.0e-6;                 ///< optimality tolerance
        double nlinfeastol = 1.0e-6;            ///< nonlinear feasibility tolerance
        bool allow_premature_termination = false; ///< Terminate with partial success if linesearch fails but we are feasible and KKT error is low enough
        int max_extra_steps = 0;                ///< Maximum number of additional steps after (partial) termination criterion has been reached to further improve accuracy

        /* Algorithmic options */
        int sparseQP = 2;                       ///< which qpOASES variant is used (dense = 0/sparse = 1/Schur = 2)
        int globalization = 1;                  ///< Globalization strategy, 0 = off, 1 = filter line search
        bool restoreFeas = true;                ///< Use feasibility restoration phase
        double restRho = 1.0;                 ///< Factors in restoration objective: f_rest = rho*||s||_2^2 + zeta*||xi_ref - xi||_2^2
        double restZeta = 1.0e-6;
        int maxLineSearch = 10;                 ///< Maximum number of steps in line search
        int maxConsecReducedSteps = 12;         ///< Maximum number of consecutive reduced steps
        int maxConsecSkippedUpdates = 100;      ///< Maximum number of consecutive skipped updates
        int blockHess = 1;                      ///< Blockwise Hessian approximation?
        int hessScaling = 2;                    ///< Scaling strategy for Hessian approximation (off = 0, 1 = Shanno-Phua, 2 = Oren-Luenberger, 3 = geometric mean of 1 and 2, 4 = centered Oren-Luenberger)
        int fallbackScaling = 4;                ///< If indefinite update is used, the type of fallback strategy, ''   ''   ''
        double iniHessDiag = 1.0;               ///< Initial Hessian guess: diagonal matrix diag(iniHessDiag)

        double colEps = 0.1;                    ///< epsilon for COL scaling strategy
        double olEps = 1.0e-4;                  ///< epsilon for first sizing in COL scaling strategy (OL scaling)
        double colTau1 = 0.5;                   ///< tau1 for COL scaling strategy
        double colTau2 = 1.0e4;                 ///< tau2 for COL scaling strategy
        double minDampQuot = 1e-12;             ///< Minimum allowed value of quotient in damping strategy
        double hessDampFac = 1./3.;             ///< damping factor for BFGS Powell modification
        int hessUpdate = 1;                     ///< Type of Hessian approximation (Identity = 0, SR1 = 1, damped BFGS = 2, None = 3, finite Diff = 4, posdef. given (e.g. Gauss-Newton) = 5, BFGS = 6)
        int fallbackUpdate = 2;                 ///< If indefinite update is used, the type of fallback strategy, ''   ''   ''
        bool indef_local_only = false;          ///< If set to true, only use positive definite fallback update until we are "close" to a local optimum
        int max_local_lenience = 2;             ///< Allow ignoring the acceptance criteria a limited number of times
        double tau_H = 2./3.;                   ///< If ||d_{SR1 + kappa*I}|| <= tau_H*||d_{BFGS}||, then d_{BFGS} is used instead
        double convKappa0 = 1./16.;             ///< Initial identity scaling factors for convexification strategy
        double convKappaMax = 8.;               ///< Maximum " "

        int autoScaling = false;                ///< activate automatic scaling heuristic for variables

        //Conditioning tolerances for SR1 update. Update is skipped if denominator goes below a bound determined by this
        double SR1_abstol = 1e-18;
        double SR1_reltol = 1e-5;
        
        bool hessLimMem = true;                 ///< Full or limited memory
        int hessMemsize = 20;                   ///< Memory size for L-BFGS updates
        int whichSecondDerv = 0;                ///< For which block should second derivatives be provided by the user (None = 0, last block = 1, all blocks = 2)
        bool skipFirstGlobalization = false;    ///< If set to true, no globalization strategy in first iteration is applied
        int convStrategy = 1;                   ///< Convexification strategy (Convex combination between Hessian and fallback = 0, added scaled identities = 1, added scaled identities to free variable indices == added scaled identities to condensed Hessian)
        int maxConvQP = 4;                      ///< How many additional QPs may be solved for convexification per iteration?
        double hess_regularizationFactor = 0.0; ///< Regularization factor added to convex Hessian approximations

        /* Options for qpOASES with condensing*/
        int max_bound_refines = 3;              ///< Options for condensed QPs
        int max_correction_steps = 5;           ///< How many additional QPs with bound correction added to dependent variables should be solved
        double dep_bound_tolerance = 1e-7;      ///< Maximum dependent variable bound violation before adding to QP

        /* Filter line search parameters */
        int maxSOCiter = 3;                     ///< Maximum number of SOC line search iterations
        double gammaTheta = 1.0e-5;             ///< see IPOPT paper
        double gammaF = 1.0e-5;                 ///< see IPOPT paper   
        double kappaSOC = 0.99;                 ///< see IPOPT paper
        double kappaF = 0.8;                    ///< see IPOPT paper
        double thetaMax = 1.0e7;                ///< see IPOPT paper
        double thetaMin = 1.0e-5;               ///< see IPOPT paper
        double delta = 1.0;                     ///< see IPOPT paper
        double sTheta = 1.1;                    ///< see IPOPT paper
        double sF = 2.3;                        ///< see IPOPT paper
        double eta = 1.0e-4;                    ///< see IPOPT paper

        /* QP solver options */
        QPSOLVER QPsol = QPSOLVER::unset;       ///< which linked QP solver (qpOASES, gurobi, ...) should be used
        QPSOLVER_options *QPsol_opts = nullptr; ///< options to be passed to the specific qp solver
        int maxItQP = 5000;                     ///< Maximum number of QP iterations per SQP iteration
        double maxTimeQP = 10000.0;             ///< Maximum number of seconds per QP solve per SQP iteration

        bool loud_SQPresult = true;
    private:
        QPSOLVER_options *default_QPsol_opts = nullptr;

    /*
     * Methods
     */
    public:
        SQPoptions();
        ~SQPoptions();
        void optionsConsistency();
        void optionsConsistency(Problemspec *problem);
        void complete_QPsol_opts();
        void reset();
};


class QPSOLVER_options{
    public:
    const QPSOLVER sol;
    double eps;
    double inf;
    double maxTimeQP;
    int maxItQP;

    protected:
    QPSOLVER_options(QPSOLVER SOL);
    //friend SQPoptions;
    //friend class QPsolver;
};

class qpOASES_options : public QPSOLVER_options{
    public:
    int printLevel;                 ///< print level of qpOASES sub-qp solver, 0 = PL_NONE, 1 = PL_LOW, 2 = PL_MEDIUM, 3 = PL_HIGH
    double terminationTolerance;    ///< Termination tolerance of qp-subproblem solver qpOASES
    qpOASES_options();
};

class gurobi_options : public QPSOLVER_options{
    public:
    //See gurobi documentation
    int Method;
    int NumericFocus;
    int OutputFlag;
    int Presolve;
    int Aggregate;
    int BarHomogeneous;
    double OptimalityTol;
    double FeasibilityTol;
    double PSDTol;

    //double regularization_factor; ///< Scaling factor for identity added to hessian when invoking gurobi

    gurobi_options();
};

class qpalm_options : public QPSOLVER_options{
    public:
    qpalm_options();
};


} // namespace blockSQP

#endif
