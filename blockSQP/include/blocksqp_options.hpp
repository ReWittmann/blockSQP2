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

namespace blockSQP{

/**
 * \brief Contains algorithmic options and parameters for SQPmethod.
 * \author Dennis Janka
 * \date 2012-2015
 */

enum class QPsolvers{
    unset = 0,
    qpOASES,
    gurobi,
    qpalm
};

class QPsolver_options;


class SQPoptions{
    public:    
    //Constants
    double eps = 1.0e-16;                   ///< values smaller than this are regarded as numerically zero
    double inf = std::numeric_limits<double>::infinity(); ///< values larger than this are regarded as numerically infinity

    //Output
    int print_level = 2;                     // information about the current iteration
    int result_print_color = 2;              // Output of the SQP result; 0: None, 1: No color, 2: color
    int debug_level = 0;                     // amount of debug information that is output to a file in each iteration
    
    //Termination criteria
    double opt_tol = 1.0e-6;         
    double feas_tol = 1.0e-6;
    bool enable_premature_termination = false;  // Allow terminating with partial success if opt error is below opt_tol**0.75, we are feasible and the linesearch fails
    int max_extra_steps = 0;                    // Maximum number of steps after tolerances were reached for improved accuracy.

    //Line search heuristics
    int max_filter_overrides = 2;

    //Derivative evaluation
    bool sparse_mode = true;                    // Decide wether dense or sparse problem functions should be used

    //Restoration phase
    bool enable_feasibility_restoration = true; 
    double restoration_rho = 1.0;                     // Restoration objective: Rho * ||s||^2 + zeta * ||xi - xi_ref||^2, s - slack variables, xi_ref - iterate at which restoration was invoked
    double restoration_zeta = 1.0e-6;

    //Full/limited memory quasi newton
    bool limited_memory = true;                       // Enable limited memory quasi newton
    int memory_size = 20;                             // Limited memory size

    //Hessian approximation
    int block_hess = 1;                             //0: Full space updates, 1: partitioned updates, 2: 2 blocks: 1 full space update block, 1 objective Hessian block
    int exact_hess_usage = 0;                       //0: No exact Hessian, 1: Exact last Hessian block, 2: Exact complete Hessian
    int hess_approximation = 1;                     //0: (Scaled) identity, 1: SR1, 2: damped BFGS, 3: None, 4: finite differences, 5: pos. def user provided (Gauss-Newton etc.), 6: undamped BFGS
    int fallback_approximation = 2;                 //As hess_approximation, must be positive definite
    
    double hess_regularization_factor = 0.0;        //Enable further regularization of pos.def. fallback Hessians by adding a scaled identity. Beneficial for some QP solvers
    
    //Hessian sizing
    double initial_hess_scale = 1.0;
    int sizing_strategy = 2;                        // Scaling strategy for Hessian approximation (off = 0, 1 = Shanno-Phua, 2 = Oren-Luenberger, 3 = geometric mean of 1 and 2, 4 = centered Oren-Luenberger)
    int fallback_sizing_strategy = 4;               // ' '                 fallback ' '
    double COL_eps = 0.1;                           // Centered Oren-Luenberger sizing parameters (see Contreras Tapia paper)
    double COL_tau_1 = 0.5;
    double COL_tau_2 = 1.0e4;
    double OL_eps = 1.0e-4;                         // Oren-Luneberger sizing epsilon

    //Quasi newton numerical tolerances
    double BFGS_damping_factor = 1./3.;
    double min_damping_quotient = 1e-12;            //Minimum quotient in damping strategy
    double SR1_abstol = 1e-18;
    double SR1_reltol = 1e-5;

    //Convexification strategy
    int conv_strategy = 1;                      //Convexification strategy, 0: convex combination between Hess and fallback, 1: add scaled identities, 2: add scaled identities to free components, requires providint vblocks to problem
    int max_conv_QPs = 4;                       //Maximum number of convexified QPs in each SQP iteration.
    double conv_tau_H = 2./3.;                  //See paper/manual
    double conv_kappa_0 = 1./16.;
    double conv_kappa_max = 2.;

    //Scaling
    int automatic_scaling = 0;                  //Select scaling heuristic, 0: Off, 1: free-dep balance 2: FUTURE

    //Advanced options for numerical experiments
    bool indef_local_only = false;              //Only use fallback as long as KKT error is "large"


    //Filter line search options
    bool enable_linesearch = true;
    int max_linesearch_steps = 10;
    int max_consec_reduced_steps = 8;                 // Reset Hessian if stepsize was reduced consecutively too often
    int max_consec_skipped_updates = 100;             // Reset Hessian if too many quasi Newton updates were skipped consecutively
    bool skip_first_linesearch = false;

    //Filter line search parameters
    int max_SOC = 3;                        ///< Maximum number of second order correction (SOC) steps
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

    //QP solver options
    QPsolvers qpsol = QPsolvers::unset;             ///< which linked QP solver (qpOASES, gurobi, ...) should be used
    QPsolver_options *qpsol_options = nullptr;      ///< options to be passed to the specific qp solver
    int max_QP_iter = 5000;                         ///< Maximum number of QP iterations per SQP iteration
    double max_QP_seconds = 3600.0;                 ///< Maximum number of seconds per QP solve per SQP iteration


    //SQPmethod subclass options, outsource into SQPoptions subclass if they become too many
    int max_bound_refines = 3;              ///< Options for condensed QPs
    int max_correction_steps = 5;           ///< How many additional QPs with bound correction added to dependent variables should be solved
    double dep_bound_tolerance = 1e-7;      ///< Maximum dependent variable bound violation before adding to QP

    private:
    //Holder if no qpsol_options were provided
    std::unique_ptr<QPsolver_options> default_qpsol_options;

    public:
    SQPoptions();
    ~SQPoptions();
    //Checks for options inconsistent with the given problem specification, then calls optionsConsistency. 
    void optionsConsistency(Problemspec *problem);
    //Checks for inconsistent options. Throw ParameterError if inconsistent options are detected. Calls complete_QP_options.
    void optionsConsistency();
    //Set default QP solver options and copy over some options from the SQP options. Assumes options are consistent. Automatically called by optionsConsistency
    void complete_QP_options();
    void reset();
};


class QPsolver_options{
    public:
    const QPsolvers sol;
    double eps;
    double inf;
    double max_QP_seconds;
    int max_QP_iter;

    protected:
    QPsolver_options(QPsolvers SOL);
    public:
    virtual ~QPsolver_options();
};

class qpOASES_options : public QPsolver_options{
    public:
    int sparsityLevel;                   ///< Method used by qpOASES: -1 (default): Infer from SQPoptions, 0 - qpOASES::SQProblem (dense), 1 - qpOASES::SQProblem (sparse), 2 - schur.

    //See qpOASES documentation
    int printLevel;                 ///< print level of qpOASES sub-qp solver, 0 = PL_NONE, 1 = PL_LOW, 2 = PL_MEDIUM, 3 = PL_HIGH
    double terminationTolerance;    ///< Termination tolerance of qp-subproblem solver qpOASES
    qpOASES_options();
};

class gurobi_options : public QPsolver_options{
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
    //double regularization_factor;         // Scaling factor for identity added to hessian when invoking gurobi

    gurobi_options();
};

class qpalm_options : public QPsolver_options{
    public:
    qpalm_options();
};


} // namespace blockSQP

#endif
