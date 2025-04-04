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
    // General options
    /*
    int print_level = 2;                     ///< information about the current iteration
    int result_print_color = 1;                     ///< use colored terminal output
    int debug_level = 0;                     ///< amount of debug information that is printed during every iteration
    double eps = 1.0e-16;                   ///< values smaller than this are regarded as numerically zero
    double inf = std::numeric_limits<double>::infinity(); ///< values larger than this are regarded as numerically infinity
    double optimality_tol = 1.0e-6;                 ///< optimality tolerance
    double feasibility_tol = 1.0e-6;            ///< nonlinear feasibility tolerance
    
    //enable
    bool enable_premature_termination = false; ///< Terminate with partial success if linesearch fails but we are feasible and KKT error is low enough
    int max_extra_steps = 0;                ///< Maximum number of additional steps after (partial) termination criterion has been reached to further improve accuracy

    */

    //TODO options separation
    // Basic options for general usage
    
    //Constants 
    double eps = 1.0e-16;                   ///< values smaller than this are regarded as numerically zero
    double inf = std::numeric_limits<double>::infinity(); ///< values larger than this are regarded as numerically infinity

    //Output
    int print_level = 2;                     // information about the current iteration
    int result_print_color = 2;              // Output of the SQP result; 0: None, 1: No color, 2: color
    int debug_level = 0;                     // amount of debug information that is output to a file in each iteration
    
    //Termination criteria
    double optimality_tol = 1.0e-6;
    double feasibility_tol = 1.0e-6;
    bool enable_premature_termination = false;
    int max_extra_steps = 0;


    bool sparse_mode = true;

    
    bool enable_feasibility_restoration = true;
    double restoration_rho = 1.0;
    double restoration_zeta = 1.0e-6;
    

    bool enable_linesearch = true;
    int max_linesearch_steps = 10;
    int max_consec_reduced_steps = 8;
    int max_consec_skipped_updates = 100;

    bool skip_first_linesearch = false;

    bool limited_memory = true;
    int memory_size = 20;

    int block_hess = 1;
    int initial_hess_scale = 1.0;
    int sizing_strategy = 2;
    int fallback_sizing_strategy = 4;
    double COL_eps = 0.1;
    double COL_tau_1 = 0.5;
    double COL_tau_2 = 1.0e4;
    double OL_eps = 1.0e-4;

    int exact_hess_usage = 0;                       //0: No exact Hessian, 1: Exact last Hessian block, 2: Exact complete Hessian
    int hess_approximation = 1;          
    int fallback_approximation = 2;
    double BFGS_damping_factor = 1./3.;
    double min_damping_quotient = 1e-12;

    double hess_regularization_factor = 0.0;    //Enable further regularization of pos.def. fallback Hessians by adding a scaled identity. Beneficial for some QP solvers
    
    int conv_strategy = 1;
    int max_conv_QPs = 4;

    double conv_tau_H = 2./3.;                         //See paper
    double conv_kappa_0 = 1./16.;
    double conv_kappa_max = 2.;

    int automatic_scaling = 0;                  //Select scaling heuristic, 0: Off, 1: free-dep balance 2: TODO 

    // Advanced options for numerical experiments
    bool indef_local_only = false;
    int max_filter_overrides = 2;


    // Filter line search parameters
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


    double SR1_abstol = 1e-18;
    double SR1_reltol = 1e-5;


    QPsolvers qpsol = QPsolvers::unset;       ///< which linked QP solver (qpOASES, gurobi, ...) should be used
    QPsolver_options *qpsol_options = nullptr; ///< options to be passed to the specific qp solver
    int max_QP_iter = 5000;                     ///< Maximum number of QP iterations per SQP iteration
    double max_QP_seconds = 10000.0;                ///< Maximum number of seconds per QP solve per SQP iteration


    //SQPmethod subclass options, outsource into SQPoptions subclass if they become too many
    int max_bound_refines = 3;              ///< Options for condensed QPs
    int max_correction_steps = 5;           ///< How many additional QPs with bound correction added to dependent variables should be solved
    double dep_bound_tolerance = 1e-7;      ///< Maximum dependent variable bound violation before adding to QP

    
    


    /*
    bool sparse_mode = 2;                       ///< which qpOASES variant is used (dense = 0/sparse = 1/Schur = 2)
    
    bool enable_linesearch = true;              ///< Globalization strategy, 0 = off, 1 = filter line search
    bool enable_feasibility_restoration = true;                ///< Use feasibility restoration phase
    double restoration_rho = 1.0;                   ///< Factors in restoration objective: f_rest = rho*||s||_2^2 + zeta*||xi_ref - xi||_2^2
    double restoration_zeta = 1.0e-6;

    int max_linesearch_steps = 10;                 ///< Maximum number of steps in line search
    int max_consec_reduced_steps = 12;         ///< Maximum number of consecutive reduced steps
    int max_consec_skipped_updates = 100;      ///< Maximum number of consecutive skipped updates

    int block_hess = 1;                      ///< Blockwise Hessian approximation?
    int sizing_strategy = 2;                    ///< Scaling strategy for Hessian approximation (off = 0, 1 = Shanno-Phua, 2 = Oren-Luenberger, 3 = geometric mean of 1 and 2, 4 = centered Oren-Luenberger)
    int fallback_sizing_strategy = 4;                ///< If indefinite update is used, the type of fallback strategy, ''   ''   ''
    double initial_hess_scale = 1.0;               ///< Initial Hessian guess: diagonal matrix diag(initial_hess_scale)

    double OL_eps = 1.0e-4;                  ///< epsilon for first sizing in COL scaling strategy (OL scaling)
    double COL_eps = 0.1;                    ///< epsilon for COL scaling strategy
    double COL_tau_1 = 0.5;                   ///< tau1 for COL scaling strategy
    double COL_tau_2 = 1.0e4;                 ///< tau2 for COL scaling strategy


    double min_damping_quotient = 1e-12;             ///< Minimum allowed value of quotient in damping strategy
    double BFGS_damping_factor = 1./3.;             ///< damping factor for BFGS Powell modification
    int hess_approximation = 1;                     ///< Type of Hessian approximation (Identity = 0, SR1 = 1, damped BFGS = 2, None = 3, finite Diff = 4, posdef. given (e.g. Gauss-Newton) = 5, BFGS = 6)
    int fallback_approximation = 2;                 ///< If indefinite update is used, the type of fallback strategy, ''   ''   ''
    
    bool indef_local_only = false;          ///< If set to true, only use positive definite fallback update until we are "close" to a local optimum
    int max_filter_overrides = 2;             ///< Allow ignoring the acceptance criteria a limited number of times
    double conv_tau_H = 2./3.;                   ///< If ||d_{SR1 + kappa*I}|| <= conv_tau_H*||d_{BFGS}||, then d_{BFGS} is used instead
    double conv_kappa_0 = 1./16.;             ///< Initial identity scaling factors for convexification strategy
    double conv_kappa_max = 2.0;               ///< Maximum " "

    int automatic_scaling = 1;                    ///< activate automatic scaling heuristic for variables

    //Conditioning tolerances for SR1 update. Update is skipped if denominator goes below a bound determined by this
    double SR1_abstol = 1e-18;
    double SR1_reltol = 1e-5;
    
    bool limited_memory = true;                 ///< Full or limited memory
    int memory_size = 20;                   ///< Memory size for L-BFGS updates
    int exact_hess_usage = 0;                ///< For which block should second derivatives be provided by the user (None = 0, last block = 1, all blocks = 2)
    bool skip_first_linesearch = false;    ///< If set to true, no enable_linesearch strategy in first iteration is applied
    int conv_strategy = 1;                   ///< Convexification strategy (Convex combination between Hessian and fallback = 0, added scaled identities = 1, added scaled identities to free variable indices == added scaled identities to condensed Hessian)
    int max_conv_QPs = 4;                      ///< How many additional QPs may be solved for convexification per iteration?
    double hess_regularization_factor = 0.0; ///< Regularization factor added to convex Hessian approximations

    //Filter line search parameters
    int max_SOC = 3;                     ///< Maximum number of SOC line search iterations
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
    QPsolvers qpsol = QPsolvers::unset;       ///< which linked QP solver (qpOASES, gurobi, ...) should be used
    QPsolver_options *qpsol_options = nullptr;           ///< options to be passed to the specific qp solver
    
    int max_QP_iter = 5000;                     ///< Maximum number of QP iterations per SQP iteration
    double max_QP_seconds = 10000.0;             ///< Maximum number of seconds per QP solve per SQP iteration

    //int result_print_color = 2;


    //Subclass options
    int max_bound_refines = 3;              ///< Options for condensed QPs
    int max_correction_steps = 5;           ///< How many additional QPs with bound correction added to dependent variables should be solved
    double dep_bound_tolerance = 1e-7;      ///< Maximum dependent variable bound violation before adding to QP
    */

    private:
    std::unique_ptr<QPsolver_options> default_qpsol_options;

    public:
    SQPoptions();
    ~SQPoptions();
    //Checks for inconsistent options. Throw ParameterError if inconsistent options are detected. Calls complete_QP_options.
    void optionsConsistency();
    //Checks for options inconsistent with the given problem specification, then calls optionsconsistency. 
    void optionsConsistency(Problemspec *problem);
    //Set default QP solver options and copy over some options from the SQP options
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
    int sparsityLevel;                   ///< Method used by qpOASES: 0 - qpOASES::SQProblem (dense), 1 - qpOASES::SQProblem (sparse), 2 - schur

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
