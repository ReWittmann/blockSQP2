/*
 * blockSQP -- Sequential quadratic programming for problems with
 *             block-diagonal Hessian matrix.
 * Copyright (C) 2012-2015 by Dennis Janka <dennis.janka@iwr.uni-heidelberg.de>
 *
 * Licensed under the zlib license. See LICENSE for more details.
 */

/*
 * blockSQP 2 -- Condensing, convexification strategies, scaling heuristics and more
 *               for blockSQP, the nonlinear programming solver by Dennis Janka.
 * Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
 * 
 * Licensed under the zlib license. See LICENSE for more details.
 */
 
 
/**
 * \file blocksqp_iterate.hpp
 * \author Dennis Janka
 * \date 2012-2015
 *
 *  Declaration of SQPiterate class that holds all variables that are
 *  updated during one SQP iteration.
 * 
 * \modifications
 *  \author Reinhold Wittmann
 *  \date 2023-2025
 */

#ifndef BLOCKSQP_ITERATE_HPP
#define BLOCKSQP_ITERATE_HPP

#include "blocksqp_defs.hpp"
#include "blocksqp_matrix.hpp"
#include "blocksqp_problemspec.hpp"
#include "blocksqp_options.hpp"
#include <memory>
#include <thread>
#include <future>

namespace blockSQP{


class SQPiterate{
    public:
        //Initialized by initIterate, 
        double obj;                                   // objective value
        double cNorm;                                 // constraint violation
        double cNormS;                                // scaled constraint violation
        double gradNorm;                              // norm of Lagrangian gradient
        double lambdaStepNorm;                        // norm of step in dual variables
        double tol;                                   // current optimality tolerance
        
        double alpha;                                 // stepsize for line search
        int nSOCS;                                    // number of second-order correction steps
        int reducedStepCount;                         // count number of consecutive reduced steps,
        int steptype;                                 // -2: Filter-overwriting step -1: Step heuristic, 0: Linesearch step, 1: Linesearch step with identity Hessian, 2: Feasibility restoration heuristic step, 3: Feasibility restoration step
        int n_id_hess;                                // Number of condecutive uses of identity hessian as fallback
        
        //Current primal-dual iterate. 
        Matrix xi;                                    // variable vector
        Matrix lambda;                                // dual variables
        Matrix constr;                                // constraint vector
        Matrix gradObj;                               // gradient of objective
        Matrix gradLagrange;                          // gradient of Lagrangian
        
        //Constraint jacobian
        Matrix constrJac;                             // full constraint Jacobian (not used in sparse mode)
        Sparse_Matrix sparse_constrJac;               // sparse constraint Jacobian (not used in dense mode)
        
        //Hessian(s), including layout
        int nBlocks;                                   ///< number of diagonal blocks in Hessian
        std::unique_ptr<int[]> blockIdx;               ///< indices in the variable vector that correspond to diagonal blocks (nBlocks+1)
        
        bool last_block_exact;
        bool last_fallback_exact;
        
        SymMatrix *hess;                               ///< [blockwise] pointer to current Hessian (-approximation) of the Lagrangian
        std::unique_ptr<SymMatrix[]> hess1;            ///< [blockwise] first Hessian approximation
        std::unique_ptr<SymMatrix[]> hess2;            ///< [blockwise] second Hessian approximation (convex)
        std::unique_ptr<SymMatrix[]> hess_conv;        ///< [blockwise] space to store alternative hessians, such as convexified indefinite Hessians

        //Current and past iteration data
        int dg_nsave;                                 /// number of saved delta-gamma pairs and their scalar products. These are required for limited memory updates and the scaling heuristic
        int dg_pos;                                   /// position of the current iterate within _____Mat
        Matrix deltaMat;                              ///< last dg_nsave primal steps
        Matrix gammaMat;                              ///< Lagrangian gradient differences for last dg_nsave steps
        Matrix deltaXi;                               ///< alias for current step (first stores full step from QP, may get downscaled in linesearch)
        Matrix gamma;                                 ///< alias for current Lagrangian gradient

        //Precalculated scalar products for COL sizing. In full memory quasi newton, they are updated at the end of each SQP iteration. In limited memory, they are calculated when applying the update
        Matrix deltaNormSqMat;                        /// last dg_nsave >= 2 squared step norms
        Matrix deltaGammaMat;                         /// last dg_nsave >= 2 delta-gamma scalar products

        //Damping may change gamma, old scalar product are required for COL sizing
        Matrix deltaNormSqOld;                        // TODO remove
        Matrix deltaOld;                              // To restore deltaNormSqOld in full memory after rescaling. TODO recompute from deltaMat instead
        Matrix deltaGammaOld;
        Matrix deltaGammaOldFallback;

        //Additional step data used by the methods
        Matrix AdeltaXi;                              ///< product of constraint Jacobian with deltaXi (from SOC for SOC iterations after the first one), calculated in secondOrderCorrection method as needed
        Matrix lambdaQP;                              ///< dual variables of QP
        
        //Convex Hessian step calculated by convexification strategy
        Matrix deltaXi_conv;
        Matrix lambdaQP_conv;

        Matrix trialXi;                               ///< new trial iterate (for line search)
        Matrix trialLambda;                           ///< Used temporarily if previous Lambda is still required, e.g. to calculate lambdaStepNorm
        Matrix trialConstr;                           ///< constraints evaluated at trial point. Calculated in linesearch and used also in SOC      
        
        //Step bounds used in the QP, calculated in solveQP right before QPsolver/Condenser is invoked
        Matrix delta_lb_var;                          ///< lower bounds for current (SOC) step
        Matrix delta_ub_var;                          ///< upper bounds for current (SOC) step
        Matrix delta_lb_con;                          ///< lower bounds for linearized (SOC) constraints
        Matrix delta_ub_con;                          ///< upper bounds for linearized (SOC) constraints

        //Miscellaneous counters
        std::unique_ptr<int[]> nquasi;                 /// number of quasi-newton updates for each block since last Hessian(block) reset
        std::unique_ptr<int[]> noUpdateCounter;        /// count skipped updates for each block, reset if options->max_consec_skipped_updates is exceeded
        int nRestIt;                                   /// Number of current restoration iterate
        int remaining_filter_overrides;                


        //Flags and associated values
        bool conv_qp_only;                             ///< If true, only convex sub-QPs are used to generate steps
        bool conv_qp_solved;                           // Indicated that convex QP was solved
        bool hess2_updated;                            // Used in limited memory where fallback Hessian is updated only when needed
        bool use_homotopy;                             // Indicates wether QP solver should use homotopy. Currently only affects qpOASES

        bool KKT_heuristic_enabled;
        double KKTerror_save;                          //Previous KKT error for comparison, steptype indicates if KKT heuristic is active

        bool nearSol;                                  // Indicates that iterates reached near a local optimum in terms of KKT error
        double milestone;                              // Lowest scaled KKT error so far

        bool solution_found;                           // Indicates that a solution satisfying the tolerances has been found and extra steps are done for improved accuracy
        int n_extra;                                   // Number of current extra step

        // 
        std::set<std::pair<double, double>> filter;   // Filter for line search, contains (constrVio, objective)


        //Convexification strategy 1 and 2
        int hess_num_accepted;                        // order of Hessian regularization for last QP, ranging from 0 (no regularization) to options.max_conv_QPs (fallback)
        int QP_num_accepted;                          // Number of QP whose step was chosen, different from hess_num_accepted because the fallback step may be chosen if its norm is greater than a regularized Hessian step
        double convKappa;                             // Last factor in convexification strategy, factors are ... , 2^-2 * convKappa, 2^-1 * convKappa, convKappa     
        
        //Scaling heuristic
        std::unique_ptr<double[]> rescaleFactors;
        int n_scaleIt;                                // How many past steps are available (not necessarily used by heuristic) to compute the scaling.
        double vfreeScale;                            // Extra save
        

        //Derived options
        double modified_hess_regularizationFactor;

        //Step backup. Used during extra step phase to save the best iterate
        bool it_saved;
        Matrix xiOpt_save;
        Matrix lambdaOpt_save;
        double objOpt_save;
        Matrix constrOpt_save;
        double tolOpt_save;
        double cNormOpt_save;
        double cNormSOpt_save;
        scaled_Problemspec *scaled_prob;              // Pointer to a scaled problem or nullptr. Used to save and restore scaling factors. 
        std::unique_ptr<double[]> scaleFactors_save;
        
        //For parallel QP solution
        int N_QP_cancels;
        std::unique_ptr<Matrix[]> par_QP_sols_prim;
        std::unique_ptr<Matrix[]> par_QP_sols_dual;
        
        
        
        
    public:
        /// Call allocation and initializing routines
        SQPiterate(Problemspec* prob, const SQPoptions* param);
        SQPiterate();
        // WARNING: No complete copy. Used only in finite difference Hessian calculation.
        SQPiterate( const SQPiterate &iter );
        /// Set initial filter, objective function, tolerances etc.
        void initIterate( SQPoptions* param );
        virtual ~SQPiterate( void );

        void save_iterate();    //Save xi, lambda, tol, cNorm and cNormS
        void restore_iterate(); //Restore the above from save
};


} // namespace blockSQP

#endif
