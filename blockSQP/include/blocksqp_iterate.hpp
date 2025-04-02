/*
 * blockSQP -- Sequential quadratic programming for problems with
 *             block-diagonal Hessian matrix.
 * Copyright (C) 2012-2015 by Dennis Janka <dennis.janka@iwr.uni-heidelberg.de>
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
 */

#ifndef BLOCKSQP_ITERATE_HPP
#define BLOCKSQP_ITERATE_HPP

#include "blocksqp_defs.hpp"
#include "blocksqp_matrix.hpp"
#include "blocksqp_problemspec.hpp"
#include "blocksqp_options.hpp"
#include <memory>

namespace blockSQP
{

/**
 * \brief Holds all variables that are updated during one SQP iteration
 * \author Dennis Janka
 * \date 2012-2015
 */
class SQPiterate{
    public:
        double obj;                                   ///< objective value
        double cNorm;                                 ///< constraint violation
        double cNormS;                                ///< scaled constraint violation
        double gradNorm;                              ///< norm of Lagrangian gradient
        double lambdaStepNorm;                        ///< norm of step in dual variables
        double tol;                                   ///< current optimality tolerance

        Matrix xi;                                    ///< variable vector
        Matrix lambda;                                ///< dual variables
        Matrix constr;                                ///< constraint vector

        Matrix constrJac;                             ///< full constraint Jacobian (not used in sparse mode)
        
        //double *jacNz;                                ///< nonzero elements of Jacobian (length)
        //int *jacIndRow;                               ///< row indices (length)
        //int *jacIndCol;                               ///< indices to first entry of columns (nCols+1)
        std::unique_ptr<double[]> jacNz;
        std::unique_ptr<int[]> jacIndRow;
        std::unique_ptr<int[]> jacIndCol;

        Matrix deltaMat;                              ///< last m primal steps
        Matrix deltaXi;                               ///< alias for current step (first stores full step from QP, then gets modified by globalization)
        Matrix gradObj;                               ///< gradient of objective
        Matrix gradLagrange;                          ///< gradient of Lagrangian
        Matrix gammaMat;                              ///< Lagrangian gradient differences for last m steps
        Matrix gamma;                                 ///< alias for current Lagrangian gradient

        //Scalar products for COL sizing. In full memory quasi newton, they are updated at the end of each SQP iteration. In limited memory, they are calculated when applying the update
        Matrix deltaNormSqMat;                          /// last m >= 2 squared step norms
        Matrix deltaGammaMat;                         /// last m >= 2 delta-gamma scalar products
        int dg_pos;                                   /// position of the current iterate within deltaNormSqMat and gammaNormMat as well as deltaMat and gammaMat if in limited memory
        int dg_nsave;                                    /// number of saved delta-gamma pairs and their scalar products. These are required for limited memory updates and the scaling heuristic

        //Damping may change gamma, old scalar product is required for COL sizing
        Matrix deltaGammaOld;
        Matrix deltaGammaOldFallback;

        //For full memory, no old steps delta and gradient differences gamma are saved. The sectioned norm of the step is required for COl sizing.
        //The step deltaOld is required to restore the sectioned norm after variable rescaling
        Matrix deltaNormSqOld;
        Matrix deltaOld;


        //int *nquasi;                                  ///< number of quasi-newton updates for each block since last hessian reset
        //int *noUpdateCounter;                         ///< count skipped updates for each block
        std::unique_ptr<int[]> nquasi;
        std::unique_ptr<int[]> noUpdateCounter;

        int nBlocks;                                  ///< number of diagonal blocks in Hessian
        std::unique_ptr<int[]> blockIdx;                                ///< indices in the variable vector that correspond to diagonal blocks (nBlocks+1)

        int nRestIt;

        SymMatrix *hess;                              ///< [blockwise] pointer to current Hessian (-approximation) of the Lagrangian
        
        std::unique_ptr<SymMatrix[]> hess1;                             ///< [blockwise] first Hessian approximation
        std::unique_ptr<SymMatrix[]> hess2;                             ///< [blockwise] second Hessian approximation (convex)
        std::unique_ptr<SymMatrix[]> hess_alt;                          ///< [blockwise] space to store alternative hessians, such as convex combinations or temporarily used (scaled) identity hessians

        bool conv_qp_only;                            ///< If true, only convex sub-QPs are used to generate steps
        bool conv_qp_solved;
        bool hess2_updated;

        int hess_num_accepted;                        ///< order of hessian convexification for last QP, ranging from 0 (no regularization) to options.maxConvQP (fallback)


        bool use_homotopy;

        //Bounds for QP step, calculated in solveQP directly before invoking QP solver / condenser
        Matrix delta_lb_var;                          ///< lower bounds for current (SOC) step
        Matrix delta_ub_var;                          ///< upper bounds for current (SOC) step
        Matrix delta_lb_con;                          ///< lower bounds for linearized (SOC) constraints
        Matrix delta_ub_con;                          ///< upper bounds for linearized (SOC) constraints
        Matrix lambdaQP;                              ///< dual variables of QP

        Matrix AdeltaXi;                              ///< product of constraint Jacobian with deltaXi (from SOC for SOC iterations after the first one), calculated in secondOrderCorrection method as needed


        int steptype;                                 ///< -2: Filter-overwriting step -1: Step heuristic, 0: Linesearch step, 1: Linesearch step with identity Hessian, 2: Feasibility restoration heuristic step, 3: Feasibility restoration step
        int n_id_hess;                                ///< Number of condecutive uses of identity hessian as fallback

        double alpha;                                 ///< stepsize for line search
        int nSOCS;                                    ///< number of second-order correction steps
        int reducedStepCount;                         ///< count number of consecutive reduced steps,

        Matrix trialXi;                               ///< new trial iterate (for line search)
        Matrix trialLambda;
        Matrix trialConstr;                           ///< constraints evaluated at trial point. Calculated in linesearch and used also in SOC                           

        std::set<std::pair<double, double>> filter;   ///< Filter contains pairs (constrVio, objective)

        
        //Parameters derived from given options, may change during iterations
        double modified_hess_regularizationFactor;
        double convKappa;

        //Ignore filter up to a limited amount of times close to a solution as rounding errors can caues errors in line search.
        int local_lenience;
        
        
        std::unique_ptr<double[]> rescaleFactors;
        int n_scaleIt;                               ///< How many past steps are available (not necessarily used by heuristic) to compute the scaling.
        double vfreeScale;
        
        bool nearSol;
        double milestone;

        //Fields used by KKT step-heuristic
        bool KKT_heuristic_active;
        //Previous KKT error for comparison
        double tol_save;
        
        int n_extra;
        bool sol_found; 
        Matrix xiOpt_save;
        Matrix lambdaOpt_save;
        double objOpt_save;
        Matrix constrOpt_save;
        double tolOpt_save;
        double cNormOpt_save;
        double cNormSOpt_save;
        std::unique_ptr<double[]> scaleFactors_save;
        scaled_Problemspec *scaled_prob;
    public:
        /// Call allocation and initializing routines
        SQPiterate(Problemspec* prob, const SQPoptions* param, bool full );
        SQPiterate();
        SQPiterate( const SQPiterate &iter );
        /// Set initial filter, objective function, tolerances etc.
        void initIterate( SQPoptions* param );
        virtual ~SQPiterate( void );

        void save_iterate();    //Save xi, lambda, tol, cNorm and cNormS
        void restore_iterate(); //Restore the above from save
};



class SCQPiterate : public SQPiterate{
    public:
    //Wrapper object for sparse jacobian arrays, need it to invoke condensing
    Sparse_Matrix Jacobian;

    Matrix condensed_h;
    Sparse_Matrix condensed_Jacobian;
    std::unique_ptr<SymMatrix[]> condensed_hess;
    Matrix condensed_lb_var;
    Matrix condensed_ub_var;
    Matrix condensed_lb_con;
    Matrix condensed_ub_con;

    double *condensed_hess_nz;
    int *condensed_hess_row;
    int *condensed_hess_colind;
    int *condensed_hess_loind;

    //Hessian convex combination factor
    double t_hess;

    //Condensed fallback hessian
    std::unique_ptr<SymMatrix[]> condensed_hess_2;

    //Solutions of condensed QP
    Matrix deltaXi_cond;
    Matrix lambdaQP_cond;

    SCQPiterate(Problemspec* prob, SQPoptions* param, Condenser* cond, bool full);
    virtual ~SCQPiterate();

};

class SCQP_correction_iterate : public SCQPiterate{
public:
    Matrix corrected_h;
    Matrix corrected_lb_con;
    Matrix corrected_ub_con;
    
    Matrix deltaXi_save;    //For restoring original step if correction yields no full step
    Matrix lambdaQP_save;   // ' '    ' ' 
    SCQP_correction_iterate(Problemspec* prob, SQPoptions* param, Condenser* cond, bool full);
};



} // namespace blockSQP

#endif
