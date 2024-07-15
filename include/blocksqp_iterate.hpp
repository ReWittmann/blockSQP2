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

namespace blockSQP
{

/**
 * \brief Holds all variables that are updated during one SQP iteration
 * \author Dennis Janka
 * \date 2012-2015
 */
class SQPiterate
{
    /*
     * Variables
     */
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
        double *jacNz;                                ///< nonzero elements of Jacobian (length)
        int *jacIndRow;                               ///< row indices (length)
        int *jacIndCol;                               ///< indices to first entry of columns (nCols+1)

        Matrix deltaMat;                              ///< last m primal steps
        Matrix deltaXi;                               ///< alias for current step (first stores full step from QP, then gets modified by globalization)
        Matrix gradObj;                               ///< gradient of objective
        Matrix gradLagrange;                          ///< gradient of Lagrangian
        Matrix gammaMat;                              ///< Lagrangian gradient differences for last m steps
        Matrix gamma;                                 ///< alias for current Lagrangian gradient

        //Scalar products for COL sizing. In full memory quasi newton, they are updated at the end of each SQP iteration. In limited memory, they are calculated when applying the up
        Matrix deltaNormMat;                          /// last m >= 2 squared step norms
        Matrix deltaGammaMat;                         /// last m >= 2 delta-gamma scalar products
        int dg_pos;                                   /// position of the current iterate within deltaNormMat and gammaNormMat as well as deltaMat and gammaMat if in limited memory

        //[blockwise] precalculated scalar products, needed for quasi-newton updates and sizing
        Matrix deltaNorm;                             ///< sTs, subvector of deltaNormMat
        Matrix deltaGamma;                            ///< sTy, subvector of deltaGammaMat

        //[blockwise] norm and scalar product of the last delta-gamma pair for the secant update was successful and secand equation is fulfilled.
        //Dampening may have been applied to gamma und thus to deltaGamma
        //These are set during the update calculation (SR1, BFGS etc.) and required for COL sizing. They may be different for each of the two maintained hessians, so
        //we need one pair for each hessian

        //For hess1
        Matrix deltaNormOld;
        Matrix deltaGammaOld;

        //For hess2
        Matrix deltaNormOldFallback;
        Matrix deltaGammaOldFallback;

        int *nquasi;                                  ///< number of quasi-newton updates for each block since last hessian reset
        int *noUpdateCounter;                         ///< count skipped updates for each block

        int nBlocks;                                  ///< number of diagonal blocks in Hessian
        int *blockIdx;                                ///< indices in the variable vector that correspond to diagonal blocks (nBlocks+1)

        SymMatrix *hess;                              ///< [blockwise] pointer to current Hessian of the Lagrangian
        SymMatrix *hess1;                             ///< [blockwise] first Hessian approximation
        SymMatrix *hess2;                             ///< [blockwise] second Hessian approximation (convexified)
        SymMatrix *hess_conv;                         ///< [blockwise] convex combination of first and second Hessian approximation if two or more additional qps are solved per iteration

        double *hessNz;                               ///< nonzero elements of Hessian (length)
        int *hessIndRow;                              ///< row indices (length)
        int *hessIndCol;                              ///< indices to first entry of columns (nCols+1)
        int *hessIndLo;                               ///< Indices to first entry of lower triangle (including diagonal) (nCols)

        bool conv_qp_solved;
        bool hess2_calculated;

        /*
         * Variables for QP solver
         */
        bool use_homotopy;

        //Bounds for QP step, calculated in solveQP directly before invoking QP solver / condenser
        Matrix delta_lb_var;                          ///< lower bounds for current (SOC) step
        Matrix delta_ub_var;                          ///< upper bounds for current (SOC) step
        Matrix delta_lb_con;                          ///< lower bounds for linearized (SOC) constraints
        Matrix delta_ub_con;                          ///< upper bounds for linearized (SOC) constraints
        Matrix lambdaQP;                              ///< dual variables of QP

        Matrix AdeltaXi;                              ///< product of constraint Jacobian with deltaXi (from SOC for SOC iterations after the first one), calculated in secondOrderCorrection method as needed

        /*
         * Variables for globalization strategy
         */
        int steptype;                                 ///< -1: KKT-error reduction step, 0: Linesearch step, 1: Step with identity hessian, 2: Feasibility restoration heuristic step, 3: Feasibility restoration step

        double alpha;                                 ///< stepsize for line search
        int nSOCS;                                    ///< number of second-order correction steps
        int reducedStepCount;                         ///< count number of consecutive reduced steps,

        Matrix trialXi;                               ///< new trial iterate (for line search)
        Matrix trialConstr;                           ///< constraints evaluated at trial point. Calculated in linesearch and used also in SOC

        std::set< std::pair<double,double> > *filter; ///< Filter contains pairs (constrVio, objective)


        //Parameters derived from given options, may change during iterations
        //int hessUpdate;
        //int fallbackUpdate;
        //int hessMemSize;


    /*
     * Methods
     */
    public:
        /// Call allocation and initializing routines
        SQPiterate( Problemspec* prob, SQPoptions* param, bool full );
        SQPiterate();
        SQPiterate( const SQPiterate &iter );
        /// Convert *hess to column compressed sparse format
        void convertHessian( int num_vars, int num_hessblocks, double eps, SymMatrix *&hess_,
                             double *&hessNz_, int *&hessIndRow_, int *&hessIndCol_, int *&hessIndLo_ );
        /// Convert *hess to double array (dense matrix)
        void convertHessian( Problemspec *prob, double eps, SymMatrix *&hess_ );
        /// Set initial filter, objective function, tolerances etc.
        void initIterate( SQPoptions* param );
        virtual ~SQPiterate( void );
};



class SCQPiterate : public SQPiterate{
public:

    //Wrapper object for sparse jacobian arrays, need it to invoke condensing
    Sparse_Matrix Jacobian;

    /*
     * Variables of condensed QP
     */

    Matrix condensed_h;
    Sparse_Matrix condensed_Jacobian;
    SymMatrix *condensed_hess;
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
    SymMatrix *condensed_hess_2;

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
    SCQP_correction_iterate(Problemspec* prob, SQPoptions* param, Condenser* cond, bool full);
};



} // namespace blockSQP

#endif
