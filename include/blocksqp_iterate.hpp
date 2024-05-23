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
        //double qpObj;                                 ///< objective value of last QP subproblem
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
        Matrix deltaXi;                               ///< alias for current step
        Matrix gradObj;                               ///< gradient of objective
        Matrix gradLagrange;                          ///< gradient of Lagrangian
        Matrix gammaMat;                              ///< Lagrangian gradient differences for last m steps
        Matrix gamma;                                 ///< alias for current Lagrangian gradient

        int nBlocks;                                  ///< number of diagonal blocks in Hessian
        int *blockIdx;                                ///< indices in the variable vector that correspond to diagonal blocks (nBlocks+1)

        SymMatrix *hess;                              ///< [blockwise] pointer to current Hessian of the Lagrangian
        SymMatrix *hess1;                             ///< [blockwise] first Hessian approximation
        SymMatrix *hess2;                             ///< [blockwise] second Hessian approximation (convexified)
        SymMatrix *hess_conv;                         ///< [blockwise] convex combination of first and second Hessian approximation if two or more additional qps are solved per iteration
        //SymMatrix *hess_save;

        double *hessNz;                               ///< nonzero elements of Hessian (length)
        int *hessIndRow;                              ///< row indices (length)
        int *hessIndCol;                              ///< indices to first entry of columns (nCols+1)
        int *hessIndLo;                               ///< Indices to first entry of lower triangle (including diagonal) (nCols)

        double conv_identity_scale;                        ///< Current scaling factor for added identity in hessian convexification


        bool conv_qp_solved;
        bool hess2_calculated;

        /*
         * Variables for QP solver
         */
        bool use_homotopy;
        Matrix delta_lb_var;                          ///< lower bounds for current step
        Matrix delta_ub_var;                          ///< upper bounds for current step
        Matrix delta_lb_con;                          ///< lower bounds for linearized constraints
        Matrix delta_ub_con;                          ///< upper bounds for linearized constraints
        Matrix lambdaQP;                              ///< dual variables of QP
        Matrix AdeltaXi;                              ///< product of constraint Jacobian with deltaXi

        /*
         * For modified BFGS updates
         */
        Matrix deltaNorm;                             ///< sTs
        Matrix deltaNormOld;                          ///< (from previous iteration)
        Matrix deltaGamma;                            ///< sTy
        Matrix deltaGammaOld;                         ///< (from previous iteration)
        int *noUpdateCounter;                         ///< count skipped updates for each block

        /*
         * Variables for globalization strategy
         */
        int steptype;                                 ///< -1: KKT-error reduction step, 0: Linesearch step, 1: Step with identity hessian
                                                      ///<  2: Feasibility restoration heuristic step, 3: Feasibility restoration step

        double solution_durations[10];                ///< Solution time of the last 10 successful qps
        int dur_pos;
        double avg_solution_duration;


        double alpha;                                 ///< stepsize for line search
        int nSOCS;                                    ///< number of second-order correction steps
        int reducedStepCount;                         ///< count number of consecutive reduced steps,
        //Matrix deltaH;                                ///< scalars for inertia correction (filter line search w indef Hessian)
        Matrix trialXi;                               ///< new trial iterate (for line search)
        std::set< std::pair<double,double> > *filter; ///< Filter contains pairs (constrVio, objective)

    /*
     * Methods
     */
    public:
        /// Call allocation and initializing routines
        SQPiterate( Problemspec* prob, SQPoptions* param, bool full );
        SQPiterate();
        SQPiterate( const SQPiterate &iter );
        /// Allocate variables that any SQP code needs
        //void allocMin( Problemspec* prob );
        /// Allocate diagonal block Hessian
        //void allocHess( SQPoptions* param );
        /// Convert *hess to column compressed sparse format
        void convertHessian( int num_vars, int num_hessblocks, double eps, SymMatrix *&hess_,
                             double *&hessNz_, int *&hessIndRow_, int *&hessIndCol_, int *&hessIndLo_ );
        /// Convert *hess to double array (dense matrix)
        void convertHessian( Problemspec *prob, double eps, SymMatrix *&hess_ );
        /// Allocate variables specifically needed by vmused SQP method
        //void allocAlg( Problemspec* prob, SQPoptions* param );
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
