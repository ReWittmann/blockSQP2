/*
 * blockSQP -- Sequential quadratic programming for problems with
 *             block-diagonal Hessian matrix.
 * Copyright (C) 2012-2015 by Dennis Janka <dennis.janka@iwr.uni-heidelberg.de>
 *
 * Licensed under the zlib license. See LICENSE for more details.
 */

/**
 * \file blocksqp_method.hpp
 * \author Dennis Janka
 * \date 2012-2015
 *
 *  Declaration of blockSQP's central SQPmethod class.
 */

#ifndef BLOCKSQP_METHOD_HPP
#define BLOCKSQP_METHOD_HPP

//#include "qpOASES.hpp"
#include "blocksqp_defs.hpp"
#include "blocksqp_matrix.hpp"
#include "blocksqp_problemspec.hpp"
#include "blocksqp_options.hpp"
#include "blocksqp_iterate.hpp"
#include "blocksqp_stats.hpp"
#include "blocksqp_qpsolver.hpp"
#include <iostream>

namespace blockSQP{

/**
 * \brief Describes an SQP method for a given problem and set of algorithmic options.
 * \author Dennis Janka
 * \date 2012-2015
 */
class SQPmethod{

    public:
        Problemspec*             prob;        ///< Problem structure (has to provide evaluation routines)
        SQPoptions*              param;       ///< Set of algorithmic options and parameters for this method
        SQPstats*                stats;       ///< Statistics object for current SQP run

        SQPiterate*              vars;        ///< All SQP variables for this method
        QPsolver*                sub_QP;      ///< Class wrapping an external QP solver

    //Feasibility restoration problem
        Problemspec*             rest_prob;
        SQPoptions*              rest_opts;
        SQPstats*                rest_stats;
        SQPmethod*               rest_method;

    protected:
        bool                     initCalled;  ///< indicates if init() has been called (necessary for run())

    /*
     * Methods
     */
    public:
        /// Construct a method for a given problem and set of algorithmic options
        SQPmethod( Problemspec *problem, SQPoptions *parameters, SQPstats *statistics );
        SQPmethod();
        virtual ~SQPmethod();

        /// Initialization, has to be called before run
        void init();
        /// Main Loop of SQP method
        int run( int maxIt, int warmStart = 0 );
        /// Call after the last call of run, to close output files etc.
        void finish();
        /// Print information about the SQP method
        void printInfo( int printLevel );
        /// Compute gradient of Lagrangian function (dense version)
        void calcLagrangeGradient( const Matrix &lambda, const Matrix &gradObj, const Matrix &constrJacFull, Matrix &gradLagrange, int flag );
        /// Compute gradient of Lagrangian function (sparse version)
        void calcLagrangeGradient( const Matrix &lambda, const Matrix &gradObj, double *jacNz, int *jacIndRow, int *jacIndCol, Matrix &gradLagrange, int flag );
        /// Overloaded function for convenience, uses current variables of SQPiterate vars
        void calcLagrangeGradient( Matrix &gradLagrange, int flag );
        /// Update optimization tolerance (similar to SNOPT) in current iterate
        bool calcOptTol();

        /*
         * Solve QP subproblem
         */
        /// Update the bounds on the current step, i.e. the QP variables
        void updateStepBounds();
        /// Update the bounds on the current step for a second order correction, i.e. lb_s = lb - constr(trialXi) + constrJac(Xi)*deltaXi = prob->lb_con - vars->trialConstr + vars->AdeltaXi
        void updateStepBoundsSOC();
        /// Solve a QP with QPOPT or qpOASES to obtain a step deltaXi and estimates for the Lagrange multipliers.
        //If conv_qp is false, solution is tries with increasingly convexified hessian approximations, else only the convex fallback hessian is used
        virtual int solveQP( Matrix &deltaXi, Matrix &lambdaQP, bool conv_qp = false );
        /// Solve a QP with convex hessian and corrected constraint bounds. vars->AdeltaXi, vars->trialConstr need to be updated before calling this method
        virtual int solve_SOC_QP( Matrix &deltaXi, Matrix &lambdaQP);
        /// Compute the next Hessian in the inner loop of increasingly convexified QPs and store it in vars->hess2
        virtual void computeNextHessian( int idx, int maxQP );
        /// Compute a convexified hessian and store it in vars->hess2, set hess to hess2
        virtual void computeConvexHessian();

        /*
         * Globalization Strategy
         */
        /// No globalization strategy
        int fullstep();
        /// Set new primal dual iterate
        void acceptStep( const Matrix &deltaXi, const Matrix &lambdaQP, double alpha, int nSOCS );
        /// Overloaded function for convenience, uses current variables of SQPiterate vars
        void acceptStep( double alpha );
        /// Reduce stepsize if a step is rejected
        void reduceStepsize( double *alpha );
        /// Determine steplength alpha by a filter based line search similar to IPOPT
        int filterLineSearch();
        /// Remove all entries from filter
        void initializeFilter();
        /// Is a pair (cNorm, obj) in the current filter?
        bool pairInFilter( double cNorm, double obj );
        /// Augment current filter by pair (cNorm, obj)
        void augmentFilter( double cNorm, double obj );
        /// Perform a second order correction step (solve QP)
        virtual bool secondOrderCorrection( double cNorm, double cNormTrial, double dfTdeltaXi, bool swCond);
        /// Start feasibility restoration heuristic
        int feasibilityRestorationHeuristic();
        /// Start feasibility restoration phase (solve NLP)
        virtual int feasibilityRestorationPhase();
        /// Check if full step reduces KKT error
        int kktErrorReduction( );

        /*
         * Hessian Approximation
         */
        /// Set initial Hessian: Identity matrix
        void calcInitialHessian(SymMatrix *hess);
        /// [blockwise] Set initial Hessian: Identity matrix
        void calcInitialHessian(int iBlock, SymMatrix *hess);

        void calcInitialHessians();

        void calcScaledInitialHessian(double scale, SymMatrix *hess);
        void calcScaledInitialHessian(int iBlock, double scale, SymMatrix *hess);

        /// Reset Hessian to identity and remove past information on Lagrange gradient and steps
        void resetHessian(SymMatrix *hess);
        /// [blockwise] Reset Hessian to identity and remove past information on Lagrange gradient and steps
        void resetHessian(int iBlock, SymMatrix *hess);
        /// Shortcut method to reset the hessian and the fallback hessian if it is in use
        void resetHessians();

        ///
        /// Update methods for the block hessian. May store data in vars for the specific hessian given (hess1 or hess2)
        ///
        /// Compute current Hessian approximation by finite differences
        int calcFiniteDiffHessian(SymMatrix *hess);
        /// Compute full memory Hessian approximations based on update formulas
        void calcHessianUpdate(int updateType, int hessScaling, SymMatrix *hess);
        /// Compute limited memory Hessian approximations based on update formulas
        void calcHessianUpdateLimitedMemory(int updateType, int hessScaling, SymMatrix *hess);
        /// [blockwise] Compute new approximation for Hessian by SR1 update
        void calcSR1( const Matrix &gamma, const Matrix &delta, int iBlock, SymMatrix *hess);
        /// [blockwise] Compute new approximation for Hessian by BFGS update with Powell modification
        void calcBFGS( const Matrix &gamma, const Matrix &delta, int iBlock, bool damping, SymMatrix *hess);
        /// Set pointer to correct step and Lagrange gradient difference in a limited memory context
        void updateDeltaGamma();

        /*
         * Scaling of Hessian Approximation
         */
        /// [blockwise] Update scalars for COL sizing of Hessian approximation (full memory, save last 2)
        void updateScalarProducts();
        /// [blockwise] Update scalars for COL sizing of Hessian approximation (limited memory, save param->hessMemsize)
        void updateScalarProductsLimitedMemory();

        /// [blockwise] Size Hessian using SP, OL, or mean sizing factor
        void sizeInitialHessian( const Matrix &gamma, const Matrix &delta, int iBlock, int option, SymMatrix *hess);
        /// [blockwise] Size Hessian using the COL scaling factor
        void sizeHessianCOL( const Matrix &gamma, const Matrix &delta, int iBlock, bool first_sizing, SymMatrix *hess);
};

////////////////////////////////////////////////////////////

//Sequential Condensed Quadratic Programming method
class SCQPmethod : public SQPmethod{
public:
    Condenser *cond;

    //Restoration problem with own condenser
    Condenser*              rest_cond;
    vblock*                 rest_vblocks;
    cblock*                 rest_cblocks;
    int*                    rest_h_sizes;
    condensing_target*      rest_targets;

    SCQPmethod(Problemspec *problem, SQPoptions *parameters, SQPstats *statistics, Condenser *CND);
    SCQPmethod();
    virtual ~SCQPmethod();

    virtual int solveQP(Matrix &deltaXi, Matrix &lambdaQP, bool conv_qp = false);
    virtual int solve_SOC_QP(Matrix &deltaXi, Matrix &lambdaQP);
    virtual int feasibilityRestorationPhase();
};


class SCQP_bound_method : public SCQPmethod{
public:
    SCQP_bound_method(Problemspec *problem, SQPoptions *parameters, SQPstats *statistics, Condenser *CND);

    virtual int solveQP(Matrix &deltaXi, Matrix &lambdaQP, bool conv_qp = false);
    virtual int solve_SOC_QP(Matrix &deltaXi, Matrix &lambdaQP);
};


class SCQP_correction_method : public SCQPmethod{
public:
    Matrix *corrections;
    Matrix *SOC_corrections;

    SCQP_correction_method(Problemspec *problem, SQPoptions *parameters, SQPstats *statistics, Condenser *CND);
    virtual ~SCQP_correction_method();

    virtual int solveQP(Matrix &deltaXi, Matrix &lambdaQP, bool conv_qp = false);
    virtual int solve_SOC_QP(Matrix &deltaXi, Matrix &lambdaQP);

    virtual int feasibilityRestorationPhase();
};



} // namespace blockSQP

#endif
