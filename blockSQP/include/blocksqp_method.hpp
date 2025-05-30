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

#include "blocksqp_defs.hpp"
#include "blocksqp_matrix.hpp"
#include "blocksqp_problemspec.hpp"
#include "blocksqp_options.hpp"
#include "blocksqp_iterate.hpp"
#include "blocksqp_stats.hpp"
#include "blocksqp_qpsolver.hpp"
#include "blocksqp_restoration.hpp"
#include <iostream>
#include <memory>

namespace blockSQP{

class SQPmethod{
    public:
        //Provided problem data and algorithmic settings
        Problemspec*             prob;        ///< Pointer to used problem, may be original problem or scalable problem wrapper
        SQPoptions*              param;       ///< Set of algorithmic options and parameters for this method
        SQPstats*                stats;       ///< Statistics object for current SQP run
        
        std::unique_ptr<SQPiterate> vars;     ///< All SQP variables for this method
        std::unique_ptr<QPsolverBase>   sub_QP;   ///< Class wrapping an external QP solver
        
        std::unique_ptr<std::unique_ptr<QPsolverBase>[]> sub_QPs_par;   //QPsolver objects for parallelizing the QP solving loop
        
        //Scalable problem used internally, wraps the original problem and is used in it's stead if automatic scaling is activated
        std::unique_ptr<scaled_Problemspec> scaled_prob;
        
        // Objects for feasibility restoration
        //Feasibility restoration problem
        std::unique_ptr<abstractRestorationProblem> rest_prob;
        std::unique_ptr<SQPoptions>  rest_param;
        std::unique_ptr<SQPstats>    rest_stats;
        std::unique_ptr<SQPmethod>   rest_method;
        //Returned iterates from restoration method. We do not assume that they are identical to those stored in the method (future scaling for restoration problems!)
        Matrix rest_xi;
        Matrix rest_lambda;
        Matrix rest_lambdaQP;

    protected:
        bool                     initCalled = false;  ///< indicates if init() has been called (necessary for run())

    public:
        /// Construct a method for a given problem and set of algorithmic options
        SQPmethod( Problemspec *problem, SQPoptions *parameters, SQPstats *statistics );
        SQPmethod();
        virtual ~SQPmethod();

        // Main interface methods
        /// Initialization, has to be called before run
        void init();
        /// Main Loop of SQP method
        SQPresult run( int maxIt, int warmStart = 0 );
        /// Call after the last call of run, to close output files etc.
        void finish();
        
        // Utility methods
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
        /// Set pointer to correct step and Lagrange gradient difference in a limited memory context
        void updateDeltaGammaData();


        // Solution of quadratic subproblems
        /// Update the bounds on the current step, i.e. the QP variables
        void updateStepBounds();
        /// Update the bounds on the current step for a second order correction, i.e. lb_s = lb - constr(trialXi) + constrJac(Xi)*deltaXi = prob->lb_con - vars->trialConstr + vars->AdeltaXi
        void updateStepBoundsSOC();
        /// Solve a QP with QPOPT or qpOASES to obtain a step deltaXi and estimates for the Lagrange multipliers.
        //If hess_type is 0, solution is tried with increasingly convexified hessian approximations. If hess_type is 1, only convex hessian approximations are used. If hess_type is 2, only the (scaled) identity is used as hessian
        virtual int solve_initial_QP(Matrix &deltaXi, Matrix &lambdaQP);
        virtual int solveQP(Matrix &deltaXi, Matrix &lambdaQP, int hess_type = 0);
        
        virtual int solveQP_par(Matrix &deltaXi, Matrix &lambdaQP);
        //virtual int solveQP_par_conv_str_2(Matrix &deltaXi, Matrix &lambdaQP);
        
        /// Solve a QP with convex hessian and corrected constraint bounds. vars->AdeltaXi, vars->trialConstr need to be updated before calling this method
        virtual int solve_SOC_QP( Matrix &deltaXi, Matrix &lambdaQP);
        
        // ??? QP_loop()
        // 
        
        /// Compute the next Hessian in the inner loop of increasingly convexified QPs and store it in vars->hess2
        virtual void computeNextHessian( int idx, int maxQP );
        /// Compute a convexified hessian and store it in vars->hess2, set hess to hess2
        virtual void computeConvexHessian();
        /// Set hess to point to a blockwise (scaled) identity hessian, (vars->hess_spec)
        virtual void setIdentityHessian();


        // Filter line search, restoration phase and associated heuristics
        /// No enable_linesearch strategy
        int fullstep();
        /// Set new primal dual iterate
        void acceptStep( const Matrix &deltaXi, const Matrix &lambdaQP, double alpha, int nSOCS );
        /// Overloaded function for convenience, uses current variables of SQPiterate vars
        void acceptStep( double alpha );
        /// Set a new iterate, ignoring the filter and remove all entries from the filter that dominate the new point
        void force_accept(const Matrix &deltaXi, const Matrix &lambdaQP, double alpha, int nSOCS);
        void force_accept(double alpha);
        /// Set a new iterate and update derivatives
        void set_iterate(const Matrix &xi, const Matrix &lambda, bool resetHessian = false);
        Matrix get_xi();
        Matrix get_lambda();
        void get_xi(Matrix &xi_hold);
        void get_lambda(Matrix &lambda_hold);
        void get_lambdaQP(Matrix &lambdaQP_hold);

        /// Reduce stepsize if a step is rejected
        void reduceStepsize( double *alpha );
        /// Determine steplength alpha by a filter based line search similar to IPOPT
        virtual int filterLineSearch();
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
        /// Main loop of restoration phase - check acceptability of the filter after each step
        int innerRestorationPhase(abstractRestorationProblem *argRestProb, SQPmethod *argRestMeth, bool argWarmStart, double min_stepsize_sum = 1.0);
        /// Check if full step reduces KKT error
        int kktErrorReduction( );


        // Hessian approximation and sizing

        /// Set initial Hessian: Identity matrix
        void calcInitialHessian(SymMatrix *hess);
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

        /// Compute current Hessian approximation by finite differences
        int calcFiniteDiffHessian(SymMatrix *hess);
        /// Compute full memory Hessian approximations based on update formulas
        void calcHessianUpdate(int updateType, int sizing, SymMatrix *hess);
        /// Compute limited memory Hessian approximations based on update formulas
        void calcHessianUpdateLimitedMemory(int updateType, int sizing, SymMatrix *hess);
        /// [blockwise] Compute new approximation for Hessian by SR1 update
        void calcSR1(int dpos, int iBlock, SymMatrix *hess);
        /// [blockwise] Compute new approximation for Hessian by BFGS update with Powell modification
        void calcBFGS(int dpos, int iBlock, SymMatrix *hess, bool damping);

        /// Oren-Luenberger sizing of initial Hessian
        void sizeInitialHessian(int dpos, int iBlock, SymMatrix *hess, int option);
        /// Centered Oren-Luenberger sizing
        void sizeHessianCOL(int dpos, int iBlock, SymMatrix *hess);


        // Rescaling of the problem (only variables)
        void calc_free_variables_scaling(double *SF);
        void apply_rescaling(const double *resfactors);
        void scaling_heuristic();
};

////////////////////////////////////////////////////////////

//Sequential Condensed Quadratic Programming method

/*
class SCQPmethod : public SQPmethod{
    public:
    Condenser *cond;
    
    //Condenser for restoration problem
    std::unique_ptr<Condenser> rest_cond;

    SCQPmethod(Problemspec *problem, SQPoptions *parameters, SQPstats *statistics, Condenser *CND);
    SCQPmethod();
    virtual ~SCQPmethod();

    // QP solution methods incorporating the condensing step
    virtual int solveQP(Matrix &deltaXi, Matrix &lambdaQP, int hess_type = 0);
    virtual int solve_SOC_QP(Matrix &deltaXi, Matrix &lambdaQP);

    // Restoration phase with slack variables omitted for dependency constraints (continuity conditions in multiple shooting) 
    virtual int feasibilityRestorationPhase();

    // Convexification strategy for condensed Hessians
    void convexify_condensed(SymMatrix *condensed_hess, int idx, int maxQP);
};


class SCQP_bound_method : public SCQPmethod{
    public:
    SCQP_bound_method(Problemspec *problem, SQPoptions *parameters, SQPstats *statistics, Condenser *CND);

    // condensed QP solution methods incorporating QP resolves with violated bounds added
    virtual int solveQP(Matrix &deltaXi, Matrix &lambdaQP, int hess_type = 0);
    virtual int solve_SOC_QP(Matrix &deltaXi, Matrix &lambdaQP);
};


class SCQP_correction_method : public SCQPmethod{
    public:
    Matrix *corrections;
    Matrix *SOC_corrections;


    SCQP_correction_method(Problemspec *problem, SQPoptions *parameters, SQPstats *statistics, Condenser *CND);
    virtual ~SCQP_correction_method();

    // condensed QP solution methods incorporating QP resolves with added corrections
    virtual int solve_SOC_QP(Matrix &deltaXi, Matrix &lambdaQP);
    virtual int bound_correction(Matrix &deltaXi_corr, Matrix &lambdaQP_corr);

    // filterLineSearch that calls modified SOC from above 
    virtual int filterLineSearch();
    // feasiblity restoration phase that calls the correction method.
    virtual int feasibilityRestorationPhase();
};

*/

} // namespace blockSQP

#endif
