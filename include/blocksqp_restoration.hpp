/*
 * blockSQP -- Sequential quadratic programming for problems with
 *             block-diagonal Hessian matrix.
 * Copyright (C) 2012-2015 by Dennis Janka <dennis.janka@iwr.uni-heidelberg.de>
 *
 * Licensed under the zlib license. See LICENSE for more details.
 */

/**
 * \file blocksqp_restoration.hpp
 * \author Dennis Janka
 * \date 2012-2015
 *
 *  Declaration of RestorationProblem class that describes a
 *  minimum l_2-norm NLP.
 */

#ifndef BLOCKSQP_RESTORATION_HPP
#define BLOCKSQP_RESTORATION_HPP

#include "blocksqp_defs.hpp"
#include "blocksqp_problemspec.hpp"

namespace blockSQP
{

/**
 * \brief Describes a minimum l_2-norm NLP for a given parent problem
 *        that is solved during the feasibility restoration phase.
 * \author Dennis Janka
 * \date 2012-2015
 */
class RestorationProblem : public Problemspec
{
    /*
     * CLASS VARIABLES
     */
    public:
        Problemspec *parent;
        Matrix xiRef;
        Matrix diagScale;
        //int neq;
        //bool *isEqCon;

        double zeta;
        double rho;

        double *jacNzOrig;
        int *jacIndRowOrig;
        int *jacIndColOrig;

    /*
     * METHODS
     */
    public:
        RestorationProblem( Problemspec *parentProblem, const Matrix &xiReference);
        virtual ~RestorationProblem();

        /// Set initial values for xi and lambda, may also set matrix for linear constraints (dense version)
        virtual void initialize( Matrix &xi, Matrix &lambda, Matrix &constrJac );

        /// Set initial values for xi and lambda, may also set matrix for linear constraints (sparse version)
        virtual void initialize( Matrix &xi, Matrix &lambda, double *&jacNz, int *&jacIndRow, int *&jacIndCol );

        /// Evaluate all problem functions and their derivatives (dense version)
        virtual void evaluate( const Matrix &xi, const Matrix &lambda,
                               double *objval, Matrix &constr,
                               Matrix &gradObj, Matrix &constrJac,
                               SymMatrix *&hess, int dmode, int *info );

        /// Evaluate all problem functions and their derivatives (sparse version)
        virtual void evaluate( const Matrix &xi, const Matrix &lambda,
                               double *objval, Matrix &constr,
                               Matrix &gradObj, double *&jacNz, int *&jacIndRow, int *&jacIndCol,
                               SymMatrix *&hess, int dmode, int *info );

        virtual void printInfo();
        virtual void printVariables( const Matrix &xi, const Matrix &lambda, int verbose );
        virtual void printConstraints( const Matrix &constr, const Matrix &lambda );
};

/*
class condensable_Restoration_Problem: public Problemspec{
public:
    Problemspec *parent;
    Condenser *parent_cond;

    Matrix xi_ref;
    Matrix diagScale;

    double zeta;
    double rho;

    Matrix constr_orig;
    double *jac_orig_nz = nullptr;
    int *jac_orig_row = nullptr;
    int *jac_orig_colind = nullptr;



public:
    condensable_Restoration_Problem(Problemspec *parent_Problem, Condenser *parent_cond, const Matrix &xi_Reference);
    ~condensable_Restoration_Problem();

    /// Set initial values for xi and lambda, may also set matrix for linear constraints (sparse version)
    virtual void initialize(Matrix &xi, Matrix &lambda, double *&jacNz, int *&jacIndRow, int *&jacIndCol);

    /// Evaluate all problem functions and their derivatives (sparse version)
    virtual void evaluate(const Matrix &xi, const Matrix &lambda,
                           double *objval, Matrix &constr,
                           Matrix &gradObj, double *&jacNz, int *&jacIndRow, int *&jacIndCol,
                           SymMatrix *&hess, int dmode, int *info);


    void build_restoration_jacobian(const Sparse_Matrix &jac_orig, Sparse_Matrix &jac_restoration);
    void recover_multipliers(const Matrix &lambda_rest, Matrix &lambda_orig);
    void recover_multipliers(const Matrix &lambda_rest, Matrix &lambda_orig, double &lambda_step_norm);

};


class feasibility_Problem: public Problemspec{
public:
    Problemspec *parent;
    Condenser *parent_cond;
    Matrix constr_orig;
    double *jac_orig_nz = nullptr;
    int *jac_orig_row = nullptr;
    int *jac_orig_colind = nullptr;

    feasibility_Problem(Problemspec *parent_Problem, Condenser *parent_cond);
    ~feasibility_Problem();

    virtual void initialize(Matrix &xi, Matrix &lambda, double *&jacNz, int *&jacIndRow, int *&jacIndCol);

    virtual void evaluate(const Matrix &xi, const Matrix &lambda,
                            double *objval, Matrix &constr,
                           Matrix &gradObj, double *&jacNz, int *&jacIndRow, int *&jacIndCol,
                           SymMatrix *&hess, int dmode, int *info);

    void build_restoration_jacobian(const Sparse_Matrix &jac_orig, Sparse_Matrix &jac_restoration);
    void recover_multipliers(const Matrix &lambda_rest, Matrix &lambda_orig);

};
*/


/* True-Constraint restoration Problem:
    Solve an NLP min square(||s||_2) + square(||Su||_2)
        s.t. lb_g <= g(x,u) - s <= ub_g
             C(x,u) = 0

    where x are dependent variables (states in multiple-shooting),
    u are free variables, and C(x,u) = 0 are x - defining conditions
    (continuity conditions in multiple-shooting)

    Only "true" constraints are relaxed.
*/


class TC_restoration_Problem: public Problemspec{
public:
    Problemspec *parent;
    Condenser *parent_cond;
    Matrix xi_ref;
    Matrix diagScale;

    Matrix xi_orig;
    Matrix slack;

    double zeta;
    double rho;

    Matrix constr_orig;
    double *jac_orig_nz = nullptr;
    int *jac_orig_row = nullptr;
    int *jac_orig_colind = nullptr;



public:
    TC_restoration_Problem(Problemspec *parent_Problem, Condenser *parent_CND, const Matrix &xi_Reference);
    virtual ~TC_restoration_Problem();

    /// Set initial values for xi and lambda, may also set matrix for linear constraints (sparse version)
    virtual void initialize(Matrix &xi, Matrix &lambda, double *&jacNz, int *&jacIndRow, int *&jacIndCol);

    /// Evaluate all problem functions and their derivatives (sparse version)
    virtual void evaluate(const Matrix &xi, const Matrix &lambda,
                           double *objval, Matrix &constr,
                           Matrix &gradObj, double *&jacNz, int *&jacIndRow, int *&jacIndCol,
                           SymMatrix *&hess, int dmode, int *info);

    virtual void reduceConstrVio(Matrix &xi, int *info);

    void recover_multipliers(const Matrix &lambda_rest, Matrix &lambda_orig);
    void recover_multipliers(const Matrix &lambda_rest, Matrix &lambda_orig, double &lambda_step_norm);
};


class TC_feasibility_Problem: public Problemspec{
public:
    Problemspec *parent;
    Condenser *parent_cond;
    Matrix xi_orig;
    Matrix slack;
    Matrix constr_orig;
    double *jac_orig_nz = nullptr;
    int *jac_orig_row = nullptr;
    int *jac_orig_colind = nullptr;

    TC_feasibility_Problem(Problemspec *parent_Problem, Condenser *parent_CND);
    virtual ~TC_feasibility_Problem();

    virtual void initialize(Matrix &xi, Matrix &lambda, double *&jacNz, int *&jacIndRow, int *&jacIndCol);

    virtual void evaluate(const Matrix &xi, const Matrix &lambda,
                            double *objval, Matrix &constr,
                           Matrix &gradObj, double *&jacNz, int *&jacIndRow, int *&jacIndCol,
                           SymMatrix *&hess, int dmode, int *info);

    //void build_restoration_jacobian(const Sparse_Matrix &jac_orig, Sparse_Matrix &jac_restoration);
    //void recover_multipliers(const Matrix &lambda_rest, Matrix &lambda_orig);
    virtual void reduceConstrVio(Matrix &xi, int *info);
};



} // namespace blockSQP

#endif

