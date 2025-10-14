/*
 * blockSQP -- Sequential quadratic programming for problems with
 *             block-diagonal Hessian matrix.
 * Copyright (C) 2012-2015 by Dennis Janka <dennis.janka@iwr.uni-heidelberg.de>
 *
 * Licensed under the zlib license. See LICENSE for more details.
 */

/*
 * blockSQP extensions -- Extensions and modifications for the 
                          blockSQP nonlinear solver by Dennis Janka
 * Copyright (C) 2023-2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
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
 * 
 * \modifications
 *  \author Reinhold Wittmann
 *  \date 2023-2025
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

// Base for restoration problem, requires methods to restore original iterates from restoration iterates. 
// Allows variable layouts for original and slack variables
class abstractRestorationProblem : public Problemspec{
    public:
    Problemspec *parent;
    //Update the reference point if it enters into the restoration problem
    virtual void update_xi_ref(const Matrix &xiReference);

    //Recover the Lagrange multipliers for the original problem from a dual iterate of the restoration problem
    virtual void recover_lambda(const Matrix &lambda_rest, Matrix &lambda_orig) = 0;
            
    //Recover the variables from the original problem from a primal iterate of the restoration problem 
    virtual void recover_xi(const Matrix &xi_rest, Matrix &xi_orig) = 0;
};


class RestorationProblem : public abstractRestorationProblem{
    public:
        Matrix xi_ref;
        Matrix diagScale;

        //Submatrix for parent evaluation
        Matrix xi_parent;
        //Submatrix containing slack variables
        Matrix slack;

        double rho;
        double zeta;

        //double *jacNzOrig;
        //int *jacIndRowOrig;
        //int *jacIndColOrig;

    public:
        RestorationProblem( Problemspec *parentProblem, const Matrix &xiReference, double param_rho, double param_zeta);
        virtual ~RestorationProblem();

        /// Set initial values for xi and lambda, may also set matrix for linear constraints (dense version)
        virtual void initialize(Matrix &xi, Matrix &lambda, Matrix &constrJac);

        /// Set initial values for xi and lambda, may also set matrix for linear constraints (sparse version)
        virtual void initialize(Matrix &xi, Matrix &lambda, double *jacNz, int *jacIndRow, int *jacIndCol);

        /// Evaluate all problem functions and their derivatives (dense version)
        virtual void evaluate(const Matrix &xi, const Matrix &lambda,
                              double *objval, Matrix &constr,
                              Matrix &gradObj, Matrix &constrJac,
                              SymMatrix *hess, int dmode, int *info);

        /// Evaluate all problem functions and their derivatives (sparse version)
        virtual void evaluate(const Matrix &xi, const Matrix &lambda,
                              double *objval, Matrix &constr,
                              Matrix &gradObj, double *jacNz, int *jacIndRow, int *jacIndCol,
                              SymMatrix *hess, int dmode, int *info);

        virtual void printInfo();
        virtual void printVariables( const Matrix &xi, const Matrix &lambda, int verbose );
        virtual void printConstraints( const Matrix &constr, const Matrix &lambda );


        virtual void update_xi_ref(const Matrix &xiReference);

        //Recover the variables from the original problem from a primal iterate of the restoration problem 
        virtual void recover_xi(const Matrix &xi_rest, Matrix &xi_orig);

        //Recover the Lagrange multipliers for the original problem from a dual iterate of the restoration problem
        virtual void recover_lambda(const Matrix &lambda_rest, Matrix &lambda_orig);
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


class TC_restoration_Problem: public abstractRestorationProblem{
    public:
    Condenser *parent_cond;
    Matrix xi_ref;
    Matrix diagScale;

    //Submatrix for parent evaluation
    Matrix xi_parent;
    //Submatrix containing slack variables
    Matrix slack;

    double rho;
    double zeta;

    Matrix constr_orig;
    
    double *jac_orig_nz = nullptr;
    int *jac_orig_row = nullptr;
    int *jac_orig_colind = nullptr;

    public:
    TC_restoration_Problem(Problemspec *parent_Problem, Condenser *parent_CND, const Matrix &xi_Reference, double param_rho, double param_zeta);
    virtual ~TC_restoration_Problem();

    void update_xi_ref(const Matrix &xiReference);

    /// Set initial values for xi and lambda, may also set matrix for linear constraints (sparse version)
    virtual void initialize(Matrix &xi, Matrix &lambda, double *jacNz, int *jacIndRow, int *jacIndCol);

    /// Evaluate all problem functions and their derivatives (sparse version)
    virtual void evaluate(const Matrix &xi, const Matrix &lambda,
                           double *objval, Matrix &constr,
                           Matrix &gradObj, double *jacNz, int *jacIndRow, int *jacIndCol,
                           SymMatrix *hess, int dmode, int *info);

    virtual void reduceConstrVio(Matrix &xi, int *info);

    virtual void recover_xi(const Matrix &xi_rest, Matrix &xi_orig);
    virtual void recover_lambda(const Matrix &lambda_rest, Matrix &lambda_orig);

    //void recover_multipliers(const Matrix &lambda_rest, Matrix &lambda_orig);
    //void recover_multipliers(const Matrix &lambda_rest, Matrix &lambda_orig, double &lambda_step_norm);
};

//Utility method to create a condenser for TC_restoration/feasibility_problem from a condenser for the parent problem. 
holding_Condenser* create_restoration_Condenser(Condenser *parent, int DEP_BOUNDS = 0);


class TC_feasibility_Problem: public Problemspec{
public:
    Problemspec *parent;
    Condenser *parent_cond;
    Matrix xi_parent;
    Matrix slack;
    Matrix constr_orig;
    double *jac_orig_nz = nullptr;
    int *jac_orig_row = nullptr;
    int *jac_orig_colind = nullptr;

    TC_feasibility_Problem(Problemspec *parent_Problem, Condenser *parent_CND);
    virtual ~TC_feasibility_Problem();

    virtual void initialize(Matrix &xi, Matrix &lambda, double *jacNz, int *jacIndRow, int *jacIndCol);

    virtual void evaluate(const Matrix &xi, const Matrix &lambda,
                            double *objval, Matrix &constr,
                           Matrix &gradObj, double *jacNz, int *jacIndRow, int *jacIndCol,
                           SymMatrix *hess, int dmode, int *info);

    //void build_restoration_jacobian(const Sparse_Matrix &jac_orig, Sparse_Matrix &jac_restoration);
    //void recover_multipliers(const Matrix &lambda_rest, Matrix &lambda_orig);
    virtual void reduceConstrVio(Matrix &xi, int *info);
};



} // namespace blockSQP

#endif

