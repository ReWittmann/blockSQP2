/*
 * blockSQP -- Sequential quadratic programming for problems with
 *             block-diagonal Hessian matrix.
 * Copyright (C) 2012-2015 by Dennis Janka <dennis.janka@iwr.uni-heidelberg.de>
 *
 * Licensed under the zlib license. See LICENSE for more details.
 */

/**
 * \file blocksqp_problemspec.hpp
 * \author Dennis Janka
 * \date 2012-2015
 *
 *  Declaration of ProblemSpec class to describe an NLP to be solved by blockSQP.
 */

#ifndef BLOCKSQP_PROBLEMSPEC_HPP
#define BLOCKSQP_PROBLEMSPEC_HPP

#include "blocksqp_defs.hpp"
#include "blocksqp_matrix.hpp"
#include "blocksqp_condensing.hpp"
#include <limits>
#include <memory>

namespace blockSQP{

/**
 * \brief Base class for problem specification as required by SQPmethod.
 * \author Dennis Janka
 * \date 2012-2015
 */
class Problemspec{
    /*
     * VARIABLES
     */
    public:
        int         nVar = -1;                                          ///< number of variables
        int         nCon = -1;                                          ///< number of constraints
        int         nnz = -1;                                           ///< number of structural nonzero entries of sparse constraint jacobian

        double      objLo = std::numeric_limits<double>::infinity();    ///< lower bound for objective
        double      objUp = std::numeric_limits<double>::infinity();    ///< upper bound for objective
        Matrix      lb_var;                                             ///< lower bounds of variables and constraints
        Matrix      ub_var;                                             ///< upper bounds of variables and constraints
        Matrix      lb_con;             
        Matrix      ub_con;             
        
        //Metadata
        int         nBlocks = -1;                                       ///< number of separable blocks of Lagrangian
        int*        blockIdx = nullptr;                                 ///< [blockwise] index in the variable vector where a block starts

        int         n_vblocks = -1;                                     ///< number of distinct variable blocks of variables
        vblock      *vblocks = nullptr;                                 ///< variable blocks, containing structure information (free/dependent, ...)
        
    /*
     * METHODS
     */
    public:
        Problemspec();
        virtual ~Problemspec();

        /// Set initial values for xi (and possibly lambda) and parts of the Jacobian that correspond to linear constraints (dense version).
        virtual void initialize( Matrix &xi,            ///< optimization variables
                                 Matrix &lambda,        ///< Lagrange multipliers
                                 Matrix &constrJac      ///< constraint Jacobian (dense)
                                 ){};

        /// Set initial values for xi (and possibly lambda) and parts of the Jacobian that correspond to linear constraints (sparse version).
        virtual void initialize( Matrix &xi,            ///< optimization variables
                                 Matrix &lambda,        ///< Lagrange multipliers
                                 double *jacNz,        ///< nonzero elements of constraint Jacobian
                                 int *jacIndRow,       ///< row indices of nonzero elements
                                 int *jacIndCol        ///< starting indices of columns
                                 ){};

        /// Evaluate objective, constraints, and derivatives (dense version).
        virtual void evaluate( const Matrix &xi,        ///< optimization variables
                               const Matrix &lambda,    ///< Lagrange multipliers
                               double *objval,          ///< objective function value
                               Matrix &constr,          ///< constraint function values
                               Matrix &gradObj,         ///< gradient of objective
                               Matrix &constrJac,       ///< constraint Jacobian (dense)
                               SymMatrix *hess,        ///< Hessian of the Lagrangian (blockwise)
                               int dmode,               ///< derivative mode
                               int *info                ///< error flag
                               ){};

        /// Evaluate objective, constraints, and derivatives (sparse version).
        virtual void evaluate( const Matrix &xi,        ///< optimization variables
                               const Matrix &lambda,    ///< Lagrange multipliers
                               double *objval,          ///< objective function value
                               Matrix &constr,          ///< constraint function values
                               Matrix &gradObj,         ///< gradient of objective
                               double *jacNz,          ///< nonzero elements of constraint Jacobian
                               int *jacIndRow,         ///< row indices of nonzero elements
                               int *jacIndCol,         ///< starting indices of columns
                               SymMatrix *hess,        ///< Hessian of the Lagrangian (blockwise)
                               int dmode,               ///< derivative mode
                               int *info                ///< error flag
                               ){};

        /// Short cut if no derivatives are needed
        virtual void evaluate( const Matrix &xi,        ///< optimization variables
                               double *objval,          ///< objective function value
                               Matrix &constr,          ///< constraint function values
                               int *info                ///< error flag
                               );

        /*
         * Optional Methods
         */
        /// Problem specific heuristic to reduce constraint violation
        virtual void reduceConstrVio( Matrix &xi,       ///< optimization variables
                                      int *info         ///< error flag
                                      ){ *info = 1; };

        /// Print information about the current problem
        virtual void printInfo(){};
};

class scaled_Problemspec: public Problemspec{
public:

    Problemspec *unscaled_prob;

    std::unique_ptr<double[]> scaling_factors;
    Matrix xi_unscaled;

    scaled_Problemspec(Problemspec *UNSCprob);
    ~scaled_Problemspec();

    //Set scaling factors
    void set_scale(const double *const scaleFacs);
    //Apply scaling factors, multiplies to current scaling factors
    void rescale(const double *const scaleFacs);


    void initialize(Matrix &xi, Matrix &lambda, Matrix &constrJac);
    void initialize(Matrix &xi, Matrix &lambda, double *jacNz, int *jacIndRow, int *jacIndCol);

    void evaluate(const Matrix &xi, const Matrix &lambda, double *objval, Matrix &constr, Matrix &gradObj, Matrix &constrJac, SymMatrix *hess, int dmode, int *info);
    void evaluate(const Matrix &xi, const Matrix &lambda, double *objval, Matrix &constr, Matrix &gradObj, double *jacNz, int *jacIndRow, int *jacIndCol, SymMatrix *hess, int dmode, int *info);
    void evaluate(const Matrix &xi, double *objval, Matrix &constr, int *info);

    void reduceConstrVio(Matrix &xi, int* info);


};



} // namespace blockSQP

#endif

