/*
 * blockSQP -- Sequential quadratic programming for problems with
 *             block-diagonal Hessian matrix.
 * Copyright (C) 2012-2015 by Dennis Janka <dennis.janka@iwr.uni-heidelberg.de>
 *
 * Licensed under the zlib license. See LICENSE for more details.
 */

/*
 * blockSQP2 -- A structure-exploiting nonlinear programming solver based
 *              on blockSQP by Dennis Janka.
 * Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
 * 
 * Licensed under the zlib license. See LICENSE for more details.
 */
 
 
/**
 * \file blocksqp_general_purpose.hpp
 * \author Dennis Janka
 * \date 2012-2015
 *
 *  Declaration of general purpose routines for matrix and vector computations.
 * 
 * \modifications
 *  \author Reinhold Wittmann
 *  \date 2023-2025
 */

#ifndef BLOCKSQP2_GENERAL_PURPOSE_HPP
#define BLOCKSQP2_GENERAL_PURPOSE_HPP

#include <blockSQP2/defs.hpp>
#include <blockSQP2/matrix.hpp>

namespace blockSQP2{

double l1VectorNorm( const Matrix &v );
double l2VectorNorm( const Matrix &v );
double lInfVectorNorm( const Matrix &v );

double l1ConstraintNorm( const Matrix &xi, const Matrix &constr, const Matrix &lb_var, const Matrix &ub_var, const Matrix &lb_con, const Matrix &ub_con, const Matrix &weights );
double l1ConstraintNorm( const Matrix &xi, const Matrix &constr, const Matrix &lb_var, const Matrix &ub_var, const Matrix &lb_con, const Matrix &ub_con );
double l2ConstraintNorm( const Matrix &xi, const Matrix &constr, const Matrix &lb_var, const Matrix &ub_var, const Matrix &lb_con, const Matrix &ub_con );
double lInfConstraintNorm( const Matrix &xi, const Matrix &constr, const Matrix &lb_var, const Matrix &ub_var, const Matrix &lb_con, const Matrix &ub_con );
double lInfConstraintNorm( const Matrix &xi, const Matrix &constr, const Matrix &lb_var, const Matrix &ub_var, const Matrix &lb_con, const Matrix &ub_con, const Matrix &weights);

double adotb( const Matrix &a, const Matrix &b );
void Atimesb( const Matrix &A, const Matrix &b, Matrix &result );
void Atimesb( double *Anz, int *AIndRow, int *AIndCol, const Matrix &b, Matrix &result );

//int calcEigenvalues( const Matrix &B, Matrix &ev );
double estimateSmallestEigenvalue( const Matrix &B );
//int inverse( const Matrix &A, Matrix &Ainv );

//Convert block Hessian to sparse 
void convertHessian(SymMatrix *const hess, int nBlocks, int nVar, double regularizationFactor,
    double *&hessNz);
void convertHessian(double eps, SymMatrix *const hess_, int nBlocks, int nVar, double regularizationFactor,
    double *&hessNz_, int *&hessIndRow_, int *&hessIndCol_, int *&hessIndLo_);

//Convert block Hessian to sparse, assume sufficient memory has been allocated to hessNz
void convertHessian_noalloc(SymMatrix *const hess, int nBlocks, int nVar, double regularizationFactor,
    double *hessNz);

void convertHessian_noalloc(double eps, SymMatrix *const hess_, int nBlocks, int nVar, double regularizationFactor,
    double *hessNz_, int *hessIndRow_, int *hessIndCol_, int *hessIndLo_);
    

} // namespace blockSQP2

#endif
