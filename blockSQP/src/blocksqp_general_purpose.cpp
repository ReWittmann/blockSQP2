/*
 * blockSQP -- Sequential quadratic programming for problems with
 *             block-diagonal Hessian matrix.
 * Copyright (C) 2012-2015 by Dennis Janka <dennis.janka@iwr.uni-heidelberg.de>
 *
 * Licensed under the zlib license. See LICENSE for more details.
 */

/**
 * \file blocksqp_general_purpose.cpp
 * \author Dennis Janka
 * \date 2012-2015
 *
 *  Implementation of general purpose routines for matrix and vector computations.
 */

#include "blocksqp_general_purpose.hpp"

namespace blockSQP
{
/**
 * Compute the inverse of a matrix
 * using LU decomposition (DGETRF and DGETRI)
 */

/*
int inverse( const Matrix &A, Matrix &Ainv )
{
    int i, j;
    int n, ldim, lwork, info = 0;
    int *ipiv;
    double *work;

    for( i=0; i<A.N(); i++ )
        for( j=0; j<A.M(); j++ )
            Ainv( j,i ) = A( j,i );

    n = Ainv.N();
    ldim = Ainv.LDIM();
    ipiv = new int[n];
    lwork = n*n;
    work = new double[lwork];

    // Compute LU factorization
    dgetrf_( &n, &n, Ainv.ARRAY(), &ldim, ipiv, &info );
    if ( info != 0 )
        printf( "WARNING: DGETRF returned info=%i\n", info );
    // Compute inverse
    dgetri_( &n, Ainv.ARRAY(), &ldim, ipiv, work, &lwork, &info );
    if ( info != 0 )
        printf( "WARNING: DGETRI returned info=%i\n", info );

    return info;
}
*/

/**
 * Compute eigenvalues of a symmetric matrix by DSPEV
 */

 /*
int calcEigenvalues( const SymMatrix &B, Matrix &ev )
{
    int n;
    SymMatrix temp;
    double *work, *dummy = 0;
    int info, iDummy = 1;
    char JOB = 'N', UPLO = 'L';

    n = B.M();
    ev.Dimension( n ).Initialize( 0.0 );
    work = new double[3*n];

    // copy Matrix, will be overwritten
    temp = SymMatrix( B );

    dspev_( &JOB, &UPLO, &n, temp.ARRAY(), ev.ARRAY(), dummy, &iDummy,
            work, &info, 1, 1 );

    delete[] work;
    return info;
}
*/

/**
 * Estimate the smalles eigenvalue of a sqare matrix
 * with the help og Gershgorin's circle theorem
 */
double estimateSmallestEigenvalue( const Matrix &B )
{
    int i, j;
    double radius;
    int dim = B.M();
    double lambdaMin = 0.0;

    // For each row, sum up off-diagonal elements
    for (i = 0; i < dim; i++){
        radius = 0.0;
        for (j = 0; j < dim; j++)
            if (j != i)
                radius += std::abs( B( i,j ) );

        if (B(i,i) - radius < lambdaMin)
            lambdaMin = B(i,i) - radius;
    }
    return lambdaMin;
}


/**
 * Compute scalar product aTb
 */
double adotb( const Matrix &a, const Matrix &b )
{
    double norm = 0.0;

    if (a.n != 1) throw ParameterError("adotb: a is not a vector");
    else if (b.n != 1) throw ParameterError("adotb: b is not a vector");
    else if (a.m != b.m) throw ParameterError("adotb: a and b have different lengths");

    for( int k=0; k<a.M(); k++ )
        norm += a(k) * b(k);

    return norm;
}

/**
 * Compute the matrix vector product for a column-compressed sparse matrix A with a vector b and store it in result
 */
void Atimesb( double *Anz, int *AIndRow, int *AIndCol, const Matrix &b, Matrix &result )
{
    int nCol = b.M();
    int nRow = result.M();
    int i, k;

    for( i=0; i<nRow; i++ )
        result( i ) = 0.0;

    for( i=0; i<nCol; i++ )
    {
        // k runs over all elements in one column
        for( k=AIndCol[i]; k<AIndCol[i+1]; k++ )
            result( AIndRow[k] ) += Anz[k] * b( i );
    }

}

/**
 * Compute the matrix vector product A*b and store it in result
 */
void Atimesb( const Matrix &A, const Matrix &b, Matrix &result )
{
    result.Initialize( 0.0 );
    for( int i=0; i<A.M(); i++ )
        for( int k=0; k<A.N(); k++ )
            result( i ) += A( i, k ) * b( k );
}

double l1VectorNorm( const Matrix &v )
{
    double norm = 0.0;

    if( v.N() != 1 )
    {
        printf("v is not a vector!\n");
    }
    else
    {
        for( int k=0; k<v.M(); k++ )
            norm += std::abs(v( k ));
    }

    return norm;
}

double l2VectorNorm( const Matrix &v )
{
    double norm = 0.0;

    if( v.N() != 1 )
    {
        printf("v is not a vector!\n");
    }
    else
    {
        for( int k=0; k<v.M(); k++ )
            norm += v( k )* v( k );
    }

    return std::sqrt(norm);
}

double lInfVectorNorm( const Matrix &v )
{
    double norm = 0.0;

    if( v.N() != 1 )
    {
        printf("v is not a vector!\n");
    }
    else
    {
        for( int k=0; k<v.M(); k++ )
            if( fabs(v( k )) > norm )
                norm = fabs(v( k ));
    }

    return norm;
}


/**
 * Calculate weighted l1 norm of constraint violations
 */
double l1ConstraintNorm(const Matrix &xi, const Matrix &constr, const Matrix &lb_var, const Matrix &ub_var, const Matrix &lb_con, const Matrix &ub_con, const Matrix &weights)
{
    double norm = 0.0;
    int i;
    int nVar = xi.M();

    if( weights.M() < constr.M() + nVar )
    {
        printf("Weight vector too short!\n");
        return 0.0;
    }

    // Weighted violation of simple bounds
    for( i=0; i<nVar; i++ )
    {
        if( xi( i ) > ub_var( i ) )
            norm += fabs(weights( i )) * (xi( i ) - ub_var( i ));
        else if( xi( i ) < lb_var( i ) )
            norm += fabs(weights( i )) * (lb_var( i ) - xi( i ));
    }

    // Calculate weighted sum of constraint violations
    for( i=0; i<constr.M(); i++ )
    {
        if( constr( i ) > ub_con(i) )
            norm += fabs(weights( nVar+i )) * (constr( i ) - ub_con(i));
        else if( constr( i ) < lb_con( i ) )
            norm += fabs(weights( nVar+i )) * (lb_con( i ) - constr( i ));
    }

    return norm;
}


/**
 * Calculate l1 norm of constraint violations
 */
double l1ConstraintNorm( const Matrix &xi, const Matrix &constr, const Matrix &lb_var, const Matrix &ub_var, const Matrix &lb_con, const Matrix &ub_con )
{
    double norm = 0.0;
    int i;
    int nVar = xi.M();

    // Violation of simple bounds
    for( i=0; i<nVar; i++ )
    {
        if( xi( i ) > ub_var( i ) )
            norm += xi( i ) - ub_var( i );
        else if( xi( i ) < lb_var( i ) )
            norm += lb_var( i ) - xi( i );
    }

    // Calculate sum of constraint violations
    for( i=0; i<constr.M(); i++ )
    {
        if( constr( i ) > ub_con( i ) )
            norm += constr( i ) - ub_con( i );
        else if( constr( i ) < lb_con( i ) )
            norm += lb_con( i ) - constr( i );
    }

    return norm;
}


/**
 * Calculate l2 norm of constraint violations
 */
double l2ConstraintNorm( const Matrix &xi, const Matrix &constr, const Matrix &lb_var, const Matrix &ub_var, const Matrix &lb_con, const Matrix &ub_con )
{
    double norm = 0.0;
    int nVar = xi.M();

    // Violation of simple bounds
    for (int i = 0; i < nVar; i++){
        if( xi( i ) > ub_var( i ) )
            norm += xi( i ) - ub_var( i );
        if( xi( i ) < lb_var( i ) )
            norm += lb_var( i ) - xi( i );
    }

    // Calculate sum of constraint violations
    for (int i = 0; i < constr.M(); i++){
        if( constr( i ) > ub_con( i ) )
            norm += pow(constr( i ) - ub_con( i ), 2);
        else if( constr( i ) < lb_con( i ) )
            norm += pow(lb_con( i ) - constr( i ), 2);
    }

    return sqrt(norm);
}

/**
 * Calculate l_Infinity norm of constraint violations
 */
double lInfConstraintNorm(const Matrix &xi, const Matrix &constr, const Matrix &lb_var, const Matrix &ub_var, const Matrix &lb_con, const Matrix &ub_con){

    double norm = 0.0;
    int nVar = xi.M();
    int nCon = constr.M();

    // Violation of simple bounds
    for(int i = 0; i < nVar; i++){

        if (lb_var(i) - xi(i) > norm){
            norm = lb_var(i) - xi(i);
        }
        else if (xi(i) - ub_var(i) > norm){
            norm = xi(i) - ub_var(i);
        }
    }

    // Find out the largest constraint violation
    for(int i = 0; i < nCon; i++){

        if (lb_con(i) - constr(i) > norm){
            norm = lb_con(i) - constr(i);
        }
        else if (constr(i) - ub_con(i) > norm){
            norm = constr(i) - ub_con(i);
        }
    }

    return norm;
}

double lInfConstraintNorm(const Matrix &xi, const Matrix &constr, const Matrix &lb_var, const Matrix &ub_var, const Matrix &lb_con, const Matrix &ub_con, const Matrix &weights){

    double norm = 0.0;
    int nVar = xi.M();
    int nCon = constr.M();

    // Violation of simple bounds
    for (int i = 0; i < nVar; i++){
        if ((lb_var(i) - xi(i))*std::abs(weights(i)) > norm){
            norm = (lb_var(i) - xi(i))*std::abs(weights(i));
        }
        else if ((xi(i) - ub_var(i))*std::abs(weights(i)) > norm){
            norm = (xi(i) - ub_var(i))*std::abs(weights(i));
        }
    }

    // Find out the largest constraint violation
    for(int i = 0; i < nCon; i++){

        if ((lb_con(i) - constr(i))*std::abs(weights(nVar + i)) > norm){
            norm = (lb_con(i) - constr(i))*std::abs(weights(nVar + i));
        }
        else if ((constr(i) - ub_con(i))*std::abs(weights(nVar + i)) > norm){
            norm = (constr(i) - ub_con(i))*std::abs(weights(nVar + i));
        }
    }
    return norm;
}



void convertHessian(blockSQP::SymMatrix *const hess, int nBlocks, int nVar, double regularizationFactor,
                                            double *&hessNz){
    if (hessNz == NULL)
        hessNz = new double[nVar * nVar];

    int bsize, bstart = 0, ind = 0;
    //Iterate over hessian blocks
    for (int h = 0; h <nBlocks; h++){
        bsize = hess[h].m;
        //Iterate over second dimension
        for (int j = 0; j < bsize; j++){
            //Iterate over first dimension
             //Segment above hessian block
            for (int i = 0; i < bstart; i++){
                hessNz[ind] = 0;
                ++ind;
            }
             //Hessian block
            for (int i = 0; i < hess[h].m; i++){
                hessNz[ind] = hess[h](i, j);
                //NEW
                if (i == j) hessNz[ind] += regularizationFactor;

                ++ind;
            }
             //Segment below hessian block
            for (int i = bstart + bsize; i < nVar; i++){
                hessNz[ind] = 0;
                ++ind;
            }
        }
        bstart += bsize;
    }
    return;
}


void convertHessian(double eps, blockSQP::SymMatrix *const hess_, int nBlocks, int nVar, double regularizationFactor,
                             double *&hessNz_, int *&hessIndRow_, int *&hessIndCol_, int *&hessIndLo_ ){
    int iBlock, count, colCountTotal, rowOffset, i, j;
    int nnz, nCols, nRows;

    // 1) count nonzero elements
    nnz = 0;
    for (iBlock=0; iBlock<nBlocks; iBlock++){
        for (i=0; i<hess_[iBlock].m; i++){
            //Always count diagonal elements (regularization)
            if (fabs(hess_[iBlock](i,i)) > eps || fabs(hess_[iBlock](i,i)) + regularizationFactor > eps)
                nnz++;

            for (j = i + 1; j < hess_[iBlock].m; j++){
                if (fabs(hess_[iBlock]( i,j )) > eps)
                    nnz += 2;
            }
        }
    }

    delete[] hessNz_;
    delete[] hessIndRow_;
    delete[] hessIndCol_;
    delete[] hessIndLo_;

    hessNz_ = new double[nnz];
    hessIndRow_ = new int[nnz];
    hessIndCol_ = new int[nVar + 1];
    hessIndLo_ = new int[nVar];

    // 2) store matrix entries columnwise in hessNz
    count = 0; // runs over all nonzero elements
    colCountTotal = 0; // keep track of position in large matrix
    rowOffset = 0;
    for (iBlock = 0; iBlock < nBlocks; iBlock++){
        nCols = hess_[iBlock].m;
        nRows = hess_[iBlock].m;

        for (i = 0; i < nCols; i++){
            // column 'colCountTotal' starts at element 'count'
            hessIndCol_[colCountTotal] = count;

            for (j = 0; j < nRows; j++){
                //if (hess_[iBlock]( i,j ) > eps || -hess_[iBlock]( i,j ) > eps ){
                if (fabs(hess_[iBlock](i,j)) > eps || (i == j && fabs(hess_[iBlock](i,j)) + regularizationFactor > eps)){
                    hessNz_[count] = hess_[iBlock](i, j);
                    if (i == j) hessNz_[count] += regularizationFactor;

                    hessIndRow_[count] = j + rowOffset;
                    count++;
                }
            }
            colCountTotal++;
        }
        rowOffset += nRows;
    }
    hessIndCol_[colCountTotal] = count;

    // 3) Set reference to lower triangular matrix
    for( j=0; j<nVar; j++ )
    {
        for( i=hessIndCol_[j]; i<hessIndCol_[j+1] && hessIndRow_[i]<j; i++);
        hessIndLo_[j] = i;
    }

    if( count != nnz ){
         std::cout << "Error in convertHessian: " << count << " elements processed, should be " << nnz << " elements!\n";
    }
}

} // namespace blockSQP
