/*
 * blockSQP -- Sequential quadratic programming for problems with
 *             block-diagonal Hessian matrix.
 * Copyright (C) 2012-2015 by Dennis Janka <dennis.janka@iwr.uni-heidelberg.de>
 *
 * Licensed under the zlib license. See LICENSE for more details.
 */

/**
 * \file blocksqp_matrix.cpp
 * \author Dennis Janka, based on VPLAN's matrix.h by Stefan Koerkel
 * \date 2012-2015
 *
 *  Implementation of Matrix and SymMatrix classes for easy access of
 *  matrix elements.
 */

#include "blocksqp_matrix.hpp"
#include <stdexcept>
#include <vector>
#include <iostream>
#include <string>
#include <chrono>
#ifdef WINDOWS
    #include "mkl.h"
#else
    #include "cblas.h"
#endif
#define MATRIX_DEBUG

namespace blockSQP
{

//#define MATRIX_DEBUG

void Error( const char *F )
{
    printf("Error: %s\n", F );
    //exit( 1 );
}

/* ----------------------------------------------------------------------- */

int Ccount = 0;
int Dcount = 0;
int Ecount = 0;

/* ----------------------------------------------------------------------- */

int Matrix::malloc( void )
{
    int len;

    if ( tflag )
        Error("malloc cannot be called with Submatrix");

    if ( ldim < m )
        ldim = m;

    len = ldim*n;

    if ( len == 0 ){
        array = NULL;
        }
    else{
        if ( ( array = new double[len] ) == NULL ){
            Error("'new' failed");
            }
        }
    return 0;
}


int Matrix::free( void )
{
    if ( tflag )
        Error("free cannot be called with Submatrix");

    if ( array != NULL )
        delete[] array;

    return 0;
}


double &Matrix::operator()( int i, int j )
{   /*
    #ifdef MATRIX_DEBUG
    if ( i < 0 || i >= m || j < 0 || j >= n )
        Error("Invalid matrix entry");
    #endif*/

    #ifdef MATRIX_DEBUG
    if ( i < 0 || i >= m || j < 0 || j >= n){
        throw std::invalid_argument("Matrix operator(): Indices (" + std::to_string(i) + ", " + std::to_string(j) + ") out of bounds for matrix of shape (" + std::to_string(m) + ", " + std::to_string(n) + ")");
    }
    #endif

    return array[i+j*ldim];
}

double &Matrix::operator()( int i, int j ) const
{   /*
    #ifdef MATRIX_DEBUG
    if ( i < 0 || i >= m || j < 0 || j >= n )
        Error("Invalid matrix entry");
    #endif*/

    #ifdef MATRIX_DEBUG
    if ( i < 0 || i >= m || j < 0 || j >= n){
        throw std::invalid_argument("Matrix operator(): Indices (" + std::to_string(i) + ", " + std::to_string(j) + ") out of bounds for matrix of shape (" + std::to_string(m) + ", " + std::to_string(n) + ")");
    }
    #endif

    return array[i+j*ldim];
}

double &Matrix::operator()( int i )
{
    #ifdef MATRIX_DEBUG
    if ( i < 0 || i >= m ){
        throw std::invalid_argument("Matrix operator(): Index " + std::to_string(i) + " out of bounds for matrix of shape (" + std::to_string(m) + ", " + std::to_string(n) + ")");
    }
    #endif

    return array[i];
}

double &Matrix::operator()( int i ) const
{
    #ifdef MATRIX_DEBUG
    if ( i < 0 || i >= m ){
        throw std::invalid_argument("Matrix operator(): Index " + std::to_string(i) + " out of bounds for matrix of shape (" + std::to_string(m) + ", " + std::to_string(n) + ")");
    }
    #endif

    return array[i];
}


Matrix::Matrix(){
    array = nullptr;
    m = 0;
    n = 0;
    ldim = 0;
    tflag = 0;
}

Matrix::Matrix( int M, int N, int LDIM )
{
    Ccount++;

    m = M;
    n = N;
    ldim = LDIM;
    tflag = 0;

    malloc();
}


Matrix::Matrix( int M, int N, double *ARRAY, int LDIM )
{
    Ccount++;

    m = M;
    n = N;
    array = ARRAY;
    ldim = LDIM;
    tflag = 0;

    if ( ldim < m )
        ldim = m;
}


Matrix::Matrix( const Matrix &A )
{
    int i, j;
    //printf("copy constructor\n");
    Ccount++;

    m = A.m;
    n = A.n;
    ldim = A.ldim;
    tflag = 0;

    malloc();

    for ( i = 0; i < m; i++ )
        for ( j = 0; j < n ; j++ )
            (*this)(i,j) = A(i,j);
            //(*this)(i,j) = A.a(i,j);
}

Matrix::Matrix(const SymMatrix &A){
    Ccount++;
    m = A.m;
    n = A.m;
    ldim = m;
    tflag = 0;
    malloc();
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            (*this)(i,j) = A(i,j);
        }
    }
    return;
}

/*
Matrix::Matrix(Matrix&& M){
    m = M.m;
    n = M.n;
    ldim = M.ldim;
    array = M.array;
    tflag = M.tflag;

    M.m = 0;
    M.n = 0;
    M.ldim = 0;
    M.array = nullptr;
    M.tflag = 0;
}*/

Matrix &Matrix::operator=( const Matrix &A )
{
    int i, j;
    //printf("assignment operator\n");
    Ecount++;

    if ( this != &A )
    {
        if ( !tflag )
        {

            free();

            m = A.m;
            n = A.n;
            ldim = A.m;

            malloc();

            for ( i = 0; i < m; i++ )
                for ( j = 0; j < n ; j++ )
                    (*this)(i,j) = A(i,j);
        }
        else
        {
            if ( m != A.m || n != A.n )
                throw std::invalid_argument(std::string("Matrix::operator=: Cannot assign matrix of size (") + std::to_string(A.m) + "," + std::to_string(A.n) + ") to submatrix of size (" + std::to_string(m) + ", " + std::to_string(n) + ")");

            for ( i = 0; i < m; i++ )
                for ( j = 0; j < n ; j++ )
                    (*this)(i,j) = A(i,j);
        }
    }

    return *this;
}


Matrix &Matrix::operator=(const SymMatrix &A){
    Ecount++;

    if (!tflag){
        free();

        m = A.m;
        n = A.m;
        ldim = m;

        malloc();
        for (int i = 0; i < m; i++){
            for (int j = 0; j < n ; j++){
                (*this)(i,j) = A(i,j);
            }
        }
    }
    else{
        if (m != A.m || n != A.m)
            Error("= operation not allowed");

        for (int i = 0; i < m; i++){
            for (int j = 0; j < n ; j++){
                (*this)(i,j) = A(i,j);
            }
        }
    }
    return *this;
}


/*
void Matrix::operator=(Matrix &&A){
    if (tflag){
        //invoke copy constructor
        (*this) = A;
    }

    if (m != A.m || n != A.n || ldim != A.ldim)
        free();
        m = A.m;
        n = A.n;
        ldim = A.ldim;

    m = A.m;
    n = A.n;
    ldim = A.ldim;
    array = A.array;

    A.m = 0;
    A.n = 0;
    A.ldim = 0;
    A.array = nullptr;

    return;
}*/

Matrix::~Matrix( void )
{
    Dcount++;
    if ( !tflag )
        free();
}

/* ----------------------------------------------------------------------- */

int Matrix::M( void ) const
{   return m;
}


int Matrix::N( void ) const
{   return n;
}


int Matrix::LDIM( void ) const
{   return ldim;
}


double *Matrix::ARRAY( void ) const
{   return array;
}

double *Matrix::release(){
    if (tflag) throw std::runtime_error("Submatrix cannot release it's memory.");
    double *ret = array;
    array = nullptr;
    m = 0; n = 0; ldim = 0; tflag = 0;
    return ret;
}

int Matrix::TFLAG( void ) const
{   return tflag;
}

/* ----------------------------------------------------------------------- */

Matrix &Matrix::Dimension( int M, int N, int LDIM )
{
    if ( M != m || N != n || ( LDIM != ldim && LDIM != -1 ) )
    {
        if ( tflag )
            Error("Cannot set new dimension for Submatrix");
        else
        {
            free();
            m = M;
            n = N;
            ldim = std::max(LDIM, m);
            malloc();
        }
    }

    return *this;
}


Matrix &Matrix::Initialize( double (*f)( int, int ) )
{
    int i, j;

    for ( i = 0; i < m; i++ )
        for ( j = 0; j < n; j++ )
            (*this)(i,j) = f(i,j);

    return *this;
}


Matrix &Matrix::Initialize( double val )
{
    int i, j;

    for ( i = 0; i < m; i++ )
        for ( j = 0; j < n; j++ )
            (*this)(i,j) = val;

    return *this;
}



/* ----------------------------------------------------------------------- */

Matrix &Matrix::Submatrix( const Matrix &A, int M, int N, int i0, int j0 )
{
    if ( i0 + M > A.m || j0 + N > A.n )
        Error("Cannot create Submatrix");

    if ( !tflag )
        free();

    tflag = 1;

    m = M;
    n = N;
    array = &A.array[i0+j0*A.ldim];
    ldim = A.ldim;

    return *this;
}


Matrix &Matrix::Arraymatrix( int M, int N, double *ARRAY, int LDIM )
{
    if ( !tflag )
        free();

    tflag = 1;

    m = M;
    n = N;
    array = ARRAY;
    ldim = LDIM;

    if ( ldim < m )
        ldim = m;

    return *this;
}


Matrix Matrix::get_slice(int m_start, int m_end, int n_start, int n_end) const{
	if (m_end < m_start || n_end < n_start || m_end > m || n_end > n){
		throw std::invalid_argument("Matrix.get_slice: Slices (" + std::to_string(m_start) + ", " + std::to_string(m_end) + "), (" + std::to_string(n_start) + ", " + std::to_string(n_end) + ") invalid for matrix of shape (" + std::to_string(m) + ", " + std::to_string(n));
	}

	int M_slc = m_end - m_start;
	int N_slc = n_end - n_start;
	double *array_slc = new double[M_slc * N_slc];
	for (int j = 0; j < N_slc; j++){
		for (int i = 0; i < M_slc; i++){
			array_slc[i + j*M_slc] = (*this)(m_start + i, n_start + j);
		}
	}
	return Matrix(M_slc, N_slc, array_slc);
}


Matrix Matrix::get_slice(int m_start, int m_end) const{
    #ifdef MATRIX_DEBUG
	if (m_end < m_start || m_end > m){
		throw std::invalid_argument("Matrix.get_slice: Slice (" + std::to_string(m_start) + ", " + std::to_string(m_end) + ") out of matrix bounds (" + std::to_string(0) + ", " + std::to_string(m) + ")");
	}
	#endif

	int M_slc = m_end - m_start;
	double *array_slc = new double[M_slc * n];
	for (int j = 0; j < n; j++){
		for (int i = 0; i < M_slc; i++){
			array_slc[i + j*M_slc] = (*this)(m_start + i, j);
		}
	}
	return Matrix(M_slc, n, array_slc);
}


Matrix Matrix::without_rows(int *starts, int *ends, int n_slices) const{
    int M = m;

    #ifdef MATRIX_DEBUG
    for (int l = 0; l<n_slices-1; l++){
		if (starts[l] > ends[l] || ends[l]>starts[l+1]){
			throw std::invalid_argument( "slice end must be greater than start and slices must be non-overlapping and sorted by starting index" );
		}
	}
	if (starts[n_slices - 1] > ends[n_slices - 1]){
		throw std::invalid_argument( "slice end must be greater than start and slices must be non-overlapping and sorted by starting index" );
	}
    #endif

    for (int snum = 0; snum<n_slices; snum++){
        M -= ends[snum] - starts[snum];
    }

    double *r_array = new double[M*n];

    int ind_1;
    int ind_2;
    for (int j = 0; j < n; j++){
        ind_1 = 0;
        ind_2 = 0;
        for (int snum = 0; snum < n_slices; snum++){
            for (int i = 0; i < starts[snum] - ind_1; i++){
                r_array[ind_2 + i + j*M] = (*this)(ind_1 + i,j);
            }
            ind_2 += starts[snum] - ind_1;
            ind_1 = ends[snum];
        }
        for (int i = 0; i < m - ind_1; i++){
            r_array[ind_2 + i + j*M] = (*this)(ind_1 + i,j);
        }
    }

    return Matrix(M, n, r_array);
}


const Matrix &Matrix::Print( FILE *f, int DIGITS, int flag ) const
{    int i, j;
     double x;

     // Flag == 1: Matlab output
     // else: plain output

    if ( flag == 1 )
        fprintf( f, "[" );

    for ( i = 0; i < m; i++ )
    {
        for ( j = 0; j < n; j++ )
        {
            x = (*this)(i,j);
            //x = a(i,j);

            if ( flag == 1 )
            {
                fprintf( f, j == 0 ? " " : ", " );
                fprintf( f, "%.*le", DIGITS, x );
            }
            else
            {
                fprintf( f, j == 0 ? "" : "  " );
                fprintf( f, "% .*le", DIGITS, x );
            }
        }
        if ( flag == 1 )
        {
            if ( i < m-1 )
            fprintf( f, ";\n" );
        }
        else
        {
            if ( i < m-1 )
            fprintf( f, "\n" );
        }
    }

    if ( flag == 1 )
        fprintf( f, " ];\n" );
    else
        fprintf( f, "\n" );

    return *this;
}


std::ostream& operator<<(std::ostream &os, const Matrix &M){
	for (int i = 0; i < M.m; i++){
		for (int j = 0; j < M.n; j++){
			os << M(i,j) << " ";
		}
		os << "\n";
	}
	return os;
}

std::ostream& operator<<(std::ostream &os, const SymMatrix &M){
	for (int i = 0; i < M.m; i++){
		for (int j = 0; j < M.m; j++){
			os << M(i,j) << " ";
		}
		os << "\n";
	}
	return os;
}


Matrix Matrix::operator+(const Matrix &M2) const{
	Matrix M3;
	#ifdef MATRIX_DEBUG
	if ((*this).m != M2.m || (*this).n != M2.n){
		throw std::invalid_argument("+: Mismatched matrix sizes");
	}
	#endif

	M3.Dimension((*this).m, (*this).n);
	for (int i = 0; i<(*this).m; i++){
		for (int j = 0; j<(*this).n; j++){
			M3(i,j) = (*this)(i,j) + M2(i,j);
		}
	}
	return M3;
}

Matrix Matrix::operator-(const Matrix &M2) const{
	Matrix M3;
	#ifdef MATRIX_DEBUG
	if ((*this).m != M2.m || (*this).n != M2.n){
		throw std::invalid_argument("Matrix -: Mismatched matrix sizes");
	}
	#endif

	M3.Dimension((*this).m, (*this).n);
	for (int i = 0; i<(*this).m; i++){
		for (int j = 0; j<(*this).n; j++){
			M3(i,j) = (*this)(i,j) - M2(i,j);
		}
	}
	return M3;
}


Matrix Matrix::operator*(const Matrix &M2) const{
    #ifdef MATRIX_DEBUG
	if (n != M2.m){
		throw std::invalid_argument(std::string("Matrix *: Mismatched matrix sizes, M1.n = ") + std::to_string(n) + std::string(", M2.m = ") + std::to_string(M2.m));
	}
    if (n == 0 && m > 0 && M2.n > 0){
        throw std::invalid_argument("Matrix *: Cannot multiply along chaining dimension zero into a true 2D matrix");
    }
    #endif
    //NEW
    if (m == 0) return Matrix(0, M2.n);
    else if (n == 0) return Matrix(m, M2.n);
    //

    double *array_3 = new double[m * M2.n];
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, M2.n, n, 1.0, array, ldim, M2.array, M2.ldim, 0., array_3, m);
    return Matrix(m, M2.n, array_3);
}


Matrix Matrix::operator*(const double alpha) const{
    double *m_array = new double[m*n];
    for (int i = 0; i<m; i++){
        for (int j = 0; j<n; j++){
            m_array[i + j*m] = (*this)(i,j)*alpha;
        }
    }
    return Matrix(m,n,m_array);
}


void Matrix::operator+=(const Matrix &M2){
    #ifdef MATRIX_DEBUG
   	if (m != M2.m || n != M2.n){
		throw std::invalid_argument("Matrix +=: Mismatched matrix sizes");
	}
    #endif

	for (int i = 0; i<M2.m; i++){
        for (int j = 0; j < M2.n; j++){
            (*this)(i,j) += M2(i,j);
        }
	}
	return;
}


void Matrix::operator-=(const Matrix &M2){
    #ifdef MATRIX_DEBUG
   	if (m != M2.m || n != M2.n){
		throw std::invalid_argument("Matrix -=: Mismatched matrix sizes");
	}
	#endif

	for (int i = 0; i<M2.m; i++){
        for (int j = 0; j < M2.n; j++){
            (*this)(i,j) -= M2(i,j);
        }
	}
	return;
}

void Matrix::operator*=(const double alpha){
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            (*this)(i,j) *= alpha;
        }
    }
    return;
}


Matrix vertcat(std::vector<Matrix> M_k){
    int n = M_k[0].n;
    int m = 0;
    double *array;

    //for (int i = 0; i < M_k.size(); i++){
    for (std::vector<Matrix>::size_type i = 0; i < M_k.size(); i++){
        #ifdef MATRIX_DEBUG
        if (M_k[i].n != n){
            throw std::invalid_argument("Matrix vertcat: Mismatched matrix sizes of " + std::to_string(n) + " and " + std::to_string(M_k[i].n));
        }
        #endif

        m += M_k[i].m;
    }

    array = new double[m*n];
    int ind;
    for (int j = 0; j<n; j++){
        ind = 0;
        for (std::vector<Matrix>::size_type k = 0; k<M_k.size(); k++){
            for (int i = 0; i<M_k[k].m; i++){
                array[ind + i + j*m] = M_k[k](i,j);
            }
           ind += M_k[k].m;
        }
    }
    return Matrix(m, n, array);
}



/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */



int SymMatrix::malloc(void){
    int len;
    len = (m*(m+1))/2.0;
    if ( len == 0 )
       array = NULL;
    else
       if ((array = new double[len]) == NULL)
          Error("'new' failed");

    return 0;
}


int SymMatrix::free(){
    if (array != NULL){
        delete[] array;
        array = nullptr;
    }
    return 0;
}


double &SymMatrix::operator()(int i, int j){
    #ifdef MATRIX_DEBUG
    if (i < 0 || i >= m || j < 0 || j >= m)
    Error("Invalid matrix entry");
    #endif

    if( i < j )//reference to upper triangular part
        return array[(j + i*ldim - (i*(i + 1))/2)];
    return array[i + j*ldim - (j*(j + 1))/2];
}


double &SymMatrix::operator()(int i, int j) const{
    #ifdef MATRIX_DEBUG
    if (i < 0 || i >= m || j < 0 || j >= m)
    Error("Invalid matrix entry");
    #endif

    if (i < j)//reference to upper triangular part
        return array[j + i*ldim - (i*(i + 1))/2];
    return array[i + j*ldim - (j*(j + 1))/2];
}


double &SymMatrix::operator()( int i )
{
    #ifdef MATRIX_DEBUG
    if ( i >= m*(m+1)/2.0 )
    Error("Invalid matrix entry");
    #endif

    return array[i];
}


double &SymMatrix::operator()( int i ) const
{
    #ifdef MATRIX_DEBUG
    if ( i >= m*(m+1)/2.0 )
    Error("Invalid matrix entry");
    #endif

    return array[i];
}


SymMatrix::SymMatrix(){
    array = nullptr;
    m = 0;
    ldim = 0;
    tflag = 0;
}

SymMatrix::SymMatrix(int M){
    m = M;
    ldim = M;
    malloc();
    tflag = 0;
}


SymMatrix::SymMatrix(int M, double *ARRAY, int LDIM){
    m = M;
    if (LDIM < m) ldim = M;
    else ldim = LDIM;
    tflag = 0;
    array = ARRAY;
}


SymMatrix::SymMatrix(int M, int N, int LDIM){
    m = M;
    ldim = M;
    tflag = 0;
    malloc();
}


SymMatrix::SymMatrix(const Matrix &A){
    m = A.m;
    ldim = A.m;
    tflag = 0;
    malloc();
    for (int j = 0; j < m; j++){
         for (int i = j; i < m; i++){
             (*this)(i,j) = A(i,j);
        }
    }
}


SymMatrix::SymMatrix(const SymMatrix &A){
    m = A.m;
    ldim = A.m;
    tflag = 0;

    malloc();
    for (int j = 0; j < m; j++){
        for (int i = j; i < m; i++){
            (*this)(i,j) = A(i,j);
        }
    }
}


SymMatrix::~SymMatrix( void ){
    Dcount++;

    if (!tflag)
        free();
}


SymMatrix &SymMatrix::Dimension(int M){
    free();
    m = M;
    ldim = M;
    //tflag = 0;

    malloc();
    return *this;
}


SymMatrix &SymMatrix::Initialize(double (*f)(int, int)){
    for (int j = 0; j < m; j++){
        for (int i = j; i < m; i++){
            (*this)(i,j) = f(i,j);
        }
    }
    return *this;
}


SymMatrix &SymMatrix::Initialize(double val){
    for (int j = 0; j < m; j++){
        for (int i = j; i < m; i++){
            (*this)(i,j) = val;
        }
    }
    return *this;
}


SymMatrix &SymMatrix::Submatrix(const Matrix &A, int M, int i0){
    //Error("SymMatrix doesn't support Submatrix");
    free();
    m = M;
    array = A.array + (i0 + i0*A.ldim - (i0 * (i0+1))/2);
    ldim = A.ldim - i0;
    tflag = 1;

    return *this;
}


SymMatrix &SymMatrix::Arraymatrix(int M, double *ARRAY){
    if (!tflag)
        free();

    tflag = 1;
    m = M;
    ldim = M;
    array = ARRAY;

    return *this;
}


SymMatrix &SymMatrix::Arraymatrix(int M, double *ARRAY, int LDIM){
    if(!tflag)
        free();

    tflag = 1;
    m = M;
    ldim = LDIM;
    array = ARRAY;

    return *this;
}


SymMatrix &SymMatrix::operator=(const SymMatrix &M2){
    if (m != M2.m){
        delete[] array;
        array = new double[(M2.m * (M2.m + 1))/2];
    }

    m = M2.m;
    ldim = m;

    for (int j = 0; j < m; j++){
        for(int i = j; i < m; i++){
            array[i + j*ldim - (j*(j+1))/2] = M2(i,j);
        }
    }

    return *this;
}


SymMatrix SymMatrix::operator+(const SymMatrix &M2) const{
    double *arr = new double[(m*(m+1))/2];
    for (int j = 0; j < m; j++){
        for (int i = j; i < m; i++){
            arr[i + j*m - j*(j+1)/2] = array[i + j*ldim - j*(j+1)/2] + M2.array[i + j*M2.ldim - j*(j+1)/2];
        }
    }

    return SymMatrix(m, arr);
}

SymMatrix SymMatrix::operator*(const double alpha) const{
    double *arr = new double[(m*(m+1)/2)];
    for (int j = 0; j < m; j++){
        for (int i = j; i < m; i++){
            arr[i + j*m - j*(j+1)/2] = array[i + j*ldim - j*(j+1)/2]*alpha;
        }
    }

    return SymMatrix(m, arr);
}


Matrix SymMatrix::get_slice(int m_start, int m_end, int n_start, int n_end) const{
	if (m_end < m_start || n_end < n_start || m_end > m || n_end > m){
		throw std::invalid_argument("Matrix.get_slice: Slices (" + std::to_string(m_start) + ", " + std::to_string(m_end) + "), (" + std::to_string(n_start) + ", " + std::to_string(n_end) + ") invalid for matrix of shape (" + std::to_string(m) + ", " + std::to_string(m));
	}

	int M_slc = m_end - m_start;
	int N_slc = n_end - n_start;
	double *array_slc = new double[M_slc * N_slc];
	for (int j = 0; j < N_slc; j++){
		for (int i = 0; i < M_slc; i++){
			array_slc[i + j*M_slc] = (*this)(m_start + i, n_start + j);
		}
	}
	return Matrix(M_slc, N_slc, array_slc);
}


const SymMatrix &SymMatrix::Print( FILE *f, int DIGITS, int flag ) const
{    int i, j;
     double x;

     // Flag == 1: Matlab output
     // else: plain output

    if ( flag == 1 )
        fprintf( f, "[" );

    for ( i = 0; i < m; i++ )
    {
        for ( j = 0; j < m; j++ )
        {
            x = (*this)(i,j);
            //x = a(i,j);

            if ( flag == 1 )
            {
                fprintf( f, j == 0 ? " " : ", " );
                fprintf( f, "%.*le", DIGITS, x );
            }
            else
            {
                fprintf( f, j == 0 ? "" : "  " );
                fprintf( f, "% .*le", DIGITS, x );
            }
        }
        if ( flag == 1 )
        {
            if ( i < m-1 )
            fprintf( f, ";\n" );
        }
        else
        {
            if ( i < m-1 )
            fprintf( f, "\n" );
        }
    }

    if ( flag == 1 )
        fprintf( f, " ];\n" );
    else
        fprintf( f, "\n" );

    return *this;
}



/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */


double delta( int i, int j )
{    return (i == j) ? 1.0 : 0.0;
}


Matrix Transpose( const Matrix &A )
{
    int i, j;
    double *array;

    if ( ( array = new double[A.N()*A.M()] ) == NULL )
        Error("'new' failed");

    for ( i = 0; i < A.N(); i++ )
        for ( j = 0; j < A.M(); j++ )
            array[i+j*A.N()] = A(j,i);

    return Matrix( A.N(), A.M(), array, A.N() );
}


Matrix &Transpose( const Matrix &A, Matrix &T )
{
    int i, j;

    T.Dimension( A.N(), A.M() );

    for ( i = 0; i < A.N(); i++ )
        for ( j = 0; j < A.M(); j++ )
            T(i,j) = A(j,i);

    return T;
}


//###########################
//###Sparse_Matrix methods###
//###########################
Sparse_Matrix::Sparse_Matrix(int M, int N, int NNZ): m(M), n(N), 
                                nz(std::make_unique<double[]>(NNZ)), row(std::make_unique<int[]>(NNZ)), colind(std::make_unique<int[]>(n+1)){
    colind[n] = NNZ;                                 
}

Sparse_Matrix::Sparse_Matrix(int M, int N, std::unique_ptr<double[]> NZ, std::unique_ptr<int[]> ROW, std::unique_ptr<int[]> COLIND):
	m(M), n(N), nz(std::move(NZ)), row(std::move(ROW)), colind(std::move(COLIND))
	{}

Sparse_Matrix::Sparse_Matrix(const Sparse_Matrix &M){
	int nnz = M.colind[M.n];
    m = M.m;
	n = M.n;
    nz = std::make_unique<double[]>(nnz);
    row = std::make_unique<int[]>(nnz);
    colind = std::make_unique<int[]>(n+1);
    std::copy(M.nz.get(), M.nz.get() + nnz, nz.get());
    std::copy(M.row.get(), M.row.get() + nnz, row.get());
    std::copy(M.colind.get(), M.colind.get() + M.n + 1, colind.get());
}

Sparse_Matrix::Sparse_Matrix(Sparse_Matrix &&M){
    m = M.m;
    n = M.n;

    nz = std::move(M.nz);
    row = std::move(M.row);
    colind = std::move(M.colind);

    M.m = 0;
    M.n = 0;
}

Sparse_Matrix::Sparse_Matrix(const CSR_Matrix &M){

    m = M.m;
    n = M.n;
    int nnz = M.rowind[M.m];
    nz = std::make_unique<double[]>(nnz);
    row = std::make_unique<int[]>(nnz);
    colind = std::make_unique<int[]>(n + 1);
    CSC_CSR(n, m, M.nz, M.col, M.rowind, nz.get(), row.get(), colind.get());
}


Sparse_Matrix::Sparse_Matrix(){
	m = 0;
	n = 0;

	nz = nullptr;
	row = nullptr;
	colind = std::unique_ptr<int[]>(new int[1]{0});
};




Sparse_Matrix &Sparse_Matrix::Dimension(int M, int N, int NNZ){

    m = M;
    n = N;
    
    nz = std::make_unique<double[]>(NNZ);
    row = std::make_unique<int[]>(NNZ);
    colind = std::make_unique<int[]>(n+1);
    colind[n] = NNZ;
    return *this;
}



void Sparse_Matrix::operator=(const Sparse_Matrix& M){
	//Set Matrix default-constructed if M is default constructed
    if (M.nz == nullptr || M.row == nullptr){
        m = 0;
        n = 0;
        nz = nullptr;
        row = nullptr;
        colind = std::unique_ptr<int[]>(new int[1]{0});
        return;
    }
    
    int nnz = M.colind[M.n];
    if (colind[n] != nnz){
        nz = std::make_unique<double[]>(nnz);
        row = std::make_unique<int[]>(nnz);
    }
    std::copy(M.nz.get(), M.nz.get() + nnz, nz.get());
    std::copy(M.row.get(), M.row.get() + nnz, row.get());
    
    
    m = M.m;
    if (n != M.n){
        n = M.n;
        colind = std::make_unique<int[]>(n+1);
    }
    std::copy(M.colind.get(), M.colind.get() + M.n + 1, colind.get());
}

void Sparse_Matrix::operator=(Sparse_Matrix &&M){
    m = M.m;
    n = M.n;
    nz = std::move(M.nz);
    row = std::move(M.row);
    colind = std::move(M.colind);
    M.m = 0; 
    M.n = 0;
}

Sparse_Matrix Sparse_Matrix::operator+(const Sparse_Matrix &M2) const{
    #ifdef MATRIX_DEBUG
	if ((*this).m != M2.m || (*this).n != M2.n){
		throw std::invalid_argument("+: Mismatched matrix sizes");
	}
	#endif

	std::unique_ptr<int[]> colind = std::make_unique<int[]>(n + 1);
	std::unique_ptr<int[]> row;
	std::unique_ptr<double[]> nz;

	std::vector<double> NZ;
	std::vector<int> ROW;

	int i1 = 0;
	int i2 = 0;
	int ind = 0;

	colind[0] = 0;
	for (int j = 0; j<M2.n; j++){
		while (true){
			if (i1 >= (*this).colind[j+1]){
				while (i2 < M2.colind[j+1]){
					NZ.push_back(M2.nz[i2]);
					ROW.push_back(M2.row[i2]);
					i2++;
					ind++;
				}
				break;
			}
			else if(i2 >= M2.colind[j+1]){
				while (i1 < (*this).colind[j+1]){
					NZ.push_back((*this).nz[i1]);
					ROW.push_back((*this).row[i1]);
					i1++;
					ind++;
				}
				break;
			}

			if ((*this).row[i1] < M2.row[i2]){
				ROW.push_back((*this).row[i1]);
				NZ.push_back((*this).nz[i1]);
				i1++;
				ind++;
			}
			else if (M2.row[i2] < (*this).row[i1]){
				ROW.push_back(M2.row[i2]);
				NZ.push_back(M2.nz[i2]);
				i2++;
				ind++;
			}
			else{
			 ROW.push_back(M2.row[i2]);
			 NZ.push_back((*this).nz[i1] + M2.nz[i2]);
			 i1++;
			 i2++;
			 ind++;
			}

		}
		colind[j+1] = ind;
	}

	nz = std::make_unique<double[]>(ind);
	row = std::make_unique<int[]>(ind);
    std::copy(NZ.begin(), NZ.end(), nz.get());
    std::copy(ROW.begin(), ROW.end(), row.get());
    
	return Sparse_Matrix(M2.m, M2.n, std::move(nz), std::move(row), std::move(colind));
}




Sparse_Matrix Sparse_Matrix::get_slice(int m_start, int m_end, int n_start, int n_end) const{
    #ifdef MATRIX_DEBUG
	if (m_end > m || n_end > n){
        std::string err_str = "slice out of matrix bounds: Matrix shape is (" + std::to_string(m) + "," + std::to_string(n) + "), m_end, n_end = " + std::to_string(m_end) + ", " + std::to_string(n_end);
		throw std::invalid_argument(err_str);
	}
	#endif

	std::vector<int> row_v = {};
	std::vector<int> col_v = {0};
	std::vector<double> nz_v;
	int nnz = 0;

	for (int j = n_start; j < n_end; j++){
		for (int i = colind[j]; i < colind[j+1]; i++){
			if (row[i] >= m_start && row[i] < m_end){
				row_v.push_back(row[i] - m_start);
				nz_v.push_back(nz[i]);
				nnz++;
			}
        }
		col_v.push_back(nnz);
	}

	std::unique_ptr<double[]> NZ = std::make_unique<double[]>(nnz);
	std::unique_ptr<int[]> ROW = std::make_unique<int[]>(nnz);
	std::unique_ptr<int[]> COLIND = std::make_unique<int[]>(n_end - n_start + 1);
    std::copy(nz_v.begin(), nz_v.end(), NZ.get());
    std::copy(row_v.begin(), row_v.end(), ROW.get());
    std::copy(col_v.begin(), col_v.end(), COLIND.get());

	return Sparse_Matrix(m_end - m_start, n_end - n_start, std::move(NZ), std::move(ROW), std::move(COLIND));
}


Matrix Sparse_Matrix::get_dense_slice(int m_start, int m_end, int n_start, int n_end) const{
    #ifdef MATRIX_DEBUG
    if (m_end > m || n_end > n){
        std::string err_str = "slice out of matrix bounds: Matrix shape is (" + std::to_string(m) + "," + std::to_string(n) + "), m_end, n_end = " + std::to_string(m_end) + ", " + std::to_string(n_end);
		throw std::invalid_argument(err_str);
	}
	#endif

    int m_ = (m_end - m_start) * (m_end - m_start >= 0);
    int n_ = (n_end - n_start) * (n_end - n_start >= 0);
    double *arr = new double[m_ * n_]();

    int i_nz;
    for (int j = n_start; j < n_end; j++){
        for (int i = colind[j]; i < colind[j+1]; i++){
            i_nz = row[i];
            if (i_nz >= m_start && i_nz < m_end){
                arr[(i_nz - m_start) + (j - n_start)*m_] = nz[i];
            }
        }
    }

    return Matrix(m_, n_, arr);
}


Matrix Sparse_Matrix::dense() const{
    double *array_d = new double[m*n];
    for (int i = 0; i < m*n; i++){
        array_d[i] = 0.0;
    }

    for (int j = 0; j < n; j++){
        for (int i = colind[j]; i < colind[j+1]; i++){
            array_d[row[i] + j*m] = nz[i];
        }
    }


    return Matrix(m, n, array_d);
}


Sparse_Matrix sparse_dense_multiply(const Sparse_Matrix &M1, const Matrix &M2){
    #ifdef MATRIX_DEBUG
	if (M1.n != M2.m){
		throw std::invalid_argument( "sparse_dense_multiply: Mismatched matrix sizes" );
	}
	if (M1.n == 0){
        throw std::invalid_argument("sparse_dense_multiply: Cannot multiply along chaining dimension zero");
    }
    #endif
    
    /*
	int index_offsets[M2.m] = {0};
	int first_rows[M2.m];
	int min_inds[M2.m];
    */

    int *index_offsets = new int[M2.m], *first_rows = new int[M2.m], *min_inds = new int[M2.m];

	int min_end;
	int min_row;
	int num_inds = 0;
	double entry;
	int c_ind = 0;

	std::vector<double> nz_m = {};
	std::vector<int> row_m = {};
	std::vector<int> colind_m = {0};
    
	for (int j2 = 0; j2 < M2.n; j2++){
		for (int k = 0; k<M1.n; k++){
			index_offsets[k] = 0;
			}
		while (true){
			num_inds = 0;
			for (int k = 0; k<M1.n; k++){
				if (M1.colind[k] + index_offsets[k] < M1.colind[k+1]){
					first_rows[k] = M1.row[M1.colind[k] + index_offsets[k]];
					num_inds++;
				}
				else{
					first_rows[k] = M1.m;
				}
			}
			if (num_inds == 0){
				break;
			}

			min_row = M1.m;
			min_end = 0;
			for (int k = 0; k<M1.n;k++){
				if (min_row > first_rows[k]){
					min_row = first_rows[k];
					min_inds[0] = k;
					min_end = 1;
				}
				else if (min_row == first_rows[k]){
					min_inds[min_end] = k;
					min_end++;
				}
			}

			entry = 0.;
			for (int k = 0; k<min_end; k++){
				entry += M1.nz[M1.colind[min_inds[k]] + index_offsets[min_inds[k]]]*M2(min_inds[k], j2);
                index_offsets[min_inds[k]] += 1;
			}
			nz_m.push_back(entry);
			row_m.push_back(min_row);
			c_ind++;
		}
		colind_m.push_back(c_ind);
	}

	std::unique_ptr<double[]> NZ = std::make_unique<double[]>(c_ind);
	std::unique_ptr<int[]> ROW = std::make_unique<int[]>(c_ind);
	std::unique_ptr<int[]> COLIND = std::make_unique<int[]>(M2.n + 1);
    std::copy(nz_m.begin(), nz_m.end(), NZ.get());
    std::copy(row_m.begin(), row_m.end(), ROW.get());
    std::copy(colind_m.begin(), colind_m.end(), COLIND.get());

    delete[] index_offsets; delete[] first_rows; delete[] min_inds;
    
	return Sparse_Matrix(M1.m, M2.n, std::move(NZ), std::move(ROW), std::move(COLIND));
}


Matrix sparse_vector_multiply(const Sparse_Matrix &M1, const Matrix &V1){
    double *arr = new double[M1.m]();

    for (int j = 0; j < M1.n; j++){
        for (int i = M1.colind[j]; i < M1.colind[j+1]; i++){
            arr[M1.row[i]] += M1.nz[i] * V1(j);
        }
    }

    return Matrix(M1.m, 1, arr);
}



Matrix transpose_multiply(const Sparse_Matrix &M1, const Matrix &M2){
    #ifdef MATRIX_DEBUG
    if (M1.m != M2.m){
        throw std::invalid_argument( "transpose_multiply: Incompatible matrix shapes of (" + std::to_string(M1.m) + ", " + std::to_string(M1.n) + "), (" + std::to_string(M2.m) + ", " + std::to_string(M2.n) + ")");
    }

    if (M1.m == 0){
        throw std::invalid_argument( "transpose_multiply: Cannot multiply from the left with matrix of size_2 = 0" );
    }
    #endif

    int M = M1.n;
    int N = M2.n;
    double *array = new double[M*N];

    for (int j = 0; j < N; j++){
        for (int i = 0; i < M; i++){
            array[i + j*M] = 0;
            for (int k = M1.colind[i]; k < M1.colind[i+1]; k++){
                array[i + j*M] += M1.nz[k]*M2(M1.row[k],j);
            }
        }
    }
    return Matrix(M, N, array);
}


void Sparse_Matrix::remove_rows(int *starts, int *ends, int nblocks){

    #ifdef MATRIX_DEBUG
	for (int l = 0; l<nblocks-1; l++){
		if (starts[l] > ends[l] || ends[l]>starts[l+1]){
			throw std::invalid_argument( "slice end must be greater than start and slices must be non-overlapping and sorted by starting index" );
		}
	}
	if (starts[nblocks - 1] > ends[nblocks - 1]){
		throw std::invalid_argument( "slice end must be greater than start and slices must be non-overlapping and sorted by starting index" );
	}
	#endif

	std::vector<double> NZ_v = {};
	std::vector<int> ROW_v = {};
	std::vector<int> COLIND_v = {0};
	COLIND_v.reserve(n+1);
	int c_ind = 0;
	bool in_slice;
	int row_ind_offset;


	for (int k = 0; k<n; k++){
		for (int j = colind[k]; j<colind[k+1]; j++){
			in_slice = false;
			row_ind_offset = 0;
			for (int l = 0; l<nblocks; l++){
				if	(row[j] >= starts[l] && row[j] < ends[l]){
					in_slice = true;
					break;
				}
				else if (row[j] >= ends[l]){
					row_ind_offset += ends[l] - starts[l];
				}
				else{
					break;
				}
			}
			if (!in_slice){
				NZ_v.push_back(nz[j]);
				ROW_v.push_back(row[j] - row_ind_offset);
				c_ind++;
			}
		}
		COLIND_v.push_back(c_ind);
	}

	nz = std::make_unique<double[]>(c_ind);
	row = std::make_unique<int[]>(c_ind);

    std::copy(NZ_v.begin(), NZ_v.end(), nz.get());
    std::copy(ROW_v.begin(), ROW_v.end(), row.get());
    std::copy(COLIND_v.begin(), COLIND_v.end(), colind.get());
    
	for (int k = 0; k<nblocks; k++){
		m -= ends[k] - starts[k];
	}
}

Sparse_Matrix Sparse_Matrix::without_nz_rows(int *starts, int *ends, int nblocks) const{
    #ifdef MATRIX_DEBUG
	for (int l = 0; l<nblocks-1; l++){
		if (starts[l] > ends[l] || ends[l]>starts[l+1]){
			throw std::invalid_argument( "slice end must be greater than start and slices must be non-overlapping and sorted by starting index" );
		}
	}
	if (starts[nblocks - 1] > ends[nblocks - 1]){
		throw std::invalid_argument( "slice end must be greater than start and slices must be non-overlapping and sorted by starting index" );
	}
	#endif

	std::vector<double> NZ_v = {};
	std::vector<int> ROW_v = {};
	std::vector<int> COLIND_v = {0};
	COLIND_v.reserve(n+1);
	int c_ind = 0;
	bool in_slice;

	for (int k = 0; k<n; k++){
		for (int j = colind[k]; j<colind[k+1]; j++){
			in_slice = false;
			for (int l = 0; l<nblocks; l++){
				if	(row[j] >= starts[l] && row[j] < ends[l]){
					in_slice = true;
					break;
				}
			}
			if (!in_slice){
				NZ_v.push_back(nz[j]);
				ROW_v.push_back(row[j]);
				c_ind++;
			}
		}
		COLIND_v.push_back(c_ind);
	}

	std::unique_ptr<double[]> M_nz = std::make_unique<double[]>(c_ind);
	std::unique_ptr<int[]> M_row = std::make_unique<int[]>(c_ind);
	std::unique_ptr<int[]> M_colind = std::make_unique<int[]>(n+1);

    std::copy(NZ_v.begin(), NZ_v.end(), M_nz.get());
    std::copy(ROW_v.begin(), ROW_v.end(), M_row.get());
    std::copy(COLIND_v.begin(), COLIND_v.end(), M_colind.get());
    
	return Sparse_Matrix(m, n, std::move(M_nz), std::move(M_row), std::move(M_colind));
}


Sparse_Matrix horzcat(Sparse_Matrix *mats, int n_mat){
	int nnz = 0;
	int m = mats[0].m;
	int n = 0;

	for (int k = 0; k<n_mat; k++){
        #ifdef MATRIX_DEBUG
        if (mats[k].m != m){
            throw std::invalid_argument("Sparse_Matrix horzcat: Incompatible matrix sizes");
        }
        #endif

		nnz += mats[k].colind[mats[k].n];
		n += mats[k].n;
	}
	std::unique_ptr<double[]> nz = std::make_unique<double[]>(nnz);
	std::unique_ptr<int[]> row = std::make_unique<int[]>(nnz);
	std::unique_ptr<int[]> colind = std::make_unique<int[]>(n+1);

	int row_offset = 0;
	int col_offset = 0;
	for (int k = 0; k < n_mat; k++){

		for (int l = 0; l < mats[k].n; l++){
			colind[col_offset + l] = mats[k].colind[l] + row_offset;
		}
		col_offset += mats[k].n;

		for (int l = 0; l<mats[k].colind[mats[k].n]; l++){
			nz[row_offset + l] = mats[k].nz[l];
			row[row_offset + l] = mats[k].row[l];
		}
		row_offset += mats[k].colind[mats[k].n];
	}
	colind[n] = nnz;

	return Sparse_Matrix(m, n, std::move(nz), std::move(row), std::move(colind));
}

Sparse_Matrix horzcat(std::vector<Sparse_Matrix> &mats){
    int nnz = 0;
    int m = mats[0].m;
    int n = 0;

	for (std::vector<Matrix>::size_type k = 0; k<mats.size(); k++){
        #ifdef MATRIX_DEBUG
        if (mats[k].m != m){
            throw std::invalid_argument("Sparse_Matrix horzcat: Incompatible matrix sizes");
        }
        #endif

		nnz += mats[k].colind[mats[k].n];
		n += mats[k].n;
	}
	std::unique_ptr<double[]> nz = std::make_unique<double[]>(nnz);
	std::unique_ptr<int[]> row = std::make_unique<int[]>(nnz);
	std::unique_ptr<int[]> colind = std::make_unique<int[]>(n+1);

	int row_offset = 0;
	int col_offset = 0;
	for (std::vector<Matrix>::size_type k = 0; k < mats.size(); k++){

		for (int l = 0; l < mats[k].n; l++){
			colind[col_offset + l] = mats[k].colind[l] + row_offset;
		}
		col_offset += mats[k].n;

		for (int l = 0; l<mats[k].colind[mats[k].n]; l++){
			nz[row_offset + l] = mats[k].nz[l];
			row[row_offset + l] = mats[k].row[l];
		}
		row_offset += mats[k].colind[mats[k].n];
	}
	colind[n] = nnz;

	return Sparse_Matrix(m, n, std::move(nz), std::move(row), std::move(colind));
}

Sparse_Matrix vertcat(std::vector<Sparse_Matrix> &mats){
    int nnz = 0;
    int m = 0;
    int n = mats[0].n;

    for (std::vector<Matrix>::size_type k = 0; k < mats.size(); k++){
        #ifdef MATRIX_DEBUG
        if (mats[k].n != n){
            throw std::invalid_argument("Sparse_Matrix vertcat: Incompatible matrix sizes");
        }
        #endif

        nnz += mats[k].colind[mats[k].n];
        m += mats[k].m;
    }
	std::unique_ptr<double[]> nz = std::make_unique<double[]>(nnz);
	std::unique_ptr<int[]> row = std::make_unique<int[]>(nnz);
	std::unique_ptr<int[]> colind = std::make_unique<int[]>(n+1);
    colind[0] = 0;

    int row_offset;
    int ind = 0;
    for (int j = 0; j<n; j++){
        row_offset = 0;
        for (std::vector<Matrix>::size_type k = 0; k < mats.size(); k++){
            for (int i = mats[k].colind[j]; i<mats[k].colind[j+1]; i++){
                nz[ind] = mats[k].nz[i];
                row[ind] = mats[k].row[i] + row_offset;
                ind++;
            }
            row_offset += mats[k].m;
        }
        colind[j+1] = ind;
    }
    return Sparse_Matrix(m, n, std::move(nz), std::move(row), std::move(colind));
}


Sparse_Matrix lr_zero_pad(int N, const Sparse_Matrix &M1, int start){
    #ifdef MATRIX_DEBUG
    if (start + M1.n > N){
        throw std::invalid_argument("lr_zero_pad: Matrix not in given bounds at given position");
    }
    #endif
    
	std::unique_ptr<double[]> nz = std::make_unique<double[]>(M1.colind[M1.n]);
	std::unique_ptr<int[]> row = std::make_unique<int[]>(M1.colind[M1.n]);
	std::unique_ptr<int[]> colind = std::make_unique<int[]>(N+1);

    colind[0] = 0;
    for (int j = 0; j < start; j++){
        colind[j+1] = 0;
    }
    for (int j = start; j < start + M1.n; j++){
        colind[j+1] = M1.colind[j+1 - start];
    }

    for (int j = start + M1.n; j < N; j++){
        colind[j+1] = M1.colind[M1.n];
    }

    std::copy(M1.nz.get(), M1.nz.get() + M1.colind[M1.n], nz.get());
    std::copy(M1.row.get(), M1.row.get() + M1.colind[M1.n], row.get());

    return Sparse_Matrix(M1.m, N, std::move(nz), std::move(row), std::move(colind));
}


Sparse_Matrix lr_zero_pad(int N, const Matrix &M1, int start){
    #ifdef MATRIX_DEBUG
    if (start + M1.n > N){
        throw std::invalid_argument("lr_zero_pad: Matrix not in given bounds at given position");
    }
    #endif

    std::unique_ptr<double[]> nz = std::make_unique<double[]>(M1.m * M1.n);
    std::unique_ptr<int[]> row = std::make_unique<int[]>(M1.m * M1.n);
    std::unique_ptr<int[]> colind = std::make_unique<int[]>(N+1);

    colind[0] = 0;
    for (int j = 0; j < start; j++){
        colind[j+1] = 0;
    }

    for (int j = 1; j <= M1.n; j++){
        colind[start+j] = M1.m * j;
    }

    for (int j = start + M1.n; j < N; j++){
        colind[j+1] = M1.m * M1.n;
    }

    for (int j = 0; j < M1.n; j++){
        for (int i = 0; i < M1.m; i++){
            nz[i + j*M1.m] = M1(i,j);
            row[i + j*M1.m] = i;
        }
    }

    return Sparse_Matrix(M1.m, N, std::move(nz), std::move(row), std::move(colind));
}



//######################################
//###CSR_Matrix methods#################
//######################################

CSR_Matrix::CSR_Matrix():
    m(0), n(0), nz(nullptr), col(nullptr), rowind(nullptr), free_data(false){}

CSR_Matrix::CSR_Matrix(int M, int N, double *NZ, int *COL, int *ROWIND, bool FD):
    m(M), n(N), nz(NZ), col(COL), rowind(ROWIND), free_data(FD){}

CSR_Matrix::CSR_Matrix(const CSR_Matrix &M1){
    m = M1.m;
    n = M1.n;

    if (M1.free_data){
        int nnz = M1.rowind[M1.m];
        nz = new double[nnz];
        col = new int[nnz];
        rowind = new int[M1.m + 1];

        std::copy(M1.nz, M1.nz + nnz, nz);
        std::copy(M1.col, M1.col + nnz, col);
        std::copy(M1.rowind, M1.rowind + M1.m + 1, rowind);
        free_data = true;
    }
    else{
        nz = M1.nz;
        col = M1.col;
        rowind = M1.rowind;
        free_data = false;
    }
}

CSR_Matrix::CSR_Matrix(CSR_Matrix &&M1){
    m = M1.m;
    n = M1.n;
    free_data = M1.free_data;
    nz = M1.nz;
    col = M1.col;
    rowind = M1.rowind;

    M1.m = 0;
    M1.n = 0;
    M1.nz = nullptr;
    M1.col = nullptr;
    M1.rowind = nullptr;
    M1.free_data = false;
}

CSR_Matrix::CSR_Matrix(const Sparse_Matrix &M1){
    m = M1.m;
    n = M1.n;
    nz = new double[M1.colind[M1.n]];
    col = new int[M1.colind[M1.n]];
    rowind = new int[M1.m + 1];

    CSC_CSR(m, n, M1.nz.get(), M1.row.get(), M1.colind.get(), nz, col, rowind);
    free_data = true;
}


CSR_Matrix::~CSR_Matrix(){
    if (free_data){
        delete[] nz;
        delete[] col;
        delete[] rowind;
    }
}

void CSR_Matrix::operator=(const CSR_Matrix &M1){
    if (free_data){
        delete[] nz;
        delete[] col;
        delete[] rowind;
    }

    m = M1.m;
    n = M1.n;

    if (M1.free_data){

        int nnz = M1.rowind[M1.m];

        nz = new double[nnz];
        col = new int[nnz];
        rowind = new int[M1.m + 1];

        std::copy(M1.nz, M1.nz + nnz, nz);
        std::copy(M1.col, M1.col + nnz, col);
        std::copy(M1.rowind, M1.rowind + M1.m + 1, rowind);
        free_data = true;
    }
    else{
        nz = M1.nz;
        col = M1.col;
        rowind = M1.rowind;
        free_data = false;
    }

    return;
}

void CSR_Matrix::operator=(CSR_Matrix &&M1){
    m = M1.m;
    n = M1.n;
    if (free_data){
        delete[] nz;
        delete[] col;
        delete[] rowind;
    }
    free_data = M1.free_data;
    nz = M1.nz;
    col = M1.col;
    rowind = M1.rowind;

    M1.m = 0;
    M1.n = 0;
    M1.nz = nullptr;
    M1.col = nullptr;
    M1.rowind = nullptr;
    M1.free_data = false;
}


Matrix CSR_Matrix::dense() const{
    double *arr = new double[m*n]();

    for (int i = 0; i < m; i++){
        for (int j = rowind[i]; j < rowind[i+1]; j++){
            arr[i + col[j]*m] = nz[j];
        }
    }

    return Matrix(m, n, arr);
}

//######################################
//###LT_Block_Matrix methods############
//######################################

int LT_Block_Matrix::malloc(void){
	int len;
	len = (m*(m+1))/2;
    if ( len == 0 )
        array = nullptr;
    else
        if ( ( array = new Matrix[len] ) == NULL )
            Error("'new' failed");

    return 0;
}

LT_Block_Matrix::LT_Block_Matrix(int M, int *m_sizes, int *n_sizes){
	m = M;
	n = M;
	m_block_sizes = m_sizes;
	n_block_sizes = n_sizes;
	malloc();
}

LT_Block_Matrix::LT_Block_Matrix(int M, int N, int *m_sizes, int *n_sizes){
	m = M;
	n = N;
	m_block_sizes = m_sizes;
	n_block_sizes = n_sizes;
	malloc();
}


LT_Block_Matrix::LT_Block_Matrix(){
	m = 0;
	n = 0;
	array = nullptr;
	m_block_sizes = nullptr;
	n_block_sizes = nullptr;
}

LT_Block_Matrix::~LT_Block_Matrix(void){
	delete[] array;
	if (m_block_sizes == n_block_sizes){
		delete[] m_block_sizes;
	}
	else{
		delete[] m_block_sizes;
		delete[] n_block_sizes;
	}
}

void LT_Block_Matrix::set(int i, int j, const Matrix &M){
    #ifdef MATRIX_DEBUG
	if (m_block_sizes[i] != M.m || n_block_sizes[j] != M.n){
		throw std::invalid_argument( "LT_Block_Matrix.set: Matrix shape does not adhere to Block_Matrix structure");
	}
	if (i >= m || j >= n){
		throw std::invalid_argument( "LT_Block_Matrix.set: received out of bounds matrix indizes (" + std::to_string(i) + ", " + std::to_string(j) + ") for blockmatrix of shape (" + std::to_string(m) + ", " + std::to_string(n) + ")");
	}
	if (j > i){
		throw std::invalid_argument( "LT_Block_Matrix.set: cannot set matrix in upper triangle block (" + std::to_string(i) + ", " + std::to_string(j) + ")");
	}
	#endif

	array[i + j*m - (j*(j+1))/2] = M;

	return;

}

const Matrix &LT_Block_Matrix::operator() (int i, int j) const{
    #ifdef MATRIX_DEBUG
	if (i >= m || j >= n){
		throw std::invalid_argument( "LT_Block_Matrix(,): received out of bounds matrix indizes (" + std::to_string(i) + ", " + std::to_string(j) + ") for blockmatrix of shape ("  + std::to_string(m) + ", " + std::to_string(n) + ")");
	}
	#endif

	if (i >= j){
		return array[i + j*m - (j*(j+1))/2];
	}
	else{
		return Matrix(m_block_sizes[i], n_block_sizes[j]).Initialize(0.);
	}
}

LT_Block_Matrix &LT_Block_Matrix::operator=(const LT_Block_Matrix &M){
	delete[] array;
	delete[] m_block_sizes;
	delete[] n_block_sizes;

	m = M.m;
	n = M.n;
	malloc();
	m_block_sizes = new int[m];
	n_block_sizes = new int[n];

	for (int i = 0; i < m; i++){
		m_block_sizes[i] = M.m_block_sizes[i];
	}
	for (int i = 0; i<n; i++){
		n_block_sizes[i] = M.n_block_sizes[i];
	}
	for (int i = 0; i < m; i++){
		for (int j = 0; j <= i; j++){
			set(i,j, M(i,j));
		}
	}

	return *this;
}

LT_Block_Matrix &LT_Block_Matrix::Dimension(int M, int N, int* m_sizes, int* n_sizes){
	delete[] array;
	delete[] m_block_sizes;
	delete[] n_block_sizes;

	m = M;
	n = N;
	m_block_sizes = m_sizes;
	n_block_sizes = n_sizes;
	malloc();

	return *this;
}


void LT_Block_Matrix::to_dense(Matrix &M) const{
	int M_d = 0;
	int N_d = 0;

	for (int i = 0; i<m; i++){
		M_d += m_block_sizes[i];
	}

	for (int i = 0; i<n; i++){
		N_d += n_block_sizes[i];
	}

	M.Dimension(M_d, N_d);
	M.Initialize(0.);

	int I_start = 0;
	int J_start;
	for (int I = 0; I<m; I++){
		J_start = 0;
		for (int J = 0; J <= std::min(I, n-1); J++ ){
			for (int i = 0; i<m_block_sizes[I]; i++){
				for (int j = 0; j<n_block_sizes[J]; j++){
					M(I_start + i, J_start + j) = (*this)(I,J)(i,j);
				}
			}
			J_start += n_block_sizes[J];
		}
		I_start += m_block_sizes[I];
	}
}

void LT_Block_Matrix::to_sym(SymMatrix &M) const{
	int M_d = 0;
	int N_d = 0;

	for (int i = 0; i<m; i++){
		M_d += m_block_sizes[i];
	}

	for (int i = 0; i<n; i++){
		N_d += n_block_sizes[i];
	}
	if (M_d != N_d) throw std::logic_error("LT_Block_Matrix: Cannot create equivalent SymMatrix - no symmetry");

	M.Dimension(M_d).Initialize(0.);

	int I_start = 0;
	int J_start;
	for (int I = 0; I<m; I++){
		J_start = 0;
		for (int J = 0; J <= std::min(I, n-1); J++ ){
			for (int i = 0; i<m_block_sizes[I]; i++){
				for (int j = 0; j<n_block_sizes[J]; j++){
					M(I_start + i, J_start + j) = (*this)(I,J)(i,j);
				}
			}
			J_start += n_block_sizes[J];
		}
		I_start += m_block_sizes[I];
	}
}


void LT_Block_Matrix::to_sparse(Sparse_Matrix &M) const{
	int M_m = 0;
	int M_n = 0;
    int M_nnz = 0;

	for (int i = 0; i<m; i++){
		M_m += m_block_sizes[i];
	}

	for (int i = 0; i<n; i++){
		M_n += n_block_sizes[i];
	}

    int qsize = std::min(m, n);
    int Lsize = M_m;
	for (int i = 0; i < qsize; i++){
        M_nnz += n_block_sizes[i] * Lsize;
        Lsize -= m_block_sizes[i];
	}

	std::unique_ptr<double[]> M_nz = std::make_unique<double[]>(M_nnz);
	std::unique_ptr<int[]> M_row = std::make_unique<int[]>(M_nnz);
    std::unique_ptr<int[]> M_colind = std::make_unique<int[]>(M_n + 1);
    M_colind[0] = 0;

    int L_start = 0, I_start, J_start = 0;
    int ind = 0, ind_2 = M_m;

    for (int J = 0; J < qsize; J++){
        for (int j = 0; j < n_block_sizes[J]; j++){
            I_start = L_start;
            for (int I = J; I < m; I++){
                for (int i = 0; i < m_block_sizes[I]; i++){
                    M_nz[ind] = (*this)(I,J)(i,j);
                    M_row[ind] = I_start + i;
                    ind++;
                }
                I_start += m_block_sizes[I];
            }
            M_colind[J_start + j + 1] = M_colind[J_start + j] + ind_2;
        }
        J_start += n_block_sizes[J];
        L_start += m_block_sizes[J];
        ind_2 -= m_block_sizes[J];
    }

    for (int J = qsize; J < n; J++){
        for (int j = 0; j < n_block_sizes[J]; j++){
            M_colind[J_start + j + 1] = M_nnz;
        }
        J_start += n_block_sizes[J];
    }

    M = Sparse_Matrix(M_m, M_n, std::move(M_nz), std::move(M_row), std::move(M_colind));
    return;
}



void CSC_CSR(const int m, const int n, const double* nz_CSC, const int* row, const int* colind, double* const nz_CSR, int* const col, int* const rowind){

    int *row_nnz = new int[m]();

    for (int i = 0; i < colind[n]; i++){
        row_nnz[row[i]] += 1;
    }

    rowind[0] = 0;
    for (int i = 0; i < m; i++){
        rowind[i+1] = rowind[i] + row_nnz[i];
    }
    int *row_offset = new int[m]();

    for (int k = 0; k < n; k++){
        for (int i = colind[k]; i < colind[k+1]; i++){
            nz_CSR[rowind[row[i]] + row_offset[row[i]]] = nz_CSC[i];
            col[rowind[row[i]] + row_offset[row[i]]] = k;
            row_offset[row[i]] += 1;
        }
    }


    delete[] row_offset;
    delete[] row_nnz;
    return;
}


CSR_Matrix sparse_dense_multiply_2(const Sparse_Matrix &M1, const Matrix &M2){

    double *nz_ = new double[M1.colind[M1.n]];
    int *col = new int[M1.colind[M1.n]];
    int *rowind = new int[M1.m + 1];

    CSC_CSR(M1.m, M1.n, M1.nz.get(), M1.row.get(), M1.colind.get(), nz_, col, rowind);

    int n_nzrow = 0;
    for (int i = 0; i < M1.m; i++){
        n_nzrow += ((rowind[i+1] - rowind[i]) > 0);
    }

    double *nz_out = new double[n_nzrow * M2.n];
    int *col_out = new int[n_nzrow * M2.n];
    int *rowind_out = new int[M1.m + 1];
    rowind_out[0] = 0;
    for (int i = 1; i <= M1.m; i++){
        rowind_out[i] = rowind_out[i-1] + (rowind[i] - rowind[i-1] > 0) * M2.n;
    }

    double sum;
    int ind = 0;
    for (int i = 0; i < M1.m; i++){
        if (rowind[i+1] - rowind[i] > 0){
            for (int j = 0; j < M2.n; j++){
                sum = 0.0;
                for (int k = rowind[i]; k < rowind[i+1]; k++){
                    sum += nz_[k] * M2(col[k], j);
                }
                nz_out[ind] = sum;
                col_out[ind] = j;
                ++ind;
            }
        }
    }

    delete[] nz_; delete[] col; delete[] rowind;

    return CSR_Matrix(M1.m, M2.n, nz_out, col_out, rowind_out, true);
}


CSR_Matrix add_fullrow(const CSR_Matrix &M1, const CSR_Matrix &M2){
    int *rowind = new int[M1.m + 1];
    rowind[0] = 0;
    int M2_row_nnz, ind;

    for (int i = 1; i <= M1.m; i++){
        M2_row_nnz = M2.rowind[i] - M2.rowind[i-1];
        rowind[i] = rowind[i-1] + (M2_row_nnz <= 0) * (M1.rowind[i] - M1.rowind[i-1]) + (M2_row_nnz > 0) * M2_row_nnz;
    }

    double *nz = new double[rowind[M1.m]];
    int *col = new int[rowind[M1.m]];

    ind = 0;
    int ind_old;
    for (int j = 0; j < M1.m; j++){
        M2_row_nnz = M2.rowind[j+1] - M2.rowind[j];
        if (M2_row_nnz > 0){
            ind_old = ind;
            for (int i = 0; i < M2.n; i++){
                nz[ind] = M2.nz[M2.rowind[j] + i];
                col[ind] = i;
                ++ind;
            }
            for (int i = M1.rowind[j]; i < M1.rowind[j+1]; i++){
                nz[ind_old + M1.col[i]] += M1.nz[i];
            }
        }
        else{
            for (int i = M1.rowind[j]; i < M1.rowind[j+1]; i++){
                nz[ind] = M1.nz[i];
                col[ind] = M1.col[i];
                ++ind;
            }
        }
    }

    return CSR_Matrix(M1.m, M1.n, nz, col, rowind, true);
}


CSR_Matrix fullrow_multiply(const CSR_Matrix &M1, const Matrix &M2){
    #ifdef MATRIX_DEBUG
    if (M1.n != M2.m || M1.n == 0){
        std::cout << "M1.n = " << M1.n << ", M2.m = " << M2.m << "\n";
        throw std::invalid_argument("fullrow_multiply: Mismatched chaining dimensions");
    }
    #endif

    int n_nzrow = 0;
    for (int i = 0; i < M1.m; i++){
        n_nzrow += ((M1.rowind[i+1] - M1.rowind[i]) > 0);
    }

    double *nz = new double[n_nzrow * M2.n];

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n_nzrow, M2.n, M1.n, 1.0, M1.nz, M1.n, M2.array, M2.ldim, 0., nz, M2.n);

    int *col = new int[n_nzrow * M2.n];
    int ind = 0;
    for (int i = 0; i < n_nzrow; i++){
        for (int j = 0; j < M2.n; j++){
            col[ind] = j;
            ++ind;
        }
    }

    int *rowind = new int[M1.m + 1];
    rowind[0] = 0;
    for (int i = 1; i <= M1.m; i++){
        rowind[i] = rowind[i-1] + M2.n * (M1.rowind[i] - M1.rowind[i-1] > 0);
    }

    return CSR_Matrix(M1.m, M2.n, nz, col, rowind, true);
}

CSR_Matrix make_fullrow(const CSR_Matrix &M){
    int n_nzrow = 0;
    int *rowind = new int[M.m + 1];
    rowind[0] = 0;
    for (int i = 0; i < M.m; i++){
        n_nzrow += ((M.rowind[i+1] - M.rowind[i]) > 0);
        rowind[i+1] = rowind[i] + M.n * ((M.rowind[i+1] - M.rowind[i]) > 0);
    }
    double *nz = new double[n_nzrow * M.n]();
    int *col = new int[n_nzrow * M.n];

    int ind = 0;
    for (int j = 0; j < M.n; j++){
        for (int i = 0; i < M.m; i++){
            col[ind] = i;
            ++ind;
        }

        for (int i = M.rowind[j]; i < M.rowind[j+1]; i++){
            nz[j*M.n + M.col[i]] = M.nz[i];
        }
    }

    return CSR_Matrix(M.m, M.n, nz, col, rowind, true);
}



} // namespace blockSQP
