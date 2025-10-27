/*
 * blockSQP -- Sequential quadratic programming for problems with
 *             block-diagonal Hessian matrix.
 * Copyright (C) 2012-2015 by Dennis Janka <dennis.janka@iwr.uni-heidelberg.de>
 *
 * Licensed under the zlib license. See LICENSE for more details.
 */

/*
 * blockSQP 2 -- Condensing, convexification strategies, scaling heuristics and more
 *               for blockSQP, the nonlinear programming solver by Dennis Janka.
 * Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
 * 
 * Licensed under the zlib license. See LICENSE for more details.
 */
 
 
/**
 * \file blocksqp_matrix.hpp
 * \author Dennis Janka, based on VPLAN's matrix.h by Stefan Koerkel
 * \date 2012-2015
 *
 *  Declaration of Matrix and SymMatrix classes.
 * 
 * \modifications
 *  \author Reinhold Wittmann
 *  \date 2023-2025
 *  Additional methods, sparse Matrix classes, LT_Block_Matrix class
 */

#ifndef BLOCKSQP_MATRIX_HPP
#define BLOCKSQP_MATRIX_HPP

#include "blocksqp_defs.hpp"
#include <iostream>
#include <vector>
#include <memory>

namespace blockSQP
{

extern int Ccount; ///< Count constructor calls
extern int Dcount; ///< Count destructor calls
extern int Ecount; ///< Count assign operator calls

/**
 * \brief Class for easy access of elements of a dense matrix.
 * \author Dennis Janka
 * \date 2012-2015
 */

class SymMatrix;

class Matrix
{  private:
      int malloc( void );                                           ///< memory allocation
      int free( void );                                             ///< memory free

   public:
      int m;                                                        ///< internal number of rows
      int n;                                                        ///< internal number of columns
      int ldim;                                                     ///< internal leading dimension not necesserily equal to m or n
      double *array;                                                ///< array of how the matrix is stored in the memory
      int tflag;                                                    ///< 1 if it is a Teilmatrix

      Matrix();
      Matrix( int, int = 1, int = -1 );                             ///< constructor with standard arguments
      Matrix( int, int, double*, int = -1 );
      Matrix( const Matrix& A );
      Matrix(const SymMatrix &A);
      //Matrix( Matrix&& M);
      virtual ~Matrix( void );
      
      int M( void ) const;                                          ///< number of rows
      int N( void ) const;                                          ///< number of columns
      int LDIM( void ) const;                                       ///< leading dimensions
      double *ARRAY( void ) const;                                  ///< returns pointer to data array
      double *release();                                            // Return pointer to data array, passing ownership
      int TFLAG( void ) const;                                      ///< returns this->tflag (1 if it is a submatrix and does not own the memory and 0 otherwise)

      virtual double &operator()(int i, int j);                     ///< access element i,j of the matrix
      virtual double &operator()(int i, int j) const;
      virtual double &operator()(int i);                            ///< access element i of the matrix (columnwise)
      virtual double &operator()(int i) const;
      virtual Matrix &operator=(const Matrix &A);                   ///< assignment operator
      virtual Matrix &operator=(const SymMatrix &A);
      
      //virtual void operator=( Matrix &&A );
      Matrix operator+(const Matrix &M2) const;
      Matrix operator-(const Matrix &M2) const;
      Matrix operator*(const Matrix &M2) const;
      Matrix operator*(const double alpha) const;
      void operator+=(const Matrix &M2);
      void operator-=(const Matrix &M2);
      void operator*=(const double alpha);

      virtual Matrix &Dimension( int, int = 1, int = -1 );          ///< set dimension (rows, columns, leading dimension)
      Matrix &Initialize(double (*)(int, int));                     ///< set matrix elements i,j to f(i,j)
      Matrix &Initialize(double);                                   ///< set all matrix elements to a constant

      /// Returns just a pointer to the full matrix
      Matrix& Submatrix( const Matrix&, int, int, int = 0, int = 0 );
      /// Matrix that points on <tt>ARRAY</tt>
      Matrix& Arraymatrix( int M, int N, double* ARRAY, int LDIM = -1 );

      //Matrix get_slice(int m_start, int m_end, int n_start, int n_end);
      Matrix get_slice(int m_start, int m_end, int n_start, int n_end) const;
      //Matrix get_slice(int m_start, int m_end);
      Matrix get_slice(int m_start, int m_end) const;
      //Matrix without_rows(int *starts, int *ends, int num_slices);
      Matrix without_rows(int *starts, int *ends, int num_slices) const;

      /** Flag == 0: bracket output
        * Flag == 1: Matlab output
        * else: plain output */
       
      const Matrix &Print( FILE* = stdout,   ///< file for output
                             int = 13,       ///< number of digits
                             int = 1         ///< Flag for format
                           ) const;
      
};

std::ostream& operator<<(std::ostream& os, const Matrix &M);
Matrix vertcat(std::vector<Matrix> Ms);

/**
 * \brief Class for easy access of elements of a dense symmetric matrix.
 * \author Dennis Janka
 * \date 2012-2015
 */
class SymMatrix{
    public:
        int m;
        int ldim;                                                        ///< size of the quadratic symmetric matrix
        double *array;                                                ///< array of how the matrix is stored in the memory
        int tflag;

    protected:
        int malloc( void );
        int free( void );

    public:
        SymMatrix();
        SymMatrix( int );
        SymMatrix( int, double*, int = -1);
        SymMatrix( int, int, int );
        SymMatrix( const Matrix& A );
        SymMatrix( const SymMatrix& A );
        virtual ~SymMatrix( void );

        virtual double &operator()( int i, int j );
        virtual double &operator()( int i, int j ) const;
        virtual double &operator()( int i );
        virtual double &operator()( int i ) const;

        SymMatrix &Dimension(int = 1);
        SymMatrix &Initialize(double (*)(int, int));
        SymMatrix &Initialize(double);

        SymMatrix& Submatrix(const Matrix&, int, int = 0);
        SymMatrix& Arraymatrix( int, double* );
        SymMatrix& Arraymatrix( int, double*, int = -1 );

        virtual SymMatrix &operator=(const SymMatrix &M2);
        SymMatrix operator+(const SymMatrix &M2) const;
        SymMatrix operator*(const double alpha) const;


        Matrix get_slice(int m_start, int m_end, int n_start, int n_end) const;

        
        const SymMatrix &Print( FILE* = stdout,   ///< file for output
                             int = 13,       ///< number of digits
                             int = 1         ///< Flag for format
                           ) const;
        
};

std::ostream& operator<<(std::ostream& os, const SymMatrix &M);


Matrix Transpose( const Matrix& A); ///< Overwrites \f$ A \f$ with its transpose \f$ A^T \f$
Matrix &Transpose( const Matrix &A, Matrix &T ); ///< Computes \f$ T = A^T \f$
double delta( int, int );


class CSR_Matrix;

class Sparse_Matrix{
	public:
		int m;
		int n;
		//int nnz;
		std::unique_ptr<double[]> nz;
		std::unique_ptr<int[]> row;
		std::unique_ptr<int[]> colind;
    //bool memoryOwned;

    Sparse_Matrix(int M, int N, int NNZ);
    Sparse_Matrix(int M, int N, std::unique_ptr<double[]> NZ, std::unique_ptr<int[]> ROW, std::unique_ptr<int[]> COLIND);
		Sparse_Matrix(const Sparse_Matrix &M);
		Sparse_Matrix(Sparse_Matrix &&M);
		Sparse_Matrix(const CSR_Matrix &M);
		Sparse_Matrix();
		//~Sparse_Matrix();
    
    Sparse_Matrix &Dimension(int M, int N, int NNZ);
    
		void operator=(const Sparse_Matrix &M);
		void operator=(Sparse_Matrix &&M);
		Sparse_Matrix operator+(const Sparse_Matrix &M2) const;
		//Sparse_Matrix get_slice(int m_start, int m_end, int n_start, int n_end);
		Sparse_Matrix get_slice(int m_start, int m_end, int n_start, int n_end) const;
		Matrix get_dense_slice(int m_start, int m_end, int n_start, int n_end) const;
		Matrix dense() const;
		void remove_rows(int *starts, int *ends, int nblocks);
		Sparse_Matrix without_nz_rows(int *starts, int *ends, int nblocks) const;
};

Sparse_Matrix sparse_dense_multiply(const Sparse_Matrix &M1, const Matrix &M2);
Matrix sparse_vector_multiply(const Sparse_Matrix &M1, const Matrix &V1);
Matrix transpose_multiply(const Sparse_Matrix &M1, const Matrix &M2);
Sparse_Matrix horzcat(Sparse_Matrix*, int);
Sparse_Matrix horzcat(std::vector<Sparse_Matrix>&);
Sparse_Matrix lr_zero_pad(int N, const Sparse_Matrix &M1, int start);
Sparse_Matrix lr_zero_pad(int N, const Matrix &M1, int start);
Sparse_Matrix vertcat(std::vector<Sparse_Matrix>&);


class CSR_Matrix{
public:
    CSR_Matrix(int M, int N, double* NZ, int* COL, int* ROWIND, bool FD = false);
    CSR_Matrix(const CSR_Matrix &M1);
    CSR_Matrix(CSR_Matrix &&M1);
    CSR_Matrix(const Sparse_Matrix &M1);
    CSR_Matrix();
    ~CSR_Matrix();
    void operator=(const CSR_Matrix &M1);
    void operator=(CSR_Matrix &&M1);

    int m;
    int n;
    double *nz;
    int *col;
    int *rowind;
    bool free_data;

    Matrix dense() const;
};


//Lower-Triangle block-matrix
class LT_Block_Matrix{
	private:
		int malloc( void );
	public:
		int *m_block_sizes;
		int *n_block_sizes;
		int m;
		int n;
		Matrix *array;

		LT_Block_Matrix(int, int*, int*);
		LT_Block_Matrix(int, int, int*, int*);
		LT_Block_Matrix();
		~LT_Block_Matrix(void);

		void set(int i, int j, const Matrix &M);
		const Matrix &operator() (int i, int j) const;
		LT_Block_Matrix &operator=(const LT_Block_Matrix &M);
		LT_Block_Matrix &Dimension(int, int, int*, int*);
		void to_dense(Matrix &M) const;
		void to_sym(SymMatrix &M) const;

		//UNTESTED
		void to_sparse(Sparse_Matrix &M) const;
};


void CSC_CSR(const int m, const int n, const double* nz_CSC, const int* row, const int* colind, double* const nz_CSR, int* const col, int* const rowind);
CSR_Matrix sparse_dense_multiply_2(const Sparse_Matrix &M1, const Matrix &M2);
CSR_Matrix fullrow_multiply(const CSR_Matrix &M1, const Matrix &M2);

CSR_Matrix add_fullrow(const CSR_Matrix &M1, const CSR_Matrix &M2);
CSR_Matrix make_fullrow(const CSR_Matrix &M);




} // namespace blockSQP
#endif
