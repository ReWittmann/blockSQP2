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

#ifndef BLOCKSQP2_MATRIX_HPP
#define BLOCKSQP2_MATRIX_HPP

#include <blockSQP2/defs.hpp>
#include <blockSQP2/lapackblas.hpp>
#include <iosfwd>
#include <vector>
#include <memory>


namespace blockSQP2{

class SymMatrix;

/**
 * \brief Class for easy access of elements of a dense matrix.
 * \author Dennis Janka
 * \date 2012-2015
 */
class Matrix{  
  private:
    void allocate(void);
    void deallocate(void);

  public:
    int m;                                                        ///< internal number of rows
    int n;                                                        ///< internal number of columns
    int ldim;                                                     ///< internal leading dimension not necesserily equal to m or n
    double *array;                                                ///< array of how the matrix is stored in the memory
    int tflag;                                                    ///< 1 if it is a Teilmatrix

    Matrix();
    Matrix(int, int = 1, int = -1);                             ///< constructor with standard arguments
    Matrix( int, int, double*, int = -1 );
    Matrix( const Matrix& A );
    Matrix(const SymMatrix &A);
    virtual ~Matrix();
    
    inline int M() const {return m;}
    inline int N() const {return n;}
    inline int LDIM() const {return ldim;}
    inline double *ARRAY() const {return array;}
    inline int TFLAG() const {return tflag;}

    double *release();
    
    inline double &operator()(int i, int j) const;
    inline double &operator()(int i) const;
    
    Matrix &operator=(const Matrix &A);
    Matrix &operator=(Matrix &&A);
    Matrix &operator=(const SymMatrix &A);
    
    //virtual void operator=( Matrix &&A );
    Matrix operator+(const Matrix &M2) const;
    Matrix operator-(const Matrix &M2) const;
    Matrix operator*(const Matrix &M2) const;
    Matrix operator*(const double alpha) const;
    void operator+=(const Matrix &M2);
    void operator-=(const Matrix &M2);
    void operator*=(const double alpha) const;

    inline Matrix &Dimension(int M, int N = 1, int LDIM = -1);
    
    Matrix &Initialize(double (*)(int, int));
    inline const Matrix &Initialize(double val) const;
    
    /// Returns just a pointer to the full matrix
    Matrix& Submatrix( const Matrix&, int, int, int = 0, int = 0 );
    /// Matrix that points on <tt>ARRAY</tt>
    Matrix& Arraymatrix( int M, int N, double* ARRAY, int LDIM = -1 );

    Matrix get_slice(int m_start, int m_end, int n_start, int n_end) const;
    Matrix get_slice(int m_start, int m_end) const;
    void get_slice(int m_start, int m_end, int n_start, int n_end, Matrix &A) const;
    void get_slice(int m_start, int m_end, Matrix &A) const;
    
    //Matrix without_rows(int *starts, int *ends, int num_slices);
    Matrix without_rows(int *starts, int *ends, int num_slices) const;
    Matrix without_rows(int const *row_indices, int row_indices_l) const;
    
    /** Flag == 0: bracket output
      * Flag == 1: Matlab output
      * else: plain output */
      
    const Matrix &Print( FILE* = stdout,   ///< file for output
                            int = 13,       ///< number of digits
                            int = 1         ///< Flag for format
                          ) const;
      
};

std::ostream& operator<<(std::ostream& os, const Matrix &M);
Matrix vertcat(std::vector<Matrix> mats);
Matrix vertcat(Matrix const *mats, int mats_l);

inline void mult(const Matrix &M1, const Matrix &M2, Matrix &M3);
inline void Tmult(const Matrix &M1, const Matrix &M2, Matrix &M3);
inline void mult_to(const Matrix &M1, const Matrix &M2, const Matrix &M3, double beta = 1.0);
inline void Tmult_to(const Matrix &M1, const Matrix &M2, const Matrix &M3, double beta = 1.0);


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
        void allocate();
        void deallocate();

    public:
        SymMatrix();
        SymMatrix(int);
        SymMatrix(int, double*, int = -1);
        SymMatrix(int, int, int);
        SymMatrix(const Matrix& A);
        SymMatrix(const SymMatrix& A);
        virtual ~SymMatrix();
        
        inline double &operator()(int i, int j) const;
        inline double &operator()(int i) const;
        
        inline SymMatrix &Dimension(int M = 1);
        
        SymMatrix &Initialize(double (*)(int, int));
        SymMatrix &Initialize(double);

        SymMatrix& Submatrix(const Matrix&, int, int = 0);
        SymMatrix& Arraymatrix(int, double*);
        SymMatrix& Arraymatrix(int, double*, int = -1);

        SymMatrix &operator=(const SymMatrix &M2);
        SymMatrix operator+(const SymMatrix &M2) const;
        SymMatrix operator*(const double alpha) const;


        Matrix get_slice(int m_start, int m_end, int n_start, int n_end) const;
        void get_slice(int m_start, int m_end, int n_start, int n_end, Matrix &A) const;

        
        const SymMatrix &Print(FILE* = stdout,   ///< file for output
                             int = 13,       ///< number of digits
                             int = 1         ///< Flag for format
                           ) const;
        
};

std::ostream& operator<<(std::ostream& os, const SymMatrix &M);


Matrix Transpose(const Matrix& A);
Matrix &Transpose(const Matrix &A, Matrix &T);
double delta(int, int);


class CSR_Matrix;

class Sparse_Matrix{
	public:
		int m;
		int n;
		std::unique_ptr<double[]> nz;
		std::unique_ptr<int[]> row;
		std::unique_ptr<int[]> colind;

    Sparse_Matrix(int M, int N, int NNZ);
    Sparse_Matrix(int M, int N, std::unique_ptr<double[]> NZ, std::unique_ptr<int[]> ROW, std::unique_ptr<int[]> COLIND);
		Sparse_Matrix(const Sparse_Matrix &M);
		Sparse_Matrix(Sparse_Matrix &&M);
		Sparse_Matrix(const CSR_Matrix &M);
		Sparse_Matrix();
    
    inline int nnz(){return colind[n];}
    Sparse_Matrix &Dimension(int M, int N, int NNZ);
    
		Sparse_Matrix &operator=(const Sparse_Matrix &M);
		Sparse_Matrix &operator=(Sparse_Matrix &&M);
		Sparse_Matrix operator+(const Sparse_Matrix &M2) const;
		Sparse_Matrix get_slice(int m_start, int m_end, int n_start, int n_end) const;
    
    void get_slice(int m_start, int m_end, int n_start, int n_end, Sparse_Matrix &A) const;
    void get_slice(int m_start, int m_end, int n_start, int n_end, Matrix &A) const;
    
		Matrix get_dense_slice(int m_start, int m_end, int n_start, int n_end) const;
    double operator()(int i, int j = 0) const;
		Matrix dense() const;
		void remove_rows(int *starts, int *ends, int nblocks);
		// Sparse_Matrix without_nz_rows(int *starts, int *ends, int nblocks) const;
    Sparse_Matrix without_rows(int const *row_indices, int row_indices_l) const;
};

Sparse_Matrix sparse_dense_multiply(const Sparse_Matrix &M1, const Matrix &M2);
Matrix sparse_vector_multiply(const Sparse_Matrix &M1, const Matrix &V1);
Matrix transpose_multiply(const Sparse_Matrix &M1, const Matrix &M2);
Sparse_Matrix horzcat(Sparse_Matrix* mats, int n_mats);
Sparse_Matrix horzcat(std::vector<Sparse_Matrix>& mats);
Sparse_Matrix lr_zero_pad(int N, const Sparse_Matrix &M1, int start);
Sparse_Matrix lr_zero_pad(int N, const Matrix &M1, int start);
Sparse_Matrix vertcat(std::vector<Sparse_Matrix>& mats);
Sparse_Matrix vertcat(Sparse_Matrix const *mats, int mats_l);

class CSR_Matrix{
public:
    CSR_Matrix(int M, int N, double* NZ, int* COL, int* ROWIND, bool FD = false);
    CSR_Matrix(const CSR_Matrix &M1);
    CSR_Matrix(CSR_Matrix &&M1);
    CSR_Matrix(const Sparse_Matrix &M1);
    CSR_Matrix();
    ~CSR_Matrix();
    CSR_Matrix &operator=(const CSR_Matrix &M1);
    CSR_Matrix &operator=(CSR_Matrix &&M1);

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
		void allocate();
	public:
		int *m_block_sizes;
		int *n_block_sizes;
		int m;
		int n;
		Matrix *array;

		LT_Block_Matrix(int, int*, int*);
		LT_Block_Matrix(int, int, int*, int*);
		LT_Block_Matrix();
		~LT_Block_Matrix();

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





#include <blockSQP2/inline/matrix.ipp>

} // namespace blockSQP2
#endif
