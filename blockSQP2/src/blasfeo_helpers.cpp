#include <blockSQP2/lapackblas.hpp>
#include <algorithm>
#include <string>
#include <stdexcept>

namespace blockSQP2{

#ifdef BSQP_BLASFEO

inline char enumTranspose_to_char(BLASFEO_CBLAS_TRANSPOSE trans){
    switch (trans) {
    case BlasfeoCblasNoTrans:   return 'N';
    case BlasfeoCblasTrans:     return 'T';
    case BlasfeoCblasConjTrans: return 'C';
    default:                   return '\0';
    }
}

extern "C" void blasfeo_cblas_dgemm(BLASFEO_CBLAS_LAYOUT layout, BLASFEO_CBLAS_TRANSPOSE TransA, BLASFEO_CBLAS_TRANSPOSE TransB,
		int M, int N, int K, double alpha, const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc){
    char ta = enumTranspose_to_char(TransA);
    char tb = enumTranspose_to_char(TransB);
    const bool row_major = layout == BlasfeoCblasRowMajor;
    
    int invalid = 0;
    if (!row_major && layout != BlasfeoCblasColMajor) invalid = 1;
    else if (!ta) invalid = 2;
    else if (!tb) invalid = 3;
    else if (M < 0) invalid = 4;
    else if (N < 0) invalid = 5;
    else if (K < 0) invalid = 6;
    else if (lda < std::max(1, row_major ? (ta == 'N' ? K : M) : (ta == 'N' ? M : K))) invalid = 9;
    else if (ldb < std::max(1, row_major ? (tb == 'N' ? N : K) : (tb == 'N' ? K : N))) invalid = 11;
    else if (ldc < std::max(1, row_major ? N : M)) invalid = 14;
    
    if (invalid) throw std::invalid_argument("wrapper - blasfeo_cblas_dgemm: Invalid arguments, error = " + std::to_string(invalid));
    if (M == 0 || N == 0) return;
    
    if (row_major) {
        std::swap(M, N);
        std::swap(A, B);
        std::swap(lda, ldb);
        std::swap(ta, tb);
    }
    
    if (alpha == 0.0 || K == 0) {
        if (beta == 1.0) return;
        for (int j = 0; j < N; ++j) {
            double *column = C + static_cast<std::ptrdiff_t>(j) * ldc;
            for (int i = 0; i < M; ++i)
                column[i] = beta == 0.0 ? 0.0 : beta * column[i];
        }
        return;
    }
    
    blasfeo_blas_dgemm(&ta, &tb, &M, &N, &K, &alpha,
                       const_cast<double *>(A), &lda,
                       const_cast<double *>(B), &ldb, &beta, C, &ldc);
}

#endif

}