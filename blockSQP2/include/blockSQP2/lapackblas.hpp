#ifndef BLOCKSQP2_LAPACKBLAS_HPP
#define BLOCKSQP2_LAPACKBLAS_HPP

#include <cblas.h>

#ifndef OPENBLAS_CONFIG_H
	#if defined(CBLAS_INT)
		typedef CBLAS_INT blasint;
	#elif defined(MKL_INT)
		typedef MKL_INT blasint;
	#else
		#error "Included cblas header defines neither blasint nor CBLAS_INT nor MKL_INT"
	#endif
#endif

#ifndef BSQP_CBLAS_SUFFIX
    #define BSQP_CBLAS_SUFFIX
#endif

#ifndef BSQP_CBLAS_PREFIX
    #define BSQP_CBLAS_PREFIX
#endif

#define BSQP_CONCAT(a,b) a##b
#define BSQP_EXPAND_CONCAT(a,b) BSQP_CONCAT(a,b)
#define BSQP_BLASFUNC(a) BSQP_EXPAND_CONCAT(BSQP_CBLAS_PREFIX, BSQP_EXPAND_CONCAT(a, BSQP_CBLAS_SUFFIX)) // #####!!! READ THIS !!!#####: If this is in an error message: Invalid cblas suffix. Check cblas.h, and use -DBSQP_CBLAS_SUFFIX= instead of -DBSQP_CBLAS_SUFFIX if there is no suffix.


#ifndef BSQP_CBLAS_ARG_PREFIX
	#define BSQP_CBLAS_ARG_PREFIX
#endif
#define BSQP_BLASARG(a) BSQP_EXPAND_CONCAT(BSQP_CBLAS_ARG_PREFIX, a)


#ifdef BSQP_BLASFEO
	#include <blasfeo.h>
	#define BSQP_CBLAS_DGEMM(LT, TA, TB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc) \
		blasfeo_cblas_dgemm(BSQP_EXPAND_CONCAT(Blasfeo, LT), BSQP_EXPAND_CONCAT(Blasfeo, TA), BSQP_EXPAND_CONCAT(Blasfeo, TB), blasint(M), blasint(N), blasint(K), double(alpha), A, blasint(lda), B, blasint(ldb), double(beta), C, blasint(ldc))

#else
	#define BSQP_CBLAS_DGEMM(LT, TA, TB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc) \
		BSQP_BLASFUNC(cblas_dgemm)(BSQP_BLASARG(LT), BSQP_BLASARG(TA), BSQP_BLASARG(TB), blasint(M), blasint(N), blasint(K), double(alpha), A, blasint(lda), B, blasint(ldb), double(beta), C, blasint(ldc))

#endif


// extern "C" void blasfeo_cblas_dgemm(
// 		BLASFEO_CBLAS_LAYOUT layout,
// 		BLASFEO_CBLAS_TRANSPOSE TransA, BLASFEO_CBLAS_TRANSPOSE TransB,
// 		int M, int N, int K, double alpha, const double *A, int lda,
// 		const double *B, int ldb, double beta, double *C, int ldc);





#endif