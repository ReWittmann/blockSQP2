
#ifndef QPOASES_LAPACKBLAS_HPP
#define QPOASES_LAPACKBLAS_HPP


#include "cblas.h"
#include "lapack.h"

#ifndef QPOS_CBLAS_SUFFIX
    #define QPOS_CBLAS_SUFFIX
#endif
#define QPOS_CONCAT(a, b) a##b
#define QPOS_EXPAND_CONCAT(a, b) QPOS_CONCAT(a, b)
#define QPOS_BLASFUNC(a) QPOS_EXPAND_CONCAT(a, QPOS_CBLAS_SUFFIX)


#ifdef __USE_SINGLE_PRECISION__
	#define CBLAS__GEMM QPOS_BLASFUNC(cblas_sgemm)

	#define POTRF spotrf_
	#define TRTRS strtrs_
	#define TRCON strcon_
#else
	#define CBLAS__GEMM QPOS_BLASFUNC(cblas_dgemm)

	#define POTRF dpotrf_
	#define TRTRS dtrtrs_
	#define TRCON dtrcon_
#endif // __USE_SINGLE_PRECISION__

#ifdef LAPACK_FORTRAN_STRLEN_END
	#define ST_EXPAND1(l1) , size_t(l1)
	#define STRLENS1(l1) ST_EXPAND1(l1)

	#define ST_EXPAND2(l1, l2) , size_t(l1), size_t(l2)
	#define STRLENS2(l1, l2) ST_EXPAND2(l1, l2)

	#define ST_EXPAND3(l1, l2, l3) , size_t(l1), size_t(l2), size_t(l3)
	#define STRLENS3(l1, l2, l3) ST_EXPAND3(l1, l2, l3)
#else
	#define STRLENS1(l1, l2, l3) 
	#define STRLENS2(l1, l2, l3) 
	#define STRLENS3(l1, l2, l3) 
#endif




#endif