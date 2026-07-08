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


#endif