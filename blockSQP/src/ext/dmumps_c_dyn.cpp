
#include "dmumps_c.h"

#ifdef WINDOWS
    #define dl_exp __declspec__(dllexpor)
#else
    #define dl_exp
#endif


extern "C" dl_exp void dmumps_c_dyn(void *mumps_struc_c_dyn){
    dmumps_c((DMUMPS_STRUC_C*) mumps_struc_c_dyn);
}
