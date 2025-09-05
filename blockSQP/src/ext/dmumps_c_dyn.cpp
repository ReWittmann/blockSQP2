
#include "dmumps_c.h"

#ifdef _MSC_VER
    #define CDLEXP extern "C" __declspec(dllexport)
#else
    #define CDLEXP extern "C" __attribute__((visibility("default")))
#endif


CDLEXP void dmumps_c_dyn(void *mumps_struc_c_dyn){
    dmumps_c((DMUMPS_STRUC_C*) mumps_struc_c_dyn);
}
