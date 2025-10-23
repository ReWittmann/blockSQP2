/*
 * blockSQP 2 -- Condensing, convexification strategies, scaling heuristics and more
 *               for blockSQP, the nonlinear programming solver by Dennis Janka.
 * Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
 * 
 * Licensed under the zlib license. See LICENSE for more details.
 */

/**
 * \file dmumps_c_dyn.hpp
 * \author Reinhold Wittmann
 * \date 2023-2025
 *
 * Wrapper for dynamic loading of the sparse linear solver MUMPS
 */


#include "dmumps_c.h"

#ifdef _MSC_VER
    #define CDLEXP extern "C" __declspec(dllexport)
#else
    #define CDLEXP extern "C" __attribute__((visibility("default")))
#endif


CDLEXP void dmumps_c_dyn(void *mumps_struc_c_dyn){
    dmumps_c((DMUMPS_STRUC_C*) mumps_struc_c_dyn);
}
