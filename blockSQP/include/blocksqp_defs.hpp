/*
 * blockSQP -- Sequential quadratic programming for problems with
 *             block-diagonal Hessian matrix.
 * Copyright (C) 2012-2015 by Dennis Janka <dennis.janka@iwr.uni-heidelberg.de>
 *
 * Licensed under the zlib license. See LICENSE for more details.
 */

/**
 * \file blocksqp_defs.hpp
 * \author Dennis Janka
 * \date 2012-2015
 *
 *  Declaration of all constants and inclusion of standard header files.
 */

#ifndef BLOCKSQP_DEFS_H
#define BLOCKSQP_DEFS_H

//#include "math.h"
#include <cmath>
#include "stdio.h"
#include "string.h"
#include <set>
#include <string>
#include <stdexcept>

namespace blockSQP{

typedef char PATHSTR[4096];

enum class RES{
    IT_FINISHED = 0,
    FEAS_SUCCESS = 1,
    success = 2,
    SUPER_SUCCESS = 3,
    LOCAL_INFEASIBILITY = -1,
    RESTORATION_FAILURE = -2,
    LINESEARCH_FAILURE = -3,
    QP_FAILURE = -4,
    EVAL_FAILURE = -5,
    MISC_ERROR = -10,
};
//Colored print output when exiting with  return print_RES(RES::__)
RES print_RES(RES rs);


class NotImplementedError : public std::logic_error{
public:
    NotImplementedError(std::string info) : std::logic_error("Missing implementation of " + info){}
};


class ParameterError : public std::logic_error{
public:
    ParameterError(std::string info) : std::logic_error(info){}
};




} // namespace blockSQP







#endif
