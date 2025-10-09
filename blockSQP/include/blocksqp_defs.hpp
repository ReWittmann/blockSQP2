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

#ifndef BLOCKSQP_DEFS_HPP
#define BLOCKSQP_DEFS_HPP

//#include "math.h"
#include <cmath>
#include "stdio.h"
#include "string.h"
#include <set>
#include <string>
#include <stdexcept>
#include <vector>
#include <chrono>

namespace blockSQP{

typedef char PATHSTR[4096];

enum class SQPresult{
    it_finished = 0,
    partial_success = 1,
    success = 2,
    super_success = 3,
    local_infeasibility = -1,
    restoration_failure = -2,
    linesearch_failure = -3,
    qp_failure = -4,
    eval_failure = -5,
    misc_error = -10,
    sensitivity_eval_failure = -100
};
//Colored print output when exiting with  return print_RES(RES::__)
SQPresult print_SQPresult(SQPresult rs, int print_level = 2);


enum class QPresult{
    success = 0,
    time_it_limit_reached,
    indef_unbounded,
    infeasible,
    other_error
};

class NotImplementedError : public std::logic_error{
public:
    NotImplementedError(std::string info) : std::logic_error("Missing implementation of " + info){}
};


class ParameterError : public std::logic_error{
public:
    ParameterError(std::string info) : std::logic_error(info){}
};



#define PAR_QP_MAX 8

} // namespace blockSQP







#endif
