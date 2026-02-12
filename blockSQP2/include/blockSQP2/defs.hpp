/*
 * blockSQP -- Sequential quadratic programming for problems with
 *             block-diagonal Hessian matrix.
 * Copyright (C) 2012-2015 by Dennis Janka <dennis.janka@iwr.uni-heidelberg.de>
 *
 * Licensed under the zlib license. See LICENSE for more details.
 */

/*
 * blockSQP 2 -- Condensing, convexification strategies, scaling heuristics and more
 *               for blockSQP, the nonlinear programming solver by Dennis Janka.
 * Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
 * 
 * Licensed under the zlib license. See LICENSE for more details.
 */
 
 
/**
 * \file blocksqp_defs.hpp
 * \author Dennis Janka
 * \date 2012-2015
 *
 *  Declaration of all constants and inclusion of standard header files.
 * 
 * \modifications
 *  \author Reinhold Wittmann
 *  \date 2023-2025
 */

#ifndef BLOCKSQP2_DEFS_HPP
#define BLOCKSQP2_DEFS_HPP

//#include "math.h"
#include <cmath>
#include "stdio.h"
#include "string.h"
#include <set>
#include <string>
#include <stdexcept>
#include <vector>
#include <chrono>

namespace blockSQP2{

typedef char PATHSTR[4096];

enum class SQPresults{
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
SQPresults print_SQPresult(SQPresults rs, int print_level = 2);

enum class Hessians : int{
    scaled_ID = 0,
    SR1 = 1,
    BFGS = 2,
    
    finite_diff = 4,
    //pos_def_exact = 5,
    //undamped_BFGS = 6,
    exact = 5,
    pos_def_exact = 6,
    undamped_BFGS = 7,
    
    last_block_default = -100
    
    /*
    exact = 5,
    pos_def_exact = 6,
    */
};

//Identity Hessian may still be "updated" through sizing
inline bool is_update(Hessians appr){
    return appr == Hessians::scaled_ID || appr == Hessians::SR1 || appr == Hessians::BFGS || appr == Hessians::undamped_BFGS;
}
inline bool is_indefinite(Hessians appr){
    return appr == Hessians::SR1 || appr == Hessians::finite_diff || appr == Hessians::exact || appr == Hessians::undamped_BFGS;
}
inline bool is_exact(Hessians appr){
    return appr == Hessians::exact || appr == Hessians::pos_def_exact;
}

std::string to_string(Hessians hess_kind);
std::string to_print_string(Hessians hess_kind);
Hessians Hessians_from_string(std::string_view Hname);

enum class Sizings : int{
    None = 0,
    SP = 1,                     //Shanno-Phua sizing
    OL = 2,                     //Oren-Luenberger sizing
    GM_SP_OL = 3,               //Geometric mean of SP and OL
    COL = 4                     //Centered Oren-Luenberger sizing
};
std::string to_string(Sizings sizing);
std::string to_string_full(Sizings sizing);
std::string to_print_string(Sizings sizing);
Sizings Sizings_from_string(std::string_view Sname);


class NotImplementedError : public std::logic_error{
public:
    NotImplementedError(std::string info) : std::logic_error("Missing implementation of " + info){}
};


class ParameterError : public std::logic_error{
public:
    ParameterError(std::string info) : std::logic_error(info){}
};



#define PAR_QP_MAX 8

} // namespace blockSQP2







#endif
