/*
 * blockSQP 2 -- Condensing, convexification strategies, scaling heuristics and more
 *               for blockSQP, the nonlinear programming solver by Dennis Janka.
 * Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
 * 
 * Licensed under the zlib license. See LICENSE for more details.
 */

/**
 * \file blocksqp_defs.cpp
 * \author Reinhold Wittmann
 * \date 2023-2025
 *
 *  A result print utility
 */

#include "blocksqp_defs.hpp"
#include <iostream>


namespace blockSQP{

SQPresult print_SQPresult(SQPresult rs, int print_level){
    if (print_level > 0){
        std::string colPrefix;
        std::string colSuffix;
        if (print_level == 2){
            #ifdef LINUX
                if (int(rs) < 0) colPrefix = "\033[1;31m";
                else colPrefix = "\033[1;32m";
                colSuffix = "\033[0m";
            #endif
        }
        
        switch (rs){
            case SQPresult::partial_success: 
                std::cout << colPrefix + "\n***CONVERGENCE PARTIALLY ACHIEVED***" + colSuffix + "\n"; 
                break;
            case SQPresult::success:
                std::cout << colPrefix + "\n***CONVERGENCE ACHIEVED***" + colSuffix + "\n";
                break;
            case SQPresult::super_success:
                std::cout << colPrefix + "\n***STRONG CONVERGENCE ACHIEVED***" + colSuffix + "\n";
                break;
            case SQPresult::local_infeasibility:
                std::cout << colPrefix + "\nLOCAL INFEASIBILITY" + colSuffix + "\n";
                break;
            case SQPresult::restoration_failure:
                std::cout << colPrefix + "\nRESTORATION ERROR" + colSuffix + "\n";
                break;
            case SQPresult::linesearch_failure:
                std::cout << colPrefix + "\nLINESEARCH ERROR" + colSuffix + "\n";
                break;
            case SQPresult::sensitivity_eval_failure:
                std::cout << colPrefix + "\nSENSITIVITY EVALUATION ERROR" + colSuffix + "\n";
                break;
            default:
                std::cout << colPrefix + "\nNLP SOLUTION UNSUCCESSFUL" + colSuffix + "\n";
        }
    }
    return rs;
}


}
