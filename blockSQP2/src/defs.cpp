/*
 * blockSQP2 -- A structure-exploiting nonlinear programming solver based
 *              on blockSQP by Dennis Janka.
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

#include <blockSQP2/defs.hpp>
#include <iostream>


namespace blockSQP2{

SQPresults print_SQPresult(SQPresults rs, int print_level){
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
            case SQPresults::partial_success: 
                std::cout << colPrefix + "\n***CONVERGENCE PARTIALLY ACHIEVED***" + colSuffix + "\n"; 
                break;
            case SQPresults::success:
                std::cout << colPrefix + "\n***CONVERGENCE ACHIEVED***" + colSuffix + "\n";
                break;
            case SQPresults::super_success:
                std::cout << colPrefix + "\n***STRONG CONVERGENCE ACHIEVED***" + colSuffix + "\n";
                break;
            case SQPresults::local_infeasibility:
                std::cout << colPrefix + "\nLOCAL INFEASIBILITY" + colSuffix + "\n";
                break;
            case SQPresults::restoration_failure:
                std::cout << colPrefix + "\nRESTORATION ERROR" + colSuffix + "\n";
                break;
            case SQPresults::linesearch_failure:
                std::cout << colPrefix + "\nLINESEARCH ERROR" + colSuffix + "\n";
                break;
            case SQPresults::sensitivity_eval_failure:
                std::cout << colPrefix + "\nSENSITIVITY EVALUATION ERROR" + colSuffix + "\n";
                break;
            default:
                std::cout << colPrefix + "\nNLP SOLUTION UNSUCCESSFUL" + colSuffix + "\n";
        }
    }
    return rs;
}

std::string to_string(Hessians hess_kind){
    switch (hess_kind){
        case Hessians::scaled_ID:       return "scaled_ID";
        case Hessians::SR1:             return "SR1";
        case Hessians::BFGS:            return "BFGS";
        case Hessians::finite_diff:     return "finite_diff";
        case Hessians::exact:           return "exact";
        case Hessians::pos_def_exact:   return "pos_def_exact";
        case Hessians::undamped_BFGS:   return "undamped_BFGS";
        case Hessians::last_block_default: break;
    }
    return "";
}

std::string to_print_string(Hessians hess_kind){
    switch (hess_kind){
        case Hessians::scaled_ID:       return "Scaled ID";
        case Hessians::SR1:             return "SR1";
        case Hessians::BFGS:            return "BFGS";
        case Hessians::finite_diff:     return "Finite differences";
        case Hessians::exact:           return "exact";
        case Hessians::pos_def_exact:   return "Pos. def. exact";
        case Hessians::undamped_BFGS:   return "Undamped BFGS";
        case Hessians::last_block_default: break;
    }
    return "";
}

Hessians Hessians_from_string(std::string_view Hname){
    if (Hname == to_string(Hessians::scaled_ID))
        return Hessians::scaled_ID;
    if (Hname == to_string(Hessians::SR1))
        return Hessians::SR1;
    if (Hname == to_string(Hessians::BFGS))
        return Hessians::BFGS;
    if (Hname == to_string(Hessians::finite_diff))
        return Hessians::finite_diff;
    if (Hname == to_string(Hessians::exact))
        return Hessians::exact;
    if (Hname == to_string(Hessians::pos_def_exact))
        return Hessians::pos_def_exact;
    if (Hname == to_string(Hessians::undamped_BFGS))
        return Hessians::undamped_BFGS;
    if (Hname == to_string(Hessians::last_block_default))
        return Hessians::last_block_default;
    throw std::invalid_argument(std::string("Hessians_from_string: Name \"") + std::string(Hname) + std::string("\" does not match any available Hessian-approximation"));
}


std::string to_string(Sizings sizing){
    switch (sizing){
        case Sizings::None:     return "None";
        case Sizings::SP:       return "SP";
        case Sizings::OL:       return "OL";
        case Sizings::GM_SP_OL: return "GM_OL_SP";
        case Sizings::COL:      return "COL";
    }
    return "";
}
std::string to_string_full(Sizings sizing){
    switch (sizing){
        case Sizings::None:     return "None";
        case Sizings::SP:       return "Shanno-Phua";
        case Sizings::OL:       return "Oren-Luenberger";
        case Sizings::GM_SP_OL: return "Geometric mean of Shanno-Phua and Oren-Luenberger";
        case Sizings::COL:      return "Centered Oren-Luenberger";
    }
    return "";
}
std::string to_print_string(Sizings sizing){
    switch (sizing){
        case Sizings::None:     return "none";
        case Sizings::SP:       return "SP";
        case Sizings::OL:       return "OL";
        case Sizings::GM_SP_OL: return "mean";
        case Sizings::COL:      return "selective sizing";
    }
    return "";
}


Sizings Sizings_from_string(std::string_view Sname){
    if (Sname == to_string(Sizings::None))
        return Sizings::None;
    if (Sname == to_string(Sizings::SP))
        return Sizings::SP;
    if (Sname == to_string(Sizings::OL))
        return Sizings::OL;
    if (Sname == to_string(Sizings::GM_SP_OL))
        return Sizings::GM_SP_OL;
    if (Sname == to_string(Sizings::COL))
        return Sizings::COL;
    throw std::invalid_argument(std::string("Sizings_from_string: Name \"") + std::string(Sname) + std::string("\" does not match any available sizing strategy"));
}

} // namespace blockSQP2
