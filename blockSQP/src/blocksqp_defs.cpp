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
                std::cout << colPrefix + "\n***CONVERGENCE PARTIALLY ACHIEVED***\n" + colSuffix; 
                break;
            case SQPresult::success:
                //std::cout << "\033[1;32m" << "\n***CONVERGENCE ACHIEVED***\n" << "\033[0m";
                std::cout << colPrefix + "\n***CONVERGENCE ACHIEVED***\n" + colSuffix;
                break;
            case SQPresult::super_success:
                std::cout << colPrefix + "\n***STRONG CONVERGENCE ACHIEVED***\n" + colSuffix;
                break;
            case SQPresult::local_infeasibility:
                std::cout << colPrefix + "\nLOCAL INFEASIBILITY\n" + colSuffix;
                break;
            case SQPresult::restoration_failure:
                std::cout << colPrefix + "\nRESTORATION ERROR\n" + colSuffix;
                break;
                case SQPresult::linesearch_failure:
                std::cout << colPrefix + "\nLINESEARCH ERROR\n" + colSuffix;
                break;
            default:
                std::cout << colPrefix + "\nNLP SOLUTION UNSUCCESSFUL\n" + colSuffix;
        }
    }
    return rs;
}


}
