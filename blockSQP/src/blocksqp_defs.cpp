#include "blocksqp_defs.hpp"
#include <iostream>


namespace blockSQP{

SQPresult loud(SQPresult rs, bool enabled){
    if (enabled){
        switch (rs){
            case SQPresult::partial_success: 
                std::cout << "\033[1;32m" << "\n***CONVERGENCE PARTIALLY ACHIEVED***\n" << "\033[0m"; 
                break;
            case SQPresult::success:
                std::cout << "\033[1;32m" << "\n***CONVERGENCE ACHIEVED***\n" << "\033[0m";
                break;
            case SQPresult::super_success:
                std::cout << "\033[1;32m" << "\n***STRONG CONVERGENCE ACHIEVED***\n" << "\033[0m";
                break;
            case SQPresult::local_infeasibility:
                std::cout << "\033[1;31m" << "\nLOCAL INFEASIBILITY\n" << "\033[0m";
                break;
            case SQPresult::restoration_failure:
                std::cout << "\033[1;31m" << "\nRESTORATION FAILURE\n" << "\033[0m";
                break;
            default:
                std::cout << "\033[1;31m" << "\nNLP SOLUTION FAILED\n" << "\033[0m";
        }
    }
    return rs;
}


}
