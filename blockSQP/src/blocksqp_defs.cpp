#include "blocksqp_defs.hpp"
#include <iostream>


namespace blockSQP{

SQPresult loud_SQPresult(SQPresult rs, bool rloud){
    if (rloud){
        if (rs == SQPresult::partial_success) std::cout << "\033[1;32m" << "\n***CONVERGENCE PARTIALLY ACHIEVED***\n" << "\033[0m";
        else if (rs == SQPresult::success) std::cout << "\033[1;32m" << "\n***CONVERGENCE ACHIEVED***\n" << "\033[0m";
        else if (rs == SQPresult::super_success) std::cout << "\033[1;32m" << "\n***STRONG CONVERGENCE ACHIEVED***\n" << "\033[0m";
        else if (int(rs) < 0) std::cout << "\033[1;31m" << "\nNLP SOLUTION FAILED\n" << "\033[0m";
    }
    return rs;
}


}
