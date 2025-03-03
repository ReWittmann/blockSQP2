#include "blocksqp_defs.hpp"
#include <iostream>


namespace blockSQP{

RES print_RES(RES rs){
    if (rs == RES::FEAS_SUCCESS) std::cout << "\033[1;32m" << "\n***CONVERGENCE PARTIALLY ACHIEVED***\n" << "\033[0m";
    else if (rs == RES::success) std::cout << "\033[1;32m" << "\n***CONVERGENCE ACHIEVED***\n" << "\033[0m";
    else if (rs == RES::SUPER_SUCCESS) std::cout << "\033[1;32m" << "\n***STRONG CONVERGENCE ACHIEVED***\n" << "\033[0m";
    else if (int(rs) < 0) std::cout << "\033[1;31m" << "\nNLP SOLUTION FAILED\n" << "\033[0m";
    return rs;
}


}
