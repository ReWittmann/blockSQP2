#include "blocksqp_defs.hpp"
#include <iostream>


namespace blockSQP{
std::vector<std::chrono::microseconds> BFGS_times;

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
            default:
                std::cout << colPrefix + "\nNLP SOLUTION UNSUCCESSFUL" + colSuffix + "\n";
        }
    }
    std::chrono::microseconds dur(0);
    for(size_t i = 0; i < BFGS_times.size(); i++){
        dur += BFGS_times[i];
    }
    dur /= BFGS_times.size();
    BFGS_times.resize(0);
    
    std::cout << "Avg BFGS time: " << dur << "\n";
    return rs;
}


}
