#include "blocksqp_plugin_loader.hpp"
#include <iostream>
#include <chrono>


int main(){
    blockSQP::plugin_loader L(8);
    void *fptr_dmumps_c = L.load_symbol("dmumps_c", 0);
    void *fptr_dmumps_c_2 = L.load_symbol("dmumps_c", 1);
    void *fptr_dmumps_c_3;
    try{
        fptr_dmumps_c_3 = L.load_symbol("wrong_symbol", 2);
    }
    catch(std::runtime_error &err){
        std::cout << err.what() << "\n";
    } 
    
    
    std::cout << "fptr_dmumps_c_ = " << fptr_dmumps_c << ", fptr_dmumps_c_2 = " << fptr_dmumps_c << "\n";
    
    std::unique_ptr<blockSQP::plugin_loader> L2 = std::make_unique<blockSQP::plugin_loader>(8);
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    std::cout << "calling destructor\n" << std::flush;
    L2 = nullptr;
    
    
    //blockSQP::plugin_loader L3(8);
    //blockSQP::plugin_loader L4(8);
    
    return 0;
}