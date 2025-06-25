#ifndef BLOCKSQP_PLUGIN_LOADER_HPP
#define BLOCKSQP_PLUGIN_LOADER_HPP

#include <thread>
#include <condition_variable>
#include <future>
#include <condition_variable>
#include <mutex>

namespace blockSQP{


//Future: add enum class Plugins{...} if more possible plugins are added. Currently, MUMPS is the only plugin handled this way.
/*
//Load several instances of a library, which are threadsafe between each other. Highly platform-specific. 
class plugin_loader{
    public:
    int N_handles;
    //std::unique_ptr<void *[]> handles;
    
    private:
    std::unique_ptr<std::jthread[]> loader_threads;
    
    //Loader thread communication
    std::mutex loader_mutex;
    std::unique_ptr<std::condition_variable[]> loader_cv;  //To resume thread execution
    
    //Loader thread inputs
    bool unloading;                     //Indicates whether thread should close all handles or load a symbol
    const char* symbol_name;            //Name of the symbol that should be loaded
    //int num_handle;                     //ID of the handle that symbol should be loaded from
    
    //Loader thread outputs
    void *loaded_symbol;                //The loaded symbol, equal to nullptr upon failure
    std::promise<bool> symbol_loaded_p; //To indicate completion of the loading process
    std::string symbol_error_msg;       //Error message in case symbol loading failed 
    
    public:
    plugin_loader(int N_handles);
    void inner_plugin_loader(std::promise<bool>plugins_loaded, int lib_num, std::string &load_error_msg);
    ~plugin_loader();
    plugin_loader(plugin_loader&) = delete;
    plugin_loader(plugin_loader&&) = default;
    
    void *load_symbol(const char* name, int i_handle);
};
*/

void load_plugins(int N_plugins);
void *get_plugin_handle(int ind);

}
#endif