#ifndef BLOCKSQP_PLUGIN_LOADER_HPP
#define BLOCKSQP_PLUGIN_LOADER_HPP

#include <thread>
#include <condition_variable>
#include <future>
#include <condition_variable>
#include <mutex>

namespace blockSQP{


//Future: add enum class Plugins{...} if more possible plugins are added. Currently, MUMPS is the only plugin handled this way.

//Load several instances of a library, which are threadsafe between each other. Highly platform-specific. 
class plugin_loader{
    private:
    std::jthread TLS_hold;
    std::mutex loader_mutex;
    std::condition_variable loader_cv;
    
    public:
    int N_handles;
    std::unique_ptr<void *[]> handles;
    
    plugin_loader(int N_handles);
    void inner_plugin_loader(std::promise<bool>plugins_loaded, int N_handles, void** handles, std::string &load_error_msg);
    ~plugin_loader();
    plugin_loader(plugin_loader&) = delete;
    plugin_loader(plugin_loader&&) = default;
    
    void *load_symbol(const char* name, int i_handle);
};

}
#endif