#include "blocksqp_plugin_loader.hpp"
#include <thread>
#include <condition_variable>
#include <future>
#include <iostream>

#ifdef LINUX
    #include <dlfcn.h>
#endif


namespace blockSQP{

namespace PATHS{
#define INNER_TO_STRING(x) #x
#define TO_STRING(x) INNER_TO_STRING(x)

#ifdef N_LINSOL_BIN
    #ifdef LINSOL_PATH_0
        const char *linsol_path_glob_0 = TO_STRING(LINSOL_PATH_0);
    #endif
    #ifdef LINSOL_PATH_1
        const char *linsol_path_glob_1 = TO_STRING(LINSOL_PATH_1);
    #endif
    #ifdef LINSOL_PATH_2
        const char *linsol_path_glob_2 = TO_STRING(LINSOL_PATH_2);
    #endif
    #ifdef LINSOL_PATH_3
        const char *linsol_path_glob_3 = TO_STRING(LINSOL_PATH_3);
    #endif
    #ifdef LINSOL_PATH_4
        const char *linsol_path_glob_4 = TO_STRING(LINSOL_PATH_4);
    #endif
    #ifdef LINSOL_PATH_5
        const char *linsol_path_glob_5 = TO_STRING(LINSOL_PATH_5);
    #endif
    #ifdef LINSOL_PATH_6
        const char *linsol_path_glob_6 = TO_STRING(LINSOL_PATH_6);
    #endif
    #ifdef LINSOL_PATH_7
        const char *linsol_path_glob_7 = TO_STRING(LINSOL_PATH_7);
    #endif

const char* get_linsol_path(int linsol_ID){
    switch (linsol_ID){
        #ifdef LINSOL_PATH_0
            case 0: return linsol_path_glob_0;
        #endif
        #ifdef LINSOL_PATH_1
            case 1: return linsol_path_glob_1;
        #endif
        #ifdef LINSOL_PATH_2
            case 2: return linsol_path_glob_2;
        #endif
        #ifdef LINSOL_PATH_3
            case 3: return linsol_path_glob_3;
        #endif
        #ifdef LINSOL_PATH_4
            case 4: return linsol_path_glob_4;
        #endif
        #ifdef LINSOL_PATH_5    
            case 5: return linsol_path_glob_5;
        #endif
        #ifdef LINSOL_PATH_6    
            case 6: return linsol_path_glob_6;
        #endif
        #ifdef LINSOL_PATH_7
            case 7: return linsol_path_glob_7;
        #endif
        //  default: return nullptr;
    }
    throw std::runtime_error("No linsol path available for ID " + std::to_string(linsol_ID) + ". Please recompile with -DLINSOL_PATH_0=* ... -DLINSOL_PATH_${N_LINSOL_BIN}=*");
}

#else
    const char* get_linsol_path(int linsol_ID){return nullptr;}
#endif
}


plugin_loader::plugin_loader(int arg_N_handles): N_handles(arg_N_handles), handles(new void*[arg_N_handles]){
    bool plugins_loaded = false;
    std::string error_msg;
        
    std::promise<bool> plugins_loaded_p;
    std::future<bool> plugins_loaded_f = plugins_loaded_p.get_future();
    TLS_hold = std::jthread(&plugin_loader::inner_plugin_loader, this, std::move(plugins_loaded_p), N_handles, handles.get(), std::ref(error_msg));
    plugins_loaded = plugins_loaded_f.get();
    if (!plugins_loaded){
        TLS_hold.join();
        throw std::runtime_error(error_msg.empty() ? "Error when trying to load dynamic libraries, no error message available" : error_msg);
    }
}



void plugin_loader::inner_plugin_loader(std::promise<bool> plugins_loaded, int N_handles, void **handles, std::string &load_error_msg){
    bool plugin_load_error = false;
    int i = 0;
    for (; i < N_handles; i++){
        std::cout << "Loading library at " << PATHS::get_linsol_path(i) << "\n";
        handles[i] = dlopen(PATHS::get_linsol_path(i), RTLD_LAZY | RTLD_LOCAL);
        if (handles[i] == nullptr){
            plugin_load_error = true;
            load_error_msg = std::string("Failed to load library at \"") + PATHS::get_linsol_path(i) + "\", dlerror(): " + dlerror();
            break;
        }
        std::cout << "Successfully loaded library\n";
    }
    
    if (plugin_load_error){
        for (i -= 1; i >= 0; i--){
            dlclose(handles[i]);
        }
        plugins_loaded.set_value(false);
        return;
    }
    else plugins_loaded.set_value(true);
    
    std::unique_lock<std::mutex> loader_lock(loader_mutex);
    std::cout << "All plugins loaded, going to sleep\n";
    loader_cv.wait(loader_lock);
    std::cout << "Woke up, unloading plugins\n";
    for (i = N_handles - 1; i >= 0; i--){
        dlclose(handles[i]);
    }
}

plugin_loader::~plugin_loader(){
    {
        std::lock_guard<std::mutex> linsol_lock(loader_mutex);
        loader_cv.notify_all();
    }
    TLS_hold.join();
}

void *plugin_loader::load_symbol(const char *symbol_name, int i_handle){
    if (i_handle >= N_handles) throw std::runtime_error("Symbol requested for handle " + std::to_string(i_handle) + ", but only " + std::to_string(N_handles) + " were loaded");
    void *sym = dlsym(handles[i_handle], symbol_name);
    if (sym == nullptr) throw std::runtime_error("Could not load symbol \"" + std::string(symbol_name) + "\", dlerror(): " + std::string(dlerror()));
    return sym;
}




}