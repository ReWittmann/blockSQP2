#include "blocksqp_load_mumps.hpp"
#include "blocksqp_defs.hpp"
#include <thread>
#include <condition_variable>
#include <future>
#include <iostream>
#include <cstring>

#ifdef LINUX
    #include <dlfcn.h>
    #include <libgen.h>
#elif defined(WINDOWS)
    #include "windows.h"
#endif


namespace blockSQP{

#define INNER_TO_STRING(x) #x
#define TO_STRING(x) INNER_TO_STRING(x)

/*
#ifdef N_LINSOL_PATHS
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

const char* get_plugin_path(int linsol_ID){
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
            default: ;
    }
    throw std::runtime_error("No linsol path available for ID " + std::to_string(linsol_ID) + ". Please recompile with -DN_LINSOL_PATHS=[N > " + std::to_string(linsol_ID) + "]-DLINSOL_PATH_0=* ... -DLINSOL_PATH_${N_LINSOL_PATHS}=*");
}
#else
    const char* get_plugin_path(int linsol_ID){return nullptr;}
#endif


#ifdef LINSOL_PATH
    const char *linsol_path_glob = TO_STRING(LINSOL_PATH);
    const char* get_plugin_path(){return linsol_path_glob;}
#else
     const char* get_plugin_path(){throw std::runtime_error("No linsol path provided");};
#endif
*/

void *glob_handle_0 = nullptr;
void *glob_handle_1 = nullptr;
void *glob_handle_2 = nullptr;
void *glob_handle_3 = nullptr;
void *glob_handle_4 = nullptr;
void *glob_handle_5 = nullptr;
void *glob_handle_6 = nullptr;
void *glob_handle_7 = nullptr;


void **get_handle_ptr(int ind){
    switch (ind){
        case 0:
            return &glob_handle_0;
        case 1:
            return &glob_handle_1;
        case 2:
            return &glob_handle_2;
        case 3:
            return &glob_handle_3;
        case 4:
            return &glob_handle_4;
        case 5:
            return &glob_handle_5;
        case 6:
            return &glob_handle_6;
        case 7:
            return &glob_handle_7;
        default:
            throw std::invalid_argument("No global handle available for index " + std::to_string(ind));
    }
}

void *get_plugin_handle(int ind){
    switch (ind){
        case 0:
            return glob_handle_0;
        case 1:
            return glob_handle_1;
        case 2:
            return glob_handle_2;
        case 3:
            return glob_handle_3;
        case 4:
            return glob_handle_4;
        case 5:
            return glob_handle_5;
        case 6:
            return glob_handle_6;
        case 7:
            return glob_handle_7;
        default:
            throw std::invalid_argument("No global handle available for index " + std::to_string(ind));
    }
}

/*
#ifdef SOLVER_MUMPS
    extern "C" void dmumps_c_dyn(void *mumps_struc_c_dyn);
#endif
*/


#ifdef LINUX
    extern "C" void dmumps_c_dyn(void* mumps_struc_c_dyn);
    /*
    void load_mumps_libs(int N_plugins){        
        void **handle;
        for (int i = 0; i < N_plugins; i++){
            handle = get_handle_ptr(i);
            if (*handle == nullptr){
                *handle = dlmopen(LM_ID_NEWLM, get_plugin_path(), RTLD_LAZY | RTLD_LOCAL);
                if (*handle == nullptr)
                    throw std::runtime_error(std::string("Failed to load library at \"") + get_plugin_path() + "\", dlerror(): " + dlerror());
            }
        }
    }
    */
    
    const char* get_mumps_module_dir(){
        Dl_info info;
        bool load_success = dladdr((void*) &dmumps_c_dyn, &info);
        if (!load_success) throw std::runtime_error(std::string("Could not get path to linsol plugins"));//, dlerror(): ") + dlerror());
        char* path_ = new char[PATH_MAX];
        std::strncpy(path_, info.dli_fname, PATH_MAX);
        path_[PATH_MAX - 1] = '\0';
        const char* dir = dirname(path_);
        return dir;
    }
    
    void load_mumps_libs(int N_plugins){
        void **handle;
        std::string dmumps_c_dyn_dir = std::string(get_mumps_module_dir()) + "/libdmumps_c_dyn.so";
        for (int i = 0; i < N_plugins; i++){
            handle = get_handle_ptr(i);
            if (*handle == nullptr){
                *handle = dlmopen(LM_ID_NEWLM, dmumps_c_dyn_dir.c_str(), RTLD_LAZY | RTLD_LOCAL);
                if (*handle == nullptr)
                    throw std::runtime_error(std::string("Failed to load library at \"") + dmumps_c_dyn_dir + "\", dlerror(): " + dlerror());
            }
        }
    }
    
    
    void *get_fptr_dmumps_c(int ID){
        if (get_plugin_handle(ID) == nullptr) load_mumps_libs(ID + 1);
        void *linsol_handle = get_plugin_handle(ID);
        void *fptr_dmumps_c = dlsym(linsol_handle, "dmumps_c_dyn");
        if (fptr_dmumps_c == nullptr) throw std::runtime_error(std::string("Error, could not load symbol dmumps_c from handle Nr. ") + std::to_string(ID) + std::string(", dlerror(): ") + std::string(dlerror()));
        return fptr_dmumps_c;
    }    
#elif defined(WINDOWS)
/*
    const char* get_mumps_module_dir(){
        HMODULE module = nullptr;
        bool load_success = GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, static_cast<LPCSTR>((void*) &dmumps_c_dyn), &module);
        if (!load_success) throw std::runtime_error("GetModuleHandleEx failed");

        char *path = new char[MAX_PATH];
        DWORD length = GetModuleFileNameA(module, path, MAX_PATH);
        if (length == 0 || length == MAX_PATH) 
            throw std::runtime_error("GetModuleFileNameA failed");
        
        return path;
    }
    */
    void load_mumps_libs(int N_plugins) {
        void** handle;
        std::string dmumps_c_dyn_dir = std::string(get_mumps_module_dir());
        for (int i = 0; i < N_plugins; i++) {
            handle = get_handle_ptr(i);
            if (*handle == nullptr){
                *handle = LoadLibrary((dmumps_c_dyn_dir + "\\dmumps_c_dyn_" + std::to_string(i) + ".dll").c_str());
                if (*handle == nullptr)
                    throw std::runtime_error(std::string("Failed to load library at \"") + dmumps_c_dyn_dir + "\"");
                std::cout << "Load successful\n";
            }
        }
    }
    
    void *get_fptr_dmumps_c(int ID){
        if (get_plugin_handle(ID) == nullptr) load_mumps_libs(ID + 1);
        void *linsol_handle = get_plugin_handle(ID);
        void *fptr_dmumps_c = (void *) GetProcAddress((HMODULE) linsol_handle, "dmumps_c_dyn");
        if (fptr_dmumps_c == nullptr) throw std::runtime_error("Could not load symbol my_dmumps_c from handle Nr. " + std::to_string(ID));
        return fptr_dmumps_c;
    }
    
#else
    void load_mumps_libs(int N_plugins){
        throw NotImplementedError("load_mumps_libs should never be called on this platform");
    }
#endif








/*

plugin_loader::plugin_loader(int arg_N_handles): 
            N_handles(arg_N_handles), loader_threads(new std::jthread[arg_N_handles]), loader_cv(new std::condition_variable[arg_N_handles]),
            unloading(false), symbol_name(nullptr), loaded_symbol(nullptr){
    bool plugin_loaded = false;
    std::string error_msg;
    
    int i = 0;
    for (; i < N_handles; i++){
        std::promise<bool> plugin_loaded_p;
        std::future<bool> plugin_loaded_f = plugin_loaded_p.get_future();
        loader_threads[i] = std::jthread(&plugin_loader::inner_plugin_loader, this, std::move(plugin_loaded_p), i, std::ref(error_msg));
        plugin_loaded = plugin_loaded_f.get();
        if (!plugin_loaded){
            loader_threads[i].join();
            break;
        }
    }
    if (!plugin_loaded){
        {
            std::lock_guard<std::mutex> loader_lock(loader_mutex); 
            unloading = true;
        }
        for (i -= 1; i >= 0; i--){
            loader_cv[i].notify_all();
            loader_threads[i].join();
        }
        throw std::runtime_error(error_msg);
    }
}



void plugin_loader::inner_plugin_loader(std::promise<bool> plugin_loaded_p, int lib_num, std::string &load_error_msg){
    std::cout << "Loading library at " << PATHS::get_linsol_path(lib_num) << "\n";
    //void *handle = dlmopen(LM_ID_NEWLM, PATHS::get_linsol_path(lib_num), RTLD_LAZY | RTLD_LOCAL);
    void *handle = dlopen(PATHS::get_linsol_path(lib_num), RTLD_LAZY | RTLD_LOCAL);
    
    //handles[i] = dlmopen(LM_ID_NEWLM, PATHS::get_linsol_path(i), RTLD_LAZY | RTLD_LOCAL);
    if (handle == nullptr){
        load_error_msg = std::string("Failed to load library at \"") + PATHS::get_linsol_path(lib_num) + "\", dlerror(): " + dlerror();
        plugin_loaded_p.set_value(false);
        return;
    }
    
    std::unique_lock<std::mutex> loader_lock(loader_mutex);
    plugin_loaded_p.set_value(true);
    
    std::cout << "Library loaded, going to sleep\n";
    loader_cv[lib_num].wait(loader_lock);
    while (!unloading){
        std::cout << "Woke up, loading symbol \"" << symbol_name << "\" from library " << lib_num << " at " << handle << "\n";
        loaded_symbol = dlsym(handle, symbol_name);
        if (loaded_symbol == nullptr){ 
            symbol_error_msg = std::string("Could not load symbol \"") + std::string(symbol_name) + "\", dlerror(): " + std::string(dlerror());
            symbol_loaded_p.set_value(false);
        }
        else symbol_loaded_p.set_value(true);
        std::cout << "Done, going to sleep\n";
        loader_cv[lib_num].wait(loader_lock);
    }
    
    std::cout << "Unloading plugin\n";
    dlclose(handle);
    return;
}

plugin_loader::~plugin_loader(){
    {
        std::lock_guard<std::mutex> linsol_lock(loader_mutex);
        unloading = true;
    }
    
    for (int i = 0; i < N_handles; i++){
        loader_cv[i].notify_all();
        loader_threads[i].join();
    }
}



void *plugin_loader::load_symbol(const char *arg_symbol_name, int arg_num_handle){
    symbol_loaded_p = std::promise<bool>();
    std::cout << "Symbol name is " << arg_symbol_name << "\n";
    std::future<bool> symbol_loaded_f = symbol_loaded_p.get_future();
    {
        std::lock_guard<std::mutex> loader_lock(loader_mutex);
        symbol_name = arg_symbol_name;
        loader_cv[arg_num_handle].notify_all();
    }
    if (!symbol_loaded_f.get()) throw std::runtime_error(symbol_error_msg);
    return loaded_symbol;
}

*/

}//namespace blockSQP