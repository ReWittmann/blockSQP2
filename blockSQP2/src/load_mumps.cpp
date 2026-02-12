/*
 * blockSQP 2 -- Condensing, convexification strategies, scaling heuristics and more
 *               for blockSQP, the nonlinear programming solver by Dennis Janka.
 * Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
 * 
 * Licensed under the zlib license. See LICENSE for more details.
 */

/**
 * \file blocksqp_mumps.cpp
 * \author Reinhold Wittmann
 * \date 2023-2025
 *
 * Implementation of functions for loading the sparse linear solver MUMPS
 */

namespace blockSQP2{

#ifdef SOLVER_MUMPS

#include <blockSQP2/load_mumps.hpp>
#include <blockSQP2/defs.hpp>
#include <iostream>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>

#ifdef LINUX
    #include <dlfcn.h>
    #include <libgen.h>
#elif defined(WINDOWS)
    #include "windows.h"
#endif


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


#ifdef LINUX
    //Flag indicating that libdmumps_c_dyn.so, build from libdmumps_c_dyn.cpp has been linked
    //In this case, rely on rpath to find and load the shared library
    
    /*
    #ifdef DMUMPS_C_DYN
    //Rely on dynamic linking and rpath to find the directory of libdmumps_c_dyn.so
    extern "C" void dmumps_c_dyn(void* mumps_struc_c_dyn);
    
    const char* get_mumps_module_dir(){
        Dl_info info;
        bool load_success = dladdr((void*) &dmumps_c_dyn, &info);
        if (!load_success) throw std::runtime_error(std::string("dladdr failed to obtain path to libdmumps_c_dyn.so from linked symbol dmumps_c_dyn"));
        char* path_ = new char[PATH_MAX];
        std::strncpy(path_, info.dli_fname, PATH_MAX);
        path_[PATH_MAX - 1] = '\0';
        const char* dir = dirname(path_);
        return dir;
    }
    #endif
    */
            
    //Search for libdmumps_c_syn.so in current directory
    const char* get_current_lib_dir(){
        Dl_info info;
        //Get address of any function (e.g. this function) in this shared object,
        bool load_success = dladdr((void*) &get_current_lib_dir, &info);
        if (!load_success) throw std::runtime_error(std::string("dladdr failed to obtain path to libblockSQP"));
        char* path_ = new char[PATH_MAX];
        std::strncpy(path_, info.dli_fname, PATH_MAX);
        path_[PATH_MAX - 1] = '\0';
        const char* dir = dirname(path_);
        return dir;
    }
    
    void load_mumps_libs(int N_plugins){
        void **handle;
        std::string dmumps_c_dyn_dir = std::string(get_current_lib_dir()) + "/libdmumps_c_dyn.so";
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
    
    std::string get_current_lib_dir(){
        HMODULE module = nullptr;
        bool load_success = GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, static_cast<LPCSTR>((void*) &get_current_lib_dir), &module);
        if (!load_success) throw std::runtime_error("GetModuleHandleEx failed");

        char path[MAX_PATH];
        DWORD length = GetModuleFileNameA(module, path, MAX_PATH);
        if (length == 0 || length == MAX_PATH) 
            throw std::runtime_error("GetModuleFileNameA failed");
        
        return std::filesystem::path(path).parent_path().string();
    }
    
    void load_mumps_libs(int N_plugins){
        void** handle;
        std::string dmumps_c_dyn_dir = get_current_lib_dir();
        for (int i = 0; i < N_plugins; i++) {
            handle = get_handle_ptr(i);
            if (*handle == nullptr){
                *handle = LoadLibrary((dmumps_c_dyn_dir + "\\dmumps_c_dyn_" + std::to_string(i) + ".dll").c_str());
                if (*handle == nullptr)
                    throw std::runtime_error(std::string("Failed to load library at \"") + dmumps_c_dyn_dir + "\\dmumps_c_dyn_" + std::to_string(i) + ".dll" + "\"");
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

#endif //SOLVER_MUMPS

} // namespace blockSQP2