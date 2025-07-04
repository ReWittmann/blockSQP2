#ifndef BLOCKSQP_PLUGIN_LOADER_HPP
#define BLOCKSQP_PLUGIN_LOADER_HPP

#ifdef SOLVER_MUMPS
namespace blockSQP{


void load_mumps_libs(int N_plugins);
void *get_plugin_handle(int ind);
void *get_fptr_dmumps_c(int ID);

}


#endif //SOLVER_MUMPS
#endif //BLOCKSQP_PLUGIN_LOADER_HPP