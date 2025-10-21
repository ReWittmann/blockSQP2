/*
 * blockSQP 2 -- Extensions for the blockSQP nonlinear
                          programming solver by Dennis Janka
 * Copyright (C) 2023-2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
 *
 * Licensed under the zlib license. See LICENSE for more details.
 */

/**
 * \file blocksqp_mumps.hpp
 * \author Reinhold Wittmann
 * \date 2023-2025
 *
 * Declaration functions for loading the sparse linear solver MUMPS
 */

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