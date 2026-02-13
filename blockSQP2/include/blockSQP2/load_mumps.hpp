/*
 * blockSQP2 -- A structure-exploiting nonlinear programming solver based
 *              on blockSQP by Dennis Janka.
 * Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
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

#ifndef BLOCKSQP2_LOAD_MUMPS_HPP
#define BLOCKSQP2_LOAD_MUMPS_HPP

#ifdef SOLVER_MUMPS
namespace blockSQP2{


void load_mumps_libs(int N_plugins);
void *get_plugin_handle(int ind);
void *get_fptr_dmumps_c(int ID);

} // namespace blockSQP2


#endif //SOLVER_MUMPS
#endif //BLOCKSQP2_LOAD_MUMPS_HPP