/*
 * blockSQP 2 -- Condensing, convexification strategies, scaling heuristics and more
 *               for blockSQP, the nonlinear programming solver by Dennis Janka.
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

#ifndef BLOCKSQP_LOAD_MUMPS_HPP
#define BLOCKSQP_LOAD_MUMPS_HPP

#ifdef SOLVER_MUMPS
namespace blockSQP{


void load_mumps_libs(int N_plugins);
void *get_plugin_handle(int ind);
void *get_fptr_dmumps_c(int ID);

}


#endif //SOLVER_MUMPS
#endif //BLOCKSQP_LOAD_MUMPS_HPP