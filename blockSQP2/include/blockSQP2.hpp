/*
 * blockSQP2 -- A structure-exploiting nonlinear programming solver based
 *              on blockSQP by Dennis Janka.
 * Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
 * 
 * Licensed under the zlib license. See LICENSE for more details.
 */

#include <blockSQP2/condensing.hpp>
#include <blockSQP2/defs.hpp>
#include <blockSQP2/general_purpose.hpp>
#include <blockSQP2/iterate.hpp>
#include <blockSQP2/matrix.hpp>
#include <blockSQP2/method.hpp>
#include <blockSQP2/options.hpp>
#include <blockSQP2/problemspec.hpp>
#include <blockSQP2/qpsolver.hpp>
#include <blockSQP2/restoration.hpp>
#include <blockSQP2/stats.hpp>