/*
 * blockSQP -- Sequential quadratic programming for problems with
 *             block-diagonal Hessian matrix.
 * Copyright (C) 2012-2015 by Dennis Janka <dennis.janka@iwr.uni-heidelberg.de>
 *
 * Licensed under the zlib license. See LICENSE for more details.
 */

/*
 * blockSQP extensions -- Extensions and modifications for the 
                          blockSQP nonlinear solver by Dennis Janka
 * Copyright (C) 2023-2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
 *
 * Licensed under the zlib license. See LICENSE for more details.
 */
 
/**
 * \file blocksqp_method.cpp
 * \author Reinhold Wittmann
 * \date 2025-
 *
 *  Constructors and helper methods of SQPmethod class,
 *  based blocksqp_main.cpp by Dennis Janka
 *
 */

 #include "blocksqp_iterate.hpp"
 #include "blocksqp_options.hpp"
 #include "blocksqp_stats.hpp"
 #include "blocksqp_method.hpp"
 #include "blocksqp_general_purpose.hpp"
 #include "blocksqp_restoration.hpp"
 #include "blocksqp_qpsolver.hpp"
 #include <fstream>
 #include <cmath>
 #include <chrono>
 #include <memory>


namespace blockSQP{


//Constructor helper methods

SQPoptions* create_restoration_options(SQPoptions *parent_options){
    SQPoptions *rest_param = new SQPoptions();

    //General restoration options
    rest_param->enable_rest = 0;
    rest_param->lim_mem = 1;
    rest_param->hess_approx = 2;
    rest_param->sizing = 2;
    rest_param->BFGS_damping_factor = 0.2;
    rest_param->sparse = parent_options->sparse;
    rest_param->max_filter_overrides = 0;
    //rest_param->sizing = 4;
    //rest_param->BFGS_damping_factor = 1./3.;

    rest_param->result_print_color = false;
    rest_param->par_QPs = false;
    //Do not print to any file
    rest_param->debug_level = 0;

    //Derived from parent method options
    rest_param->opt_tol = parent_options->opt_tol;
    rest_param->feas_tol = parent_options->feas_tol;
    rest_param->qpsol = parent_options->qpsol;
    rest_param->qpsol_options = parent_options->qpsol_options;

    return rest_param;
}



SQPmethod::SQPmethod(Problemspec *problem, SQPoptions *parameters, SQPstats *statistics): 
        prob(problem), param(parameters), stats(statistics){
    // Check if there are options that are infeasible and set defaults accordingly
    param->optionsConsistency(problem);
    
    //Create scalable problem wrapper if automatic scaling is enabled.
    if (param->automatic_scaling){
        scaled_prob = std::make_unique<scaled_Problemspec>(prob);
        prob = scaled_prob.get();
    }    
    vars = std::make_unique<SQPiterate>(prob, param);

    // Create a solver object for quadratic subproblems.
    sub_QP = std::unique_ptr<QPsolverBase>(create_QPsolver(prob, vars.get(), param->qpsol_options));
    
    // If parallel solution of QPs is enabled, use dedicated QPsolver instances instead
    if (param->par_QPs){
        sub_QPs_par = create_QPsolvers_par(prob, vars.get(), param);
        QP_threads = std::make_unique<std::jthread[]>(param->max_conv_QPs); //One QP can run on main thread
    }
    
    
    //Setup the feasibility restoration problem
    if (param->enable_rest){
        rest_param = std::unique_ptr<SQPoptions>(create_restoration_options(param));
        
        rest_xi.Dimension(prob->nVar + prob->nCon);
        rest_lambda.Dimension(prob->nVar + prob->nCon + prob->nCon);
        rest_lambdaQP.Dimension(prob->nVar + prob->nCon + prob->nCon);
   }
}

SQPmethod::SQPmethod(): prob(nullptr), param(nullptr), stats(nullptr), vars(nullptr), sub_QP(nullptr), sub_QPs_par(nullptr), scaled_prob(nullptr),
    rest_prob(nullptr), rest_param(nullptr), rest_stats(nullptr), rest_method(nullptr), initCalled(false){}

SQPmethod::~SQPmethod(){}



//Experimental

bound_correction_method::bound_correction_method(Problemspec *problem, SQPoptions *parameters, SQPstats *statistics):
    SQPmethod(problem, parameters, statistics){
        if (prob->cond == nullptr) throw std::invalid_argument("bound_correction_method invoked for problem with no condenser!");
        if (prob->cond->add_dep_bounds > 0) throw std::invalid_argument("bound_correction_method: Condenser adds dependent variable bounds!");
        //Note: The dimensions of rest_xi, rest_lambda and rest_lambdaQP are set larger than necessary by base class.
        //      Should not cause problems, so leave them for now
    }



}