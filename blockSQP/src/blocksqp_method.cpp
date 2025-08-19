/*
 * blockSQP -- Sequential quadratic programming for problems with
 *             block-diagonal Hessian matrix.
 * Copyright (C) 2012-2015 by Dennis Janka <dennis.janka@iwr.uni-heidelberg.de>
 *
 * Licensed under the zlib license. See LICENSE for more details.
 */

/**
 * \file blocksqp_method.cpp
 * \author Reinhold Wittmann
 * \date 2025-
 *
 *  Constructors and helper methods of SQPmethod class
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
        rest_prob = std::make_unique<RestorationProblem>(prob, Matrix(), param->rest_rho, param->rest_zeta);
        //rest_stats = std::make_unique<SQPstats>(stats->outpath);

        rest_xi.Dimension(rest_prob->nVar);
        rest_lambda.Dimension(rest_prob->nVar + rest_prob->nCon);
        rest_lambdaQP.Dimension(rest_prob->nVar + rest_prob->nCon);
   }
}

SQPmethod::SQPmethod(): prob(nullptr), param(nullptr), stats(nullptr), vars(nullptr), sub_QP(nullptr), sub_QPs_par(nullptr), scaled_prob(nullptr),
    rest_prob(nullptr), rest_param(nullptr), rest_stats(nullptr), rest_method(nullptr), initCalled(false){}

SQPmethod::~SQPmethod(){}

/*

SCQPmethod::SCQPmethod(Problemspec *problem, SQPoptions *parameters, SQPstats *statistics, Condenser *CND){

    prob = problem;
    param = parameters; param->optionsConsistency();
    stats = statistics;
    cond = CND;

    if (param->automatic_scaling){
        scaled_prob = std::make_unique<scaled_Problemspec>(prob);
        prob = scaled_prob.get();
    }
    vars = std::make_unique<SCQPiterate>(prob, param, cond);

    // Check if there are options that are infeasible and set defaults accordingly
    if (param->sparse == 0) throw ParameterError("Condensing only works with sparse QPs");
    if (param->block_hess != 1) throw ParameterError("Condensing requires block diagonal hessian for efficient linear algebra");
    
    sub_QP = std::unique_ptr<QPsolverBase>(create_QPsolver(prob, vars.get(), param->qpsol_options));
    
    if (param->enable_rest){
        rest_cond = std::unique_ptr<Condenser>(create_restoration_Condenser(cond, 0));
        rest_param = std::unique_ptr<SQPoptions>(create_restoration_options(param));
        rest_prob = std::make_unique<TC_restoration_Problem>(prob, cond, Matrix(), param->rest_rho, param->rest_zeta);
        rest_stats = std::make_unique<SQPstats>(stats->outpath);

        rest_xi.Dimension(rest_prob->nVar);
        rest_lambda.Dimension(rest_prob->nVar + rest_prob->nCon);
        rest_lambdaQP.Dimension(rest_prob->nVar + rest_prob->nCon);
    }
}

SCQPmethod::SCQPmethod(){}


SCQPmethod::~SCQPmethod(){}


SCQP_bound_method::SCQP_bound_method(Problemspec *problem, SQPoptions *parameters, SQPstats *statistics, Condenser *CND){
    cond = CND;
    if (cond->add_dep_bounds != 1){
        std::cout << "SCQP_bound_method: Condenser needs to add inactive dependent variable bounds, changing condenser add_dep_bound option to 1\n";
        cond->set_dep_bound_handling(1);
    }

    prob = problem;
    param = parameters; param->optionsConsistency();
    stats = statistics;

    if (param->automatic_scaling){
        scaled_prob = std::make_unique<scaled_Problemspec>(prob);
        prob = scaled_prob.get();
    }
    vars = std::make_unique<SCQPiterate>(prob, param, cond);

    // Check if there are options that are infeasible and set defaults accordingly
    if (param->sparse == 0){
        throw std::invalid_argument("SCQPmethod: Error, condensing only works with sparse QPs");
    }
    if (param->block_hess != 1){
        throw std::invalid_argument("SCQPmethod: Error, condensing requires block diagonal hessian for efficient linear algebra");
    }

    //sub_QP = std::unique_ptr<QPsolver>(create_QPsolver(cond->condensed_num_vars, cond->condensed_num_cons, cond->condensed_num_hessblocks, cond->condensed_blockIdx, param));
    sub_QP = std::unique_ptr<QPsolverBase>(create_QPsolver(prob, vars.get(), param));

    
    if (param->enable_rest){
        rest_cond = std::unique_ptr<Condenser>(create_restoration_Condenser(cond, 0));
        rest_param = std::unique_ptr<SQPoptions>(create_restoration_options(param));
        rest_prob = std::make_unique<TC_restoration_Problem>(prob, cond, Matrix(), param->rest_rho, param->rest_zeta);
        rest_stats = std::make_unique<SQPstats>(stats->outpath);

        rest_xi.Dimension(rest_prob->nVar);
        rest_lambda.Dimension(rest_prob->nVar + rest_prob->nCon);
        rest_lambdaQP.Dimension(rest_prob->nVar + rest_prob->nCon);
    }
}


SCQP_correction_method::SCQP_correction_method(Problemspec *problem, SQPoptions *parameters, SQPstats *statistics, Condenser *CND){
    cond = CND;
    if (cond->add_dep_bounds > 0){
        std::cout << "Warning: Condenser adds dependent variable bounds to constraint matrix, performance may be impeded\n";
    }
    
    prob = problem;
    param = parameters; param->optionsConsistency();
    stats = statistics;
    
    if (param->automatic_scaling){
        scaled_prob = std::make_unique<scaled_Problemspec>(prob);
        prob = scaled_prob.get();
    }
    vars = std::make_unique<SCQP_correction_iterate>(prob, param, cond);
    
    // Check if there are options that are infeasible and set defaults accordingly
    if (param->sparse == 0){
        throw std::invalid_argument("SCQPmethod: Error, condensing only works with sparse QPs");
    }
    if (param->block_hess != 1){
        throw std::invalid_argument("SCQPmethod: Error, condensing requires block diagonal hessian for efficient linear algebra");
    }
    
    sub_QP = std::unique_ptr<QPsolverBase>(create_QPsolver(prob, vars.get(), param->qpsol_options));
    
    corrections = new Matrix[cond->num_targets];
    SOC_corrections = new Matrix[cond->num_targets];
    for (int tnum = 0; tnum < cond->num_targets; tnum++){
        corrections[tnum].Dimension(cond->targets_data[tnum].n_dep).Initialize(0.);
        SOC_corrections[tnum].Dimension(cond->targets_data[tnum].n_dep).Initialize(0.);
    }
    
    if (param->enable_rest){
        rest_cond = std::unique_ptr<Condenser>(create_restoration_Condenser(cond, 0));
        rest_param = std::unique_ptr<SQPoptions>(create_restoration_options(param));
        rest_prob = std::make_unique<TC_restoration_Problem>(prob, cond, Matrix(), param->rest_rho, param->rest_zeta);
        rest_stats = std::make_unique<SQPstats>(stats->outpath);
        
        rest_xi.Dimension(rest_prob->nVar);
        rest_lambda.Dimension(rest_prob->nVar + rest_prob->nCon);
        rest_lambdaQP.Dimension(rest_prob->nVar + rest_prob->nCon);
    }
}


SCQP_correction_method::~SCQP_correction_method(){
    delete[] corrections;
    delete[] SOC_corrections;
}

*/



}