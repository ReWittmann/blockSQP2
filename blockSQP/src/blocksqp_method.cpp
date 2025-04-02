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
    rest_param->restoreFeas = 0;
    rest_param->hessLimMem = 1;
    rest_param->hessUpdate = 2;
    rest_param->hessScaling = 2;
    rest_param->hessDampFac = 0.2;
    //rest_param->hessScaling = 4;
    //rest_param->hessDampFac = 1./3.;

    rest_param->loud_SQPresult = false;
    //Do not print to any file
    rest_param->debugLevel = 0;

    //Derived from parent method options
    rest_param->opttol = parent_options->opttol;
    rest_param->nlinfeastol = parent_options->nlinfeastol;
    rest_param->QPsol = parent_options->QPsol;
    rest_param->QPsol_opts = parent_options->QPsol_opts;

    return rest_param;
}   



SQPmethod::SQPmethod( Problemspec *problem, SQPoptions *parameters, SQPstats *statistics ): prob(problem), param(parameters), stats(statistics){
    // Check if there are options that are infeasible and set defaults accordingly
    param->optionsConsistency(problem);
    
    if (param->autoScaling){
        //scaled_prob = new scaled_Problemspec(problem);
        scaled_prob = std::make_unique<scaled_Problemspec>(problem);
        prob = scaled_prob.get();
    }
    else{
        scaled_prob = nullptr;
        prob = problem;
    }
    vars = std::make_unique<SQPiterate>(prob, param, true);

    // Create a solver object for quadratic subproblems.
    sub_QP = std::unique_ptr<QPsolver>(create_QPsolver(prob->nVar, prob->nCon, vars->nBlocks, vars->blockIdx.get(), param));
    
    initCalled = false;
    
    //Setup the feasibility restoration problem
    if (param->restoreFeas){
        rest_param = std::unique_ptr<SQPoptions>(create_restoration_options(param));
        rest_prob = std::make_unique<RestorationProblem>(prob, Matrix(), param->restRho, param->restZeta);
        rest_stats = std::make_unique<SQPstats>(stats->outpath);

        rest_xi.Dimension(rest_prob->nVar);
        rest_lambda.Dimension(rest_prob->nVar + rest_prob->nCon);
        rest_lambdaQP.Dimension(rest_prob->nVar + rest_prob->nCon);
   }
}

SQPmethod::SQPmethod(): prob(nullptr), param(nullptr), stats(nullptr), vars(nullptr), sub_QP(nullptr),
    rest_prob(nullptr), rest_param(nullptr), rest_stats(nullptr), rest_method(nullptr), scaled_prob(nullptr), initCalled(false){}

SQPmethod::~SQPmethod(){}


SCQPmethod::SCQPmethod(Problemspec *problem, SQPoptions *parameters, SQPstats *statistics, Condenser *CND){

    prob = problem;
    param = parameters; param->optionsConsistency();
    stats = statistics;
    cond = CND;

    if (param->autoScaling){
        //scaled_prob = new scaled_Problemspec(problem);
        scaled_prob = std::make_unique<scaled_Problemspec>(problem);
        prob = scaled_prob.get();
    }
    else{
        scaled_prob = nullptr;
        prob = problem;
    }
    vars = std::make_unique<SCQPiterate>(prob, param, cond, true);

    // Check if there are options that are infeasible and set defaults accordingly
    if (param->sparseQP == 0) throw ParameterError("Condensing only works with sparse QPs");
    if (param->blockHess != 1) throw ParameterError("Condensing requires block diagonal hessian for efficient linear algebra");
    
    sub_QP = std::unique_ptr<QPsolver>(create_QPsolver(cond->condensed_num_vars, cond->condensed_num_cons, cond->condensed_num_hessblocks, cond->condensed_blockIdx, param));
    initCalled = false;

    if (param->restoreFeas){
        rest_cond = std::unique_ptr<Condenser>(create_restoration_Condenser(cond, 0));
        rest_param = std::unique_ptr<SQPoptions>(create_restoration_options(param));
        rest_prob = std::make_unique<TC_restoration_Problem>(prob, cond, Matrix(), param->restRho, param->restZeta);
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

    if (param->autoScaling){
        //scaled_prob = new scaled_Problemspec(problem);
        scaled_prob = std::make_unique<scaled_Problemspec>(problem);
        prob = scaled_prob.get();
    }
    else{
        scaled_prob = nullptr;
        prob = problem;
    }
    vars = std::make_unique<SCQPiterate>(prob, param, cond, true);

    // Check if there are options that are infeasible and set defaults accordingly
    if (param->sparseQP == 0){
        throw std::invalid_argument("SCQPmethod: Error, condensing only works with sparse QPs");
    }
    if (param->blockHess != 1){
        throw std::invalid_argument("SCQPmethod: Error, condensing requires block diagonal hessian for efficient linear algebra");
    }

    sub_QP = std::unique_ptr<QPsolver>(create_QPsolver(cond->condensed_num_vars, cond->condensed_num_cons, cond->condensed_num_hessblocks, cond->condensed_blockIdx, param));

    initCalled = false;

    if (param->restoreFeas){
        rest_cond = std::unique_ptr<Condenser>(create_restoration_Condenser(cond, 0));
        rest_param = std::unique_ptr<SQPoptions>(create_restoration_options(param));
        rest_prob = std::make_unique<TC_restoration_Problem>(prob, cond, Matrix(), param->restRho, param->restZeta);
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

    if (param->autoScaling){
        //scaled_prob = new scaled_Problemspec(problem);
        scaled_prob = std::make_unique<scaled_Problemspec>(problem);
        prob = scaled_prob.get();
    }
    else{
        scaled_prob = nullptr;
        prob = problem;
    }
    vars = std::make_unique<SCQP_correction_iterate>(prob, param, cond, true);

    // Check if there are options that are infeasible and set defaults accordingly
    if (param->sparseQP == 0){
        throw std::invalid_argument("SCQPmethod: Error, condensing only works with sparse QPs");
    }
    if (param->blockHess != 1){
        throw std::invalid_argument("SCQPmethod: Error, condensing requires block diagonal hessian for efficient linear algebra");
    }

    sub_QP = std::unique_ptr<QPsolver>(create_QPsolver(cond->condensed_num_vars, cond->condensed_num_cons, cond->condensed_num_hessblocks, cond->condensed_blockIdx, param));

    initCalled = false;

    corrections = new Matrix[cond->num_targets];
    SOC_corrections = new Matrix[cond->num_targets];
    for (int tnum = 0; tnum < cond->num_targets; tnum++){
        corrections[tnum].Dimension(cond->targets_data[tnum].n_dep).Initialize(0.);
        SOC_corrections[tnum].Dimension(cond->targets_data[tnum].n_dep).Initialize(0.);
    }

    if (param->restoreFeas){
        rest_cond = std::unique_ptr<Condenser>(create_restoration_Condenser(cond, 0));
        rest_param = std::unique_ptr<SQPoptions>(create_restoration_options(param));
        rest_prob = std::make_unique<TC_restoration_Problem>(prob, cond, Matrix(), param->restRho, param->restZeta);
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









}