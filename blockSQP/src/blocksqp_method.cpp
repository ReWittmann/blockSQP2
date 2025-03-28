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


std::unique_ptr<SQPoptions> default_restorationOptions(SQPoptions *param){
    std::unique_ptr<SQPoptions> rest_opts = std::make_unique<SQPoptions>();

    rest_opts->restoreFeas = 0;
    rest_opts->hessUpdate = 2;
    rest_opts->hessLimMem = 1;
    rest_opts->hessScaling = 2;
    rest_opts->opttol = param->opttol;
    rest_opts->nlinfeastol = param->nlinfeastol;
    rest_opts->QPsol = param->QPsol;
    rest_opts->QPsol_opts = param->QPsol_opts;
    rest_opts->hessDampFac = 0.2;
    rest_opts->loud_SQPresult = false;

    return rest_opts;
}   




SQPmethod::SQPmethod( Problemspec *problem, SQPoptions *parameters, SQPstats *statistics ): prob(problem), param(parameters), stats(statistics){
    // Check if there are options that are infeasible and set defaults accordingly
    param->optionsConsistency(problem);
    
    if (param->autoScaling){
        scaled_prob = new scaled_Problemspec(problem);
        prob = scaled_prob;
    }
    else{
        scaled_prob = nullptr;
        prob = problem;
    }
    vars = new SQPiterate(prob, param, true);

    // Create a solver object for quadratic subproblems.
    sub_QP = create_QPsolver(prob->nVar, prob->nCon, vars->nBlocks, param);
    
    initCalled = false;
    
    //Setup the feasibility restoration problem
    if (param->restoreFeas){
        rest_opts = new SQPoptions();
        // Set options for the SQP method for this problem
        rest_opts->globalization = 1;
        rest_opts->whichSecondDerv = 0;
        rest_opts->restoreFeas = 0;
        rest_opts->hessUpdate = 2;
        rest_opts->hessLimMem = 1;
        rest_opts->hessScaling = 2;
        rest_opts->opttol = param->opttol;
        rest_opts->nlinfeastol = param->nlinfeastol;
        rest_opts->QPsol = param->QPsol;
        rest_opts->QPsol_opts = param->QPsol_opts;
        rest_opts->hessDampFac = 0.2;
        
        rest_opts->loud_SQPresult = false;

        //rest_opts->autoScaling = param->autoScaling;
        
        rest_prob = nullptr;
        rest_stats = nullptr;
        rest_method = nullptr;
    }
    else{
        rest_prob = nullptr;
        rest_opts = nullptr;
        rest_stats = nullptr;
        rest_method = nullptr;
    }
}

SQPmethod::SQPmethod(): prob(nullptr), param(nullptr), stats(nullptr), vars(nullptr), sub_QP(nullptr),
    rest_prob(nullptr), rest_opts(nullptr), rest_stats(nullptr), rest_method(nullptr), scaled_prob(nullptr), initCalled(false){}

SQPmethod::~SQPmethod(){
    delete vars;
    delete sub_QP;

    delete scaled_prob;
    delete rest_prob;
    delete rest_opts;
    delete rest_stats;
    delete rest_method;
}




SCQPmethod::SCQPmethod( Problemspec *problem, SQPoptions *parameters, SQPstats *statistics, Condenser *CND){

    prob = problem;
    param = parameters; param->optionsConsistency();
    stats = statistics;
    cond = CND;

    if (param->autoScaling){
        scaled_prob = new scaled_Problemspec(problem);
        prob = scaled_prob;
    }
    else{
        scaled_prob = nullptr;
        prob = problem;
    }
    vars = new SCQPiterate(prob, param, cond, true);

    // Check if there are options that are infeasible and set defaults accordingly
    if (param->sparseQP == 0){
        throw std::invalid_argument("SCQPmethod: Error, condensing only works with sparse QPs");
    }
    if (param->blockHess != 1){
        throw std::invalid_argument("SCQPmethod: Error, condensing requires block diagonal hessian for efficient linear algebra");
    }


    sub_QP = create_QPsolver(cond->condensed_num_vars, cond->condensed_num_cons, cond->condensed_num_hessblocks, param);

    initCalled = false;

    if (param->restoreFeas){
        //Setup condenser for the restoration problem
        int N_vblocks = cond->num_vblocks + cond->num_true_cons;
        int N_cblocks = cond->num_cblocks;
        int N_hessblocks = cond->num_hessblocks + cond->num_true_cons;
        int N_targets = cond->num_targets;

        rest_vblocks = new vblock[N_vblocks];
        rest_cblocks = new cblock[N_cblocks];
        rest_h_sizes = new int[N_hessblocks];
        rest_targets = new condensing_target[N_targets];

        for (int i = 0; i<cond->num_vblocks; i++){
            rest_vblocks[i] = cond->vblocks[i];
        }
        for (int i = cond->num_vblocks; i < N_vblocks; i++){
            rest_vblocks[i] = vblock(1, false);
        }

        for (int i = 0; i<cond->num_cblocks; i++){
            rest_cblocks[i] = cond->cblocks[i];
        }

        for (int i = 0; i<cond->num_hessblocks; i++){
            rest_h_sizes[i] = cond->hess_block_sizes[i];
        }
        for (int i = cond->num_hessblocks; i<N_hessblocks; i++){
            rest_h_sizes[i] = 1;
        }

        for (int i = 0; i<cond->num_targets; i++){
            rest_targets[i] = cond->targets[i];
        }
        rest_cond = new Condenser(rest_vblocks, N_vblocks, rest_cblocks, N_cblocks, rest_h_sizes, N_hessblocks, rest_targets, N_targets, 0);

        //Setup options for the restoration problem
        rest_opts = new SQPoptions();
        rest_opts->globalization = 1;
        rest_opts->whichSecondDerv = 0;
        rest_opts->restoreFeas = 0;
        //rest_opts->hessUpdate = param->hessUpdate;
        rest_opts->hessLimMem = 1;
        rest_opts->hessUpdate = 2;
        rest_opts->hessScaling = 4;
        rest_opts->maxConvQP = param->maxConvQP;
        rest_opts->opttol = param->opttol;
        rest_opts->nlinfeastol = param->nlinfeastol;
        rest_opts->QPsol = param->QPsol;
        rest_opts->QPsol_opts = param->QPsol_opts;
        
        //rest_opts->autoScaling = param->autoScaling;
        
        rest_prob = nullptr;
        rest_stats = nullptr;
        rest_method = nullptr;
    }
    else{
        rest_vblocks = nullptr;
        rest_cblocks = nullptr;
        rest_h_sizes = nullptr;
        rest_targets = nullptr;
        rest_cond = nullptr;
        rest_opts = nullptr;

        rest_prob = nullptr;
        rest_stats = nullptr;
        rest_method = nullptr;
    }
}

SCQPmethod::SCQPmethod(): cond(nullptr), rest_cond(nullptr), rest_vblocks(nullptr), rest_cblocks(nullptr), rest_h_sizes(nullptr), rest_targets(nullptr)
{};

SCQPmethod::~SCQPmethod(){
    delete[] rest_vblocks;
    delete[] rest_cblocks;
    delete[] rest_h_sizes;
    delete[] rest_targets;
    delete rest_cond;
}


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
        scaled_prob = new scaled_Problemspec(problem);
        prob = scaled_prob;
    }
    else{
        scaled_prob = nullptr;
        prob = problem;
    }
    vars = new SCQPiterate(prob, param, cond, true);

    // Check if there are options that are infeasible and set defaults accordingly
    if (param->sparseQP == 0){
        throw std::invalid_argument("SCQPmethod: Error, condensing only works with sparse QPs");
    }
    if (param->blockHess != 1){
        throw std::invalid_argument("SCQPmethod: Error, condensing requires block diagonal hessian for efficient linear algebra");
    }

    sub_QP = create_QPsolver(cond->condensed_num_vars, cond->condensed_num_cons, cond->condensed_num_hessblocks, param);

    initCalled = false;

    if (param->restoreFeas){
        //Setup condenser for the restoration problem
        int N_vblocks = cond->num_vblocks + cond->num_true_cons;
        int N_cblocks = cond->num_cblocks;
        int N_hessblocks = cond->num_hessblocks + cond->num_true_cons;
        int N_targets = cond->num_targets;

        rest_vblocks = new vblock[N_vblocks];
        rest_cblocks = new cblock[N_cblocks];
        rest_h_sizes = new int[N_hessblocks];
        rest_targets = new condensing_target[N_targets];

        for (int i = 0; i<cond->num_vblocks; i++){
            rest_vblocks[i] = cond->vblocks[i];
        }
        for (int i = cond->num_vblocks; i < N_vblocks; i++){
            rest_vblocks[i] = vblock(1, false);
        }

        for (int i = 0; i<cond->num_cblocks; i++){
            rest_cblocks[i] = cond->cblocks[i];
        }

        for (int i = 0; i<cond->num_hessblocks; i++){
            rest_h_sizes[i] = cond->hess_block_sizes[i];
        }
        for (int i = cond->num_hessblocks; i<N_hessblocks; i++){
            rest_h_sizes[i] = 1;
        }

        for (int i = 0; i<cond->num_targets; i++){
            rest_targets[i] = cond->targets[i];
        }
        rest_cond = new Condenser(rest_vblocks, N_vblocks, rest_cblocks, N_cblocks, rest_h_sizes, N_hessblocks, rest_targets, N_targets, 0);

        //Setup options for the restoration problem
        rest_opts = new SQPoptions();
        rest_opts->globalization = 1;
        rest_opts->whichSecondDerv = 0;
        rest_opts->restoreFeas = 0;
        //rest_opts->hessUpdate = param->hessUpdate;
        rest_opts->hessLimMem = 1;
        rest_opts->hessUpdate = 2;
        rest_opts->hessScaling = 4;
        rest_opts->maxConvQP = param->maxConvQP;
        rest_opts->opttol = param->opttol;
        rest_opts->nlinfeastol = param->nlinfeastol;
        rest_opts->QPsol = param->QPsol;

        //rest_opts->autoScaling = param->autoScaling;

        rest_prob = nullptr;
        rest_stats = nullptr;
        rest_method = nullptr;
    }
    else{
        rest_vblocks = nullptr;
        rest_cblocks = nullptr;
        rest_h_sizes = nullptr;
        rest_targets = nullptr;
        rest_cond = nullptr;
        rest_opts = nullptr;

        rest_prob = nullptr;
        rest_stats = nullptr;
        rest_method = nullptr;
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
        scaled_prob = new scaled_Problemspec(problem);
        prob = scaled_prob;
    }
    else{
        scaled_prob = nullptr;
        prob = problem;
    }
    vars = new SCQP_correction_iterate(prob, param, cond, true);

    // Check if there are options that are infeasible and set defaults accordingly
    if (param->sparseQP == 0){
        throw std::invalid_argument("SCQPmethod: Error, condensing only works with sparse QPs");
    }
    if (param->blockHess != 1){
        throw std::invalid_argument("SCQPmethod: Error, condensing requires block diagonal hessian for efficient linear algebra");
    }

    sub_QP = create_QPsolver(cond->condensed_num_vars, cond->condensed_num_cons, cond->condensed_num_hessblocks, param);

    initCalled = false;

    corrections = new Matrix[cond->num_targets];
    SOC_corrections = new Matrix[cond->num_targets];
    for (int tnum = 0; tnum < cond->num_targets; tnum++){
        corrections[tnum].Dimension(cond->targets_data[tnum].n_dep).Initialize(0.);
        SOC_corrections[tnum].Dimension(cond->targets_data[tnum].n_dep).Initialize(0.);
    }

    if (param->restoreFeas){
        //Setup condenser for the restoration problem
        int N_vblocks = cond->num_vblocks + cond->num_true_cons;
        int N_cblocks = cond->num_cblocks;
        int N_hessblocks = cond->num_hessblocks + cond->num_true_cons;
        int N_targets = cond->num_targets;

        rest_vblocks = new vblock[N_vblocks];
        rest_cblocks = new cblock[N_cblocks];
        rest_h_sizes = new int[N_hessblocks];
        rest_targets = new condensing_target[N_targets];

        for (int i = 0; i<cond->num_vblocks; i++){
            rest_vblocks[i] = cond->vblocks[i];
        }
        for (int i = cond->num_vblocks; i < N_vblocks; i++){
            rest_vblocks[i] = vblock(1, false);
        }

        for (int i = 0; i<cond->num_cblocks; i++){
            rest_cblocks[i] = cond->cblocks[i];
        }

        for (int i = 0; i<cond->num_hessblocks; i++){
            rest_h_sizes[i] = cond->hess_block_sizes[i];
        }
        for (int i = cond->num_hessblocks; i<N_hessblocks; i++){
            rest_h_sizes[i] = 1;
        }

        for (int i = 0; i<cond->num_targets; i++){
            rest_targets[i] = cond->targets[i];
        }
        rest_cond = new Condenser(rest_vblocks, N_vblocks, rest_cblocks, N_cblocks, rest_h_sizes, N_hessblocks, rest_targets, N_targets, 0);

        //Setup options for the restoration problem
        rest_opts = new SQPoptions();
        rest_opts->globalization = 1;
        rest_opts->whichSecondDerv = 0;
        rest_opts->restoreFeas = 0;
        rest_opts->hessLimMem = 1;
        rest_opts->hessUpdate = 2;
        rest_opts->hessScaling = 4;
        rest_opts->maxConvQP = param->maxConvQP;
        rest_opts->opttol = param->opttol;
        rest_opts->nlinfeastol = param->nlinfeastol;
        rest_opts->QPsol = param->QPsol;
        rest_opts->QPsol_opts = param->QPsol_opts;
        rest_opts->max_correction_steps = param->max_correction_steps;
        
        //rest_opts->autoScaling = param->autoScaling;

        rest_prob = nullptr;
        rest_stats = nullptr;
        rest_method = nullptr;
    }
    else{
        rest_vblocks = nullptr;
        rest_cblocks = nullptr;
        rest_h_sizes = nullptr;
        rest_targets = nullptr;
        rest_cond = nullptr;
        rest_opts = nullptr;

        rest_prob = nullptr;
        rest_stats = nullptr;
        rest_method = nullptr;
    }
}


SCQP_correction_method::~SCQP_correction_method(){
    delete[] corrections;
    delete[] SOC_corrections;
}









}