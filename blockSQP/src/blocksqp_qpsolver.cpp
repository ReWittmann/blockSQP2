/**
 * \file blocksqp_qpsolver.cpp
 * \author Reinhold Wittmann
 * \date 2024-
 *
 *  Implementation of interfaces to third party qp solvers
 *
 */


#include "blocksqp_qpsolver.hpp"
#include "blocksqp_general_purpose.hpp"
#include "blocksqp_problemspec.hpp"
#include "blocksqp_defs.hpp"
#include <cmath>
#include <chrono>

namespace blockSQP{


//QPsolver base class implemented methods
QPsolver::QPsolver(int n_QP_var, int n_QP_con, int n_QP_hessblocks, QPsolver_options *QPopts): nVar(n_QP_var), nCon(n_QP_con), nHess(n_QP_hessblocks), Qparam(QPopts){
    //For managing QP solution times
    default_time_limit = Qparam->max_QP_secs;
    custom_time_limit = Qparam->max_QP_secs;
    time_limit_type = 0;
    skip_timeRecord = false;

    dur_pos = 9; dur_count = 0;
    QPtime_avg = default_time_limit/2.5;
    for (int i = 0; i < 10; i++){
        solution_durations[i] = default_time_limit/2.5;
    }
    //Problem information
    convex_QP = false;

    //Flags
    use_hotstart = false;
};

QPsolver::~QPsolver(){}

void QPsolver::set_constr(const Matrix &constr_jac){
    throw blockSQP::NotImplementedError("QPsolver::set_constr(const Matrix &constr_jac)");
}

void QPsolver::set_constr(double *const jac_nz, int *const jac_row, int *const jac_colind){
    throw blockSQP::NotImplementedError("QPsolver::set_constr(const Sparse_Matrix &constr_jac)");
}

int QPsolver::get_QP_it(){return 0;}
double QPsolver::get_solutionTime(){return solution_durations[dur_pos];}


void QPsolver::recordTime(double solTime){
    //std::cout << "recorded time " << solTime << "\n";
    dur_pos = (dur_pos + 1)%10;
    solution_durations[dur_pos] = solTime;
    dur_count += int(dur_count < 10);
    QPtime_avg = 0.0;
    for (int i = 0; i < dur_count; i++){
        QPtime_avg += solution_durations[(dur_pos - i + 10)%10];
    }
    QPtime_avg /= dur_count;
    return;
}

void QPsolver::reset_timeRecord(){
    dur_pos = 0;
    dur_count = 0;
    QPtime_avg = default_time_limit/2.5;
}

void QPsolver::custom_timeLimit(double CTlim){
    time_limit_type = 2;
    custom_time_limit = CTlim;
    return;
}

//QPsolver factory, handle checks for linked QP solvers
QPsolver *create_QPsolver(int n_QP_var, int n_QP_con, int n_QP_hessblocks, int *blockIdx, SQPoptions *param){

    if (param->qpsol != param->qpsol_options->sol) throw ParameterError("qpsol_options for wrong qpsol, should have been caught by SQPoptions::optionsConsistency");
    //param->complete_QP_sol_opts has already been called through param->optionsConsistency in SQPmethod constructor, else we would need to call it here so all stanard QPsolvers options get added to param->qpsol_options

    #ifdef QPSOLVER_QPOASES
    if (param->qpsol == QPsolvers::qpOASES)
        return new qpOASES_solver(n_QP_var, n_QP_con, n_QP_hessblocks, blockIdx, static_cast<qpOASES_options*>(param->qpsol_options));
    #endif
    #ifdef QPSOLVER_GUROBI
    if (param->qpsol == QPsolvers::gurobi)
        return new gurobi_solver(n_QP_var, n_QP_con, n_QP_hessblocks, static_cast<gurobi_options*>(param->qpsol_options));
    #endif
    #ifdef QPSOLVER_QPALM
    if (param->qpsol == QPsolvers::qpalm)
        return new qpalm_solver(n_QP_var, n_QP_con, n_QP_hessblocks, static_cast<qpalm_options*>(param->qpsol_options));
    #endif

    throw ParameterError("Selected QP solver not specified and linked, should have been caught by SQPoptions::optionsConsistency");
}


////////////////////////////////////////////////////////////////
/////////////Interfaces to (third party) QP solvers/////////////
////////////////////////////////////////////////////////////////



///////////////////////
///qpOASES interface///
///////////////////////

#ifdef QPSOLVER_QPOASES

qpOASES_solver::qpOASES_solver(int n_QP_var, int n_QP_con, int n_QP_hessblocks, int *blockIdx, qpOASES_options *QPopts):
                    QPsolver(n_QP_var, n_QP_con, n_QP_hessblocks, QPopts){
    if (static_cast<qpOASES_options*>(Qparam)->sparsityLevel < -1)
        throw ParameterError("qpOASES_solver class cannot choose sparsityLevel automatically, set to 0 - dense, 1 - sparse or 2 - schur");
    else if (static_cast<qpOASES_options*>(Qparam)->sparsityLevel < 0 || static_cast<qpOASES_options*>(Qparam)->sparsityLevel > 2)
        throw ParameterError("Invalid value sparsityLevel option for qpOASES_solver");

    if (static_cast<qpOASES_options*>(Qparam)->sparsityLevel < 2){
        qp = std::unique_ptr<qpOASES::SQProblem>(new qpOASES::SQProblem(nVar, nCon));
        qpSave = std::unique_ptr<qpOASES::SQProblem>(new qpOASES::SQProblem(nVar, nCon));
        qpCheck = std::unique_ptr<qpOASES::SQProblem>(new qpOASES::SQProblem(nVar, nCon));
    }
    else{
        qp = std::unique_ptr<qpOASES::SQProblemSchur>(new qpOASES::SQProblemSchur(nVar, nCon, qpOASES::HST_UNKNOWN, 50));
        qpSave = std::unique_ptr<qpOASES::SQProblemSchur>(new qpOASES::SQProblemSchur(nVar, nCon, qpOASES::HST_UNKNOWN, 50));
        qpCheck = std::unique_ptr<qpOASES::SQProblemSchur>(new qpOASES::SQProblemSchur(nVar, nCon, qpOASES::HST_UNKNOWN, 50));
    }
    
    //Owned
    A_qp = nullptr;
    H_qp = nullptr;

    lb = std::make_unique<double[]>(nVar);
    ub = std::make_unique<double[]>(nVar);
    lbA = std::make_unique<double[]>(nCon);
    ubA = std::make_unique<double[]>(nCon);

    h_qp = nullptr;
    if (static_cast<qpOASES_options*>(Qparam)->sparsityLevel > 0){
        int hess_nzCount = 0;
        for (int i = 0; i < n_QP_hessblocks; i++){
            hess_nzCount += (blockIdx[i+1] - blockIdx[i])*(blockIdx[i+1] - blockIdx[i]);
        }
        //Allocate enough memory to support all structurally nonzero elements being nonzero.
        hess_nz = std::make_unique<double[]>(hess_nzCount);
        hess_row = std::make_unique<int[]>(hess_nzCount);
        hess_colind = std::make_unique<int[]>(n_QP_var + 1);
        hess_loind = std::make_unique<int[]>(n_QP_var + 1);
    }
    else hess_nz = std::make_unique<double[]>(nVar*nVar);
    
    //Options
    opts.enableEqualities = qpOASES::BT_TRUE;
    opts.initialStatusBounds = qpOASES::ST_INACTIVE;
    switch(static_cast<qpOASES_options*>(Qparam)->printLevel){
        case 0: opts.printLevel = qpOASES::PL_NONE;     break;
        case 1: opts.printLevel = qpOASES::PL_LOW;      break;
        case 2: opts.printLevel = qpOASES::PL_MEDIUM;   break;
        case 3: opts.printLevel = qpOASES::PL_HIGH;     break;
    }
    opts.numRefinementSteps = 2;
    opts.epsLITests =  2.2204e-08;
    opts.terminationTolerance = static_cast<qpOASES_options*>(Qparam)->terminationTolerance;
}

qpOASES_solver::~qpOASES_solver(){}

void qpOASES_solver::set_lin(const Matrix &grad_obj){
    h_qp = grad_obj.array;
    return;
}


void qpOASES_solver::set_bounds(const Matrix &lb_x, const Matrix &ub_x, const Matrix &lb_A, const Matrix &ub_A){
    //by default, qpOASES defines +-inifinity as +-1e20 (see qpOASES Constants.hpp), set bounds accordingly
    for (int i = 0; i < nVar; i++){
        if (lb_x(i) > -Qparam->inf)
            lb[i] = lb_x(i);
        else
            lb[i] = -1e20;

        if (ub_x(i) < Qparam->inf)
            ub[i] = ub_x(i);
        else
            ub[i] = 1e20;
    }
    for (int i = 0; i < nCon; i++){
        if (lb_A(i) > -Qparam->inf)
            lbA[i] = lb_A(i);
        else
            lbA[i] = -1e20;

        if (ub_A(i) < Qparam->inf)
            ubA[i] = ub_A(i);
        else
            ubA[i] = 1e20;
    }
    return;
}


void qpOASES_solver::set_constr(const Matrix &constr_jac){
    Transpose(constr_jac, jacT);
    A_qp = std::make_unique<qpOASES::DenseMatrix>(nCon, nVar, nVar, jacT.array);
    return;
}

void qpOASES_solver::set_constr(double *const jac_nz, int *const jac_row, int *const jac_colind){
    A_qp = std::make_unique<qpOASES::SparseMatrix>(nCon, nVar, jac_row, jac_colind, jac_nz);
    matrices_changed = true;
    return;
}

void qpOASES_solver::set_hess(SymMatrix *const hess, bool pos_def, double regularizationFactor){
    convex_QP = pos_def;
    double regFactor;
    if (convex_QP)
        regFactor = regularizationFactor;
    else
        regFactor = 0.0;

    if (static_cast<qpOASES_options*>(Qparam)->sparsityLevel > 0){
        convertHessian_noalloc(Qparam->eps, hess, nHess, nVar, regFactor, hess_nz.get(), hess_row.get(), hess_colind.get(), hess_loind.get());
        H_qp = std::make_unique<qpOASES::SymSparseMat>(nVar, nVar, hess_row.get(), hess_colind.get(), hess_nz.get());
        dynamic_cast<qpOASES::SymSparseMat*>(H_qp.get())->createDiagInfo();
    }
    else{
        convertHessian_noalloc(hess, nHess, nVar, regFactor, hess_nz.get());
        H_qp = std::make_unique<qpOASES::SymDenseMat>(nVar, nVar, nVar, hess_nz.get());
    }
    matrices_changed = true;
    return;
}

int qpOASES_solver::solve(Matrix &deltaXi, Matrix &lambdaQP){
    double QPtime;

    if (convex_QP)  opts.enableInertiaCorrection = qpOASES::BT_TRUE;
    else            opts.enableInertiaCorrection = qpOASES::BT_FALSE;

    qp->setOptions(opts);

    // Other variables for qpOASES

    //Set time limit to prevent wasting time on ill conditioned QPs:
    // 0 - limit by 2.5*(average solution time), 2 - limit by custom time, else - limit by maximum time set in options
    if (time_limit_type == 0)
        QPtime = std::min(2.5*QPtime_avg, default_time_limit);
    else if (time_limit_type == 2)
        QPtime = custom_time_limit;
    else
        QPtime = default_time_limit;

    //std::cout << "QPtime = " << QPtime << "\n";
    QP_it = Qparam->max_QP_it;
    qpOASES::SolutionAnalysis solAna;
    qpOASES::returnValue ret;

    if ((qp->getStatus() == qpOASES::QPS_HOMOTOPYQPSOLVED ||
         qp->getStatus() == qpOASES::QPS_SOLVED) && use_hotstart){
        if (matrices_changed)
            ret = qp->hotstart(H_qp.get(), h_qp, A_qp.get(), lb.get(), ub.get(), lbA.get(), ubA.get(), QP_it, &QPtime);
        else
            ret = qp->hotstart(h_qp, lb.get(), ub.get(), lbA.get(), ubA.get(), QP_it, &QPtime);
    }
    else
        ret = qp->init(H_qp.get(), h_qp, A_qp.get(), lb.get(), ub.get(), lbA.get(), ubA.get(), QP_it, &QPtime);


    if (!convex_QP && ret == qpOASES::SUCCESSFUL_RETURN){
        if (static_cast<qpOASES_options*>(Qparam)->sparsityLevel == 2){
            *dynamic_cast<qpOASES::SQProblemSchur*>(qpCheck.get()) = *dynamic_cast<qpOASES::SQProblemSchur*>(qp.get());
            ret = solAna.checkCurvatureOnStronglyActiveConstraints(dynamic_cast<qpOASES::SQProblemSchur*>(qpCheck.get()));
        }
        else{
            *qpCheck = *qp;
            ret = solAna.checkCurvatureOnStronglyActiveConstraints(qpCheck.get());
        }
    }

    if (deltaXi.m != nVar) throw std::invalid_argument("QPsolver.solve: Error in argument deltaXi, wrong matrix size");
    if (lambdaQP.m != nVar + nCon) throw std::invalid_argument("QPsolver.solve: Error in argument lambdaQP, wrong matrix size");


    // Return codes: 0 - success, 1 - took too long/too many steps, 2 definiteness condition violated or QP unbounded, 3 - QP was infeasible, 4 - other error
    if (ret == qpOASES::SUCCESSFUL_RETURN){
        use_hotstart = true;
        matrices_changed = false;
        
        qp->getPrimalSolution(deltaXi.array);
        qp->getDualSolution(lambdaQP.array);
        if (!skip_timeRecord) recordTime(QPtime);
        else skip_timeRecord = false;

        QP_it += 1;
        *qpSave = *qp;

        return 0;
    }

    *qp = *qpSave;

    //std::cout << "QP could not be solved, qpOASES ret is " << ret << "\n";

    if (ret == qpOASES::RET_SETUP_AUXILIARYQP_FAILED)
        QP_it = 1;
    
    if( ret == qpOASES::RET_MAX_NWSR_REACHED )
        return 1;
    else if( ret == qpOASES::RET_HESSIAN_NOT_SPD ||
             ret == qpOASES::RET_HESSIAN_INDEFINITE ||
             ret == qpOASES::RET_INIT_FAILED_UNBOUNDEDNESS ||
             ret == qpOASES::RET_QP_UNBOUNDED ||
             ret == qpOASES::RET_HOTSTART_STOPPED_UNBOUNDEDNESS ){
        return 2;}
    else if( ret == qpOASES::RET_INIT_FAILED_INFEASIBILITY ||
             ret == qpOASES::RET_QP_INFEASIBLE ||
             ret == qpOASES::RET_HOTSTART_STOPPED_INFEASIBILITY ){
        return 3;}
    return 4;
}

int qpOASES_solver::get_QP_it(){return QP_it;}

#endif




//////////////////////
///gurobi interface///
//////////////////////

#ifdef QPSOLVER_GUROBI
gurobi_solver::gurobi_solver(int n_QP_var, int n_QP_con, int n_QP_hessblocks, gurobi_options *QPopts): QPsolver(n_QP_var, n_QP_con, n_QP_hessblocks, QPopts), obj_lin(0), obj_quad(0){
    //Check for inconsistent options before construction
    env = new GRBEnv();
    model = new GRBModel(env);
    
    model->set(GRB_IntParam_OutputFlag, static_cast<gurobi_options*>(Qparam)->OutputFlag);
    model->set(GRB_IntParam_Method, static_cast<gurobi_options*>(Qparam)->Method);
    model->set(GRB_IntParam_NumericFocus, static_cast<gurobi_options*>(Qparam)->NumericFocus);
    model->set(GRB_IntParam_Presolve, static_cast<gurobi_options*>(Qparam)->Presolve);
    model->set(GRB_IntParam_Aggregate, static_cast<gurobi_options*>(Qparam)->Aggregate);

    model->set(GRB_DoubleParam_OptimalityTol, static_cast<gurobi_options*>(Qparam)->OptimalityTol);
    model->set(GRB_DoubleParam_FeasibilityTol, static_cast<gurobi_options*>(Qparam)->FeasibilityTol);
    model->set(GRB_DoubleParam_PSDTol, static_cast<gurobi_options*>(Qparam)->PSDTol);

    model->set(GRB_IntParam_NonConvex, 0);


    QP_vars = model->addVars(nVar, GRB_CONTINUOUS);
    QP_cons_lb = model->addConstrs(nCon);
    QP_cons_ub = model->addConstrs(nCon);
    for (int i = 0; i < nCon; i++){
        QP_cons_lb[i].set(GRB_CharAttr_Sense, GRB_GREATER_EQUAL);
        QP_cons_ub[i].set(GRB_CharAttr_Sense, GRB_LESS_EQUAL);
    }
}

gurobi_solver::~gurobi_solver(){
    delete[] QP_vars;
    delete[] QP_cons_lb;
    delete[] QP_cons_ub;
    delete model;
    delete env;
}

void gurobi_solver::set_lin(const Matrix &grad_obj){
    obj_lin = 0;
    obj_lin.addTerms(grad_obj.array, QP_vars, nVar);
    return;
}

void gurobi_solver::set_bounds(const Matrix &lb_x, const Matrix &ub_x, const Matrix &lb_A, const Matrix &ub_A){
    for (int i = 0; i < nVar; i++){
        if (lb_x(i) > -Qparam->inf)
            QP_vars[i].set(GRB_DoubleAttr_LB, lb_x(i));
        else
            QP_vars[i].set(GRB_DoubleAttr_LB, -GRB_INFINITY);

        if (ub_x(i) < Qparam->inf)
            QP_vars[i].set(GRB_DoubleAttr_UB, ub_x(i));
        else
            QP_vars[i].set(GRB_DoubleAttr_UB, GRB_INFINITY);
    }
    for (int i = 0; i < nCon; i++){
        if (lb_A(i) > -Qparam->inf)
            QP_cons_lb[i].set(GRB_DoubleAttr_RHS, lb_A(i));
        else
            QP_cons_lb[i].set(GRB_DoubleAttr_RHS, -GRB_INFINITY);

        if (ub_A(i) < Qparam->inf)
            QP_cons_ub[i].set(GRB_DoubleAttr_RHS, ub_A(i));
        else
            QP_cons_ub[i].set(GRB_DoubleAttr_RHS, GRB_INFINITY);
    }
    return;
}

void gurobi_solver::set_constr(const Matrix &constr_jac){
    for (int i = 0; i < nCon; i++){
        for (int j = 0; j < nVar; j++){
            model->chgCoeff(QP_cons_lb[i], QP_vars[j], constr_jac(i,j));
            model->chgCoeff(QP_cons_ub[i], QP_vars[j], constr_jac(i,j));
        }
    }
    return;
}

void gurobi_solver::set_constr(double *const jac_nz, int *const jac_row, int *const jac_colind){
    for (int j = 0; j < nVar; j++){
        for (int i = jac_colind[j]; i < jac_colind[j+1]; i++){
            model->chgCoeff(QP_cons_lb[jac_row[i]], QP_vars[j], jac_nz[i]);
            model->chgCoeff(QP_cons_ub[jac_row[i]], QP_vars[j], jac_nz[i]);
        }
    }
    return;
}

void gurobi_solver::set_hess(SymMatrix *const hess, bool pos_def, double regularizationFactor){
    convex_QP = pos_def;
    double regFactor;
    if (convex_QP)
        regFactor = regularizationFactor;
    else
        regFactor = 0;

    obj_quad = 0;

    int offset = 0;
    for (int k = 0; k < nHess; k++){
        for (int i = 0; i < hess[k].m; i++){
            for (int j = 0; j < i; j++){
                obj_quad += QP_vars[offset + i] * QP_vars[offset + j] * hess[k](i,j);
            }
            obj_quad += 0.5 * QP_vars[offset + i] * QP_vars[offset + i] * (hess[k](i,i) + regFactor);
        }
        offset += hess[k].m;
    }

    //Unnecessary right now as gurobi only supplies lagrange multipliers for convex QPs
    //model->set(GRB_IntParam_NonConvex, int(!pos_def));
    return;
}


int gurobi_solver::solve(Matrix &deltaXi, Matrix &lambdaQP){
    model->setObjective(obj_quad + obj_lin, GRB_MINIMIZE);

    //Set time limit to prevent wasting time on ill conditioned QPs:
    // 0 - limit by 2.5*(average solution time), 2 - limit by custom time, else - limit by maximum time set in options
    if (time_limit_type == 0)
        model->set(GRB_DoubleParam_TimeLimit, std::min(2.5*QPtime_avg, default_time_limit));
    else if (time_limit_type == 2)
        model->set(GRB_DoubleParam_TimeLimit, custom_time_limit);
    else
        model->set(GRB_DoubleParam_TimeLimit, default_time_limit);


    try{
        model->optimize();
    }
    catch (GRBException &e){
        return 4;
    }

    int ret = model->get(GRB_IntAttr_Status);
    if (ret == 2){
        for (int i = 0; i < nVar; i++){
            deltaXi(i) = QP_vars[i].get(GRB_DoubleAttr_X);
            lambdaQP(i) = QP_vars[i].get(GRB_DoubleAttr_RC);
        }
        for (int i = 0; i < nCon; i++){
            lambdaQP(nVar + i) = QP_cons_lb[i].get(GRB_DoubleAttr_Pi);
            lambdaQP(nVar + i) += QP_cons_ub[i].get(GRB_DoubleAttr_Pi);
        }

        if (!skip_timeRecord) recordTime(model->get(GRB_DoubleAttr_Runtime));
        else skip_timeRecord = false;

        return 0;
    }
    else if (ret == 3)
        return 3;
    else if (ret == 4)
        return 2;
    else if (ret == 7 || ret == 9 || ret == 16)
        return 1;
    return 4;
}


int gurobi_solver::get_QP_it(){
    if (model->get(GRB_IntParam_Method) == 2)
        return model->get(GRB_IntAttr_BarIterCount);
    else if (model->get(GRB_IntParam_Method) == 0 || model->get(GRB_IntParam_Method) == 1)
        return int(model->get(GRB_DoubleAttr_IterCount));
    return model->get(GRB_IntAttr_BarIterCount) + int(model->get(GRB_DoubleAttr_IterCount));
}

#endif



/////////////////////
///qpalm interface///
/////////////////////

#ifdef QPSOLVER_QPALM
qpalm_solver::qpalm_solver(int n_QP_var, int n_QP_con, int n_QP_hessblocks, qpalm_options *QPopts): QPsolver(n_QP_var, n_QP_con, n_QP_hessblocks, QPopts),
data(qpalm::index_t(n_QP_var), qpalm::index_t(n_QP_var + n_QP_con)), Q(n_QP_var, n_QP_var), q(n_QP_var), A(n_QP_con + n_QP_var, n_QP_var), lb(n_QP_con + n_QP_var), ub(n_QP_con + n_QP_var){
    
    
    settings.eps_abs     = 1e-9;
    settings.eps_abs_in = 1.0e-4;
    settings.eps_rel     = 1e-9;
    settings.eps_rel_in = 1.0e-4;
    settings.eps_prim_inf = 1e-8;
    settings.eps_dual_inf = 1e-6;
    
    settings.max_iter    = 1000;
    settings.inner_max_iter = 100;
    
    settings.verbose = 1;
}
qpalm_solver::~qpalm_solver(){};

void qpalm_solver::set_lin(const Matrix &grad_obj){
    for (int i = 0; i < nVar; i++){
        q(i) = grad_obj(i);
    }
    return;
}

void qpalm_solver::set_bounds(const Matrix &lb_x, const Matrix &ub_x, const Matrix &lb_A, const Matrix &ub_A){
    for (int i = 0; i < nCon; i++){
        lb(i) = lb_A(i);
        ub(i) = ub_A(i);
    }
    for (int i = 0; i < nVar; i++){
        lb(nCon + i) = lb_x(i);
        ub(nCon + i) = ub_x(i);
    }
    return;
}

void qpalm_solver::set_constr(const Matrix &constr_jac){
    triplets.reserve((nVar+nCon)*nCon);
    triplets.resize(0);
    for (int i = 0; i < nCon; i++){
        for (int j = 0; j < nVar; j++){
            triplets.push_back(qpalm::triplet_t(i, j, constr_jac(i,j)));
        }
    }
    for (int i = 0; i < nVar; i++){
        triplets.push_back(qpalm::triplet_t(nCon + i, i, 1.0));
    }
    A.setFromTriplets(triplets.begin(), triplets.end());
    return;
}

void qpalm_solver::set_constr(double *const jac_nz, int *const jac_row, int *const jac_colind){
    triplets.reserve(jac_colind[nVar] + nVar);
    triplets.resize(0);
    for (int j = 0; j < nVar; j++){
        for (int i = jac_colind[j]; i < jac_colind[j+1]; i++){
            triplets.push_back(qpalm::triplet_t(jac_row[i], j, jac_nz[i]));
        }
    }
    for (int i = 0; i < nVar; i++){
        triplets.push_back(qpalm::triplet_t(nCon + i, i, 1.0));
    }
    A.setFromTriplets(triplets.begin(), triplets.end());
    return;
}

void qpalm_solver::set_hess(SymMatrix *const hess, bool pos_def, double regularizationFactor){
    triplets.resize(0);
    int offset = 0;
    for (int iBlock = 0; iBlock < nHess; iBlock++){
        for (int i = 0; i < hess[iBlock].m; i++){
            for (int j = 0; j < hess[iBlock].m; j++){
                triplets.push_back(qpalm::triplet_t(offset + i, offset + j, hess[iBlock](i,j) + regularizationFactor*int(i == j)));
            }
        }
        offset += hess[iBlock].m;
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());
    settings.nonconvex = !pos_def;
    return;
}

int qpalm_solver::solve(Matrix &deltaXi, Matrix &lambdaQP){
    data.set_Q(Q);
    data.q = q;
    data.set_A(A);
    data.c = 0;
    data.bmin = lb;
    data.bmax = ub;

    //Set time limit to prevent wasting time on ill conditioned QPs:
    // 0 - limit by 2.5*(average solution time), 2 - limit by custom time, else - limit by maximum time set in options
    if (time_limit_type == 0)
        settings.time_limit = std::min(2.5*QPtime_avg, default_time_limit);
    else if (time_limit_type == 2)
        settings.time_limit = custom_time_limit;
    else
        settings.time_limit = default_time_limit;

    qpalm::Solver solver = {data, settings};
    solver.solve();

    qpalm::SolutionView sol = solver.get_solution();
    info = solver.get_info();
    std::cout << "qpalm returned, info is " << info.status << "\n";
    
    if (!strcmp(info.status, "solved")){
        for (int i = 0; i < nCon; i++){
            //qpalm defines Lagrangian as f + lambda^T g, we have f - lambda^T g. Change sign of Lagrange multipliers.
            lambdaQP(nVar + i) = -sol.y(i);
        }
        for (int i = 0; i < nVar; i++){
            deltaXi(i) = sol.x(i);
            lambdaQP(i) = -sol.y(nCon + i);
        }

        if (!skip_timeRecord) recordTime(info.run_time);
        else skip_timeRecord = false;

        return 0;
    }
    return 4;
}

int qpalm_solver::get_QP_it(){return info.iter;};

#endif



}//namespace blockSQP



