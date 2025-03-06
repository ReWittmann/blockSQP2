#include "blocksqp_qpsolver.hpp"
#include "blocksqp_general_purpose.hpp"
#include "blocksqp_problemspec.hpp"
#include "blocksqp_defs.hpp"
#include <cmath>
#include <chrono>


#ifdef QPSOLVER_QPOASES
    #include "qpOASES.hpp"
#endif

#ifdef QPSOLVER_GUROBI
    #include "gurobi_c++.h"
#endif


namespace blockSQP{

void convertHessian(blockSQP::SymMatrix *const hess, int nBlocks, int nVar, double regularizationFactor,
                                            double *&hessNz){
    if (hessNz == NULL)
        hessNz = new double[nVar * nVar];

    int bsize, bstart = 0, ind = 0;
    //Iterate over hessian blocks
    for (int h = 0; h <nBlocks; h++){
        bsize = hess[h].m;
        //Iterate over second dimension
        for (int j = 0; j < bsize; j++){
            //Iterate over first dimension
             //Segment above hessian block
            for (int i = 0; i < bstart; i++){
                hessNz[ind] = 0;
                ++ind;
            }
             //Hessian block
            for (int i = 0; i < hess[h].m; i++){
                hessNz[ind] = hess[h](i, j);
                //NEW
                if (i == j) hessNz[ind] += regularizationFactor;

                ++ind;
            }
             //Segment below hessian block
            for (int i = bstart + bsize; i < nVar; i++){
                hessNz[ind] = 0;
                ++ind;
            }
        }
        bstart += bsize;
    }
    return;
}


void convertHessian(double eps, blockSQP::SymMatrix *const hess_, int nBlocks, int nVar, double regularizationFactor,
                             double *&hessNz_, int *&hessIndRow_, int *&hessIndCol_, int *&hessIndLo_ ){
    int iBlock, count, colCountTotal, rowOffset, i, j;
    int nnz, nCols, nRows;

    // 1) count nonzero elements
    nnz = 0;
    for (iBlock=0; iBlock<nBlocks; iBlock++){
        for (i=0; i<hess_[iBlock].m; i++){
            //Always count diagonal elements (regularization)
            if (fabs(hess_[iBlock](i,i)) > eps || fabs(hess_[iBlock](i,i)) + regularizationFactor > eps)
                nnz++;

            for (j = i + 1; j < hess_[iBlock].m; j++){
                if (fabs(hess_[iBlock]( i,j )) > eps)
                    nnz += 2;
            }
        }
    }

    delete[] hessNz_;
    delete[] hessIndRow_;
    delete[] hessIndCol_;
    delete[] hessIndLo_;

    hessNz_ = new double[nnz];
    hessIndRow_ = new int[nnz];
    hessIndCol_ = new int[nVar + 1];
    hessIndLo_ = new int[nVar];

    // 2) store matrix entries columnwise in hessNz
    count = 0; // runs over all nonzero elements
    colCountTotal = 0; // keep track of position in large matrix
    rowOffset = 0;
    for (iBlock = 0; iBlock < nBlocks; iBlock++){
        nCols = hess_[iBlock].m;
        nRows = hess_[iBlock].m;

        for (i = 0; i < nCols; i++){
            // column 'colCountTotal' starts at element 'count'
            hessIndCol_[colCountTotal] = count;

            for (j = 0; j < nRows; j++){
                //if (hess_[iBlock]( i,j ) > eps || -hess_[iBlock]( i,j ) > eps ){
                if (fabs(hess_[iBlock](i,j)) > eps || (i == j && fabs(hess_[iBlock](i,j)) + regularizationFactor > eps)){
                    hessNz_[count] = hess_[iBlock](i, j);
                    if (i == j) hessNz_[count] += regularizationFactor;

                    hessIndRow_[count] = j + rowOffset;
                    count++;
                }
            }
            colCountTotal++;
        }
        rowOffset += nRows;
    }
    hessIndCol_[colCountTotal] = count;

    // 3) Set reference to lower triangular matrix
    for( j=0; j<nVar; j++ )
    {
        for( i=hessIndCol_[j]; i<hessIndCol_[j+1] && hessIndRow_[i]<j; i++);
        hessIndLo_[j] = i;
    }

    if( count != nnz ){
         std::cout << "Error in convertHessian: " << count << " elements processed, should be " << nnz << " elements!\n";
    }
}


///////////////////////

//QPsolver base class implemented methods
QPsolver::QPsolver(int n_QP_var, int n_QP_con, int n_QP_hessblocks, QPSOLVER_options *QPopts): nVar(n_QP_var), nCon(n_QP_con), nHess(n_QP_hessblocks), Qparam(QPopts){
    //For managing QP solution times
    default_time_limit = Qparam->maxTimeQP;
    custom_time_limit = Qparam->maxTimeQP;
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
QPsolver *create_QPsolver(int n_QP_var, int n_QP_con, int n_QP_hessblocks, SQPoptions *param){

    if (param->QPsol != param->QPsol_opts->sol) throw ParameterError("QPsol_opts for wrong QPsol, should have been caught by SQPoptions::optionsConsistency");
    //param->complete_QP_sol_opts has already been called through param->optionsConsistency in SQPmethod constructor, else we would need to call it here so all stanard QPsolvers options get added to param->QPsol_opts

    #ifdef QPSOLVER_QPOASES
    if (param->QPsol == QPSOLVER::qpOASES)
        return new qpOASES_solver(n_QP_var, n_QP_con, n_QP_hessblocks, param->sparseQP, static_cast<qpOASES_options*>(param->QPsol_opts));
    #endif
    #ifdef QPSOLVER_GUROBI
    if (param->QPsol == QPSOLVER::gurobi)
        return new gurobi_solver(n_QP_var, n_QP_con, n_QP_hessblocks, static_cast<gurobi_options*>(param->QPsol_opts));
    #endif
    #ifdef QPSOLVER_QPALM
    if (param->QPsol == QPSOLVER::qpalm)
        return new qpalm_solver(n_QP_var, n_QP_con, n_QP_hessblocks, static_cast<qpalm_options*>(param->QPsol_opts));
    #endif

    throw ParameterError("Selected QP solver not specified and linked, should have been caught by SQPoptions::optionsConsistency");
}

//Interfaces to (third party) QP solvers

#ifdef QPSOLVER_QPOASES

qpOASES_solver::qpOASES_solver(int n_QP_var, int n_QP_con, int n_QP_hessblocks, int SPARSE, qpOASES_options *QPopts): QPsolver(n_QP_var, n_QP_con, n_QP_hessblocks, QPopts), sparseQP(SPARSE){
    //Initialize all qpOASES objects and data formats
    if (sparseQP < 2){
        qp = new qpOASES::SQProblem(nVar, nCon);
        qpSave = new qpOASES::SQProblem(nVar, nCon);
        qp_check = new qpOASES::SQProblem(nVar, nCon);
    }
    else{
        qp = new qpOASES::SQProblemSchur(nVar, nCon, qpOASES::HST_UNKNOWN, 50);
        qpSave = new qpOASES::SQProblemSchur(nVar, nCon, qpOASES::HST_UNKNOWN, 50);
        qp_check = new qpOASES::SQProblemSchur(nVar, nCon, qpOASES::HST_UNKNOWN, 50);
    }
    //Owned
    A_qp = nullptr;
    H_qp = nullptr;
    lb = new double[nVar];
    ub = new double[nVar];
    lbA = new double[nCon];
    ubA = new double[nCon];

    //Not owned
    h_qp = nullptr;

    //Owned
    if (sparseQP) hess_nz = nullptr;
    else hess_nz = new double[nVar*nVar];

    hess_row = nullptr;
    hess_colind = nullptr;
    hess_loind = nullptr;
    
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

qpOASES_solver::~qpOASES_solver(){
    delete[] hess_nz;
    delete[] hess_row;
    delete[] hess_colind;
    delete[] hess_loind;
    delete[] lb;
    delete[] ub;
    delete[] lbA;
    delete[] ubA;

    delete A_qp;
    delete H_qp;
    delete qp;
    delete qpSave;
    delete qp_check;

}

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
    delete A_qp;
    Transpose(constr_jac, jacT);
    A_qp = new qpOASES::DenseMatrix(nCon, nVar, nVar, jacT.array);
    return;
}

void qpOASES_solver::set_constr(double *const jac_nz, int *const jac_row, int *const jac_colind){
    delete A_qp;
    A_qp = new qpOASES::SparseMatrix(nCon, nVar,
            jac_row, jac_colind, jac_nz);

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

    delete H_qp;

    if (sparseQP){
        convertHessian(Qparam->eps, hess, nHess, nVar, regFactor, hess_nz, hess_row, hess_colind, hess_loind);
        H_qp = new qpOASES::SymSparseMat(nVar, nVar, hess_row, hess_colind, hess_nz);
        dynamic_cast<qpOASES::SymSparseMat*>(H_qp)->createDiagInfo();
    }
    else{
        convertHessian(hess, nHess, nVar, regFactor, hess_nz);
        H_qp = new qpOASES::SymDenseMat(nVar, nVar, nVar, hess_nz);
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
    QP_it = Qparam->maxItQP;
    qpOASES::SolutionAnalysis solAna;
    qpOASES::returnValue ret;

    if ((qp->getStatus() == qpOASES::QPS_HOMOTOPYQPSOLVED ||
         qp->getStatus() == qpOASES::QPS_SOLVED) && use_hotstart){
        if (matrices_changed)
            ret = qp->hotstart(H_qp, h_qp, A_qp, lb, ub, lbA, ubA, QP_it, &QPtime);
        else
            ret = qp->hotstart(h_qp, lb, ub, lbA, ubA, QP_it, &QPtime);
    }
    else
        ret = qp->init(H_qp, h_qp, A_qp, lb, ub, lbA, ubA, QP_it, &QPtime);


    if (!convex_QP && ret == qpOASES::SUCCESSFUL_RETURN){
        if (sparseQP == 2){
            *dynamic_cast<qpOASES::SQProblemSchur*>(qp_check) = *dynamic_cast<qpOASES::SQProblemSchur*>(qp);
            ret = solAna.checkCurvatureOnStronglyActiveConstraints(dynamic_cast<qpOASES::SQProblemSchur*>(qp_check));
        }
        else{
            *qp_check = *qp;
            ret = solAna.checkCurvatureOnStronglyActiveConstraints(qp_check);
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
            //Diagonal part, add a regularization, else QP solution seems to fail in some instances
            //obj_quad += 0.5 * QP_vars[offset + i] * QP_vars[offset + i] * (hess[k](i,i) + static_cast<gurobi_options*>(Qparam)->regularization_factor);
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

/*
double gurobi_solver::get_solutionTime(){
    return model->get(GRB_DoubleAttr_Runtime);
}*/
#endif


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

    settings.time_limit = 1.0;
    qpalm::Solver solver = {data, settings};
    //qpalm::Solver solver(&data, settings);
    solver.solve();

    qpalm::SolutionView sol = solver.get_solution();
    info = solver.get_info();
    std::cout << "qpalm returned, info is " << info.status << "\n";
    
    if (!strcmp(info.status, "solved")){
        for (int i = 0; i < nCon; i++){
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



}

//


