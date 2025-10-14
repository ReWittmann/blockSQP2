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
 * \file blocksqp_restoration.cpp
 * \author Dennis Janka
 * \date 2012-2015
 *
 *  Implementation of RestorationProblem class that describes a
 *  minimum l_2-norm NLP.
 * 
 * \modifications
 *  \author Reinhold Wittmann
 *  \date 2023-2025
 */
 
 
#include "blocksqp_restoration.hpp"
#include "blocksqp_matrix.hpp"
#include <cmath>
#include <limits>

namespace blockSQP{


void abstractRestorationProblem::update_xi_ref(const Matrix &xiReference){return;}


RestorationProblem::RestorationProblem(Problemspec *parentProblem, const Matrix &xiReference, double param_rho, double param_zeta): rho(param_rho), zeta(param_zeta){
    parent = parentProblem;
    
    n_vblocks = parent->n_vblocks + 1;
    vblocks = new vblock[parent->n_vblocks + 1];
    std::copy(parent->vblocks, parent->vblocks + parent->n_vblocks, vblocks);
    vblocks[parent->n_vblocks] = vblock(parent->nCon, false);
    
    /*
    xiRef.Dimension( parent->nVar ).Initialize(0.);
    for(int i=0; i<parent->nVar; i++)
        xiRef( i ) = xiReference( i );
    */
    
    xi_ref = xiReference;

    /* nCon slack variables */
    nVar = parent->nVar + parent->nCon;
    nCon = parent->nCon;
    nnz = parent->nnz + parent->nCon;

    /* Block structure: One additional block for every slack variable */
    nBlocks = parent->nBlocks+nCon;
    blockIdx = new int[nBlocks+1];
    for(int i = 0; i < parent->nBlocks + 1; i++)
        blockIdx[i] = parent->blockIdx[i];
    for(int i = parent->nBlocks + 1; i<nBlocks + 1; i++)
        blockIdx[i] = blockIdx[i-1] + 1;

    /* Set bounds */
    objLo = 0.0;
    objUp = 1.0e20;

    lb_var.Dimension(nVar).Initialize(-1.0e20);
    ub_var.Dimension(nVar).Initialize(1.0e20);
    for (int i = 0; i < parent->nVar; i++){
        lb_var(i) = parent->lb_var(i);
        ub_var(i) = parent->ub_var(i);
    }

    lb_con.Dimension(nCon);
    ub_con.Dimension(nCon);
    for (int i = 0; i < nCon; i++){
        lb_con(i) = parent->lb_con(i);
        ub_con(i) = parent->ub_con(i);
    }

}

RestorationProblem::~RestorationProblem(){
    delete[] blockIdx;
    delete[] vblocks;
    //delete[] jacNzOrig;
    //delete[] jacIndRowOrig;
    //delete[] jacIndColOrig;
}


void RestorationProblem::update_xi_ref(const Matrix &xiReference){
    xi_ref = xiReference;
}

void RestorationProblem::evaluate(
        const Matrix &xi, const Matrix &lambda, double *objval, Matrix &constr,
        Matrix &gradObj, double *jacNz, int *jacIndRow, int *jacIndCol,
        SymMatrix *hess, int dmode, int *info){
    int iCon, i;
    double diff, regTerm;
    Matrix xiOrig, slack;

    // The first nVar elements of the variable vector correspond to the variables of the original problem
    xiOrig.Submatrix( xi, parent->nVar, 1, 0, 0 );
    slack.Submatrix( xi, parent->nCon, 1, parent->nVar, 0 );

    // Evaluate constraints of the original problem
    parent->evaluate( xiOrig, lambda, objval, constr,
                      gradObj, jacNz, jacIndRow, jacIndCol, hess, dmode, info );

    // Subtract slacks
    for( iCon=0; iCon<nCon; iCon++ )
        constr(iCon) -= slack(iCon);

    
    /* Evaluate objective: minimize slacks plus deviation from reference point */
    if( dmode < 0 )
        return;

    *objval = 0.0;

    // First part: sum of slack variables
    for( i=0; i<nCon; i++ )
        *objval += slack( i ) * slack( i );
    *objval = 0.5 * rho * (*objval);

    // Second part: regularization term
    regTerm = 0.0;
    for (i = 0; i < parent->nVar; i++){
        diff = xiOrig(i) - xi_ref(i);
        regTerm += diagScale(i) * diagScale(i) * diff * diff;
    }
    regTerm = 0.5 * zeta * regTerm;
    *objval += regTerm;
    
    if (dmode > 0){
        // compute objective gradient

        // gradient w.r.t. xi (regularization term)
        for (i = 0; i < parent->nVar; i++)
            gradObj(i) = zeta*diagScale(i)*diagScale(i) * (xiOrig(i) - xi_ref(i));

        // gradient w.r.t. slack variables
        for (i = parent->nVar; i < nVar; i++)
            gradObj(i) = rho * xi(i);
    }

    *info = 0;
}


void RestorationProblem::evaluate( const Matrix &xi, const Matrix &lambda,
                                   double *objval, Matrix &constr,
                                   Matrix &gradObj, Matrix &constrJac,
                                   SymMatrix *hess, int dmode, int *info )
{
    int iCon, i;
    double diff, regTerm;
    Matrix xiOrig, constrJacOrig;
    Matrix slack;

    // The first nVar elements of the variable vector correspond to the variables of the original problem
    xiOrig.Submatrix(xi, parent->nVar, 1, 0, 0);
    slack.Submatrix(xi, parent->nCon, 1, parent->nVar, 0);
    if (dmode != 0)
        constrJacOrig.Submatrix(constrJac, parent->nCon, parent->nVar, 0, 0);

    // Evaluate constraints of the original problem
    parent->evaluate(xiOrig, lambda, objval, constr,
                     gradObj, constrJacOrig, hess, dmode, info);

    // Subtract slacks
    for (iCon = 0; iCon < nCon; iCon++){
        constr(iCon) -= slack(iCon);
    }

    //Evaluate objective: minimize slacks plus deviation from reference point
    if( dmode < 0 )
        return;

    *objval = 0.0;

    // First part: sum of slack variables
    for( i=0; i<nCon; i++ )
        *objval += slack( i ) * slack( i );
    *objval = 0.5 * rho * (*objval);
    
    // Second part: regularization term
    regTerm = 0.0;
    for( i=0; i<parent->nVar; i++ )
    {
        diff = xiOrig( i ) - xi_ref( i );
        regTerm += diagScale(i) * diagScale(i) * diff * diff;
    }
    regTerm = 0.5 * zeta * regTerm;
    *objval += regTerm;

    if( dmode > 0 )
    {// compute objective gradient

        // gradient w.r.t. xi (regularization term)
        for( i=0; i<parent->nVar; i++ )
            gradObj( i ) = zeta * diagScale( i ) * diagScale( i ) * (xiOrig( i ) - xi_ref( i ));

        // gradient w.r.t. slack variables
        for( i=parent->nVar; i<nVar; i++ )
            gradObj( i ) = rho * slack( i );
    }

    *info = 0;
}


void RestorationProblem::initialize(Matrix &xi, Matrix &lambda, double *jacNz, int *jacIndRow, int *jacIndCol){
    int i, info;
    double objval;
    Matrix constrRef;

    xi_parent.Submatrix(xi, parent->nVar, 1, 0, 0);
    slack.Submatrix(xi, parent->nCon, 1, parent->nVar, 0);

    // Allocate the sparse jacobian of the parent problem
    
    //jacNzOrig = new double[parent->nnz];
    //jacIndRowOrig = new int[parent->nnz];
    //jacIndColOrig = new int[parent->nVar + 1];


    //parent->initialize( xiOrig, lambda, jacNzOrig, jacIndRowOrig, jacIndColOrig );
    parent->initialize(xi_parent, lambda, jacNz, jacIndRow, jacIndCol);
    //nnzOrig = jacIndColOrig[parent->nVar];

    // Copy sparse Jacobian from parent problem
    /*
    for (i = 0; i < nnzOrig; i++){
        jacNz[i] = jacNzOrig[i];
        jacIndRow[i] = jacIndRowOrig[i];
    }
    for (i = 0; i <= parent->nVar; i++){
        jacIndCol[i] = jacIndColOrig[i];
    }
    */
    

    // Jacobian entries for slacks (one nonzero entry per column)
    for (i = parent->nnz; i < nnz; i++){
        jacNz[i] = -1.0;
        jacIndRow[i] = i - parent->nnz;
    }
    for (i = parent->nVar; i < nVar + 1; i++){
        jacIndCol[i] = parent->nnz + i - parent->nVar;
    }

    // The reference point is the starting value for the restoration phase
    for (i = 0; i < parent->nVar; i++){
        xi_parent(i) = xi_ref(i);
    }

    // Initialize slack variables such that the constraints are feasible
    constrRef.Dimension(nCon);
    parent->evaluate(xi_parent, &objval, constrRef, &info);

    for (i = 0; i < nCon; i++){
        if (constrRef(i) < parent->lb_con(i))// if lower bound is violated
            slack(i) = constrRef(i) - parent->lb_con(i);
        else if (constrRef(i) > parent->ub_con(i))// if upper bound is violated
            slack(i) = constrRef(i) - parent->ub_con(i);
    }

    // Set diagonal scaling matrix
    diagScale.Dimension(parent->nVar).Initialize(1.0);
    for (i = 0; i < parent->nVar; i++){
        if (fabs(xi_ref(i)) > 1.0)
            diagScale(i) = 1.0/fabs(xi_ref(i));
    }

    lambda.Initialize(0.0);
}


void RestorationProblem::initialize(Matrix &xi, Matrix &lambda, Matrix &constrJac){
    int i, info;
    double objval;
    Matrix constrJacOrig, constrRef;

    xi_parent.Submatrix( xi, parent->nVar, 1, 0, 0 );
    slack.Submatrix( xi, parent->nCon, 1, parent->nVar, 0 );
    constrJacOrig.Submatrix( constrJac, parent->nCon, parent->nVar, 0, 0 );

    // Call initialize of the parent problem to set up linear constraint matrix correctly
    parent->initialize( xi_parent, lambda, constrJacOrig );

    // Jacobian entries for slacks
    for( i=0; i<parent->nCon; i++ )
        constrJac( i, parent->nVar+i ) = -1.0;

    // The reference point is the starting value for the restoration phase
    for( i=0; i<parent->nVar; i++ )
        xi_parent( i ) = xi_ref( i );

    // Initialize slack variables such that the constraints are feasible
    constrRef.Dimension( nCon );
    parent->evaluate( xi_parent, &objval, constrRef, &info );

    for( i=0; i<nCon; i++ )
    {
        if(constrRef(i) <= parent->lb_con(i))// if lower bound is violated
            slack( i ) = constrRef( i ) - parent->lb_con(i);
        else if( constrRef( i ) > parent->ub_con(i) )// if upper bound is violated
            slack( i ) = constrRef( i ) - parent->ub_con(i);
    }

    // Set diagonal scaling matrix
    diagScale.Dimension( parent->nVar ).Initialize( 1.0 );
    for( i=0; i<parent->nVar; i++ )
        if( fabs( xi_ref( i ) ) > 1.0 )
            diagScale( i ) = 1.0 / fabs( xi_ref( i ) );

    lambda.Initialize( 0.0 );
}


void RestorationProblem::printVariables( const Matrix &xi, const Matrix &lambda, int verbose )
{
    int k;

    printf("\n<|----- Original Variables -----|>\n");
    for( k=0; k<parent->nVar; k++ )
        //printf("%7i: %-30s   %7g <= %10.3g <= %7g   |   mul=%10.3g\n", k+1, parent->varNames[k], bl(k), xi(k), bu(k), lambda(k));
        printf("%7i: x%-5i   %7g <= %10.3g <= %7g   |   mul=%10.3g\n", k+1, k, lb_var(k), xi(k), ub_var(k), lambda(k));
    printf("\n<|----- Slack Variables -----|>\n");
    for( k=parent->nVar; k<nVar; k++ )
        printf("%7i: slack   %7g <= %10.3g <= %7g   |   mul=%10.3g\n", k+1, lb_var(k), xi(k), ub_var(k), lambda(k));
}


void RestorationProblem::printConstraints( const Matrix &constr, const Matrix &lambda )
{
    printf("\n<|----- Constraints -----|>\n");
    for( int k=0; k<nCon; k++ )
        //printf("%5i: %-30s   %7g <= %10.4g <= %7g   |   mul=%10.3g\n", k+1, parent->conNames[parent->nVar+k], bl(nVar+k), constr(k), bu(nVar+k), lambda(nVar+k));
        printf("%5i: c%-5i   %7g <= %10.4g <= %7g   |   mul=%10.3g\n", k+1, k, lb_con(k), constr(k), ub_con(k), lambda(nVar+k));
}


void RestorationProblem::printInfo()
{
    printf("Minimum 2-norm NLP to find a point acceptable to the filter\n");
}


void RestorationProblem::recover_xi(const Matrix &xi_rest, Matrix &xi_orig){
    for (int i = 0; i < parent->nVar; i++){
        xi_orig(i) = xi_rest(i);
    }
    return;
}

void RestorationProblem::recover_lambda(const Matrix &lambda_rest, Matrix &lambda_orig){
    for (int i = 0; i < parent->nVar; i++){
        lambda_orig(i) = lambda_rest(i);
    }
    for (int i = parent->nVar; i < parent->nVar + parent->nCon; i++){
        lambda_orig(i) = lambda_rest(parent->nCon + i);
    }
    return;
}


//##############################################


/*


condensable_Restoration_Problem::condensable_Restoration_Problem(Problemspec *parent_Problem, Condenser *parent_CND, const Matrix &xi_Reference): parent(parent_Problem), parent_cond(parent_CND), xi_ref(xi_Reference)
{


    // nCon slack variables
    nVar = parent->nVar + parent->nCon;
    nCon = parent->nCon + parent->nVar - parent_cond->condensed_num_vars;
    nnz = parent->nnz + parent->nCon + (parent->nVar - parent_cond->condensed_num_vars);

    // Block structure: One additional block for every slack variable
    nBlocks = parent->nBlocks + parent->nCon;
    blockIdx = new int[nBlocks + 1];
    for(int i = 0; i<parent->nBlocks + 1; i++){
        blockIdx[i] = parent->blockIdx[i];
    }
    for(int i = parent->nBlocks + 1; i<nBlocks+1; i++){
        blockIdx[i] = blockIdx[i-1]+1;
    }

    //Set bounds, no bounds for dependent variables, add dependent variable bounds to constraints
    objLo = 0.0;
    objUp = 1.0e20;

    //Bounds for original variables, bounds for slack variables, bounds for original constraints and conditions, bounds for dependent variables as constraints
    lb_var.Dimension(nVar).Initialize(-1.0e20);
    ub_var.Dimension(nVar).Initialize(1.0e20);

    //Variable bounds: No bounds for dependent variables
    int ind_1 = 0;
    for (int i = 0; i < parent_cond->num_vblocks; i++){
        if (!parent_cond->vblocks[i].dependent){
            for (int j = ind_1; j < ind_1 + parent_cond->vblocks[i].size; j++){
                lb_var(j) = parent->lb_var(j);
                ub_var(j) = parent->ub_var(j);
            }
        }
        ind_1 += parent_cond->vblocks[i].size;
    }
    //Slack variables are unbounded

    //Bounds for constraints and conditions
    lb_con.Dimension(nCon).Initialize(-1.0e20);
    ub_con.Dimension(nCon).Initialize(1.0e20);

    for (int i = 0; i < parent->nCon; i++){
        lb_con(i) = parent->lb_con(i);
        ub_con(i) = parent->ub_con(i);
    }

    //Bounds for dependent variables that are added as relaxed constraints
    ind_1 = parent->nCon;
    int ind_2 = 0;
    for (int i = 0; i < parent_cond->num_vblocks; i++){
        if (parent_cond->vblocks[i].dependent){
            for (int j = 0; j < parent_cond->vblocks[i].size; j++){
                lb_con(ind_1 + j) = parent->lb_var(ind_2 + j);
                ub_con(ind_1 + j) = parent->ub_var(ind_2 + j);
            }
            ind_1 += parent_cond->vblocks[i].size;
            ind_2 += parent_cond->vblocks[i].size;
        }
        else{
            ind_2 += parent_cond->vblocks[i].size;
        }
    }

}

condensable_Restoration_Problem::~condensable_Restoration_Problem(){

    delete[] jac_orig_nz;
    delete[] jac_orig_row;
    delete[] jac_orig_colind;
}


void condensable_Restoration_Problem::build_restoration_jacobian(const Sparse_Matrix &jac_orig, Sparse_Matrix &jac_restoration){

    int num_dep_vars = parent->nVar - parent_cond->condensed_num_vars;

    //Upper part of restoration jacobian: Slacks for true constraints
    double *constr_slack_nz = new double[parent_cond->num_true_cons];
    int *constr_slack_row = new int[parent_cond->num_true_cons];
    int *constr_slack_colind = new int[parent->nCon + 1];
    int ind_1 = 0;
    int ind_2 = 0;
    constr_slack_colind[0] = 0;

    for (int i = 0; i < parent_cond->num_cblocks; i++){
        if (parent_cond->cblocks[i].removed){
            ind_2 += parent_cond->cblocks[i].size;
        }
        else{
            for (int j = 0; j < parent_cond->cblocks[i].size; j++){
                constr_slack_nz[ind_1 + j] = -1.0;
                constr_slack_row[ind_1 + j] = ind_2 + j;
                constr_slack_colind[ind_1 + j + 1] = ind_1 + j + 1;
            }
            ind_1 += parent_cond->cblocks[i].size;
            ind_2 += parent_cond->cblocks[i].size;
        }
    }

    for (int i = parent_cond->num_true_cons; i < parent->nCon; i++){
        constr_slack_colind[i + 1] = parent_cond->num_true_cons;
    }

    Sparse_Matrix constr_slack(parent->nCon, parent->nCon, parent_cond->num_true_cons, constr_slack_nz, constr_slack_row, constr_slack_colind);

    //Lower part of restoration jacobian: Dependent variable bounds, relaxed with slack variables
    double *dep_bounds_nz = new double[num_dep_vars * 2];
    int *dep_bounds_row = new int[num_dep_vars * 2];
    int *dep_bounds_colind = new int[parent->nVar + parent->nCon + 1];
    ind_1 = 0;
    ind_2 = 0;
    dep_bounds_colind[0] = 0;

    //Left part: Derivatives with respect to the dependent variables
    for (int i = 0; i < parent_cond->num_vblocks; i++){
        if (parent_cond->vblocks[i].dependent){
            for (int j = 0; j < parent_cond->vblocks[i].size; j++){
                dep_bounds_nz[ind_1 + j] = 1.;
                dep_bounds_row[ind_1 + j] = ind_1 + j;
                dep_bounds_colind[ind_2 + j + 1] = ind_1 + j + 1;
            }
            ind_1 += parent_cond->vblocks[i].size;
            ind_2 += parent_cond->vblocks[i].size;
        }
        else{
            for (int j = 0; j < parent_cond->vblocks[i].size; j++){
                dep_bounds_colind[ind_2 + j + 1] = ind_1;
            }
            ind_2 += parent_cond->vblocks[i].size;
        }
    }

    //Middle part: No nonzero entries, slack variables already used up for true constraints

    for (int i = 0; i < parent_cond->num_true_cons; i++){
        dep_bounds_colind[ind_2 + i + 1] = ind_1;
    }
    ind_2 += parent_cond->num_true_cons;

    //Right part: Derivatives with respect to the remaining slack variables
    for (int i = 0; i < num_dep_vars; i++){
        dep_bounds_nz[ind_1 + i] = -1.;
        dep_bounds_row[ind_1 + i] = i;
        dep_bounds_colind[ind_2 + i + 1] = ind_1 + i + 1;
    }

    Sparse_Matrix dep_bounds(num_dep_vars, parent->nVar + parent->nCon, 2*num_dep_vars, dep_bounds_nz, dep_bounds_row, dep_bounds_colind);

    std::vector<Sparse_Matrix> mats(2);
    mats[0] = std::move(jac_orig);
    mats[1] = std::move(constr_slack);

    mats[0] = horzcat(mats);
    mats[1] = std::move(dep_bounds);

    jac_restoration = vertcat(mats);
}



void condensable_Restoration_Problem::initialize( Matrix &xi, Matrix &lambda, double *&jacNz, int *&jacIndRow, int *&jacIndCol )
{

    int info;
    double objval;
    Matrix xi_orig, slack;

    xi_orig.Submatrix( xi, parent->nVar, 1, 0, 0 );
    slack.Submatrix( xi, parent->nCon, 1, parent->nVar, 0 );

    //Initialize the sparse jacobian of the parent problem

    jac_orig_nz = new double[parent->nnz];
    jac_orig_row = new int[parent->nnz];
    jac_orig_colind = new int[parent->nVar + 1];

    // Call initialize of the parent problem. There, the sparse Jacobian is intialized
    parent->initialize(xi_orig, lambda, jac_orig_nz, jac_orig_row, jac_orig_colind);

    //Save original jacobian and build restoration-jacobian
    Sparse_Matrix jac_orig(parent->nCon, parent->nVar, jac_orig_colind[parent->nVar], jac_orig_nz, jac_orig_row, jac_orig_colind);
    Sparse_Matrix jac_restoration;

    build_restoration_jacobian(jac_orig, jac_restoration);

    for (int i = 0; i < jac_restoration.nnz; i++){
        jacNz[i] = jac_restoration.nz[i];
        jacIndRow[i] = jac_restoration.row[i];
    }
    for (int i = 0; i <= jac_restoration.n; i++){
        jacIndCol[i] = jac_restoration.colind[i];
    }

    //jacNz = jac_restoration.nz;
    //jacIndRow = jac_restoration.row;
    //jacIndCol = jac_restoration.colind;

    jac_restoration.nz = nullptr;
    jac_restoration.row = nullptr;
    jac_restoration.colind = nullptr;
    jac_restoration.m = 0; jac_restoration.n = 0; jac_restoration.nnz = 0;

    jac_orig.nz = nullptr;
    jac_orig.row = nullptr;
    jac_orig.colind = nullptr;
    jac_orig.m = 0; jac_orig.n = 0; jac_orig.nnz = 0;


    // The reference point is the starting value for the restoration phase
    for(int i=0; i<parent->nVar; i++){
        xi_orig(i) = xi_ref(i);
    }

    // Initialize slack variables such that the constraints are feasible, allocate and use vector for original constraints
    constr_orig.Dimension(parent->nCon);
    parent->evaluate(xi_orig, &objval, constr_orig, &info);

    int ind_1 = 0;
    int ind_2 = 0;
    for (int i = 0; i < parent_cond->num_cblocks; i++){
        if (parent_cond->cblocks[i].removed){
            ind_2 += parent_cond->cblocks[i].size;
        }
        else{
            for (int j = 0; j < parent_cond->cblocks[i].size; j++){
                if (constr_orig(ind_2 + j) < parent->lb_con(ind_2 + j)){
                    slack(ind_1 + j) = constr_orig(ind_2 + j) - parent->lb_con(ind_2 + j);
                }
                else if (constr_orig(ind_2 + j) > parent->ub_con(ind_2 + j)){
                    slack(ind_1 + j) = constr_orig(ind_2 + j) - parent->ub_con(ind_2 + j);
                }
                else{
                    slack(ind_1 + j) = 0.;
                }
            }
            ind_1 += parent_cond->cblocks[i].size;
            ind_2 += parent_cond->cblocks[i].size;
        }
    }


    //Leave slack variables for dependent variable bounds as zero, as setting them to ensure feasibility requires condensed problem data.
    for (int i = parent_cond->num_true_cons; i < parent->nCon; i++){
        slack(i) = 0;
    }

    // Set diagonal scaling matrix
    diagScale.Dimension(parent->nVar).Initialize(1.0);
    for(int i = 0; i < parent->nVar; i++){
        if(fabs(xi_ref(i)) > 1.0){
            diagScale(i) = 1.0/fabs(xi_ref(i));
        }
    }

    // Regularization factor zeta and rho \todo wie setzen?
    zeta = 1.0e-3;
    rho = 1.0e3;

    lambda.Initialize(0.0);
}



void condensable_Restoration_Problem::evaluate(
                                const Matrix &xi, const Matrix &lambda,
                                double *objval, Matrix &constr,
                                Matrix &gradObj, double *&jacNz, int *&jacIndRow, int *&jacIndCol,
                                SymMatrix *&hess, int dmode, int *info){

    double diff, regTerm;
    Matrix xi_orig, slack, relaxed_constr, dep_bounds;

    // The first nVar elements of the variable vector correspond to the variables of the original problem
    xi_orig.Submatrix( xi, parent->nVar, 1, 0, 0 );
    slack.Submatrix( xi, parent->nCon, 1, parent->nVar, 0 );

    // Evaluate constraints of the original problem
    parent->evaluate(xi_orig, lambda, objval, constr_orig,
                      gradObj, jac_orig_nz, jac_orig_row, jac_orig_colind, hess, dmode, info);

    // Subtract slacks from true constraints (not conditions)
    relaxed_constr.Submatrix(constr, parent->nCon, 1, 0, 0);

    int ind_1 = 0;
    int ind_2 = 0;

    for (int i = 0; i < parent_cond->num_cblocks; i++){
        if (parent_cond->cblocks[i].removed){
            for (int j = 0; j < parent_cond->cblocks[i].size; j++){
                relaxed_constr(ind_1 + j) = constr_orig(ind_1 + j);
            }
            ind_1 += parent_cond->cblocks[i].size;
        }
        else{
            for (int j = 0; j < parent_cond->cblocks[i].size; j++){
                relaxed_constr(ind_1 + j) = constr_orig(ind_1 + j) - slack(ind_2 + j);
            }
            ind_1 += parent_cond->cblocks[i].size;
            ind_2 += parent_cond->cblocks[i].size;
        }
    }

    dep_bounds.Submatrix(constr, parent->nVar - parent_cond->condensed_num_vars, 1, parent->nCon, 0);
    ind_1 = 0;
    ind_2 = 0;

    for (int i = 0; i < parent_cond->num_vblocks; i++){
        if (parent_cond->vblocks[i].dependent){
            for (int j = 0; j < parent_cond->vblocks[i].size; j++){
                dep_bounds(ind_1 + j) = xi_orig(ind_2 + j) - slack(parent_cond->num_true_cons + ind_1 + j);
            }
            ind_1 += parent_cond->vblocks[i].size;
            ind_2 += parent_cond->vblocks[i].size;
        }
        else{
            ind_2 += parent_cond->vblocks[i].size;
        }
    }

    // Evaluate objective: minimize slacks plus deviation from reference point
    if( dmode < 0 ){
        return;
    }

    *objval = 0.0;

    // First part: sum of slack variables
    for (int i = 0; i < parent->nCon; i++){
        *objval += slack( i ) * slack( i );
    }
    *objval = 0.5 * rho * (*objval);

    // Second part: regularization term
    regTerm = 0.0;
    for(int i = 0; i < parent->nVar; i++){
        diff = xi_orig(i) - xi_ref(i);
        regTerm += diagScale(i) * diagScale(i) * diff * diff;
    }
    regTerm = 0.5 * zeta * regTerm;
    *objval += regTerm;

    if(dmode > 0)
    {// compute objective gradient

        // gradient w.r.t. xi (regularization term)
        for(int i=0; i<parent->nVar; i++){
            gradObj(i) = zeta * diagScale(i) * diagScale(i) * (xi_orig(i) - xi_ref(i));
        }

        // gradient w.r.t. slack variables
        for(int i=parent->nVar; i<nVar; i++){
            gradObj(i) = rho * xi(i);
        }

        Sparse_Matrix jac_orig(parent->nCon, parent->nVar, jac_orig_colind[parent->nVar], jac_orig_nz, jac_orig_row, jac_orig_colind);
        Sparse_Matrix jac_restoration;
        build_restoration_jacobian(jac_orig, jac_restoration);

        for (int i = 0; i < jac_restoration.nnz; i++){
            jacNz[i] = jac_restoration.nz[i];
            jacIndRow[i] = jac_restoration.row[i];
        }
        for (int i = 0; i <= jac_restoration.n; i++){
            jacIndCol[i] = jac_restoration.colind[i];
        }

        jac_orig.nz = nullptr;
        jac_orig.row = nullptr;
        jac_orig.colind = nullptr;
        jac_orig.m = 0; jac_orig.n = 0; jac_restoration.nnz = 0;
    }

    *info = 0;
}


void condensable_Restoration_Problem::recover_multipliers(const Matrix &lambda_rest, Matrix &lambda_orig){
    int ind_1 = 0;
    int ind_2 = parent->nVar + 2 * parent->nCon;

    for (int i = 0; i < parent_cond->num_vblocks; i++){
        if (parent_cond->vblocks[i].dependent){
            for (int j = 0; j < parent_cond->vblocks[i].size; j++){
                lambda_orig(ind_1 + j) = lambda_rest(ind_2 + j);
            }
            ind_1 += parent_cond->vblocks[i].size;
            ind_2 += parent_cond->vblocks[i].size;
        }
        else{
            for (int j = 0; j < parent_cond->vblocks[i].size; j++){
                lambda_orig(ind_1 + j) = lambda_rest(ind_1 + j);
            }
            ind_1 += parent_cond->vblocks[i].size;
        }
    }

    for (int i = parent->nVar; i < parent->nVar + parent->nCon; i++){
        lambda_orig(i) = lambda_rest(parent->nCon + i);
    }
}


void condensable_Restoration_Problem::recover_multipliers(const Matrix &lambda_rest, Matrix &lambda_orig, double &lambda_step_norm){
    int ind_1 = 0;
    int ind_2 = parent->nVar + 2 * parent->nCon;
    lambda_step_norm = 0.;

    for (int i = 0; i < parent_cond->num_vblocks; i++){
        if (parent_cond->vblocks[i].dependent){
            for (int j = 0; j < parent_cond->vblocks[i].size; j++){
                if (lambda_step_norm < fabs(lambda_rest(ind_2 + j) - lambda_orig(ind_1 + j))){
                    lambda_step_norm = fabs(lambda_rest(ind_2 + j) - lambda_orig(ind_1 + j));
                }
                lambda_orig(ind_1 + j) = lambda_rest(ind_2 + j);
            }
            ind_1 += parent_cond->vblocks[i].size;
            ind_2 += parent_cond->vblocks[i].size;
        }
        else{
            for (int j = 0; j < parent_cond->vblocks[i].size; j++){
                if (lambda_step_norm < fabs(lambda_rest(ind_1 + j) - lambda_orig(ind_1 + j))){
                    lambda_step_norm = fabs(lambda_rest(ind_1 + j) - lambda_orig(ind_1 + j));
                }
                lambda_orig(ind_1 + j) = lambda_rest(ind_1 + j);
            }
            ind_1 += parent_cond->vblocks[i].size;
        }
    }

    for (int i = parent->nVar; i < parent->nVar + parent->nCon; i++){
        lambda_orig(i) = lambda_rest(parent->nCon + i);
    }
}



feasibility_Problem::feasibility_Problem(Problemspec *parent_Problem, Condenser *parent_CND): parent(parent_Problem), parent_cond(parent_CND){
    //Create condenser-class for restoration problem


    // nCon slack variables
    nVar = parent->nVar + parent->nCon;
    nCon = parent->nCon + parent->nVar - parent_cond->condensed_num_vars;

    // Block structure: One additional block for every slack variable
    nBlocks = parent->nBlocks + parent->nCon;
    blockIdx = new int[nBlocks + 1];
    for(int i = 0; i<parent->nBlocks + 1; i++){
        blockIdx[i] = parent->blockIdx[i];
    }
    for(int i = parent->nBlocks + 1; i<nBlocks+1; i++){
        blockIdx[i] = blockIdx[i-1]+1;
    }

    //Set bounds, no bounds for dependent variables, add dependent variable bounds to constraints
    objLo = 0.0;
    objUp = 1.0e20;

    //Bounds for original variables, bounds for slack variables, bounds for original constraints and conditions, bounds for dependent variables as constraints
    lb_var.Dimension(nVar).Initialize(-1.0e20);
    ub_var.Dimension(nVar).Initialize(1.0e20);

    //Variable bounds: No bounds for dependent variables
    int ind_1 = 0;
    for (int i = 0; i < parent_cond->num_vblocks; i++){
        if (!parent_cond->vblocks[i].dependent){
            for (int j = ind_1; j < ind_1 + parent_cond->vblocks[i].size; j++){
                lb_var(j) = parent->lb_var(j);
                ub_var(j) = parent->ub_var(j);
            }
        }
        ind_1 += parent_cond->vblocks[i].size;
    }
    //Slack variables are unbounded

    //Bounds for constraints and conditions
    lb_con.Dimension(nCon).Initialize(-1.0e20);
    ub_con.Dimension(nCon).Initialize(1.0e20);

    for (int i = 0; i < parent->nCon; i++){
        lb_con(i) = parent->lb_con(i);
        ub_con(i) = parent->ub_con(i);
    }

    //Bounds for dependent variables that are added as relaxed constraints
    ind_1 = parent->nCon;
    int ind_2 = 0;
    for (int i = 0; i < parent_cond->num_vblocks; i++){
        if (parent_cond->vblocks[i].dependent){
            for (int j = 0; j < parent_cond->vblocks[i].size; j++){
                lb_con(ind_1 + j) = parent->lb_var(ind_2 + j);
                ub_con(ind_1 + j) = parent->ub_var(ind_2 + j);
            }
            ind_1 += parent_cond->vblocks[i].size;
            ind_2 += parent_cond->vblocks[i].size;
        }
        else{
            ind_2 += parent_cond->vblocks[i].size;
        }
    }
}

feasibility_Problem::~feasibility_Problem(){

    delete[] jac_orig_nz;
    delete[] jac_orig_row;
    delete[] jac_orig_colind;
}

void feasibility_Problem::initialize(Matrix &xi, Matrix &lambda, double *&jacNz, int *&jacIndRow, int *&jacIndCol){

    int info;
    double objval;
    Matrix xi_orig, slack;

    xi_orig.Submatrix( xi, parent->nVar, 1, 0, 0 );
    slack.Submatrix( xi, parent->nCon, 1, parent->nVar, 0 );

    jac_orig_nz = new double[parent->nnz];
    jac_orig_row = new int[parent->nnz];
    jac_orig_colind = new int[parent->nVar + 1];

    // Call initialize of the parent problem. There, the sparse Jacobian is allocated
    parent->initialize(xi_orig, lambda, jac_orig_nz, jac_orig_row, jac_orig_colind);

    //Save original jacobian and build restoration-jacobian
    Sparse_Matrix jac_orig(parent->nCon, parent->nVar, jac_orig_colind[parent->nVar], jac_orig_nz, jac_orig_row, jac_orig_colind);
    Sparse_Matrix jac_restoration;

    build_restoration_jacobian(jac_orig, jac_restoration);

    jacNz = jac_restoration.nz;
    jacIndRow = jac_restoration.row;
    jacIndCol = jac_restoration.colind;

    jac_restoration.nz = nullptr;
    jac_restoration.row = nullptr;
    jac_restoration.colind = nullptr;
    jac_restoration.m = 0; jac_restoration.n = 0; jac_restoration.nnz = 0;

    jac_orig.nz = nullptr;
    jac_orig.row = nullptr;
    jac_orig.colind = nullptr;
    jac_orig.m = 0; jac_orig.n = 0; jac_orig.nnz = 0;

    // Initialize slack variables such that the constraints are feasible, allocate and use matrix for original constraints
    constr_orig.Dimension(parent->nCon);
    parent->evaluate(xi_orig, &objval, constr_orig, &info);

    int ind_1 = 0;
    int ind_2 = 0;
    for (int i = 0; i < parent_cond->num_cblocks; i++){
        if (parent_cond->cblocks[i].removed){
            ind_2 += parent_cond->cblocks[i].size;
        }
        else{
            for (int j = 0; j < parent_cond->cblocks[i].size; j++){
                if (constr_orig(ind_2 + j) < parent->lb_con(ind_2 + j)){
                    slack(ind_1 + j) = constr_orig(ind_2 + j) - parent->lb_con(ind_2 + j);
                }
                else if (constr_orig(ind_2 + j) > parent->ub_con(ind_2 + j)){
                    slack(ind_1 + j) = constr_orig(ind_2 + j) - parent->ub_con(ind_2 + j);
                }
                else{
                    slack(ind_1 + j) = 0.;
                }
            }
            ind_1 += parent_cond->cblocks[i].size;
            ind_2 += parent_cond->cblocks[i].size;
        }
    }


    //Leave slack variables for dependent variable bounds as zero, as setting them to ensure feasibility requires condensed problem data.
    for (int i = parent_cond->num_true_cons; i < parent->nCon; i++){
        slack(i) = 0;
    }

    lambda.Initialize(0.0);
}

void feasibility_Problem::evaluate(const Matrix &xi, const Matrix &lambda,
                            double *objval, Matrix &constr,
                           Matrix &gradObj, double *&jacNz, int *&jacIndRow, int *&jacIndCol,
                           SymMatrix *&hess, int dmode, int *info){

    double diff, regTerm;
    Matrix xi_orig, slack, relaxed_constr, dep_bounds;

    // The first nVar elements of the variable vector correspond to the variables of the original problem
    xi_orig.Submatrix( xi, parent->nVar, 1, 0, 0 );
    slack.Submatrix( xi, parent->nCon, 1, parent->nVar, 0 );

    // Evaluate constraints of the original problem
    parent->evaluate(xi_orig, lambda, objval, constr_orig,
                      gradObj, jac_orig_nz, jac_orig_row, jac_orig_colind, hess, dmode, info);

    // Subtract slacks from true constraints (not conditions)
    relaxed_constr.Submatrix(constr, parent->nCon, 1, 0, 0);

    int ind_1 = 0;
    int ind_2 = 0;

    for (int i = 0; i < parent_cond->num_cblocks; i++){
        if (parent_cond->cblocks[i].removed){
            for (int j = 0; j < parent_cond->cblocks[i].size; j++){
                relaxed_constr(ind_1 + j) = constr_orig(ind_1 + j);
            }
            ind_1 += parent_cond->cblocks[i].size;
        }
        else{
            for (int j = 0; j < parent_cond->cblocks[i].size; j++){
                relaxed_constr(ind_1 + j) = constr_orig(ind_1 + j) - slack(ind_2 + j);
            }
            ind_1 += parent_cond->cblocks[i].size;
            ind_2 += parent_cond->cblocks[i].size;
        }
    }

    dep_bounds.Submatrix(constr, parent->nVar - parent_cond->condensed_num_vars, 1, parent->nCon, 0);
    ind_1 = 0;
    ind_2 = 0;

    for (int i = 0; i < parent_cond->num_vblocks; i++){
        if (parent_cond->vblocks[i].dependent){
            for (int j = 0; j < parent_cond->vblocks[i].size; j++){
                dep_bounds(ind_1 + j) = xi_orig(ind_2 + j) - slack(parent_cond->num_true_cons + ind_1 + j);
            }
            ind_1 += parent_cond->vblocks[i].size;
            ind_2 += parent_cond->vblocks[i].size;
        }
        else{
            ind_2 += parent_cond->vblocks[i].size;
        }
    }

    // Evaluate objective: minimize slacks plus deviation from reference point
    if( dmode < 0 ){
        return;
    }

    *objval = 0.0;

    // Objective: 0.5 * Square sum of slack variables
    for (int i = 0; i < parent->nCon; i++){
        *objval += slack( i ) * slack( i );
    }
    *objval = 0.5 * (*objval);


    if(dmode > 0)
    {// compute objective gradient

        for(int i=0; i<parent->nVar; i++){
            gradObj(i) = 0;
        }

        for(int i=parent->nVar; i<nVar; i++){
            gradObj(i) = xi(i);
        }

        Sparse_Matrix jac_orig(parent->nCon, parent->nVar, jac_orig_colind[parent->nVar], jac_orig_nz, jac_orig_row, jac_orig_colind);
        Sparse_Matrix jac_restoration;
        build_restoration_jacobian(jac_orig, jac_restoration);

        for (int i = 0; i < jac_restoration.nnz; i++){
            jacNz[i] = jac_restoration.nz[i];
            jacIndRow[i] = jac_restoration.row[i];
        }
        for (int i = 0; i <= jac_restoration.n; i++){
            jacIndCol[i] = jac_restoration.colind[i];
        }

        jac_orig.nz = nullptr;
        jac_orig.row = nullptr;
        jac_orig.colind = nullptr;
        jac_orig.m = 0; jac_orig.n = 0; jac_restoration.nnz = 0;
    }

    *info = 0;
}


void feasibility_Problem::build_restoration_jacobian(const Sparse_Matrix &jac_orig, Sparse_Matrix &jac_restoration){

    int num_dep_vars = parent->nVar - parent_cond->condensed_num_vars;

    //Upper part of restoration jacobian: Slacks for true constraints
    double *constr_slack_nz = new double[parent_cond->num_true_cons];
    int *constr_slack_row = new int[parent_cond->num_true_cons];
    int *constr_slack_colind = new int[parent->nCon + 1];
    int ind_1 = 0;
    int ind_2 = 0;
    constr_slack_colind[0] = 0;

    for (int i = 0; i < parent_cond->num_cblocks; i++){
        if (parent_cond->cblocks[i].removed){
            ind_2 += parent_cond->cblocks[i].size;
        }
        else{
            for (int j = 0; j < parent_cond->cblocks[i].size; j++){
                constr_slack_nz[ind_1 + j] = -1.0;
                constr_slack_row[ind_1 + j] = ind_2 + j;
                constr_slack_colind[ind_1 + j + 1] = ind_1 + j + 1;
            }
            ind_1 += parent_cond->cblocks[i].size;
            ind_2 += parent_cond->cblocks[i].size;
        }
    }

    for (int i = parent_cond->num_true_cons; i < parent->nCon; i++){
        constr_slack_colind[i + 1] = parent_cond->num_true_cons;
    }

    Sparse_Matrix constr_slack(parent->nCon, parent->nCon, parent_cond->num_true_cons, constr_slack_nz, constr_slack_row, constr_slack_colind);

    //Lower part of restoration jacobian: Dependent variable bounds, relaxed with slack variables
    double *dep_bounds_nz = new double[num_dep_vars * 2];
    int *dep_bounds_row = new int[num_dep_vars * 2];
    int *dep_bounds_colind = new int[parent->nVar + parent->nCon + 1];
    ind_1 = 0;
    ind_2 = 0;
    dep_bounds_colind[0] = 0;

    //Left part: Derivatives with respect to the dependent variables
    for (int i = 0; i < parent_cond->num_vblocks; i++){
        if (parent_cond->vblocks[i].dependent){
            for (int j = 0; j < parent_cond->vblocks[i].size; j++){
                dep_bounds_nz[ind_1 + j] = 1.;
                dep_bounds_row[ind_1 + j] = ind_1 + j;
                dep_bounds_colind[ind_2 + j + 1] = ind_1 + j + 1;
            }
            ind_1 += parent_cond->vblocks[i].size;
            ind_2 += parent_cond->vblocks[i].size;
        }
        else{
            for (int j = 0; j < parent_cond->vblocks[i].size; j++){
                dep_bounds_colind[ind_2 + j + 1] = ind_1;
            }
            ind_2 += parent_cond->vblocks[i].size;
        }
    }

    //Middle part: No nonzero entries, slack variables already used up for true constraints

    for (int i = 0; i < parent_cond->num_true_cons; i++){
        dep_bounds_colind[ind_2 + i + 1] = ind_1;
    }
    ind_2 += parent_cond->num_true_cons;

    //Right part: Derivatives with respect to the remaining slack variables
    for (int i = 0; i < num_dep_vars; i++){
        dep_bounds_nz[ind_1 + i] = -1.;
        dep_bounds_row[ind_1 + i] = i;
        dep_bounds_colind[ind_2 + i + 1] = ind_1 + i + 1;
    }

    Sparse_Matrix dep_bounds(num_dep_vars, parent->nVar + parent->nCon, 2*num_dep_vars, dep_bounds_nz, dep_bounds_row, dep_bounds_colind);

    std::vector<Sparse_Matrix> mats(2);
    mats[0] = std::move(jac_orig);
    mats[1] = std::move(constr_slack);

    mats[0] = horzcat(mats);
    mats[1] = std::move(dep_bounds);

    jac_restoration = vertcat(mats);
}


void feasibility_Problem::recover_multipliers(const Matrix &lambda_rest, Matrix &lambda_orig){
    int ind_1 = 0;
    int ind_2 = parent->nVar + 2 * parent->nCon;

    for (int i = 0; i < parent_cond->num_vblocks; i++){
        if (parent_cond->vblocks[i].dependent){
            for (int j = 0; j < parent_cond->vblocks[i].size; j++){
                lambda_orig(ind_1 + j) = lambda_rest(ind_2 + j);
            }
            ind_1 += parent_cond->vblocks[i].size;
            ind_2 += parent_cond->vblocks[i].size;
        }
        else{
            for (int j = 0; j < parent_cond->vblocks[i].size; j++){
                lambda_orig(ind_1 + j) = lambda_rest(ind_1 + j);
            }
            ind_1 += parent_cond->vblocks[i].size;
        }
    }

    for (int i = parent->nVar; i < parent->nVar + parent->nCon; i++){
        lambda_orig(i) = lambda_rest(parent->nCon + i);
    }
}
*/



TC_restoration_Problem::TC_restoration_Problem(Problemspec *parent_Problem, Condenser *parent_CND, const Matrix &xi_Reference,
                                                 double param_rho, double param_zeta):
        parent_cond(parent_CND), xi_ref(xi_Reference), rho(param_rho), zeta(param_zeta){
    
    parent = parent_Problem;
    // one slack variable for each true (not used for condensing) constraint
    nVar = parent->nVar + parent_cond->num_true_cons;
    nCon = parent->nCon;
    nnz = parent->nnz + parent_cond->num_true_cons;

    // Block structure: One additional block for every slack variable
    nBlocks = parent->nBlocks + parent_cond->num_true_cons;
    blockIdx = new int[nBlocks + 1];
    for(int i = 0; i<parent->nBlocks + 1; i++){
        blockIdx[i] = parent->blockIdx[i];
    }
    for(int i = parent->nBlocks + 1; i<nBlocks+1; i++){
        blockIdx[i] = blockIdx[i-1]+1;
    }

    //Set bounds, no bounds for dependent variables
    objLo = 0.0;
    objUp = 1.0e20;

    //Bounds for original variables, bounds for slack variables, bounds for original constraints and conditions, bounds for dependent variables as constraints
    lb_var.Dimension(nVar).Initialize(-std::numeric_limits<double>::infinity());
    ub_var.Dimension(nVar).Initialize(std::numeric_limits<double>::infinity());

    //Variable bounds
    for (int i = 0; i < parent->nVar; i++){
        lb_var(i) = parent->lb_var(i);
        ub_var(i) = parent->ub_var(i);
    }

    //No bounds for slack variables

    //Bounds for constraints and conditions
    lb_con.Dimension(nCon);//.Initialize(-std::numeric_limits<double>::infinity());
    ub_con.Dimension(nCon);//.Initialize(std::numeric_limits<double>::infinity());

    for (int i = 0; i < parent->nCon; i++){
        lb_con(i) = parent->lb_con(i);
        ub_con(i) = parent->ub_con(i);
    }

}

TC_restoration_Problem::~TC_restoration_Problem(){
    delete[] jac_orig_nz;
    delete[] jac_orig_row;
    delete[] jac_orig_colind;
    delete[] blockIdx;
}


void TC_restoration_Problem::update_xi_ref(const Matrix &xiReference){
    xi_ref = xiReference;
}

void TC_restoration_Problem::initialize(Matrix &xi, Matrix &lambda, double *jacNz, int *jacIndRow, int *jacIndCol){

    int info;
    double objval;

    xi_parent.Submatrix( xi, parent->nVar, 1, 0, 0 );
    slack.Submatrix( xi, parent_cond->num_true_cons, 1, parent->nVar, 0 );

    //Allocate the sparse jacobian of the parent problem
    jac_orig_nz = new double[parent->nnz];
    jac_orig_row = new int[parent->nnz];
    jac_orig_colind = new int[parent->nVar + 1];

    // Call initialize of the parent problem. There, the sparse Jacobian is intialized
    parent->initialize(xi_parent, lambda, jac_orig_nz, jac_orig_row, jac_orig_colind);

    //Initialize restoration jacobian: Slacks only for true constraints
    for (int i = 0; i < parent->nnz; i++){
        jacIndRow[i] = jac_orig_row[i];
    }
    for (int i = 0; i <= parent->nVar; i++){
        jacIndCol[i] = jac_orig_colind[i];
    }

    //Add slack part
    int ind_1 = parent->nnz;
    int ind_2 = 0;
    int ind_3 = parent->nVar;
    
    //B: Jacobian of true constraints (not (continuity-) conditions)
    //C: Jacobian of (continuity-) conditions
    //Create restoration Jacobian from B and C, adding slacks only for B:
    //[B I]
    //[C 0]
    //Matrix row may not be sorted according to B anc C, 
    //so iterate over constraint blocks corresponding to B
    for (int i = 0; i < parent_cond->num_cblocks; i++){
        if (!parent_cond->cblocks[i].removed){
            for (int j = 0; j < parent_cond->cblocks[i].size; j++){
                jacNz[ind_1 + j] = -1.0;
                jacIndRow[ind_1 + j] = ind_2 + j;
                jacIndCol[ind_3 + j + 1] = ind_1 + j + 1;
                //jacIndCol[ind_3 + 1 + j] = jacIndCol[ind_3 + j] + 1;
            }
            ind_1 += parent_cond->cblocks[i].size;
            ind_3 += parent_cond->cblocks[i].size;
        }
        ind_2 += parent_cond->cblocks[i].size;
    }


    // The reference point is the starting value for the restoration phase
    for(int i=0; i<parent->nVar; i++){
        xi_parent(i) = xi_ref(i);
    }

    // Initialize slack variables such that the constraints are feasible, allocate and use vector for original constraints
    constr_orig.Dimension(parent->nCon);
    parent->evaluate(xi_parent, &objval, constr_orig, &info);

    ind_1 = 0;
    ind_2 = 0;
    for (int i = 0; i < parent_cond->num_cblocks; i++){
        if (!parent_cond->cblocks[i].removed){
            for (int j = 0; j < parent_cond->cblocks[i].size; j++){
                if (constr_orig(ind_2 + j) < parent->lb_con(ind_2 + j)){
                    slack(ind_1 + j) = constr_orig(ind_2 + j) - parent->lb_con(ind_2 + j);
                }
                else if (constr_orig(ind_2 + j) > parent->ub_con(ind_2 + j)){
                    slack(ind_1 + j) = constr_orig(ind_2 + j) - parent->ub_con(ind_2 + j);
                }
                else{
                    slack(ind_1 + j) = 0.;
                }
            }
            ind_1 += parent_cond->cblocks[i].size;
        }
        ind_2 += parent_cond->cblocks[i].size;
    }


    // Set diagonal scaling matrix
    diagScale.Dimension(parent->nVar).Initialize(1.0);
    for(int i = 0; i < parent->nVar; i++){
        if(fabs(xi_ref(i)) > 1.0){
            diagScale(i) = 1.0/fabs(xi_ref(i));
        }
    }

    // Regularization factor zeta and rho \todo wie setzen?
    //zeta = 1.0e-3;
    //rho = 1.0e3;

    lambda.Initialize(0.0);
}



void TC_restoration_Problem::evaluate(
                                const Matrix &xi, const Matrix &lambda,
                                double *objval, Matrix &constr,
                                Matrix &gradObj, double *jacNz, int *jacIndRow, int *jacIndCol,
                                SymMatrix *hess, int dmode, int *info){

    double diff, regTerm;

    // The first nVar elements of the variable vector correspond to the variables of the original problem
    xi_parent.Submatrix( xi, parent->nVar, 1, 0, 0 );
    slack.Submatrix( xi, parent_cond->num_true_cons, 1, parent->nVar, 0 );

    // Evaluate constraints of the original problem
    parent->evaluate(xi_parent, lambda, objval, constr_orig,
                      gradObj, jacNz, jacIndRow, jacIndCol, hess, dmode, info);

    // Subtract slacks from true constraints (not conditions)
    int ind_1 = 0;
    int ind_2 = 0;

    for (int i = 0; i < parent_cond->num_cblocks; i++){
        if (parent_cond->cblocks[i].removed){
            for (int j = 0; j < parent_cond->cblocks[i].size; j++){
                constr(ind_1 + j) = constr_orig(ind_1 + j);
            }
            ind_1 += parent_cond->cblocks[i].size;
        }
        else{
            for (int j = 0; j < parent_cond->cblocks[i].size; j++){
                constr(ind_1 + j) = constr_orig(ind_1 + j) - slack(ind_2 + j);
            }
            ind_1 += parent_cond->cblocks[i].size;
            ind_2 += parent_cond->cblocks[i].size;
        }
    }


    // Evaluate objective: minimize slacks plus deviation from reference point
    if( dmode < 0 ){
        return;
    }

    *objval = 0.0;

    // First part: sum of slack variables
    for (int i = 0; i < parent_cond->num_true_cons; i++){
        *objval += slack( i ) * slack( i );
    }
    *objval = 0.5 * rho * (*objval);

    // Second part: regularization term
    regTerm = 0.0;
    for(int i = 0; i < parent->nVar; i++){
        diff = xi_parent(i) - xi_ref(i);
        regTerm += diagScale(i) * diagScale(i) * diff * diff;
    }
    regTerm = 0.5 * zeta * regTerm;
    *objval += regTerm;

    if(dmode > 0){
        // gradient w.r.t. xi (regularization term)
        for(int i=0; i<parent->nVar; i++){
            gradObj(i) = zeta * diagScale(i) * diagScale(i) * (xi_parent(i) - xi_ref(i));
        }

        // gradient w.r.t. slack variables
        for(int i=parent->nVar; i<nVar; i++){
            gradObj(i) = rho * xi(i);
        }

    }

    *info = 0;
}


void TC_restoration_Problem::reduceConstrVio(Matrix &xi, int *info){
    xi_parent.Submatrix(xi, parent->nVar, 1, 0, 0);
    parent->reduceConstrVio(xi_parent, info);
    return;
}

/*
void TC_restoration_Problem::recover_multipliers(const Matrix &lambda_rest, Matrix &lambda_orig){

    for (int i = 0; i < parent->nVar; i++){
        lambda_orig(i) = lambda_rest(i);
    }

    int ind_1 = parent->nVar;
    int ind_2 = parent->nVar + parent_cond->num_true_cons;

    for (int i = 0; i < parent->nCon; i++){
        lambda_orig(ind_1 + i) = lambda_rest(ind_2 + i);
    }
    return;
}
*/

void TC_restoration_Problem::recover_xi(const Matrix &xi_rest, Matrix &xi_orig){
    for (int i = 0; i < parent->nVar; i++){
        xi_orig(i) = xi_rest(i);
    }
    return;
}

void TC_restoration_Problem::recover_lambda(const Matrix &lambda_rest, Matrix &lambda_orig){
    for (int i = 0; i < parent->nVar; i++){
        lambda_orig(i) = lambda_rest(i);
    }
    int ind_1 = parent->nVar;
    int ind_2 = parent->nVar + parent_cond->num_true_cons;

    for (int i = 0; i < parent->nCon; i++){
        lambda_orig(ind_1 + i) = lambda_rest(ind_2 + i);
    }
    return;
}



holding_Condenser* create_restoration_Condenser(Condenser *parent, int DEP_BOUNDS){
    int N_vblocks = parent->num_vblocks + parent->num_true_cons;
    int N_cblocks = parent->num_cblocks;
    int N_hessblocks = parent->num_hessblocks + parent->num_true_cons;
    int N_targets = parent->num_targets;

	std::unique_ptr<vblock[]> rest_vblocks = std::make_unique<vblock[]>(N_vblocks);
    std::unique_ptr<cblock[]> rest_cblocks = std::make_unique<cblock[]>(N_cblocks);
	std::unique_ptr<int[]> rest_hess_block_sizes = std::make_unique<int[]>(N_hessblocks);
	std::unique_ptr<condensing_target[]> rest_targets = std::make_unique<condensing_target[]>(N_targets);

    for (int i = 0; i < parent->num_vblocks; i++){
        rest_vblocks[i] = parent->vblocks[i];
    }
    for (int i = parent->num_vblocks; i < N_vblocks; i++){
        rest_vblocks[i] = vblock(1, false);
    }

    for (int i = 0; i < parent->num_cblocks; i++){
        rest_cblocks[i] = parent->cblocks[i];
    }

    for (int i = 0; i<parent->num_hessblocks; i++){
        rest_hess_block_sizes[i] = parent->hess_block_sizes[i];
    }
    for (int i = parent->num_hessblocks; i<N_hessblocks; i++){
        rest_hess_block_sizes[i] = 1;
    }

    for (int i = 0; i<parent->num_targets; i++){
        rest_targets[i] = parent->targets[i];
    }

    return new holding_Condenser(std::move(rest_vblocks), N_vblocks, std::move(rest_cblocks), N_cblocks, std::move(rest_hess_block_sizes), N_hessblocks, std::move(rest_targets), N_targets, DEP_BOUNDS);
}





/*
void TC_restoration_Problem::recover_multipliers(const Matrix &lambda_rest, Matrix &lambda_orig, double &lambda_step_norm){

    lambda_step_norm = 0.;

    for (int i = 0; i < parent->nVar; i++){
        if (fabs(lambda_rest(i) - lambda_orig(i)) > lambda_step_norm){
            lambda_step_norm = fabs(lambda_rest(i) - lambda_orig(i));
        }
        lambda_orig(i) = lambda_rest(i);
    }

    int ind_1 = parent->nVar;
    int ind_2 = parent->nVar + parent_cond->num_true_cons;

    for (int i = 0; i < parent->nCon; i++){
        if (fabs(lambda_rest(ind_2 + i) - lambda_orig(ind_1 + i)) < lambda_step_norm){
            lambda_step_norm = fabs(lambda_rest(ind_2 + i) - lambda_orig(ind_1 + i));
        }
        lambda_orig(ind_1 + i) = lambda_rest(ind_2 + i);
    }

    return;
}
*/
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TC_feasibility_Problem::TC_feasibility_Problem(Problemspec *parent_Problem, Condenser *parent_CND): parent(parent_Problem), parent_cond(parent_CND){

    // one slack variable for each true (not used for condensing) constraint
    nVar = parent->nVar + parent_cond->num_true_cons;
    nCon = parent->nCon;
    nnz = parent->nnz + parent_cond->num_true_cons;

    // Block structure: One additional block for every slack variable
    nBlocks = parent->nBlocks + parent_cond->num_true_cons;
    blockIdx = new int[nBlocks + 1];
    for(int i = 0; i<parent->nBlocks + 1; i++){
        blockIdx[i] = parent->blockIdx[i];
    }
    for(int i = parent->nBlocks + 1; i<nBlocks+1; i++){
        blockIdx[i] = blockIdx[i-1]+1;
    }

    //Set bounds, no bounds for dependent variables
    objLo = 0.0;
    objUp = 1.0e20;

    //Bounds for original variables, bounds for slack variables, bounds for original constraints and conditions, bounds for dependent variables as constraints
    lb_var.Dimension(nVar).Initialize(-std::numeric_limits<double>::infinity());
    ub_var.Dimension(nVar).Initialize(std::numeric_limits<double>::infinity());

    //Variable bounds
    for (int i = 0; i < parent->nVar; i++){
        lb_var(i) = parent->lb_var(i);
        ub_var(i) = parent->ub_var(i);
    }

    //No bounds for slack variables

    //Bounds for constraints and conditions
    lb_con.Dimension(nCon).Initialize(-std::numeric_limits<double>::infinity());
    ub_con.Dimension(nCon).Initialize(std::numeric_limits<double>::infinity());

    for (int i = 0; i < parent->nCon; i++){
        lb_con(i) = parent->lb_con(i);
        ub_con(i) = parent->ub_con(i);
    }
}


TC_feasibility_Problem::~TC_feasibility_Problem(){
    delete[] jac_orig_nz;
    delete[] jac_orig_row;
    delete[] jac_orig_colind;
    delete[] blockIdx;
}


void TC_feasibility_Problem::initialize(Matrix &xi, Matrix &lambda, double *jacNz, int *jacIndRow, int *jacIndCol){

    int info;
    double objval;

    xi_parent.Submatrix( xi, parent->nVar, 1, 0, 0 );
    slack.Submatrix( xi, parent_cond->num_true_cons, 1, parent->nVar, 0 );

    //Allocate the sparse jacobian of the parent problem
    jac_orig_nz = new double[parent->nnz];
    jac_orig_row = new int[parent->nnz];
    jac_orig_colind = new int[parent->nVar + 1];

    // Call initialize of the parent problem. There, the sparse Jacobian is intialized
    parent->initialize(xi_parent, lambda, jac_orig_nz, jac_orig_row, jac_orig_colind);

    //Initialize restoration jacobian: Slacks only for true constraints
    for (int i = 0; i < parent->nnz; i++){
        jacIndRow[i] = jac_orig_row[i];
    }
    for (int i = 0; i <= parent->nVar; i++){
        jacIndCol[i] = jac_orig_colind[i];
    }

    //Add slack part
    int ind_1 = parent->nnz;
    int ind_2 = 0;
    int ind_3 = parent->nVar;

    for (int i = 0; i < parent_cond->num_cblocks; i++){
        if (!parent_cond->cblocks[i].removed){
            for (int j = 0; j < parent_cond->cblocks[i].size; j++){
                jacNz[ind_1 + j] = -1.0;
                jacIndRow[ind_1 + j] = ind_2 + j;
                jacIndCol[ind_3 + j + 1] = ind_1 + j + 1;
                //jacIndCol[ind_3 + 1 + j] = jacIndCol[ind_3 + j] + 1;
            }
            ind_1 += parent_cond->cblocks[i].size;
            ind_3 += parent_cond->cblocks[i].size;
        }
        ind_2 += parent_cond->cblocks[i].size;
    }


    // Initialize slack variables such that the constraints are feasible, allocate and use vector for original constraints
    constr_orig.Dimension(parent->nCon);
    parent->evaluate(xi_parent, &objval, constr_orig, &info);

    ind_1 = 0;
    ind_2 = 0;
    for (int i = 0; i < parent_cond->num_cblocks; i++){
        if (!parent_cond->cblocks[i].removed){
            for (int j = 0; j < parent_cond->cblocks[i].size; j++){
                if (constr_orig(ind_2 + j) < parent->lb_con(ind_2 + j)){
                    slack(ind_1 + j) = constr_orig(ind_2 + j) - parent->lb_con(ind_2 + j);
                }
                else if (constr_orig(ind_2 + j) > parent->ub_con(ind_2 + j)){
                    slack(ind_1 + j) = constr_orig(ind_2 + j) - parent->ub_con(ind_2 + j);
                }
                else{
                    slack(ind_1 + j) = 0.;
                }
            }
            ind_1 += parent_cond->cblocks[i].size;
        }
        ind_2 += parent_cond->cblocks[i].size;
    }

    lambda.Initialize(0.0);
}



void TC_feasibility_Problem::evaluate(
                                const Matrix &xi, const Matrix &lambda,
                                double *objval, Matrix &constr,
                                Matrix &gradObj, double *jacNz, int *jacIndRow, int *jacIndCol,
                                SymMatrix *hess, int dmode, int *info){

    // The first nVar elements of the variable vector correspond to the variables of the original problem
    xi_parent.Submatrix( xi, parent->nVar, 1, 0, 0 );
    slack.Submatrix( xi, parent_cond->num_true_cons, 1, parent->nVar, 0 );

    for (int i = 0; i < parent->nVar; i++){
        if (std::isnan(xi_parent(i))){
            std::cout << "Submatrix value is nan!\n" << "Index = " << i << "\nMatrix value = " << xi(i) << "\n";
            throw std::invalid_argument("Submatrix value is nan!");
        }
    }

    // Evaluate constraints of the original problem
    parent->evaluate(xi_parent, lambda, objval, constr_orig,
                      gradObj, jacNz, jacIndRow, jacIndCol, hess, dmode, info);

    // Subtract slacks from true constraints (not conditions)
    int ind_1 = 0;
    int ind_2 = 0;

    for (int i = 0; i < parent_cond->num_cblocks; i++){
        if (parent_cond->cblocks[i].removed){
            for (int j = 0; j < parent_cond->cblocks[i].size; j++){
                constr(ind_1 + j) = constr_orig(ind_1 + j);
            }
            ind_1 += parent_cond->cblocks[i].size;
        }
        else{
            for (int j = 0; j < parent_cond->cblocks[i].size; j++){
                constr(ind_1 + j) = constr_orig(ind_1 + j) - slack(ind_2 + j);
            }
            ind_1 += parent_cond->cblocks[i].size;
            ind_2 += parent_cond->cblocks[i].size;
        }
    }


    // Evaluate objective: minimize slacks plus deviation from reference point
    if( dmode < 0 ){
        return;
    }

    *objval = 0.0;

    // Objective value: 0.5 * || s ||_2
    for (int i = 0; i < parent_cond->num_true_cons; i++){
        *objval += slack( i ) * slack( i );
    }
    *objval = 0.5 * (*objval);


    if(dmode > 0){

        for(int i=0; i<parent->nVar; i++){
            gradObj(i) = 0.;
        }

        // gradient w.r.t. slack variables
        for(int i=parent->nVar; i<nVar; i++){
            gradObj(i) = xi(i);
        }

    }

    *info = 0;
}


void TC_feasibility_Problem::reduceConstrVio(Matrix &xi, int *info){

    xi_parent.Submatrix(xi, parent->nVar, 1, 0, 0);
    parent->reduceConstrVio(xi_parent, info);

    return;
}













} // namespace blockSQP














