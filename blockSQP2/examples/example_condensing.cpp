/*
 * blockSQP 2 -- Condensing, convexification strategies, scaling heuristics and more
 *               for blockSQP, the nonlinear programming solver by Dennis Janka.
 * Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
 * 
 * Licensed under the zlib license. See LICENSE for more details.
 */

/**
 * \file example_condensing.hpp
 * \author Reinhold Wittmann
 * \date 2023-2025
 *
 * Example demonstrating the use of QP condensing.
 */

#include <blockSQP2/condensing.hpp>
#include "qpOASES.hpp"
#include <iostream>

blockSQP2::SymMatrix identity(int n){
    blockSQP2::SymMatrix S(n);
    S.Initialize(0.);
    for (int i = 0; i < n; i++){
        S(i,i) = 1.;
    }
    return S;
}


void convertHessian(double eps, blockSQP2::SymMatrix *&hess_, int nBlocks, int nVar,
                             double *&hessNz_, int *&hessIndRow_, int *&hessIndCol_, int *&hessIndLo_ ){
    int iBlock, count, colCountTotal, rowOffset, i, j;
    int nnz, nCols, nRows;

    // 1) count nonzero elements
    nnz = 0;
    for( iBlock=0; iBlock<nBlocks; iBlock++ )
        for( i=0; i<hess_[iBlock].m; i++ )
            for( j=i; j<hess_[iBlock].m; j++ )
                if( fabs(hess_[iBlock]( i,j )) > eps )
                {
                    nnz++;
                    if( i != j )// off-diagonal elements count twice
                        nnz++;
                }

    if( hessNz_ != NULL ) delete[] hessNz_;
    if( hessIndRow_ != NULL ) delete[] hessIndRow_;

    hessNz_ = new double[nnz];
    hessIndRow_ = new int[nnz + (nVar+1) + nVar];
    hessIndCol_ = hessIndRow_ + nnz;
    hessIndLo_ = hessIndCol_ + (nVar+1);


    // 2) store matrix entries columnwise in hessNz
    count = 0; // runs over all nonzero elements
    colCountTotal = 0; // keep track of position in large matrix
    rowOffset = 0;
    for( iBlock=0; iBlock<nBlocks; iBlock++ )
    {
        nCols = hess_[iBlock].m;
        nRows = hess_[iBlock].m;

        for( i=0; i<nCols; i++ )
        {
            // column 'colCountTotal' starts at element 'count'
            hessIndCol_[colCountTotal] = count;

            for( j=0; j<nRows; j++ ){
                if( (hess_[iBlock]( i,j ) > eps) || (-hess_[iBlock]( i,j ) > eps) )
                {
                    hessNz_[count] = hess_[iBlock]( i, j );
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


int main(){
    
    //Layout information
    blockSQP2::vblock *vblocks = new blockSQP2::vblock[7];
    vblocks[0] = blockSQP2::vblock(1,false);
    
    vblocks[1] = blockSQP2::vblock(2,true);
    vblocks[2] = blockSQP2::vblock(1,false);
    
    vblocks[3] = blockSQP2::vblock(2,true);
    vblocks[4] = blockSQP2::vblock(1,false);
    
    vblocks[5] = blockSQP2::vblock(2,true);
    vblocks[6] = blockSQP2::vblock(1,false);
    
    
    blockSQP2::cblock *cblocks = new blockSQP2::cblock[4];
    cblocks[0] = blockSQP2::cblock(2);
    cblocks[1] = blockSQP2::cblock(2);
    cblocks[2] = blockSQP2::cblock(2);
    cblocks[3] = blockSQP2::cblock(1);
    
    int *hsizes = new int[4]{1,3,3,3};
    
    blockSQP2::condensing_target *targets = new blockSQP2::condensing_target[1];
    targets[0] = blockSQP2::condensing_target(3,0,7,0,3);
    
    blockSQP2::Condenser *cond = new blockSQP2::Condenser(vblocks, 7, cblocks, 4, hsizes, 4, targets, 1);
    
    
    cond->print_info();
    
    
    //Uncondensed Problem
    std::unique_ptr<double[]> NZ = std::unique_ptr<double[]>(new double[26]{-1,-2,1,1,-2,1,1,-1,1,-1,-2,1,1,-2,1,1,-1,1,-1,-2,1,1,1,1,1,1});
    std::unique_ptr<int[]> ROW = std::unique_ptr<int[]>(new int[26]{0,1,6,0,2,6,1,3,6,2,3,6,2,4,6,3,5,6,4,5,6,4,6,5,6,6});
    std::unique_ptr<int[]> COLIND = std::unique_ptr<int[]>(new int[11]{0,3,6,9,12,15,18,21,23,25,26});
    
    blockSQP2::Sparse_Matrix con_jac(7, 10, std::move(NZ), std::move(ROW), std::move(COLIND));
    
    std::cout << "A =\n" << con_jac.dense() << "\n";
    
    
    //full_block = 0.25 * ([1,-1,1]*[1,-1,1]^T + [1,1,0]*[1,1,0]^T + [-1,1,2]*[-1,1,2]^T)
    blockSQP2::SymMatrix full_block(3);
    full_block(0,0) = 0.75; full_block(1,0) = -0.25; full_block(2,0) = -0.25; full_block(1,1) = 0.75; full_block(2,1) = 0.25; full_block(2,2) = 1.25;
    
    
    blockSQP2::SymMatrix *hess = new blockSQP2::SymMatrix[4];
    hess[0] = identity(1);
    //hess[1] = identity(3);
    //hess[2] = identity(3);
    //hess[3] = identity(3);
    hess[1] = full_block;
    hess[2] = full_block;
    hess[3] = full_block;
    
    
    //std::cout << "The quadratic objective is xi^T H xi, with H block-structured with 4 identity-blocks of sizes 1,3,3,3\n";
	std::cout << "The quadratic objective is xi^T H xi, with H block-structured with 4 blocks, [1] +\n" << full_block << "x3\n";
    
    blockSQP2::Matrix grad_obj(10,1);
    grad_obj.Initialize(1.);
    
    blockSQP2::Matrix lb_var(10,1);
    blockSQP2::Matrix ub_var(10,1);
    blockSQP2::Matrix lb_con(7,1);
    blockSQP2::Matrix ub_con(7,1);
    
    //default: +-0.3
    lb_var.Initialize(-0.3);
    ub_var.Initialize(0.3);
    lb_con.Initialize(0.1);
    ub_con.Initialize(0.1);
    lb_con(6) = -1.9;
    ub_con(6) = 1.9;
    
    std::cout << "c =\n[0.1\n0.1\n0.1\n0.1\n0.1\n0.1]\n";
    
    //Solve uncondensed problem
    qpOASES::SQProblem* qp;
    qpOASES::returnValue ret;
    
    qpOASES::Matrix *A_qp;
    qpOASES::SymmetricMatrix *H;
    
    double *hess_nz = nullptr;
    int *hess_row = nullptr;
    int *hess_colind = nullptr;
    int *hess_loind = nullptr;
    
    qp = new qpOASES::SQProblemSchur( con_jac.n, con_jac.m, qpOASES::HST_UNKNOWN, 50 );
    
    A_qp = new qpOASES::SparseMatrix(con_jac.m, con_jac.n,
                con_jac.row.get(), con_jac.colind.get(), con_jac.nz.get());
    
    convertHessian(1.0e-15, hess, 4, con_jac.n, hess_nz, hess_row, hess_colind, hess_loind);
    std::cout << "converted Hessians\n";
    
    
    H = new qpOASES::SymSparseMat(con_jac.n, con_jac.n, hess_row, hess_colind, hess_nz);
    dynamic_cast<qpOASES::SymSparseMat*>(H)->createDiagInfo();
    
    double *g = grad_obj.array;
    double *lb = lb_var.array;
    double *ub = ub_var.array;
    double *lbA = lb_con.array;
    double *ubA = ub_con.array;
    double cpu_time = 10000;
    int max_it = 10000;
    
    qpOASES::Options opts;
    opts.enableInertiaCorrection = qpOASES::BT_FALSE;
    opts.enableEqualities = qpOASES::BT_TRUE;
    opts.initialStatusBounds = qpOASES::ST_INACTIVE;
    opts.printLevel = qpOASES::PL_LOW; //PL_LOW, PL_HIGH, PL_MEDIUM, PL_None
    opts.numRefinementSteps = 2;
    opts.epsLITests =  2.2204e-08;
    qp->setOptions( opts );
    
    ret = qp->init(H, g, A_qp, lb, ub, lbA, ubA, max_it, &cpu_time);
    
    
    blockSQP2::Matrix xi(10);
    blockSQP2::Matrix lambda(17);
    qp->getPrimalSolution(xi.array);
    qp->getDualSolution(lambda.array);
    
    
    delete qp;
    delete A_qp;
    delete H;
    
    
    //Condense the QP
    blockSQP2::SymMatrix *condensed_hess = new blockSQP2::SymMatrix[cond->condensed_num_hessblocks];
    blockSQP2::Sparse_Matrix condensed_Jacobian;
    blockSQP2::Matrix condensed_h;
    blockSQP2::Matrix condensed_lb_var;
    blockSQP2::Matrix condensed_ub_var;
    blockSQP2::Matrix condensed_lb_con;
    blockSQP2::Matrix condensed_ub_con;
    
    cond->full_condense(grad_obj, con_jac, hess, lb_var, ub_var, lb_con, ub_con,
        condensed_h, condensed_Jacobian, condensed_hess, condensed_lb_var, condensed_ub_var, condensed_lb_con, condensed_ub_con
    );
    
    
    std::cout << "Condensed hess =\n" << condensed_hess[0] << "\n";
    std::cout << "Condensed jacobian =\n" << condensed_Jacobian.dense() << "\n";
    std::cout << "Condensed linear form h =\n" << condensed_h << "\n";
    std::cout << "Condensed lb_var =\n" << condensed_lb_var << "\n\ncondensed_ub_var =\n" << condensed_ub_var << "\n";
    std::cout << "Condensed lb_con =\n" << condensed_lb_con << "\n\ncondensed_ub_con =\n" << condensed_ub_con << "\n";
    
    
    //Solve the condensed QP
    qpOASES::SQProblem* qp_cond;
    qpOASES::returnValue ret_cond;
    
    qpOASES::Matrix *A_qp_cond;
    qpOASES::SymmetricMatrix *H_cond;
    
    double *hess_nz_cond = nullptr;
    int *hess_row_cond = nullptr;
    int *hess_colind_cond = nullptr;
    int *hess_loind_cond = nullptr;
    
    qp_cond = new qpOASES::SQProblemSchur( condensed_Jacobian.n, condensed_Jacobian.m, qpOASES::HST_UNKNOWN, 50 );
    
    A_qp_cond = new qpOASES::SparseMatrix(condensed_Jacobian.m, condensed_Jacobian.n,
                condensed_Jacobian.row.get(), condensed_Jacobian.colind.get(), condensed_Jacobian.nz.get());
    
    convertHessian(1.0e-15, condensed_hess, 1, condensed_Jacobian.n, hess_nz_cond, hess_row_cond, hess_colind_cond, hess_loind_cond);
    std::cout << "converted Hessians\n";
    
    
    H_cond = new qpOASES::SymSparseMat(condensed_Jacobian.n, condensed_Jacobian.n, hess_row_cond, hess_colind_cond, hess_nz_cond);
    dynamic_cast<qpOASES::SymSparseMat*>(H_cond)->createDiagInfo();
    
    double *g_cond = condensed_h.array;
    double *lb_cond = condensed_lb_var.array;
    double *ub_cond = condensed_ub_var.array;
    double *lbA_cond = condensed_lb_con.array;
    double *ubA_cond = condensed_ub_con.array;
    double cpu_time_cond = 10000;
    int max_it_cond = 10000;
    
    qp_cond->setOptions( opts );
    
    ret_cond = qp_cond->init(H_cond, g_cond, A_qp_cond, lb_cond, ub_cond, lbA_cond, ubA_cond, max_it_cond, &cpu_time_cond);
    std::cout << "Solver of condensed QP returned, ret is " << ret_cond << "\n";
    
    blockSQP2::Matrix xi_cond(4);
    blockSQP2::Matrix lambda_cond(11);
    qp_cond->getPrimalSolution(xi_cond.array);
    qp_cond->getDualSolution(lambda_cond.array);
    
    std::cout << "xi_cond=\n" << xi_cond;
    std::cout << "lambda_cond=\n" << lambda_cond;
    
    
    blockSQP2::Matrix xi_rest;
    blockSQP2::Matrix lambda_rest;

    cond->recover_var_mult(xi_cond, lambda_cond, xi_rest, lambda_rest);

    std::cout << "Primal solution of uncondensed QP is\n" << xi << "\n";
    std::cout << "Dual solution of uncondensed QP is\n" << lambda << "\n";
    
    std::cout << "Primal solution of condensed QP after restoration is\n" << xi_rest << "\n";
    std::cout << "Dual solution of condensed QP after restoration is\n" << lambda_rest << "\n";
    
    
    delete qp_cond;
    delete A_qp_cond;
    delete H_cond;
    
    delete[] vblocks;
    delete[] cblocks;
    delete[] hsizes;
    delete[] targets;
    delete[] hess;
    delete[] condensed_hess;
    
    delete[] hess_nz;
    delete[] hess_row;
    
    delete[] hess_nz_cond;
    delete[] hess_row_cond;
    
    delete cond;
    return 0;
}

