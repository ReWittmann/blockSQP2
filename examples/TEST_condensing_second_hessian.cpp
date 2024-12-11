#include "blocksqp_condensing.hpp"
#include "qpOASES.hpp"
#include <iostream>

blockSQP::SymMatrix identity(int n){
    blockSQP::SymMatrix S(n);
    S.Initialize(0.);
    for (int i = 0; i < n; i++){
        S(i,i) = 1.;
    }
    return S;
}

void convertHessian(double eps, blockSQP::SymMatrix *&hess_, int nBlocks, int nVar,
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

    //Convex combination factor
    double t = 0.5;


    //Layout information
    blockSQP::vblock *vblocks = new blockSQP::vblock[7];
    vblocks[0] = blockSQP::vblock(1,false);

    vblocks[1] = blockSQP::vblock(2,true);
    vblocks[2] = blockSQP::vblock(1,false);

    vblocks[3] = blockSQP::vblock(2,true);
    vblocks[4] = blockSQP::vblock(1,false);

    vblocks[5] = blockSQP::vblock(2,true);
    vblocks[6] = blockSQP::vblock(1,false);


    blockSQP::cblock *cblocks = new blockSQP::cblock[4];
    cblocks[0] = blockSQP::cblock(2);
    cblocks[1] = blockSQP::cblock(2);
    cblocks[2] = blockSQP::cblock(2);
    cblocks[3] = blockSQP::cblock(1);

    int *hsizes = new int[4]{1,3,3,3};

    blockSQP::condensing_target *targets = new blockSQP::condensing_target[1];
    targets[0] = blockSQP::condensing_target(3,0,7,0,3);

    blockSQP::Condenser *C = new blockSQP::Condenser(vblocks, 7, cblocks, 4, hsizes, 4, targets, 1);


    C->print_debug();


    //Uncondensed Problem
    double *NZ = new double[26]{-1,-2,1,1,-2,1,1,-1,1,-1,-2,1,1,-2,1,1,-1,1,-1,-2,1,1,1,1,1,1};
    int *ROW = new int[26]{0,1,6,0,2,6,1,3,6,2,3,6,2,4,6,3,5,6,4,5,6,4,6,5,6,6};
    int *COLIND = new int[11]{0,3,6,9,12,15,18,21,23,25,26};

    blockSQP::Sparse_Matrix con_jac(7,10,26, NZ, ROW, COLIND);

    std::cout << "A =\n" << con_jac.dense() << "\n";


    //full_block = 0.25 * ([1,-1,1]*[1,-1,1]^T + [1,1,0]*[1,1,0]^T + [-1,1,2]*[-1,1,2]^T)
    blockSQP::SymMatrix full_block(3);
    full_block(0,0) = 0.75; full_block(1,0) = -0.25; full_block(2,0) = -0.25; full_block(1,1) = 0.75; full_block(2,1) = 0.25; full_block(2,2) = 1.25;
    //std::cout << "full_block =\n" << full_block << "\n";


    blockSQP::SymMatrix *hess = new blockSQP::SymMatrix[4];
    hess[0] = identity(1);
    hess[1] = identity(3);
    hess[2] = identity(3);
    hess[3] = identity(3);
    //hess[1] = full_block;
    //hess[2] = full_block;
    //hess[3] = full_block;

    blockSQP::SymMatrix *hess_2 = new blockSQP::SymMatrix[4];
    hess_2[0] = identity(1);

    hess_2[1] = full_block;
    hess_2[2] = full_block;
    hess_2[3] = full_block;

    blockSQP::SymMatrix *hess_CC = new blockSQP::SymMatrix[4];
    for (int i = 0; i < 4; i++){
        hess_CC[i] = hess[i]*(1-t) + hess_2[i]*t;
    }


    std::cout << "The quadratic objective is xi^T H xi, with H block-structured with 4 identity-blocks of sizes 1,3,3,3\n";


    blockSQP::Matrix grad_obj(10,1);
    grad_obj.Initialize(1.);

    blockSQP::Matrix lb_var(10,1);
    blockSQP::Matrix ub_var(10,1);
    blockSQP::Matrix lb_con(7,1);
    blockSQP::Matrix ub_con(7,1);

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
                con_jac.row, con_jac.colind, con_jac.nz);

    convertHessian(1.0e-15, hess_CC, 4, con_jac.n, hess_nz, hess_row, hess_colind, hess_loind);
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
    std::cout << "Solver of uncondensed QP returned, ret is " << ret << "\n";



    blockSQP::Matrix xi(10);
    blockSQP::Matrix lambda(17);
    qp->getPrimalSolution(xi.array);
    qp->getDualSolution(lambda.array);

    //std::cout << "Primal solution of uncondensed QP is\n" << xi << "\n";
    //std::cout << "Dual solution of uncondensed QP is\n" << lambda << "\n";


    delete qp;
    delete A_qp;
    delete H;


    ///#############################
    //Condense the problem with the convex combination hessian

    blockSQP::SymMatrix *condensed_hess_CC = nullptr;
    blockSQP::Sparse_Matrix condensed_Jacobian_CC;
    blockSQP::Matrix condensed_h_CC;
    blockSQP::Matrix condensed_lb_var_CC;
    blockSQP::Matrix condensed_ub_var_CC;
    blockSQP::Matrix condensed_lb_con_CC;
    blockSQP::Matrix condensed_ub_con_CC;



    C->full_condense(grad_obj, con_jac, hess_CC, lb_var, ub_var, lb_con, ub_con,
        condensed_h_CC, condensed_Jacobian_CC, condensed_hess_CC, condensed_lb_var_CC, condensed_ub_var_CC, condensed_lb_con_CC, condensed_ub_con_CC
    );


    std::cout << "Condensed hess =\n" << condensed_hess_CC[0] << "\n";
    std::cout << "Condensed jacobian =\n" << condensed_Jacobian_CC.dense() << "\n";
    std::cout << "Condensed linear form h =\n" << condensed_h_CC << "\n";
    std::cout << "Condensed lb_var =\n" << condensed_lb_var_CC << "\n\ncondensed_ub_var =\n" << condensed_ub_var_CC << "\n";
    std::cout << "Condensed lb_con =\n" << condensed_lb_con_CC << "\n\ncondensed_ub_con =\n" << condensed_ub_con_CC << "\n";


    //Solve the condensed QP
    qpOASES::SQProblem* qp_cond_CC;
    qpOASES::returnValue ret_cond_CC;

    qpOASES::Matrix *A_qp_cond_CC;
    qpOASES::SymmetricMatrix *H_cond_CC;

    double *hess_nz_cond_CC = nullptr;
    int *hess_row_cond_CC = nullptr;
    int *hess_colind_cond_CC = nullptr;
    int *hess_loind_cond_CC = nullptr;

    qp_cond_CC = new qpOASES::SQProblemSchur( condensed_Jacobian_CC.n, condensed_Jacobian_CC.m, qpOASES::HST_UNKNOWN, 50 );

    A_qp_cond_CC = new qpOASES::SparseMatrix(condensed_Jacobian_CC.m, condensed_Jacobian_CC.n,
                condensed_Jacobian_CC.row, condensed_Jacobian_CC.colind, condensed_Jacobian_CC.nz);

    convertHessian(1.0e-15, condensed_hess_CC, 1, condensed_Jacobian_CC.n, hess_nz_cond_CC, hess_row_cond_CC, hess_colind_cond_CC, hess_loind_cond_CC);
    std::cout << "converted Hessians\n";


    H_cond_CC = new qpOASES::SymSparseMat(condensed_Jacobian_CC.n, condensed_Jacobian_CC.n, hess_row_cond_CC, hess_colind_cond_CC, hess_nz_cond_CC);
    dynamic_cast<qpOASES::SymSparseMat*>(H_cond_CC)->createDiagInfo();

    double *g_cond_CC = condensed_h_CC.array;
    double *lb_cond_CC = condensed_lb_var_CC.array;
    double *ub_cond_CC = condensed_ub_var_CC.array;
    double *lbA_cond_CC = condensed_lb_con_CC.array;
    double *ubA_cond_CC = condensed_ub_con_CC.array;
    double cpu_time_cond_CC = 10000;
    int max_it_cond_CC = 10000;

    qp_cond_CC->setOptions( opts );

    ret_cond_CC = qp_cond_CC->init(H_cond_CC, g_cond_CC, A_qp_cond_CC, lb_cond_CC, ub_cond_CC, lbA_cond_CC, ubA_cond_CC, max_it_cond_CC, &cpu_time_cond_CC);
    std::cout << "Solver of condensed QP returned, ret is " << ret_cond_CC << "\n";

    /*
    double *xi_cond_arr = new double[4];
    double *lambda_cond_arr = new double[10];
    qp_cond->getPrimalSolution(xi_cond_arr);
    qp_cond->getDualSolution(lambda_cond_arr);

    blockSQP::Matrix xi_cond(4,1,xi_cond_arr);
    blockSQP::Matrix lambda_cond(10,1,lambda_cond_arr);
    */

    blockSQP::Matrix xi_cond_CC(4);
    blockSQP::Matrix lambda_cond_CC(11);
    qp_cond_CC->getPrimalSolution(xi_cond_CC.array);
    qp_cond_CC->getDualSolution(lambda_cond_CC.array);

    std::cout << "xi_cond_CC=\n" << xi_cond_CC;
    std::cout << "lambda_cond_CC=\n" << lambda_cond_CC;


    blockSQP::Matrix xi_rest_CC;
    blockSQP::Matrix lambda_rest_CC;

    C->recover_var_mult(xi_cond_CC, lambda_cond_CC, xi_rest_CC, lambda_rest_CC);

    std::cout << "Primal solution of uncondensed convex-combination QP is\n" << xi << "\n";
    std::cout << "Dual solution of uncondensed convex-combination QP is\n" << lambda << "\n";

    std::cout << "Primal solution of condensed convex-combination QP after restoration is\n" << xi_rest_CC << "\n";
    std::cout << "Dual solution of condensed convex-combination QP after restoration is\n" << lambda_rest_CC << "\n";


    delete qp_cond_CC;
    delete A_qp_cond_CC;
    delete H_cond_CC;



    ///#############################

    //Condense the QP with the first hessian
    blockSQP::SymMatrix *condensed_hess = nullptr;
    blockSQP::Sparse_Matrix condensed_Jacobian;
    blockSQP::Matrix condensed_h;
    blockSQP::Matrix condensed_lb_var;
    blockSQP::Matrix condensed_ub_var;
    blockSQP::Matrix condensed_lb_con;
    blockSQP::Matrix condensed_ub_con;

    C->full_condense(grad_obj, con_jac, hess, lb_var, ub_var, lb_con, ub_con,
        condensed_h, condensed_Jacobian, condensed_hess, condensed_lb_var, condensed_ub_var, condensed_lb_con, condensed_ub_con
    );



    /// NEW HESSIAN ######################################################
    // Condense a second hessian and calculate convex combination

    blockSQP::SymMatrix *condensed_hess_2 = new blockSQP::SymMatrix[1];
    blockSQP::Matrix condensed_h_2;
    C->fallback_hessian_condense(hess_2, condensed_h_2, condensed_hess_2);

    std::cout << "Second condensed hessian is\n" << condensed_hess_2[0] << "\n";
    std::cout << "Condensed_h_2 = \n" << condensed_h_2 << "\n";

    blockSQP::SymMatrix *conv_cond_hess = new blockSQP::SymMatrix[1];

    conv_cond_hess[0] = condensed_hess[0] * (1 - t) + condensed_hess_2[0] * t;

    std::cout << "The convex combination of both condensed hessians is\n" << conv_cond_hess[0] << "\n";

    blockSQP::Matrix conv_cond_h = condensed_h * (1 - t) + condensed_h_2 * t;

    std::cout << "The convex combination of both condensed linear terms is\n" << conv_cond_h << "\n";

//Solve the condensed QP with convex combination hessian
    qpOASES::SQProblem* qp_conv;
    qpOASES::returnValue ret_conv;

    qpOASES::Matrix *A_qp_conv;
    qpOASES::SymmetricMatrix *H_conv;

    double *hess_nz_conv = nullptr;
    int *hess_row_conv = nullptr;
    int *hess_colind_conv = nullptr;
    int *hess_loind_conv = nullptr;

    qp_conv = new qpOASES::SQProblemSchur( condensed_Jacobian.n, condensed_Jacobian.m, qpOASES::HST_UNKNOWN, 50 );

    A_qp_conv = new qpOASES::SparseMatrix(condensed_Jacobian.m, condensed_Jacobian.n,
                condensed_Jacobian.row, condensed_Jacobian.colind, condensed_Jacobian.nz);

    convertHessian(1.0e-15, conv_cond_hess, 1, condensed_Jacobian.n, hess_nz_conv, hess_row_conv, hess_colind_conv, hess_loind_conv);
    std::cout << "converted convex combination Hessians\n";


    H_conv = new qpOASES::SymSparseMat(condensed_Jacobian.n, condensed_Jacobian.n, hess_row_conv, hess_colind_conv, hess_nz_conv);

    dynamic_cast<qpOASES::SymSparseMat*>(H_conv)->createDiagInfo();

    double *g_conv = conv_cond_h.array;
    double *lb_conv = condensed_lb_var.array;
    double *ub_conv = condensed_ub_var.array;
    double *lbA_conv = condensed_lb_con.array;
    double *ubA_conv = condensed_ub_con.array;
    double cpu_time_conv = 10000;
    int max_it_conv = 10000;

    qp_conv->setOptions( opts );


    std::cout << "Solving QP with convex combination hessian\n" << std::flush;
    ret_conv = qp_conv->init(H_conv, g_conv, A_qp_conv, lb_conv, ub_conv, lbA_conv, ubA_conv, max_it_conv, &cpu_time_conv);
    std::cout << "Solver of convex combination condensed QP returned, ret is " << ret_conv << "\n" << std::flush;



    blockSQP::Matrix xi_conv(4);
    blockSQP::Matrix lambda_conv(11);
    qp_conv->getPrimalSolution(xi_conv.array);
    qp_conv->getDualSolution(lambda_conv.array);

    std::cout << "xi_conv=\n" << xi_conv;
    std::cout << "lambda_conv=\n" << lambda_conv;


    blockSQP::Matrix xi_conv_rest;
    blockSQP::Matrix lambda_conv_rest;

    C->convex_combination_recover(xi_conv, lambda_conv, t, xi_conv_rest, lambda_conv_rest);

    std::cout << "Primal solution of convex-combination QP after restoration is\n" << xi_conv_rest << "\n";
    std::cout << "Dual solution of convex-combination QP after restoration is\n" << lambda_conv_rest << "\n";


    delete qp_conv;
    delete A_qp_conv;
    delete H_conv;



    ///////


    //##############

    delete[] vblocks;
    delete[] cblocks;
    delete[] hsizes;
    delete[] targets;
    delete[] hess;
    delete[] hess_2;
    delete[] condensed_hess;
    delete[] condensed_hess_2;
    delete[] conv_cond_hess;
    delete[] hess_CC;

    delete[] hess_nz;
    delete[] hess_row;

    delete[] hess_nz_cond_CC;
    delete[] hess_row_cond_CC;

    delete[] hess_nz_conv;
    delete[] hess_row_conv;

    delete C;

    return 0;

}

