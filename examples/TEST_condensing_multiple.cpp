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
        for( i=0; i<hess_[iBlock].N(); i++ )
            for( j=i; j<hess_[iBlock].N(); j++ )
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
        nCols = hess_[iBlock].N();
        nRows = hess_[iBlock].M();

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
    blockSQP::vblock *vblocks = new blockSQP::vblock[15];
    vblocks[0] = blockSQP::vblock(1,false);

    vblocks[1] = blockSQP::vblock(2,true);
    vblocks[2] = blockSQP::vblock(1,false);

    vblocks[3] = blockSQP::vblock(2,true);
    vblocks[4] = blockSQP::vblock(1,false);

    vblocks[5] = blockSQP::vblock(2,true);
    vblocks[6] = blockSQP::vblock(1,false);


    vblocks[7] = blockSQP::vblock(1,false);


    vblocks[8] = blockSQP::vblock(1,false);

    vblocks[9] = blockSQP::vblock(2,true);
    vblocks[10] = blockSQP::vblock(1,false);

    vblocks[11] = blockSQP::vblock(2,true);
    vblocks[12] = blockSQP::vblock(1,false);

    vblocks[13] = blockSQP::vblock(2,true);
    vblocks[14] = blockSQP::vblock(1,false);



    blockSQP::cblock *cblocks = new blockSQP::cblock[7];
    cblocks[0] = blockSQP::cblock(2);
    cblocks[1] = blockSQP::cblock(2);
    cblocks[2] = blockSQP::cblock(2);
    cblocks[3] = blockSQP::cblock(2);
    cblocks[4] = blockSQP::cblock(2);
    cblocks[5] = blockSQP::cblock(2);
    cblocks[6] = blockSQP::cblock(1);

    int *hsizes = new int[9]{1,3,3,3, 1, 1,3,3,3};

    blockSQP::condensing_target *targets = new blockSQP::condensing_target[2];
    targets[0] = blockSQP::condensing_target(3,0,7,0,3);
    targets[1] = blockSQP::condensing_target(3,8,15,3,6);

    blockSQP::Condenser *C = new blockSQP::Condenser(vblocks, 15, cblocks, 7, hsizes, 9, targets, 2);
    C->print_debug();


    //Uncondensed Problem
    //double *NZ = new double[26]{-1,-2,1,1,-2,1,1,-1,1,-1,-2,1,1,-2,1,1,-1,1,-1,-2,1,1,1,1,1,1};
    //int *ROW = new int[26]{0,1,6,0,2,6,1,3,6,2,3,6,2,4,6,3,5,6,4,5,6,4,6,5,6,6};
    //int *COLIND = new int[11]{0,3,6,9,12,15,18,21,23,25,26};

    double *NZ = new double[53]{-1,-2,1, 1,-2,1, 1,-1,1, -1,-2,1, 1,-2,1, 1,-1,1, -1,-2,1, 1,1, 1,1, 1, 1, 1,1,1, 1,0.5,1, 1,1,1, -1,-1,1, 1,1,1, 1,0.5,1, 1,1,1, 1,1, 1,1, 1};
    int *ROW = new int[53]{0,1,12, 0,2,12, 1,3,12, 2,3,12, 2,4,12, 3,5,12, 4,5,12, 4,12, 5,12, 12, 12, 6,7,12, 6,8,12, 7,9,12, 8,9,12, 8,10,12, 9,11,12, 10,11,12, 10,12, 11,12, 12};
    int *COLIND = new int[22]{0,3,6,9,12,15,18,21, 23, 25, 26, 27, 30, 33, 36, 39, 42, 45, 48, 50, 52, 53};

    blockSQP::Sparse_Matrix con_jac(13,21,53, NZ, ROW, COLIND);

    std::cout << "A =\n" << con_jac.dense() << "\n";

    blockSQP::SymMatrix full_block(3);
    full_block(0,0) = 0.75; full_block(1,0) = -0.25; full_block(2,0) = -0.25; full_block(1,1) = 0.75; full_block(2,1) = 0.25; full_block(2,2) = 1.25;
    //std::cout << "full_block =\n" << full_block << "\n";


    blockSQP::SymMatrix *hess = new blockSQP::SymMatrix[9];
    hess[0] = identity(1);
    hess[1] = full_block;
    hess[2] = full_block;
    hess[3] = full_block;

    hess[4] = identity(1);

    hess[5] = identity(1);
    hess[6] = identity(3);
    hess[7] = identity(3);
    hess[8] = identity(3);


    std::cout << "The quadratic objective is xi^T H xi, with H block-structured with 4 identity-blocks of sizes 1,3,3,3\n";


    blockSQP::Matrix grad_obj(21,1);
    grad_obj.Initialize(1.);
    grad_obj(10) = -0.1;

    blockSQP::Matrix lb_var(21,1);
    blockSQP::Matrix ub_var(21,1);
    blockSQP::Matrix lb_con(13,1);
    blockSQP::Matrix ub_con(13,1);

    lb_var.Initialize(-0.3);
    ub_var.Initialize(0.3);
    lb_con.Initialize(0.1);
    ub_con.Initialize(0.1);
    lb_con(12) = -2.5;
    ub_con(12) = 2.5;

    std::cout << "c_T0 = c_T1 =\n[0.1\n0.1\n0.1\n0.1\n0.1\n0.1]\n";

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

    convertHessian(1.0e-15, hess, 9, con_jac.n, hess_nz, hess_row, hess_colind, hess_loind);
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



    blockSQP::Matrix xi(21);
    blockSQP::Matrix lambda(34);
    qp->getPrimalSolution(xi.array);
    qp->getDualSolution(lambda.array);

    //std::cout << "Primal solution of uncondensed QP is\n" << xi << "\n";
    //std::cout << "Dual solution of uncondensed QP is\n" << lambda << "\n";


    delete qp;
    delete A_qp;
    delete H;


    //Condense the QP
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

    std::cout << "Condensed hess 0 =\n" << condensed_hess[0] << "\nCondensed hess 1=\n" << condensed_hess[1] << "\n";
    std::cout << "Condensed jacobian =\n" << condensed_Jacobian.dense() << "\n";
    std::cout << "Condensed linear form h =\n" << "condensed_h" << "\n";
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
                condensed_Jacobian.row, condensed_Jacobian.colind, condensed_Jacobian.nz);

    convertHessian(1.0e-15, condensed_hess, 3, condensed_Jacobian.n, hess_nz_cond, hess_row_cond, hess_colind_cond, hess_loind_cond);
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



    blockSQP::Matrix xi_cond(9);
    blockSQP::Matrix lambda_cond(22);
    qp_cond->getPrimalSolution(xi_cond.array);
    qp_cond->getDualSolution(lambda_cond.array);

    std::cout << "xi_cond=\n" << xi_cond;
    std::cout << "lambda_cond=\n" << lambda_cond;


    blockSQP::Matrix xi_rest;
    blockSQP::Matrix lambda_rest;

    C->recover_var_mult(xi_cond, lambda_cond, xi_rest, lambda_rest);

    std::cout << "Primal solution of uncondensed QP is\n" << xi << "\n";
    std::cout << "Dual solution of uncondensed QP is\n" << lambda << "\n";

    std::cout << "Primal solution of condensed QP after restoration is\n" << xi_rest << "\n";
    std::cout << "Dual solution of condensed QP after restoration is\n" << lambda_rest << "\n";




    delete qp_cond;
    delete A_qp_cond;
    delete H_cond;

    //##############


    delete[] vblocks;
    delete[] cblocks;
    delete[] hsizes;
    delete[] targets;
    delete[] hess;
    delete[] condensed_hess;

    delete C;

    return 0;

}

