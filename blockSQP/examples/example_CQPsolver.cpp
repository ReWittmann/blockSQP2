#include "blocksqp_condensing.hpp"
#include "blocksqp_qpsolver.hpp"
#include "blocksqp_options.hpp"
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

    
    //Layout information
    std::unique_ptr<blockSQP::vblock[]> vblocks = std::make_unique<blockSQP::vblock[]>(7);
    vblocks[0] = blockSQP::vblock(1,false);

    vblocks[1] = blockSQP::vblock(2,true);
    vblocks[2] = blockSQP::vblock(1,false);

    vblocks[3] = blockSQP::vblock(2,true);
    vblocks[4] = blockSQP::vblock(1,false);

    vblocks[5] = blockSQP::vblock(2,true);
    vblocks[6] = blockSQP::vblock(1,false);


    std::unique_ptr<blockSQP::cblock[]> cblocks = std::make_unique<blockSQP::cblock[]>(4);
    cblocks[0] = blockSQP::cblock(2);
    cblocks[1] = blockSQP::cblock(2);
    cblocks[2] = blockSQP::cblock(2);
    cblocks[3] = blockSQP::cblock(1);

    std::unique_ptr<int[]> hsizes = std::unique_ptr<int[]>(new int[4]{1,3,3,3});
    std::unique_ptr<int[]> blockIdx = std::unique_ptr<int[]>(new int[5]{0,1,3,7,10});

    std::unique_ptr<blockSQP::condensing_target[]> targets = std::make_unique<blockSQP::condensing_target[]>(1);
    targets[0] = blockSQP::condensing_target(3,0,7,0,3);

    blockSQP::Condenser C(vblocks.get(), 7, cblocks.get(), 4, hsizes.get(), 4, targets.get(), 1);


    C.print_debug();


    //Uncondensed Problem
    std::unique_ptr<double[]> NZ = std::unique_ptr<double[]>(new double[26]{-1,-2,1,1,-2,1,1,-1,1,-1,-2,1,1,-2,1,1,-1,1,-1,-2,1,1,1,1,1,1});
    std::unique_ptr<int[]> ROW = std::unique_ptr<int[]>(new int[26]{0,1,6,0,2,6,1,3,6,2,3,6,2,4,6,3,5,6,4,5,6,4,6,5,6,6});
    std::unique_ptr<int[]> COLIND = std::unique_ptr<int[]>(new int[11]{0,3,6,9,12,15,18,21,23,25,26});

    blockSQP::Sparse_Matrix con_jac(7, 10, std::move(NZ), std::move(ROW), std::move(COLIND));

    std::cout << "A =\n" << con_jac.dense() << "\n";


    //full_block = 0.25 * ([1,-1,1]*[1,-1,1]^T + [1,1,0]*[1,1,0]^T + [-1,1,2]*[-1,1,2]^T)
    blockSQP::SymMatrix full_block(3);
    full_block(0,0) = 0.75; full_block(1,0) = -0.25; full_block(2,0) = -0.25; full_block(1,1) = 0.75; full_block(2,1) = 0.25; full_block(2,2) = 1.25;

    std::unique_ptr<blockSQP::SymMatrix[]> hess = std::make_unique<blockSQP::SymMatrix[]>(4);
    hess[0] = identity(1);
    hess[1] = full_block;
    hess[2] = full_block;
    hess[3] = full_block;

	std::cout << "The quadratic objective is xi^T H xi, with H block-structured with 4 blocks, [1] +\n" << full_block << "x3\n";

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
    
    
    blockSQP::qpOASES_options QPOPTS;
    QPOPTS.sparsityLevel = 2;
    
    std::unique_ptr<blockSQP::QPsolverBase> QPsol = std::make_unique<blockSQP::qpOASES_solver>(10, 7, 4, blockIdx.get(), &QPOPTS);
    QPsol->set_hess(hess.get(), true);
    QPsol->set_lin(grad_obj);
    QPsol->set_bounds(lb_var,ub_var,lb_con,ub_con);
    QPsol->set_constr(con_jac.nz.get(), con_jac.row.get(), con_jac.colind.get());
    
    blockSQP::Matrix xi(10);
    blockSQP::Matrix lambda(17);
    QPsol->solve(xi, lambda);
    
    std::unique_ptr<blockSQP::QPsolverBase> QPsol_cond = std::make_unique<blockSQP::qpOASES_solver>(
        C.condensed_num_vars, C.condensed_num_cons, C.condensed_num_hessblocks, C.condensed_blockIdx, &QPOPTS);
    
    blockSQP::CQPsolver CQPsol(QPsol_cond.get(), &C);
    
    CQPsol.set_hess(hess.get(), true);
    CQPsol.set_lin(grad_obj);
    CQPsol.set_bounds(lb_var,ub_var,lb_con,ub_con);
    CQPsol.set_constr(con_jac.nz.get(), con_jac.row.get(), con_jac.colind.get());
    
    blockSQP::Matrix xi_rest(10);
    blockSQP::Matrix lambda_rest(17);
    CQPsol.solve(xi_rest, lambda_rest);
    std::cout << "xi = \n" << xi << "\nxi_rest = \n" << xi_rest << "\n";
    
    
    /*
    
    //Condense the QP
    blockSQP::SymMatrix *condensed_hess = new blockSQP::SymMatrix[C->condensed_num_hessblocks];
    blockSQP::Sparse_Matrix condensed_Jacobian;
    blockSQP::Matrix condensed_h;
    blockSQP::Matrix condensed_lb_var;
    blockSQP::Matrix condensed_ub_var;
    blockSQP::Matrix condensed_lb_con;
    blockSQP::Matrix condensed_ub_con;

    C->full_condense(grad_obj, con_jac, hess, lb_var, ub_var, lb_con, ub_con,
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


    blockSQP::Matrix xi_cond(4);
    blockSQP::Matrix lambda_cond(11);
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

    delete[] hess_nz_cond;
    delete[] hess_row_cond;

    delete C;

    */
    
    return 0;

}

