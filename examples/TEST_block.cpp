#include "blocksqp_matrix.hpp"
#include <iostream>

int main(){
int *msizes = new int[2]{2,2};
int *nsizes = new int[3]{2,2,2};

blockSQP::LT_Block_Matrix B(2, 3, msizes, nsizes);

blockSQP::Matrix M_00(2,2);
M_00.Initialize(0.);
M_00(1,0) = 3.0;

blockSQP::Matrix M_10(2,2);
M_10.Initialize(-1.);
M_10(0,1) = 5.0;

blockSQP::Matrix M_11(2,2);
M_11.Initialize(1.);
M_11(1,1) = 7.0;

B.set(0,0,M_00);
B.set(1,0,M_10);
B.set(1,1,M_11);

blockSQP::Matrix B_dense;
B.to_dense(B_dense);

std::cout << B_dense << "\n";

blockSQP::Sparse_Matrix B_sparse;
B.to_sparse(B_sparse);

blockSQP::Sparse_Matrix B_pad = lr_zero_pad(10, B_sparse, 3);

std::cout << B_sparse.dense() << "\n";
std::cout << B_pad.dense() << "\n";
std::cout << "B_sparse.nnz: " << B_sparse.nnz << ", B_sparse.colind[B_sparse.n] " << B_sparse.colind[B_sparse.n] << "\n";

return 0;;
}
