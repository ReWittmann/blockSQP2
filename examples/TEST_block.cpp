#include "blocksqp_matrix.hpp"
#include "blocksqp_general_purpose.hpp"
#include <iostream>

void convertHessian( double eps, blockSQP::SymMatrix const *hess, int nBlocks, int nVar,
                                            double *&hessNz){
    if( hessNz == NULL )
        hessNz = new double[nVar * nVar];

    int bsize, bstart = 0, ind = 0;
    //Iterate over hessian blocks
    for (int h = 0; h < nBlocks; h++){
        bsize = hess[h].m;
        //Iterate of second dimension
        for (int j = 0; j < bsize; j++){
            //Segment above hessian block
            for (int i = 0; i < bstart; i++){
                hessNz[ind] = 0;
                ++ind;
            }
            //Hessian block
            for (int i = 0; i < bsize; i++){
                hessNz[ind] = hess[h](i, j);
                ++ind;
            }
            //Segement below hessian block
            for (int i = bstart + bsize; i < nVar; i++){
                hessNz[ind] = 0;
                ++ind;
            }
        }
        bstart += bsize;
    }
    std::cout << "ind = " << ind << "\n";

    return;
}


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



blockSQP::SymMatrix *hess = new blockSQP::SymMatrix[2];
hess[0].Dimension(3).Initialize(1);
hess[1].Dimension(2).Initialize(2);


double *HNZ = new double[25];
convertHessian(1e-14, hess, 2, 5, HNZ);

for (int i = 0; i < 5; i++){
    for (int j = 0; j < 5; j++){
        std::cout << HNZ[i + j*5] << " ";
    }
    std::cout << "\n";
}
delete[] HNZ;


throw blockSQP::NotImplementedError("TEST");

return 0;
}
