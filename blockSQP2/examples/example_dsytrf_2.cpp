#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <qpOASES/LapackBlas.hpp>


static void printMat(const char* name, const double* A, int n)
{
    std::cout << name << "\n";
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
            std::cout << A[i + j*n] << "\t";
        std::cout << "\n";
    }
    std::cout << "\n";
}

static void printVec(const char* name, const double* x, int n)
{
    std::cout << name << "\n";
    for(int i=0;i<n;i++)
        std::cout << x[i] << "\n";
    std::cout << "\n";
}

int main()
{
    const int n = 3;
    const int lda = n;

    //
    // Symmetric indefinite matrix
    //
    // [ 0  1  0  0 ]
    // [ 1  0  0  0 ]
    // [ 0  0  2  1 ]
    // [ 0  0  1  3 ]
    //
    // eigenvalues:
    //  -1, +1, 1.381966, 3.618034
    //
    // double A[n*n] =
    // {
    //     0,1,0,0,
    //     1,0,0,0,
    //     0,0,2,1,
    //     0,0,1,3
    // };
    
    double A[n*n] = 
    {2.221e-13, 0, 1, 
    0, 2.221e-13, -1, 
    1, -1, 0,};

    

    double Aorig[n*n];
    std::memcpy(Aorig,A,sizeof(A));

    printMat("Original A",A,n);

    int ipiv[n];
    int info;

    //
    // Workspace query
    //
    double work_query;
    int lwork = -1;

    // dsytrf_("L",
    //         &n,
    //         A,
    //         &lda,
    //         ipiv,
    //         &work_query,
    //         &lwork,
    //         &info);
    SYTRF("L",
            &n,
            A,
            &lda,
            ipiv,
            &work_query,
            &lwork,
            &info STRLENS1(1));

    std::cout << "query info = " << info << "\n";
    std::cout << "recommended lwork = "
              << static_cast<int>(work_query)
              << "\n\n";

    lwork = static_cast<int>(work_query);
    std::vector<double> work(lwork);

    //
    // Factorization
    //
    // dsytrf_("L",
    //         &n,
    //         A,
    //         &lda,
    //         ipiv,
    //         work.data(),
    //         &lwork,
    //         &info);
    SYTRF("L",
            &n,
            A,
            &lda,
            ipiv,
            work.data(),
            &lwork,
            &info STRLENS1(1));

    std::cout << "factorization info = "
              << info << "\n\n";

    printMat("Factorized storage",A,n);

    std::cout << "IPIV:\n";
    for(int i=0;i<n;i++)
        std::cout << ipiv[i] << " ";
    std::cout << "\n\n";

    //
    // Solve A x = b
    //
    // double b[n] = {1,2,3,4};
    double b[n] = {-20, 10, 0};
        
    int val = 1;
    // dsytrs_("L",
    //         &n,
    //         &val,
    //         A,
    //         &lda,
    //         ipiv,
    //         b,
    //         &n,
    //         &info);
    SYTRS("L",
            &n,
            &val,
            A,
            &lda,
            ipiv,
            b,
            &n,
            &info STRLENS1(1));

    std::cout << "solve info = "
              << info << "\n\n";

    printVec("solution x",b,n);

    //
    // residual
    //
    double r[n];

    for(int i=0;i<n;i++)
    {
        double Ax = 0.0;
        for(int j=0;j<n;j++)
            Ax += Aorig[i+j*n] * b[j];

        r[i] = Ax;
    }

    printVec("A*x",r,n);

    return 0;
}