#include <blockSQP2.hpp>
using namespace blockSQP2;

#include <chrono>
using namespace std::chrono;

#define DIM 3



// void mult(const Matrix &M1, const Matrix &M2, Matrix &M3){
//     if (M3.m != M1.m || M3.n != M2.n) [[unlikely]] M3.Dimension(M1.m, M2.n);
    
//     BSQP_BLASFUNC(cblas_dgemm)(CblasColMajor, CblasNoTrans, CblasNoTrans, blasint(M1.m), blasint(M2.n), blasint(M1.n), 1.0, M1.array, blasint(M1.ldim), M2.array, blasint(M2.ldim), 0., M3.array, blasint(M3.ldim));
// }

void hand_mult(const Matrix &M1, const Matrix &M2, Matrix &M3){
    // if (M3.m != M1.m || M3.n != M2.n) [[unlikely]] M3.Dimension(M1.m, M2.n);
    // if (M1.m == 0 || M2.n == 0) [[unlikely]] return;
    
    M3.Initialize(0.);
    
    for (int k = 0; k < M2.n; k++){
        for (int i = 0; i < M1.m; i++){
            for (int j = 0; j < M1.n; j++){
                M3.array[i + k*M3.m] += M1.array[i + j*M1.m]*M2.array[j + k*M2.m];
                // M3(i, k) += M1(i,j)*M2(j,k);
            }
        }
    }
}


int main(){
    Matrix A(DIM,DIM); A.Initialize(1.0);
   
    Matrix B(DIM,1); B.Initialize(1.0);
    Matrix C(DIM,1);
    
    
    
    steady_clock::time_point T0 = steady_clock::now();
    for (int i = 0; i < 10000; i++){
        C = A*B;
    }
    steady_clock::time_point T1 = steady_clock::now();
    
    std::cout << "Multiplications took " << duration_cast<nanoseconds>(T1 - T0) << "\n";
    
    
    T0 = steady_clock::now();
    for (int i = 0; i < 10000; i++){
        C = A*B;
    }
    T1 = steady_clock::now();
    
    std::cout << "Multiplications took " << duration_cast<nanoseconds>(T1 - T0) << "\n";
    
    
    
    T0 = steady_clock::now();
    for (int i = 0; i < 10000; i++){
        mult(A,B,C);
    }
    T1 = steady_clock::now();
    
    std::cout << "mults took " << duration_cast<nanoseconds>(T1 - T0) << "\n";
    
    
    double *A2 = new double[DIM*DIM]; std::fill(A2, A2 + DIM*DIM, 1.0);
    double *B2 = new double[DIM]; std::fill(B2, B2 + DIM, 1.0);
    double *C2 = new double[DIM];

    
    T0 = steady_clock::now();
    for (int i = 0; i < 10000; i++){
        BSQP_BLASFUNC(cblas_dgemm)(CblasColMajor, CblasNoTrans, CblasNoTrans, blasint(DIM), blasint(1), blasint(DIM), 1.0, A2, blasint(DIM), B2, blasint(DIM), 0., C2, blasint(DIM));
    }
    T1 = steady_clock::now();
    std::cout << "Direct cblas calls took " << duration_cast<nanoseconds>(T1 - T0) << "\n";
    
    
    
    
    T0 = steady_clock::now();
    for (int i = 0; i < 10000; i++){
        // double *TEMP = new double[DIM];
        if (i*i > 123409223432345){
            throw;
            // B2[0] = 12.0;
        }
        
        BSQP_BLASFUNC(cblas_dgemm)(CblasColMajor, CblasNoTrans, CblasNoTrans, blasint(DIM), blasint(1), blasint(DIM), 1.0, A2, blasint(DIM), B2, blasint(DIM), 0., C2, blasint(DIM));
        // std::copy(TEMP, TEMP+DIM, C2);
        // delete[] TEMP;
        // BSQP_BLASFUNC(cblas_dgemv)(CblasColMajor, CblasNoTrans, blasint(3), blasint(3), 1.0, A2, blasint(3), B2, blasint(1), 0., C2, blasint(1));
    }
        T1 = steady_clock::now();
    std::cout << "Direct cblas calls with overhead took " << duration_cast<nanoseconds>(T1 - T0) << "\n";
    
    
    T0 = steady_clock::now();
    for (int i = 0; i < 10000; i++){
        double *TEMP = new double[DIM];
        if (long(TEMP) > 123409223432345) throw;
        
        std::copy(B2, B2+DIM, TEMP);
        delete[] TEMP;
    }
        T1 = steady_clock::now();
    std::cout << "Allocations took " << duration_cast<nanoseconds>(T1 - T0) << "\n";
    
    
    
    
    T0 = steady_clock::now();
    for (int i = 0; i < 10000; i++){
        hand_mult(A,B,C);
    }
        T1 = steady_clock::now();
    std::cout << "Hand mult took " << duration_cast<nanoseconds>(T1 - T0) << "\n";
    
    hand_mult(A,B,C);
    std::cout << C << "\n";
    
    if (DIM == 3){
        T0 = steady_clock::now();
        for (int i = 0; i < 10000; i++){
            B2[0] = i; B2[1] = i; B2[2] = i;
            
            C2[0] = A2[0]*B2[0] + A2[3]*B2[1] + A2[6]*B2[2];
            C2[1] = A2[1]*B2[0] + A2[4]*B2[1] + A2[7]*B2[2];
            C2[1] = A2[2]*B2[0] + A2[5]*B2[1] + A2[8]*B2[2];
        }
            // BSQP_BLASFUNC(cblas_dgemm)(CblasColMajor, CblasNoTrans, CblasNoTrans, blasint(3), blasint(1), blasint(3), 1.0, A2, blasint(3), B2, blasint(3), 0., C2, blasint(3));
            // BSQP_BLASFUNC(cblas_dgemv)(CblasColMajor, CblasNoTrans, blasint(3), blasint(3), 1.0, A2, blasint(3), B2, blasint(1), 0., C2, blasint(1));
        T1 = steady_clock::now();
        std::cout << "Handwritten unrolled loop took " << duration_cast<nanoseconds>(T1 - T0) << "\n";
    }
    
    
    
    
    double *BM2 = new double[DIM*DIM];
    double *CM2 = new double[DIM*DIM];
    
    if (DIM == 3){
        T0 = steady_clock::now();
        // for (int i = 0; i < 10000; i++){
            CM2[0] = A2[0]*BM2[0] + A2[3]*BM2[1] + A2[6]*BM2[2];
            CM2[1] = A2[1]*BM2[0] + A2[4]*BM2[1] + A2[7]*BM2[2];
            CM2[2] = A2[2]*BM2[0] + A2[5]*BM2[1] + A2[8]*BM2[2];
            
            CM2[3] = A2[0]*BM2[3] + A2[3]*BM2[4] + A2[6]*BM2[5];
            CM2[4] = A2[1]*BM2[3] + A2[4]*BM2[4] + A2[7]*BM2[5];
            CM2[5] = A2[2]*BM2[3] + A2[5]*BM2[4] + A2[8]*BM2[5];
            
            CM2[6] = A2[0]*BM2[6] + A2[3]*BM2[7] + A2[6]*BM2[8];
            CM2[7] = A2[1]*BM2[6] + A2[4]*BM2[7] + A2[7]*BM2[8];
            CM2[8] = A2[2]*BM2[6] + A2[5]*BM2[7] + A2[8]*BM2[8];
        // }
        T1 = steady_clock::now();
        std::cout << "unrolled 3x3 took " << duration_cast<nanoseconds>(T1 - T0) << "\n";
    }
    
    if (DIM == 4){
        T0 = steady_clock::now();
        // for (int i = 0; i < 10000; i++){
            CM2[0] = A2[0]*BM2[0] + A2[4]*BM2[1] + A2[8]*BM2[2] + A2[12]*BM2[3];
            CM2[1] = A2[1]*BM2[0] + A2[5]*BM2[1] + A2[9]*BM2[2] + A2[13]*BM2[3];
            CM2[2] = A2[2]*BM2[0] + A2[6]*BM2[1] + A2[10]*BM2[2] + A2[14]*BM2[3];
            CM2[2] = A2[3]*BM2[0] + A2[7]*BM2[1] + A2[11]*BM2[2] + A2[15]*BM2[3];
            
            CM2[3] = A2[0]*BM2[4] + A2[4]*BM2[5] + A2[8]*BM2[6] + A2[12]*BM2[7];
            CM2[4] = A2[1]*BM2[4] + A2[5]*BM2[5] + A2[9]*BM2[6] + A2[13]*BM2[7];
            CM2[5] = A2[2]*BM2[4] + A2[6]*BM2[5] + A2[10]*BM2[6] + A2[14]*BM2[7];
            CM2[2] = A2[3]*BM2[4] + A2[7]*BM2[5] + A2[11]*BM2[6] + A2[15]*BM2[7];
            
            CM2[6] = A2[0]*BM2[8] + A2[4]*BM2[9] + A2[8]*BM2[10] + A2[12]*BM2[11];
            CM2[7] = A2[1]*BM2[8] + A2[5]*BM2[9] + A2[9]*BM2[10] + A2[13]*BM2[11];
            CM2[8] = A2[2]*BM2[8] + A2[6]*BM2[9] + A2[10]*BM2[10] + A2[14]*BM2[11];
            CM2[2] = A2[3]*BM2[8] + A2[7]*BM2[9] + A2[11]*BM2[10] + A2[15]*BM2[11];
            
            CM2[6] = A2[0]*BM2[12] + A2[4]*BM2[13] + A2[8]*BM2[11] + A2[12]*BM2[15];
            CM2[7] = A2[1]*BM2[12] + A2[5]*BM2[13] + A2[9]*BM2[11] + A2[13]*BM2[15];
            CM2[8] = A2[2]*BM2[12] + A2[6]*BM2[13] + A2[10]*BM2[11] + A2[14]*BM2[15];
            CM2[2] = A2[3]*BM2[12] + A2[7]*BM2[13] + A2[11]*BM2[11] + A2[15]*BM2[15];
        // }
        T1 = steady_clock::now();
        std::cout << "unrolled 4x4 took " << duration_cast<nanoseconds>(T1 - T0) << "\n";
    }
    
    
    T0 = steady_clock::now();
    // for (int i = 0; i < 10000; i++){
        BSQP_BLASFUNC(cblas_dgemm)(CblasColMajor, CblasNoTrans, CblasNoTrans, blasint(DIM), blasint(DIM), blasint(DIM), 1.0, A2, blasint(DIM), BM2, blasint(DIM), 0., CM2, blasint(DIM));
    // }
    T1 = steady_clock::now();
    std::cout << "cblas nxn took " << duration_cast<nanoseconds>(T1 - T0) << "\n";
    
    
    T0 = steady_clock::now();
    for (int j = 0; j < DIM; j++){
        for (int i = 0; i < DIM; i++){
            A2[i + j*DIM] *= 2.1;
        }
    }
    T1 = steady_clock::now();
    std::cout << "Manual scaling took " << duration_cast<nanoseconds>(T1 - T0) << "\n";
    
    T0 = steady_clock::now();
    BSQP_BLASFUNC(cblas_dscal)(blasint(DIM*DIM), 2.1, A2, blasint(1));
    T1 = steady_clock::now();
    std::cout << "dscal took " << duration_cast<nanoseconds>(T1 - T0) << "\n";
    
    // else
    //     BSQP_BLASFUNC(cblas_dgemv)(CblasColMajor, CblasNoTrans, blasint(m), blasint(n), 1.0, array, blasint(ldim), M2.array, blasint(1), 0., array_3, blasint(1));

    
    return 0;
}