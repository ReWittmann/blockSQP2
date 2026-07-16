
inline double &Matrix::operator()(int i, int j) const {
    #ifdef MATRIX_DEBUG 
    if ( i < 0 || i >= m || j < 0 || j >= n) throw std::invalid_argument("Matrix operator(): Indices (" + std::to_string(i) + ", " + std::to_string(j) + ") out of bounds for matrix of shape (" + std::to_string(m) + ", " + std::to_string(n) + ")"); 
    #endif
    return array[i+j*ldim];
}

inline double &Matrix::operator()(int i) const {
    #ifdef MATRIX_DEBUG
    if (i < 0 || i >= m) throw std::invalid_argument("Matrix operator(): Index " + std::to_string(i) + " out of bounds for matrix of shape (" + std::to_string(m) + ", " + std::to_string(n) + ")");
    #endif
    return array[i];
}

inline Matrix &Matrix::Dimension(int M, int N, int LDIM){
    if (M != m || N != n || (LDIM != ldim && LDIM != -1)){
        if (tflag) throw std::logic_error("Cannot set new dimension for Submatrix");
        else{
            deallocate();
            m = M;
            n = N;
            ldim = (std::max)(m, LDIM);
            allocate();
        }
    }
    return *this;
}

inline const Matrix &Matrix::Initialize(double val) const{
    for (int j = 0; j < n; j++) std::fill(array + j*ldim, array + m + j*ldim, val);
    return *this;
}



#define BSQP_SMALLMULT


#ifndef BSQP_SMALLMULT

inline void mult(const Matrix &M1, const Matrix &M2, Matrix &M3){
    #ifdef MATRIX_DEBUG
    if (M1.n != M2.m) throw std::invalid_argument("mult: Chaining dimension mismatch");
    if (M1.n == 0) throw std::invalid_argument("mult: Chaining dimension zero");
    #endif
  
    if (M3.m != M1.m || M3.n != M2.n) [[unlikely]] M3.Dimension(M1.m, M2.n);
    if (M1.m == 0 || M2.n == 0) [[unlikely]] return; 
    BSQP_BLASFUNC(cblas_dgemm)(CblasColMajor, CblasNoTrans, CblasNoTrans, blasint(M1.m), blasint(M2.n), blasint(M1.n), double(1.0), M1.array, blasint(M1.ldim), M2.array, blasint(M2.ldim), double(0.), M3.array, blasint(M3.ldim));
}

inline void Tmult(const Matrix &M1, const Matrix &M2, Matrix &M3){
    #ifdef MATRIX_DEBUG
    if (M1.m != M2.m) throw std::invalid_argument("Tmult: Chaining dimension mismatch");
    if (M1.m == 0) throw std::invalid_argument("Tmult: Chaining dimension zero");
    #endif
  
    if (M3.m != M1.n || M3.n != M2.n) [[unlikely]] M3.Dimension(M1.n, M2.n);
    if (M1.n == 0 || M2.n == 0) [[unlikely]] return; 
    BSQP_BLASFUNC(cblas_dgemm)(CblasColMajor, CblasTrans, CblasNoTrans, blasint(M1.n), blasint(M2.n), blasint(M1.m), double(1.0), M1.array, blasint(M1.ldim), M2.array, blasint(M2.ldim), double(0.), M3.array, blasint(M3.ldim));
}

inline void mult_to(const Matrix &M1, const Matrix &M2, const Matrix &M3, double beta){
    #ifdef MATRIX_DEBUG
      if (M1.n != M2.m) throw std::invalid_argument("mult_to: Chaining dimension mismatch");
      if (M1.n == 0) throw std::invalid_argument("mult_to: Chaining dimension zero");
      if (M3.m != M1.m || M3.n != M2.n) throw std::invalid_argument("mult_to: Output matrix dimension mismatch");
    #endif
    if (M1.m == 0 || M2.n == 0) [[unlikely]] return;
    BSQP_BLASFUNC(cblas_dgemm)(CblasColMajor, CblasNoTrans, CblasNoTrans, blasint(M1.m), blasint(M2.n), blasint(M1.n), double(1.0), M1.array, blasint(M1.ldim), M2.array, blasint(M2.ldim), beta, M3.array, blasint(M3.ldim));
}

inline void Tmult_to(const Matrix &M1, const Matrix &M2, const Matrix &M3, double beta){
    #ifdef MATRIX_DEBUG
      if (M1.m != M2.m) throw std::invalid_argument("Tmult_to: Chaining dimension mismatch");
      if (M1.m == 0) throw std::invalid_argument("Tmult_to: Chaining dimension zero");
      if (M3.m != M1.n || M3.n != M2.n) throw std::invalid_argument("Tmult_to: Output matrix dimension mismatch");
    #endif
    if (M1.n == 0 || M2.n == 0) [[unlikely]] return;
    BSQP_BLASFUNC(cblas_dgemm)(CblasColMajor, CblasTrans, CblasNoTrans, blasint(M1.n), blasint(M2.n), blasint(M1.m), double(1.0), M1.array, blasint(M1.ldim), M2.array, blasint(M2.ldim), beta, M3.array, blasint(M3.ldim));
}

#else //defined(BSQP_SMALLMULT) 


template<int M, int N> inline double (*rcst(double *ptr))[M*N]{return reinterpret_cast<double (*)[M*N]>(ptr);}

template <int M, int N, int K> inline void smallMultInternal(const double (* __restrict A)[M*K], const double (* __restrict B)[K*N], double(* __restrict C)[M*N]){
    for (int j = 0; j < N; j++){
        for (int i = 0; i < M; i++){
            double sum = 0;
            for (int k = 0; k < K; k++){
                sum += (*A)[i + k*M]*(*B)[k + K*j];
            }
            (*C)[i + j*M] = sum;
        }
    }
}
template<int M, int N, int K> inline void smallMult(double * __restrict A, double * __restrict B, double * __restrict C){
    smallMultInternal<M,N,K>(rcst<M,K>(A), rcst<K,N>(B), rcst<M,N>(C));
}


template <int M, int N, int K> inline void smallMultToInternal(const double (* __restrict A)[M*K], const double (* __restrict B)[K*N], double(* __restrict C)[M*N]){
    for (int j = 0; j < N; j++){
        for (int i = 0; i < M; i++){
            double sum = 0;
            for (int k = 0; k < K; k++){
                sum += (*A)[i + k*M]*(*B)[k + K*j];
            }
            (*C)[i + j*M] += sum;
        }
    }
}
template<int M, int N, int K> inline void smallMultTo(double * __restrict A, double * __restrict B, double * __restrict C){
    smallMultToInternal<M,N,K>(rcst<M,K>(A), rcst<K,N>(B), rcst<M,N>(C));
}


template <int M, int N, int K> inline void smallTmultInternal(const double (* __restrict A)[K*M], const double (* __restrict B)[K*N], double(* __restrict C)[M*N]){
    for (int j = 0; j < N; j++){
        for (int i = 0; i < M; i++){
            double sum = 0;
            for (int k = 0; k < K; k++){
                sum += (*A)[k + i*K]*(*B)[k + K*j];
            }
            (*C)[i + j*M] = sum;
        }
    }
}
template<int M, int N, int K> inline void smallTmult(double * __restrict A, double * __restrict B, double * __restrict C){
    smallTmultInternal<M,N,K>(rcst<K,M>(A), rcst<K,N>(B), rcst<M,N>(C));
}


template <int M, int N, int K> inline void smallTmultToInternal(const double (* __restrict A)[K*M], const double (* __restrict B)[K*N], double(* __restrict C)[M*N]){
    for (int j = 0; j < N; j++){
        for (int i = 0; i < M; i++){
            double sum = 0;
            for (int k = 0; k < K; k++){
                sum += (*A)[k + i*K]*(*B)[k + K*j];
            }
            (*C)[i + j*M] += sum;
        }
    }
}
template<int M, int N, int K> inline void smallTmultTo(double * __restrict A, double * __restrict B, double * __restrict C){
    smallTmultToInternal<M,N,K>(rcst<K,M>(A), rcst<K,N>(B), rcst<M,N>(C));
}

inline void mult(const Matrix &M1, const Matrix &M2, Matrix &M3){
    #ifdef MATRIX_DEBUG
    if (M1.n != M2.m) throw std::invalid_argument("mult: Chaining dimension mismatch");
    if (M1.n == 0) throw std::invalid_argument("mult: Chaining dimension zero");
    #endif
    
    if (M3.m != M1.m || M3.n != M2.n) [[unlikely]] M3.Dimension(M1.m, M2.n);
    if (M1.m == 0 || M2.n == 0) [[unlikely]] return;
    if (M1.m > 3 || M1.n > 3 || M2.n > 3 || M1.ldim != M1.m || M2.ldim != M2.m || M3.ldim != M3.m)
        BSQP_BLASFUNC(cblas_dgemm)(CblasColMajor, CblasNoTrans, CblasNoTrans, blasint(M1.m), blasint(M2.n), blasint(M1.n), double(1.0), M1.array, blasint(M1.ldim), M2.array, blasint(M2.ldim), double(0.), M3.array, blasint(M3.ldim));
    else{
        switch (M1.m){
            case 1:
                switch (M2.n){
                    case 1:
                        switch (M1.n){
                            case 1:
                                smallMult<1,1,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallMult<1,1,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallMult<1,1,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                        break;
                    case 2:
                        switch (M1.n){
                            case 1:
                                smallMult<1,2,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallMult<1,2,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallMult<1,2,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                        break;
                    case 3:
                        switch (M1.n){
                            case 1:
                                smallMult<1,3,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallMult<1,3,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallMult<1,3,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                    break;
                }
                break;
            case 2:
                switch (M2.n){
                    case 1:
                        switch (M1.n){
                            case 1:
                                smallMult<2,1,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallMult<2,1,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallMult<2,1,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                        break;
                    case 2:
                        switch (M1.n){
                            case 1:
                                smallMult<2,2,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallMult<2,2,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallMult<2,2,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                        break;
                    case 3:
                        switch (M1.n){
                            case 1:
                                smallMult<2,3,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallMult<2,3,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallMult<2,3,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                        break;
                }
                break;
            case 3:
                switch (M2.n){
                    case 1:
                        switch (M1.n){
                            case 1:
                                smallMult<3,1,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallMult<3,1,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallMult<3,1,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                        break;
                    case 2:
                        switch (M1.n){
                            case 1:
                                smallMult<3,2,1>(M1.array, M2.array, M3.array);
                            break;
                            case 2:
                                smallMult<3,2,2>(M1.array, M2.array, M3.array);
                            break;
                            case 3:
                                smallMult<3,2,3>(M1.array, M2.array, M3.array);
                        }
                        break;
                    case 3:
                        switch (M1.n){
                            case 1:
                                smallMult<3,3,1>(M1.array, M2.array, M3.array);
                            break;
                            case 2:
                                smallMult<3,3,2>(M1.array, M2.array, M3.array);
                            break;
                            case 3:
                                smallMult<3,3,3>(M1.array, M2.array, M3.array);
                        }
                        break;
                }
                break;
        }
    }
}


inline void Tmult(const Matrix &M1, const Matrix &M2, Matrix &M3){
    #ifdef MATRIX_DEBUG
    if (M1.m != M2.m) throw std::invalid_argument("Tmult: Chaining dimension mismatch");
    if (M1.m == 0) throw std::invalid_argument("Tmult: Chaining dimension zero");
    #endif
  
    if (M3.m != M1.n || M3.n != M2.n) [[unlikely]] M3.Dimension(M1.n, M2.n);
    if (M1.n == 0 || M2.n == 0) [[unlikely]] return; 
    if (M1.n > 3 || M1.m > 3 || M2.n > 3 || M1.ldim != M1.m || M2.ldim != M2.m || M3.ldim != M3.m)
        BSQP_BLASFUNC(cblas_dgemm)(CblasColMajor, CblasTrans, CblasNoTrans, blasint(M1.n), blasint(M2.n), blasint(M1.m), double(1.0), M1.array, blasint(M1.ldim), M2.array, blasint(M2.ldim), double(0.), M3.array, blasint(M3.ldim));
    else{
        switch (M1.n){
            case 1:
                switch (M2.n){
                    case 1:
                        switch (M1.m){
                            case 1:
                                smallTmult<1,1,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallTmult<1,1,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallTmult<1,1,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                        break;
                    case 2:
                        switch (M1.m){
                            case 1:
                                smallTmult<1,2,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallTmult<1,2,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallTmult<1,2,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                        break;
                    case 3:
                        switch (M1.m){
                            case 1:
                                smallTmult<1,3,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallTmult<1,3,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallTmult<1,3,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                    break;
                }
                break;
            case 2:
                switch (M2.n){
                    case 1:
                        switch (M1.m){
                            case 1:
                                smallTmult<2,1,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallTmult<2,1,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallTmult<2,1,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                        break;
                    case 2:
                        switch (M1.m){
                            case 1:
                                smallTmult<2,2,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallTmult<2,2,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallTmult<2,2,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                        break;
                    case 3:
                        switch (M1.m){
                            case 1:
                                smallTmult<2,3,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallTmult<2,3,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallTmult<2,3,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                        break;
                }
                break;
            case 3:
                switch (M2.n){
                    case 1:
                        switch (M1.m){
                            case 1:
                                smallTmult<3,1,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallTmult<3,1,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallTmult<3,1,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                        break;
                    case 2:
                        switch (M1.m){
                            case 1:
                                smallTmult<3,2,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallTmult<3,2,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallTmult<3,2,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                        break;
                    case 3:
                        switch (M1.m){
                            case 1:
                                smallTmult<3,3,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallTmult<3,3,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallTmult<3,3,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                    break;
                }
                break;
        }
    }
}


inline void mult_to(const Matrix &M1, const Matrix &M2, const Matrix &M3, double beta){
    #ifdef MATRIX_DEBUG
      if (M1.n != M2.m) throw std::invalid_argument("mult_to: Chaining dimension mismatch");
      if (M1.n == 0) throw std::invalid_argument("mult_to: Chaining dimension zero");
      if (M3.m != M1.m || M3.n != M2.n) throw std::invalid_argument("mult_to: Output matrix dimension mismatch");
    #endif
    if (M1.m == 0 || M2.n == 0) [[unlikely]] return;
    if (M1.m > 3 || M1.n > 3 || M2.n > 3 || M1.ldim != M1.m || M2.ldim != M2.m || M3.ldim != M3.m)
        BSQP_BLASFUNC(cblas_dgemm)(CblasColMajor, CblasNoTrans, CblasNoTrans, blasint(M1.m), blasint(M2.n), blasint(M1.n), double(1.0), M1.array, blasint(M1.ldim), M2.array, blasint(M2.ldim), beta, M3.array, blasint(M3.ldim));
    else{
        M3 *= beta; //NOTE: M3 may be uninitialized. Matrix::operator*= forwards to Matrix::Initialize in case beta is zero, so its ok.
        switch (M1.m){
            case 1:
                switch (M2.n){
                    case 1:
                        switch (M1.n){
                            case 1:
                                smallMultTo<1,1,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallMultTo<1,1,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallMultTo<1,1,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                        break;
                    case 2:
                        switch (M1.n){
                            case 1:
                                smallMultTo<1,2,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallMultTo<1,2,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallMultTo<1,2,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                        break;
                    case 3:
                        switch (M1.n){
                            case 1:
                                smallMultTo<1,3,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallMultTo<1,3,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallMultTo<1,3,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                        break;
                }
                break;
            case 2:
                switch (M2.n){
                    case 1:
                        switch (M1.n){
                            case 1:
                                smallMultTo<2,1,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallMultTo<2,1,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallMultTo<2,1,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                        break;
                    case 2:
                        switch (M1.n){
                            case 1:
                                smallMultTo<2,2,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallMultTo<2,2,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallMultTo<2,2,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                        break;
                    case 3:
                        switch (M1.n){
                            case 1:
                                smallMultTo<2,3,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallMultTo<2,3,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallMultTo<2,3,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                        break;
                }
                break;
            case 3:
                switch (M2.n){
                    case 1:
                        switch (M1.n){
                            case 1:
                                smallMultTo<3,1,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallMultTo<3,1,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallMultTo<3,1,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                        break;
                    case 2:
                        switch (M1.n){
                            case 1:
                                smallMultTo<3,2,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallMultTo<3,2,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallMultTo<3,2,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                        break;
                    case 3:
                        switch (M1.n){
                            case 1:
                                smallMultTo<3,3,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallMultTo<3,3,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallMultTo<3,3,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                    break;
                }
                break;
        }
    }
}


inline void Tmult_to(const Matrix &M1, const Matrix &M2, const Matrix &M3, double beta){
    #ifdef MATRIX_DEBUG
      if (M1.m != M2.m) throw std::invalid_argument("Tmult_to: Chaining dimension mismatch");
      if (M1.m == 0) throw std::invalid_argument("Tmult_to: Chaining dimension zero");
      if (M3.m != M1.n || M3.n != M2.n) throw std::invalid_argument("Tmult_to: Output matrix dimension mismatch");
    #endif
    if (M1.n == 0 || M2.n == 0) [[unlikely]] return;
    if (M1.n > 3 || M1.m > 3 || M2.n > 3 || M1.ldim != M1.m || M2.ldim != M2.m || M3.ldim != M3.m)
    // if (true)
        BSQP_BLASFUNC(cblas_dgemm)(CblasColMajor, CblasTrans, CblasNoTrans, blasint(M1.n), blasint(M2.n), blasint(M1.m), double(1.0), M1.array, blasint(M1.ldim), M2.array, blasint(M2.ldim), beta, M3.array, blasint(M3.ldim));
    else{
        M3 *= beta; //NOTE: M3 may be uninitialized. Matrix::operator*= forwards to Matrix::Initialize in case beta is zero, so its ok.
        switch (M1.n){
            case 1:
                switch (M2.n){
                    case 1:
                        switch (M1.m){
                            case 1:
                                smallTmultTo<1,1,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallTmultTo<1,1,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallTmultTo<1,1,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                        break;
                    case 2:
                        switch (M1.m){
                            case 1:
                                smallTmultTo<1,2,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallTmultTo<1,2,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallTmultTo<1,2,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                        break;
                    case 3:
                        switch (M1.m){
                            case 1:
                                smallTmultTo<1,3,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallTmultTo<1,3,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallTmultTo<1,3,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                        break;
                }
                break;
            case 2:
                switch (M2.n){
                    case 1:
                        switch (M1.m){
                            case 1:
                                smallTmultTo<2,1,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallTmultTo<2,1,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallTmultTo<2,1,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                        break;
                    case 2:
                        switch (M1.m){
                            case 1:
                                smallTmultTo<2,2,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallTmultTo<2,2,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallTmultTo<2,2,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                        break;
                    case 3:
                        switch (M1.m){
                            case 1:
                                smallTmultTo<2,3,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallTmultTo<2,3,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallTmultTo<2,3,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                        break;
                }
                break;
            case 3:
                switch (M2.n){
                    case 1:
                        switch (M1.m){
                            case 1:
                                smallTmultTo<3,1,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallTmultTo<3,1,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallTmultTo<3,1,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                        break;
                    case 2:
                        switch (M1.m){
                            case 1:
                                smallTmultTo<3,2,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallTmultTo<3,2,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallTmultTo<3,2,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                        break;
                    case 3:
                        switch (M1.m){
                            case 1:
                                smallTmultTo<3,3,1>(M1.array, M2.array, M3.array);
                                break;
                            case 2:
                                smallTmultTo<3,3,2>(M1.array, M2.array, M3.array);
                                break;
                            case 3:
                                smallTmultTo<3,3,3>(M1.array, M2.array, M3.array);
                                break;
                        }
                        break;
                }
                break;
        }
    }
}

#endif //elif defined(BSQP_SMALLMULT)



inline double &SymMatrix::operator()(int i, int j) const {
  #ifdef MATRIX_DEBUG
    if (i < 0 || i >= m || j < 0 || j >= m) throw std::invalid_argument("SymMatrix::operator(): Indices (" + std::to_string(i) + ", " + std::to_string(j) + ") out of bounds for matrix of shape (" + std::to_string(m) + ", " + std::to_string(m) + ")");
  #endif
  if (i < j) return array[j + i*ldim - (i*(i + 1))/2];
  return array[i + j*ldim - (j*(j + 1))/2];
}
        
inline double &SymMatrix::operator()(int i) const {
  #ifdef MATRIX_DEBUG
    if (i >= m*(m+1)/2.0) throw std::invalid_argument("SymMatrix operator(): Index " + std::to_string(i) + " out of bounds for SymMatrix of size " + std::to_string(m));
  #endif
  return array[i];
}
        
inline SymMatrix &SymMatrix::Dimension(int M){
    if (m != M){
    deallocate();
    m = M;
    ldim = M;
    allocate();
    }
    return *this;
}