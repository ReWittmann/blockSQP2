/**
 * \file py_blockSQP.cpp
 * \author Reinhold Wittmann
 * \date 2022-
 *
 * Pybind11 based python interface for the blockSQP nonlinear programming solver
 */


#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <tuple>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include "blocksqp_options.hpp"
#include "blocksqp_method.hpp"
#include "blocksqp_condensing.hpp"
#include "qpOASES.hpp"
#include "blocksqp_restoration.hpp"

namespace py = pybind11;

class int_pointer_interface{
    public:
    int size;
    int *ptr;

    public:
    int_pointer_interface(): size(0), ptr(nullptr){}
    int_pointer_interface(int *ptr_, int size_): size(size_), ptr(ptr_){} //Causes linker warning -Walloc-size-larger-than=
    ~int_pointer_interface(){}
};

class double_pointer_interface{
    public:
    int size;
    double *ptr;

    public:
    double_pointer_interface(): size(0), ptr(nullptr){}
    double_pointer_interface(double *ptr_, int size_): size(size_), ptr(ptr_){} //Causes linker warning -Walloc-size-larger-than=
    ~double_pointer_interface(){}
};


template <typename T> class T_array{
    public:
    int size;
    T *ptr;

    public:
    T_array(): size(0), ptr(nullptr){}
    T_array(int size_): size(size_), ptr(new T[size_]){} //Causes linker warning -Walloc-size-larger-than=

    ~T_array(){
        delete[] ptr;
    }

    void resize(int size_){
        delete[] ptr;
        size = size_;
        ptr = new T[size];
    }
};

typedef T_array<int> int_array;
typedef T_array<double> double_array;
typedef T_array<double_pointer_interface> double_pointer_interface_array;
typedef T_array<blockSQP::vblock> vblock_array;
typedef T_array<blockSQP::cblock> cblock_array;
typedef T_array<blockSQP::condensing_target> condensing_targets;
typedef T_array<blockSQP::SymMatrix> SymMat_array;


class condensing_args{
public:
    blockSQP::Matrix grad_obj;
    blockSQP::Sparse_Matrix con_jac;
    SymMat_array hess;
    blockSQP::Matrix lb_var;
    blockSQP::Matrix ub_var;
    blockSQP::Matrix lb_con;
    blockSQP::Matrix ub_con;

    SymMat_array condensed_hess;
    blockSQP::Sparse_Matrix condensed_Jacobian;
    blockSQP::Matrix condensed_h;
    blockSQP::Matrix condensed_lb_var;
    blockSQP::Matrix condensed_ub_var;
    blockSQP::Matrix condensed_lb_con;
    blockSQP::Matrix condensed_ub_con;

    blockSQP::Matrix delta_Xi;
    blockSQP::Matrix delta_Lambda;

    blockSQP::Matrix delta_Xi_cond;
    blockSQP::Matrix delta_Lambda_cond;
    blockSQP::Matrix delta_Xi_rest;
    blockSQP::Matrix delta_Lambda_rest;

    blockSQP::Condenser *C;


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

                for (j = 0; j < nRows; j++){
                    if ((hess_[iBlock](i, j) > eps) || (-hess_[iBlock](i, j) > eps)){
                        hessNz_[count] = hess_[iBlock](i, j);
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
        for (j = 0; j < nVar; j++){
            for (i = hessIndCol_[j]; i < hessIndCol_[j+1] && hessIndRow_[i] < j; i++);
            hessIndLo_[j] = i;
        }

        if( count != nnz ){
             std::cout << "Error in convertHessian: " << count << " elements processed, should be " << nnz << " elements!\n";
        }
    }


    void solve_QPs(){
        double a = 0;
        for (int i = 0; i < condensed_Jacobian.colind[condensed_Jacobian.n]; i++){
            a += condensed_Jacobian.nz[i];
        }
        std::cout << "Sum on nonzeros in condensed_Jacobian is " << a << "\n";

        blockSQP::Sparse_Matrix red_con_jac = condensed_Jacobian;
        blockSQP::Matrix red_lb_con = condensed_lb_con;
        blockSQP::Matrix red_ub_con = condensed_ub_con;

        std::cout << "con_jac.nnz = " << con_jac.colind[con_jac.n] << ", con_jac.m = " << con_jac.m << ", con_jac.n = " << con_jac.n << "\n";

        qpOASES::SQProblem* qp;
        qpOASES::SQProblem* qp_cond;
        qpOASES::returnValue ret;
        qpOASES::returnValue ret_cond;

        qpOASES::Matrix *A_qp;
        qpOASES::Matrix *A_qp_cond;
        qpOASES::SymmetricMatrix *H;
        qpOASES::SymmetricMatrix *H_cond;

        double *hess_nz = nullptr;
        int *hess_row = nullptr;
        int *hess_colind = nullptr;
        int *hess_loind = nullptr;

        double *hess_cond_nz = nullptr;
        int *hess_cond_row = nullptr;
        int *hess_cond_colind = nullptr;
        int *hess_cond_loind = nullptr;

        qp = new qpOASES::SQProblemSchur( con_jac.n, con_jac.m, qpOASES::HST_UNKNOWN, 50 );
        qp_cond = new qpOASES::SQProblemSchur( red_con_jac.n, red_con_jac.m, qpOASES::HST_UNKNOWN, 50 );

        A_qp = new qpOASES::SparseMatrix(con_jac.m, con_jac.n,
                    con_jac.row.get(), con_jac.colind.get(), con_jac.nz.get());
        A_qp_cond = new qpOASES::SparseMatrix(red_con_jac.m, red_con_jac.n,
                    red_con_jac.row.get(), red_con_jac.colind.get(), red_con_jac.nz.get());


        convertHessian(1.0e-15, hess.ptr, hess.size, con_jac.n, hess_nz, hess_row, hess_colind, hess_loind);
        convertHessian(1.0e-15, condensed_hess.ptr, C->condensed_num_hessblocks, red_con_jac.n, hess_cond_nz, hess_cond_row, hess_cond_colind, hess_cond_loind);
        std::cout << "converted Hessians\n";


        H = new qpOASES::SymSparseMat(con_jac.n, con_jac.n, hess_row, hess_colind, hess_nz);
        dynamic_cast<qpOASES::SymSparseMat*>(H)->createDiagInfo();
        H_cond = new qpOASES::SymSparseMat(red_con_jac.n, red_con_jac.n, hess_cond_row, hess_cond_colind, hess_cond_nz);
        dynamic_cast<qpOASES::SymSparseMat*>(H_cond)->createDiagInfo();


        double *g = grad_obj.array;
        double *lb = lb_var.array;
        double *ub = ub_var.array;
        double *lbA = lb_con.array;
        double *ubA = ub_con.array;
        double cpu_time = 600;
        int max_it = 10000000;

        double *g_cond = condensed_h.array;
        double *lb_cond = condensed_lb_var.array;
        double *ub_cond = condensed_ub_var.array;
        double *lbA_cond = red_lb_con.array;
        double *ubA_cond = red_ub_con.array;
        double cpu_time_cond = 600;
        int max_it_cond = 10000000;

        qpOASES::Options opts;
        opts.enableInertiaCorrection = qpOASES::BT_FALSE;
        opts.enableEqualities = qpOASES::BT_TRUE;
        opts.initialStatusBounds = qpOASES::ST_INACTIVE;
        opts.printLevel = qpOASES::PL_NONE; //PL_LOW, PL_HIGH, PL_MEDIUM, PL_None
        opts.numRefinementSteps = 2;
        opts.epsLITests =  2.2204e-08;
        qp->setOptions(opts);
        qp_cond->setOptions(opts);


        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        ret = qp->init(H, g, A_qp, lb, ub, lbA, ubA, max_it, &cpu_time);
        std::cout << "Solver of uncondensed QP returned, ret is " << ret << "\n";

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Finished solution of uncondensed QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms\n";


        begin = std::chrono::steady_clock::now();

        ret_cond = qp_cond->init(H_cond, g_cond, A_qp_cond, lb_cond, ub_cond, lbA_cond, ubA_cond, max_it_cond, &cpu_time_cond);
        std::cout << "Solver of condensed QP returned, ret is " << ret_cond << "\n";

        end = std::chrono::steady_clock::now();
        std::cout << "Finished solution of condensed QP in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms\n";


        delta_Xi.Dimension(con_jac.n, 1);
        delta_Xi.Initialize(0.);
        delta_Lambda.Dimension(con_jac.n + con_jac.m, 1);
        delta_Lambda.Initialize(0.);

        delta_Xi_cond.Dimension(red_con_jac.n, 1);
        delta_Xi_cond.Initialize(0.);
        delta_Lambda_cond.Dimension(red_con_jac.n + red_con_jac.m,1);
        delta_Lambda_cond.Initialize(0.);

        qp->getPrimalSolution(delta_Xi.array);
        qp->getDualSolution(delta_Lambda.array);

        qp_cond->getPrimalSolution(delta_Xi_cond.array);
        qp_cond->getDualSolution(delta_Lambda_cond.array);

        C->recover_var_mult(delta_Xi_cond, delta_Lambda_cond, delta_Xi_rest, delta_Lambda_rest);

        delete qp;
        delete qp_cond;
        delete A_qp;
        delete A_qp_cond;
        delete H;
        delete H_cond;
        delete[] hess_nz;
        delete[] hess_row;
        delete[] hess_cond_nz;
        delete[] hess_cond_row;
    }
};



struct Prob_Data{
    double_pointer_interface xi;			///< optimization variables
    double_pointer_interface lambda;		///< Lagrange multipliers
    double objval;				            ///< objective function value
    double_pointer_interface constr;		///< constraint function values
    double_pointer_interface gradObj;		///< gradient of objective

    blockSQP::Matrix constrJac;
    double_pointer_interface jacNz;		    ///< nonzero elements of constraint Jacobian
    int_pointer_interface jacIndRow;		///< row indices of nonzero elements
	int_pointer_interface jacIndCol;		///< starting indices of columns

    //Each hessian blocks elements are a double array, wrapper by double_pointer_interface
    //These are the once again wrapped by doubel_pointer_interface_interface
    double_pointer_interface_array hess_arr;
    int dmode;				                ///< derivative mode
    int info;				                ///< error flag
};


class Problemform : public blockSQP::Problemspec
{
public:
    Problemform(){}

    virtual ~Problemform(){
        delete[] blockIdx;
        delete[] vblocks;
    };

    Prob_Data Cpp_Data; //values that get evaluated and returned by the evaluate methods

    void init_Cpp_Data(bool Sparse_QP, int nnz){
        Cpp_Data.xi.size = nVar;
        Cpp_Data.lambda.size = nVar + nCon;
        Cpp_Data.gradObj.size = nVar;
        Cpp_Data.constr.size = nCon;

        if (Sparse_QP){
            Cpp_Data.jacNz.size = nnz;
            Cpp_Data.jacIndRow.size = nnz;
            Cpp_Data.jacIndCol.size = nVar + 1;
        }
        else{
            //Cpp_Data.constrJac.size = nCon * nVar;
            Cpp_Data.constrJac.m = nCon;
            Cpp_Data.constrJac.n = nVar;
            Cpp_Data.constrJac.ldim = -1;
            Cpp_Data.constrJac.tflag = 1;
        }

        Cpp_Data.hess_arr.size = nBlocks;
        double_pointer_interface *h_arrays = new double_pointer_interface[nBlocks];
        for (int j = 0; j < nBlocks; j++){
            h_arrays[j].size = ((blockIdx[j+1] - blockIdx[j]) * (blockIdx[j+1] - blockIdx[j] + 1))/2 ;
        }
        Cpp_Data.hess_arr.ptr = h_arrays;
        Cpp_Data.info = 0;
    }


    //Methods to be implemented on python side:
    virtual void initialize_dense(){}; //initialize Cpp_Data (dense jacobian)
    virtual void initialize_sparse(){}; //initialize Cpp_Data (sparse jacobian)
    virtual void evaluate_dense(){}; //evaluate and write Cpp_Data (dense jacobian)
    virtual void evaluate_sparse(){}; //evaluate and write Cpp_Data (sparse jacobian)
    virtual void evaluate_simple(){}; //evaluate and write Cpp_Data (no derivatives)

    virtual void update_inits(){};
    virtual void update_evals(){};
    virtual void update_simple(){};
    virtual void update_xi(){};

    virtual void get_objval(){};

    virtual void restore_continuity(){};


    void initialize(blockSQP::Matrix &xi, blockSQP::Matrix &lambda, blockSQP::Matrix &constrJac) override {
        Cpp_Data.xi.ptr = xi.array;
        Cpp_Data.lambda.ptr = lambda.array;
        Cpp_Data.constrJac.array = constrJac.array;

        update_inits();
        initialize_dense();
    }


    void initialize(blockSQP::Matrix &xi, blockSQP::Matrix &lambda, double *jacNz, int *jacIndRow, int *jacIndCol) override {
        Cpp_Data.xi.ptr = xi.array;
        Cpp_Data.lambda.ptr = lambda.array;
        Cpp_Data.jacIndRow.ptr = jacIndRow;
        Cpp_Data.jacIndCol.ptr = jacIndCol;

        update_inits();
        initialize_sparse();
    }


    void evaluate(const blockSQP::Matrix &xi, const blockSQP::Matrix &lambda, 
            double *objval, blockSQP::Matrix &constr, blockSQP::Matrix &gradObj, blockSQP::Matrix &constrJac,
            blockSQP::SymMatrix *hess, int dmode, int *info) override {
        Cpp_Data.xi.ptr = xi.array;
        Cpp_Data.lambda.ptr = lambda.array;
        Cpp_Data.dmode = dmode;
        Cpp_Data.constr.ptr = constr.array;
        Cpp_Data.gradObj.ptr = gradObj.array;
        Cpp_Data.constrJac.array = constrJac.array;

        if (dmode == 3){
            for (int j = 0; j < nBlocks; j++)
                Cpp_Data.hess_arr.ptr[j].ptr = hess[j].array;
        }
        else if (dmode == 2){
            Cpp_Data.hess_arr.ptr[nBlocks - 1].ptr = hess[nBlocks - 1].array;
        }

        update_evals();
        evaluate_dense();
        get_objval();

        *objval = Cpp_Data.objval;
        *info = 0;
    }


    void evaluate(const blockSQP::Matrix &xi, const blockSQP::Matrix &lambda, double *objval, blockSQP::Matrix &constr, 
                    blockSQP::Matrix &gradObj, double *jacNz, int *jacIndRow, int *jacIndCol, blockSQP::SymMatrix *hess, 
                        int dmode, int *info) override {
        Cpp_Data.xi.ptr = xi.array;
        Cpp_Data.lambda.ptr = lambda.array;
        Cpp_Data.dmode = dmode;
        Cpp_Data.constr.ptr = constr.array;
        Cpp_Data.gradObj.ptr = gradObj.array;
        Cpp_Data.jacNz.ptr = jacNz;
        Cpp_Data.jacIndRow.ptr = jacIndRow;
        Cpp_Data.jacIndCol.ptr = jacIndCol;

        if (dmode == 3){
            for (int j = 0; j < nBlocks; j++)
                Cpp_Data.hess_arr.ptr[j].ptr = hess[j].array;
        }
        else if (dmode == 2){
            Cpp_Data.hess_arr.ptr[nBlocks - 1].ptr = hess[nBlocks - 1].array;
        }

        update_evals();
        evaluate_sparse();
        get_objval();

        *objval = Cpp_Data.objval;
        *info = 0;
    }


    void evaluate(const blockSQP::Matrix &xi, double *objval, blockSQP::Matrix &constr, int *info) override {
        Cpp_Data.xi.ptr = xi.array;
        Cpp_Data.constr.ptr = constr.array;

        update_simple();
        evaluate_simple();
        get_objval();

        *objval = Cpp_Data.objval;
        *info = 0;
    }


    void set_blockIdx(py::array_t<int> arr){
        py::buffer_info buff = arr.request();
        nBlocks = buff.size - 1;
        blockIdx = new int[buff.size];
        std::copy((int*)buff.ptr, (int*)buff.ptr + buff.size, blockIdx);
    }

    void set_vblocks(vblock_array &VB){
        n_vblocks = VB.size;
        vblocks = new blockSQP::vblock[n_vblocks];
        for (int i = 0; i < n_vblocks; i++){
            vblocks[i] = VB.ptr[i];
        }
    }

    void reduceConstrVio(blockSQP::Matrix &xi, int *info) override {
        Cpp_Data.xi.ptr = xi.array;
        update_xi();
        restore_continuity();
        *info = Cpp_Data.info;
    }
};


class Py_Problemform: public Problemform{
    void initialize_dense() override {
        PYBIND11_OVERRIDE(void, Problemform, initialize_dense,);
    }

    void initialize_sparse() override {
        PYBIND11_OVERRIDE(void, Problemform, initialize_sparse,);
    }

    void evaluate_dense() override {
        PYBIND11_OVERRIDE(void, Problemform, evaluate_dense,);
    }

    void evaluate_sparse() override {
        PYBIND11_OVERRIDE(void, Problemform, evaluate_sparse,);
    }

    void evaluate_simple() override {
        PYBIND11_OVERRIDE(void, Problemform, evaluate_simple,);
    }

    void update_inits() override {
        PYBIND11_OVERRIDE(void, Problemform, update_inits,);
    }
    void update_evals() override {
        PYBIND11_OVERRIDE(void, Problemform, update_evals,);
    }
    void update_simple() override {
        PYBIND11_OVERRIDE(void, Problemform, update_simple,);
    }

    void update_xi() override {
        PYBIND11_OVERRIDE(void, Problemform, update_xi,);
    }

    void get_objval() override {
        PYBIND11_OVERRIDE(void, Problemform, get_objval,);
    }

    void restore_continuity() override {
        PYBIND11_OVERRIDE(void, Problemform, restore_continuity,);
    }
};



PYBIND11_MODULE(py_blockSQP, m){

py::class_<blockSQP::Matrix>(m, "Matrix", py::buffer_protocol())
	.def_buffer([](blockSQP::Matrix &mtrx) -> py::buffer_info{
		return py::buffer_info(
			mtrx.array,
			sizeof(double),
			py::format_descriptor<double>::format(),
			2,
			{mtrx.m,mtrx.n},
			{sizeof(double), sizeof(double)*mtrx.m}
			);
		})
    .def(py::init<>())
	.def(py::init<int,int,int>(), py::arg("M"), py::arg("N") = 1, py::arg("LDIM") = -1)
	.def(py::init<const blockSQP::Matrix&>())
	.def(py::init<const blockSQP::SymMatrix&>())
	.def("Dimension", &blockSQP::Matrix::Dimension, py::arg("M"), py::arg("N") = 1, py::arg("LDIM") = -1)
	.def("Initialize", static_cast<blockSQP::Matrix& (blockSQP::Matrix::*)(double)>(&blockSQP::Matrix::Initialize))
	.def_readwrite("ldim",&blockSQP::Matrix::ldim)
	.def_readwrite("m",&blockSQP::Matrix::m)
	.def_readwrite("n",&blockSQP::Matrix::n)
    .def("__setitem__", [](blockSQP::Matrix &M, std::tuple<int, int> inds, double val) -> void{M(std::get<0>(inds), std::get<1>(inds)) = val; return;})
	.def("__getitem__", [](blockSQP::Matrix &M, std::tuple<int, int> inds) -> double{return M(std::get<0>(inds), std::get<1>(inds));})
	.def("__setitem__", [](blockSQP::Matrix &M, int ind, double val) -> void{M(ind) = val; return;})
	.def("__getitem__", [](blockSQP::Matrix &M, int ind) -> double{return M(ind);})
	.def_property("array", nullptr, [](blockSQP::Matrix &mtrx, py::array_t<double> arr){
		py::buffer_info buff = arr.request();
		mtrx.array = (double*)buff.ptr;
		});

py::class_<blockSQP::SymMatrix>(m, "SymMatrix")
    .def(py::init<>())
    .def(py::init<int>(), py::arg("M") = 1)
    .def(py::init<const blockSQP::Matrix&>())
	.def("Dimension", [](blockSQP::SymMatrix &M1, int m) -> void{M1.Dimension(m);})
	.def("Initialize", static_cast<blockSQP::SymMatrix & (blockSQP::SymMatrix::*)(double)>(&blockSQP::SymMatrix::Initialize))
	.def_readwrite("ldim",&blockSQP::SymMatrix::ldim)
	.def_readwrite("m",&blockSQP::SymMatrix::m)
	.def("__setitem__", [](blockSQP::SymMatrix &M, std::tuple<int, int> inds, double val) -> void{M(std::get<0>(inds), std::get<1>(inds)) = val; return;})
	.def("__getitem__", [](blockSQP::SymMatrix &M, std::tuple<int, int> inds) -> double{return M(std::get<0>(inds), std::get<1>(inds));})
    .def("set", [](blockSQP::SymMatrix &M1, int i, int j, const blockSQP::Matrix &M2) -> void{
        M1.Dimension(M2.m);
        for (int i = 0; i < M2.m; i++){
            for (int j = 0; j <= i; j++){
                M1(i,j) = M1(i,j);
            }
        }
        return;
    })
    .def("tril", [](blockSQP::SymMatrix &M)->double_pointer_interface{return double_pointer_interface(M.array, (M.m*(M.m + 1))/2);})
    ;

py::class_<blockSQP::Sparse_Matrix>(m, "Sparse_Matrix")
    .def(py::init<>())
    .def(py::init([](int M, int N, double_array &nz, int_array &row, int_array &colind) -> blockSQP::Sparse_Matrix*{
        std::unique_ptr<double[]> NZ = std::unique_ptr<double[]>(nz.ptr); std::unique_ptr<int[]> ROW = std::unique_ptr<int[]>(row.ptr); std::unique_ptr<int[]> COLIND = std::unique_ptr<int[]>(colind.ptr);
        nz.size = 0; nz.ptr = nullptr; row.size = 0; row.ptr = nullptr; colind.size = 0; colind.ptr = nullptr;
        return new blockSQP::Sparse_Matrix(M, N, std::move(NZ), std::move(ROW), std::move(COLIND));
    }), py::return_value_policy::take_ownership)
    .def_readonly("m", &blockSQP::Sparse_Matrix::m)
    .def_readonly("n", &blockSQP::Sparse_Matrix::n)
    .def("dense", &blockSQP::Sparse_Matrix::dense)
    .def_property("NZ", [](blockSQP::Sparse_Matrix &M)->double_pointer_interface{double_pointer_interface nonzeros; nonzeros.size = M.colind[M.n]; nonzeros.ptr = M.nz.get(); return nonzeros;}, nullptr)
    .def_property("ROW", [](blockSQP::Sparse_Matrix &M)->int_pointer_interface{int_pointer_interface row; row.size = M.colind[M.n]; row.ptr = M.row.get(); return row;}, nullptr)
    .def_property("COLIND", [](blockSQP::Sparse_Matrix &M)->int_pointer_interface{int_pointer_interface colind; colind.size = M.n + 1; colind.ptr = M.colind.get(); return colind;}, nullptr)
    ;

py::class_<blockSQP::SQPoptions>(m, "SQPoptions")
	.def(py::init<>())
	.def("optionsConsistency", static_cast<void (blockSQP::SQPoptions::*)()>(&blockSQP::SQPoptions::optionsConsistency))
    .def("optionsConsistency", static_cast<void (blockSQP::SQPoptions::*)(blockSQP::Problemspec*)>(&blockSQP::SQPoptions::optionsConsistency))
    .def("reset", &blockSQP::SQPoptions::reset)
	.def_readwrite("print_level",&blockSQP::SQPoptions::print_level)
	.def_readwrite("result_print_color",&blockSQP::SQPoptions::result_print_color)
	.def_readwrite("debug_level",&blockSQP::SQPoptions::debug_level)
	.def_readwrite("eps",&blockSQP::SQPoptions::eps)
	.def_readwrite("inf",&blockSQP::SQPoptions::inf)
	.def_readwrite("opt_tol",&blockSQP::SQPoptions::opt_tol)
	.def_readwrite("feas_tol",&blockSQP::SQPoptions::feas_tol)
	.def_readwrite("sparse",&blockSQP::SQPoptions::sparse)
	.def_readwrite("enable_linesearch",&blockSQP::SQPoptions::enable_linesearch)
	.def_readwrite("enable_rest",&blockSQP::SQPoptions::enable_rest)
	.def_readwrite("max_linesearch_steps",&blockSQP::SQPoptions::max_linesearch_steps)
	.def_readwrite("max_consec_reduced_steps",&blockSQP::SQPoptions::max_consec_reduced_steps)
	.def_readwrite("max_consec_skipped_updates",&blockSQP::SQPoptions::max_consec_skipped_updates)
	.def_readwrite("max_QP_it",&blockSQP::SQPoptions::max_QP_it)
	.def_readwrite("block_hess",&blockSQP::SQPoptions::block_hess)
	.def_readwrite("sizing",&blockSQP::SQPoptions::sizing)
	.def_readwrite("fallback_sizing",&blockSQP::SQPoptions::fallback_sizing)
	.def_readwrite("max_QP_secs",&blockSQP::SQPoptions::max_QP_secs)
	.def_readwrite("initial_hess_scale",&blockSQP::SQPoptions::initial_hess_scale)
	.def_readwrite("COL_eps",&blockSQP::SQPoptions::COL_eps)
	.def_readwrite("OL_eps", &blockSQP::SQPoptions::OL_eps)
	.def_readwrite("COL_tau_1",&blockSQP::SQPoptions::COL_tau_1)
	.def_readwrite("COL_tau_2",&blockSQP::SQPoptions::COL_tau_2)
	.def_readwrite("min_damping_quotient",&blockSQP::SQPoptions::min_damping_quotient)
	.def_readwrite("SR1_abstol",&blockSQP::SQPoptions::SR1_abstol)
	.def_readwrite("SR1_reltol",&blockSQP::SQPoptions::SR1_reltol)
	.def_readwrite("BFGS_damping_factor",&blockSQP::SQPoptions::BFGS_damping_factor)
	.def_readwrite("hess_approx",&blockSQP::SQPoptions::hess_approx)
	.def_readwrite("fallback_approx",&blockSQP::SQPoptions::fallback_approx)
	.def_readwrite("indef_local_only", &blockSQP::SQPoptions::indef_local_only)
	.def_readwrite("lim_mem",&blockSQP::SQPoptions::lim_mem)
	.def_readwrite("mem_size",&blockSQP::SQPoptions::mem_size)
	.def_readwrite("exact_hess",&blockSQP::SQPoptions::exact_hess)
	.def_readwrite("skip_first_linesearch",&blockSQP::SQPoptions::skip_first_linesearch)
	.def_readwrite("conv_strategy",&blockSQP::SQPoptions::conv_strategy)
	.def_readwrite("max_conv_QPs",&blockSQP::SQPoptions::max_conv_QPs)
	.def_readwrite("reg_factor", &blockSQP::SQPoptions::reg_factor)
	.def_readwrite("max_SOC",&blockSQP::SQPoptions::max_SOC)
	.def_readwrite("max_bound_refines", &blockSQP::SQPoptions::max_bound_refines)
	.def_readwrite("max_correction_steps", &blockSQP::SQPoptions::max_correction_steps)
	.def_readwrite("dep_bound_tolerance", &blockSQP::SQPoptions::dep_bound_tolerance)
	.def_readwrite("max_filter_overrides", &blockSQP::SQPoptions::max_filter_overrides)
	.def_readwrite("conv_tau_H", &blockSQP::SQPoptions::conv_tau_H)
	.def_readwrite("conv_kappa_0", &blockSQP::SQPoptions::conv_kappa_0)
    .def_readwrite("conv_kappa_max", &blockSQP::SQPoptions::conv_kappa_max)
	.def_readwrite("rest_zeta", &blockSQP::SQPoptions::rest_zeta)
	.def_readwrite("rest_rho", &blockSQP::SQPoptions::rest_rho)
    .def_readwrite("automatic_scaling", &blockSQP::SQPoptions::automatic_scaling)

    .def_readwrite("kappaF", &blockSQP::SQPoptions::kappaF)
    .def_readwrite("max_extra_steps", &blockSQP::SQPoptions::max_extra_steps)
    .def_readwrite("enable_premature_termination", &blockSQP::SQPoptions::enable_premature_termination)
	.def_property("qpsol", [](blockSQP::SQPoptions &opts)->std::string{
        if (opts.qpsol == blockSQP::QPsolvers::qpOASES) return "qpOASES";
        else if (opts.qpsol == blockSQP::QPsolvers::gurobi) return "gurobi";
        else if (opts.qpsol == blockSQP::QPsolvers::qpalm) return "qpalm";
        return "unset";
        },
    [](blockSQP::SQPoptions &opts, std::string &QPsolver_name){
        if (QPsolver_name == "qpOASES") opts.qpsol = blockSQP::QPsolvers::qpOASES;
        else if (QPsolver_name == "gurobi") opts.qpsol = blockSQP::QPsolvers::gurobi;
        else if (QPsolver_name  == "qpalm") opts.qpsol = blockSQP::QPsolvers::qpalm;
        else throw blockSQP::ParameterError("Unknown QP solver, known (no neccessarily linked) are qpOASES, gurobi, qpalm");
        return;
    }
    )
    .def_readwrite("qpsol_options", &blockSQP::SQPoptions::qpsol_options)
    .def_readwrite("par_QPs", &blockSQP::SQPoptions::par_QPs)
    .def_readwrite("test_opt_2", &blockSQP::SQPoptions::test_opt_2)
    .def_readwrite("test_join_all", &blockSQP::SQPoptions::test_join_all)
    .def_readwrite("test_qp_hotstart", &blockSQP::SQPoptions::test_qp_hotstart)
	;

py::class_<blockSQP::QPsolver_options>(m, "QPsolver_options");

py::class_<blockSQP::qpOASES_options, blockSQP::QPsolver_options>(m, "qpOASES_options")
    .def(py::init<>())
    .def_readwrite("sparsityLevel", &blockSQP::qpOASES_options::sparsityLevel)
    .def_readwrite("printLevel", &blockSQP::qpOASES_options::printLevel)
    .def_readwrite("terminationTolerance", &blockSQP::qpOASES_options::terminationTolerance)
    ;

py::class_<blockSQP::gurobi_options, blockSQP::QPsolver_options>(m, "gurobi_options")
    .def(py::init<>())
    .def_readwrite("Method", &blockSQP::gurobi_options::Method)
    .def_readwrite("NumericFocus", &blockSQP::gurobi_options::NumericFocus)
    .def_readwrite("OutputFlag", &blockSQP::gurobi_options::OutputFlag)
    .def_readwrite("Presolve", &blockSQP::gurobi_options::Presolve)
    .def_readwrite("Aggregate", &blockSQP::gurobi_options::Aggregate)
    .def_readwrite("BarHomogeneous", &blockSQP::gurobi_options::BarHomogeneous)
    .def_readwrite("OptimalityTol", &blockSQP::gurobi_options::OptimalityTol)
    .def_readwrite("FeasibilityTol", &blockSQP::gurobi_options::FeasibilityTol)
    .def_readwrite("PSDTol", &blockSQP::gurobi_options::PSDTol)
    ;

py::class_<int_array>(m,"int_array",py::buffer_protocol())
	.def(py::init<>())
	.def(py::init<int>())
	.def_readonly("size", &int_array::size)
	.def_buffer([](int_array &inter) -> py::buffer_info{
		return py::buffer_info(
			inter.ptr,
			sizeof(int),
			py::format_descriptor<int>::format(),
			1,
			{inter.size},
			{sizeof(int)}
			);
		})
	.def("resize", &int_array::resize)
	.def("__getitem__", [](int_array &arr, int i) -> int{return arr.ptr[i];})
	.def("__setitem__", [](int_array &arr, int i, int val) -> void{arr.ptr[i] = val;});

py::class_<double_array>(m,"double_array",py::buffer_protocol())
	.def(py::init<>())
	.def(py::init<int>())
	.def_readonly("size", &double_array::size)
	.def_buffer([](double_array &arr) -> py::buffer_info{
		return py::buffer_info(
			arr.ptr,
			sizeof(double),
			py::format_descriptor<double>::format(),
			1,
			{arr.size},
			{sizeof(double)}
			);
		})
	.def("resize", &double_array::resize)
	.def("__getitem__", [](double_array &arr, int i) -> double{return arr.ptr[i];})
	.def("__setitem__", [](double_array &arr, int i, double val) -> void{arr.ptr[i] = val;});

py::class_<int_pointer_interface>(m,"int_pointer_interface",py::buffer_protocol())
	.def(py::init<>())
	.def_readonly("size", &int_pointer_interface::size)
	.def_buffer([](int_pointer_interface &inter) -> py::buffer_info{
		return py::buffer_info(
			inter.ptr,
			sizeof(int),
			py::format_descriptor<int>::format(),
			1,
			{inter.size},
			{sizeof(int)}
			);
		});

py::class_<double_pointer_interface>(m,"double_pointer_interface",py::buffer_protocol())
	.def(py::init<>())
	.def_readonly("size", &double_pointer_interface::size)
	.def_buffer([](double_pointer_interface &inter) -> py::buffer_info{
		return py::buffer_info(
			inter.ptr,
			sizeof(double),
			py::format_descriptor<double>::format(),
			1,
			{inter.size},
			{sizeof(double)}
			);
		});


py::class_<double_pointer_interface_array>(m,"double_pointer_interface_array")
	.def(py::init<>())
	.def("__getitem__", [](double_pointer_interface_array &arr, int i)->double_pointer_interface*{return arr.ptr + i;}, py::return_value_policy::reference)
	;

py::class_<Prob_Data>(m,"Prob_Data")
	.def_readwrite("xi", &Prob_Data::xi)
	.def_readwrite("lam", &Prob_Data::lambda)
	.def_readwrite("objval", &Prob_Data::objval)
	.def_readwrite("constr", &Prob_Data::constr)
	.def_readwrite("gradObj", &Prob_Data::gradObj)
	.def_readwrite("constrJac", &Prob_Data::constrJac)
	.def_readwrite("jacIndRow", &Prob_Data::jacIndRow)
	.def_readwrite("jacNz", &Prob_Data::jacNz)
	.def_readwrite("jacIndCol", &Prob_Data::jacIndCol)
	.def_readwrite("dmode", &Prob_Data::dmode)
	.def_readwrite("info", &Prob_Data::info)
	.def_readwrite("hess_arr", &Prob_Data::hess_arr)
	;

py::enum_<blockSQP::SQPresult>(m, "SQPresult")
    .value("it_finished", blockSQP::SQPresult::it_finished)
    .value("partial_success", blockSQP::SQPresult::partial_success)
    .value("success", blockSQP::SQPresult::success)
    .value("super_success", blockSQP::SQPresult::super_success)
    .value("local_infeasibility", blockSQP::SQPresult::local_infeasibility)
    .value("restoration_failuer", blockSQP::SQPresult::restoration_failure)
    .value("linesearch_failure", blockSQP::SQPresult::linesearch_failure)
    .value("qp_failure", blockSQP::SQPresult::qp_failure)
    .value("eval_failure", blockSQP::SQPresult::eval_failure)
    .value("misc_error", blockSQP::SQPresult::misc_error)
    ;

py::class_<blockSQP::Problemspec>(m, "Problemspec");

py::class_<Problemform, blockSQP::Problemspec, Py_Problemform>(m,"Problemform")
	.def(py::init<>())
	.def("init_Cpp_Data", &Problemform::init_Cpp_Data)
	.def("initialize_dense", &Problemform::initialize_dense)
	.def("initialize_sparse", &Problemform::initialize_sparse)
	.def("evaluate_dense", &Problemform::evaluate_dense)
	.def("evaluate_sparse", &Problemform::evaluate_sparse)
	.def("evaluate_simple", &Problemform::evaluate_simple)
	.def("update_inits", &Problemform::update_inits)
	.def("update_evals", &Problemform::update_evals)
	.def("update_simple", &Problemform::update_simple)
	.def("update_xi", &Problemform::update_xi)
	.def("get_objval", &Problemform::get_objval)
	.def("restore_continuity", &Problemform::restore_continuity)
	.def_readwrite("Cpp_Data", &Problemform::Cpp_Data)
	.def_readwrite("nVar", &Problemform::nVar)
	.def_readwrite("nCon", &Problemform::nCon)
	.def_readwrite("nnz", &Problemform::nnz)
	.def_readwrite("objLo", &Problemform::objLo)
	.def_readwrite("objUp", &Problemform::objUp)
	.def_readwrite("lb_var", &Problemform::lb_var)
	.def_readwrite("ub_var", &Problemform::ub_var)
	.def_readwrite("lb_con", &Problemform::lb_con)
	.def_readwrite("ub_con", &Problemform::ub_con)
	.def_readonly("nBlocks", &Problemform::nBlocks)
	.def_property("blockIdx", nullptr, &Problemform::set_blockIdx)
    .def_property("vblocks", nullptr, &Problemform::set_vblocks)
    .def_property("cond", nullptr, [](Problemform &prob, blockSQP::Condenser *cond){prob.cond = cond;})
    //.def_property("vblocks", nullptr, [](Problemform &P, vblock_array &v_arr){P.vblocks = v_arr.ptr; P.n_vblocks = v_arr.size;})
	;

py::class_<blockSQP::SQPstats>(m,"SQPstats")
	.def(py::init<char*>())
	.def_readwrite("itCount", &blockSQP::SQPstats::itCount)
	.def_readwrite("qpIterations", &blockSQP::SQPstats::qpIterations)
	.def_readwrite("qpIterations2", &blockSQP::SQPstats::qpIterations2)
	.def_readwrite("qpItTotal", &blockSQP::SQPstats::qpItTotal)
	.def_readwrite("qpResolve", &blockSQP::SQPstats::qpResolve)
	.def_readwrite("nFunCalls", &blockSQP::SQPstats::nFunCalls)
	.def_readwrite("nDerCalls", &blockSQP::SQPstats::nDerCalls)
	.def_readwrite("nRestHeurCalls", &blockSQP::SQPstats::nRestHeurCalls)
	.def_readwrite("nRestPhaseCalls", &blockSQP::SQPstats::nRestPhaseCalls)
	.def_readwrite("rejectedSR1", &blockSQP::SQPstats::rejectedSR1)
	.def_readwrite("hessSkipped", &blockSQP::SQPstats::hessSkipped)
	.def_readwrite("hessDamped", &blockSQP::SQPstats::hessDamped)
	.def_readwrite("nTotalUpdates", &blockSQP::SQPstats::nTotalUpdates)
	.def_readwrite("nTotalSkippedUpdates", &blockSQP::SQPstats::nTotalSkippedUpdates)
	.def_readwrite("averageSizingFactor", &blockSQP::SQPstats::averageSizingFactor);

py::class_<blockSQP::SQPmethod>(m, "SQPmethod")
	.def(py::init<blockSQP::Problemspec*, blockSQP::SQPoptions*, blockSQP::SQPstats*>())
	 //.def_readonly("vars", &blockSQP::SQPmethod::vars)
    .def_property_readonly("vars", [](blockSQP::SQPmethod *meth){return meth->vars.get();})
	.def_readonly("stats", &blockSQP::SQPmethod::stats)
	.def("init", &blockSQP::SQPmethod::init)
	.def("run", &blockSQP::SQPmethod::run, py::arg("maxIt"), py::arg("warmStart") = 0)
	.def("finish", &blockSQP::SQPmethod::finish)
    .def("resetHessians", static_cast<void (blockSQP::SQPmethod::*)()>(&blockSQP::SQPmethod::resetHessians))
    .def_readwrite("param", &blockSQP::SQPmethod::param)
    .def("set_iterate_", &blockSQP::SQPmethod::set_iterate)
    .def("get_xi", static_cast<blockSQP::Matrix (blockSQP::SQPmethod::*)()>(&blockSQP::SQPmethod::get_xi), py::return_value_policy::take_ownership)
    .def("get_lambda", static_cast<blockSQP::Matrix (blockSQP::SQPmethod::*)()>(&blockSQP::SQPmethod::get_lambda), py::return_value_policy::take_ownership)
    .def("get_scaleFactors", [](blockSQP::SQPmethod &M){if (M.param->automatic_scaling) return double_pointer_interface(M.scaled_prob->scaling_factors.get(), M.scaled_prob->nVar); else return double_pointer_interface();}, py::return_value_policy::take_ownership)
    .def("get_rescaleFactors", [](blockSQP::SQPmethod &M){if (M.param->automatic_scaling) return double_pointer_interface(M.vars->rescaleFactors.get(), M.scaled_prob->nVar); else return double_pointer_interface();}, py::return_value_policy::take_ownership)
    .def("arr_apply_rescaling", [](blockSQP::SQPmethod &M, double_array *arr){if (M.param->automatic_scaling){M.apply_rescaling(arr->ptr);} return;})
    .def("dec_nquasi", [](blockSQP::SQPmethod &M){for (int iBlock = 0; iBlock < M.vars->nBlocks; iBlock++){if (M.vars->nquasi[iBlock] > 0) M.vars->nquasi[iBlock] -= 1;} return;})
    ;
/*
py::class_<blockSQP::SCQPmethod, blockSQP::SQPmethod>(m, "SCQPmethod")
    .def(py::init<blockSQP::Problemspec*, blockSQP::SQPoptions*, blockSQP::SQPstats*, blockSQP::Condenser*>())
    ;

py::class_<blockSQP::SCQP_bound_method, blockSQP::SCQPmethod>(m, "SCQP_bound_method")
    .def(py::init<blockSQP::Problemspec*, blockSQP::SQPoptions*, blockSQP::SQPstats*, blockSQP::Condenser*>())
    ;

py::class_<blockSQP::SCQP_correction_method, blockSQP::SCQPmethod>(m, "SCQP_correction_method")
    .def(py::init<blockSQP::Problemspec*, blockSQP::SQPoptions*, blockSQP::SQPstats*, blockSQP::Condenser*>())
    ;
*/

py::class_<blockSQP::SQPiterate>(m, "SQPiterate")
	.def_readonly("obj", &blockSQP::SQPiterate::obj)
	.def_readonly("cNorm", &blockSQP::SQPiterate::cNorm)
	.def_readonly("cNormS", &blockSQP::SQPiterate::cNormS)
	.def_readonly("gradNorm", &blockSQP::SQPiterate::gradNorm)
	.def_readonly("lambdaStepNorm", &blockSQP::SQPiterate::lambdaStepNorm)
	.def_readonly("tol", &blockSQP::SQPiterate::tol)
	.def_readonly("xi", &blockSQP::SQPiterate::xi)
	.def_readonly("lam", &blockSQP::SQPiterate::lambda)
	.def_readonly("constr", &blockSQP::SQPiterate::constr)
	.def_readonly("constrJac", &blockSQP::SQPiterate::constrJac)
	.def_readonly("trialXi", &blockSQP::SQPiterate::trialXi)
	.def_readwrite("use_homotopy", &blockSQP::SQPiterate::use_homotopy)
	.def("get_hess1_block", [](blockSQP::SQPiterate &vars, int i){return vars.hess1[i];})
	.def("get_hess2_block", [](blockSQP::SQPiterate &vars, int i){return vars.hess2[i];})
	.def_readonly("filter", &blockSQP::SQPiterate::filter)
	.def_readonly("gradLagrange", &blockSQP::SQPiterate::gradLagrange)
	.def_readonly("gammaMat", &blockSQP::SQPiterate::gammaMat)
	.def_readonly("deltaXi", &blockSQP::SQPiterate::deltaXi)
	.def_readonly("deltaMat", &blockSQP::SQPiterate::deltaMat)
    .def_readonly("deltaNormSqMat", &blockSQP::SQPiterate::deltaNormSqMat)
    .def_readonly("deltaGammaMat", &blockSQP::SQPiterate::deltaGammaMat)
    .def_readonly("dg_pos", &blockSQP::SQPiterate::dg_pos)
	;

py::class_<blockSQP::RestorationProblem, blockSQP::Problemspec>(m, "RestorationProblem")
        .def(py::init<blockSQP::Problemspec*, blockSQP::Matrix&, double, double>())
        //.def("__init__", [](Problemform* P, blockSQP::Matrix& M, double d1, double d2){return new RestorationProblem(P, M, d1, d2);})
        ;
    
    
    
py::class_<blockSQP::SCQPiterate, blockSQP::SQPiterate>(m, "SCQPiterate")
	.def_readonly("condensed_Jacobian", &blockSQP::SCQPiterate::condensed_Jacobian)
	.def_readonly("condensed_lb_var", &blockSQP::SCQPiterate::condensed_lb_var)
	.def_readonly("condensed_ub_var", &blockSQP::SCQPiterate::condensed_ub_var)
	.def_readonly("condensed_lb_con", &blockSQP::SCQPiterate::condensed_lb_con)
	.def_readonly("condensed_ub_con", &blockSQP::SCQPiterate::condensed_ub_con)
	;

//Condensing classes and structs
py::class_<blockSQP::vblock>(m, "vblock")
    .def(py::init<int, bool>())
    .def_readwrite("size", &blockSQP::vblock::size)
    .def_readwrite("dependent", &blockSQP::vblock::dependent);

py::class_<vblock_array>(m, "vblock_array")
	.def(py::init<>())
	.def(py::init<int>())
	.def_readonly("size", &vblock_array::size)
	.def("resize", &vblock_array::resize)
	.def("__getitem__", [](vblock_array &arr, int i) -> blockSQP::vblock *{return arr.ptr + i;}, py::return_value_policy::reference)
	.def("__setitem__", [](vblock_array &arr, int i, blockSQP::vblock B) -> void{arr.ptr[i] = B;});

py::class_<blockSQP::cblock>(m, "cblock")
    .def(py::init<int>())
    .def_readwrite("size", &blockSQP::cblock::size);

py::class_<cblock_array>(m, "cblock_array")
    .def(py::init<>())
    .def(py::init<int>())
    .def_readonly("size", &cblock_array::size)
    .def("resize", &cblock_array::resize)
	.def("__getitem__", [](cblock_array &arr, int i) -> blockSQP::cblock *{return arr.ptr + i;}, py::return_value_policy::reference)
	.def("__setitem__", [](cblock_array &arr, int i, blockSQP::cblock B) -> void{arr.ptr[i] = B;});

py::class_<blockSQP::condensing_target>(m, "condensing_target")
    .def(py::init<int, int, int, int, int>())
    .def_readwrite("n_stages", &blockSQP::condensing_target::n_stages)
    .def_readwrite("first_free", &blockSQP::condensing_target::first_free)
    .def_readwrite("vblock_end", &blockSQP::condensing_target::vblock_end)
    .def_readwrite("first_cond", &blockSQP::condensing_target::first_cond)
    .def_readwrite("cblock_end", &blockSQP::condensing_target::cblock_end);

py::class_<condensing_targets>(m, "condensing_targets")
    .def(py::init<>())
    .def(py::init<int>())
    .def_readonly("size", &condensing_targets::size)
    .def("resize", &condensing_targets::resize)
	.def("__getitem__", [](condensing_targets &arr, int i) -> blockSQP::condensing_target *{return arr.ptr + i;}, py::return_value_policy::reference)
	.def("__setitem__", [](condensing_targets &arr, int i, blockSQP::condensing_target B) -> void{arr.ptr[i] = B;});

py::class_<blockSQP::Condenser>(m, "Condenser")
    .def(py::init(
        [](vblock_array &v_arr, cblock_array &c_arr, int_array &hess_sizes, condensing_targets &c_targets, int add_dep_bounds) -> blockSQP::Condenser*
        {return new blockSQP::Condenser(v_arr.ptr, v_arr.size, c_arr.ptr, c_arr.size, hess_sizes.ptr, hess_sizes.size, c_targets.ptr, c_targets.size, add_dep_bounds);}),
        py::arg("vBlocks"), py::arg("cBlocks"), py::arg("hBlocks"), py::arg("Targets"), py::arg("add_dep_bounds") = 2, py::return_value_policy::take_ownership)
    .def("print_debug", &blockSQP::Condenser::print_debug)
    .def("condense_args", [](blockSQP::Condenser &C, condensing_args &args) -> void{
            args.C = &C;
            args.condensed_hess.resize(C.condensed_num_hessblocks);
            C.full_condense(args.grad_obj, args.con_jac, args.hess.ptr, args.lb_var, args.ub_var, args.lb_con, args.ub_con,
                args.condensed_h, args.condensed_Jacobian, args.condensed_hess.ptr, args.condensed_lb_var, args.condensed_ub_var, args.condensed_lb_con, args.condensed_ub_con
            );
            args.condensed_hess.size = C.condensed_num_hessblocks;
            return;})
    .def_readonly("num_hessblocks", &blockSQP::Condenser::num_hessblocks)
    .def_readonly("num_vars", &blockSQP::Condenser::num_vars)
    .def_readonly("num_cons", &blockSQP::Condenser::num_cons)
    .def_readonly("condensed_num_hessblocks", &blockSQP::Condenser::condensed_num_hessblocks)
    .def_readonly("condensed_num_vars", &blockSQP::Condenser::condensed_num_vars)
    .def_readonly("num_true_cons", &blockSQP::Condenser::num_true_cons);

py::class_<SymMat_array>(m, "SymMat_array")
    .def(py::init<>())
    .def(py::init<int>())
    .def_readonly("size", &SymMat_array::size)
    .def("resize", &SymMat_array::resize)
	//.def("__getitem__", [](SymMat_array &arr, int i) -> blockSQP::SymMatrix *{return arr.ptr + i;}, py::return_value_policy::reference)
	.def("__getitem__", [](SymMat_array &arr, int i) -> blockSQP::SymMatrix *{return arr.ptr + i;}, py::return_value_policy::reference)
	.def("__setitem__", [](SymMat_array &arr, int i, blockSQP::SymMatrix &B) -> void{arr.ptr[i] = B;});


py::class_<condensing_args>(m, "condensing_args")
    .def(py::init<>())
    .def_readwrite("grad_obj", &condensing_args::grad_obj)
    .def_readwrite("con_jac", &condensing_args::con_jac)
    .def_readwrite("hess", &condensing_args::hess)
    .def_readwrite("lb_var", &condensing_args::lb_var)
    .def_readwrite("ub_var", &condensing_args::ub_var)
    .def_readwrite("lb_con", &condensing_args::lb_con)
    .def_readwrite("ub_con", &condensing_args::ub_con)
    .def_readonly("condensed_hess", &condensing_args::condensed_hess)
    .def_readonly("condensed_Jacobian", &condensing_args::condensed_Jacobian)
    .def_readonly("condensed_h", &condensing_args::condensed_h)
    .def_readonly("condensed_lb_var", &condensing_args::condensed_lb_var)
    .def_readonly("condensed_ub_var", &condensing_args::condensed_ub_var)
    .def_readonly("condensed_lb_con", &condensing_args::condensed_lb_con)
    .def_readonly("condensed_ub_con", &condensing_args::condensed_ub_con)
    .def_readonly("delta_Xi", &condensing_args::delta_Xi)
    .def_readonly("delta_Lambda", &condensing_args::delta_Lambda)
    .def_readonly("delta_Xi_cond", &condensing_args::delta_Xi_cond)
    .def_readonly("delta_Lambda_cond", &condensing_args::delta_Lambda_cond)
    .def_readonly("delta_Xi_rest", &condensing_args::delta_Xi_rest)
    .def_readonly("delta_Lambda_rest", &condensing_args::delta_Lambda_rest)
    .def("solve_QPs", &condensing_args::solve_QPs)
    ;


py::class_<blockSQP::TC_restoration_Problem, blockSQP::Problemspec>(m, "TC_restoration_Problem")
    .def(py::init<blockSQP::Problemspec*, blockSQP::Condenser*, const blockSQP::Matrix&, double, double>())
    .def_readonly("nVar", &blockSQP::TC_restoration_Problem::nVar)
    .def_readonly("nCon", &blockSQP::TC_restoration_Problem::nCon)
    .def_readwrite("xi_ref", &blockSQP::TC_restoration_Problem::xi_ref)
    ;

py::class_<blockSQP::TC_feasibility_Problem, blockSQP::Problemspec>(m, "TC_feasibility_Problem")
    .def(py::init<blockSQP::Problemspec*, blockSQP::Condenser*>())
    .def_readonly("nVar", &blockSQP::TC_feasibility_Problem::nVar)
    .def_readonly("nCon", &blockSQP::TC_feasibility_Problem::nCon)
	.def_readwrite("nnz", &blockSQP::TC_feasibility_Problem::nnz)
	.def_readwrite("objLo", &blockSQP::TC_feasibility_Problem::objLo)
	.def_readwrite("objUp", &blockSQP::TC_feasibility_Problem::objUp)
	.def_readwrite("lb_var", &blockSQP::TC_feasibility_Problem::lb_var)
	.def_readwrite("ub_var", &blockSQP::TC_feasibility_Problem::ub_var)
	.def_readwrite("lb_con", &blockSQP::TC_feasibility_Problem::lb_con)
	.def_readwrite("ub_con", &blockSQP::TC_feasibility_Problem::ub_con)
	.def_readonly("nBlocks", &blockSQP::TC_feasibility_Problem::nBlocks)
	.def_readonly("xi_orig", &blockSQP::TC_feasibility_Problem::xi_parent)
	.def_readonly("slack", &blockSQP::TC_feasibility_Problem::slack)
	.def_property("jac_orig_nz", [](blockSQP::TC_feasibility_Problem &P)->double_pointer_interface{double_pointer_interface nonzeros; nonzeros.size = P.nnz; nonzeros.ptr = P.jac_orig_nz; return nonzeros;}, nullptr)
	.def_property("jac_orig_row", [](blockSQP::TC_feasibility_Problem &P)->int_pointer_interface{int_pointer_interface row; row.size = P.nnz; row.ptr = P.jac_orig_row; return row;}, nullptr)
	.def_property("jac_orig_colind", [](blockSQP::TC_feasibility_Problem &P)->int_pointer_interface{int_pointer_interface colind; colind.size = P.nVar + 1; colind.ptr = P.jac_orig_colind; return colind;}, nullptr)
	;

py::class_<blockSQP::scaled_Problemspec, blockSQP::Problemspec>(m, "scaled_Problemspec")
    .def(py::init<blockSQP::Problemspec*>())
    .def("arr_set_scale", [](blockSQP::scaled_Problemspec &P, double_array &arr){P.set_scale(arr.ptr);});
}















