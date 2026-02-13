/*
 * blockSQP2 -- A structure-exploiting nonlinear programming solver based
 *              on blockSQP by Dennis Janka.
 * Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
 * 
 * Licensed under the zlib license. See LICENSE for more details.
 */

/**
 * \file pyblockSQP2.cpp
 * \author Reinhold Wittmann
 * \date 2022-2025
 *
 * Pybind11 based python interface to blockSQP2
 */


#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/native_enum.h>
#include <tuple>
#include <iostream>
#include <fstream>
#include <chrono>
using namespace std::chrono;
#include <string>
#include <blockSQP2/options.hpp>
#include <blockSQP2/method.hpp>
#include <blockSQP2/condensing.hpp>
#include <blockSQP2/restoration.hpp>

#include "qpOASES.hpp"

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
    T_array(const T_array& arr): size(arr.size){ptr = new T[size]; std::copy(arr.ptr, arr.ptr + arr.size, ptr);}
    T_array &operator=(const T_array& arr){delete[] ptr; size = arr.size; ptr = new T[size]; std::copy(arr.ptr, arr.ptr + arr.size, ptr); return *this;}
    
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
typedef T_array<blockSQP2::vblock> vblock_array;
typedef T_array<blockSQP2::cblock> cblock_array;
typedef T_array<blockSQP2::condensing_target> condensing_targets;
typedef T_array<blockSQP2::SymMatrix> SymMat_array;



//For testing condensing from python
class condensing_args{
public:
    blockSQP2::Matrix grad_obj;
    blockSQP2::Sparse_Matrix con_jac;
    SymMat_array hess;
    blockSQP2::Matrix lb_var;
    blockSQP2::Matrix ub_var;
    blockSQP2::Matrix lb_con;
    blockSQP2::Matrix ub_con;

    SymMat_array condensed_hess;
    blockSQP2::Sparse_Matrix condensed_Jacobian;
    blockSQP2::Matrix condensed_h;
    blockSQP2::Matrix condensed_lb_var;
    blockSQP2::Matrix condensed_ub_var;
    blockSQP2::Matrix condensed_lb_con;
    blockSQP2::Matrix condensed_ub_con;

    blockSQP2::Matrix deltaXi;
    blockSQP2::Matrix lambdaQP;

    blockSQP2::Matrix deltaXi_cond;
    blockSQP2::Matrix lambdaQP_cond;
    blockSQP2::Matrix deltaXi_rest;
    blockSQP2::Matrix lambdaQP_rest;
    
    blockSQP2::Condenser *C;
    
    
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
        blockSQP2::Sparse_Matrix red_con_jac = condensed_Jacobian;
        blockSQP2::Matrix red_lb_con = condensed_lb_con;
        blockSQP2::Matrix red_ub_con = condensed_ub_con;
        
        qpOASES::SQProblem* qp;
        qpOASES::SQProblem* qp_cond;
        qpOASES::returnValue ret = qpOASES::TERMINAL_LIST_ELEMENT;
        qpOASES::returnValue ret_cond = qpOASES::TERMINAL_LIST_ELEMENT;
        
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
        
        
        steady_clock::time_point begin = steady_clock::now();
        ret = qp->init(H, g, A_qp, lb, ub, lbA, ubA, max_it, &cpu_time);
        steady_clock::time_point end = steady_clock::now();
        
        if (ret != qpOASES::SUCCESSFUL_RETURN){
            std::cout << "Could not solve the QP\n";
            return;
        }
        else{
            std::cout << "Solving the QP took " << duration_cast<milliseconds>(end - begin) << "\n";
        }

        
        begin = steady_clock::now();
        ret_cond = qp_cond->init(H_cond, g_cond, A_qp_cond, lb_cond, ub_cond, lbA_cond, ubA_cond, max_it_cond, &cpu_time_cond);
        end = steady_clock::now();
        
        if (ret_cond != qpOASES::SUCCESSFUL_RETURN){
            std::cout << "Could not solve the condensed QP\n";
            return;
        }
        else{
            if (C->add_dep_bounds == 2)
                std::cout << "Solving the condensed QP with implicit bounds took " << duration_cast<milliseconds>(end - begin) << "\n";
            else if (C->add_dep_bounds == 0)
                std::cout << "Solving the condensed QP without implicit bounds took " << duration_cast<milliseconds>(end - begin) << "\n";
        }
            
        deltaXi.Dimension(con_jac.n, 1);
        deltaXi.Initialize(0.);
        lambdaQP.Dimension(con_jac.n + con_jac.m, 1);
        lambdaQP.Initialize(0.);
        
        deltaXi_cond.Dimension(red_con_jac.n, 1);
        deltaXi_cond.Initialize(0.);
        lambdaQP_cond.Dimension(red_con_jac.n + red_con_jac.m,1);
        lambdaQP_cond.Initialize(0.);
        
        qp->getPrimalSolution(deltaXi.array);
        qp->getDualSolution(lambdaQP.array);
        
        qp_cond->getPrimalSolution(deltaXi_cond.array);
        qp_cond->getDualSolution(lambdaQP_cond.array);
        
        C->recover_var_mult(deltaXi_cond, lambdaQP_cond, deltaXi_rest, lambdaQP_rest);
        
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
//


struct Prob_Data{
    double_pointer_interface xi;			///< optimization variables
    double_pointer_interface lambda;		///< Lagrange multipliers
    double objval;				            ///< objective function value
    double_pointer_interface constr;		///< constraint function values
    double_pointer_interface gradObj;		///< gradient of objective

    blockSQP2::Matrix constrJac;
    double_pointer_interface jacNz;		    ///< nonzero elements of constraint Jacobian
    int_pointer_interface jacIndRow;		///< row indices of nonzero elements
	int_pointer_interface jacIndCol;		///< starting indices of columns

    //Each hessian blocks elements are a double array, wrapper by double_pointer_interface
    //These are the once again wrapped by double_pointer_interface_interface
    double_pointer_interface_array hess_arr;
    int dmode;				                ///< derivative mode
    int info;				                ///< error flag
};


class PyProblemspec : public blockSQP2::Problemspec
{
public:
    PyProblemspec(){}
    
    virtual ~PyProblemspec(){
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
        
        if (nBlocks > 0){
            Cpp_Data.hess_arr.size = nBlocks;
            double_pointer_interface *h_arrays = new double_pointer_interface[nBlocks];
            for (int j = 0; j < nBlocks; j++){
                h_arrays[j].size = ((blockIdx[j+1] - blockIdx[j]) * (blockIdx[j+1] - blockIdx[j] + 1))/2 ;
            }
            Cpp_Data.hess_arr.ptr = h_arrays;
        }
        
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
    virtual void update_lambda(){};
    
    virtual void get_objval(){};
    
    virtual void restore_continuity(){};
    virtual void call_stepModification(){};
    
    
    void initialize(blockSQP2::Matrix &xi, blockSQP2::Matrix &lambda, blockSQP2::Matrix &constrJac) override {
        Cpp_Data.xi.ptr = xi.array;
        Cpp_Data.lambda.ptr = lambda.array;
        Cpp_Data.constrJac.array = constrJac.array;

        update_inits();
        initialize_dense();
    }
    
    
    void initialize(blockSQP2::Matrix &xi, blockSQP2::Matrix &lambda, double *jacNz, int *jacIndRow, int *jacIndCol) override {
        Cpp_Data.xi.ptr = xi.array;
        Cpp_Data.lambda.ptr = lambda.array;
        Cpp_Data.jacIndRow.ptr = jacIndRow;
        Cpp_Data.jacIndCol.ptr = jacIndCol;

        update_inits();
        initialize_sparse();
    }
    
    
    void evaluate(const blockSQP2::Matrix &xi, const blockSQP2::Matrix &lambda, 
            double *objval, blockSQP2::Matrix &constr, blockSQP2::Matrix &gradObj, blockSQP2::Matrix &constrJac,
            blockSQP2::SymMatrix *hess, int dmode, int *info) override {
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
        
        *info = Cpp_Data.info;
        if (*info > 0) [[unlikely]] return;
        
        get_objval();
        *objval = Cpp_Data.objval;
    }
    
    
    void evaluate(const blockSQP2::Matrix &xi, const blockSQP2::Matrix &lambda, double *objval, blockSQP2::Matrix &constr, 
                    blockSQP2::Matrix &gradObj, double *jacNz, int *jacIndRow, int *jacIndCol, blockSQP2::SymMatrix *hess, 
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
        
        *info = Cpp_Data.info;
        if (*info > 0) [[unlikely]] return;
        
        get_objval();
        *objval = Cpp_Data.objval;
    }
    
    
    void evaluate(const blockSQP2::Matrix &xi, double *objval, blockSQP2::Matrix &constr, int *info) override {
        Cpp_Data.xi.ptr = xi.array;
        Cpp_Data.constr.ptr = constr.array;
        
        update_simple();
        evaluate_simple();
        
        *info = Cpp_Data.info;
        if (*info > 0) [[unlikely]] return;
        
        get_objval();
        *objval = Cpp_Data.objval;
    }
    
    
    void set_blockIdx(py::array_t<int> arr){
        py::buffer_info buff = arr.request();
        nBlocks = buff.size - 1;
        delete[] blockIdx;
        blockIdx = new int[buff.size];
        std::copy((int*)buff.ptr, (int*)buff.ptr + buff.size, blockIdx);
    }
    
    void set_vblocks(vblock_array &VB){
        n_vblocks = VB.size;
        delete[] vblocks;
        vblocks = new blockSQP2::vblock[n_vblocks];
        for (int i = 0; i < n_vblocks; i++){
            vblocks[i] = VB.ptr[i];
        }
    }
    
    void reduceConstrVio(blockSQP2::Matrix &xi, int *info) override {
        Cpp_Data.xi.ptr = xi.array;
        update_xi();
        restore_continuity();
        *info = Cpp_Data.info;
    }
    
    
    void stepModification(blockSQP2::Matrix &trialXi, blockSQP2::Matrix &trialLambda, int *info) override {
        Cpp_Data.xi.ptr = trialXi.array;
        Cpp_Data.lambda.ptr = trialLambda.array;
        update_xi(); update_lambda();
        call_stepModification();
        *info = Cpp_Data.info;
    }
};


class PyProblemspecTrampoline: public PyProblemspec{
    void initialize_dense() override {
        PYBIND11_OVERRIDE(void, PyProblemspec, initialize_dense,);
    }
    
    void initialize_sparse() override {
        PYBIND11_OVERRIDE(void, PyProblemspec, initialize_sparse,);
    }
    
    void evaluate_dense() override {
        PYBIND11_OVERRIDE(void, PyProblemspec, evaluate_dense,);
    }
    
    void evaluate_sparse() override {
        PYBIND11_OVERRIDE(void, PyProblemspec, evaluate_sparse,);
    }
    
    void evaluate_simple() override {
        PYBIND11_OVERRIDE(void, PyProblemspec, evaluate_simple,);
    }
    
    void update_inits() override {
        PYBIND11_OVERRIDE(void, PyProblemspec, update_inits,);
    }
    void update_evals() override {
        PYBIND11_OVERRIDE(void, PyProblemspec, update_evals,);
    }
    void update_simple() override {
        PYBIND11_OVERRIDE(void, PyProblemspec, update_simple,);
    }
    
    void update_xi() override {
        PYBIND11_OVERRIDE(void, PyProblemspec, update_xi,);
    }
    
    void update_lambda() override {
        PYBIND11_OVERRIDE(void, PyProblemspec, update_lambda,);
    }
    
    void get_objval() override {
        PYBIND11_OVERRIDE(void, PyProblemspec, get_objval,);
    }
    
    void restore_continuity() override {
        PYBIND11_OVERRIDE(void, PyProblemspec, restore_continuity,);
    }
    
    void call_stepModification() override {
        PYBIND11_OVERRIDE(void, PyProblemspec, call_stepModification,);
    }
};



PYBIND11_MODULE(pyblockSQP2, m){

py::class_<blockSQP2::Matrix>(m, "Matrix", py::buffer_protocol())
	.def_buffer([](blockSQP2::Matrix &mtrx) -> py::buffer_info{
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
	.def(py::init<const blockSQP2::Matrix&>())
	.def(py::init<const blockSQP2::SymMatrix&>())
	.def("Dimension", &blockSQP2::Matrix::Dimension, py::arg("M"), py::arg("N") = 1, py::arg("LDIM") = -1)
	.def("Initialize", static_cast<blockSQP2::Matrix& (blockSQP2::Matrix::*)(double)>(&blockSQP2::Matrix::Initialize))
	.def_readwrite("ldim",&blockSQP2::Matrix::ldim)
	.def_readwrite("m",&blockSQP2::Matrix::m)
	.def_readwrite("n",&blockSQP2::Matrix::n)
    .def("__setitem__", [](blockSQP2::Matrix &M, std::tuple<int, int> inds, double val) -> void{M(std::get<0>(inds), std::get<1>(inds)) = val; return;})
	.def("__getitem__", [](blockSQP2::Matrix &M, std::tuple<int, int> inds) -> double{return M(std::get<0>(inds), std::get<1>(inds));})
	.def("__setitem__", [](blockSQP2::Matrix &M, int ind, double val) -> void{M(ind) = val; return;})
	.def("__getitem__", [](blockSQP2::Matrix &M, int ind) -> double{return M(ind);})
	.def_property("array", nullptr, [](blockSQP2::Matrix &mtrx, py::array_t<double> arr){
		py::buffer_info buff = arr.request();
		mtrx.array = (double*)buff.ptr;
		});

py::class_<blockSQP2::SymMatrix>(m, "SymMatrix")
    .def(py::init<>())
    .def(py::init<int>(), py::arg("M") = 1)
    .def(py::init<const blockSQP2::Matrix&>())
	.def("Dimension", [](blockSQP2::SymMatrix &M1, int m) -> void{M1.Dimension(m);})
	.def("Initialize", static_cast<blockSQP2::SymMatrix & (blockSQP2::SymMatrix::*)(double)>(&blockSQP2::SymMatrix::Initialize))
	.def_readwrite("ldim",&blockSQP2::SymMatrix::ldim)
	.def_readwrite("m",&blockSQP2::SymMatrix::m)
	.def("__setitem__", [](blockSQP2::SymMatrix &M, std::tuple<int, int> inds, double val) -> void{M(std::get<0>(inds), std::get<1>(inds)) = val; return;})
	.def("__getitem__", [](blockSQP2::SymMatrix &M, std::tuple<int, int> inds) -> double{return M(std::get<0>(inds), std::get<1>(inds));})
    .def("set", [](blockSQP2::SymMatrix &M1, int i, int j, const blockSQP2::Matrix &M2) -> void{
        M1.Dimension(M2.m);
        for (int i = 0; i < M2.m; i++){
            for (int j = 0; j <= i; j++){
                M1(i,j) = M1(i,j);
            }
        }
        return;
    })
    .def("tril", [](blockSQP2::SymMatrix &M)->double_pointer_interface{return double_pointer_interface(M.array, (M.m*(M.m + 1))/2);})
    ;

py::class_<blockSQP2::Sparse_Matrix>(m, "Sparse_Matrix")
    .def(py::init<>())
    .def(py::init([](int M, int N, double_array &nz, int_array &row, int_array &colind) -> blockSQP2::Sparse_Matrix*{
        std::unique_ptr<double[]> NZ = std::unique_ptr<double[]>(nz.ptr); std::unique_ptr<int[]> ROW = std::unique_ptr<int[]>(row.ptr); std::unique_ptr<int[]> COLIND = std::unique_ptr<int[]>(colind.ptr);
        nz.size = 0; nz.ptr = nullptr; row.size = 0; row.ptr = nullptr; colind.size = 0; colind.ptr = nullptr;
        return new blockSQP2::Sparse_Matrix(M, N, std::move(NZ), std::move(ROW), std::move(COLIND));
    }), py::return_value_policy::take_ownership)
    .def_readonly("m", &blockSQP2::Sparse_Matrix::m)
    .def_readonly("n", &blockSQP2::Sparse_Matrix::n)
    .def("dense", &blockSQP2::Sparse_Matrix::dense)
    .def_property("NZ", [](blockSQP2::Sparse_Matrix &M)->double_pointer_interface{double_pointer_interface nonzeros; nonzeros.size = M.colind[M.n]; nonzeros.ptr = M.nz.get(); return nonzeros;}, nullptr)
    .def_property("ROW", [](blockSQP2::Sparse_Matrix &M)->int_pointer_interface{int_pointer_interface row; row.size = M.colind[M.n]; row.ptr = M.row.get(); return row;}, nullptr)
    .def_property("COLIND", [](blockSQP2::Sparse_Matrix &M)->int_pointer_interface{int_pointer_interface colind; colind.size = M.n + 1; colind.ptr = M.colind.get(); return colind;}, nullptr)
    .def_property("nnz", [](blockSQP2::Sparse_Matrix &M)->int{return M.colind[M.n];}, nullptr)
    ;
    
py::class_<blockSQP2::SQPoptions>(m, "SQPoptions")
	.def(py::init<>())
	.def("optionsConsistency", static_cast<void (blockSQP2::SQPoptions::*)()>(&blockSQP2::SQPoptions::optionsConsistency))
    .def("optionsConsistency", static_cast<void (blockSQP2::SQPoptions::*)(blockSQP2::Problemspec*)>(&blockSQP2::SQPoptions::optionsConsistency))
	.def_readwrite("print_level",&blockSQP2::SQPoptions::print_level)
	.def_readwrite("result_print_color",&blockSQP2::SQPoptions::result_print_color)
	.def_readwrite("debug_level",&blockSQP2::SQPoptions::debug_level)
	.def_readwrite("eps",&blockSQP2::SQPoptions::eps)
	.def_readwrite("inf",&blockSQP2::SQPoptions::inf)
	.def_readwrite("opt_tol",&blockSQP2::SQPoptions::opt_tol)
	.def_readwrite("feas_tol",&blockSQP2::SQPoptions::feas_tol)
	.def_readwrite("sparse",&blockSQP2::SQPoptions::sparse)
	.def_readwrite("enable_linesearch",&blockSQP2::SQPoptions::enable_linesearch)
	.def_readwrite("enable_rest",&blockSQP2::SQPoptions::enable_rest)
	.def_readwrite("max_linesearch_steps",&blockSQP2::SQPoptions::max_linesearch_steps)
	.def_readwrite("max_consec_reduced_steps",&blockSQP2::SQPoptions::max_consec_reduced_steps)
	.def_readwrite("max_consec_skipped_updates",&blockSQP2::SQPoptions::max_consec_skipped_updates)
	.def_readwrite("max_QP_it",&blockSQP2::SQPoptions::max_QP_it)
	.def_readwrite("block_hess",&blockSQP2::SQPoptions::block_hess)
	// .def_readwrite("sizing",&blockSQP2::SQPoptions::sizing)
	// .def_readwrite("fallback_sizing",&blockSQP2::SQPoptions::fallback_sizing)
    
    .def_property("sizing", 
        [](blockSQP2::SQPoptions &opts){return blockSQP2::to_string(opts.sizing);},
        [](blockSQP2::SQPoptions &opts, std::string_view Sname){opts.sizing = blockSQP2::Sizings_from_string(Sname);}
    )
    .def_property("fallback_sizing", 
        [](blockSQP2::SQPoptions &opts){return blockSQP2::to_string(opts.fallback_sizing);},
        [](blockSQP2::SQPoptions &opts, std::string_view Sname){opts.fallback_sizing = blockSQP2::Sizings_from_string(Sname);}
    )
    
    
	.def_readwrite("max_QP_secs",&blockSQP2::SQPoptions::max_QP_secs)
	.def_readwrite("initial_hess_scale",&blockSQP2::SQPoptions::initial_hess_scale)
	.def_readwrite("COL_eps",&blockSQP2::SQPoptions::COL_eps)
	.def_readwrite("OL_eps", &blockSQP2::SQPoptions::OL_eps)
	.def_readwrite("COL_tau_1",&blockSQP2::SQPoptions::COL_tau_1)
	.def_readwrite("COL_tau_2",&blockSQP2::SQPoptions::COL_tau_2)
	.def_readwrite("min_damping_quotient",&blockSQP2::SQPoptions::min_damping_quotient)
	.def_readwrite("SR1_abstol",&blockSQP2::SQPoptions::SR1_abstol)
	.def_readwrite("SR1_reltol",&blockSQP2::SQPoptions::SR1_reltol)
	.def_readwrite("BFGS_damping_factor",&blockSQP2::SQPoptions::BFGS_damping_factor)
	// .def_readwrite("hess_approx",&blockSQP2::SQPoptions::hess_approx)
	// .def_readwrite("fallback_approx",&blockSQP2::SQPoptions::fallback_approx)
    // .def_property("hess_approx", 
    //     [](blockSQP2::SQPoptions &opts){return int(opts.hess_approx);},
    //     [](blockSQP2::SQPoptions &opts, int num){opts.hess_approx = blockSQP2::Hessians(num);}
    // )
    // .def_property("fallback_approx", 
    //     [](blockSQP2::SQPoptions &opts){return int(opts.fallback_approx);},
    //     [](blockSQP2::SQPoptions &opts, int num){opts.fallback_approx = blockSQP2::Hessians(num);}
    // )
    
    .def_property("hess_approx", 
        [](blockSQP2::SQPoptions &opts){return blockSQP2::to_string(opts.hess_approx);},
        [](blockSQP2::SQPoptions &opts, std::string_view Hname){opts.hess_approx = blockSQP2::Hessians_from_string(Hname);}
    )
    .def_property("fallback_approx", 
        [](blockSQP2::SQPoptions &opts){return blockSQP2::to_string(opts.fallback_approx);},
        [](blockSQP2::SQPoptions &opts, std::string_view Hname){opts.fallback_approx = blockSQP2::Hessians_from_string(Hname);}
    )
    
	.def_readwrite("indef_local_only", &blockSQP2::SQPoptions::indef_local_only)
	.def_readwrite("lim_mem",&blockSQP2::SQPoptions::lim_mem)
	.def_readwrite("mem_size",&blockSQP2::SQPoptions::mem_size)
	// .def_readwrite("exact_hess",&blockSQP2::SQPoptions::exact_hess)
	.def_readwrite("skip_first_linesearch",&blockSQP2::SQPoptions::skip_first_linesearch)
	.def_readwrite("conv_strategy",&blockSQP2::SQPoptions::conv_strategy)
	.def_readwrite("max_conv_QPs",&blockSQP2::SQPoptions::max_conv_QPs)
	.def_readwrite("reg_factor", &blockSQP2::SQPoptions::reg_factor)
	.def_readwrite("max_SOC",&blockSQP2::SQPoptions::max_SOC)
	//.def_readwrite("max_bound_refines", &blockSQP2::SQPoptions::max_bound_refines)
	//.def_readwrite("max_correction_steps", &blockSQP2::SQPoptions::max_correction_steps)
	//.def_readwrite("dep_bound_tolerance", &blockSQP2::SQPoptions::dep_bound_tolerance)
	.def_readwrite("max_filter_overrides", &blockSQP2::SQPoptions::max_filter_overrides)
	.def_readwrite("conv_tau_H", &blockSQP2::SQPoptions::conv_tau_H)
	.def_readwrite("conv_kappa_0", &blockSQP2::SQPoptions::conv_kappa_0)
    .def_readwrite("conv_kappa_max", &blockSQP2::SQPoptions::conv_kappa_max)
	.def_readwrite("rest_zeta", &blockSQP2::SQPoptions::rest_zeta)
	.def_readwrite("rest_rho", &blockSQP2::SQPoptions::rest_rho)
    .def_readwrite("automatic_scaling", &blockSQP2::SQPoptions::automatic_scaling)

    .def_readwrite("kappaF", &blockSQP2::SQPoptions::kappaF)
    .def_readwrite("kappaSOC", &blockSQP2::SQPoptions::kappaSOC)
    .def_readwrite("gammaTheta", &blockSQP2::SQPoptions::gammaTheta)
    .def_readwrite("gammaF", &blockSQP2::SQPoptions::gammaF)
    .def_readwrite("eta", &blockSQP2::SQPoptions::eta)
    .def_readwrite("max_extra_steps", &blockSQP2::SQPoptions::max_extra_steps)
    .def_readwrite("enable_premature_termination", &blockSQP2::SQPoptions::enable_premature_termination)
	.def_property("qpsol", [](blockSQP2::SQPoptions &opts)->std::string{
        if (opts.qpsol == blockSQP2::QPsolvers::qpOASES) return "qpOASES";
        else if (opts.qpsol == blockSQP2::QPsolvers::gurobi) return "gurobi";
        else if (opts.qpsol == blockSQP2::QPsolvers::qpalm) return "qpalm";
        return "unset";
        },
    [](blockSQP2::SQPoptions &opts, std::string &QPsolver_name){
        if (QPsolver_name == "qpOASES") opts.qpsol = blockSQP2::QPsolvers::qpOASES;
        else if (QPsolver_name == "gurobi") opts.qpsol = blockSQP2::QPsolvers::gurobi;
        else if (QPsolver_name  == "qpalm") opts.qpsol = blockSQP2::QPsolvers::qpalm;
        else throw blockSQP2::ParameterError("Unknown QP solver, known (no neccessarily linked) are qpOASES, gurobi, qpalm");
        return;
        }
    )
    .def_readwrite("qpsol_options", &blockSQP2::SQPoptions::qpsol_options)
    .def_readwrite("par_QPs", &blockSQP2::SQPoptions::par_QPs)
    .def_readwrite("enable_QP_cancellation", &blockSQP2::SQPoptions::enable_QP_cancellation)
    .def_readwrite("test_opt_1", &blockSQP2::SQPoptions::test_opt_1)
    .def_readwrite("test_opt_2", &blockSQP2::SQPoptions::test_opt_2)
    .def_readwrite("test_opt_3", &blockSQP2::SQPoptions::test_opt_3)
    .def_readwrite("test_opt_4", &blockSQP2::SQPoptions::test_opt_4)
    .def_readwrite("test_opt_5", &blockSQP2::SQPoptions::test_opt_5)
    .def_readwrite("indef_delay", &blockSQP2::SQPoptions::indef_delay)
	;

py::class_<blockSQP2::QPsolver_options>(m, "QPsolver_options");

py::class_<blockSQP2::qpOASES_options, blockSQP2::QPsolver_options>(m, "qpOASES_options")
    .def(py::init<>())
    .def_readwrite("sparsityLevel", &blockSQP2::qpOASES_options::sparsityLevel)
    .def_readwrite("printLevel", &blockSQP2::qpOASES_options::printLevel)
    .def_readwrite("terminationTolerance", &blockSQP2::qpOASES_options::terminationTolerance)
    ;

py::class_<blockSQP2::gurobi_options, blockSQP2::QPsolver_options>(m, "gurobi_options")
    .def(py::init<>())
    .def_readwrite("Method", &blockSQP2::gurobi_options::Method)
    .def_readwrite("NumericFocus", &blockSQP2::gurobi_options::NumericFocus)
    .def_readwrite("OutputFlag", &blockSQP2::gurobi_options::OutputFlag)
    .def_readwrite("Presolve", &blockSQP2::gurobi_options::Presolve)
    .def_readwrite("Aggregate", &blockSQP2::gurobi_options::Aggregate)
    .def_readwrite("BarHomogeneous", &blockSQP2::gurobi_options::BarHomogeneous)
    .def_readwrite("OptimalityTol", &blockSQP2::gurobi_options::OptimalityTol)
    .def_readwrite("FeasibilityTol", &blockSQP2::gurobi_options::FeasibilityTol)
    .def_readwrite("PSDTol", &blockSQP2::gurobi_options::PSDTol)
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

// py::enum_<blockSQP2::SQPresults>(m, "SQPresults")
py::native_enum<blockSQP2::SQPresults>(m, "SQPresults", "enum.Enum")
    .value("it_finished", blockSQP2::SQPresults::it_finished)
    .value("partial_success", blockSQP2::SQPresults::partial_success)
    .value("success", blockSQP2::SQPresults::success)
    .value("super_success", blockSQP2::SQPresults::super_success)
    .value("local_infeasibility", blockSQP2::SQPresults::local_infeasibility)
    .value("restoration_failuer", blockSQP2::SQPresults::restoration_failure)
    .value("linesearch_failure", blockSQP2::SQPresults::linesearch_failure)
    .value("qp_failure", blockSQP2::SQPresults::qp_failure)
    .value("eval_failure", blockSQP2::SQPresults::eval_failure)
    .value("misc_error", blockSQP2::SQPresults::misc_error)
    .finalize()
    ;

py::class_<blockSQP2::Problemspec>(m, "blockSQP2Problemspec");

py::class_<PyProblemspec, blockSQP2::Problemspec, PyProblemspecTrampoline>(m,"PyProblemspec")
	.def(py::init<>())
	.def("init_Cpp_Data", &PyProblemspec::init_Cpp_Data)
	.def("initialize_dense", &PyProblemspec::initialize_dense)
	.def("initialize_sparse", &PyProblemspec::initialize_sparse)
	.def("evaluate_dense", &PyProblemspec::evaluate_dense)
	.def("evaluate_sparse", &PyProblemspec::evaluate_sparse)
	.def("evaluate_simple", &PyProblemspec::evaluate_simple)
	.def("update_inits", &PyProblemspec::update_inits)
	.def("update_evals", &PyProblemspec::update_evals)
	.def("update_simple", &PyProblemspec::update_simple)
	.def("update_xi", &PyProblemspec::update_xi)
	.def("get_objval", &PyProblemspec::get_objval)
	.def("restore_continuity", &PyProblemspec::restore_continuity)
	.def_readwrite("Cpp_Data", &PyProblemspec::Cpp_Data)
	.def_readwrite("nVar", &PyProblemspec::nVar)
	.def_readwrite("nCon", &PyProblemspec::nCon)
	.def_readwrite("nnz", &PyProblemspec::nnz)
	.def_readwrite("objLo", &PyProblemspec::objLo)
	.def_readwrite("objUp", &PyProblemspec::objUp)
	.def_readwrite("lb_var", &PyProblemspec::lb_var)
	.def_readwrite("ub_var", &PyProblemspec::ub_var)
	.def_readwrite("lb_con", &PyProblemspec::lb_con)
	.def_readwrite("ub_con", &PyProblemspec::ub_con)
	.def_readonly("nBlocks", &PyProblemspec::nBlocks)
	.def_property("blockIdx", nullptr, &PyProblemspec::set_blockIdx)
    .def_property("vblocks", nullptr, &PyProblemspec::set_vblocks)
    .def_property("cond", nullptr, [](PyProblemspec &prob, blockSQP2::Condenser *cond){prob.cond = cond;})
    //.def_property("vblocks", nullptr, [](PyProblemspec &P, vblock_array &v_arr){P.vblocks = v_arr.ptr; P.n_vblocks = v_arr.size;})
	;

py::class_<blockSQP2::SQPstats>(m,"SQPstats")
	.def(py::init<char*>())
	.def_readwrite("itCount", &blockSQP2::SQPstats::itCount)
	.def_readwrite("qpIterations", &blockSQP2::SQPstats::qpIterations)
	.def_readwrite("qpIterations2", &blockSQP2::SQPstats::qpIterations2)
	.def_readwrite("qpItTotal", &blockSQP2::SQPstats::qpItTotal)
	.def_readwrite("qpResolve", &blockSQP2::SQPstats::qpResolve)
	.def_readwrite("nFunCalls", &blockSQP2::SQPstats::nFunCalls)
	.def_readwrite("nDerCalls", &blockSQP2::SQPstats::nDerCalls)
	.def_readwrite("nRestHeurCalls", &blockSQP2::SQPstats::nRestHeurCalls)
	.def_readwrite("nRestPhaseCalls", &blockSQP2::SQPstats::nRestPhaseCalls)
	.def_readwrite("rejectedSR1", &blockSQP2::SQPstats::rejectedSR1)
	.def_readwrite("hessSkipped", &blockSQP2::SQPstats::hessSkipped)
	.def_readwrite("hessDamped", &blockSQP2::SQPstats::hessDamped)
	.def_readwrite("nTotalUpdates", &blockSQP2::SQPstats::nTotalUpdates)
	.def_readwrite("nTotalSkippedUpdates", &blockSQP2::SQPstats::nTotalSkippedUpdates)
	.def_readwrite("averageSizingFactor", &blockSQP2::SQPstats::averageSizingFactor);

py::class_<blockSQP2::SQPmethod>(m, "SQPmethod")
	.def(py::init<blockSQP2::Problemspec*, blockSQP2::SQPoptions*, blockSQP2::SQPstats*>())
	 //.def_readonly("vars", &blockSQP2::SQPmethod::vars)
    .def_property_readonly("vars", [](blockSQP2::SQPmethod *meth){return meth->vars.get();})
	.def_readonly("stats", &blockSQP2::SQPmethod::stats)
	.def("init", &blockSQP2::SQPmethod::init)
	.def("run", &blockSQP2::SQPmethod::run, py::arg("maxIt"), py::arg("warmStart") = 0)
	.def("finish", &blockSQP2::SQPmethod::finish)
    .def("resetHessians", static_cast<void (blockSQP2::SQPmethod::*)()>(&blockSQP2::SQPmethod::resetHessians))
    .def_readwrite("param", &blockSQP2::SQPmethod::param)
    .def("set_iterate_", &blockSQP2::SQPmethod::set_iterate)
    .def("get_xi", static_cast<blockSQP2::Matrix (blockSQP2::SQPmethod::*)()>(&blockSQP2::SQPmethod::get_xi), py::return_value_policy::take_ownership)
    .def("get_lambda", static_cast<blockSQP2::Matrix (blockSQP2::SQPmethod::*)()>(&blockSQP2::SQPmethod::get_lambda), py::return_value_policy::take_ownership)
    .def("get_scaleFactors", [](blockSQP2::SQPmethod &M){if (M.param->automatic_scaling) return double_pointer_interface(M.scaled_prob->scaling_factors.get(), M.scaled_prob->nVar); else return double_pointer_interface();}, py::return_value_policy::take_ownership)
    .def("get_rescaleFactors", [](blockSQP2::SQPmethod &M){if (M.param->automatic_scaling) return double_pointer_interface(M.vars->rescaleFactors.get(), M.scaled_prob->nVar); else return double_pointer_interface();}, py::return_value_policy::take_ownership)
    .def("arr_apply_rescaling", [](blockSQP2::SQPmethod &M, double_array *arr){if (M.param->automatic_scaling){M.apply_rescaling(arr->ptr);} return;})
    .def("dec_nquasi", [](blockSQP2::SQPmethod &M){for (int iBlock = 0; iBlock < M.vars->nBlocks; iBlock++){if (M.vars->nquasi[iBlock] > 0) M.vars->nquasi[iBlock] -= 1;} return;})
    ;


py::class_<blockSQP2::bound_correction_method, blockSQP2::SQPmethod>(m, "bound_correction_method")
    .def(py::init<blockSQP2::Problemspec*, blockSQP2::SQPoptions*, blockSQP2::SQPstats*>())
    ;


py::class_<blockSQP2::SQPiterate>(m, "SQPiterate")
	.def_readonly("obj", &blockSQP2::SQPiterate::obj)
	.def_readonly("cNorm", &blockSQP2::SQPiterate::cNorm)
	.def_readonly("cNormS", &blockSQP2::SQPiterate::cNormS)
	.def_readonly("gradNorm", &blockSQP2::SQPiterate::gradNorm)
	.def_readonly("lambdaStepNorm", &blockSQP2::SQPiterate::lambdaStepNorm)
	.def_readonly("tol", &blockSQP2::SQPiterate::tol)
	.def_readonly("xi", &blockSQP2::SQPiterate::xi)
	.def_readonly("lam", &blockSQP2::SQPiterate::lambda)
	.def_readonly("constr", &blockSQP2::SQPiterate::constr)
	.def_readonly("constrJac", &blockSQP2::SQPiterate::constrJac)
	.def_readonly("trialXi", &blockSQP2::SQPiterate::trialXi)
	.def_readwrite("use_homotopy", &blockSQP2::SQPiterate::use_homotopy)
	.def("get_hess1_block", [](blockSQP2::SQPiterate &vars, int i){return vars.hess1[i];})
	.def("get_hess2_block", [](blockSQP2::SQPiterate &vars, int i){return vars.hess2[i];})
	.def_readonly("filter", &blockSQP2::SQPiterate::filter)
	.def_readonly("gradLagrange", &blockSQP2::SQPiterate::gradLagrange)
    .def_readonly("gradObj", &blockSQP2::SQPiterate::gradObj)
	.def_readonly("gammaMat", &blockSQP2::SQPiterate::gammaMat)
	.def_readonly("deltaXi", &blockSQP2::SQPiterate::deltaXi)
	.def_readonly("deltaMat", &blockSQP2::SQPiterate::deltaMat)
    .def_readonly("deltaNormSqMat", &blockSQP2::SQPiterate::deltaNormSqMat)
    .def_readonly("deltaGammaMat", &blockSQP2::SQPiterate::deltaGammaMat)
    .def_readonly("dg_pos", &blockSQP2::SQPiterate::dg_pos)
    .def("print_hess2", [](blockSQP2::SQPiterate *vars){
        if (vars->hess2 == nullptr) return;
        for (int i = 0; i < vars->nBlocks; i++){
            std::cout << vars->hess2[i] << "\n";
        }
    })
    .def_readonly("hess_num_accepted", &blockSQP2::SQPiterate::hess_num_accepted)                   
    .def_readonly("QP_num_accepted", &blockSQP2::SQPiterate::QP_num_accepted)
    .def("set_hess1", [](blockSQP2::SQPiterate &vars, SymMat_array &arr){
        for (int i = 0; i < vars.nBlocks; i++){
            vars.hess1[i] = arr.ptr[i];
        }
    })
    .def("set_hess2", [](blockSQP2::SQPiterate &vars, SymMat_array &arr){
        for (int i = 0; i < vars.nBlocks; i++){
            vars.hess2[i] = arr.ptr[i];
        }
    })
    .def("set_hess1_block", [](blockSQP2::SQPiterate &vars, int i, blockSQP2::SymMatrix &M){vars.hess1[i] = M;})
	.def("set_hess2_block", [](blockSQP2::SQPiterate &vars, int i, blockSQP2::SymMatrix &M){vars.hess2[i] = M;})
	;

py::class_<blockSQP2::RestorationProblem, blockSQP2::Problemspec>(m, "RestorationProblem")
        .def(py::init<blockSQP2::Problemspec*, blockSQP2::Matrix&, double, double>())
        ;

    
//Condensing classes and structs
py::class_<blockSQP2::vblock>(m, "vblock")
    .def(py::init<int, bool>())
    .def_readwrite("size", &blockSQP2::vblock::size)
    .def_readwrite("dependent", &blockSQP2::vblock::dependent);

py::class_<vblock_array>(m, "vblock_array")
	.def(py::init<>())
	.def(py::init<int>())
	.def_readonly("size", &vblock_array::size)
	.def("resize", &vblock_array::resize)
	.def("__getitem__", [](vblock_array &arr, int i) -> blockSQP2::vblock *{return arr.ptr + i;}, py::return_value_policy::reference)
	.def("__setitem__", [](vblock_array &arr, int i, blockSQP2::vblock B) -> void{arr.ptr[i] = B;});

py::class_<blockSQP2::cblock>(m, "cblock")
    .def(py::init<int>())
    .def_readwrite("size", &blockSQP2::cblock::size);

py::class_<cblock_array>(m, "cblock_array")
    .def(py::init<>())
    .def(py::init<int>())
    .def_readonly("size", &cblock_array::size)
    .def("resize", &cblock_array::resize)
	.def("__getitem__", [](cblock_array &arr, int i) -> blockSQP2::cblock *{return arr.ptr + i;}, py::return_value_policy::reference)
	.def("__setitem__", [](cblock_array &arr, int i, blockSQP2::cblock B) -> void{arr.ptr[i] = B;});

py::class_<blockSQP2::condensing_target>(m, "condensing_target")
    .def(py::init<int, int, int, int, int>())
    .def_readwrite("n_stages", &blockSQP2::condensing_target::n_stages)
    .def_readwrite("first_free", &blockSQP2::condensing_target::first_free)
    .def_readwrite("vblock_end", &blockSQP2::condensing_target::vblock_end)
    .def_readwrite("first_cond", &blockSQP2::condensing_target::first_cond)
    .def_readwrite("cblock_end", &blockSQP2::condensing_target::cblock_end);

py::class_<condensing_targets>(m, "condensing_targets")
    .def(py::init<>())
    .def(py::init<int>())
    .def_readonly("size", &condensing_targets::size)
    .def("resize", &condensing_targets::resize)
	.def("__getitem__", [](condensing_targets &arr, int i) -> blockSQP2::condensing_target *{return arr.ptr + i;}, py::return_value_policy::reference)
	.def("__setitem__", [](condensing_targets &arr, int i, blockSQP2::condensing_target B) -> void{arr.ptr[i] = B;});

py::class_<blockSQP2::Condenser>(m, "Condenser")
    .def(py::init(
        [](vblock_array &v_arr, cblock_array &c_arr, int_array &hess_sizes, condensing_targets &c_targets, int add_dep_bounds) -> blockSQP2::Condenser*
        {return new blockSQP2::Condenser(v_arr.ptr, v_arr.size, c_arr.ptr, c_arr.size, hess_sizes.ptr, hess_sizes.size, c_targets.ptr, c_targets.size, add_dep_bounds);}),
        py::arg("vBlocks"), py::arg("cBlocks"), py::arg("hBlocks"), py::arg("Targets"), py::arg("add_dep_bounds") = 2, py::return_value_policy::take_ownership)
    .def("print_info", &blockSQP2::Condenser::print_info)
    .def("condense_args", [](blockSQP2::Condenser &C, condensing_args &args) -> void{
            args.C = &C;
            args.condensed_hess.resize(C.condensed_num_hessblocks);
            steady_clock::time_point T0 = steady_clock::now();
            C.full_condense(args.grad_obj, args.con_jac, args.hess.ptr, args.lb_var, args.ub_var, args.lb_con, args.ub_con,
                args.condensed_h, args.condensed_Jacobian, args.condensed_hess.ptr, args.condensed_lb_var, args.condensed_ub_var, args.condensed_lb_con, args.condensed_ub_con
            );
            steady_clock::time_point T1 = steady_clock::now();
            std::cout << "Condensing took " << duration_cast<microseconds>(T1 - T0) << "\n";
            return;})
    .def_readonly("num_hessblocks", &blockSQP2::Condenser::num_hessblocks)
    .def_readonly("num_vars", &blockSQP2::Condenser::num_vars)
    .def_readonly("num_cons", &blockSQP2::Condenser::num_cons)
    .def_readonly("condensed_num_hessblocks", &blockSQP2::Condenser::condensed_num_hessblocks)
    .def_readonly("condensed_num_vars", &blockSQP2::Condenser::condensed_num_vars)
    .def_readonly("num_true_cons", &blockSQP2::Condenser::num_true_cons);

py::class_<SymMat_array>(m, "SymMat_array")
    .def(py::init<>())
    .def(py::init<int>())
    .def_readonly("size", &SymMat_array::size)
    .def("resize", &SymMat_array::resize)
	//.def("__getitem__", [](SymMat_array &arr, int i) -> blockSQP2::SymMatrix *{return arr.ptr + i;}, py::return_value_policy::reference)
	.def("__getitem__", [](SymMat_array &arr, int i) -> blockSQP2::SymMatrix *{return arr.ptr + i;}, py::return_value_policy::reference)
	.def("__setitem__", [](SymMat_array &arr, int i, blockSQP2::SymMatrix &B) -> void{arr.ptr[i] = B;});


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
    .def_readonly("deltaXi", &condensing_args::deltaXi)
    .def_readonly("lambdaQP", &condensing_args::lambdaQP)
    .def_readonly("deltaXi_cond", &condensing_args::deltaXi_cond)
    .def_readonly("lambdaQP_cond", &condensing_args::lambdaQP_cond)
    .def_readonly("deltaXi_rest", &condensing_args::deltaXi_rest)
    .def_readonly("lambdaQP_rest", &condensing_args::lambdaQP_rest)
    .def("solve_QPs", &condensing_args::solve_QPs)
    ;


py::class_<blockSQP2::TC_restoration_Problem, blockSQP2::Problemspec>(m, "TC_restoration_Problem")
    .def(py::init<blockSQP2::Problemspec*, const blockSQP2::Matrix&, double, double>())
    .def_readonly("nVar", &blockSQP2::TC_restoration_Problem::nVar)
    .def_readonly("nCon", &blockSQP2::TC_restoration_Problem::nCon)
    .def_readwrite("xi_ref", &blockSQP2::TC_restoration_Problem::xi_ref)
    ;

py::class_<blockSQP2::TC_feasibility_Problem, blockSQP2::Problemspec>(m, "TC_feasibility_Problem")
    .def(py::init<blockSQP2::Problemspec*>())
    .def_readonly("nVar", &blockSQP2::TC_feasibility_Problem::nVar)
    .def_readonly("nCon", &blockSQP2::TC_feasibility_Problem::nCon)
	.def_readwrite("nnz", &blockSQP2::TC_feasibility_Problem::nnz)
	.def_readwrite("objLo", &blockSQP2::TC_feasibility_Problem::objLo)
	.def_readwrite("objUp", &blockSQP2::TC_feasibility_Problem::objUp)
	.def_readwrite("lb_var", &blockSQP2::TC_feasibility_Problem::lb_var)
	.def_readwrite("ub_var", &blockSQP2::TC_feasibility_Problem::ub_var)
	.def_readwrite("lb_con", &blockSQP2::TC_feasibility_Problem::lb_con)
	.def_readwrite("ub_con", &blockSQP2::TC_feasibility_Problem::ub_con)
	.def_readonly("nBlocks", &blockSQP2::TC_feasibility_Problem::nBlocks)
	.def_readonly("xi_orig", &blockSQP2::TC_feasibility_Problem::xi_parent)
	.def_readonly("slack", &blockSQP2::TC_feasibility_Problem::slack)
	.def_property("jac_orig_nz", [](blockSQP2::TC_feasibility_Problem &P)->double_pointer_interface{double_pointer_interface nonzeros; nonzeros.size = P.nnz; nonzeros.ptr = P.jac_orig_nz; return nonzeros;}, nullptr)
	.def_property("jac_orig_row", [](blockSQP2::TC_feasibility_Problem &P)->int_pointer_interface{int_pointer_interface row; row.size = P.nnz; row.ptr = P.jac_orig_row; return row;}, nullptr)
	.def_property("jac_orig_colind", [](blockSQP2::TC_feasibility_Problem &P)->int_pointer_interface{int_pointer_interface colind; colind.size = P.nVar + 1; colind.ptr = P.jac_orig_colind; return colind;}, nullptr)
	;

py::class_<blockSQP2::scaled_Problemspec, blockSQP2::Problemspec>(m, "scaled_Problemspec")
    .def(py::init<blockSQP2::Problemspec*>())
    .def("arr_set_scale", [](blockSQP2::scaled_Problemspec &P, double_array &arr){P.set_scale(arr.ptr);});
}















