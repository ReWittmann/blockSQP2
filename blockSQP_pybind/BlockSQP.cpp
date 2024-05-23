#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <tuple>
#include <iostream>
#include <fstream>
#include <chrono>
#include "blocksqp_method.hpp"
#include "blocksqp_condensing.hpp"
#include "qpOASES.hpp"
#include "blocksqp_restoration.hpp"

namespace py = pybind11;


class int_array{
    public:
    int size = 0;
    int *ptr;

    public:
    int_array(){
        ptr = nullptr;
        size = 0;
    }

    int_array(int size_): size(size_){
        ptr = new int[size];
    }

    ~int_array(){
    delete[] ptr;
    }

    void resize(int size_){
        delete[] ptr;
        size = size_;
        ptr = new int[size];
    }
};

class double_array{
    public:
    int size = 0;
    double *ptr;

    public:
    double_array(){
        ptr = nullptr;
        size = 0;
    }

    double_array(int size_): size(size_){
        ptr = new double[size];
    }

    ~double_array(){
        delete[] ptr;
    }

    void resize(int size_){
        delete[] ptr;
        size = size_;
        ptr = new double[size];
    }
};


class int_pointer_interface{
    public:
    int size = 0;
    int *ptr;

    public:
    int_pointer_interface(){
        ptr = nullptr;
        size = 0;
    }
};

class double_pointer_interface{
    public:
    int size = 0;
    double *ptr;

    public:
    double_pointer_interface(){
        ptr = nullptr;
        size = 0;
    }
};

class vblock_array{
    public:
    int size = 0;
    blockSQP::vblock *ptr;

    vblock_array(int size_): size(size_){
        ptr = new blockSQP::vblock[size];
    }

    vblock_array(){
        ptr = nullptr;
        size = 0;
    }

    ~vblock_array(){
        delete[] ptr;
    }

    void resize(int size_){
        delete[] ptr;
        size = size_;
        ptr = new blockSQP::vblock[size];
    }
};

class cblock_array{
    public:
    int size = 0;
    blockSQP::cblock *ptr;

    cblock_array(int size_): size(size_){
        ptr = new blockSQP::cblock[size];
    }

    cblock_array(){
        ptr = nullptr;
        size = 0;
    }

    ~cblock_array(){
        delete[] ptr;
    }

    void resize(int size_){
        delete[] ptr;
        size = size_;
        ptr = new blockSQP::cblock[size];
    }
};

class condensing_targets{
    public:
    int size = 0;
    blockSQP::condensing_target *ptr;

    condensing_targets(int size_): size(size_){
        ptr = new blockSQP::condensing_target[size];
    }

    condensing_targets(){
        ptr = nullptr;
        size = 0;
    }

    ~condensing_targets(){
        delete[] ptr;
    }

     void resize(int size_){
        delete[] ptr;
        size = size_;
        ptr = new blockSQP::condensing_target[size];
    }
};

class SymMat_array{
public:
    int size = 0;
    blockSQP::SymMatrix *ptr;

    SymMat_array(int size_): size(size_){
        ptr = new blockSQP::SymMatrix[size];
    }

    SymMat_array(){
        ptr = nullptr;
        size = 0;
    }

    ~SymMat_array(){
        delete[] ptr;
    }

    void resize(int size_){
        delete[] ptr;
        size = size_;
        ptr = new blockSQP::SymMatrix[size];
    }
};


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


    void solve_QPs(){
        double a = 0;
        for (int i = 0; i < condensed_Jacobian.nnz; i++){
            a += condensed_Jacobian.nz[i];
        }
        std::cout << "Sum on nonzeros in condensed_Jacobian is " << a << "\n";

        blockSQP::Sparse_Matrix red_con_jac = condensed_Jacobian;
        blockSQP::Matrix red_lb_con = condensed_lb_con;
        blockSQP::Matrix red_ub_con = condensed_ub_con;

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
                    con_jac.row, con_jac.colind, con_jac.nz);
        A_qp_cond = new qpOASES::SparseMatrix(red_con_jac.m, red_con_jac.n,
                    red_con_jac.row, red_con_jac.colind, red_con_jac.nz);

        convertHessian(1.0e-15, hess.ptr, hess.size, con_jac.n, hess_nz, hess_row, hess_colind, hess_loind);
        convertHessian(1.0e-15, condensed_hess.ptr, C->condensed_num_hessblocks, red_con_jac.n, hess_cond_nz, hess_cond_row, hess_cond_colind, hess_cond_loind);
        std::cout << "converted Hessians\n";

        double S_h_nz = 0;
        for (int i = 0; i < hess_cond_colind[C->condensed_num_vars]; i++){
            S_h_nz += hess_cond_nz[i];
        }
        std::cout << "Sum hess_cond_nz = " << S_h_nz << "\n";


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
        opts.printLevel = qpOASES::PL_LOW; //PL_LOW, PL_HIGH, PL_MEDIUM, PL_None
        opts.numRefinementSteps = 2;
        opts.epsLITests =  2.2204e-08;
        qp->setOptions( opts );
        qp_cond->setOptions( opts);

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

    }

};


class TC_restoration_Test{
public:
    blockSQP::TC_restoration_Problem *rest_Prob;
    double_array rest_nz;
    int_array rest_row;
    int_array rest_colind;

    blockSQP::Matrix xi_rest;
    blockSQP::Matrix lambda_rest;

    TC_restoration_Test(blockSQP::TC_restoration_Problem *RP): rest_Prob(RP){
        xi_rest.Dimension(rest_Prob->nVar);
        lambda_rest.Dimension(rest_Prob->nVar + rest_Prob->nCon);

        rest_nz.resize(rest_Prob->nnz);
        rest_row.resize(rest_Prob->nnz);
        rest_colind.resize(rest_Prob->nVar + 1);
    }

    void init(){
        rest_Prob->initialize(xi_rest, lambda_rest, rest_nz.ptr, rest_row.ptr, rest_colind.ptr);
    }
};



/*
struct Prob_Data{


	blockSQP::Matrix xi;			///< optimization variables
	blockSQP::Matrix lambda;		///< Lagrange multipliers
	double objval;				///< objective function value
	blockSQP::Matrix constr;		///< constraint function values
	blockSQP::Matrix gradObj;		///< gradient of objective
	blockSQP::Matrix constrJac;		///< constraint Jacobian (dense)
	double_pointer_interface jacNz;		///< nonzero elements of constraint Jacobian
	int_pointer_interface jacIndRow;		///< row indices of nonzero elements
	int_pointer_interface jacIndCol;		///< starting indices of columns
	blockSQP::SymMatrix *hess;		///< Hessian of the Lagrangian (blockwise)
	int dmode;				///< derivative mode
	int info;				///< error flag
};*/

struct Prob_Data{
    double_pointer_interface xi;			///< optimization variables
    double_pointer_interface lambda;		///< Lagrange multipliers
    double objval;				            ///< objective function value
    double_pointer_interface constr;		///< constraint function values
    double_pointer_interface gradObj;		///< gradient of objective
    //double_pointer_interface constrJac;		///< constraint Jacobian (dense)
    blockSQP::Matrix constrJac;
    double_pointer_interface jacNz;		    ///< nonzero elements of constraint Jacobian
    int_pointer_interface jacIndRow;		///< row indices of nonzero elements
	int_pointer_interface jacIndCol;		///< starting indices of columns
    blockSQP::SymMatrix *hess;		        ///< Hessian of the Lagrangian (blockwise)
    int dmode;				                ///< derivative mode
    int info;				                ///< error flag
};


class Problemform : public blockSQP::Problemspec
{
public:
    virtual ~Problemform(){};

    void call_initialize(){
    initialize_dense();
    }

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

        Cpp_Data.info = 0;
    }


    /*
    void init_Cpp_Data(bool Sparse_QP, int nnz){

        Cpp_Data.xi.m = nVar; Cpp_Data.xi.ldim = -1; Cpp_Data.xi.n = 1; Cpp_Data.xi.tflag = 0;
        Cpp_Data.lambda.m = nVar + nCon; Cpp_Data.lambda.ldim = -1; Cpp_Data.lambda.n = 1; Cpp_Data.lambda.tflag = 0;
        Cpp_Data.gradObj.m = nVar; Cpp_Data.gradObj.ldim = -1; Cpp_Data.gradObj.n = 1; Cpp_Data.gradObj.tflag = 0;
        Cpp_Data.constr.m = nCon; Cpp_Data.constr.ldim = -1; Cpp_Data.constr.n = 1; Cpp_Data.constr.tflag = 0;

        if (not Sparse_QP){
            Cpp_Data.constrJac.m = nCon; Cpp_Data.constrJac.ldim = -1; Cpp_Data.constrJac.n = nVar; Cpp_Data.constrJac.tflag = 0;
    //		Cpp_Data.constrJac.Dimension(nCon, nVar).Initialize(0.0);
        }
        else{
            Cpp_Data.jacNz.resize(nnz);
            Cpp_Data.jacIndRow.resize(nnz);
            Cpp_Data.jacIndCol.resize(nVar+1);
        }

        Cpp_Data.info = 0;
    }
    */

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


    void initialize(blockSQP::Matrix &xi, blockSQP::Matrix &lambda, double *&jacNz, int *&jacIndRow, int *&jacIndCol) override {

        Cpp_Data.xi.ptr = xi.array;
        Cpp_Data.lambda.ptr = lambda.array;
        Cpp_Data.jacIndRow.ptr = jacIndRow;
        Cpp_Data.jacIndCol.ptr = jacIndCol;

        update_inits();
        initialize_sparse();
    }


    void evaluate(const blockSQP::Matrix &xi, const blockSQP::Matrix &lambda, double *objval, blockSQP::Matrix &constr, blockSQP::Matrix &gradObj, blockSQP::Matrix &constrJac, blockSQP::SymMatrix *&hess, int dmode, int *info) override {

        Cpp_Data.xi.ptr = xi.array;
        Cpp_Data.lambda.ptr = lambda.array;
        Cpp_Data.dmode = dmode;
        Cpp_Data.constr.ptr = constr.array;
        Cpp_Data.gradObj.ptr = gradObj.array;
        Cpp_Data.constrJac.array = constrJac.array;

        update_evals();
        evaluate_dense();
        get_objval();

        *objval = Cpp_Data.objval;
        *info = 0;
    }


    void evaluate(const blockSQP::Matrix &xi, const blockSQP::Matrix &lambda, double *objval, blockSQP::Matrix &constr, blockSQP::Matrix &gradObj, double *&jacNz, int *&jacIndRow, int *&jacIndCol, blockSQP::SymMatrix *&hess, int dmode, int *info) override {

        Cpp_Data.xi.ptr = xi.array;
        Cpp_Data.lambda.ptr = lambda.array;
        Cpp_Data.dmode = dmode;
        Cpp_Data.constr.ptr = constr.array;
        Cpp_Data.gradObj.ptr = gradObj.array;
        Cpp_Data.jacNz.ptr = jacNz;
        Cpp_Data.jacIndRow.ptr = jacIndRow;
        Cpp_Data.jacIndCol.ptr = jacIndCol;

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
        blockIdx = (int*)buff.ptr;
        nBlocks = buff.size - 1;
    }


    void reduceConstrVio(blockSQP::Matrix &xi, int *info){
        Cpp_Data.xi.ptr = xi.array;
        update_xi();
        std::cout << "reduceConstrVio called\n";
        restore_continuity();

        *info = Cpp_Data.info;
        std::cout << "info is " << *info << "\n";
        //*info = 1;
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



PYBIND11_MODULE(BlockSQP, m) {

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

py::class_<blockSQP::SymMatrix, blockSQP::Matrix>(m, "SymMatrix")
    .def(py::init<>())
    .def(py::init<int,int,int>(), py::arg("M") = 1, py::arg("N") = 1, py::arg("LDIM") = -1)
    .def(py::init<const blockSQP::Matrix&>())
	.def("Dimension", [](blockSQP::SymMatrix &M1, int m, int n) -> void{M1.Dimension(m,n,m);})
	.def("Initialize", static_cast<blockSQP::SymMatrix& (blockSQP::SymMatrix::*)(double)>(&blockSQP::SymMatrix::Initialize))
	.def_readwrite("ldim",&blockSQP::SymMatrix::ldim)
	.def_readwrite("m",&blockSQP::SymMatrix::m)
	.def_readwrite("n",&blockSQP::SymMatrix::n)
	.def("__setitem__", [](blockSQP::SymMatrix &M, std::tuple<int, int> inds, double val) -> void{M(std::get<0>(inds), std::get<1>(inds)) = val; return;})
	.def("__getitem__", [](blockSQP::SymMatrix &M, std::tuple<int, int> inds) -> double{return M(std::get<0>(inds), std::get<1>(inds));})
    .def("set", [](blockSQP::SymMatrix &M1, int i, int j, const blockSQP::Matrix &M2) -> void{
        M1.Dimension(M2.m, M2.n, M2.m);
        for (int i = 0; i < M2.m; i++){
            for (int j = 0; j <= i; j++){
                M1(i,j) = M1(i,j);
            }
        }
        return;
    });

py::class_<blockSQP::Sparse_Matrix>(m, "Sparse_Matrix")
    .def(py::init<>())
    .def(py::init([](int M, int N, int nnz, double_array &nz, int_array &row, int_array &colind) -> blockSQP::Sparse_Matrix*{
        double *NZ = nz.ptr; int *ROW = row.ptr; int *COLIND = colind.ptr;
        nz.size = 0; nz.ptr = nullptr; row.size = 0; row.ptr = nullptr; colind.size = 0; colind.ptr = nullptr;
        return new blockSQP::Sparse_Matrix(M, N, nnz, NZ, ROW, COLIND);
    }), py::return_value_policy::take_ownership)
    .def_readonly("m", &blockSQP::Sparse_Matrix::m)
    .def_readonly("n", &blockSQP::Sparse_Matrix::n)
    .def_readonly("nnz", &blockSQP::Sparse_Matrix::nnz)
    .def("dense", &blockSQP::Sparse_Matrix::dense)
    .def_property("NZ", [](blockSQP::Sparse_Matrix &M)->double_pointer_interface{double_pointer_interface nonzeros; nonzeros.size = M.nnz; nonzeros.ptr = M.nz; return nonzeros;}, nullptr)
    .def_property("ROW", [](blockSQP::Sparse_Matrix &M)->int_pointer_interface{int_pointer_interface row; row.size = M.nnz; row.ptr = M.row; return row;}, nullptr)
    .def_property("COLIND", [](blockSQP::Sparse_Matrix &M)->int_pointer_interface{int_pointer_interface colind; colind.size = M.n + 1; colind.ptr = M.colind; return colind;}, nullptr)
    ;

py::class_<blockSQP::SQPoptions>(m, "SQPoptions")
	.def(py::init<>())
	.def("optionsConsistency",&blockSQP::SQPoptions::optionsConsistency)
	.def_readwrite("printLevel",&blockSQP::SQPoptions::printLevel)
	.def_readwrite("printColor",&blockSQP::SQPoptions::printColor)
	.def_readwrite("debugLevel",&blockSQP::SQPoptions::debugLevel)
	.def_readwrite("qpOASES_print_level", &blockSQP::SQPoptions::qpOASES_print_level)
	.def_readwrite("qpOASES_terminationTolerance", &blockSQP::SQPoptions::qpOASES_terminationTolerance)
	.def_readwrite("eps",&blockSQP::SQPoptions::eps)
	.def_readwrite("inf",&blockSQP::SQPoptions::inf)
	.def_readwrite("opttol",&blockSQP::SQPoptions::opttol)
	.def_readwrite("nlinfeastol",&blockSQP::SQPoptions::nlinfeastol)
	.def_readwrite("sparseQP",&blockSQP::SQPoptions::sparseQP)
	.def_readwrite("globalization",&blockSQP::SQPoptions::globalization)
	.def_readwrite("restoreFeas",&blockSQP::SQPoptions::restoreFeas)
	.def_readwrite("maxLineSearch",&blockSQP::SQPoptions::maxLineSearch)
	.def_readwrite("maxConsecReducedSteps",&blockSQP::SQPoptions::maxConsecReducedSteps)
	.def_readwrite("maxConsecSkippedUpdates",&blockSQP::SQPoptions::maxConsecSkippedUpdates)
	.def_readwrite("maxItQP",&blockSQP::SQPoptions::maxItQP)
	.def_readwrite("blockHess",&blockSQP::SQPoptions::blockHess)
	.def_readwrite("hessScaling",&blockSQP::SQPoptions::hessScaling)
	.def_readwrite("fallbackScaling",&blockSQP::SQPoptions::fallbackScaling)
	.def_readwrite("maxTimeQP",&blockSQP::SQPoptions::maxTimeQP)
	.def_readwrite("iniHessDiag",&blockSQP::SQPoptions::iniHessDiag)
	.def_readwrite("colEps",&blockSQP::SQPoptions::colEps)
	.def_readwrite("colTau1",&blockSQP::SQPoptions::colTau1)
	.def_readwrite("colTau2",&blockSQP::SQPoptions::colTau2)
	.def_readwrite("hessDamp",&blockSQP::SQPoptions::hessDamp)
	.def_readwrite("hessDampFac",&blockSQP::SQPoptions::hessDampFac)
	.def_readwrite("hessUpdate",&blockSQP::SQPoptions::hessUpdate)
	.def_readwrite("fallbackUpdate",&blockSQP::SQPoptions::fallbackUpdate)
	.def_readwrite("hessLimMem",&blockSQP::SQPoptions::hessLimMem)
	.def_readwrite("hessMemsize",&blockSQP::SQPoptions::hessMemsize)
	.def_readwrite("whichSecondDerv",&blockSQP::SQPoptions::whichSecondDerv)
	.def_readwrite("skipFirstGlobalization",&blockSQP::SQPoptions::skipFirstGlobalization)
	.def_readwrite("convStrategy",&blockSQP::SQPoptions::convStrategy)
	.def_readwrite("maxConvQP",&blockSQP::SQPoptions::maxConvQP)
	.def_readwrite("maxSOCiter",&blockSQP::SQPoptions::maxSOCiter)
	.def_readwrite("max_bound_refines", &blockSQP::SQPoptions::max_bound_refines)
	.def_readwrite("max_correction_steps", &blockSQP::SQPoptions::max_correction_steps)
	.def_readwrite("dep_bound_tolerance", &blockSQP::SQPoptions::dep_bound_tolerance)
	.def_readwrite("integration_correction_stepsize", &blockSQP::SQPoptions::integration_correction_stepsize)
	//Test options
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
	.def_readwrite("info", &Prob_Data::info);

py::class_<blockSQP::Problemspec>(m, "Problemspec");

py::class_<Problemform, blockSQP::Problemspec, Py_Problemform>(m,"Problemform", py::buffer_protocol())
	.def(py::init<>())
//##############
	.def("call_initialize", &Problemform::call_initialize)
//##############
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
//	.def(py::init([](Problemform* prob, blockSQP::SQPoptions* opts, blockSQP::SQPstats* stats){return new blockSQP::SQPmethod(prob, opts, stats); }), py::return_value_policy::take_ownership)
	.def_readonly("vars", &blockSQP::SQPmethod::vars)
	.def_readonly("stats", &blockSQP::SQPmethod::stats)
	.def("init", &blockSQP::SQPmethod::init)
	.def("run", &blockSQP::SQPmethod::run, py::arg("maxIt"), py::arg("warmStart") = 0)
	.def("finish", &blockSQP::SQPmethod::finish)
    .def("resetHessian", static_cast<void (blockSQP::SQPmethod::*)()>(&blockSQP::SQPmethod::resetHessian))
    .def_readwrite("param", &blockSQP::SQPmethod::param)
    ;

py::class_<blockSQP::SCQPmethod, blockSQP::SQPmethod>(m, "SCQPmethod")
    .def(py::init<blockSQP::Problemspec*, blockSQP::SQPoptions*, blockSQP::SQPstats*, blockSQP::Condenser*>())
    ;

py::class_<blockSQP::SCQP_bound_method, blockSQP::SCQPmethod>(m, "SCQP_bound_method")
    .def(py::init<blockSQP::Problemspec*, blockSQP::SQPoptions*, blockSQP::SQPstats*, blockSQP::Condenser*>())
    ;

py::class_<blockSQP::SCQP_correction_method, blockSQP::SCQPmethod>(m, "SCQP_correction_method")
    .def(py::init<blockSQP::Problemspec*, blockSQP::SQPoptions*, blockSQP::SQPstats*, blockSQP::Condenser*>())
    ;

py::class_<blockSQP::SQPiterate>(m, "SQPiterate")
	.def_readonly("obj", &blockSQP::SQPiterate::obj)
	//.def_readonly("qpObj", &blockSQP::SQPiterate::qpObj)
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
	.def("__getitem__", [](vblock_array &arr, int i) -> blockSQP::vblock{return arr.ptr[i];})
	.def("__setitem__", [](vblock_array &arr, int i, blockSQP::vblock B) -> void{arr.ptr[i] = B;});

py::class_<blockSQP::cblock>(m, "cblock")
    .def(py::init<int>())
    .def_readwrite("size", &blockSQP::cblock::size);

py::class_<cblock_array>(m, "cblock_array")
    .def(py::init<>())
    .def(py::init<int>())
    .def_readonly("size", &cblock_array::size)
    .def("resize", &cblock_array::resize)
	.def("__getitem__", [](cblock_array &arr, int i) -> blockSQP::cblock{return arr.ptr[i];})
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
	.def("__getitem__", [](condensing_targets &arr, int i) -> blockSQP::condensing_target{return arr.ptr[i];})
	.def("__setitem__", [](condensing_targets &arr, int i, blockSQP::condensing_target B) -> void{arr.ptr[i] = B;});

py::class_<blockSQP::Condenser>(m, "Condenser")
    .def(py::init(
        [](vblock_array &v_arr, cblock_array &c_arr, int_array &hess_sizes, condensing_targets &c_targets, int add_dep_bounds) -> blockSQP::Condenser*
        {return new blockSQP::Condenser(v_arr.ptr, v_arr.size, c_arr.ptr, c_arr.size, hess_sizes.ptr, hess_sizes.size, c_targets.ptr, c_targets.size, add_dep_bounds);}),
        py::arg("vBlocks"), py::arg("cBlocks"), py::arg("hBlocks"), py::arg("Targets"), py::arg("add_dep_bounds") = 2, py::return_value_policy::take_ownership)
    .def("print_debug", &blockSQP::Condenser::print_debug)
    .def("condense_args", [](blockSQP::Condenser &C, condensing_args &args) -> void{
            args.C = &C;
            C.full_condense(args.grad_obj, args.con_jac, args.hess.ptr, args.lb_var, args.ub_var, args.lb_con, args.ub_con,
                args.condensed_h, args.condensed_Jacobian, args.condensed_hess.ptr, args.condensed_lb_var, args.condensed_ub_var, args.condensed_lb_con, args.condensed_ub_con
            );
            args.condensed_hess.size = C.condensed_num_hessblocks;
            return;} )
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
	.def("__getitem__", [](SymMat_array &arr, int i) -> blockSQP::SymMatrix {return arr.ptr[i];})
	.def("__setitem__", [](SymMat_array &arr, int i, blockSQP::SymMatrix B) -> void{arr.ptr[i] = B;});

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
    .def(py::init<blockSQP::Problemspec*, blockSQP::Condenser*, const blockSQP::Matrix&>())
    .def_readonly("nVar", &blockSQP::TC_restoration_Problem::nVar)
    .def_readonly("nCon", &blockSQP::TC_restoration_Problem::nCon)
    .def_readwrite("xi_ref", &blockSQP::TC_restoration_Problem::xi_ref)
    ;

py::class_<TC_restoration_Test>(m, "TC_restoration_Test")
    .def(py::init<blockSQP::TC_restoration_Problem *>())
    .def("init", &TC_restoration_Test::init)
    .def_readonly("rest_nz", &TC_restoration_Test::rest_nz)
    .def_readonly("rest_row", &TC_restoration_Test::rest_row)
    .def_readonly("rest_colind", &TC_restoration_Test::rest_colind);


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
	.def_readonly("xi_orig", &blockSQP::TC_feasibility_Problem::xi_orig)
	.def_readonly("slack", &blockSQP::TC_feasibility_Problem::slack)
	.def_property("jac_orig_nz", [](blockSQP::TC_feasibility_Problem &P)->double_pointer_interface{double_pointer_interface nonzeros; nonzeros.size = P.nnz; nonzeros.ptr = P.jac_orig_nz; return nonzeros;}, nullptr)
	.def_property("jac_orig_row", [](blockSQP::TC_feasibility_Problem &P)->int_pointer_interface{int_pointer_interface row; row.size = P.nnz; row.ptr = P.jac_orig_row; return row;}, nullptr)
	.def_property("jac_orig_colind", [](blockSQP::TC_feasibility_Problem &P)->int_pointer_interface{int_pointer_interface colind; colind.size = P.nVar + 1; colind.ptr = P.jac_orig_colind; return colind;}, nullptr)
	;

}















