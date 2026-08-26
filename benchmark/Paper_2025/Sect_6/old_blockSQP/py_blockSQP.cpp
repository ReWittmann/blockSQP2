/*
 * blockSQP_reference_build - build system and Python interface for blockSQP
 * Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
 *
 * Licensed under the zlib license. See LICENSE for more details.
 */

/**
 * \file py_blockSQP.cpp
 * \author Reinhold Wittmann
 * \date 2022-2025
 *
 * Implementation of a python interface to the blockSQP 
 * nonlinear programming solver - C++ side part
 */


#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "blocksqp_method.hpp"

namespace py = pybind11;



class int_array_interface{
public:
int size = 0;
int *ptr;
};

class double_array_interface{
public:
int size = 0;
double *ptr;
};



//From new version (less questionable code)
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

typedef T_array<double_array_interface> double_pointer_interface_array;




struct Prob_Data{

	
	blockSQP::Matrix xi;			///< optimization variables
	blockSQP::Matrix lambda;		///< Lagrange multipliers
	double objval;				///< objective function value
	blockSQP::Matrix constr;		///< constraint function values
	blockSQP::Matrix gradObj;		///< gradient of objective
	blockSQP::Matrix constrJac;		///< constraint Jacobian (dense)
	double_array_interface jacNz;		///< nonzero elements of constraint Jacobian
	int_array_interface jacIndRow;		///< row indices of nonzero elements
	int_array_interface jacIndCol;		///< starting indices of columns
	blockSQP::SymMatrix *hess;		///< Hessian of the Lagrangian (blockwise)
	
	//Each hessian blocks elements are a double array, wrapper by double_pointer_interface
    //These are the once again wrapped by double_pointer_interface_interface
    double_pointer_interface_array hess_arr;
	
	int dmode;				///< derivative mode
	int info;				///< error flag
};




class Problemform : public blockSQP::Problemspec
{
	public:
	Problemform(){
		blockIdx = nullptr;
	}
	virtual ~Problemform(){
		delete[] blockIdx;
	};
	
	//###########Test_Methods
	void call_initialize(){
		initialize_dense();
	}

	//###########




	Prob_Data Cpp_Data; //values that get evaluated and returned by the evaluate methods


	void init_Cpp_Data(bool Sparse_QP, int nnz){

		Cpp_Data.xi.m = nVar; Cpp_Data.xi.ldim = -1; Cpp_Data.xi.n = 1; Cpp_Data.xi.tflag = 1; delete[] Cpp_Data.xi.array;
		Cpp_Data.lambda.m = nVar + nCon; Cpp_Data.lambda.ldim = -1; Cpp_Data.lambda.n = 1; Cpp_Data.lambda.tflag = 1; delete[] Cpp_Data.lambda.array;
		Cpp_Data.gradObj.m = nVar; Cpp_Data.gradObj.ldim = -1; Cpp_Data.gradObj.n = 1; Cpp_Data.gradObj.tflag = 1; delete[] Cpp_Data.gradObj.array;
		Cpp_Data.constr.m = nCon; Cpp_Data.constr.ldim = -1; Cpp_Data.constr.n = 1; Cpp_Data.constr.tflag = 1; delete[] Cpp_Data.constr.array;
		
		if (not Sparse_QP){
			Cpp_Data.constrJac.m = nCon; Cpp_Data.constrJac.n = nVar; Cpp_Data.constrJac.ldim = -1; Cpp_Data.constrJac.tflag = 1; delete[] Cpp_Data.constrJac.array;
		}
		else{
			Cpp_Data.jacNz.size = nnz;
			Cpp_Data.jacNz.ptr = new double[nnz];
			
			int *RowColLow = new int[nnz + nVar+1 + nVar];
			Cpp_Data.jacIndRow.size = nnz;
			Cpp_Data.jacIndRow.ptr = RowColLow;
			Cpp_Data.jacIndCol.size = nVar+1;
			Cpp_Data.jacIndCol.ptr = RowColLow + nnz;
		}
		
        Cpp_Data.hess_arr.size = nBlocks;
        double_array_interface *h_arrays = new double_array_interface[nBlocks];
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

	virtual void get_objval(){};

	
	void initialize(blockSQP::Matrix &xi, blockSQP::Matrix &lambda, blockSQP::Matrix &constrJac) override {

		Cpp_Data.xi.array = xi.array;
		Cpp_Data.lambda.array = lambda.array;
		Cpp_Data.constrJac.array = constrJac.array;

		update_inits();
		initialize_dense();
	}
	
	
	void initialize(blockSQP::Matrix &xi, blockSQP::Matrix &lambda, double *&jacNz, int *&jacIndRow, int *&jacIndCol) override {

		Cpp_Data.xi.array = xi.array;
		Cpp_Data.lambda.array = lambda.array;

		update_inits();
		initialize_sparse();

		jacNz = Cpp_Data.jacNz.ptr;
		jacIndRow = Cpp_Data.jacIndRow.ptr;
		jacIndCol = Cpp_Data.jacIndCol.ptr;
		
	}
	
	
	void evaluate(const blockSQP::Matrix &xi, const blockSQP::Matrix &lambda, double *objval, blockSQP::Matrix &constr, blockSQP::Matrix &gradObj, blockSQP::Matrix &constrJac, blockSQP::SymMatrix *&hess, int dmode, int *info) override {

		Cpp_Data.xi.array = xi.array;
		Cpp_Data.lambda.array = lambda.array;
		Cpp_Data.dmode = dmode;
		Cpp_Data.constr.array = constr.array;
		
		if (dmode > 0){
			Cpp_Data.gradObj.array = gradObj.array;
			Cpp_Data.constrJac.array = constrJac.array;
		}
		
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
		*info = Cpp_Data.info;
	}
	
	
	void evaluate(const blockSQP::Matrix &xi, const blockSQP::Matrix &lambda, double *objval, blockSQP::Matrix &constr, blockSQP::Matrix &gradObj, double *&jacNz, int *&jacIndRow, int *&jacIndCol, blockSQP::SymMatrix *&hess, int dmode, int *info) override {

		Cpp_Data.xi.array = xi.array;
		Cpp_Data.lambda.array = lambda.array;
		Cpp_Data.dmode = dmode;
		Cpp_Data.constr.array = constr.array;
		
		if (dmode > 0){
			Cpp_Data.gradObj.array = gradObj.array;
			Cpp_Data.jacNz.ptr = jacNz;
			Cpp_Data.jacIndRow.ptr = jacIndRow;
			Cpp_Data.jacIndCol.ptr = jacIndCol;
		}
		
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
		*info = Cpp_Data.info;
		
	}
	
	
	void evaluate(const blockSQP::Matrix &xi, double *objval, blockSQP::Matrix &constr, int *info) override {
		Cpp_Data.xi.array = xi.array;
		Cpp_Data.constr.array = constr.array;
		
		update_simple();
		evaluate_simple();
		get_objval();	
		
		*objval = Cpp_Data.objval;
		*info = Cpp_Data.info;
	}
	
	
	void set_blockIdx(py::array_t<int> arr){
		py::buffer_info buff = arr.request();
		//blockIdx = (int*)buff.ptr;
		delete[] blockIdx;
		blockIdx = new int[buff.size];
		for (int i = 0; i < buff.size; i++){
			blockIdx[i] = static_cast<int*>(buff.ptr)[i];
		}
		nBlocks = buff.size - 1;
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

	void get_objval() override {
		PYBIND11_OVERRIDE(void, Problemform, get_objval,);
	}
};



PYBIND11_MODULE(py_blockSQP, m) {

py::class_<blockSQP::Matrix>(m, "Matrix", py::buffer_protocol())
	.def_buffer([](blockSQP::Matrix &mtrx) -> py::buffer_info{
		return py::buffer_info(
			mtrx.array,
			sizeof(double),
			py::format_descriptor<double>::format(),
			2,
			{mtrx.m,mtrx.n},
//			{sizeof(double)*mtrx.n, sizeof(double)}
			{sizeof(double), sizeof(double)*mtrx.m}
			);
		})
	.def(py::init<int,int,int>(), py::arg("M") = 1, py::arg("N") = 1, py::arg("LDIM") = -1)
	.def(py::init<const blockSQP::Matrix&>())
	.def("Dimension", &blockSQP::Matrix::Dimension, py::arg("M"), py::arg("N") = 1, py::arg("LDIM") = -1)
	.def("Initialize", static_cast<blockSQP::Matrix& (blockSQP::Matrix::*)(double)>(&blockSQP::Matrix::Initialize))
	.def("acc", [](blockSQP::Matrix &mtrx, int i, int j){return mtrx(i,j);})
	.def_readwrite("ldim",&blockSQP::Matrix::ldim)
	.def_readwrite("m",&blockSQP::Matrix::m)
	.def_readwrite("n",&blockSQP::Matrix::n)
	.def_property("array", nullptr, [](blockSQP::Matrix &mtrx, py::array_t<double> arr){
		py::buffer_info buff = arr.request();
		mtrx.array = (double*)buff.ptr;
		});

py::class_<blockSQP::SymMatrix, blockSQP::Matrix>(m, "SymMatrix", py::buffer_protocol());


py::class_<blockSQP::SQPoptions>(m, "SQPoptions")
	.def(py::init<>())
	.def("optionsConsistency",&blockSQP::SQPoptions::optionsConsistency)
	.def_readwrite("printLevel",&blockSQP::SQPoptions::printLevel)
	.def_readwrite("printColor",&blockSQP::SQPoptions::printColor)
	.def_readwrite("debugLevel",&blockSQP::SQPoptions::debugLevel)
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
	.def_readwrite("maxSOCiter",&blockSQP::SQPoptions::maxSOCiter);


py::class_<int_array_interface>(m,"int_array_interface",py::buffer_protocol())
	.def(py::init<>())
	.def_buffer([](int_array_interface &inter) -> py::buffer_info{
		return py::buffer_info(
			inter.ptr,
			sizeof(int),
			py::format_descriptor<int>::format(),
			1,
			{inter.size},
			{sizeof(int)}
			);
		})
	//.def("resize", &int_array_interface::resize)
	;



py::class_<double_array_interface>(m,"double_array_interface",py::buffer_protocol())
	.def(py::init<>())
	.def_buffer([](double_array_interface &inter) -> py::buffer_info{
		return py::buffer_info(
			inter.ptr,
			sizeof(double),
			py::format_descriptor<double>::format(),
			1,
			{inter.size},
			{sizeof(double)}
			);
		})
	//.def("resize", &double_array_interface::resize)
	;
	

py::class_<double_pointer_interface_array>(m,"double_pointer_interface_array")
	.def(py::init<>())
	.def("__getitem__", [](double_pointer_interface_array &arr, int i)->double_array_interface*{return arr.ptr + i;}, py::return_value_policy::reference)
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

py::class_<Problemform, Py_Problemform>(m,"Problemform", py::buffer_protocol())
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
	.def("get_objval", &Problemform::get_objval)
	.def_readwrite("Cpp_Data", &Problemform::Cpp_Data)
	.def_readwrite("nVar", &Problemform::nVar)
	.def_readwrite("nCon", &Problemform::nCon)
	.def_readwrite("nnCon", &Problemform::nnCon)
	.def_readwrite("objLo", &Problemform::objLo)
	.def_readwrite("objUp", &Problemform::objUp)
	.def_readwrite("bl", &Problemform::bl)
	.def_readwrite("bu", &Problemform::bu)
	.def_readonly("nBlocks", &Problemform::nBlocks)
	.def_property("blockIdx", nullptr, &Problemform::set_blockIdx);

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

py::class_<blockSQP::SQPmethod>(m,"SQPmethod")
//	.def(py::init<blockSQP::Problemspec*, blockSQP::SQPoptions*, blockSQP::SQPstats*>())
	.def(py::init([](Problemform* prob, blockSQP::SQPoptions* opts, blockSQP::SQPstats* stats){return new blockSQP::SQPmethod(prob, opts, stats); }), py::return_value_policy::take_ownership)
	.def_readonly("vars", &blockSQP::SQPmethod::vars)
	.def_readonly("stats", &blockSQP::SQPmethod::stats)
	.def("init", &blockSQP::SQPmethod::init)
	.def("run", &blockSQP::SQPmethod::run, py::arg("maxIt"), py::arg("warmStart") = 0)
	.def("finish", &blockSQP::SQPmethod::finish);
	
	
py::class_<blockSQP::SQPiterate>(m, "SQPiterate")
	.def_readonly("obj", &blockSQP::SQPiterate::obj)
	.def_readonly("qpObj", &blockSQP::SQPiterate::qpObj)
	.def_readonly("cNorm", &blockSQP::SQPiterate::cNorm)
	.def_readonly("cNormS", &blockSQP::SQPiterate::cNormS)
	.def_readonly("gradNorm", &blockSQP::SQPiterate::gradNorm)
	.def_readonly("lambdaStepNorm", &blockSQP::SQPiterate::lambdaStepNorm)
	.def_readonly("tol", &blockSQP::SQPiterate::tol)
	.def_readonly("xi", &blockSQP::SQPiterate::xi)
	.def_readonly("lam", &blockSQP::SQPiterate::lambda)
	.def_readonly("constr", &blockSQP::SQPiterate::constr)
	.def_readonly("constrJac", &blockSQP::SQPiterate::constrJac)
	.def_readonly("hess", &blockSQP::SQPiterate::hess)
	.def_readonly("hess1", &blockSQP::SQPiterate::hess1)
	.def_readonly("hess2", &blockSQP::SQPiterate::hess2);
	
}




