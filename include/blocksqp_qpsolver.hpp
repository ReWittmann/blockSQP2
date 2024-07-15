#ifndef BLOCKSQP_QPSOLVER_HPP
#define BLOCKSQP_QPSOLVER_HPP


#include "blocksqp_matrix.hpp"
#include "blocksqp_options.hpp"
#include "blocksqp_problemspec.hpp"


#ifdef QPSOLVER_QPOASES
    #include "qpOASES.hpp"
#endif

#ifdef QPSOLVER_GUROBI
    #include "gurobi_c++.h"
#endif


namespace blockSQP{



//Solver class for quadratic programs of the form

/////////////////////////////
// min_x 0.5x^T H x + x^T h//
// s.t.					   //
//	   lb_A <= A*x <= ub_A //
//	   lb_x <=  x  <= ub_x //
/////////////////////////////

class QPsolver{
public:
    int nVar;
    int nCon;
    int nHess;

    SQPoptions *param;

    //Store the solution time of the last 10 successful QPs,
    //use it to limit the solution time of future QPs
    double solution_durations[10];
    int dur_pos;
    double avg_solution_duration;

    //Solution time options
    double time_limit;
    int time_limit_type;

    //One time QP solving options (reset if a QP solve is successful)
    bool record_time;
    bool use_hotstart;
    bool convex_QP;


	//Arguments: Number of QP variables, number of linear constraints, options
	///TODO: Check options for validity when constructing QPsolver
    QPsolver(int n_QP_var, int n_QP_con, int n_QP_hessblocks, SQPoptions *opts);
    virtual ~QPsolver();

    //Setters for QP data. Only one of the setters for the constraint matrix (dense or sparse) is required
    virtual void set_lin(const Matrix &grad_obj) = 0;
    virtual void set_bounds(const Matrix &lb_x, const Matrix &ub_x, const Matrix &lb_A, const Matrix &ub_A) = 0;
    virtual void set_constr(const Matrix &constr_jac);
    virtual void set_constr(double *const jac_nz, int *const jac_row, int *const jac_colind);
    virtual void set_hess(SymMatrix *const hess) = 0;

    //Solve the QP and write the primal/dual result in deltaXi/lambdaQP.
    //IMPORTANT: deltaXi and lambdaQP have to remain unchanged if the QP solution fails.
    virtual int solve(Matrix &deltaXi, Matrix &lambdaQP) = 0;

    //Statistics
    virtual int get_QP_it();
};


QPsolver *create_QPsolver(int n_QP_var, int n_QP_con, int n_QP_hessblocks, SQPoptions *opts);


#ifdef QPSOLVER_QPOASES
    #include "qpOASES.hpp"
    class qpOASES_solver : public QPsolver{
    public:

        qpOASES::SQProblem*      qp;               ///< qpOASES qp object
        qpOASES::SQProblem*      qpSave;           ///< qpOASES qp object
        qpOASES::SQProblem*      qp_check;         ///< for applying solution analysis

        qpOASES::Matrix* A_qp;                     ///< qpOASES constraint matrix
        qpOASES::SymmetricMatrix* H_qp;            ///< qpOASES quadratic objective matrix
        double* h_qp;                              ///< linear term in objective
        double *lb, *ub, *lbA, *ubA;               ///< bounds for variables, bounds for constraints

        Matrix jacT;                               ///< transpose of the dense constraint jacobian

        double *hess_nz;
        int *hess_row, *hess_colind, *hess_loind;

        bool matrices_changed;

        int QP_it;
        double QP_time;

        qpOASES_solver(int n_QP_var, int n_QP_con, int n_QP_hessblocks, SQPoptions *opts);
        ~qpOASES_solver();

        void set_lin(const Matrix &grad_obj);
        void set_bounds(const Matrix &lb_x, const Matrix &ub_x, const Matrix &lb_A, const Matrix &ub_A);

        void set_constr(const Matrix &constr_jac);
        void set_constr(double *const jac_nz, int *const jac_row, int *const jac_colind);
        void set_hess(SymMatrix *const hess);

        int solve(Matrix &deltaXi, Matrix &lambdaQP);

        int get_QP_it();
    };
#endif



#ifdef QPSOLVER_GUROBI
    #include "gurobi_c++.h"

    class gurobi_solver : public QPsolver{
    public:
        GRBEnv *env;
        GRBModel *model;
        GRBVar* QP_vars;
        //GRBConstr* QP_vars_lb;
        //GRBConstr* QP_vars_ub;
        GRBConstr* QP_cons_lb;
        GRBConstr* QP_cons_ub;

        GRBLinExpr obj_lin;
        GRBQuadExpr obj_quad;


        gurobi_solver(int n_QP_var, int n_QP_con, int n_QP_hessblocks, SQPoptions *opts);
        ~gurobi_solver();

        void set_lin(const Matrix &grad_obj);
        void set_bounds(const Matrix &lb_x, const Matrix &ub_x, const Matrix &lb_A, const Matrix &ub_A);

        void set_constr(const Matrix &constr_jac);
        void set_constr(double *const jac_nz, int *const jac_row, int *const jac_colind);
        void set_hess(SymMatrix *const hess);

        int solve(Matrix &deltaXi, Matrix &lambdaQP);

        int get_QP_it();
    };
#endif



}


#endif
