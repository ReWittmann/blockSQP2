#ifndef BLOCKSQP_QPSOLVER_HPP
#define BLOCKSQP_QPSOLVER_HPP


#include "blocksqp_matrix.hpp"
#include "blocksqp_options.hpp"
#include "blocksqp_problemspec.hpp"
#include "blocksqp_condensing.hpp"
#include "blocksqp_iterate.hpp"
#include "blocksqp_load_mumps.hpp"
#include <memory>
#include <thread>
#include <future>
#include <chrono>
using namespace std::chrono;

#ifdef QPSOLVER_QPOASES
    #include "qpOASES.hpp"
#endif

#ifdef QPSOLVER_GUROBI
    #include "gurobi_c++.h"
#endif

#ifdef QPSOLVER_QPALM
    #define QPALM_TIMING
    #include <qpalm.hpp>
    #include <vector>
#endif

namespace blockSQP{


//QP solver interface
class QPsolverBase{
    public:
    virtual ~QPsolverBase();
    
    virtual void set_lin(const Matrix &grad_obj) = 0;
    virtual void set_bounds(const Matrix &lb_x, const Matrix &ub_x, const Matrix &lb_A, const Matrix &ub_A) = 0;
    virtual void set_constr(const Matrix &constr_jac) = 0;
    virtual void set_constr(double *const jac_nz, int *const jac_row, int *const jac_colind) = 0;
    //Set hessian and pass on whether hessian is supposedly positive definite
    virtual void set_hess(SymMatrix *const hess, bool pos_def = false, double regularizationFactor = 0.0) = 0;

    virtual void set_timeLimit(int limit_type, double custom_limit_secs = -1.0) = 0;
    virtual void set_use_hotstart(bool use_hom) = 0;
    
    //Set QP/active set of which to hotstart from. QP solver dependent, no effect by default, very important for qpOASES
    virtual void set_hotstart_point(QPsolverBase *hot_QP);
    
    //Statistics
    virtual int get_QP_it() = 0;
    virtual double get_solutionTime() = 0;
    
    //Solve the QP and write the primal/dual result in deltaXi/lambdaQP.
    //IMPORTANT: deltaXi and lambdaQP have to remain unchanged if the QP solution fails.
    virtual int solve(Matrix &deltaXi, Matrix &lambdaQP) = 0;
    //Overload for calling with a jthread.
    //virtual int solve(std::stop_token stopRequest, bool *hasFinished, Matrix &deltaXi, Matrix &lambdaQP);
    virtual void solve(std::stop_token stopRequest, std::promise<int> QP_result, Matrix &deltaXi, Matrix &lambdaQP);
};


//Solver class for quadratic programs of the form

//////////////////////////////
// min_x 0.5x^T H x + x^T h //
// s.t.					    //
//	   lb_A <= A*x <= ub_A  //
//	   lb_x <=  x  <= ub_x  //
//////////////////////////////

class QPsolver : public QPsolverBase{
    public:
    int nVar;
    int nCon;
    int nHess;

    const QPsolver_options *Qparam;
    
    //Store the solution time of the last 10 successful QPs,
    //use it to limit the solution time of future QPs
    double solution_durations[10];
    int dur_pos, dur_count;
    double QPtime_avg;
    
    
    //Solution time options
    double default_time_limit, custom_time_limit;
    //0: 2.5*average of past 10, 1: default_time_limit, 2: custom_time_limit
    int time_limit_type;
    
    //Set by set_hess
    bool convex_QP;
    
    //One time QP solving options (reset if a QP solve is successful)
    //bool record_time;
    bool skip_timeRecord;
    bool use_hotstart;
    
    
	//Arguments: Number of QP variables, number of linear constraints, options
    QPsolver(int n_QP_var, int n_QP_con, int n_QP_hessblocks, const QPsolver_options *QPopts);
    virtual ~QPsolver();
    
    //Time recording utility shared by all QP solvers
    void recordTime(double solTime);
    void reset_timeRecord();
    
    //Setters for QP data. Only one of the setters for the constraint matrix (dense or sparse) is required
    virtual void set_lin(const Matrix &grad_obj) = 0;
    virtual void set_bounds(const Matrix &lb_x, const Matrix &ub_x, const Matrix &lb_A, const Matrix &ub_A) = 0;
    
    //Either a sparse or a dense setter for the constraint Jacobian is required, can only (?) throw exception if an unimplemented setter is called.
    virtual void set_constr(const Matrix &constr_jac);
    virtual void set_constr(double *const jac_nz, int *const jac_row, int *const jac_colind);
    
    //Set hessian and pass on whether hessian is supposedly positive definite
    virtual void set_hess(SymMatrix *const hess, bool pos_def = false, double regularizationFactor = 0.0) = 0;
    
    //Solve the QP and write the primal/dual result in deltaXi/lambdaQP.
    //IMPORTANT: deltaXi and lambdaQP have to remain unchanged if the QP solution fails.
    virtual int solve(Matrix &deltaXi, Matrix &lambdaQP) = 0;
    
    virtual void set_timeLimit(int limit_type, double custom_limit_secs = -1.0);
    virtual void set_use_hotstart(bool use_hom);
    
    //Statistics
    virtual int get_QP_it();
    virtual double get_solutionTime();
};

//QP solver with condensing step.
//Requires QPsolverBase instantiated for the condensed QPs of size cond->num_cons, cond->num_vars, ...
class CQPsolver : public QPsolverBase{
    public:
    QPsolverBase *inner_QPsol;
    bool QPsol_own;
    std::unique_ptr<Condenser> cond;
    
    std::unique_ptr<SymMatrix[]> hess_qp;
    bool convex_QP;
    double regF;
    
    Matrix h_qp, lb_x, ub_x, lb_A, ub_A;
    Sparse_Matrix sparse_A_qp;
    
    std::unique_ptr<SymMatrix[]> hess_cond;
    Matrix h_cond, lb_x_cond, ub_x_cond, lb_A_cond, ub_A_cond;
    Sparse_Matrix sparse_A_cond;
    
    Matrix xi_cond, lambda_cond;
    
    //Flags indicating which data was updates, may avoid unnecessary work
    bool h_updated, A_updated, bounds_updated, hess_updated;
    
    CQPsolver(QPsolverBase *arg_CQPsol, const Condenser *arg_cond, bool arg_QPsol_own = false);
    CQPsolver(std::unique_ptr<QPsolverBase> arg_CQPsol, const Condenser *arg_cond);
    ~CQPsolver();
    
    virtual void set_lin(const Matrix &grad_obj);
    virtual void set_bounds(const Matrix &lb_x, const Matrix &ub_x, const Matrix &lb_A, const Matrix &ub_A);
    
    // TODO implement once condensing supports dense constraint Jacobians    
    virtual void set_constr(const Matrix &constr_jac);
    virtual void set_constr(double *const jac_nz, int *const jac_row, int *const jac_colind);
    
    //Set hessian and pass on whether hessian is supposedly positive definite
    virtual void set_hess(SymMatrix *const hess, bool pos_def = false, double regularizationFactor = 0.0);

    //Solve the QP and write the primal/dual result in deltaXi/lambdaQP.
    //IMPORTANT: deltaXi and lambdaQP have to remain unchanged if the QP solution fails.
    virtual int solve(Matrix &deltaXi, Matrix &lambdaQP);
    virtual void solve(std::stop_token stopRequest, std::promise<int> QP_result, Matrix &deltaXi, Matrix &lambdaQP);
    
    virtual void set_timeLimit(int limit_type, double custom_limit_secs = -1.0);
    void set_use_hotstart(bool use_hom);
    
    //Statistics
    virtual int get_QP_it();
    virtual double get_solutionTime();
    
};




//Helper factory to create QPsolver with given SQPoptions. This assumes opts->OptionsConsistency has already been called to check for inconsistent options.
//Preprocessor conditions for linked QP solvers are handled here.

//QPsolver *create_QPsolver(int n_QP_var, int n_QP_con, int n_QP_hessblocks, int *blockIdx, SQPoptions *opts);

QPsolverBase *create_QPsolver(const Problemspec *prob, const SQPiterate *vars, const SQPoptions *param);
QPsolverBase *create_QPsolver(const Problemspec *prob, const SQPiterate *vars, const QPsolver_options *Qparam);

std::unique_ptr<std::unique_ptr<QPsolverBase>[]> create_QPsolvers_par(const Problemspec *prob, const SQPiterate *vars, const SQPoptions *param, int N_QP = -1);



//QP solver implementations
#ifdef QPSOLVER_QPOASES
    class qpOASES_solver : public QPsolver{
        public:
        qpOASES::Options opts;
        
        std::unique_ptr<qpOASES::SQProblem> qp;
        std::unique_ptr<qpOASES::SQProblem> qpSave;
        std::unique_ptr<qpOASES::SQProblem>  qpCheck; 
        
        std::unique_ptr<qpOASES::Matrix> A_qp;
        std::unique_ptr<qpOASES::SymmetricMatrix> H_qp;
        
        std::unique_ptr<double[]> h_qp;                                       // linear term in objective
        std::unique_ptr<double[]> A_qp_nz;
        std::unique_ptr<int[]> A_qp_row;
        std::unique_ptr<int[]> A_qp_colind; 
        
        std::unique_ptr<double[]> lb, ub, lbA, ubA;         // bounds for QP variables, bounds for linearized constraints
        
        Matrix jacT;                                        // transpose of the dense constraint jacobian
        
        std::unique_ptr<double[]> hess_nz;
        std::unique_ptr<int[]> hess_row, hess_colind, hess_loind; //Copy of sparse constraint Jacobian
        
        bool matrices_changed;
        
        int QP_it;
        
        //qpOASES_solver(int n_QP_var, int n_QP_con, int n_QP_hessblocks, int *blockIdx, qpOASES_options *QPopts);
        
        qpOASES_solver(int n_QP_var, int n_QP_con, int n_QP_hessblocks, int *blockIdx, const qpOASES_options *QPopts);
        
        qpOASES_solver(int n_QP_var, int n_QP_con, int n_QP_hessblocks, const QPsolver_options *QPopts);
        void init_QP_common(int *blockIdx); //Initialize data that is independent of QP type (dense/sparse/schur)
        ~qpOASES_solver();
        
        void set_lin(const Matrix &grad_obj);
        void set_bounds(const Matrix &lb_x, const Matrix &ub_x, const Matrix &lb_A, const Matrix &ub_A);

        void set_constr(const Matrix &constr_jac);
        void set_constr(double *const jac_nz, int *const jac_row, int *const jac_colind);
        void set_hess(SymMatrix *const hess, bool pos_def = false, double regularizationFactor = 0.0);
        
        void set_hotstart_point(QPsolverBase *hot_QP);
        void set_hotstart_point(qpOASES_solver *hot_QP);
        
        int solve(Matrix &deltaXi, Matrix &lambdaQP);
        //int solve(std::stop_token stopRequest, bool *hasFinished, int *result, Matrix &deltaXi, Matrix &lambdaQP);
        void solve(std::stop_token stopRequest, std::promise<int> QP_result, Matrix &deltaXi, Matrix &lambdaQP);
        int get_QP_it();
    };
    
    //For using qpOASES with MUMPS. Mumps is meant to be loaded by the caller several times,
    //such that the loaded modules are thread safe (e.g. via dlmopen on linux).
    //Then different instances of this class will be threadsafe between each other. 
    class threadsafe_qpOASES_MUMPS_solver : public qpOASES_solver{
        public:
        void *linsol_handle;
        threadsafe_qpOASES_MUMPS_solver(int n_QP_var, int n_QP_con, int n_QP_hessblocks, int *blockIdx, const qpOASES_options *QPopts, int linsol_ID);
        
        threadsafe_qpOASES_MUMPS_solver(int n_QP_var, int n_QP_con, int n_QP_hessblocks, int *blockIdx, const qpOASES_options *QPopts, void *fptr_dmumps_c);
        ~threadsafe_qpOASES_MUMPS_solver();
        
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


        gurobi_solver(int n_QP_var, int n_QP_con, int n_QP_hessblocks, gurobi_options *QPopts);
        ~gurobi_solver();

        void set_lin(const Matrix &grad_obj);
        void set_bounds(const Matrix &lb_x, const Matrix &ub_x, const Matrix &lb_A, const Matrix &ub_A);

        void set_constr(const Matrix &constr_jac);
        void set_constr(double *const jac_nz, int *const jac_row, int *const jac_colind);
        void set_hess(SymMatrix *const hess, bool pos_def = false, double regularizationFactor = 0.0);

        int solve(Matrix &deltaXi, Matrix &lambdaQP);

        int get_QP_it();
        //double get_solutionTime();
    };
#endif

#ifdef QPSOLVER_QPALM
    class qpalm_solver : public QPsolver{
    public:
        qpalm::Data data;
        qpalm::sparse_mat_t Q;
        qpalm::vec_t q;
        qpalm::sparse_mat_t A;
        qpalm::vec_t lb;
        qpalm::vec_t ub;
        qpalm::Settings settings;
        std::vector<qpalm::triplet_t> triplets;
        qpalm::Info info;

        qpalm_solver(int n_QP_var, int n_QP_con, int n_QP_hessblocks, qpalm_options *QPopts);
        ~qpalm_solver();

        void set_lin(const Matrix &grad_obj);
        void set_bounds(const Matrix &lb_x, const Matrix &ub_x, const Matrix &lb_A, const Matrix &ub_A);

        void set_constr(const Matrix &constr_jac);
        void set_constr(double *const jac_nz, int *const jac_row, int *const jac_colind);
        void set_hess(SymMatrix *const hess, bool pos_def = false, double regularizationFactor = 0.0);

        int solve(Matrix &deltaXi, Matrix &lambdaQP);

        int get_QP_it();
    };
#endif


}


#endif
