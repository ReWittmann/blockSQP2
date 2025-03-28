/**
 * \file blocksqp_condensing.hpp
 * \author Reinhold Wittmann
 * \date 2023-
 *
 * Declaration of methods and data structures for Condenser class
 */

#ifndef BLOCKSQP_CONDENSING_HPP
#define BLOCKSQP_CONDENSING_HPP

#include "blocksqp_defs.hpp"
#include "blocksqp_matrix.hpp"
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cassert>
#include <memory>


namespace blockSQP{


struct vblock{
    vblock();
	vblock(int SIZE, bool DEP);
	int size;
	bool dependent;
	bool removed;
};

//Future: Have more general dependency graphs for variables for advanced condensing in more than one pass
/*
struct vblock{
    vblock();
    vblock(int SIZE, bool DEP, bool COND1, bool COND2);
    int size;
    bool dependent;
    //?maybe
        bool dependent1;
        bool dependent2;
    //

    bool cond1;
    bool cond2;
    
    int ndep_in;
    vblock* deps_in;
    cblock* couplings_in;

    int ndep_out;
    vblock* deps_out;
    cblock* couplings_out;
}
*/


struct cblock{
	cblock(int SIZE);
	cblock();
	int size;
	bool removed;
};


struct condensing_target{
    condensing_target();
	condensing_target(int N_stages, int ffree, int B_end, int f_cond, int C_end);
	//Number of intermediate shooting stages
	int n_stages;
	//First free variable block that following dependent blocks depend on
	int first_free;
	//End of variable blocks of target/first vblock not corresponding to target and associated hessian blocks
	int vblock_end;
	//First and last continuity conditions
	int first_cond;
	int cblock_end;
};


struct condensing_data{

    //Ranges of variables in the slice of variables of the condensing target, alternating between independent and dependent variables
	std::vector<int> alt_vranges;
	//Ranges of stage-continuity-conditions
	std::vector<int> cond_ranges;
	//Number of free variables for each stage
	std::vector<int> free_sizes;
	//Number of components of the continuity conditions
	std::vector<int> cond_sizes;
	//Number of free and dependent variables.
    int n_free;
    int n_dep;

    //Blocks of the structured constraint-jacobian and hessian, together with linear terms and bounds
	std::vector<blockSQP::Matrix> A_k; //A_1, ..., A_{n_stages - 1}
	std::vector<blockSQP::Matrix> B_k; //B_0, ..., B_{n_stages - 1}
	std::vector<blockSQP::Matrix> c_k; //c_0, ..., c_{n_stages - 1}
	std::vector<blockSQP::Matrix> r_k; //r_0, ..., r_{n_stages}
	std::vector<blockSQP::Matrix> q_k; //q_1, ..., q_{n_stages}

	std::vector<blockSQP::Matrix> R_k; //R_0, ..., R_{n_stages}
	std::vector<blockSQP::Matrix> Q_k; //Q_1, ..., Q_{n_stages}
	std::vector<blockSQP::Matrix> S_k; //S_1, ..., S_{n_stages}

	std::vector<blockSQP::Matrix> g_k; //g_0, ..., g_{n_stages - 1}
	blockSQP::LT_Block_Matrix G;       //n_stages x (n_stages + 1)
	std::vector<blockSQP::Matrix> h_k; //h_0, ..., h_{n_stages}
	blockSQP::LT_Block_Matrix H;       //(n_stages + 1) x (n_stages + 1)

	//Horizontal slices of (horizontal jacobian slice corresponding to target variables), separated in free- and dependent-variable-slices,
	//without continuity conditions
    std::vector<blockSQP::Sparse_Matrix> J_free_k; //J_f_0, ..., J_f_{n_stages}
    std::vector<blockSQP::Sparse_Matrix> J_dep_k; //J_d_1, ..., J_d_{n_stages - 1}

    std::vector<blockSQP::CSR_Matrix> J_d_CSR_k; //J_d_CSR_1, ..., J_d_CSR_{n_stages}
    std::vector<blockSQP::Sparse_Matrix> J_reduced_k; //J_reduced_0, ..., J_reduced_{n_stages}

    //See condensing paper (Andersson 2017)
    blockSQP::Matrix g;
    blockSQP::Matrix h;
    //blockSQP::Matrix G_dense;
    blockSQP::Sparse_Matrix G_sparse;
    blockSQP::SymMatrix H_dense;
    blockSQP::Sparse_Matrix J_reduced;

	//Dependent and free variable bounds
	blockSQP::Matrix D_lb;
	blockSQP::Matrix D_ub;
	blockSQP::Matrix F_lb;
	blockSQP::Matrix F_ub;

	//Jacobian w.r.t. all free variables of target, times vector g from condensing. Needed to offset constraint-bounds.
	blockSQP::Matrix Jtimes_g;

    //Convexification coefficient of the hessian, H = H_1 * (1 - t_h) + H_2 * t_h
    double t_H;

	//Copy of the blocks of the original hessian, to calculate convex combinations with fallback hessian
	std::vector<blockSQP::Matrix> R_k_1;
	std::vector<blockSQP::Matrix> Q_k_1;
	std::vector<blockSQP::Matrix> S_k_1;
	std::vector<blockSQP::Matrix> h_k_1;

    blockSQP::Matrix h_1;
    blockSQP::SymMatrix H_dense_1;

    //Blocks of an alternative/fallback - hessian, on which the linear term of the condensed QP also depends
    std::vector<blockSQP::Matrix> R_k_2;
    std::vector<blockSQP::Matrix> Q_k_2;
    std::vector<blockSQP::Matrix> S_k_2;

    std::vector<blockSQP::Matrix> h_k_2;
    blockSQP::LT_Block_Matrix H_2;

    blockSQP::Matrix h_2;
    blockSQP::SymMatrix H_dense_2;

    //Slices of different bounds (e.g. during SOC)
    /*
    std::vector<blockSQP::Matrix> c_k_2;
    std::vector<blockSQP::Matrix> r_k_2;
    std::vector<blockSQP::Matrix> q_k_2;
    */

};


class Condenser{

    public:
    //Constructor arguments
	int num_cblocks;
	int num_vblocks;
	int num_hessblocks;
	int num_targets;

	cblock* cblocks;
	vblock* vblocks;
	int* hess_block_sizes;
	condensing_target* targets;

    //Layout data calculated from constructor arguments
	int num_vars;
	int num_cons;
    int condensed_num_vars;
    //Number of constraints in condensed QPs that are not dependent variable bounds = number of constraints in uncondensed QP that are not conditions used for condensing
    int num_true_cons;

    int condensed_num_hessblocks;

	int* cranges;
	int* vranges;

	int* c_starts;
	int* c_ends;
	int* v_starts;
	int* v_ends;
	int* h_starts;
	int* h_ends;
    int* condensed_v_starts;
    int* condensed_v_ends;

	int* hess_block_ranges;

    int* condensed_hess_block_sizes;

	//Additional option: How should dependent variable bounds be added to the condensed QP:
    //  0: not, 1: inactive, -inf<= Gu + g <= inf, 2: active, lb_dep <= Gu + g <= ub_dep
	int add_dep_bounds;
	//Number of constraints and conditions of original QP, if dependent variable bounds are kept, else number of "true" constraints
	int condensed_num_cons;

    ///QP specific data
	//QP-specific data for each condensable variable-condition-structure
	condensing_data *targets_data;

    //Slices of the gradient of the objective
    std::vector<blockSQP::Matrix> T_grad_obj;
    std::vector<blockSQP::Matrix> O_grad_obj;

	//Horizontal slices of linear constraints matrix (Jacobian) for T-target variables and O-other variables
	std::vector<blockSQP::Sparse_Matrix> T_Slices;
    std::vector<blockSQP::Sparse_Matrix> O_Slices;

    ///Condensed QP data

    //Bounds on dependent variables in condensed QP, which can be manually added to a QP condensed with option add_dep_bounds = 1
    Matrix lb_dep_var;
    Matrix ub_dep_var;

	Condenser(vblock* VBLOCKS, int n_VBLOCKS, cblock* CBLOCKS, int n_CBLOCKS, int* HSIZES, int n_HBLOCKS, condensing_target* TARGETS, int n_TARGETS, int DEP_BOUNDS = 2);
	//Condenser(const Condenser &C2);
	Condenser(Condenser &&C);
	virtual ~Condenser();


    void print_debug();

    //Setter of changing how dependent variable bounds are added
    void set_dep_bound_handling(int opt);

    //For starting index start, get index of hessian block that starts at start
	int get_hessblock_index(int start);

	//Complete condensing for new quadratic subproblem
	void full_condense(const blockSQP::Matrix &grad_obj, const blockSQP::Sparse_Matrix &con_jac, const blockSQP::SymMatrix *const hess,
                        const blockSQP::Matrix &lb_var, const blockSQP::Matrix &ub_var, const blockSQP::Matrix &lb_con, const blockSQP::Matrix &ub_con,
            blockSQP::Matrix &condensed_h, blockSQP::Sparse_Matrix &condensed_Jacobian, blockSQP::SymMatrix *&condensed_hess,
                        blockSQP::Matrix &condensed_lb_var, blockSQP::Matrix &condensed_ub_var, blockSQP::Matrix &condensed_lb_con, blockSQP::Matrix &condensed_ub_con);

	//Condensing for a single block of variables and conditions with condensable structure
    void single_condense(int tnum, const blockSQP::Matrix &grad_obj, const blockSQP::Sparse_Matrix &B_Jac, const blockSQP::SymMatrix *const sub_hess,
                            const blockSQP::Matrix &B_lb_var, const blockSQP::Matrix &B_ub_var, const blockSQP::Matrix &lb_con);

    //Recovery of dependent variables and condition-lagrange-multipliers from qp-solution X_cond
	void recover_var_mult(const blockSQP::Matrix &xi_cond, const blockSQP::Matrix &lambda_cond,
                            blockSQP::Matrix &xi_full, blockSQP::Matrix &lambda_full);

    //Recover dependent variables and continuity condition multipliers for a single condensable structure
    //mu: multipliers for free variable bounds, lambda: multipliers for dependent variable bounds, nu: multipliers for continuity conditions, sigma: multipliers for (true) constraints
    void single_recover(int tnum, const blockSQP::Matrix &xi_free, const blockSQP::Matrix &mu, const blockSQP::Matrix &lambda, const blockSQP::Matrix &sigma,
                            blockSQP::Matrix &xi_full, blockSQP::Matrix &nu, blockSQP::Matrix &mu_lambda);



    ///BlockSQP specific methods: Condense a positive definite fallback hessian approximation for lifting an indefinite hessian via convex combinations

    //Update condensed QP with a different hessian. This also affects the linear term in the condensed QP.
    void fallback_hessian_condense(const blockSQP::SymMatrix *const hess_2, blockSQP::Matrix &condensed_h_2, blockSQP::SymMatrix *&condensed_hess_2);
    void single_hess_condense(int tnum, const blockSQP::SymMatrix *const sub_hess);

    //Recover dependent variables and condition multipliers for a convex combination of the original and fallback hessian (1-t)*hess1 + t*hess2
    void convex_combination_recover(const blockSQP::Matrix &xi_cond, const blockSQP::Matrix &lambda_cond, const double t, blockSQP::Matrix &xi_full, blockSQP::Matrix &lambda_full);

    //mu: multipliers for free variable bounds, lambda: multipliers for dependent variable bounds, nu: multipliers for continuity conditions, sigma: multipliers for (true) constraints
    void single_convex_combination_recover(int tnum, const blockSQP::Matrix &xi_free, const blockSQP::Matrix &mu, const blockSQP::Matrix &lambda, const blockSQP::Matrix &sigma, const double t,
                            blockSQP::Matrix &xi_full, blockSQP::Matrix &nu, blockSQP::Matrix &mu_lambda);


    //Update condensed QP with a new hessian
    void new_hessian_condense(const blockSQP::SymMatrix *const hess, blockSQP::Matrix &condensed_h, blockSQP::SymMatrix *&condensed_hess);
    void single_new_hess_condense(int tnum, const blockSQP::SymMatrix *const sub_hess);

    //Update condensed QP with new constraint bounds. grad_obj must not be changed and is used to construct the new linear term condensed_h, which DOES change
    void SOC_condense(const blockSQP::Matrix &grad_obj, const blockSQP::Matrix &lb_con, const blockSQP::Matrix &ub_con, blockSQP::Matrix &condensed_h, blockSQP::Matrix &condensed_lb_con, blockSQP::Matrix &condensed_ub_con);
    void single_SOC_condense(int tnum, const blockSQP::Matrix &lb_con);

    //Update condensed QP with new constraint bounds and add the correction term to the dependent variables
    void correction_condense(const blockSQP::Matrix &grad_obj, const blockSQP::Matrix &lb_con, const blockSQP::Matrix &ub_con, const blockSQP::Matrix *const target_corrections, blockSQP::Matrix &condensed_h, blockSQP::Matrix &condensed_lb_con, blockSQP::Matrix &condensed_ub_con);
    void single_correction_condense(int tnum, const blockSQP::Matrix &lb_con, const blockSQP::Matrix &correction);

    //Recover dependent variables and dependent variable bounds and condition multipliers. The correction term used in condensing has to be passed again.
    void recover_correction_var_mult(const blockSQP::Matrix &xi_cond, const blockSQP::Matrix &lambda_cond, const blockSQP::Matrix *const target_corrections, blockSQP::Matrix &xi_full, blockSQP::Matrix &lambda_full);
    void single_correction_recover(int tnum, const blockSQP::Matrix &xi_free, const blockSQP::Matrix &mu, const blockSQP::Matrix &lambda, const blockSQP::Matrix &sigma, const blockSQP::Matrix &correction,
                            blockSQP::Matrix &xi_full, blockSQP::Matrix &nu, blockSQP::Matrix &mu_lambda);

};


//Condenser for restoration problem of parent condenser problem
class autonomous_Condenser : public Condenser{
    public:
    autonomous_Condenser(
                        std::unique_ptr<vblock[]> VBLOCKS, int n_VBLOCKS, 
                        std::unique_ptr<cblock[]> CBLOCKS, int n_CBLOCKS, 
                        std::unique_ptr<int[]> HSIZES, int n_HBLOCKS, 
                        std::unique_ptr<condensing_target[]> TARGETS, int n_TARGETS, 
                        int DEP_BOUNDS = 2);
	std::unique_ptr<vblock[]> auto_vblocks;
    std::unique_ptr<cblock[]> auto_cblocks;
	std::unique_ptr<int[]> auto_hess_block_sizes;
	std::unique_ptr<condensing_target[]> auto_targets;
};

std::unique_ptr<autonomous_Condenser> create_restoration_Condenser(Condenser *parent, int DEP_BOUNDS = 2);



}//namespace blockSQP

#endif

