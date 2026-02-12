/*
 * blockSQP 2 -- Condensing, convexification strategies, scaling heuristics and more
 *               for blockSQP, the nonlinear programming solver by Dennis Janka.
 * Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
 * 
 * Licensed under the zlib license. See LICENSE for more details.
 */
 

/**
 * \file condensing.hpp
 * \author Reinhold Wittmann
 * \date 2023-2025
 *
 * Declaration of methods and data structures for Condenser class
 */

#ifndef BLOCKSQP2_CONDENSING_HPP
#define BLOCKSQP2_CONDENSING_HPP

#include <blockSQP2/defs.hpp>
#include <blockSQP2/matrix.hpp>

#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cassert>
#include <memory>


namespace blockSQP2{


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
    
    // Matchings may either be x_k - F(x_k-1, u_k-1) = 0 => -A_k, -B_k = B_Jac(*:*,*:*) or F(x_k-1, u_k-1) - x_k = 0 => A_k, B_k = B_Jac(*:*,*:*)
    // std::abs(match_sign) = 1 with the sign being the sign of x_k in the matching conditions. It is assumed to be the same for all matchings of a target
    // and inferred from passed Jacobian. 
    double matching_sign; 
    
    //Blocks of the structured constraint-jacobian and hessian, together with linear terms and bounds
	std::vector<Matrix> A_k; //A_1, ..., A_{n_stages - 1}
	std::vector<Matrix> B_k; //B_0, ..., B_{n_stages - 1}
	std::vector<Matrix> c_k; //c_0, ..., c_{n_stages - 1}
	std::vector<Matrix> r_k; //r_0, ..., r_{n_stages}
	std::vector<Matrix> q_k; //q_1, ..., q_{n_stages}
    
	std::vector<Matrix> R_k; //R_0, ..., R_{n_stages}
	std::vector<Matrix> Q_k; //Q_1, ..., Q_{n_stages}
	std::vector<Matrix> S_k; //S_1, ..., S_{n_stages}
    
	std::vector<Matrix> g_k; //g_0, ..., g_{n_stages - 1}
	LT_Block_Matrix G;       //n_stages x (n_stages + 1)
	std::vector<Matrix> h_k; //h_0, ..., h_{n_stages}
	LT_Block_Matrix H;       //(n_stages + 1) x (n_stages + 1)
    
	//Horizontal slices of (horizontal jacobian slice corresponding to target variables), separated in free- and dependent-variable-slices,
	//without continuity conditions
    std::vector<Sparse_Matrix> J_free_k; //J_f_0, ..., J_f_{n_stages}
    std::vector<Sparse_Matrix> J_dep_k; //J_d_1, ..., J_d_{n_stages - 1}

    std::vector<CSR_Matrix> J_d_CSR_k; //J_d_CSR_1, ..., J_d_CSR_{n_stages}
    std::vector<Sparse_Matrix> J_reduced_k; //J_reduced_0, ..., J_reduced_{n_stages}
    
    //See condensing paper (Andersson 2013)
    Matrix g;
    Matrix h;
    //Matrix G_dense;
    Sparse_Matrix G_sparse;
    SymMatrix H_dense;
    Sparse_Matrix J_reduced;

	//Dependent and free variable bounds
	Matrix D_lb;
	Matrix D_ub;
	Matrix F_lb;
	Matrix F_ub;

	//Jacobian w.r.t. all free variables of target, times vector g from condensing. Needed to offset constraint-bounds.
	Matrix Jtimes_g;

    //Convexification coefficient of the hessian, H = H_1 * (1 - t_h) + H_2 * t_h
    double t_H;

	//Copy of the blocks of the original hessian, to calculate convex combinations with fallback hessian
	std::vector<Matrix> R_k_1;
	std::vector<Matrix> Q_k_1;
	std::vector<Matrix> S_k_1;
	std::vector<Matrix> h_k_1;

    Matrix h_1;
    SymMatrix H_dense_1;

    //Blocks of an alternative/fallback - hessian, on which the linear term of the condensed QP also depends
    std::vector<Matrix> R_k_2;
    std::vector<Matrix> Q_k_2;
    std::vector<Matrix> S_k_2;

    std::vector<Matrix> h_k_2;
    LT_Block_Matrix H_2;

    Matrix h_2;
    SymMatrix H_dense_2;
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
    int* condensed_blockIdx;

	//Additional option: How should dependent variable bounds be added to the condensed QP:
    //  0: not, 1: inactive, -inf<= Gu + g <= inf, 2: active, lb_dep <= Gu + g <= ub_dep
	int add_dep_bounds;
	//Number of constraints and conditions of original QP, if dependent variable bounds are kept, else number of "true" constraints
	int condensed_num_cons;

    ///QP specific data
	//QP-specific data for each condensable variable-condition-structure
	condensing_data *targets_data;

    //Slices of the gradient of the objective
    std::vector<Matrix> T_grad_obj;
    std::vector<Matrix> O_grad_obj;

	//Horizontal slices of linear constraints matrix (Jacobian) for T-target variables and O-other variables
	std::vector<Sparse_Matrix> T_Slices;
    std::vector<Sparse_Matrix> O_Slices;

    ///Condensed QP data

    //Bounds on dependent variables in condensed QP, which can be manually added to a QP condensed with option add_dep_bounds = 1
    Matrix lb_dep_var;
    Matrix ub_dep_var;

	Condenser(vblock* VBLOCKS, int n_VBLOCKS, cblock* CBLOCKS, int n_CBLOCKS, int* HSIZES, int n_HBLOCKS, condensing_target* TARGETS, int n_TARGETS, int DEP_BOUNDS = 2);
	Condenser(Condenser &&C);
	virtual ~Condenser();
    
    //Construct a new Condenser sharing the layout data of an existing condenser.
    static Condenser *layout_copy(const Condenser *CND);

    void print_info();

    //Setter of changing how dependent variable bounds are added
    void set_dep_bound_handling(int opt);

    //For starting index start, get index of hessian block that starts at start
	int get_hessblock_index(int start);

	//Complete condensing for new quadratic subproblem
	void full_condense(const Matrix &grad_obj, const Sparse_Matrix &con_jac, const SymMatrix *const hess,
                        const Matrix &lb_var, const Matrix &ub_var, const Matrix &lb_con, const Matrix &ub_con,
            Matrix &condensed_h, Sparse_Matrix &condensed_Jacobian, SymMatrix *condensed_hess,
                        Matrix &condensed_lb_var, Matrix &condensed_ub_var, Matrix &condensed_lb_con, Matrix &condensed_ub_con);

	//Condensing for a single block of variables and conditions with condensable structure
    void single_condense(int tnum, const Matrix &grad_obj, const Sparse_Matrix &B_Jac, const SymMatrix *const sub_hess,
                            const Matrix &B_lb_var, const Matrix &B_ub_var, const Matrix &lb_con);

    //Recovery of dependent variables and condition-lagrange-multipliers from qp-solution X_cond
	void recover_var_mult(const Matrix &xi_cond, const Matrix &lambda_cond,
                            Matrix &xi_full, Matrix &lambda_full);

    //Recover dependent variables and continuity condition multipliers for a single condensable structure
    //mu: multipliers for free variable bounds, lambda: multipliers for dependent variable bounds, nu: multipliers for continuity conditions, sigma: multipliers for (true) constraints
    void single_recover(int tnum, const Matrix &xi_free, const Matrix &mu, const Matrix &lambda, const Matrix &sigma,
                            Matrix &xi_full, Matrix &nu, Matrix &mu_lambda);



    ///Methods for special cases.
    //They are modded versions of the four primary methods above, so currently there is still a lot of code duplication

    //Update condensed QP with a different hessian. This also affects the linear term in the condensed QP.
    void fallback_hessian_condense(const SymMatrix *const hess_2, Matrix &condensed_h_2, SymMatrix *condensed_hess_2);
    void single_hess_condense(int tnum, const SymMatrix *const sub_hess);

    //Recover dependent variables and condition multipliers for a convex combination of the original and fallback hessian (1-t)*hess1 + t*hess2
    void convex_combination_recover(const Matrix &xi_cond, const Matrix &lambda_cond, const double t, Matrix &xi_full, Matrix &lambda_full);

    //mu: multipliers for free variable bounds, lambda: multipliers for dependent variable bounds, nu: multipliers for continuity conditions, sigma: multipliers for (true) constraints
    void single_convex_combination_recover(int tnum, const Matrix &xi_free, const Matrix &mu, const Matrix &lambda, const Matrix &sigma, const double t,
                            Matrix &xi_full, Matrix &nu, Matrix &mu_lambda);

    
    //Update condensed QP with a new hessian
    void new_hessian_condense(const SymMatrix *const hess, Matrix &condensed_h, SymMatrix *condensed_hess);
    void single_new_hess_condense(int tnum, const SymMatrix *const sub_hess);

    //Update condensed QP with new constraint bounds. grad_obj must not be changed and is used to construct the new linear term condensed_h, which DOES change
    void SOC_condense(const Matrix &grad_obj, const Matrix &lb_con, const Matrix &ub_con, Matrix &condensed_h, Matrix &condensed_lb_con, Matrix &condensed_ub_con);
    void single_SOC_condense(int tnum, const Matrix &lb_con);

    //Update condensed QP with new constraint bounds and add the correction term to the dependent variables
    void correction_condense(const Matrix &grad_obj, const Matrix &lb_con, const Matrix &ub_con, const Matrix *const target_corrections, Matrix &condensed_h, Matrix &condensed_lb_con, Matrix &condensed_ub_con);
    void single_correction_condense(int tnum, const Matrix &lb_con, const Matrix &correction);

    //Recover dependent variables and dependent variable bounds and condition multipliers. The correction term used in condensing has to be passed again.
    void recover_correction_var_mult(const Matrix &xi_cond, const Matrix &lambda_cond, const Matrix *const target_corrections, Matrix &xi_full, Matrix &lambda_full);
    void single_correction_recover(int tnum, const Matrix &xi_free, const Matrix &mu, const Matrix &lambda, const Matrix &sigma, const Matrix &correction,
                            Matrix &xi_full, Matrix &nu, Matrix &mu_lambda);
};


//Condenser holding its own layout information
class holding_Condenser : public Condenser{
    public:
    holding_Condenser(
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

//holding_Condenser* create_restoration_Condenser(Condenser *parent, int DEP_BOUNDS = 2);



} // namespace blockSQP2

#endif

