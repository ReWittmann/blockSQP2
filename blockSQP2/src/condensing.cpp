/*
 * blockSQP2 -- A structure-exploiting nonlinear programming solver based
 *              on blockSQP by Dennis Janka.
 * Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
 * 
 * Licensed under the zlib license. See LICENSE for more details.
 */

/**
 * \file blocksqp_condensing.cpp
 * \author Reinhold Wittmann
 * \date 2023-2025
 *
 * Implementation of methods and data structures for Condenser class
 */

#include <blockSQP2/condensing.hpp>
#include <blockSQP2/matrix.hpp>
#include <string>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cassert>
#include <limits>

#ifdef __cpp_lib_format
    #include <format>
#endif

#include <chrono>
#include <thread>
#include <fstream>

namespace blockSQP2{

vblock::vblock(int SIZE, bool DEP): size(SIZE), dependent(DEP), removed(false)
{}

vblock::vblock(): size(0), dependent(false), removed(false)
{}

cblock::cblock(int SIZE): size(SIZE), removed(false)
{}

cblock::cblock(): size(0), removed(false)
{}

condensing_target::condensing_target(int N_stages, int ffree, int B_end, int fcond, int C_end):
	n_stages(N_stages), first_free(ffree), vblock_end(B_end), first_cond(fcond), cblock_end(C_end){
		if (vblock_end <= first_free || cblock_end <= first_cond){
            throw std::invalid_argument("vblock end index must be greater than index of first free variable block and cblock end index must be greater than index of first condition");
		}
		if (cblock_end - first_cond != n_stages){
            throw std::invalid_argument("Number of stages must be equal to number of conditions in given range");
		}
	}

condensing_target::condensing_target():
    n_stages(0), first_free(0), vblock_end(0), first_cond(0), cblock_end(0){
    }



Condenser::Condenser(
	vblock* VBLOCKS,            int n_VBLOCKS,
    cblock* CBLOCKS,            int n_CBLOCKS,
	int* HSIZES,                int n_HBLOCKS,
	condensing_target* TARGETS, int n_TARGETS,
	int DEP_BOUNDS):
		num_cblocks(n_CBLOCKS), num_vblocks(n_VBLOCKS), num_hessblocks(n_HBLOCKS), num_targets(n_TARGETS),
		cblocks(CBLOCKS), vblocks(VBLOCKS), hess_block_sizes(HSIZES), targets(TARGETS), add_dep_bounds(DEP_BOUNDS)
		{

            //Sanitize inputs
            for (int tnum = 0; tnum < num_targets; tnum++){
                if (targets[tnum].cblock_end > num_cblocks || targets[tnum].first_cond < 0){
                    throw std::invalid_argument("Condenser: Target " + std::to_string(tnum) + " condition blocks range out of bounds");
                }

                if (targets[tnum].vblock_end > num_vblocks || targets[tnum].first_free < 0){
                    throw std::invalid_argument("Condenser: Target " + std::to_string(tnum) + " variable blocks range out of bounds");
                }

                if (targets[tnum].n_stages != targets[tnum].cblock_end - targets[tnum].first_cond){
                    throw std::invalid_argument("Condenser: Number of target condition blocks must be equal to number of stages");
                }
            }

            //Initialize data concerning the full QP
			cranges = new int[num_cblocks+1];
			vranges = new int[num_vblocks+1];

			c_starts = new int[num_targets];
			c_ends = new int[num_targets];
			v_starts = new int[num_targets];
			v_ends = new int[num_targets];
			h_starts = new int[num_targets];
			h_ends = new int[num_targets];
			condensed_v_starts = new int[num_targets];
			condensed_v_ends = new int[num_targets];

			hess_block_ranges = new int[num_hessblocks+1];
			targets_data = new condensing_data[num_targets];
			std::sort(targets, targets+num_targets, [](condensing_target T1, condensing_target T2) -> bool{return T1.first_free < T2.first_free;});
            T_Slices.reserve(num_targets);
            T_grad_obj.reserve(num_targets);
            O_Slices.reserve(num_targets + 1);
            O_grad_obj.reserve(num_targets + 1);

			vranges[0] = 0;
			for (int i = 1; i<= num_vblocks; i++){
				vranges[i] = vranges[i-1] + vblocks[i-1].size;
			}
			num_vars = vranges[num_vblocks];

			cranges[0] = 0;
			for (int i = 1; i<= num_cblocks; i++){
				cranges[i] = cranges[i-1] + cblocks[i-1].size;
			}
			num_cons = cranges[num_cblocks];

			hess_block_ranges[0] = 0;
			for (int i = 1; i<= num_hessblocks; i++){
				hess_block_ranges[i] = hess_block_ranges[i-1] + hess_block_sizes[i-1];
			}

			v_starts[0] = vranges[targets[0].first_free];
			v_ends[0] = vranges[targets[0].vblock_end];
			c_starts[0] = cranges[targets[0].first_cond];
			c_ends[0] = cranges[targets[0].cblock_end];
			for (int i = 1; i < num_targets; i++){
				v_starts[i] = vranges[targets[i].first_free];
				v_ends[i] = vranges[targets[i].vblock_end];
				c_starts[i] = cranges[targets[i].first_cond];
				c_ends[i] = cranges[targets[i].cblock_end];
				if (v_starts[i] < v_ends[i-1] || c_starts[i] < c_ends[i-1]){
					throw std::invalid_argument("Shooting-structure variable/condition-slices overlapping!");
				}
			}

            for (int i = 0; i < num_targets; i++){
                h_starts[i] = get_hessblock_index(v_starts[i]);
                h_ends[i] = get_hessblock_index(v_ends[i]);
            }

            //Set condensing conditions as removed
            for (int i = 0; i < num_targets; i++){
                for (int j = targets[i].first_cond; j < targets[i].cblock_end; j++){
                    cblocks[j].removed = true;
				}
            }


            int num_free;
            int offset = 0;
			for (int tnum = 0; tnum < num_targets; tnum++){
                condensed_v_starts[tnum] = v_starts[tnum] - offset;
                num_free = 0;
                for (int i = targets[tnum].first_free; i<targets[tnum].vblock_end; i++){
                    if (vblocks[i].dependent){
                        offset += vblocks[i].size;
                    }
                    else{
                        num_free += vblocks[i].size;
                    }
                }
                condensed_v_ends[tnum] = condensed_v_starts[tnum] + num_free;
			}
            condensed_num_vars = num_vars - offset;

            num_true_cons = num_cons;
            condensed_num_hessblocks = num_hessblocks;
            for (int i = 0; i < num_targets; i++){
                num_true_cons -= c_ends[i] - c_starts[i];
                condensed_num_hessblocks -= h_ends[i] - h_starts[i] - 1;
            }

            if (add_dep_bounds){
                condensed_num_cons = num_cons;
            }
            else{
                condensed_num_cons = num_true_cons;
            }

            //Initialize data for individual condensing target structures

			int ind;
			int f_ind;
			int d_ind;
			bool dep;
			int n_stages;
			for (int tnum = 0; tnum < num_targets; tnum++){

                n_stages = targets[tnum].n_stages;
                //Generate layout-information
				targets_data[tnum].alt_vranges.resize(n_stages * 2 + 2);
				targets_data[tnum].cond_ranges.resize(n_stages + 1);
				targets_data[tnum].free_sizes.resize(n_stages + 1);
				targets_data[tnum].cond_sizes.resize(n_stages);

				targets_data[tnum].alt_vranges[0] = 0;
				targets_data[tnum].alt_vranges[1] = 0;
				targets_data[tnum].free_sizes[0] = 0;
				targets_data[tnum].n_free = 0;
				targets_data[tnum].n_dep = 0;
				ind = 1;
				f_ind = 0;
				d_ind = 0;
				dep = false;
				for (int i = targets[tnum].first_free; i<targets[tnum].vblock_end; i++){
					if (dep == vblocks[i].dependent){
						targets_data[tnum].alt_vranges[ind] += vblocks[i].size;

                        if (!vblocks[i].dependent){
                            targets_data[tnum].free_sizes[f_ind] += vblocks[i].size;
                            targets_data[tnum].n_free += vblocks[i].size;
                        }
                        else{
                            targets_data[tnum].n_dep += vblocks[i].size;
                        }
					}
					else{
						dep = vblocks[i].dependent;
						ind++;
						targets_data[tnum].alt_vranges[ind] = targets_data[tnum].alt_vranges[ind - 1] + vblocks[i].size;

                        if (!vblocks[i].dependent){
                            f_ind++;
                            if (f_ind > n_stages){
                                throw std::invalid_argument("Condenser: Number of free slices following dependent slices cannot be greater then the number of stages");
                            }

                            targets_data[tnum].free_sizes[f_ind] = vblocks[i].size;
                            targets_data[tnum].n_free += vblocks[i].size;
                        }
                        else{
                            targets_data[tnum].n_dep += vblocks[i].size;
                            d_ind++;
                        }
					}
				}

				if (vblocks[targets[tnum].vblock_end - 1].dependent){
                    targets_data[tnum].free_sizes[n_stages] = 0;
                    targets_data[tnum].alt_vranges[2*n_stages + 1] = targets_data[tnum].alt_vranges[2*n_stages];
				}

                //sanitizing
				if (targets[tnum].n_stages != d_ind){
                    throw std::invalid_argument("Condenser: Number of dependent variable slices inbetween free variable blocks does not match number of stages");
				}


				for (int i = 0; i < n_stages; i++){
					targets_data[tnum].cond_ranges[i] = cranges[targets[tnum].first_cond + i];
					targets_data[tnum].cond_sizes[i] = cblocks[targets[tnum].first_cond + i].size;
				}
				targets_data[tnum].cond_ranges[n_stages] = cranges[targets[tnum].cblock_end];


                //sanitizing
                for (int i = 0; i < n_stages; i++){
                    if (targets_data[tnum].cond_sizes[i] != targets_data[tnum].alt_vranges[2*i + 2] - targets_data[tnum].alt_vranges[2*i + 1]){
                        throw std::invalid_argument("Condenser: Number of dependent variables of each stage must match number of defining conditions");
                    }
                }


				//Allocate matrices and vectors for the condensing algorithm
                targets_data[tnum].A_k.resize(n_stages - 1);
                targets_data[tnum].B_k.resize(n_stages);
                targets_data[tnum].c_k.resize(n_stages);
                targets_data[tnum].r_k.resize(n_stages + 1);
                targets_data[tnum].q_k.resize(n_stages);

                targets_data[tnum].R_k.resize(n_stages + 1);
                targets_data[tnum].Q_k.resize(n_stages);
                targets_data[tnum].S_k.resize(n_stages);


                int *m_sizes = new int[n_stages];
                int *n_sizes = new int[n_stages + 1];
                int *h_sizes = new int[n_stages + 1];
                for (int i = 0; i < n_stages; i++){
                    m_sizes[i] = targets_data[tnum].cond_sizes[i];
                    n_sizes[i] = targets_data[tnum].free_sizes[i];
                    h_sizes[i] = targets_data[tnum].free_sizes[i];
                }
                n_sizes[n_stages] = targets_data[tnum].free_sizes[n_stages];
                h_sizes[n_stages] = targets_data[tnum].free_sizes[n_stages];

                targets_data[tnum].g_k.resize(n_stages);
                targets_data[tnum].G.Dimension(n_stages, n_stages + 1, m_sizes, n_sizes);
                targets_data[tnum].h_k.resize(n_stages + 1);
                targets_data[tnum].H.Dimension(n_stages + 1, n_stages + 1, h_sizes, h_sizes);

                targets_data[tnum].J_free_k.resize(n_stages + 1);
                targets_data[tnum].J_dep_k.resize(n_stages);

                targets_data[tnum].J_d_CSR_k.resize(n_stages);
                targets_data[tnum].J_reduced_k.resize(n_stages + 1);

                //Allocate additional matrices and vectors in case an additional QP with fallback hessian needs to be condensed
                int *h_sizes_2 = new int[n_stages + 1];
                for (int i = 0; i <= n_stages; i++){
                    h_sizes_2[i] = targets_data[tnum].free_sizes[i];
                }

                targets_data[tnum].R_k_2.resize(n_stages + 1);
                targets_data[tnum].Q_k_2.resize(n_stages);
                targets_data[tnum].S_k_2.resize(n_stages);
                targets_data[tnum].h_k_2.resize(n_stages + 1);
                targets_data[tnum].H_2.Dimension(n_stages + 1, n_stages + 1, h_sizes_2, h_sizes_2);
			}

            condensed_hess_block_sizes = new int[condensed_num_hessblocks];
            int ind_1 = 0;
            int ind_2 = 0;
            for (int tnum = 0; tnum < num_targets; tnum++){
                for (int i = 0; i < h_starts[tnum] - ind_2; i++){
                    condensed_hess_block_sizes[ind_1 + i] = hess_block_sizes[ind_2 + i];
                }
                ind_1 += h_starts[tnum] - ind_2;
                condensed_hess_block_sizes[ind_1] = targets_data[tnum].n_free;
                ind_1++;
                ind_2 = h_ends[tnum];
            }
            for (int i = 0; i < num_hessblocks - ind_2; i++){
                condensed_hess_block_sizes[ind_1 + i] = hess_block_sizes[ind_2 + i];
            }

            condensed_blockIdx = new int[condensed_num_hessblocks + 1];
            condensed_blockIdx[0] = 0;
            for (int i = 1; i < condensed_num_hessblocks + 1; i++){
                condensed_blockIdx[i] = condensed_blockIdx[i-1] + condensed_hess_block_sizes[i-1];
            }
		}
/*
Condenser(const Condenser &C){
    num_cblocks = C.num_cblocks;
    num_vblocks = C.num_vblocks;
    num_hessblocks = C.num_hessblocks;
    num_targets = C.num_targets;

    cblocks = C.cblocks;
    vblocks = C.vblocks;
    hess_block_sizes = C.hess_block_sizes;
    targets = C.targets;

    num_vars = C.num_vars;
    num_cons = C.num_cons;
    condensed_num_vars = C.condensed_num_vars;
}*/

Condenser::Condenser(Condenser &&C){
    num_cblocks = C.num_cblocks;
    num_vblocks = C.num_vblocks;
    num_hessblocks = C.num_hessblocks;
    num_targets = C.num_targets;

    cblocks = C.cblocks;
    vblocks = C.vblocks;
    hess_block_sizes = C.hess_block_sizes;
    targets = C.targets;

    num_vars = C.num_vars;
    num_cons = C.num_cons;
    condensed_num_vars = C.condensed_num_vars;
    num_true_cons = C.num_true_cons;
    condensed_num_hessblocks = C.condensed_num_hessblocks;
    condensed_hess_block_sizes = C.condensed_hess_block_sizes;
    condensed_blockIdx = C.condensed_blockIdx;

    cranges = C.cranges;
    vranges = C.vranges;
	c_starts = C.c_starts;
	c_ends = C.c_ends;
	v_starts = C.v_starts;
	v_ends = C.v_ends;
	h_starts = C.h_starts;
	h_ends = C.h_ends;
    condensed_v_starts = C.condensed_v_starts;
    condensed_v_ends = C.condensed_v_ends;
	hess_block_ranges = C.hess_block_ranges;

    C.cranges = nullptr;
    C.vranges = nullptr;
    C.c_starts = nullptr;
    C.c_ends = nullptr;
    C.v_starts = nullptr;
    C.v_ends = nullptr;
    C.h_starts = nullptr;
    C.h_ends = nullptr;
    C.condensed_v_starts = nullptr;
    C.condensed_v_ends = nullptr;
    C.hess_block_ranges = nullptr;

    add_dep_bounds = C.add_dep_bounds;
    condensed_num_cons = C.condensed_num_cons;

    targets_data = C.targets_data;
    C.targets_data = nullptr;

    T_grad_obj = std::move(C.T_grad_obj);
    O_grad_obj = std::move(C.O_grad_obj);

    T_Slices = std::move(C.T_Slices);
    O_Slices = std::move(C.O_Slices);

    lb_dep_var = C.lb_dep_var;
    ub_dep_var = C.ub_dep_var;
}


Condenser::~Condenser(){
	delete[] cranges;
	delete[] vranges;

	delete[] c_starts;
	delete[] c_ends;
	delete[] v_starts;
	delete[] v_ends;
	delete[] h_starts;
	delete[] h_ends;
    delete[] condensed_hess_block_sizes;
    delete[] condensed_blockIdx;
	delete[] condensed_v_starts;
	delete[] condensed_v_ends;
    delete[] hess_block_ranges;
    delete[] targets_data;

}
/*
void Condenser::calc_ranges(){
	vranges[0] = 0;
	for (int i = 1; i<= num_vblocks; i++){
		vranges[i] = vranges[i-1] + vblocks[i].size;
	}

	cranges[0] = 0;
	for (int i = 1; i<= num_cblocks; i++){
		cranges[i] = cranges[i-1] + cblocks[i].size;
	}

	hess_block_ranges[0] = 0;
	for (int i = 1; i<= num_hessblocks; i++){
		hess_block_ranges[i] = hess_block_ranges[i-1] + hess_block_sizes[i];
	}
	return;
}
*/


Condenser *Condenser::layout_copy(const Condenser *cond){
    return new Condenser(cond->vblocks, cond->num_vblocks, cond->cblocks, cond->num_cblocks, cond->hess_block_sizes, cond->num_hessblocks, cond->targets, cond->num_targets, cond->add_dep_bounds);
}

void Condenser::print_info(){
    std::cout<< "num_targets: " << num_targets << "\n";
    std::cout<< "num_vars: " << num_vars << "\n";
    std::cout<< "num_cons: " << num_cons << "\n";
    std::cout<< "num_hessblocks: " << num_hessblocks << "\n";
    std::cout<< "condensed_num_vars: " << condensed_num_vars << "\n";
    std::cout<< "num_true_cons: " << num_true_cons << "\n";
    std::cout<< "condensed_num_hessblocks: " << condensed_num_hessblocks << "\n";
    std::cout<< "add_dep_bounds: " << add_dep_bounds << "\n";
    std::cout<< "condensed_num_cons: " << condensed_num_cons << "\n";

    std::cout<< "vranges: ";
    for (int i = 0; i<num_vblocks+1; i++){
        std::cout<< vranges[i] << " ";
    }
    std::cout<< "\n";

    std::cout<< "cranges: ";
    for (int i = 0; i<num_cblocks+1; i++){
        std::cout<< cranges[i] << " ";
    }
    std::cout<< "\n";

    std::cout<< "hess_block_ranges: ";
    for (int i = 0; i<=num_hessblocks; i++){
        std::cout<< hess_block_ranges[i] << " ";
    }
    std::cout<< "\n";

    std::cout<< "v_starts = ";
    for (int tnum = 0; tnum < num_targets; tnum++){
        std::cout << v_starts[tnum] << " ";
    }
    std::cout<<"\n";

    std::cout<< "v_ends = ";
    for (int tnum = 0; tnum < num_targets; tnum++){
        std::cout << v_ends[tnum] << " ";
    }
    std::cout<<"\n";

    std::cout<< "h_starts = ";
    for (int tnum = 0; tnum < num_targets; tnum++){
        std::cout << h_starts[tnum] << " ";
    }
    std::cout<<"\n";

    std::cout<< "h_ends = ";
    for (int tnum = 0; tnum < num_targets; tnum++){
        std::cout << h_ends[tnum] << " ";
    }
    std::cout<<"\n";

    std::cout<< "condensed_v_starts = ";
    for (int tnum = 0; tnum < num_targets; tnum++){
        std::cout << condensed_v_starts[tnum] << " ";
    }
    std::cout<<"\n";

    std::cout<< "condensed_v_ends = ";
    for (int tnum = 0; tnum < num_targets; tnum++){
        std::cout << condensed_v_ends[tnum] << " ";
    }
    std::cout<<"\n";

    std::cout<< "c_starts = ";
    for (int tnum = 0; tnum < num_targets; tnum++){
        std::cout << c_starts[tnum] << " ";
    }
    std::cout<<"\n";

    std::cout<< "c_ends = ";
    for (int tnum = 0; tnum < num_targets; tnum++){
        std::cout << c_ends[tnum] << " ";
    }
    std::cout<<"\n";

    std::cout << "condensed_hess_block_sizes = ";
    for (int i = 0; i < condensed_num_hessblocks; i++){
        std::cout << condensed_hess_block_sizes[i] << " ";
    }
    std::cout << "\n";

    std::cout<< "cblock presence status = \n";
    for (int i = 0; i< num_cblocks; i++){
        std::cout << cblocks[i].removed << ", ";
    }
    std::cout<< "\n";

    for (int tnum = 0; tnum < num_targets; tnum++){
        std::cout << "target " << tnum << " n_stages: " << targets[tnum].n_stages << "\n"; 

        std::cout<< "target " << tnum << " alt_vranges: ";
        for (int i = 0; i < 2*targets[tnum].n_stages + 2; i++){
            std::cout<<targets_data[tnum].alt_vranges[i] << " ";
        }
        std::cout<< "\n";

        std::cout<< "target " << tnum << " free_sizes: ";
        for (int i = 0; i <= targets[tnum].n_stages; i++){
            std::cout<< targets_data[tnum].free_sizes[i] << " ";
        }
        std::cout<< "\n";

        std::cout<< "target " << tnum << " cond_sizes: ";
        for (int i = 0; i < targets[tnum].n_stages; i++){
            std::cout<< targets_data[tnum].cond_sizes[i] << " ";
        }
        std::cout<< "\n";

        std::cout<< "target " << tnum << " cond_ranges: ";
        for (int i = 0; i <= targets[tnum].n_stages; i++){
            std::cout<< targets_data[tnum].cond_ranges[i] << " ";
        }
        std::cout<< "\n";
    }
    return;
}

void Condenser::set_dep_bound_handling(int DEP_BOUNDS){
    add_dep_bounds = DEP_BOUNDS;
    if (add_dep_bounds)
        condensed_num_cons = num_cons;
    else
        condensed_num_cons = num_true_cons;
    return;
}


int Condenser::get_hessblock_index(int v_ind){
	for (int i = 0; i <= num_hessblocks; i++){
		if (hess_block_ranges[i] == v_ind){
			return i;
		}
	}
    throw std::invalid_argument("Variable-block start " + std::to_string(v_ind) + " not matching hessian block start/end");
}



void Condenser::full_condense(const Matrix &grad_obj, const Sparse_Matrix &con_jac, const SymMatrix *const hess, const Matrix &lb_var, const Matrix &ub_var, const Matrix &lb_con, const Matrix &ub_con,
    Matrix &condensed_h, Sparse_Matrix &condensed_Jacobian, SymMatrix *condensed_hess, Matrix &condensed_lb_var, Matrix &condensed_ub_var, Matrix &condensed_lb_con, Matrix &condensed_ub_con
){
	T_Slices.resize(0);
	O_Slices.resize(0);
	T_grad_obj.resize(0);
	O_grad_obj.resize(0);

    std::vector<Matrix> T_lb_var;
    std::vector<Matrix> T_ub_var;
    std::vector<Matrix> O_lb_var;
    std::vector<Matrix> O_ub_var;
    T_lb_var.reserve(num_targets);
    T_ub_var.reserve(num_targets);
    O_lb_var.reserve(num_targets + 1);
    O_ub_var.reserve(num_targets + 1);

	O_Slices.push_back(con_jac.get_slice(0, con_jac.m, 0, v_starts[0]));
    O_lb_var.push_back(lb_var.get_slice(0, v_starts[0], 0, 1));
    O_ub_var.push_back(ub_var.get_slice(0, v_starts[0], 0, 1));
    O_grad_obj.push_back(grad_obj.get_slice(0, v_starts[0], 0, 1));

	T_Slices.push_back(con_jac.get_slice(0, con_jac.m, v_starts[0], v_ends[0]));
    T_lb_var.push_back(lb_var.get_slice(v_starts[0], v_ends[0], 0, 1));
    T_ub_var.push_back(ub_var.get_slice(v_starts[0], v_ends[0], 0, 1));
    T_grad_obj.push_back(grad_obj.get_slice(v_starts[0], v_ends[0], 0, 1));

	for (int i = 1; i < num_targets; i++){
        O_Slices.push_back(con_jac.get_slice(0, con_jac.m, v_ends[i-1], v_starts[i]));
        O_lb_var.push_back(lb_var.get_slice(v_ends[i-1], v_starts[i], 0, 1));
        O_ub_var.push_back(ub_var.get_slice(v_ends[i-1], v_starts[i], 0, 1));
        O_grad_obj.push_back(grad_obj.get_slice(v_ends[i-1], v_starts[i], 0, 1));

        T_Slices.push_back(con_jac.get_slice(0, con_jac.m, v_starts[i], v_ends[i]));
        T_lb_var.push_back(lb_var.get_slice(v_starts[i], v_ends[i], 0, 1));
        T_ub_var.push_back(ub_var.get_slice(v_starts[i], v_ends[i], 0, 1));
        T_grad_obj.push_back(grad_obj.get_slice(v_starts[i], v_ends[i], 0, 1));
	}
	O_Slices.push_back(con_jac.get_slice(0, con_jac.m, v_ends[num_targets - 1], num_vars));
    O_lb_var.push_back(lb_var.get_slice(v_ends[num_targets - 1], num_vars, 0, 1));
    O_ub_var.push_back(ub_var.get_slice(v_ends[num_targets - 1], num_vars, 0, 1));
    O_grad_obj.push_back(grad_obj.get_slice(v_ends[num_targets - 1], num_vars, 0, 1));

    //Assert that lower and upper bounds of condensing conditions are equal
    for (int tnum = 0; tnum < num_targets; tnum++){
        for (int i = c_starts[tnum]; i < c_ends[tnum]; i++){
            if (lb_con(i) - ub_con(i) >= 1e-12 || ub_con(i) - lb_con(i) >= 1e-12){
                throw std::invalid_argument("Error, Condensing conditions not equality constrained, difference (ub - lb)[" + std::to_string(i) + "] = " + std::to_string(ub_con(i) - lb_con(i)));
            }
        }
    }

    for (int i = 0; i < num_targets; i++){
        single_condense(i, T_grad_obj[i], T_Slices[i], &(hess[h_starts[i]]), T_lb_var[i], T_ub_var[i], lb_con);
        O_Slices[i].remove_rows(c_starts, c_ends, num_targets);
    }
    O_Slices[num_targets].remove_rows(c_starts, c_ends, num_targets);



//Assemble reduced constraint-jacobian (condensed jacobian without dependent-variable bounds)
    std::vector<Sparse_Matrix> reduced_Slices(2*num_targets + 1);
    for (int i = 0; i<num_targets; i++){
        reduced_Slices[2*i] = O_Slices[i];
        reduced_Slices[2*i+1] = targets_data[i].J_reduced;
    }
    reduced_Slices[2*num_targets] = O_Slices[num_targets];
    Sparse_Matrix reduced_Jacobian = horzcat(reduced_Slices);

//Assemble condensed block-hessian
    int ind_1 = 0;
    int ind_2 = 0;
    for (int tnum = 0; tnum < num_targets; tnum++){
        for (int i = 0; i < h_starts[tnum] - ind_2; i++){
            condensed_hess[ind_1 + i] = hess[ind_2 + i];
        }
        ind_1 += h_starts[tnum] - ind_2;
        ind_2 = h_ends[tnum];
        condensed_hess[ind_1] = targets_data[tnum].H_dense;
        ind_1++;
    }
    for (int i = 0; i < num_hessblocks - ind_2; i++){
        condensed_hess[ind_1 + i] = hess[ind_2 + i];
    }

//Assemble reduced constraint-bounds (without dependent-variable bounds)
    Matrix reduced_lb_con = lb_con.without_rows(c_starts, c_ends, num_targets);
    Matrix reduced_ub_con = ub_con.without_rows(c_starts, c_ends, num_targets);

    for (int i = 0; i < num_targets; i++){
        reduced_lb_con -= targets_data[i].Jtimes_g;
        reduced_ub_con -= targets_data[i].Jtimes_g;
    }

//Assemble free variable-bounds
    std::vector<Matrix> condensed_lb_var_k(2*num_targets + 1);
    std::vector<Matrix> condensed_ub_var_k(2*num_targets + 1);

    for (int i = 0; i < num_targets; i++){
        condensed_lb_var_k[2*i] = O_lb_var[i];
        condensed_lb_var_k[2*i+1] = targets_data[i].F_lb;
        condensed_ub_var_k[2*i] = O_ub_var[i];
        condensed_ub_var_k[2*i+1] = targets_data[i].F_ub;
    }
    condensed_lb_var_k[2*num_targets] = O_lb_var[num_targets];
    condensed_ub_var_k[2*num_targets] = O_ub_var[num_targets];

    condensed_lb_var = vertcat(condensed_lb_var_k);
    condensed_ub_var = vertcat(condensed_ub_var_k);

//Add dependent variable bounds to constraints
    if (add_dep_bounds == 2){
        std::vector<Sparse_Matrix> condensed_Jacobian_Slices(num_targets + 1);
        std::vector<Matrix> condensed_lb_con_k(num_targets + 1);
        std::vector<Matrix> condensed_ub_con_k(num_targets + 1);
        condensed_Jacobian_Slices[0] = reduced_Jacobian;
        condensed_lb_con_k[0] = reduced_lb_con;
        condensed_ub_con_k[0] = reduced_ub_con;
        for (int tnum = 0; tnum < num_targets; tnum++){
            condensed_Jacobian_Slices[tnum + 1] = lr_zero_pad(condensed_num_vars, targets_data[tnum].G_sparse, condensed_v_starts[tnum]);
            condensed_lb_con_k[tnum + 1] = targets_data[tnum].D_lb - targets_data[tnum].g;
            condensed_ub_con_k[tnum + 1] = targets_data[tnum].D_ub - targets_data[tnum].g;
        }

        condensed_Jacobian = vertcat(condensed_Jacobian_Slices);
        condensed_lb_con = vertcat(condensed_lb_con_k);
        condensed_ub_con = vertcat(condensed_ub_con_k);
    }
    else if (add_dep_bounds == 1){
        std::vector<Sparse_Matrix> condensed_Jacobian_Slices(num_targets + 1);
        std::vector<Matrix> condensed_lb_con_k(num_targets + 1);
        std::vector<Matrix> condensed_ub_con_k(num_targets + 1);
        condensed_Jacobian_Slices[0] = reduced_Jacobian;
        condensed_lb_con_k[0] = reduced_lb_con;
        condensed_ub_con_k[0] = reduced_ub_con;
        for (int tnum = 0; tnum < num_targets; tnum++){
            condensed_Jacobian_Slices[tnum + 1] = lr_zero_pad(condensed_num_vars, targets_data[tnum].G_sparse, condensed_v_starts[tnum]);
            condensed_lb_con_k[tnum + 1] = Matrix(targets_data[tnum].n_dep).Initialize(-std::numeric_limits<double>::infinity());
            condensed_ub_con_k[tnum + 1] = Matrix(targets_data[tnum].n_dep).Initialize(std::numeric_limits<double>::infinity());
        }

        condensed_Jacobian = vertcat(condensed_Jacobian_Slices);
        condensed_lb_con = vertcat(condensed_lb_con_k);
        condensed_ub_con = vertcat(condensed_ub_con_k);

        //Save bounds on dependent variables so a user can manually add them to the qp
        std::vector<Matrix> lb_dep_var_k(num_targets);
        std::vector<Matrix> ub_dep_var_k(num_targets);
        for (int tnum = 0; tnum < num_targets; tnum++){
            lb_dep_var_k[tnum] = targets_data[tnum].D_lb - targets_data[tnum].g;
            ub_dep_var_k[tnum] = targets_data[tnum].D_ub - targets_data[tnum].g;
        }
        lb_dep_var = vertcat(lb_dep_var_k);
        ub_dep_var = vertcat(ub_dep_var_k);
    }
    else{
        condensed_Jacobian = reduced_Jacobian;
        condensed_lb_con = reduced_lb_con;
        condensed_ub_con = reduced_ub_con;
    }

//Assemble condensed_h, vector of linear term in objective
    std::vector<Matrix> condensed_h_k(2*num_targets+1);
    for (int i = 0; i < num_targets; i++){
        condensed_h_k[2*i] = O_grad_obj[i];
        condensed_h_k[2*i+1] = targets_data[i].h;
    }
    condensed_h_k[2*num_targets] = O_grad_obj[num_targets];
    condensed_h = vertcat(condensed_h_k);
    return;
}


void Condenser::single_condense(int tnum, const Matrix &grad_obj, const Sparse_Matrix &B_Jac, const SymMatrix *const sub_hess, const Matrix &B_lb_var, const Matrix &B_ub_var, const Matrix &lb_con){

	int n_stages = targets[tnum].n_stages;
	condensing_data &Data = targets_data[tnum];
    
    // Matchings may either be x_k - F(x_k-1, u_k-1) = 0 => -A_k, -B_k = B_Jac(*:*,*:*) or F(x_k-1, u_k-1) - x_k = 0 => A_k, B_k = B_Jac(*:*,*:*)
    // Set match_sense to the sign before x_k in the matching, slices of B_Jac require the opposite sign.
    // double match_sense = B_Jac(Data.cond_ranges[0], Data.alt_vranges[1]);
    Data.matching_sign = B_Jac(Data.cond_ranges[0], Data.alt_vranges[1]);
    #ifdef __cpp_lib_format
        if (std::abs(std::abs(Data.matching_sign) - 1.) > 1e-12) throw std::logic_error(std::format("Error during condensing: Expected constr_jac({}:{},{}:{}) == +-I, but constr_jac({},{}) = {}", Data.cond_ranges[0], Data.cond_ranges[1], Data.alt_vranges[1], Data.alt_vranges[2], Data.cond_ranges[0], Data.alt_vranges[1], Data.matching_sign));
    #else
        if (std::abs(std::abs(Data.matching_sign) - 1.) > 1e-12) throw std::logic_error("Error during condensing: Constraint Jacobian not matching provided layout data");
    #endif
    Data.matching_sign = std::round(Data.matching_sign);
    
	//Extract relevant subvectors and -matrices
	Data.B_k[0] = B_Jac.get_slice(Data.cond_ranges[0], Data.cond_ranges[1], Data.alt_vranges[0], Data.alt_vranges[1]).dense();
    Data.B_k[0] *= -Data.matching_sign; //
    Data.c_k[0] = lb_con.get_slice(Data.cond_ranges[0], Data.cond_ranges[1]);
    Data.c_k[0] *= Data.matching_sign;
	Data.r_k[0] = grad_obj.get_slice(Data.alt_vranges[0], Data.alt_vranges[1]);

	Data.R_k[0] = sub_hess[0];

	for (int i = 1; i<n_stages; i++){
        double matching_sign = B_Jac(Data.cond_ranges[i], Data.alt_vranges[2*i+1]);
        #ifdef __cpp_lib_format
            if (std::abs(std::abs(matching_sign) - 1.) > 1e-12) throw std::logic_error(std::format("Error during condensing: Expected constr_jac({}:{},{}:{}) == +-I, but constr_jac({},{}) = {}", Data.cond_ranges[i], Data.cond_ranges[i+1], Data.alt_vranges[2*i+1], Data.alt_vranges[2*i+2], Data.cond_ranges[i], Data.alt_vranges[2*i+1], matching_sign));
        #else
            if (std::abs(std::abs(matching_sign) - 1.) > 1e-12) throw std::logic_error("Error during condensing: Constraint Jacobian not matching provided layout data");
        #endif
        if (Data.matching_sign != std::round(matching_sign)) throw std::logic_error("Error during condensing: All matchings of a target must have the same sign, i.e. all x_k+1 - F(x_k-1,...) = 0 or all F(x_k-1,...) - x_k+1 = 0");
        
		Data.B_k[i] = B_Jac.get_slice(Data.cond_ranges[i], Data.cond_ranges[i+1], Data.alt_vranges[2*i], Data.alt_vranges[2*i+1]).dense();
        Data.B_k[i] *= -Data.matching_sign;
		Data.c_k[i] = lb_con.get_slice(Data.cond_ranges[i], Data.cond_ranges[i+1]);
        Data.c_k[i] *= Data.matching_sign;
		Data.r_k[i] = grad_obj.get_slice(Data.alt_vranges[2*i], Data.alt_vranges[2*i+1]);

		Data.A_k[i-1] = B_Jac.get_slice(Data.cond_ranges[i], Data.cond_ranges[i+1], Data.alt_vranges[2*i-1], Data.alt_vranges[2*i]).dense();
        Data.A_k[i-1] *= -Data.matching_sign;

		Data.q_k[i-1] = grad_obj.get_slice(Data.alt_vranges[2*i-1], Data.alt_vranges[2*i]);

		Data.R_k[i] = sub_hess[i].get_slice(Data.alt_vranges[2*i] - Data.alt_vranges[2*i-1], Data.alt_vranges[2*i+1] - Data.alt_vranges[2*i-1], Data.alt_vranges[2*i] - Data.alt_vranges[2*i-1], Data.alt_vranges[2*i+1] - Data.alt_vranges[2*i-1]);
		Data.Q_k[i-1] = sub_hess[i].get_slice(0, Data.alt_vranges[2*i] - Data.alt_vranges[2*i-1], 0, Data.alt_vranges[2*i] - Data.alt_vranges[2*i-1]);
		Data.S_k[i-1] = sub_hess[i].get_slice(0, Data.alt_vranges[2*i]- Data.alt_vranges[2*i-1], Data.alt_vranges[2*i] - Data.alt_vranges[2*i-1], Data.alt_vranges[2*i+1] - Data.alt_vranges[2*i-1]);
	}

	Data.q_k[n_stages - 1] = grad_obj.get_slice(Data.alt_vranges[2*n_stages - 1], Data.alt_vranges[2*n_stages]);
	Data.r_k[n_stages] = grad_obj.get_slice(Data.alt_vranges[2*n_stages], Data.alt_vranges[2*n_stages+1]);

	Data.R_k[n_stages] = sub_hess[n_stages].get_slice(Data.alt_vranges[2*n_stages] - Data.alt_vranges[2*n_stages-1], Data.alt_vranges[2*n_stages + 1] - Data.alt_vranges[2*n_stages-1], Data.alt_vranges[2*n_stages] - Data.alt_vranges[2*n_stages-1], Data.alt_vranges[2*n_stages + 1] - Data.alt_vranges[2*n_stages-1]);
	Data.Q_k[n_stages - 1] = sub_hess[n_stages].get_slice(0, Data.alt_vranges[2*n_stages] - Data.alt_vranges[2*n_stages-1], 0, Data.alt_vranges[2*n_stages] - Data.alt_vranges[2*n_stages-1]);
	Data.S_k[n_stages - 1] = sub_hess[n_stages].get_slice(0, Data.alt_vranges[2*n_stages] - Data.alt_vranges[2*n_stages-1], Data.alt_vranges[2*n_stages] - Data.alt_vranges[2*n_stages-1], Data.alt_vranges[2*n_stages+1] - Data.alt_vranges[2*n_stages-1]);

	std::vector<Matrix> w_k(n_stages);
	std::vector<Matrix> W_ik(n_stages);


	//calculate g
	Data.g_k[0] = Data.c_k[0];
	for (int i = 1; i < n_stages; i++){
		Data.g_k[i] = Data.A_k[i-1]*Data.g_k[i-1] + Data.c_k[i];
	}

	//calculate G
	for (int i = 0; i<n_stages; i++){
		Data.G.set(i,i, Data.B_k[i]);
		for (int k = i+1; k < n_stages; k++){
			Data.G.set(k, i, Data.A_k[k-1]*Data.G(k-1,i));
		}
	}

	//calculate h
	Data.h_k[n_stages] = Data.r_k[n_stages] + Transpose(Data.S_k[n_stages - 1]) * Data.g_k[n_stages - 1];
	w_k[n_stages - 1] = Data.q_k[n_stages - 1] + Data.Q_k[n_stages - 1] * Data.g_k[n_stages - 1];

	for (int k = n_stages-1; k >=1; k--){
		Data.h_k[k] = Data.r_k[k] + Transpose(Data.S_k[k-1]) * Data.g_k[k-1] + Transpose(Data.B_k[k])*w_k[k];
		w_k[k-1] = Data.q_k[k-1] + Data.Q_k[k-1] * Data.g_k[k-1] + Transpose(Data.A_k[k-1]) * w_k[k];
	}
	Data.h_k[0] = Data.r_k[0] + Transpose(Data.B_k[0])*w_k[0];

	//calculate H
	for (int i = 0; i < n_stages; i++){
		W_ik[n_stages-1] = Data.Q_k[n_stages-1] * Data.G(n_stages-1, i);
		Data.H.set(n_stages, i, Transpose(Data.S_k[n_stages-1]) * Data.G(n_stages-1, i));
		for (int k = n_stages-1; k >= i+1; k--){
			Data.H.set(k,i, Transpose(Data.S_k[k-1])*Data.G(k-1, i) + Transpose(Data.B_k[k])*W_ik[k]);
			W_ik[k-1] = Transpose(Data.A_k[k-1])*W_ik[k] + Data.Q_k[k-1]*Data.G(k-1,i);
		}
		Data.H.set(i, i, Data.R_k[i] + Transpose(Data.B_k[i])*W_ik[i]);
	}
	Data.H.set(n_stages, n_stages, Data.R_k[n_stages]);


    //Calculate slice of condensed constraint-jacobian corresponding to target variables
    std::vector<Matrix> D_lb_k(n_stages);
    std::vector<Matrix> D_ub_k(n_stages);
    std::vector<Matrix> F_lb_k(n_stages + 1);
    std::vector<Matrix> F_ub_k(n_stages + 1);

    Data.G.to_sparse(Data.G_sparse);
    Data.H.to_sym(Data.H_dense);
    Data.g = vertcat(Data.g_k);
    Data.h = vertcat(Data.h_k);

	Data.J_free_k[0] = B_Jac.get_slice(0, B_Jac.m, 0, Data.alt_vranges[1]);
	Data.J_free_k[0].remove_rows(c_starts, c_ends, num_targets);
	F_lb_k[0] = B_lb_var.get_slice(0, Data.alt_vranges[1], 0, 1);
	F_ub_k[0] = B_ub_var.get_slice(0, Data.alt_vranges[1], 0, 1);
	for (int i = 1; i <= n_stages; i++){
		Data.J_dep_k[i-1] = B_Jac.get_slice(0, B_Jac.m, Data.alt_vranges[2*i-1], Data.alt_vranges[2*i]);
		Data.J_free_k[i] = B_Jac.get_slice(0, B_Jac.m, Data.alt_vranges[2*i], Data.alt_vranges[2*i+1]);
        Data.J_dep_k[i-1].remove_rows(c_starts, c_ends, num_targets);
        Data.J_free_k[i].remove_rows(c_starts, c_ends, num_targets);

		D_lb_k[i-1] = B_lb_var.get_slice(Data.alt_vranges[2*i - 1], Data.alt_vranges[2*i], 0, 1);
		D_ub_k[i-1] = B_ub_var.get_slice(Data.alt_vranges[2*i - 1], Data.alt_vranges[2*i], 0, 1);
		F_lb_k[i] = B_lb_var.get_slice(Data.alt_vranges[2*i], Data.alt_vranges[2*i+1], 0, 1);
        F_ub_k[i] = B_ub_var.get_slice(Data.alt_vranges[2*i], Data.alt_vranges[2*i+1], 0, 1);
	}

    //Calculate J_reduced = J_free + J_dep * G = J_free + J_dep * inv(A) * B

    Data.D_lb = vertcat(D_lb_k);
    Data.D_ub = vertcat(D_ub_k);
    Data.F_lb = vertcat(F_lb_k);
    Data.F_ub = vertcat(F_ub_k);

    Sparse_Matrix J_d = horzcat(Data.J_dep_k);

    CSR_Matrix J_fullrow;
    Data.J_reduced_k[n_stages] = Data.J_free_k[n_stages];

    Data.J_d_CSR_k[n_stages - 1] = CSR_Matrix(Data.J_dep_k[n_stages - 1]);
    Data.J_reduced_k[n_stages - 1] = Sparse_Matrix(add_fullrow(CSR_Matrix(Data.J_free_k[n_stages - 1]), sparse_dense_multiply_2(Data.J_d_CSR_k[n_stages - 1], Data.B_k[n_stages - 1])));

    if (n_stages > 1){
        J_fullrow = sparse_dense_multiply_2(Data.J_d_CSR_k[n_stages - 1], Data.A_k[n_stages - 2]);
        Data.J_d_CSR_k[n_stages - 2] = add_fullrow(CSR_Matrix(Data.J_dep_k[n_stages - 2]), J_fullrow);
        Data.J_reduced_k[n_stages - 2] = Sparse_Matrix(add_fullrow(CSR_Matrix(Data.J_free_k[n_stages - 2]), sparse_dense_multiply_2(Data.J_d_CSR_k[n_stages - 2], Data.B_k[n_stages - 2])));
    }

    for (int i = n_stages - 3; i >= 0; i--){
        J_fullrow = add_fullrow(sparse_dense_multiply_2(CSR_Matrix(Data.J_dep_k[i+1]), Data.A_k[i]), fullrow_multiply(J_fullrow, Data.A_k[i]));
        Data.J_d_CSR_k[i] = add_fullrow(CSR_Matrix(Data.J_dep_k[i]), J_fullrow);
        Data.J_reduced_k[i] = Sparse_Matrix(add_fullrow(CSR_Matrix(Data.J_free_k[i]), sparse_dense_multiply_2(Data.J_d_CSR_k[i], Data.B_k[i])));
    }

    Data.Jtimes_g = sparse_vector_multiply(J_d, Data.g);
    Data.J_reduced = horzcat(Data.J_reduced_k);

    return;
}


void Condenser::recover_var_mult(const Matrix &xi_cond, const Matrix &lambda_cond,
                                    Matrix &xi_full, Matrix &lambda_full){

    std::vector<Matrix> O_xi_cond(num_targets + 1);
    std::vector<Matrix> T_xi_cond(num_targets);
    std::vector<Matrix> O_mu(num_targets + 1);
    std::vector<Matrix> T_mu(num_targets);
    std::vector<Matrix> T_lambda(num_targets);
    std::vector<Matrix> O_sigma(num_targets + 1);
    Matrix sigma = lambda_cond.get_slice(condensed_num_vars, condensed_num_vars + num_true_cons);

    std::vector<Matrix> T_xi_full(num_targets);
    std::vector<Matrix> T_nu(num_targets);
    std::vector<Matrix> T_mu_lambda(num_targets);

    std::vector<Matrix> xi_full_k(2*num_targets + 1);
    std::vector<Matrix> lambda_full_k(2*(2*num_targets + 1));

    //Get slices corresponding to targets free variables and other free variables
    int ind = 0;
    for (int i = 0; i < num_targets; i++){
        O_xi_cond[i] = xi_cond.get_slice(ind, condensed_v_starts[i]);
        T_xi_cond[i] = xi_cond.get_slice(condensed_v_starts[i], condensed_v_ends[i]);

        O_mu[i] = lambda_cond.get_slice(ind, condensed_v_starts[i]);
        T_mu[i] = lambda_cond.get_slice(condensed_v_starts[i], condensed_v_ends[i]);

        ind = condensed_v_ends[i];
    }
    O_xi_cond[num_targets] = xi_cond.get_slice(condensed_v_ends[num_targets - 1], condensed_num_vars);
    O_mu[num_targets] = lambda_cond.get_slice(condensed_v_ends[num_targets - 1], condensed_num_vars);


    //Slice constraint multipliers to later insert continuity condition multipliers
    ind = 0;
    int ind_2 = condensed_num_vars;
    for (int i = 0; i < num_targets; i++){
        O_sigma[i] = lambda_cond.get_slice(ind_2, ind_2 + c_starts[i] - ind);
        ind_2 += c_starts[i] - ind;
        ind = c_ends[i];
    }
    O_sigma[num_targets] = lambda_cond.get_slice(ind_2, ind_2 + num_cons - ind);


    //Get multipliers for dependent variable bounds, or set them to zero if dependent variable bounds weren't added to constraints
    ind = condensed_num_vars + num_true_cons;
    if (add_dep_bounds){
        for (int i = 0; i < num_targets; i++){
            T_lambda[i] = lambda_cond.get_slice(ind, ind + targets_data[i].n_dep);
            ind += targets_data[i].n_dep;
        }
    }
    else{
        for (int i = 0; i < num_targets; i++){
            T_lambda[i].Dimension(targets_data[i].n_dep).Initialize(0.);
        }
    }


    //Recover dependent variables, compose them with free variables to vector T_xi_full, recover continuity condition multipliers nu,
    //assemble multipliers for free and dependent variable bounds
    for (int i = 0; i < num_targets; i++){
        single_recover(i, T_xi_cond[i], T_mu[i], T_lambda[i], sigma, T_xi_full[i], T_nu[i], T_mu_lambda[i]);
    }

    //Assemble complete vectors of uncondensed variables and corresponding bound-constraint multipliers
    for (int i = 0; i < num_targets; i++){
        xi_full_k[2*i] = O_xi_cond[i];
        xi_full_k[2*i + 1] = T_xi_full[i];
        lambda_full_k[2*i] = O_mu[i];
        lambda_full_k[2*i + 1] = T_mu_lambda[i];
    }
    xi_full_k[2*num_targets] = O_xi_cond[num_targets];
    lambda_full_k[2*num_targets] = O_mu[num_targets];

    //Append constraint and condition multipliers to bound-constraint multipliers
    ind = 2*num_targets + 1;
    for (int i = 0; i < num_targets; i++){
        lambda_full_k[ind + 2*i] = O_sigma[i];
        lambda_full_k[ind + 2*i + 1] = T_nu[i];
    }
    lambda_full_k[ind + 2*num_targets] = O_sigma[num_targets];

    xi_full = vertcat(xi_full_k);
    lambda_full = vertcat(lambda_full_k);

    return;
}

void Condenser::single_recover(int tnum, const Matrix &xi_free, const Matrix &mu, const Matrix &lambda, const Matrix &sigma,
                            Matrix &xi_full, Matrix &nu, Matrix &mu_lambda){
    int n_stages = targets[tnum].n_stages;
    condensing_data &Data = targets_data[tnum];

    std::vector<Matrix> xi_free_k(n_stages + 1);
    std::vector<Matrix> xi_dep_k(n_stages);
    std::vector<Matrix> nu_k(n_stages);
    std::vector<Matrix> lambda_k(n_stages);
    std::vector<Matrix> mu_k(n_stages + 1);
    std::vector<Matrix> xi_full_k(2*n_stages + 1);
    std::vector<Matrix> mu_lambda_k(2*n_stages + 1);

    int s_ind = 0;
    //int dep_size;

    //Get free variables of each stage
    for (int i = 0; i <= n_stages; i++){
        xi_free_k[i] = xi_free.get_slice(s_ind, s_ind + Data.free_sizes[i]);
        s_ind += Data.free_sizes[i];
    }

    //Get multipliers for each stage-state bound
    s_ind = 0;
    for (int i = 0; i < n_stages; i++){
        lambda_k[i] = lambda.get_slice(s_ind, s_ind + Data.cond_sizes[i]);
        s_ind += Data.cond_sizes[i];
    }

    //Get multipliers for free variable bounds for each stage
    s_ind = 0;
    for (int i = 0; i <= n_stages; i++){
        mu_k[i] = mu.get_slice(s_ind, s_ind + Data.free_sizes[i]);
        s_ind += Data.free_sizes[i];
    }

    //Recover dependent variables
    xi_dep_k[0] = Data.B_k[0]*xi_free_k[0] + Data.c_k[0];
    for (int i = 1; i < n_stages; i++){
        xi_dep_k[i] = Data.A_k[i-1]*xi_dep_k[i-1] + Data.B_k[i]*xi_free_k[i] + Data.c_k[i];
    }

    //Assemble original vector of free and dependent variables and corresponding vector of bound-constraint multipliers
    for (int i = 0; i < n_stages; i++){
        xi_full_k[2*i] = xi_free_k[i];
        xi_full_k[2*i + 1] = xi_dep_k[i];
        mu_lambda_k[2*i] = mu_k[i];
        mu_lambda_k[2*i + 1] = lambda_k[i];
    }
    xi_full_k[2*n_stages] = xi_free_k[n_stages];
    mu_lambda_k[2*n_stages] = mu_k[n_stages];

    //Calculate adjoint variables backwards in time
    //Definition of Lagrangian: 0.5 xT H x + qT * x - lambdaT * x - muT * (Ax - b)

    Matrix J_T_sigma(Data.cond_sizes[n_stages-1]);
    if (num_true_cons == 0){
        J_T_sigma.Initialize(0.);
    }
    else{
        J_T_sigma = transpose_multiply(Data.J_dep_k[n_stages - 1], sigma);
    }

    //If there are no free variables in the final stage N, S_N and xi_free_N have second and first dimension zero respectively and cannot be multiplied. We have to manually omit the term
    
    if (Data.alt_vranges[2*n_stages+1] - Data.alt_vranges[2*n_stages] > 0)
        nu_k[n_stages - 1] = Data.S_k[n_stages - 1] * xi_free_k[n_stages] + Data.Q_k[n_stages - 1] * xi_dep_k[n_stages - 1] + Data.q_k[n_stages - 1] - lambda_k[n_stages - 1] - J_T_sigma;
    else
        nu_k[n_stages - 1] = Data.Q_k[n_stages - 1] * xi_dep_k[n_stages - 1] + Data.q_k[n_stages - 1] - lambda_k[n_stages - 1] - J_T_sigma;
    
    for (int i = n_stages - 2; i>= 0; i--){
        if (num_true_cons == 0){
            J_T_sigma.Dimension(Data.cond_sizes[i]);
            J_T_sigma.Initialize(0.);
        }
        else{
            J_T_sigma = transpose_multiply(Data.J_dep_k[i], sigma);
        }

        nu_k[i] = Data.S_k[i] * xi_free_k[i+1] + Data.Q_k[i] * xi_dep_k[i] + Data.q_k[i] - lambda_k[i] + Transpose(Data.A_k[i]) * nu_k[i+1] - J_T_sigma;
    }
    
    nu = vertcat(nu_k);
    nu *= Data.matching_sign;
    xi_full = vertcat(xi_full_k);
    mu_lambda = vertcat(mu_lambda_k);

    return;
}


void Condenser::fallback_hessian_condense(const SymMatrix *const hess_2, Matrix &condensed_h_2, SymMatrix *condensed_hess_2){

    for (int tnum = 0; tnum < num_targets; tnum++){
        single_hess_condense(tnum, hess_2 + h_starts[tnum]);
    }

    //if (condensed_hess_2 == nullptr) condensed_hess_2 = new SymMatrix[condensed_num_hessblocks];
    

    //Assemble second condensed block hessian
    int ind_1 = 0;
    int ind_2 = 0;

    for (int tnum = 0; tnum < num_targets; tnum++){
        for (int i = 0; i < h_starts[tnum] - ind_2; i++){
            condensed_hess_2[ind_1 + i] = hess_2[ind_2 + i];
        }
        ind_1 += h_starts[tnum] - ind_2;
        ind_2 = h_ends[tnum];
        condensed_hess_2[ind_1] = targets_data[tnum].H_dense_2;
        ind_1++;
    }
    for (int i = 0; i < num_hessblocks - ind_2; i++){
        condensed_hess_2[ind_1 + i] = hess_2[ind_2 + i];
    }

    //Assemble second linear term vector h_2
    std::vector<Matrix> condensed_h_k_2(2*num_targets + 1);
    for (int tnum = 0; tnum < num_targets; tnum++){
        condensed_h_k_2[2*tnum] = O_grad_obj[tnum];
        condensed_h_k_2[2*tnum + 1] = targets_data[tnum].h_2;
    }
    condensed_h_k_2[2*num_targets] = O_grad_obj[num_targets];

    condensed_h_2 = vertcat(condensed_h_k_2);
}


void Condenser::single_hess_condense(int tnum, const SymMatrix *const sub_hess){
    condensing_data &Data = targets_data[tnum];

    int n_stages = targets[tnum].n_stages;

    Data.R_k_2[0] = sub_hess[0];
    for (int i = 1; i <= n_stages; i++){
        Data.Q_k_2[i - 1] = sub_hess[i].get_slice(0, Data.alt_vranges[2*i] - Data.alt_vranges[2*i-1], 0, Data.alt_vranges[2*i] - Data.alt_vranges[2*i-1]);
        Data.S_k_2[i - 1] = sub_hess[i].get_slice(0, Data.alt_vranges[2*i] - Data.alt_vranges[2*i-1], Data.alt_vranges[2*i] - Data.alt_vranges[2*i-1], Data.alt_vranges[2*i+1] - Data.alt_vranges[2*i-1]);
        Data.R_k_2[i] = sub_hess[i].get_slice(Data.alt_vranges[2*i] - Data.alt_vranges[2*i-1], Data.alt_vranges[2*i+1] - Data.alt_vranges[2*i-1], Data.alt_vranges[2*i] - Data.alt_vranges[2*i-1], Data.alt_vranges[2*i+1] - Data.alt_vranges[2*i-1]);
    }

    //Calculate new h
    std::vector<Matrix> w_k(n_stages);
    w_k[n_stages - 1] = Data.q_k[n_stages - 1] + Data.Q_k_2[n_stages - 1] * Data.g_k[n_stages - 1];

    Data.h_k_2[n_stages] = Data.r_k[n_stages] + Transpose(Data.S_k_2[n_stages - 1]) * Data.g_k[n_stages - 1];
    for (int k = n_stages - 1; k > 0; k--){
        Data.h_k_2[k] = Data.r_k[k] + Transpose(Data.S_k_2[k-1]) * Data.g_k[k-1] + Transpose(Data.B_k[k]) * w_k[k];
        w_k[k-1] = Data.Q_k_2[k-1] * Data.g_k[k-1] + Data.q_k[k-1] + Transpose(Data.A_k[k-1]) * w_k[k];
    }
    Data.h_k_2[0] = Data.r_k[0] + Transpose(Data.B_k[0]) * w_k[0];

    //Calculate new H
    for (int i = 0; i < n_stages; i++){
        w_k[n_stages - 1] = Data.Q_k_2[n_stages - 1] * Data.G(n_stages - 1, i);
        Data.H_2.set(n_stages, i, Transpose(Data.S_k_2[n_stages - 1]) * Data.G(n_stages - 1,i));
        for (int k = n_stages - 1; k > i; k--){
            Data.H_2.set(k, i, Transpose(Data.S_k_2[k-1]) * Data.G(k-1,i) + Transpose(Data.B_k[k]) * w_k[k]);
            w_k[k-1] = Data.Q_k_2[k-1] * Data.G(k-1, i) + Transpose(Data.A_k[k-1]) * w_k[k];
        }
        Data.H_2.set(i,i, Data.R_k_2[i] + Transpose(Data.B_k[i])*w_k[i]);
    }
    Data.H_2.set(n_stages, n_stages, Data.R_k_2[n_stages]);

    Data.h_2 = vertcat(Data.h_k_2);
    Data.H_2.to_sym(Data.H_dense_2);

    return;
}

void Condenser::convex_combination_recover(const Matrix &xi_cond, const Matrix &lambda_cond, const double t, Matrix &xi_full, Matrix &lambda_full){
    std::vector<Matrix> O_xi_cond(num_targets + 1);
    std::vector<Matrix> T_xi_cond(num_targets);
    std::vector<Matrix> O_mu(num_targets + 1);
    std::vector<Matrix> T_mu(num_targets);
    std::vector<Matrix> T_lambda(num_targets);
    std::vector<Matrix> O_sigma(num_targets + 1);
    Matrix sigma = lambda_cond.get_slice(condensed_num_vars, condensed_num_vars + num_true_cons);

    std::vector<Matrix> T_xi_full(num_targets);
    std::vector<Matrix> T_nu(num_targets);
    std::vector<Matrix> T_mu_lambda(num_targets);

    std::vector<Matrix> xi_full_k(2*num_targets + 1);
    std::vector<Matrix> lambda_full_k(2*(2*num_targets + 1));

    //Get slices corresponding to targets free variables and other free variables
    int ind = 0;
    for (int i = 0; i < num_targets; i++){
        O_xi_cond[i] = xi_cond.get_slice(ind, condensed_v_starts[i]);
        T_xi_cond[i] = xi_cond.get_slice(condensed_v_starts[i], condensed_v_ends[i]);

        O_mu[i] = lambda_cond.get_slice(ind, condensed_v_starts[i]);
        T_mu[i] = lambda_cond.get_slice(condensed_v_starts[i], condensed_v_ends[i]);

        ind = condensed_v_ends[i];
    }
    O_xi_cond[num_targets] = xi_cond.get_slice(condensed_v_ends[num_targets - 1], condensed_num_vars);
    O_mu[num_targets] = lambda_cond.get_slice(condensed_v_ends[num_targets - 1], condensed_num_vars);


    //Slice constraint multipliers to later insert continuity condition multipliers
    ind = 0;
    int ind_2 = condensed_num_vars;
    for (int i = 0; i < num_targets; i++){
        O_sigma[i] = lambda_cond.get_slice(ind_2, ind_2 + c_starts[i] - ind);
        ind_2 += c_starts[i] - ind;
        ind = c_ends[i];
    }
    O_sigma[num_targets] = lambda_cond.get_slice(ind_2, ind_2 + num_cons - ind);


    //Get multipliers for dependent variable bounds, or set them to zero if dependent variable bounds weren't added to constraints
    ind = condensed_num_vars + num_true_cons;
    if (add_dep_bounds){
        for (int i = 0; i < num_targets; i++){
            T_lambda[i] = lambda_cond.get_slice(ind, ind + targets_data[i].n_dep);
            ind += targets_data[i].n_dep;
        }
    }
    else{
        for (int i = 0; i < num_targets; i++){
            T_lambda[i].Dimension(targets_data[i].n_dep).Initialize(0.);
        }
    }

    //Recover dependent variables, compose them with free variables to vector T_xi_full, recover continuity condition multipliers nu,
    //assemble multipliers for free and dependent variable bounds
    for (int i = 0; i < num_targets; i++){
        single_convex_combination_recover(i, T_xi_cond[i], T_mu[i], T_lambda[i], sigma, t, T_xi_full[i], T_nu[i], T_mu_lambda[i]);
    }

    //Assemble complete vectors of uncondensed variables and corresponding bound-constraint multipliers
    for (int i = 0; i < num_targets; i++){
        xi_full_k[2*i] = O_xi_cond[i];
        xi_full_k[2*i + 1] = T_xi_full[i];
        lambda_full_k[2*i] = O_mu[i];
        lambda_full_k[2*i + 1] = T_mu_lambda[i];
    }
    xi_full_k[2*num_targets] = O_xi_cond[num_targets];
    lambda_full_k[2*num_targets] = O_mu[num_targets];

    //Append constraint and condition multipliers to bound-constraint multipliers
    ind = 2*num_targets + 1;
    for (int i = 0; i < num_targets; i++){
        lambda_full_k[ind + 2*i] = O_sigma[i];
        lambda_full_k[ind + 2*i + 1] = T_nu[i];
    }
    lambda_full_k[ind + 2*num_targets] = O_sigma[num_targets];

    xi_full = vertcat(xi_full_k);
    lambda_full = vertcat(lambda_full_k);

    return;
}

void Condenser::single_convex_combination_recover(int tnum, const Matrix &xi_free, const Matrix &mu, const Matrix &lambda, const Matrix &sigma, const double t,
                            Matrix &xi_full, Matrix &nu, Matrix &mu_lambda){
    int n_stages = targets[tnum].n_stages;
    condensing_data &Data = targets_data[tnum];

    std::vector<Matrix> xi_free_k(n_stages + 1);
    std::vector<Matrix> xi_dep_k(n_stages);
    std::vector<Matrix> nu_k(n_stages);
    std::vector<Matrix> lambda_k(n_stages);
    std::vector<Matrix> mu_k(n_stages + 1);
    std::vector<Matrix> xi_full_k(2*n_stages + 1);
    std::vector<Matrix> mu_lambda_k(2*n_stages + 1);

    int s_ind = 0;
    //int dep_size;

    //Get free variables of each stage
    for (int i = 0; i <= n_stages; i++){
        xi_free_k[i] = xi_free.get_slice(s_ind, s_ind + Data.free_sizes[i]);
        s_ind += Data.free_sizes[i];
    }

    //Get multipliers for each stage-state bound
    s_ind = 0;
    for (int i = 0; i < n_stages; i++){
        lambda_k[i] = lambda.get_slice(s_ind, s_ind + Data.cond_sizes[i]);
        s_ind += Data.cond_sizes[i];
    }

    //Get multipliers for free variable bounds for each stage
    s_ind = 0;
    for (int i = 0; i <= n_stages; i++){
        mu_k[i] = mu.get_slice(s_ind, s_ind + Data.free_sizes[i]);
        s_ind += Data.free_sizes[i];
    }

    //Recover dependent variables
    xi_dep_k[0] = Data.B_k[0]*xi_free_k[0] + Data.c_k[0];
    for (int i = 1; i < n_stages; i++){
        xi_dep_k[i] = Data.A_k[i-1]*xi_dep_k[i-1] + Data.B_k[i]*xi_free_k[i] + Data.c_k[i];
    }

    //Assemble original vector of free and dependent variables and corresponding vector of bound-constraint multipliers
    for (int i = 0; i < n_stages; i++){
        xi_full_k[2*i] = xi_free_k[i];
        xi_full_k[2*i + 1] = xi_dep_k[i];
        mu_lambda_k[2*i] = mu_k[i];
        mu_lambda_k[2*i + 1] = lambda_k[i];
    }
    xi_full_k[2*n_stages] = xi_free_k[n_stages];
    mu_lambda_k[2*n_stages] = mu_k[n_stages];

    //Calculate adjoint variables backward in time
    //Definition of Lagrangian: 0.5 xT H x + qT * x + lambdaT * x + muT * (Ax - b) + sigmaT J

    Matrix J_T_sigma(Data.cond_sizes[n_stages-1]);
    if (num_true_cons == 0){
        J_T_sigma.Initialize(0.);
    }
    else{
        J_T_sigma = transpose_multiply(Data.J_dep_k[n_stages - 1], sigma);
    }

    nu_k[n_stages - 1] = (Data.S_k[n_stages - 1]*(1-t) + Data.S_k_2[n_stages - 1]*t) * xi_free_k[n_stages] + (Data.Q_k[n_stages - 1]*(1-t) + Data.Q_k_2[n_stages - 1]*t) * xi_dep_k[n_stages - 1] + Data.q_k[n_stages - 1] - lambda_k[n_stages - 1] - J_T_sigma;
    for (int i = n_stages - 2; i>= 0; i--){
        if (num_true_cons == 0){
            J_T_sigma.Dimension(Data.cond_sizes[i]);
            J_T_sigma.Initialize(0.);
        }
        else{
            J_T_sigma = transpose_multiply(Data.J_dep_k[i], sigma);
        }

        nu_k[i] = (Data.S_k[i]*(1-t) + Data.S_k_2[i]*t) * xi_free_k[i+1] + (Data.Q_k[i]*(1-t) + Data.Q_k_2[i]*t) * xi_dep_k[i] + Data.q_k[i] - lambda_k[i] + Transpose(Data.A_k[i]) * nu_k[i+1] - J_T_sigma;
    }

    nu = vertcat(nu_k);
    nu *= Data.matching_sign;
    xi_full = vertcat(xi_full_k);
    mu_lambda = vertcat(mu_lambda_k);

    return;
}




void Condenser::new_hessian_condense(const SymMatrix *const hess, Matrix &condensed_h, SymMatrix *condensed_hess){

    for (int tnum = 0; tnum < num_targets; tnum++){
        single_new_hess_condense(tnum, hess + h_starts[tnum]);
    }

    //if (condensed_hess == nullptr) condensed_hess = new SymMatrix[condensed_num_hessblocks];

    //Assemble second condensed block hessian
    int ind_1 = 0;
    int ind_2 = 0;

    for (int tnum = 0; tnum < num_targets; tnum++){
        for (int i = 0; i < h_starts[tnum] - ind_2; i++){
            condensed_hess[ind_1 + i] = hess[ind_2 + i];
        }
        ind_1 += h_starts[tnum] - ind_2;
        ind_2 = h_ends[tnum];
        condensed_hess[ind_1] = targets_data[tnum].H_dense;
        ind_1++;
    }
    for (int i = 0; i < num_hessblocks - ind_2; i++){
        condensed_hess[ind_1 + i] = hess[ind_2 + i];
    }

    //Assemble new linear term vector h
    std::vector<Matrix> condensed_h_k(2*num_targets + 1);
    for (int tnum = 0; tnum < num_targets; tnum++){
        condensed_h_k[2*tnum] = O_grad_obj[tnum];
        condensed_h_k[2*tnum + 1] = targets_data[tnum].h;
    }
    condensed_h_k[2*num_targets] = O_grad_obj[num_targets];

    condensed_h = vertcat(condensed_h_k);
}


void Condenser::single_new_hess_condense(int tnum, const SymMatrix *const sub_hess){
    condensing_data &Data = targets_data[tnum];

    int n_stages = targets[tnum].n_stages;

    Data.R_k[0] = sub_hess[0];
    for (int i = 1; i <= n_stages; i++){
        Data.Q_k[i - 1] = sub_hess[i].get_slice(0, Data.alt_vranges[2*i] - Data.alt_vranges[2*i-1], 0, Data.alt_vranges[2*i] - Data.alt_vranges[2*i-1]);
        Data.S_k[i - 1] = sub_hess[i].get_slice(0, Data.alt_vranges[2*i] - Data.alt_vranges[2*i-1], Data.alt_vranges[2*i] - Data.alt_vranges[2*i-1], Data.alt_vranges[2*i+1] - Data.alt_vranges[2*i-1]);
        Data.R_k[i] = sub_hess[i].get_slice(Data.alt_vranges[2*i] - Data.alt_vranges[2*i-1], Data.alt_vranges[2*i+1] - Data.alt_vranges[2*i-1], Data.alt_vranges[2*i] - Data.alt_vranges[2*i-1], Data.alt_vranges[2*i+1] - Data.alt_vranges[2*i-1]);
    }

    //Calculate new h
    std::vector<Matrix> w_k(n_stages);
    w_k[n_stages - 1] = Data.q_k[n_stages - 1] + Data.Q_k[n_stages - 1] * Data.g_k[n_stages - 1];

    Data.h_k[n_stages] = Data.r_k[n_stages] + Transpose(Data.S_k[n_stages - 1]) * Data.g_k[n_stages - 1];
    for (int k = n_stages - 1; k > 0; k--){
        Data.h_k[k] = Data.r_k[k] + Transpose(Data.S_k[k-1]) * Data.g_k[k-1] + Transpose(Data.B_k[k]) * w_k[k];
        w_k[k-1] = Data.Q_k[k-1] * Data.g_k[k-1] + Data.q_k[k-1] + Transpose(Data.A_k[k-1]) * w_k[k];
    }
    Data.h_k[0] = Data.r_k[0] + Transpose(Data.B_k[0]) * w_k[0];

    //Calculate new H
    for (int i = 0; i < n_stages; i++){
        w_k[n_stages - 1] = Data.Q_k[n_stages - 1] * Data.G(n_stages - 1, i);
        Data.H.set(n_stages, i, Transpose(Data.S_k[n_stages - 1]) * Data.G(n_stages - 1,i));
        for (int k = n_stages - 1; k > i; k--){
            Data.H.set(k, i, Transpose(Data.S_k[k-1]) * Data.G(k-1,i) + Transpose(Data.B_k[k]) * w_k[k]);
            w_k[k-1] = Data.Q_k[k-1] * Data.G(k-1, i) + Transpose(Data.A_k[k-1]) * w_k[k];
        }
        Data.H.set(i,i, Data.R_k[i] + Transpose(Data.B_k[i])*w_k[i]);
    }
    Data.H.set(n_stages, n_stages, Data.R_k[n_stages]);

    Data.h = vertcat(Data.h_k);
    Data.H.to_sym(Data.H_dense);

    return;
}





void Condenser::SOC_condense(const Matrix &grad_obj, const Matrix &lb_con, const Matrix &ub_con, Matrix &condensed_h, Matrix &condensed_lb_con, Matrix &condensed_ub_con){

	O_grad_obj.resize(0);

    O_grad_obj.push_back(grad_obj.get_slice(0, v_starts[0], 0, 1));
	for (int i = 1; i < num_targets; i++){
        O_grad_obj.push_back(grad_obj.get_slice(v_ends[i-1], v_starts[i], 0, 1));
	}
    O_grad_obj.push_back(grad_obj.get_slice(v_ends[num_targets - 1], num_vars, 0, 1));

    //Assert that lower and upper bounds of condensing conditions are equal
    for (int tnum = 0; tnum < num_targets; tnum++){
        for (int i = c_starts[tnum]; i < c_ends[tnum]; i++){
            if (lb_con(i) - ub_con(i) >= 1e-14 || ub_con(i) - lb_con(i) >= 1e-14){
                throw std::invalid_argument("Error, Condensing conditions not equality constrained, difference (ub - lb)[" + std::to_string(i) + "] = " + std::to_string(ub_con(i) - lb_con(i)));
            }
        }
    }
    
    for (int i = 0; i < num_targets; i++){
        single_SOC_condense(i, lb_con);
    }
    
//Assemble reduced constraint-bounds (without dependent-variable bounds)
    Matrix reduced_lb_con = lb_con.without_rows(c_starts, c_ends, num_targets);
    Matrix reduced_ub_con = ub_con.without_rows(c_starts, c_ends, num_targets);
    for (int i = 0; i < num_targets; i++){
        reduced_lb_con -= targets_data[i].Jtimes_g;
        reduced_ub_con -= targets_data[i].Jtimes_g;
    }
//Add dependent variable bounds to constraints
    if (add_dep_bounds == 2){
        std::vector<Matrix> condensed_lb_con_k(num_targets + 1);
        std::vector<Matrix> condensed_ub_con_k(num_targets + 1);
        condensed_lb_con_k[0] = reduced_lb_con;
        condensed_ub_con_k[0] = reduced_ub_con;
        for (int tnum = 0; tnum < num_targets; tnum++){
            condensed_lb_con_k[tnum + 1] = targets_data[tnum].D_lb - targets_data[tnum].g;
            condensed_ub_con_k[tnum + 1] = targets_data[tnum].D_ub - targets_data[tnum].g;
        }
        condensed_lb_con = vertcat(condensed_lb_con_k);
        condensed_ub_con = vertcat(condensed_ub_con_k);
    }
    else if (add_dep_bounds == 1){
        std::vector<Matrix> condensed_lb_con_k(num_targets + 1);
        std::vector<Matrix> condensed_ub_con_k(num_targets + 1);
        condensed_lb_con_k[0] = reduced_lb_con;
        condensed_ub_con_k[0] = reduced_ub_con;
        for (int tnum = 0; tnum < num_targets; tnum++){
            condensed_lb_con_k[tnum + 1] = Matrix(targets_data[tnum].n_dep).Initialize(-std::numeric_limits<double>::infinity());
            condensed_ub_con_k[tnum + 1] = Matrix(targets_data[tnum].n_dep).Initialize(std::numeric_limits<double>::infinity());
        }
        condensed_lb_con = vertcat(condensed_lb_con_k);
        condensed_ub_con = vertcat(condensed_ub_con_k);

        //Save bounds on dependent variables so a user can manually add them to the qp
        std::vector<Matrix> lb_dep_var_k(num_targets);
        std::vector<Matrix> ub_dep_var_k(num_targets);
        for (int tnum = 0; tnum < num_targets; tnum++){
            lb_dep_var_k[tnum] = targets_data[tnum].D_lb - targets_data[tnum].g;
            ub_dep_var_k[tnum] = targets_data[tnum].D_ub - targets_data[tnum].g;
        }
        lb_dep_var = vertcat(lb_dep_var_k);
        ub_dep_var = vertcat(ub_dep_var_k);
    }
    else{
        condensed_lb_con = reduced_lb_con;
        condensed_ub_con = reduced_ub_con;
    }

//Assemble condensed_h, vector of linear term in objective
    std::vector<Matrix> condensed_h_k(2*num_targets+1);
    for (int i = 0; i < num_targets; i++){
        condensed_h_k[2*i] = O_grad_obj[i];
        condensed_h_k[2*i+1] = targets_data[i].h;
    }
    condensed_h_k[2*num_targets] = O_grad_obj[num_targets];
    condensed_h = vertcat(condensed_h_k);

    //std::chrono::steady_clock::time_point T2 = std::chrono::steady_clock::now();

    return;

}


void Condenser::single_SOC_condense(int tnum, const Matrix &lb_con){
	int n_stages = targets[tnum].n_stages;
	condensing_data &Data = targets_data[tnum];


	//Extract the updated c_k
	Data.c_k[0] = lb_con.get_slice(Data.cond_ranges[0], Data.cond_ranges[1]);
    Data.c_k[0] *= Data.matching_sign;

	for (int i = 1; i<n_stages; i++){
		Data.c_k[i] = lb_con.get_slice(Data.cond_ranges[i], Data.cond_ranges[i+1]);
        Data.c_k[i] *= Data.matching_sign;
	}


	std::vector<Matrix> w_k(n_stages);

    //std::chrono::steady_clock::time_point T1 = std::chrono::steady_clock::now();
	//calculate g
	Data.g_k[0] = Data.c_k[0];
	for (int i = 1; i < n_stages; i++){
		Data.g_k[i] = Data.A_k[i-1]*Data.g_k[i-1] + Data.c_k[i];
	}
    
	//calculate h
	Data.h_k[n_stages] = Data.r_k[n_stages] + Transpose(Data.S_k[n_stages - 1]) * Data.g_k[n_stages - 1];
	w_k[n_stages - 1] = Data.q_k[n_stages - 1] + Data.Q_k[n_stages - 1] * Data.g_k[n_stages - 1];

	for (int k = n_stages-1; k >=1; k--){
		Data.h_k[k] = Data.r_k[k] + Transpose(Data.S_k[k-1]) * Data.g_k[k-1] + Transpose(Data.B_k[k])*w_k[k];
		w_k[k-1] = Data.q_k[k-1] + Data.Q_k[k-1] * Data.g_k[k-1] + Transpose(Data.A_k[k-1]) * w_k[k];
	}
	Data.h_k[0] = Data.r_k[0] + Transpose(Data.B_k[0])*w_k[0];
    

    Data.g = vertcat(Data.g_k);
    Data.h = vertcat(Data.h_k);

    Sparse_Matrix J_d(horzcat(Data.J_dep_k));
    Data.Jtimes_g = sparse_dense_multiply(J_d, Data.g).dense();
    
    return;
}



void Condenser::correction_condense(const Matrix &grad_obj, const Matrix &lb_con, const Matrix &ub_con, const Matrix *const target_corrections, Matrix &condensed_h, Matrix &condensed_lb_con, Matrix &condensed_ub_con){

	O_grad_obj.resize(0);

    O_grad_obj.push_back(grad_obj.get_slice(0, v_starts[0], 0, 1));
	for (int i = 1; i < num_targets; i++){
        O_grad_obj.push_back(grad_obj.get_slice(v_ends[i-1], v_starts[i], 0, 1));
	}
    O_grad_obj.push_back(grad_obj.get_slice(v_ends[num_targets - 1], num_vars, 0, 1));

    //Assert that lower and upper bounds of condensing conditions are equal
    for (int tnum = 0; tnum < num_targets; tnum++){
        for (int i = c_starts[tnum]; i < c_ends[tnum]; i++){
            if (lb_con(i) - ub_con(i) >= 1e-14 || ub_con(i) - lb_con(i) >= 1e-14){
                throw std::invalid_argument("Error, Condensing conditions not equality constrained, difference (ub - lb)[" + std::to_string(i) + "] = " + std::to_string(ub_con(i) - lb_con(i)));
            }
        }
    }

    for (int i = 0; i < num_targets; i++){
        single_correction_condense(i, lb_con, target_corrections[i]);
    }

//Assemble reduced constraint-bounds (without dependent-variable bounds)
    Matrix reduced_lb_con = lb_con.without_rows(c_starts, c_ends, num_targets);
    Matrix reduced_ub_con = ub_con.without_rows(c_starts, c_ends, num_targets);
    for (int i = 0; i < num_targets; i++){
        reduced_lb_con -= targets_data[i].Jtimes_g;
        reduced_ub_con -= targets_data[i].Jtimes_g;
    }
//Add dependent variable bounds to constraints
    if (add_dep_bounds == 2){
        std::vector<Matrix> condensed_lb_con_k(num_targets + 1);
        std::vector<Matrix> condensed_ub_con_k(num_targets + 1);
        condensed_lb_con_k[0] = reduced_lb_con;
        condensed_ub_con_k[0] = reduced_ub_con;
        for (int tnum = 0; tnum < num_targets; tnum++){
            condensed_lb_con_k[tnum + 1] = targets_data[tnum].D_lb - targets_data[tnum].g;
            condensed_ub_con_k[tnum + 1] = targets_data[tnum].D_ub - targets_data[tnum].g;
        }
        condensed_lb_con = vertcat(condensed_lb_con_k);
        condensed_ub_con = vertcat(condensed_ub_con_k);
    }
    else if (add_dep_bounds == 1){
        std::vector<Matrix> condensed_lb_con_k(num_targets + 1);
        std::vector<Matrix> condensed_ub_con_k(num_targets + 1);
        condensed_lb_con_k[0] = reduced_lb_con;
        condensed_ub_con_k[0] = reduced_ub_con;
        for (int tnum = 0; tnum < num_targets; tnum++){
            condensed_lb_con_k[tnum + 1] = Matrix(targets_data[tnum].n_dep).Initialize(-std::numeric_limits<double>::infinity());
            condensed_ub_con_k[tnum + 1] = Matrix(targets_data[tnum].n_dep).Initialize(std::numeric_limits<double>::infinity());
        }
        condensed_lb_con = vertcat(condensed_lb_con_k);
        condensed_ub_con = vertcat(condensed_ub_con_k);

        //Save bounds on dependent variables so a user can manually add them to the qp
        std::vector<Matrix> lb_dep_var_k(num_targets);
        std::vector<Matrix> ub_dep_var_k(num_targets);
        for (int tnum = 0; tnum < num_targets; tnum++){
            lb_dep_var_k[tnum] = targets_data[tnum].D_lb - targets_data[tnum].g;
            ub_dep_var_k[tnum] = targets_data[tnum].D_ub - targets_data[tnum].g;
        }
        lb_dep_var = vertcat(lb_dep_var_k);
        ub_dep_var = vertcat(ub_dep_var_k);
    }
    else{
        condensed_lb_con = reduced_lb_con;
        condensed_ub_con = reduced_ub_con;
    }

//Assemble condensed_h, vector of linear term in objective
    std::vector<Matrix> condensed_h_k(2*num_targets+1);
    for (int i = 0; i < num_targets; i++){
        condensed_h_k[2*i] = O_grad_obj[i];
        condensed_h_k[2*i+1] = targets_data[i].h;
    }
    condensed_h_k[2*num_targets] = O_grad_obj[num_targets];
    condensed_h = vertcat(condensed_h_k);

    //std::chrono::steady_clock::time_point T2 = std::chrono::steady_clock::now();

    return;

}


void Condenser::single_correction_condense(int tnum, const Matrix &lb_con, const Matrix &correction){
	int n_stages = targets[tnum].n_stages;
	condensing_data &Data = targets_data[tnum];


	//Extract the updated c_k and correction term slices
	std::vector<Matrix> corr_k(n_stages);
	int ind_1 = 0;
    for (int i = 0; i < n_stages; i++){
        corr_k[i] = correction.get_slice(ind_1, ind_1 + Data.cond_sizes[i]);
        ind_1 += Data.cond_sizes[i];
    }

	Data.c_k[0] = lb_con.get_slice(Data.cond_ranges[0], Data.cond_ranges[1]);
    Data.c_k[0] *= Data.matching_sign;
	for (int i = 1; i<n_stages; i++){
		Data.c_k[i] = lb_con.get_slice(Data.cond_ranges[i], Data.cond_ranges[i+1]);
        Data.c_k[i] *= Data.matching_sign;
	}


	std::vector<Matrix> w_k(n_stages);

    //std::chrono::steady_clock::time_point T1 = std::chrono::steady_clock::now();

	//calculate g
	Data.g_k[0] = Data.c_k[0];
	for (int i = 1; i < n_stages; i++){
		Data.g_k[i] = Data.A_k[i-1]*Data.g_k[i-1] + Data.c_k[i];
	}

    //Add corrections
    for (int i = 0; i < n_stages; i++){
        Data.g_k[i] += corr_k[i];
    }

	//calculate h
	Data.h_k[n_stages] = Data.r_k[n_stages] + Transpose(Data.S_k[n_stages - 1]) * Data.g_k[n_stages - 1];
	w_k[n_stages - 1] = Data.q_k[n_stages - 1] + Data.Q_k[n_stages - 1] * Data.g_k[n_stages - 1];

	for (int k = n_stages-1; k >=1; k--){
		Data.h_k[k] = Data.r_k[k] + Transpose(Data.S_k[k-1]) * Data.g_k[k-1] + Transpose(Data.B_k[k])*w_k[k];
		w_k[k-1] = Data.q_k[k-1] + Data.Q_k[k-1] * Data.g_k[k-1] + Transpose(Data.A_k[k-1]) * w_k[k];
	}
	Data.h_k[0] = Data.r_k[0] + Transpose(Data.B_k[0])*w_k[0];


    Data.g = vertcat(Data.g_k);
    Data.h = vertcat(Data.h_k);

    Sparse_Matrix J_d(horzcat(Data.J_dep_k));
    Data.Jtimes_g = sparse_dense_multiply(J_d, Data.g).dense();

    return;
}


void Condenser::recover_correction_var_mult(const Matrix &xi_cond, const Matrix &lambda_cond, const Matrix *const target_corrections,
                                    Matrix &xi_full, Matrix &lambda_full){

    std::vector<Matrix> O_xi_cond(num_targets + 1);
    std::vector<Matrix> T_xi_cond(num_targets);
    std::vector<Matrix> O_mu(num_targets + 1);
    std::vector<Matrix> T_mu(num_targets);
    std::vector<Matrix> T_lambda(num_targets);
    std::vector<Matrix> O_sigma(num_targets + 1);
    Matrix sigma = lambda_cond.get_slice(condensed_num_vars, condensed_num_vars + num_true_cons);

    std::vector<Matrix> T_xi_full(num_targets);
    std::vector<Matrix> T_nu(num_targets);
    std::vector<Matrix> T_mu_lambda(num_targets);

    std::vector<Matrix> xi_full_k(2*num_targets + 1);
    std::vector<Matrix> lambda_full_k(2*(2*num_targets + 1));

    //Get slices corresponding to targets free variables and other free variables
    int ind = 0;
    for (int i = 0; i < num_targets; i++){
        O_xi_cond[i] = xi_cond.get_slice(ind, condensed_v_starts[i]);
        T_xi_cond[i] = xi_cond.get_slice(condensed_v_starts[i], condensed_v_ends[i]);

        O_mu[i] = lambda_cond.get_slice(ind, condensed_v_starts[i]);
        T_mu[i] = lambda_cond.get_slice(condensed_v_starts[i], condensed_v_ends[i]);

        ind = condensed_v_ends[i];
    }
    O_xi_cond[num_targets] = xi_cond.get_slice(condensed_v_ends[num_targets - 1], condensed_num_vars);
    O_mu[num_targets] = lambda_cond.get_slice(condensed_v_ends[num_targets - 1], condensed_num_vars);


    //Slice constraint multipliers to later insert continuity condition multipliers
    ind = 0;
    int ind_2 = condensed_num_vars;
    for (int i = 0; i < num_targets; i++){
        O_sigma[i] = lambda_cond.get_slice(ind_2, ind_2 + c_starts[i] - ind);
        ind_2 += c_starts[i] - ind;
        ind = c_ends[i];
    }
    O_sigma[num_targets] = lambda_cond.get_slice(ind_2, ind_2 + num_cons - ind);


    //Get multipliers for dependent variable bounds, or set them to zero if dependent variable bounds weren't added to constraints
    ind = condensed_num_vars + num_true_cons;
    if (add_dep_bounds){
        for (int i = 0; i < num_targets; i++){
            T_lambda[i] = lambda_cond.get_slice(ind, ind + targets_data[i].n_dep);
            ind += targets_data[i].n_dep;
        }
    }
    else{
        for (int i = 0; i < num_targets; i++){
            T_lambda[i].Dimension(targets_data[i].n_dep).Initialize(0.);
        }
    }

    //Recover dependent variables, compose them with free variables to vector T_xi_full, recover continuity condition multipliers nu,
    //assemble multipliers for free and dependent variable bounds
    for (int i = 0; i < num_targets; i++){
        single_correction_recover(i, T_xi_cond[i], T_mu[i], T_lambda[i], sigma, target_corrections[i], T_xi_full[i], T_nu[i], T_mu_lambda[i]);
    }

    //Assemble complete vectors of uncondensed variables and corresponding bound-constraint multipliers
    for (int i = 0; i < num_targets; i++){
        xi_full_k[2*i] = O_xi_cond[i];
        xi_full_k[2*i + 1] = T_xi_full[i];
        lambda_full_k[2*i] = O_mu[i];
        lambda_full_k[2*i + 1] = T_mu_lambda[i];
    }
    xi_full_k[2*num_targets] = O_xi_cond[num_targets];
    lambda_full_k[2*num_targets] = O_mu[num_targets];

    //Append constraint and condition multipliers to bound-constraint multipliers
    ind = 2*num_targets + 1;
    for (int i = 0; i < num_targets; i++){
        lambda_full_k[ind + 2*i] = O_sigma[i];
        lambda_full_k[ind + 2*i + 1] = T_nu[i];
    }
    lambda_full_k[ind + 2*num_targets] = O_sigma[num_targets];

    xi_full = vertcat(xi_full_k);
    lambda_full = vertcat(lambda_full_k);

    return;
}

void Condenser::single_correction_recover(int tnum, const Matrix &xi_free, const Matrix &mu, const Matrix &lambda, const Matrix &sigma, const Matrix &correction,
                            Matrix &xi_full, Matrix &nu, Matrix &mu_lambda){
    int n_stages = targets[tnum].n_stages;
    condensing_data &Data = targets_data[tnum];

    std::vector<Matrix> xi_free_k(n_stages + 1);
    std::vector<Matrix> xi_dep_k(n_stages);
    std::vector<Matrix> nu_k(n_stages);
    std::vector<Matrix> lambda_k(n_stages);
    std::vector<Matrix> mu_k(n_stages + 1);
    std::vector<Matrix> xi_full_k(2*n_stages + 1);
    std::vector<Matrix> mu_lambda_k(2*n_stages + 1);

    int s_ind = 0;
    //int dep_size;

    //Get free variables of each stage
    for (int i = 0; i <= n_stages; i++){
        xi_free_k[i] = xi_free.get_slice(s_ind, s_ind + Data.free_sizes[i]);
        s_ind += Data.free_sizes[i];
    }

    //Get multipliers for each stage-state bound
    s_ind = 0;
    for (int i = 0; i < n_stages; i++){
        lambda_k[i] = lambda.get_slice(s_ind, s_ind + Data.cond_sizes[i]);
        s_ind += Data.cond_sizes[i];
    }

    //Get multipliers for free variable bounds for each stage
    s_ind = 0;
    for (int i = 0; i <= n_stages; i++){
        mu_k[i] = mu.get_slice(s_ind, s_ind + Data.free_sizes[i]);
        s_ind += Data.free_sizes[i];
    }

    //Slice correction values
	std::vector<Matrix> corr_k(n_stages);
	int ind_1 = 0;
    for (int i = 0; i < n_stages; i++){
        corr_k[i] = correction.get_slice(ind_1, ind_1 + Data.cond_sizes[i]);
        ind_1 += Data.cond_sizes[i];
    }

    //Recover dependent variables
    xi_dep_k[0] = Data.B_k[0]*xi_free_k[0] + Data.c_k[0] + corr_k[0];
    for (int i = 1; i < n_stages; i++){
        xi_dep_k[i] = Data.A_k[i-1]*xi_dep_k[i-1] + Data.B_k[i]*xi_free_k[i] + Data.c_k[i] + corr_k[i];
    }

    //Assemble original vector of free and dependent variables and corresponding vector of bound-constraint multipliers
    for (int i = 0; i < n_stages; i++){
        xi_full_k[2*i] = xi_free_k[i];
        xi_full_k[2*i + 1] = xi_dep_k[i];
        mu_lambda_k[2*i] = mu_k[i];
        mu_lambda_k[2*i + 1] = lambda_k[i];
    }
    xi_full_k[2*n_stages] = xi_free_k[n_stages];
    mu_lambda_k[2*n_stages] = mu_k[n_stages];

    //Calculate adjoint variables backwards in time
    //Definition of Lagrangian: 0.5 xT H x + qT * x - lambdaT * x - muT * (Ax - b)

    Matrix J_T_sigma(Data.cond_sizes[n_stages-1]);
    if (num_true_cons == 0){
        J_T_sigma.Initialize(0.);
    }
    else{
        J_T_sigma = transpose_multiply(Data.J_dep_k[n_stages - 1], sigma);
    }

    nu_k[n_stages - 1] = Data.S_k[n_stages - 1] * xi_free_k[n_stages] + Data.Q_k[n_stages - 1] * xi_dep_k[n_stages - 1] + Data.q_k[n_stages - 1] - lambda_k[n_stages - 1] - J_T_sigma;
    for (int i = n_stages - 2; i>= 0; i--){
        if (num_true_cons == 0){
            J_T_sigma.Dimension(Data.cond_sizes[i]);
            J_T_sigma.Initialize(0.);
        }
        else{
            J_T_sigma = transpose_multiply(Data.J_dep_k[i], sigma);
        }

        nu_k[i] = Data.S_k[i] * xi_free_k[i+1] + Data.Q_k[i] * xi_dep_k[i] + Data.q_k[i] - lambda_k[i] + Transpose(Data.A_k[i]) * nu_k[i+1] - J_T_sigma;
    }

    nu = vertcat(nu_k);
    nu *= Data.matching_sign;
    xi_full = vertcat(xi_full_k);
    mu_lambda = vertcat(mu_lambda_k);

    return;
}


holding_Condenser::holding_Condenser(
                    std::unique_ptr<vblock[]> VBLOCKS, int n_VBLOCKS, 
                    std::unique_ptr<cblock[]> CBLOCKS, int n_CBLOCKS, 
                    std::unique_ptr<int[]> HSIZES, int n_HBLOCKS,
                    std::unique_ptr<condensing_target[]> TARGETS, int n_TARGETS, 
                    int DEP_BOUNDS):
                        Condenser(VBLOCKS.get(), n_VBLOCKS, CBLOCKS.get(), n_CBLOCKS, HSIZES.get(), n_HBLOCKS, TARGETS.get(), n_TARGETS, DEP_BOUNDS),
                        auto_vblocks(std::move(VBLOCKS)), auto_cblocks(std::move(CBLOCKS)), auto_hess_block_sizes(std::move(HSIZES)), auto_targets(std::move(TARGETS))
                        {}


//Moved to restoration.cpp
/*
holding_Condenser* create_restoration_Condenser(Condenser *parent, int DEP_BOUNDS){
    int N_vblocks = parent->num_vblocks + parent->num_true_cons;
    int N_cblocks = parent->num_cblocks;
    int N_hessblocks = parent->num_hessblocks + parent->num_true_cons;
    int N_targets = parent->num_targets;

	std::unique_ptr<vblock[]> rest_vblocks = std::make_unique<vblock[]>(N_vblocks);
    std::unique_ptr<cblock[]> rest_cblocks = std::make_unique<cblock[]>(N_cblocks);
	std::unique_ptr<int[]> rest_hess_block_sizes = std::make_unique<int[]>(N_hessblocks);
	std::unique_ptr<condensing_target[]> rest_targets = std::make_unique<condensing_target[]>(N_targets);

    for (int i = 0; i < parent->num_vblocks; i++){
        rest_vblocks[i] = parent->vblocks[i];
    }
    for (int i = parent->num_vblocks; i < N_vblocks; i++){
        rest_vblocks[i] = vblock(1, false);
    }

    for (int i = 0; i < parent->num_cblocks; i++){
        rest_cblocks[i] = parent->cblocks[i];
    }

    for (int i = 0; i<parent->num_hessblocks; i++){
        rest_hess_block_sizes[i] = parent->hess_block_sizes[i];
    }
    for (int i = parent->num_hessblocks; i<N_hessblocks; i++){
        rest_hess_block_sizes[i] = 1;
    }

    for (int i = 0; i<parent->num_targets; i++){
        rest_targets[i] = parent->targets[i];
    }

    return new holding_Condenser(std::move(rest_vblocks), N_vblocks, std::move(rest_cblocks), N_cblocks, std::move(rest_hess_block_sizes), N_hessblocks, std::move(rest_targets), N_targets, DEP_BOUNDS);
}
*/




} // namespace blockSQP2








