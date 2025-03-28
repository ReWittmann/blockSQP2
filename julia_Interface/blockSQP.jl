#=
 * \file blockSQP.jl
 * \author Reinhold Wittmann
 * \date 2024-
 *
 * Julia side functions and structures for the blockSQP nonlinear programming solver
=#

module blockSQP
	import Base.setproperty!, Base.getproperty
	
	using CxxWrap
	@readmodule(()->joinpath(pwd(), "libblockSQP_jl.so"))
	@wraptypes
	@wrapfunctions
  
	function __init__()
		@initcxx
	end
	
	function fnothing()
	end
	
	export setindex!

	function initialize_dense(Prob::Ptr{Nothing}, xi::CxxPtr{Float64}, lam::CxxPtr{Float64}, Jac::CxxPtr{Float64})
		Jprob = unsafe_pointer_to_objref(Prob)::BlockSQP_Problem
		xi_arr = unsafe_wrap(Array{Float64, 1}, xi.cpp_object, Jprob.nVar, own = false)
		lam_arr = unsafe_wrap(Array{Float64, 1}, lam.cpp_object, Jprob.nVar + Jprob.nCon, own = false)
		
		xi_arr[:] = Jprob.x_start
		lam_arr[:] = Jprob.lam_start
		return
	end
	
	function evaluate_dense(Prob::Ptr{Nothing}, xi::ConstCxxPtr{Float64}, lam::ConstCxxPtr{Float64}, objval::CxxPtr{Float64}, constr::CxxPtr{Float64}, gradObj::CxxPtr{Float64}, constrJac::CxxPtr{Float64}, hess::CxxPtr{CxxPtr{Float64}}, dmode::Int32, info::CxxPtr{Int32})
		
		Jprob = unsafe_pointer_to_objref(Prob)::BlockSQP_Problem
		xi_arr = unsafe_wrap(Array{Float64, 1}, xi.cpp_object, Jprob.nVar, own = false)
		lam_arr = unsafe_wrap(Array{Float64, 1}, lam.cpp_object, Jprob.nVar + Jprob.nCon, own = false)
		constr_arr = unsafe_wrap(Array{Float64, 1}, constr.cpp_object, Jprob.nCon, own = false)
		
		objval[] = Jprob.f(xi_arr)
		constr_arr[:] = Jprob.g(xi_arr)
		
		if dmode > 0
			gradObj_arr = unsafe_wrap(Array{Float64, 1}, gradObj.cpp_object, Jprob.nVar, own = false)
			constrJac_arr = unsafe_wrap(Array{Float64, 2}, constrJac.cpp_object, (Jprob.nCon, Jprob.nVar), own = false)
			gradObj_arr[:] = Jprob.grad_f(xi_arr)
			constrJac_arr[:,:] = Jprob.jac_g(xi_arr)
			
			if dmode == 2
				hess_arr = unsafe_wrap(Array{CxxPtr{Float64}, 1}, hess.cpp_object, Jprob.n_hessblocks, own = true)

				s = Jprob.blockIdx[Jprob.n_hessblocks + 1] - Jprob.blockIdx[Jprob.n_hessblocks]
				hess_last = unsafe_wrap(Array{Float64,1}, hess_arr[Jprob.n_hessblocks].cpp_object, Int32((s*(s+Int32(1)))//(Int32(2))), own = false)
				hess_last[:] = Jprob.last_hessBlock(xi_arr, lam_arr[Jprob.nVar + 1 : Jprob.nVar + Jprob.nCon])
			elseif dmode == 3
				hessPTR_arr = unsafe_wrap(Array{CxxPtr{Float64}, 1}, hess.cpp_object, Jprob.n_hessblocks, own = true)
				hess_arr = Array{Array{Float64, 1}, 1}(undef, Jprob.n_hessblocks)
				for i = 1:Jprob.n_hessblocks
					Bsize = Jprob.blockIdx[i+1] - Jprob.blockIdx[i]
					hess_arr[i] = unsafe_wrap(Array{Float64,1}, hessPTR_arr[i].cpp_object, Int32((Bsize*(Bsize + Int32(1)))//Int32(2)), own = false)
				end

				hess_eval = Jprob.hess(xi_arr, lam_arr[Jprob.nVar + 1 : Jprob.nVar + Jprob.nCon])
				for i = 1:Jprob.n_hessblocks
					hess_arr[i][:] = hess_eval[i]
				end
			end
		end
		info[] = Int32(0);
		return
	end
	
	function evaluate_simple(Prob::Ptr{Nothing}, xi::ConstCxxPtr{Float64}, objval::CxxPtr{Float64}, constr::CxxPtr{Float64}, info::CxxPtr{Int32})
		Jprob = unsafe_pointer_to_objref(Prob)::BlockSQP_Problem
		xi_arr = unsafe_wrap(Array{Float64, 1}, xi.cpp_object, Jprob.nVar, own = false)
		constr_arr = unsafe_wrap(Array{Float64, 1}, constr.cpp_object, Jprob.nCon, own = false)
		
		objval[] = Jprob.f(xi_arr)
		constr_arr[:] = Jprob.g(xi_arr)

		info[] = Int32(0);
		return
	end
		
	
	function initialize_sparse(Prob::Ptr{Nothing}, xi::CxxPtr{Float64}, lam::CxxPtr{Float64}, jac_nz::CxxPtr{Float64}, jac_row::CxxPtr{Int32}, jac_colind::CxxPtr{Int32})
		Jprob = unsafe_pointer_to_objref(Prob)::BlockSQP_Problem
		xi_arr = unsafe_wrap(Array{Float64, 1}, xi.cpp_object, Jprob.nVar, own = false)
		lam_arr = unsafe_wrap(Array{Float64, 1}, lam.cpp_object, Jprob.nVar + Jprob.nCon, own = false)
		jac_row_arr = unsafe_wrap(Array{Int32, 1}, jac_row.cpp_object, Jprob.nnz, own = false)
		jac_colind_arr = unsafe_wrap(Array{Int32, 1}, jac_colind.cpp_object, Jprob.nVar + 1, own = false)

		xi_arr[:] = Jprob.x_start
		lam_arr[:] = Jprob.lam_start
		jac_row_arr[:] = Jprob.jac_g_row
		jac_colind_arr[:] = Jprob.jac_g_colind
		return
	end


	function evaluate_sparse(Prob::Ptr{Nothing}, xi::ConstCxxPtr{Float64}, lam::ConstCxxPtr{Float64}, objval::CxxPtr{Float64}, constr::CxxPtr{Float64}, gradObj::CxxPtr{Float64}, jac_nz::CxxPtr{Float64}, jac_row::CxxPtr{Int32}, jac_colind::CxxPtr{Int32}, hess::CxxPtr{CxxPtr{Float64}}, dmode::Int32, info::CxxPtr{Int32})
		Jprob = unsafe_pointer_to_objref(Prob)::BlockSQP_Problem
		xi_arr = unsafe_wrap(Array{Float64, 1}, xi.cpp_object, Jprob.nVar, own = false)
		lam_arr = unsafe_wrap(Array{Float64, 1}, lam.cpp_object, Jprob.nVar + Jprob.nCon, own = false)
		constr_arr = unsafe_wrap(Array{Float64, 1}, constr.cpp_object, Jprob.nCon, own = false)
		jac_nz_arr = unsafe_wrap(Array{Float64, 1}, jac_nz.cpp_object, Jprob.nnz, own = false)
		
		objval[] = Jprob.f(xi_arr)
		constr_arr[:] = Jprob.g(xi_arr)
		
		if dmode > 0
			gradObj_arr = unsafe_wrap(Array{Float64, 1}, gradObj.cpp_object, Jprob.nVar, own = false)
			gradObj_arr[:] = Jprob.grad_f(xi_arr)
			jac_nz_arr[:] = Jprob.jac_g_nz(xi_arr)
			
			if dmode == 2
				hess_arr = unsafe_wrap(Array{CxxPtr{Float64}, 1}, hess.cpp_object, Jprob.n_hessblocks, own = true)

				s_last = Jprob.blockIdx[Jprob.n_hessblocks + 1] - Jprob.blockIdx[Jprob.n_hesslbocks]
				hess_last = unsafe_wrap(Array{Float64,1}, hess_arr[Jprob.n_hessblocks].cpp_object, Int32((s*(s+Int32(1)))//(Int32(2))), own = false)
				hess_last[:] = Jprob.last_hessBlock(xi_arr, lam_arr[Jprob.nVar + 1 : Jprob.nVar + Jprob.nCon])
			end
			
			if dmode == 3
				hessPTR_arr = unsafe_wrap(Array{CxxPtr{Float64}, 1}, hess.cpp_object, Jprob.n_hessblocks, own = true)
				hess_arr = Array{Array{Float64, 1}, 1}(undef, Jprob.n_hessblocks)
				for i = 1:Jprob.n_hessblocks
					Bsize = Jprob.blockIdx[i+1] - Jprob.blockIdx[i]
					hess_arr[i] = unsafe_wrap(Array{Float64,1}, hessPTR_arr[i].cpp_object, Int32((Bsize*(Bsize + Int32(1)))//Int32(2)), own = false)
				end

				hess_eval = Jprob.hess(xi_arr, lam_arr[Jprob.nVar + 1 : Jprob.nVar + Jprob.nCon])
				for i = 1:Jprob.n_hessblocks
					hess_arr[i][:] = hess_eval[i]
				end
			end
		end
		info[] = Int32(0);
		return
	end
	
	
	function reduceConstrVio(Prob::Ptr{Nothing}, xi::CxxPtr{Float64}, info::CxxPtr{Int32})
		Jprob = unsafe_pointer_to_objref(Prob)::BlockSQP_Problem
		if Jprob.continuity_restoration == fnothing
			info[] = Int32(1)
		else
			xi_arr = unsafe_wrap(Array{Float64, 1}, xi.cpp_object, Jprob.nVar, own = false)
			xi_arr[:] = Jprob.continuity_restoration(xi_arr)
		end
		return
	end



	mutable struct BlockSQP_Problem
		#Cxx_Problem::Problemform
		
		nVar::Int32
		nCon::Int32
		nnz::Int32
		blockIdx::Array{Int32, 1}
		vblocks::Array{vblock, 1}

		lb_var::Array{Float64, 1}
		ub_var::Array{Float64, 1}
		lb_con::Array{Float64, 1}
		ub_con::Array{Float64, 1}
		objLo::Float64
		objUp::Float64
		
		f::Function
		g::Function
		grad_f::Function
		jac_g::Function
		jac_g_nz::Function
		continuity_restoration::Function
		last_hessBlock::Function
		hess::Function
		
		jac_g_row::Array{Int32, 1}
		jac_g_colind::Array{Int32, 1}
		
		x_start::Array{Float64, 1}
		lam_start::Array{Float64, 1}
		
		BlockSQP_Problem(nVar::Int32, nCon::Int32) = new(nVar, nCon, Int32(-1), Int32[0, nVar], vblock[vblock(1, false)], 
						Float64[], Float64[], Float64[], Float64[], -Inf, Inf,
						fnothing, fnothing, fnothing, fnothing, fnothing, fnothing, fnothing, fnothing,
						Int32[], Int32[], Float64[], Float64[]
		)
	end


	struct Solver
		BSQP_solver::SQPmethod
		Cxx_Problem::Problemform
		BSQP_options::SQPoptions
		QPsol_opts::QPSOLVER_options
		BSQP_stats::SQPstats
		Jul_Problem::BlockSQP_Problem
		Options::Dict
		Solver(J_prob::BlockSQP_Problem, param::Dict, cxx_stats::SQPstats) = begin
			#Create problem class on the C++ side
			cxx_prob = Problemform(J_prob.nVar, J_prob.nCon)
			set_scope(cxx_prob, pointer_from_objref(J_prob))
			set_dense_init(cxx_prob, @safe_cfunction(initialize_dense, Nothing, (Ptr{Nothing}, CxxPtr{Float64}, CxxPtr{Float64}, CxxPtr{Float64})))
			set_dense_eval(cxx_prob, @safe_cfunction(evaluate_dense, Nothing, (Ptr{Nothing}, ConstCxxPtr{Float64}, ConstCxxPtr{Float64}, CxxPtr{Float64}, CxxPtr{Float64}, CxxPtr{Float64}, CxxPtr{Float64}, CxxPtr{CxxPtr{Float64}}, Int32, CxxPtr{Int32})))
			set_simple_eval(cxx_prob, @safe_cfunction(evaluate_simple, Nothing, (Ptr{Nothing}, ConstCxxPtr{Float64}, CxxPtr{Float64}, CxxPtr{Float64}, CxxPtr{Int32})))
			set_sparse_init(cxx_prob, @safe_cfunction(initialize_sparse, Nothing, (Ptr{Nothing}, CxxPtr{Float64}, CxxPtr{Float64}, CxxPtr{Float64}, CxxPtr{Int32}, CxxPtr{Int32})))
			set_sparse_eval(cxx_prob, @safe_cfunction(evaluate_sparse, Nothing, (Ptr{Nothing}, ConstCxxPtr{Float64}, ConstCxxPtr{Float64}, CxxPtr{Float64}, CxxPtr{Float64}, CxxPtr{Float64}, CxxPtr{Float64}, CxxPtr{Int32}, CxxPtr{Int32}, CxxPtr{CxxPtr{Float64}}, Int32, CxxPtr{Int32})))
			set_continuity_restoration(cxx_prob, @safe_cfunction(reduceConstrVio, Nothing, (Ptr{Nothing}, CxxPtr{Float64}, CxxPtr{Int32})))
			set_blockIdx(cxx_prob, J_prob.blockIdx)
			set_nnz(cxx_prob, J_prob.nnz)
			set_bounds(cxx_prob, J_prob.lb_var, J_prob.ub_var, J_prob.lb_con, J_prob.ub_con, J_prob.objLo, J_prob.objUp)
			
			VB = vblock_array(length(J_prob.vblocks))
			for i = 1:length(J_prob.vblocks)
				array_set(VB, i, J_prob.vblocks[i])
			end
			set_vblocks(cxx_prob, CxxRef(VB))
			
			#OLD: Create options class on the C++ side
			#cxx_opts = BSQP_options(param)
			
			###################################################################################################################
			#NEW: Create default options class on the C++ side
			cxx_opts = SQPoptions()
			opt_keys = keys(param)
			
			cxx_QPsol_opts = NULL_QPSOLVER_options()
			#Pass the options. TODO: Is there a more elegant way to do this?
			if "printLevel" in opt_keys
				set_printLevel(cxx_opts, Int32(param["printLevel"]))
			end
			if "printColor" in opt_keys
				set_printColor(cxx_opts, Int32(param["printColor"]))
			end
			if "debugLevel" in opt_keys
				set_debugLevel(cxx_opts, Int32(param["debugLevel"]))
			end
			if "eps" in opt_keys
				set_eps(cxx_opts, Float64(param["eps"]))
			end
			if "inf" in opt_keys
				set_inf(cxx_opts, Float64(param["inf"]))
			end
			if "opttol" in opt_keys
				set_opttol(cxx_opts, Float64(param["opttol"]))
			end
			if "nlinfeastol" in opt_keys
				set_nlinfeastol(cxx_opts, Float64(param["nlinfeastol"]))
			end
			if "sparseQP" in opt_keys
				set_sparseQP(cxx_opts, Int32(param["sparseQP"]))
			end
			if "globalization" in opt_keys
				set_globalization(cxx_opts, Int32(param["globalization"]))
			end
			if "restoreFeas" in opt_keys
				set_restoreFeas(cxx_opts, Int32(param["restoreFeas"]))
			end
			if "maxLineSearch" in opt_keys
				set_maxLineSearch(cxx_opts, Int32(param["maxLineSearch"]))
			end
			if "maxConsecReducedSteps" in opt_keys
				set_maxConsecReducedSteps(cxx_opts, Int32(param["maxConsecReducedSteps"]))
			end
			if "maxConsecSkippedUpdates" in opt_keys
				set_maxConsecSkippedUpdates(cxx_opts, Int32(param["maxConsecSkippedUpdates"]))
			end
			if "maxItQP" in opt_keys
				set_maxItQP(cxx_opts, Int32(param["maxItQP"]))
			end
			if "blockHess" in opt_keys
				set_blockHess(cxx_opts, Int32(param["blockHess"]))
			end
			if "hessScaling" in opt_keys
				set_hessScaling(cxx_opts, Int32(param["hessScaling"]))
			end
			if "fallbackScaling" in opt_keys
				set_fallbackScaling(cxx_opts, Int32(param["fallbackScaling"]))
			end
			if "maxTimeQP" in opt_keys
				set_maxTimeQP(cxx_opts, Int32(param["maxTimeQP"]))
			end
			if "iniHessDiag" in opt_keys
				set_iniHessDiag(cxx_opts, Float64(param["iniHessDiag"]))
			end
			if "colEps" in opt_keys
				set_colEps(cxx_opts, Float64(param["colEps"]))
			end
			if "olEps" in opt_keys
				set_olEps(cxx_opts, Float64(param["olEps"]))
			end
			if "colTau1" in opt_keys
				set_colTau1(cxx_opts, Float64(param["colTau1"]))
			end
			if "colTau2" in opt_keys
				set_colTau2(cxx_opts, Float64(param["colTau2"]))
			end
			if "hessDamp" in opt_keys
				set_hessDamp(cxx_opts, Int32(param["hessDamp"]))
			end
			if "hessDampFac" in opt_keys
				set_hessDampFac(cxx_opts, Float64(param["hessDampFac"]))
			end
			if "minDampQuot" in opt_keys
				set_minDampQuot(cxx_opts, Float64(param["minDampQuot"]))
			end
			if "hessUpdate" in opt_keys
				set_hessUpdate(cxx_opts, Int32(param["hessUpdate"]))
			end
			if "fallbackUpdate" in opt_keys
				set_fallbackUpdate(cxx_opts, Int32(param["fallbackUpdate"]))
			end
			if "indef_local_only" in opt_keys
				set_indef_local_only(cxx_opts, Bool(param["indef_local_only"]))
			end
			if "hessLimMem" in opt_keys
				set_hessLimMem(cxx_opts, Int32(param["hessLimMem"]))
			end
			if "hessMemsize" in opt_keys
				set_hessMemsize(cxx_opts, Int32(param["hessMemsize"]))
			end
			if "whichSecondDerv" in opt_keys
				set_whichSecondDerv(cxx_opts, Int32(param["whichSecondDerv"]))
			end
			if "skipFirstGlobalization" in opt_keys
				set_skipFirstGlobalization(cxx_opts, Int32(param["skipFirstGlobalization"]))
			end
			if "convStrategy" in opt_keys
				set_convStrategy(cxx_opts, Int32(param["convStrategy"]))
			end
			if "maxConvQP" in opt_keys
				set_maxConvQP(cxx_opts, Int32(param["maxConvQP"]))
			end
			if "maxSOCiter" in opt_keys
				set_maxSOCiter(cxx_opts, Int32(param["maxSOCiter"]))
			end
			if "max_bound_refines" in opt_keys
				set_max_bound_refines(cxx_opts, Int32(param["max_bound_refines"]))
			end
			if "max_correction_steps" in opt_keys
				set_max_correction_steps(cxx_opts, Int32(param["max_correction_steps"]))
			end
			if "dep_bound_tolerance" in opt_keys
				set_dep_bound_tolerance(cxx_opts, Float64(param["dep_bound_tolerance"]))
			end
			if "QPsol" in opt_keys
				set_QPsol(cxx_opts, StdString(param["QPsol"]))
			end
			if "QPsol_opts" in opt_keys
				QPsol_param = param["QPsol_opts"]
				QPopt_keys = keys(QPsol_param)
				if get_QPsol(cxx_opts) == "qpOASES"
					cxx_QPsol_opts = qpOASES_options()
					if "printLevel" in QPopt_keys
						set_printLevel(cxx_QPsol_opts, Int32(QPsol_param["printLevel"]))
					end
					if "terminationTolerance" in QPopt_keys
						set_terminationTolerance(cxx_QPsol_opts, Float64(QPsol_param["terminationTolerance"]))
					end
				elseif get_QPsol(cxx_opts) == "gurobi"
					cxx_QPsol_opts = gurobi_options()
					if "Method" in QPopt_keys
						set_Method(cxx_QPsol_opts, Int32(QPsol_param["Method"]))
					end
					if "NumericFocus" in QPopt_keys
						set_NumericFocus(cxx_QPsol_opts, Int32(QPsol_param["NumericFocus"]))
					end
					if "OutputFlag" in QPopt_keys
						set_OutputFlag(cxx_QPsol_opts, Int32(QPsol_param["OutputFlag"]))
					end
					if "Presolve" in QPopt_keys
						set_Presolve(cxx_QPsol_opts, Int32(QPsol_param["Presolve"]))
					end
					if "Aggregate" in QPopt_keys
						set_Aggregate(cxx_QPsol_opts, Int32(QPsol_param["Aggregate"]))
					end
					if "BarHomogeneous" in QPopt_keys
						set_BarHomogeneous(cxx_QPsol_opts, Int32(QPsol_param["BarHomogeneous"]))
					end
					if "OptimalityTol" in QPopt_keys
						set_OptimalityTol(cxx_QPsol_opts, Float64(QPsol_param["OptimalityTol"]))
					end
					if ("FeasibilityTol" in QPopt_keys)
						set_FeasibilityTol(cxx_QPsol_opts, Float64(QPsol_param["FeasibilityTol"]))
					end
					if ("PSDTol" in QPopt_keys)
						set_PSDTol(cxx_QPsol_opts, Float64(QPsol_param["PSDTol"]))
					end
				end
				set_QPsol_opts(cxx_opts, CxxPtr(cxx_QPsol_opts))
				#There is no need to throw errors for unknown QP solver specifications here as an error would already have been thrown when setting this option on the C++ side
			end
			if "autoScaling" in opt_keys
				set_autoScaling(cxx_opts, Bool(param["autoScaling"]))
			end
			if "max_extra_steps" in opt_keys
				set_max_extra_steps(cxx_opts, Int32(param["max_extra_steps"]))
			end
			if "max_local_lenience" in opt_keys
				set_max_local_lenience(cxx_opts, Int32(param["max_local_lenience"]))
			end

			###################################################################################################################

			#Create method class on the C++ side
			cxx_method = SQPmethod(CxxPtr(cxx_prob), CxxPtr(cxx_opts), CxxPtr(cxx_stats))

			new(cxx_method, cxx_prob, cxx_opts, cxx_QPsol_opts, cxx_stats, J_prob, param)
		end
	end
	
	#=
	function setproperty!(B_prob::BlockSQP_Problem, name::Symbol, B_index::Array{Int32, 1})
		if name == :blockIdx
			B_prob.n_hessblocks = Int32(length(B_index) - 1)
			invoke(setproperty!, Tuple{BlockSQP_Problem, Symbol, Any}, B_prob, :_blockIdx, B_index)
		else
			invoke(setproperty!, Tuple{BlockSQP_Problem, Symbol, Any}, B_prob, name, B_index)
		end
	end
	=#
	
	function make_sparse(B_prob::BlockSQP_Problem, nnz::Int32, jac_nz::Function, jac_row::Array{Int32, 1}, jac_col::Array{Int32, 1})
		B_prob.jac_g_nz = jac_nz
		B_prob.jac_g_row = jac_row
		B_prob.jac_g_col = jac_col
		B_prob.nnz = nnz
	end
	

	function init!(meth::Solver)
		cpp_init(meth.BSQP_solver)
	end
  	
	function run!(meth::Solver, maxIt::Integer, warmStart::Integer)
		return cpp_run(meth.BSQP_solver, Int32(maxIt), Int32(warmStart))
	end

	function finish!(meth::Solver)
		cpp_finish(meth.BSQP_solver)
	end

	function get_primal_solution(meth::Solver)
		# xi_ptr = get_primal(meth.BSQP_solver)
		return unsafe_wrap(Array{Float64, 1}, get_primal(meth.BSQP_solver).cpp_object, meth.Jul_Problem.nVar, own = true)
		#return copy(xi_arr)
	end

	function get_dual_solution(meth::Solver)
		return unsafe_wrap(Array{Float64, 1}, get_primal(meth.BSQP_solver).cpp_object, meth.Jul_Problem.nVar + meth.Jul_Problem.nCon, own = true)
	end


	function lower_to_full!(arr1::Array{Float64, 1}, arr2::Array{Float64, 1}, n::Int32)
		for i = 1:n
			for j = 0:(i-1)
				arr2[i + j*n] = arr1[i + j*n - Int64(j*(j+1)//2)]
			end
		end

		for i = 1:n
			for j = i:(n-1)
				arr2[i+j*n] = arr1[(j+1) + (i-1)*n - Int64(i*(i-1)//2)]
			end
		end
	end

	function full_to_lower!(arr1::Array{Float64, 1}, arr2::Array{Float64, 1}, n::Int32)
		for i = 1:n
			for j = 0:(i-1)
				arr2[i + j*n - Int64(j*(j+1)//2)] = arr1[i + j*n]
			end
		end
	end


	mutable struct Condenser
		cxx_Condenser::Cpp_Condenser
		cxx_vblocks::vblock_array
		cxx_cblocks::cblock_array
		cxx_hsizes::int_array
		cxx_targets::condensing_targets
		Condenser(VBLOCKS::Array{vblock, 1}, CBLOCKS::Array{cblock, 1}, HSIZES::Array{Int32, 1}, TARGETS::Array{condensing_target, 1}, DEP_BOUNDS::Int32 = Int32(2)) = begin
			vblock_arr = vblock_array(Int32(length(VBLOCKS)))
			for i = 1:length(VBLOCKS)
				array_set(vblock_arr, i, VBLOCKS[i])
			end
			
			cblock_arr = cblock_array(Int32(length(CBLOCKS)))
			for i = 1:length(CBLOCKS)
				array_set(cblock_arr, i, CBLOCKS[i])
			end

			hsize_arr = int_array(Int32(length(HSIZES)))
			for i = 1:length(HSIZES)
				array_set(hsize_arr, i, HSIZES[i])
			end

			target_arr = condensing_targets(Int32(length(TARGETS)))
			for i = 1:length(TARGETS)
				array_set(target_arr, i, TARGETS[i])
			end
			
			cond = construct_Condenser(CxxPtr(vblock_arr), CxxRef(cblock_arr), CxxRef(hsize_arr), CxxRef(target_arr), DEP_BOUNDS)
			print_debug(cond)
			new(cond, vblock_arr, cblock_arr, hsize_arr, target_arr)
		end

	end

	function condensed_num_hessblocks(cond::Condenser)
		return get_condensed_num_hessblocks(cond.cxx_Condenser)
	end
	
	mutable struct sparse_Matrix
		m::Int32
		n::Int32
		nz::Array{Float64, 1}
		row::Array{Int32, 1}
		colind::Array{Int32, 1}
	end
	sparse_Matrix() = sparse_Matrix(Int32(0), Int32(0), Float64[], Int32[], Int32[])

	
	function full_condense(J_cond::Condenser, grad_obj::Array{Float64, 1}, constr_jac::sparse_Matrix, hess::Array{Array{Float64, 2}, 1}, lb_var::Array{Float64, 1}, ub_var::Array{Float64, 1}, lb_con::Array{Float64, 1}, ub_con::Array{Float64, 1})
		#condensed_h::Array{Float64, 1}, condensed_jacobian::sparse_Matrix, condensed_hess::Array{Array{Float64, 2}, 1}, condensed_lb_var::Array{Float64, 1}, condensed_ub_var::Array{Float64, 1}, condensed_lb_con::Array{Float64, 1}, condensed_ub_con::Array{Float64, 1})
		
		cond = J_cond.cxx_Condenser
		nVar = get_num_vars(cond)
		nCon = get_num_cons(cond)
		nnz = length(constr_jac.nz)
		num_hessblocks = get_num_hessblocks(cond)

		M_grad_obj = BSQP_Matrix(nVar, Int32(1))
		grad_obj_data = unsafe_wrap(Array{Float64, 1}, show_ptr(M_grad_obj).cpp_object, nVar, own = false)
		grad_obj_data[:] = grad_obj

		M_constr_jac = alloc_Sparse_Matrix(nCon, nVar, nnz)
		constr_nz_data = unsafe_wrap(Array{Float64, 1}, show_nz(M_constr_jac).cpp_object, nnz, own = false)
		constr_row_data = unsafe_wrap(Array{Int32, 1}, show_row(M_constr_jac).cpp_object, nnz, own = false)
		constr_colind_data = unsafe_wrap(Array{Int32, 1}, show_colind(M_constr_jac).cpp_object, nVar+1, own = false)
		constr_nz_data[:] = constr_jac.nz
		constr_row_data[:] = constr_jac.row
		constr_colind_data[:] = constr_jac.colind


		hsize_ptr = get_hess_block_sizes(cond)
		hess_block_sizes = unsafe_wrap(Array{Int32, 1}, hsize_ptr.cpp_object, num_hessblocks, own = false)

		M_hess = SymMat_array(num_hessblocks)
		for i = 1:num_hessblocks
			hsize = hess_block_sizes[i]
			M_hblock = array_get_ptr(M_hess, i)
			set_size!(M_hblock, hsize)
			
			hblock_data = unsafe_wrap(Array{Float64, 1}, show_ptr(M_hblock).cpp_object, Int64(hsize*(hsize + 1)//2), own = false)
			full_to_lower!(reshape(hess[i], Int64(hsize^2)), hblock_data, hsize)
		end

		M_lb_var = BSQP_Matrix(nVar, Int32(1))
		lb_var_data = unsafe_wrap(Array{Float64, 1}, show_ptr(M_lb_var).cpp_object, nVar, own = false)
		lb_var_data[:] = lb_var
		
		M_ub_var = BSQP_Matrix(nVar, Int32(1))
		ub_var_data = unsafe_wrap(Array{Float64, 1}, show_ptr(M_ub_var).cpp_object, nVar, own = false)
		ub_var_data[:] = ub_var

		M_lb_con = BSQP_Matrix(nCon, Int32(1))
		lb_con_data = unsafe_wrap(Array{Float64, 1}, show_ptr(M_lb_con).cpp_object, nCon, own = false)
		lb_con_data[:] = lb_con

		M_ub_con = BSQP_Matrix(nCon, Int32(1))
		ub_con_data = unsafe_wrap(Array{Float64, 1}, show_ptr(M_ub_con).cpp_object, nCon, own = false)
		ub_con_data[:] = ub_con


		#Return arguments, condensed QP
		condensed_num_hessblocks = get_condensed_num_hessblocks(cond)

		M_condensed_h = BSQP_Matrix()
		M_condensed_jacobian = Sparse_Matrix()
		M_condensed_hess = SymMat_array(condensed_num_hessblocks)
		M_condensed_lb_var = BSQP_Matrix()
		M_condensed_ub_var = BSQP_Matrix()
		M_condensed_lb_con = BSQP_Matrix()
		M_condensed_ub_con = BSQP_Matrix()

		Cxx_full_condense!(cond, M_grad_obj, M_constr_jac, M_hess, M_lb_var, M_ub_var, M_lb_con, M_ub_con,
			M_condensed_h, M_condensed_jacobian, M_condensed_hess, M_condensed_lb_var, M_condensed_ub_var, M_condensed_lb_con, M_condensed_ub_con)

		condensed_num_vars = get_condensed_num_vars(cond)
		condensed_num_cons = get_condensed_num_cons(cond)
		condensed_nnz = get_nnz(M_condensed_jacobian)

		condensed_h = unsafe_wrap(Array{Float64, 1}, release!(M_condensed_h).cpp_object, condensed_num_vars, own = true)

		c_nz_ptr = show_nz(M_condensed_jacobian)
		c_row_ptr = show_row(M_condensed_jacobian)
		c_colind_ptr = show_colind(M_condensed_jacobian)
		disown!(M_condensed_jacobian)

		condensed_jacobian = sparse_Matrix()
		condensed_jacobian.m = condensed_num_cons
		condensed_jacobian.n = condensed_num_vars
		condensed_jacobian.nz = unsafe_wrap(Array{Float64, 1}, c_nz_ptr.cpp_object, condensed_nnz, own = true)
		condensed_jacobian.row = unsafe_wrap(Array{Int32, 1}, c_row_ptr.cpp_object, condensed_nnz, own = true)
		condensed_jacobian.colind = unsafe_wrap(Array{Int32, 1}, c_colind_ptr.cpp_object, condensed_num_vars + 1, own = true)

		condensed_hess = Array{Array{Float64, 2}, 1}(undef, condensed_num_hessblocks)

		for i = 1:condensed_num_hessblocks
			chblock = array_get_ptr(M_condensed_hess, i)
			b_size = size_1(chblock)
			chblock_data = unsafe_wrap(Array{Float64, 1}, show_ptr(chblock).cpp_object, Int64(b_size*(b_size+1)//2), own = false)
			condensed_hess[i] = Array{Float64, 2}(undef, b_size, b_size)
			lower_to_full!(chblock_data, reshape(condensed_hess[i], Int64(b_size)^2), b_size)
		end

		condensed_lb_var = unsafe_wrap(Array{Float64, 1}, release!(M_condensed_lb_var).cpp_object, condensed_num_vars, own = true)
		condensed_ub_var = unsafe_wrap(Array{Float64, 1}, release!(M_condensed_ub_var).cpp_object, condensed_num_vars, own = true)
		condensed_lb_con = unsafe_wrap(Array{Float64, 1}, release!(M_condensed_lb_con).cpp_object, condensed_num_cons, own = true)
		condensed_ub_con = unsafe_wrap(Array{Float64, 1}, release!(M_condensed_ub_con).cpp_object, condensed_num_cons, own = true)
		
		return condensed_h, condensed_jacobian, condensed_hess, condensed_lb_var, condensed_ub_var, condensed_lb_con, condensed_ub_con

	end

	function recover_var_mult(J_cond::Condenser, xi_cond::Array{Float64, 1}, lambda_cond::Array{Float64, 1})
		cond = J_cond.cxx_Condenser

		nVar = get_num_vars(cond)
		nCon = get_num_cons(cond)
		condensed_num_vars = get_condensed_num_vars(cond)
		condensed_num_cons = get_condensed_num_cons(cond)

		M_xi_cond = BSQP_Matrix(condensed_num_vars, Int32(1))
		M_lambda_cond = BSQP_Matrix(condensed_num_vars + condensed_num_cons, Int32(1))
		
		xi_cond_data = unsafe_wrap(Array{Float64, 1}, show_ptr(M_xi_cond).cpp_object, condensed_num_vars, own = false)
		xi_cond_data[:] = xi_cond

		lambda_cond_data = unsafe_wrap(Array{Float64, 1}, show_ptr(M_lambda_cond).cpp_object, condensed_num_vars + condensed_num_cons, own = false)
		lambda_cond_data[:] = lambda_cond

		M_xi_rest = BSQP_Matrix()
		M_lambda_rest = BSQP_Matrix()
		Cxx_recover_var_mult!(cond, M_xi_cond, M_lambda_cond, M_xi_rest, M_lambda_rest)

		xi_rest = unsafe_wrap(Array{Float64, 1}, release!(M_xi_rest).cpp_object, nVar, own = true)
		lambda_rest = unsafe_wrap(Array{Float64, 1}, release!(M_lambda_rest).cpp_object, nVar + nCon, own = true)

		return xi_rest, lambda_rest
	end
	
	
	struct condensing_Solver
		BSQP_solver::SCQPmethod
		Cxx_Problem::Problemform
		BSQP_options::SQPoptions
		BSQP_stats::SQPstats
		Jul_Problem::BlockSQP_Problem
		Options::Dict
		Jul_Condenser::Condenser
	end

	condensing_Solver(J_prob::BlockSQP_Problem, param::Dict, cxx_stats::SQPstats, J_cond::Condenser, dep_bound_handling::String = "opt_bounds") = begin
		#Create problem class on the C++ side
		cxx_prob = Problemform(J_prob.nVar, J_prob.nCon)
		set_scope(cxx_prob, pointer_from_objref(J_prob))
		set_dense_init(cxx_prob, @safe_cfunction(initialize_dense, Nothing, (Ptr{Nothing}, CxxPtr{Float64}, CxxPtr{Float64}, CxxPtr{Float64})))
		set_dense_eval(cxx_prob, @safe_cfunction(evaluate_dense, Nothing, (Ptr{Nothing}, ConstCxxPtr{Float64}, ConstCxxPtr{Float64}, CxxPtr{Float64}, CxxPtr{Float64}, CxxPtr{Float64}, CxxPtr{Float64}, CxxPtr{CxxPtr{Float64}}, Int32, CxxPtr{Int32})))
		set_simple_eval(cxx_prob, @safe_cfunction(evaluate_simple, Nothing, (Ptr{Nothing}, ConstCxxPtr{Float64}, CxxPtr{Float64}, CxxPtr{Float64}, CxxPtr{Int32})))
		set_sparse_init(cxx_prob, @safe_cfunction(initialize_sparse, Nothing, (Ptr{Nothing}, CxxPtr{Float64}, CxxPtr{Float64}, CxxPtr{Float64}, CxxPtr{Int32}, CxxPtr{Int32})))
		set_sparse_eval(cxx_prob, @safe_cfunction(evaluate_sparse, Nothing, (Ptr{Nothing}, ConstCxxPtr{Float64}, ConstCxxPtr{Float64}, CxxPtr{Float64}, CxxPtr{Float64}, CxxPtr{Float64}, CxxPtr{Float64}, CxxPtr{Int32}, CxxPtr{Int32}, CxxPtr{CxxPtr{Float64}}, Int32, CxxPtr{Int32})))
		set_continuity_restoration(cxx_prob, @safe_cfunction(reduceConstrVio, Nothing, (Ptr{Nothing}, CxxPtr{Float64}, CxxPtr{Int32})))
		set_blockIdx(cxx_prob, J_prob.blockIdx)
		set_nnz(cxx_prob, J_prob.nnz)
		set_bounds(cxx_prob, J_prob.lb_var, J_prob.ub_var, J_prob.lb_con, J_prob.ub_con, J_prob.objLo, J_prob.objUp)

		#Create options class on the C++ side
		cxx_opts = BSQP_options(param)

		#Create method class on the C++ side
		if dep_bound_handling == "step_constraints"
			cxx_method = SCQPmethod(CxxPtr(cxx_prob), CxxPtr(cxx_opts), CxxPtr(cxx_stats), CxxPtr(J_cond.cxx_Condenser))
		elseif dep_bound_handling == "opt_bounds"
			cxx_method = SCQP_bound_method(CxxPtr(cxx_prob), CxxPtr(cxx_opts), CxxPtr(cxx_stats), CxxPtr(J_cond.cxx_Condenser))
		elseif dep_bound_handling == "correction"
			cxx_method = SCQP_correction_method(CxxPtr(cxx_prob), CxxPtr(cxx_opts), CxxPtr(cxx_stats), CxxPtr(J_cond.cxx_Condenser))
		end

		condensing_Solver(cxx_method, cxx_prob, cxx_opts, cxx_stats, J_prob, param, J_cond)
	end
	
	
end
