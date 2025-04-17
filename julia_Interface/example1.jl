include("./blockSQP.jl")
using .blockSQP


###Example problem taken from blockSQP paper (Janka 2016) (dense version)###

#min  x1^2 - 0.5x2^2
#       s.t. 0 <= x1 - x2 <= 0
#           -Inf < x1, x2 < Inf

#Initial point: x1 = 10, x2 = 10, lambda = [0., 0., 0.]

#Create problem with 2 variables and 1 constraint
prob = blockSQP.jlProblem(Int32(2), Int32(1))

#Set objective, constraints and their first derivatives
prob.f = x::Array{Float64, 1} -> x[1]^2 - 0.5*x[2]^2
prob.g = x::Array{Float64, 1} -> Float64[x[1] - x[2]]
prob.grad_f = x::Array{Float64, 1} -> Float64[2*x[1], -x[2]]
prob.jac_g = x::Array{Float64, 1} -> Float64[1 -1]

#Set bounds
prob.lb_var = Float64[-Inf, -Inf]
prob.ub_var = Float64[Inf, Inf]
prob.lb_con = Float64[0.0]
prob.ub_con = Float64[0.0]

#Set start-end indices of hessian blocks (blockIdx[0] = 0, blockIdx[-1] = nVar)
prob.blockIdx = Int32[0, 1, 2]

prob.vblocks = blockSQP.vblock[blockSQP.vblock(1, true), blockSQP.vblock(1, true)]

#Set initial values
prob.x_start = Float64[10.0, 10.0]
prob.lam_start = Float64[0., 0., 0.]

#Set options
opts = Dict()
opts["optimality_tol"] = 1.0e-12
opts["feasibility_tol"] = 1.0e-12
opts["enable_linesearch"] = false
opts["hess_approximation"] = 2
opts["fallback_approximation"] = 2
opts["sizing_strategy"] = 0
opts["fallback_sizing_strategy"] = 0
opts["limited_memory"] = true
opts["memory_size"] = 20
opts["max_consec_skipped_updates"] = 200
opts["block_hess"] = 1
opts["exact_hess_usage"] = 0
opts["sparse_mode"] = false
opts["print_level"] = 2
opts["debug_level"] = 0
opts["qpsol"] = "qpOASES"
opts["qpsol_options"] = Dict([("printLevel", 1), ("terminationTolerance", 1e-10)])

#opts["QPsol"] = "gurobi"
#opts["QPsol_opts"] = Dict([("OutputFlag", 1), ("NumericFocus", 3)])

stats = blockSQP.SQPstats("./")
#cxx_opts = blockSQP.BSQP_options(opts)

meth = blockSQP.Solver(prob, opts, stats)
blockSQP.init!(meth)
ret = blockSQP.run!(meth, Int32(100), Int32(1))
blockSQP.finish!(meth)

x_opt = blockSQP.get_primal_solution(meth)
lam_opt = blockSQP.get_dual_solution(meth)

print("Primal solution\n", x_opt, "\nDual solution\n", lam_opt, "\n")

