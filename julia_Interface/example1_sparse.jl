include("./blockSQP.jl")
using .blockSQP


###Example problem taken from blockSQP paper (Janka 2016) (sparse version)###

#min  x1^2 - 0.5x2^2
#       s.t. 0 <= x1 - x2 <= 0
#           -Inf < x1, x2 < Inf

#Initial point: x1 = 10, x2 = 10, lambda = [0., 0., 0.]

#Create problem with 2 variables and 1 constraint
prob = blockSQP.BlockSQP_Problem(Int32(2), Int32(1))

#Set objective, constraints and their first derivatives
prob.f = x::Array{Float64, 1} -> x[1]^2 - 0.5*x[2]^2
prob.g = x::Array{Float64, 1} -> Float64[x[1] - x[2]]
prob.grad_f = x::Array{Float64, 1} -> Float64[2*x[1], -x[2]]

#Sparse constraint jacobian
prob.nnz = 2
prob.jac_g_row = Int32[0, 0]
prob.jac_g_colind = Int32[0, 1, 2]
prob.jac_g_nz = x::Array{Float64, 1} -> Float64[1, -1]

#Set bounds
prob.lb_var = Float64[-Inf, -Inf]
prob.ub_var = Float64[Inf, Inf]
prob.lb_con = Float64[0.0]
prob.ub_con = Float64[0.0]

#Set start-end indices of hessian blocks (blockIdx[0] = 0, blockIdx[-1] = nVar)
prob.blockIdx = Int32[0, 1, 2]

#Set initial values
prob.x_start = Float64[10.0, 10.0]
prob.lam_start = Float64[0., 0., 0.]

#Set options
opts = Dict()
opts["opttol"] = 1.0e-12
opts["nlinfeastol"] = 1.0e-12
opts["globalization"] = 0
opts["hessUpdate"] = 1
opts["fallbackUpdate"] = 2
opts["hessScaling"] = 0
opts["fallbackScaling"] = 0
opts["hessLimMem"] = 1
opts["hessMemsize"] = 20
opts["maxConsecSkippedUpdates"] = 200
opts["blockHess"] = 1
opts["whichSecondDerv"] = 0
opts["sparseQP"] = 2
opts["printLevel"] = 2
opts["debugLevel"] = 0
opts["QPsol"] = "qpOASES"

stats = blockSQP.SQPstats("./")



meth = blockSQP.Solver(prob, opts, stats)

blockSQP.init!(meth)
ret = blockSQP.run!(meth, Int32(100), Int32(1))
blockSQP.finish!(meth)

x_opt = blockSQP.get_primal_solution(meth)
lam_opt = blockSQP.get_dual_solution(meth)

print("Primal solution\n", x_opt, "\nDual solution\n", lam_opt, "\n")




