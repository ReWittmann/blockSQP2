include("./Jul_BlockSQP.jl")
using .BlockSQP


###Example problem taken from blockSQP paper (Janka 2016) (dense version)###

#min  x1^2 - 0.5x2^2
#       s.t. 0 <= x1 - x2 <= 0
#           -Inf < x1, x2 < Inf

#Initial point: x1 = 10, x2 = 10, lambda = [0., 0., 0.]

#Create problem with 2 variables and 1 constraint
prob = BlockSQP.BlockSQP_Problem(Int32(2), Int32(1))

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

#Set initial values
prob.x_start = Float64[10.0, 10.0]
prob.lam_start = Float64[0., 0., 0.]

#Set options
opts = Dict()
opts["opttol"] = 1.0e-12
opts["nlinfeastol"] = 1.0e-12
opts["globalization"] = 0
opts["hessUpdate"] = 2
opts["fallbackUpdate"] = 2
opts["hessScaling"] = 0
opts["fallbackScaling"] = 0
opts["hessLimMem"] = 1
opts["hessMemsize"] = 20
opts["maxConsecSkippedUpdates"] = 200
opts["blockHess"] = 0
opts["whichSecondDerv"] = 0
opts["sparseQP"] = 0
opts["printLevel"] = 2
opts["debugLevel"] = 0

stats = BlockSQP.SQPstats("./")



meth = BlockSQP.Solver(prob, opts, stats)

BlockSQP.init(meth)
ret = BlockSQP.run(meth, Int32(100), Int32(1))

BlockSQP.finish(meth)

x_opt = BlockSQP.get_primal_solution(meth)
lam_opt = BlockSQP.get_dual_solution(meth)

print("Primal solution\n", x_opt, "\nDual solution\n", lam_opt, "\n")

