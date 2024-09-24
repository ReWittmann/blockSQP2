include("./Jul_BlockSQP.jl")
using .BlockSQP
using Pkg
Pkg.add("DynamicOED")
using DynamicOED
using DynamicOED.ModelingToolkit
using ForwardDiff
using DynamicOED.OrdinaryDiffEq
using DynamicOED.ComponentArrays
using Optimization, OptimizationMOI
using Ipopt
using Plots

@variables t
@variables x(t)=1.0 [description = "State"]
@parameters p[1:1]=-2.0 [description = "Fixed parameter", tunable = true]
@variables obs(t) [description = "Observed", measurement_rate = 10]
D = Differential(t)

@named simple_system = ODESystem([
        D(x) ~ p[1] * x,
    ], tspan = (0.0, 1.0),
    observed = obs .~ [x.^2])

@named oed = OEDSystem(simple_system)
oed = structural_simplify(oed)

# Augment the original problem to an OED problem
oed_problem = OEDProblem(structural_simplify(oed), DCriterion())

# Define an MTK Constraint system over the grid variables
optimization_variables = states(oed_problem)

constraint_equations = [
      sum(optimization_variables.measurements.w₁) ≲ 3,
]

@named constraint_set = ConstraintsSystem(constraint_equations, optimization_variables, Num[])

# Initialize the optimization problem
optimization_problem = OptimizationProblem(oed_problem, AutoForwardDiff(),
      constraints = constraint_set,
      integer_constraints = false)

# Solven for the optimal values of the observed variables
res = solve(optimization_problem, Ipopt.Optimizer())

#### Now do the same with blockSQP

f_blocksqp = (x) -> optimization_problem.f(x, SciMLBase.NullParameters())
∇f_blocksqp = (x) -> ForwardDiff.gradient(f_blocksqp, x)

g_blocksqp = (x) -> [sum(x[1:10])]
jac_const = reshape(vcat(ones(10), 0.0), (1,:))
∇g_blocksqp = (x) -> jac_const

lb = DynamicOED.generate_variable_bounds(oed_problem.system, oed_problem.timegrid, true)[:]
ub = DynamicOED.generate_variable_bounds(oed_problem.system, oed_problem.timegrid, false)[:]
x_init = DynamicOED.get_initial_variables(oed_problem)[:]

n_var = Int32(length(lb))
n_cons = Int32(1)

prob = BlockSQP.BlockSQP_Problem(n_var, n_cons)

#Set objective, constraints and their first derivatives
prob.f = x::Array{Float64, 1} -> f_blocksqp(x)#x[1]^2 - 0.5*x[2]^2
prob.g = x::Array{Float64, 1} -> g_blocksqp(x) #Float64[x[1] - x[2]]
prob.grad_f = x::Array{Float64, 1} -> ∇f_blocksqp(x) #Float64[2*x[1], -x[2]]
prob.jac_g = x::Array{Float64, 1} -> ∇g_blocksqp(x)

#Set bounds
prob.lb_var = lb # Float64[-Inf, -Inf]
prob.ub_var = ub #Float64[Inf, Inf]
prob.lb_con = Float64[-Inf]
prob.ub_con = Float64[3.0] # maximum of 3 measurements


# Set initial values
prob.x_start = x_init
prob.lam_start = zeros(length(x_init) + n_cons) #Float64[0., 0., 0.]


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
opts["blockHess"] = 1
opts["whichSecondDerv"] = 0
opts["sparseQP"] = 0
opts["printLevel"] = 2
opts["debugLevel"] = 0
opts["which_QPsolver"] = "qpOASES"

stats = BlockSQP.SQPstats("./")
cxx_opts = BlockSQP.BSQP_options(opts)
meth = BlockSQP.Solver(prob, opts, stats)
BlockSQP.init(meth)

ret = BlockSQP.run(meth, Int32(500), Int32(1))

BlockSQP.finish(meth)

x_opt = BlockSQP.get_primal_solution(meth)
lam_opt = BlockSQP.get_dual_solution(meth)

print("Primal solution\n", x_opt, "\nDual solution\n", lam_opt, "\n")

rep_first(x) = vcat(x[1], x)
tgrid_ = oed_problem.timegrid.timegrids[1]
tgrid = vcat(first(first(tgrid_)), last.(tgrid_))
plot(tgrid, rep_first(res.u[1:10]), label="Ipopt", title="Sampling", linetype=:steppre, xticks=0:0.1:1.0)
plot!(tgrid, rep_first(x_opt[1:10]), label="blockSQP", linetype=:steppre)