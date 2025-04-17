include("./blockSQP.jl")
using .blockSQP

vblocks = Array{blockSQP.vblock, 1}(undef, 7)
vblocks[1] = blockSQP.vblock(Int32(1), false)

vblocks[2] = blockSQP.vblock(Int32(2), true)
vblocks[3] = blockSQP.vblock(Int32(1), false)

vblocks[4] = blockSQP.vblock(Int32(2), true)
vblocks[5] = blockSQP.vblock(Int32(1), false)

vblocks[6] = blockSQP.vblock(Int32(2), true)
vblocks[7] = blockSQP.vblock(Int32(1), false)

cblocks = Array{blockSQP.cblock, 1}(undef, 4)
cblocks[1] = blockSQP.cblock(Int32(2))
cblocks[2] = blockSQP.cblock(Int32(2))
cblocks[3] = blockSQP.cblock(Int32(2))
cblocks[4] = blockSQP.cblock(Int32(1))

hsizes = Int32[1, 3, 3, 3]

targets = Array{blockSQP.condensing_target, 1}(undef, 1)
targets[1] = blockSQP.condensing_target(Int32(3), Int32(0), Int32(7), Int32(0), Int32(3))

cond = blockSQP.Condenser(vblocks, cblocks, hsizes, targets, Int32(2))

print("Created condenser julia struct\n")


NZ = Float64[-1,-2,1,1,-2,1,1,-1,1,-1,-2,1,1,-2,1,1,-1,1,-1,-2,1,1,1,1,1,1]
ROW = Int32[0,1,6,0,2,6,1,3,6,2,3,6,2,4,6,3,5,6,4,5,6,4,6,5,6,6]
COLIND = Int32[0,3,6,9,12,15,18,21,23,25,26]
con_jac = blockSQP.sparse_Matrix(7, 10, NZ, ROW, COLIND)


full_block = Float64[0.75;-0.25;-0.25;; -0.25; 0.75; 0.25;; -0.25; 0.25; 1.25]

hess = Array{Array{Float64, 2}, 1}(undef, 4)
hess[1] = Float64[1.0;;]
hess[2] = full_block
hess[3] = full_block
hess[4] = full_block

grad_obj = ones(10)

lb_var = Float64[-0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3]
ub_var = Float64[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
lb_con = Float64[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, -1.9]
ub_con = Float64[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.9]


condensed_h, condensed_jacobian, condensed_hess, condensed_lb_var, condensed_ub_var, condensed_lb_con, condensed_ub_con = blockSQP.full_condense(cond, grad_obj, con_jac, hess, lb_var, ub_var, lb_con, ub_con)
#    condensed_h, condensed_jacobian, condensed_hess, condensed_lb_var, condensed_ub_var, condensed_lb_con, condensed_ub_con


conDENSEd_jacobian = zeros(7,4)
for j = 1:4
    for i = condensed_jacobian.colind[j]:(condensed_jacobian.colind[j + 1] - 1)
        conDENSEd_jacobian[condensed_jacobian.row[i + 1] + 1, j] = condensed_jacobian.nz[i + 1]
    end
end


print("condensed_h=\n") 
display(condensed_h) 
print("\ncondensed_jacobian=\n")
display(conDENSEd_jacobian)
print("\ncondensed_hess=\n")
display(condensed_hess[1])
print("\ncondensed_lb_var=\n")
display(condensed_lb_var)
print("\ncondensed_lb_con=\n")
display(condensed_lb_con)
print("\ncondensed_ub_con=\n")
display(condensed_ub_con)
print("\n")

#TODO obtain this solution with a QP solver

xi_cond = Float64[-0.2, -0.05, -0.05, -0.2]
lambda_cond = Float64[0, 0, 0, 0, 0.7375, 0, 0.4625, 0, 0.275, 0, 0.23125]


xi_rest, lambda_rest = blockSQP.recover_var_mult(cond, xi_cond, lambda_cond)

print("xi_rest=\n")
display(xi_rest)
print("\nlambda_rest=\n")
display(lambda_rest)
