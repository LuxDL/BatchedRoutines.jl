module BatchedRoutinesCUDALinearSolveExt

using BatchedRoutines: UniformBlockDiagonalOperator, getdata
using CUDA: CUDA
using LinearAlgebra: LinearAlgebra
using LinearSolve: LinearSolve

const CuUniformBlockDiagonalOperator{T} = UniformBlockDiagonalOperator{
    T, <:CUDA.AnyCuArray{T, 3}}

function LinearSolve.init_cacheval(
        alg::LinearSolve.SVDFactorization, A::CuUniformBlockDiagonalOperator, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool, ::LinearSolve.OperatorAssumptions)
    return nothing
end

function LinearSolve.init_cacheval(
        alg::LinearSolve.QRFactorization, A::CuUniformBlockDiagonalOperator, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool, ::LinearSolve.OperatorAssumptions)
    A_ = UniformBlockDiagonalOperator(similar(getdata(A), 0, 0, 1))
    return LinearAlgebra.qr!(A_) # ignore the pivot since CUDA doesn't support it
end

end
