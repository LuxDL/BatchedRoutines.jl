module BatchedRoutinesCUDAExt

using BatchedRoutines: AbstractBatchedMatrixFactorization, BatchedRoutines,
                       UniformBlockDiagonalMatrix, batchview, nbatches
using CUDA: CUBLAS, CUDA, CuArray, CuMatrix, CuPtr, CuVector, DenseCuArray, DenseCuMatrix
using ConcreteStructs: @concrete
using LinearAlgebra: BLAS, ColumnNorm, LinearAlgebra, NoPivot, RowMaximum, RowNonZero, mul!

const CuBlasFloat = Union{Float16, Float32, Float64, ComplexF32, ComplexF64}

const CuUniformBlockDiagonalMatrix{T} = UniformBlockDiagonalMatrix{T, <:CuArray{T, 3}}

include("batched_mul.jl")
include("factorization.jl")

end
