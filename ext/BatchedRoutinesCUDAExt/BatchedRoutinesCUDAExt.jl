module BatchedRoutinesCUDAExt

using BatchedRoutines: AbstractBatchedMatrixFactorization, BatchedRoutines,
                       UniformBlockDiagonalOperator, batchview, nbatches
using CUDA: CUBLAS, CUDA, CUSOLVER, CuArray, CuMatrix, CuPtr, CuVector, DenseCuArray,
            DenseCuMatrix
using ConcreteStructs: @concrete
using LinearAlgebra: BLAS, ColumnNorm, LinearAlgebra, NoPivot, RowMaximum, RowNonZero, mul!

const CuBlasFloat = Union{Float16, Float32, Float64, ComplexF32, ComplexF64}

const CuUniformBlockDiagonalOperator{T} = UniformBlockDiagonalOperator{
    T, <:CUDA.AnyCuArray{T, 3}}

include("low_level.jl")

include("batched_mul.jl")
include("factorization.jl")

end
