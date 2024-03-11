@concrete struct CuBatchedLU{T} <: AbstractBatchedMatrixFactorization
    factors
    pivot_array
    info
    size
end

BatchedRoutines.nbatches(LU::CuBatchedLU) = nbatches(LU.factors)
function BatchedRoutines.batchview(LU::CuBatchedLU)
    return zip(batchview(LU.factors), batchview(LU.pivot_array), LU.info)
end
function BatchedRoutines.batchview(LU::CuBatchedLU, idx::Int)
    return batchview(LU.factors, idx), batchview(LU.pivot_array, idx), LU.info[idx]
end
Base.size(LU::CuBatchedLU) = LU.size
Base.size(LU::CuBatchedLU, i::Integer) = LU.size[i]
Base.eltype(::CuBatchedLU{T}) where {T} = T

function Base.show(io::IO, LU::CuBatchedLU)
    return print(io, "CuBatchedLU() with Batch Count: $(nbatches(LU))")
end

for pT in (:RowMaximum, :RowNonZero, :NoPivot)
    @eval begin
        function LinearAlgebra.lu!(A::CuUniformBlockDiagonalMatrix, pivot::$pT; kwargs...)
            return LinearAlgebra.lu!(A, !(pivot isa NoPivot); kwargs...)
        end
    end
end

function LinearAlgebra.lu!(
        A::CuUniformBlockDiagonalMatrix, pivot::Bool=true; check::Bool=true, kwargs...)
    pivot_array, info_, factors = CUBLAS.getrf_strided_batched!(A.data, pivot)
    info = Array(info_)
    check && LinearAlgebra.checknonsingular.(info)
    return CuBatchedLU{eltype(A)}(factors, pivot_array, info, size(A)[1:(end - 1)])
end

function LinearAlgebra.ldiv!(A::CuBatchedLU, b::CuMatrix)
    @assert nbatches(A) == nbatches(b)
    getrs_strided_batched!('N', A.factors, A.pivot_array, b)
    return b
end

function LinearAlgebra.ldiv!(X::CuMatrix, A::CuBatchedLU, b::CuMatrix)
    copyto!(X, b)
    return LinearAlgebra.ldiv!(A, X)
end

# Low Level Wrappers
for (fname, elty) in ((:cublasDgetrsBatched, :Float64), (:cublasSgetrsBatched, :Float32),
    (:cublasZgetrsBatched, :ComplexF64), (:cublasCgetrsBatched, :ComplexF32))
    @eval begin
        function getrs_batched!(trans::Char, n, nrhs, Aptrs::CuVector{CuPtr{$elty}},
                lda, p, Bptrs::CuVector{CuPtr{$elty}}, ldb)
            batchSize = length(Aptrs)
            info = Array{Cint}(undef, batchSize)
            CUBLAS.$fname(
                CUBLAS.handle(), trans, n, nrhs, Aptrs, lda, p, Bptrs, ldb, info, batchSize)
            CUDA.unsafe_free!(Aptrs)
            CUDA.unsafe_free!(Bptrs)
            return info
        end
    end
end

function getrs_strided_batched!(trans::Char, F::DenseCuArray{<:Any, 3}, p::DenseCuMatrix,
        B::Union{DenseCuArray{<:Any, 3}, DenseCuMatrix})
    m, n = size(F, 1), size(F, 2)
    m != n && throw(DimensionMismatch("All matrices must be square!"))
    lda = max(1, stride(F, 2))
    ldb = max(1, stride(B, 2))
    nrhs = ifelse(ndims(B) == 2, 1, size(B, 2))

    Fptrs = CUBLAS.unsafe_strided_batch(F)
    Bptrs = CUBLAS.unsafe_strided_batch(B)
    info = getrs_batched!(trans, n, nrhs, Fptrs, lda, p, Bptrs, ldb)

    return B, info
end