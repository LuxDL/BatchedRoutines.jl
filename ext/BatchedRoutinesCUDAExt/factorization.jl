# LU Factorization
@concrete struct CuBatchedLU{T} <: AbstractBatchedMatrixFactorization{T}
    factors
    pivot_array
    info
    size
end

const AdjCuBatchedLU{T} = LinearAlgebra.AdjointFactorization{T, <:CuBatchedLU{T}}
const TransCuBatchedLU{T} = LinearAlgebra.TransposeFactorization{T, <:CuBatchedLU{T}}
const AdjOrTransCuBatchedLU{T} = Union{AdjCuBatchedLU{T}, TransCuBatchedLU{T}}

const AllCuBatchedLU{T} = Union{CuBatchedLU{T}, AdjOrTransCuBatchedLU{T}}

BatchedRoutines.nbatches(LU::CuBatchedLU) = nbatches(LU.factors)
function BatchedRoutines.batchview(LU::CuBatchedLU)
    return zip(batchview(LU.factors), batchview(LU.pivot_array), LU.info)
end
function BatchedRoutines.batchview(LU::CuBatchedLU, idx::Int)
    return batchview(LU.factors, idx), batchview(LU.pivot_array, idx), LU.info[idx]
end
Base.size(LU::CuBatchedLU) = LU.size
Base.size(LU::CuBatchedLU, i::Integer) = LU.size[i]

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

function LinearAlgebra.ldiv!(A::AdjOrTransCuBatchedLU, b::CuMatrix)
    @assert nbatches(A) == nbatches(b)
    getrs_strided_batched!('T', parent(A).factors, parent(A).pivot_array, b)
    return b
end

function LinearAlgebra.ldiv!(X::CuMatrix, A::AllCuBatchedLU, b::CuMatrix)
    copyto!(X, b)
    return LinearAlgebra.ldiv!(A, X)
end

# QR Factorization
@concrete struct CuBatchedQR{T} <: AbstractBatchedMatrixFactorization{T}
    factors
    τ
    size
end

const AdjCuBatchedQR{T} = LinearAlgebra.AdjointFactorization{T, <:CuBatchedQR{T}}
const TransCuBatchedQR{T} = LinearAlgebra.TransposeFactorization{T, <:CuBatchedQR{T}}
const AdjOrTransCuBatchedQR{T} = Union{AdjCuBatchedQR{T}, TransCuBatchedQR{T}}

const AllCuBatchedQR{T} = Union{CuBatchedQR{T}, AdjOrTransCuBatchedQR{T}}

BatchedRoutines.nbatches(QR::CuBatchedQR) = length(QR.factors)
BatchedRoutines.batchview(QR::CuBatchedQR) = zip(QR.factors, QR.τ)
BatchedRoutines.batchview(QR::CuBatchedQR, idx::Int) = QR.factors[idx], QR.τ[idx]
Base.size(QR::CuBatchedQR) = QR.size
Base.size(QR::CuBatchedQR, i::Integer) = QR.size[i]

function Base.show(io::IO, QR::CuBatchedQR)
    return print(io, "CuBatchedQR() with Batch Count: $(nbatches(QR))")
end

function LinearAlgebra.qr!(::CuUniformBlockDiagonalMatrix, ::ColumnNorm; kwargs...)
    throw(ArgumentError("ColumnNorm is not supported for batched CUDA QR factorization!"))
end

function LinearAlgebra.qr!(A::CuUniformBlockDiagonalMatrix, ::NoPivot; kwargs...)
    τ, factors = CUBLAS.geqrf_batched!(collect(batchview(A)))
    return CuBatchedQR{eltype(A)}(factors, τ, size(A))
end

# TODO: Handle Adjoint and Transpose for QR
function LinearAlgebra.ldiv!(A::CuBatchedQR, b::CuMatrix)
    @assert nbatches(A) == nbatches(b)
    (; τ, factors) = A
    n, m = size(A) .÷ nbatches(A)
    for i in 1:nbatches(A)
        CUSOLVER.ormqr!('L', 'C', batchview(factors, i), batchview(τ, i), batchview(b, i))
    end
    vecX = [reshape(view(bᵢ, 1:m), :, 1) for bᵢ in batchview(b)]
    if n != m
        sqF = [F_[1:m, 1:m] for F_ in batchview(factors)]
    else
        sqF = collect(batchview(factors))
    end
    CUBLAS.trsm_batched!('L', 'U', 'N', 'N', one(eltype(A)), sqF, vecX)
    return b
end

function LinearAlgebra.ldiv!(X::CuMatrix, A::CuBatchedQR, b::CuMatrix)
    @assert size(X, 1) ≤ size(b, 1)
    b_ = LinearAlgebra.ldiv!(A, copy(b))
    copyto!(X, view(b_, 1:size(X, 1), :))
    return X
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
