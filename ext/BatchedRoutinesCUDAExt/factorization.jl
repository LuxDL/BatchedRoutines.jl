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
        function LinearAlgebra.lu!(A::CuUniformBlockDiagonalOperator, pivot::$pT; kwargs...)
            return LinearAlgebra.lu!(A, !(pivot isa NoPivot); kwargs...)
        end
    end
end

function LinearAlgebra.lu!(
        A::CuUniformBlockDiagonalOperator, pivot::Bool=true; check::Bool=true, kwargs...)
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

function LinearAlgebra.qr!(A::CuUniformBlockDiagonalOperator; kwargs...)
    return LinearAlgebra.qr!(A, NoPivot(); kwargs...)
end

function LinearAlgebra.qr!(::CuUniformBlockDiagonalOperator, ::ColumnNorm; kwargs...)
    throw(ArgumentError("ColumnNorm is not supported for batched CUDA QR factorization!"))
end

function LinearAlgebra.qr!(A::CuUniformBlockDiagonalOperator, ::NoPivot; kwargs...)
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

# Direct Ldiv
function BatchedRoutines.__internal_backslash(
        op::CuUniformBlockDiagonalOperator{T1}, b::AbstractMatrix{T2}) where {T1, T2}
    T = promote_type(T1, T2)
    return __internal_backslash(T != T1 ? T.(op) : op, T != T2 ? T.(b) : b)
end

function BatchedRoutines.__internal_backslash(
        op::CuUniformBlockDiagonalOperator{T}, b::AbstractMatrix{T}) where {T}
    size(op, 1) != length(b) && throw(DimensionMismatch("size(op, 1) != length(b)"))
    x = similar(b, T, size(BatchedRoutines.getdata(op), 2), nbatches(op))
    m, n = size(op)
    if n < m       # Underdetermined: LQ or QR with ColumnNorm
        error("Underdetermined systems are not supported yet! Please open an issue if you \
               care about this feature.")
    elseif n == m  # Square: LU with Pivoting
        p, _, F = CUBLAS.getrf_strided_batched!(copy(BatchedRoutines.getdata(op)), true)
        copyto!(x, b)
        getrs_strided_batched!('N', F, p, x)
    else           # Overdetermined: QR
        CUBLAS.gels_batched!('N', batchview(copy(BatchedRoutines.getdata(op))),
            [reshape(bᵢ, :, 1) for bᵢ in batchview(b)])
        copyto!(x, view(b, 1:n, :))
    end
    return x
end
