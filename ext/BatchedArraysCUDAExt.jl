module BatchedArraysCUDAExt

using BatchedArrays, CUDA, LinearAlgebra
import BatchedArrays: _batch_print, nbatches, batchview, __batched_gemm!
import ConcreteStructs: @concrete

const CuBatchedArray = BatchedArray{T, N, <:CUDA.AnyCuArray{T}} where {T, N}
const CuBatchedVector = CuBatchedArray{T, 1} where {T}
const CuBatchedMatrix = CuBatchedArray{T, 2} where {T}
const CuBatchedVecOrMat{T} = Union{CuBatchedVector{T}, CuBatchedMatrix{T}} where {T}

const CuBlasFloat = Union{Float16, Float32, Float64, ComplexF32, ComplexF64}

# ---------------------
# Matrix Multiplication
# ---------------------
function __batched_gemm!(::Type{<:CuArray{<:CuBlasFloat}}, transA::Char, transB::Char,
        α::Number, A, org_A, B, org_B, β::Number, C)
    CUBLAS.gemm_strided_batched!(transA, transB, α, A.data, B.data, β, C.data)
    return C
end

# ----------
# Batched QR
# ----------
@concrete struct CuBatchedQR{T}
    factors
    τ
    size
end

nbatches(QR::CuBatchedQR) = length(QR.factors)
batchview(QR::CuBatchedQR) = zip(QR.factors, QR.τ)
batchview(QR::CuBatchedQR, idx::Int) = (QR.factors[idx], QR.τ[idx])
Base.size(QR::CuBatchedQR) = QR.size
Base.size(QR::CuBatchedQR, i::Integer) = QR.size[i]

function Base.show(io::IO, QR::CuBatchedQR)
    return print(io, "CuBatchedQR() with $(_batch_print(nbatches(QR)))")
end

LinearAlgebra.qr(A::CuBatchedMatrix, args...; kwargs...) = qr!(copy(A), args...; kwargs...)

function LinearAlgebra.qr!(A::CuBatchedMatrix, ::LinearAlgebra.ColumnNorm, args...;
        kwargs...)
    throw("CUBLAS batched QR does not support column pivoting.")
end

function LinearAlgebra.qr!(A::CuBatchedMatrix, args...; kwargs...)
    τ, factors = CUBLAS.geqrf_batched!(batchview(A))
    return CuBatchedQR{eltype(A)}(factors, τ, size(A)[1:(end - 1)])
end

# FIXME (medium-priority): Unfortunately there is no direct batched solver in CUSOLVER
function LinearAlgebra.ldiv!(A::CuBatchedQR{T1}, b::CuBatchedVector{T2}) where {T1, T2}
    @assert nbatches(A) == nbatches(b)
    for i in 1:nbatches(A)
        Fᵢ, τᵢ = batchview(A, i)
        ldiv!(QR(Fᵢ, τᵢ), batchview(b, i))
    end
    return b
end

function LinearAlgebra.:\(A::CuBatchedQR{T1}, b_::CuBatchedVector{T2}) where {T1, T2}
    b = copy(b_)
    @assert nbatches(A) == nbatches(b)
    X = similar(b, promote_type(T1, T2), size(A, 1))
    for i in 1:nbatches(A)
        Fᵢ, τᵢ = batchview(A, i)
        ldiv!(batchview(X, i), QR(Fᵢ, τᵢ), batchview(b, i))
    end
    return X
end

# ----------
# Batched LU
# ----------
@concrete struct CuBatchedLU{T}
    factors
    pivot_array
    info
    size
end

nbatches(LU::CuBatchedLU) = nbatches(LU.factors)
batchview(LU::CuBatchedLU) = zip(batchview(LU.factors), batchview(LU.pivot_array), LU.info)
function batchview(LU::CuBatchedLU, idx::Int)
    return batchview(LU.factors, idx), batchview(LU.pivot_array, idx), LU.info[idx]
end
Base.size(LU::CuBatchedLU) = LU.size
Base.size(LU::CuBatchedLU, i::Integer) = LU.size[i]

function Base.show(io::IO, LU::CuBatchedLU)
    return print(io, "CuBatchedLU() with $(_batch_print(nbatches(LU)))")
end

LinearAlgebra.lu(A::CuBatchedMatrix, args...; kwargs...) = lu!(copy(A), args...; kwargs...)

function LinearAlgebra.lu!(A::CuBatchedMatrix, args...; check::Bool=true, kwargs...)
    pivot = length(args) == 0 ? RowMaximum() : first(pivot)
    pivot_array, info_, factors = CUBLAS.getrf_strided_batched!(A.data, !(pivot isa NoPivot))
    info = Array(info_)
    check && LinearAlgebra.checknonsingular.(info)
    return CuBatchedLU{eltype(A)}(factors, pivot_array, info, size(A)[1:(end - 1)])
end

# FIXME (medium-priority): Unfortunately there is no direct batched solver in CUSOLVER
function LinearAlgebra.ldiv!(A::CuBatchedLU{T1}, b::CuBatchedVector{T2}) where {T1, T2}
    @assert nbatches(A) == nbatches(b)
    for i in 1:nbatches(A)
        Fᵢ, pᵢ, info = batchview(A, i)
        ldiv!(LU(Fᵢ, pᵢ, info), batchview(b, i))
    end
    return b
end

function LinearAlgebra.:\(A::CuBatchedLU{T1}, b_::CuBatchedVector{T2}) where {T1, T2}
    b = copy(b_)
    @assert nbatches(A) == nbatches(b)
    X = similar(b, promote_type(T1, T2), size(A, 1))
    for i in 1:nbatches(A)
        Fᵢ, pᵢ, info = batchview(A, i)
        ldiv!(batchview(X, i), LU(Fᵢ, pᵢ, Int(info)), batchview(b, i))
    end
    return X
end

## --------
## Direct \
## --------

# See https://github.com/JuliaGPU/CUDA.jl/blob/dcd0970aea794acb81d8097485710b25986eac4f/lib/cusolver/linalg.jl#L54

end
