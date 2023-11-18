module BatchedArraysCUDAExt

using BatchedArrays, CUDA, LinearAlgebra
import BatchedArrays: _batch_print, nbatches, batchview
import ConcreteStructs: @concrete

const CuBatchedArray = BatchedArray{T, N, <:CUDA.AnyCuArray{T, N}} where {T, N}
const CuBatchedVector = CuBatchedArray{T, 2} where {T}
const CuBatchedMatrix = CuBatchedArray{T, 3} where {T}
const CuBatchedVecOrMat = Union{CuBatchedVector, CuBatchedMatrix}

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
    τ, factors = CUBLAS.geqrf_batched!(collect(batchview(A)))
    return CuBatchedQR{eltype(A)}(factors, τ, size(A)[1:(end - 1)])
end

# FIXME: Unfortunately there is no direct batched solver in CUSOLVER
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

## --------
## Direct \
## --------

# See https://github.com/JuliaGPU/CUDA.jl/blob/dcd0970aea794acb81d8097485710b25986eac4f/lib/cusolver/linalg.jl#L54

end
