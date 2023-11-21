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

function LinearAlgebra.qr!(A::CuBatchedMatrix, ::ColumnNorm, args...; kwargs...)
    throw("CUBLAS batched QR does not support column pivoting.")
end

function LinearAlgebra.qr!(A::CuBatchedMatrix, ::NoPivot; kwargs...)
    τ, factors = CUBLAS.geqrf_batched!(batchview(A))
    return CuBatchedQR{eltype(A)}(factors, τ, size(A)[1:(end - 1)])
end

# FIXME (medium-priority): Unfortunately there is no direct batched solver in CUSOLVER
function LinearAlgebra.ldiv!(A::CuBatchedQR, b::CuBatchedVector)
    @assert nbatches(A) == nbatches(b)
    for i in 1:nbatches(A)
        Fᵢ, τᵢ = batchview(A, i)
        ldiv!(QR(Fᵢ, τᵢ), batchview(b, i))
    end
    return b
end

function LinearAlgebra.ldiv!(X::CuBatchedVector, A::CuBatchedQR, b::CuBatchedVector)
    @assert nbatches(A) == nbatches(b) == nbatches(X)
    copyto!(X.data, b.data)
    return ldiv!(A, X)
end

function LinearAlgebra.:\(A::CuBatchedQR{T1}, b_::CuBatchedVector{T2}) where {T1, T2}
    @assert nbatches(A) == nbatches(b)
    b = copy(b_)
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

for pT in (:RowMaximum, :RowNonZero, :NoPivot)
    @eval begin
        function LinearAlgebra.lu!(A::CuBatchedMatrix, pivot::$pT; kwargs...)
            return lu!(A, !(pivot isa NoPivot); kwargs...)
        end
    end
end

function LinearAlgebra.lu!(A::CuBatchedMatrix, pivot::Bool=true; check::Bool=true,
        kwargs...)
    pivot_array, info_, factors = CUBLAS.getrf_strided_batched!(A.data, pivot)
    info = Array(info_)
    check && LinearAlgebra.checknonsingular.(info)
    return CuBatchedLU{eltype(A)}(factors, pivot_array, info, size(A)[1:(end - 1)])
end

function LinearAlgebra.ldiv!(A::CuBatchedLU, b::CuBatchedVector)
    @assert nbatches(A) == nbatches(b)
    getrs_strided_batched!('N', A.factors, A.pivot_array, b.data)
    return b
end

function LinearAlgebra.ldiv!(X::CuBatchedVector, A::CuBatchedLU, b::CuBatchedVector)
    copyto!(X.data, b.data)
    return ldiv!(A, X)
end

function LinearAlgebra.:\(A::CuBatchedLU, b_::CuBatchedVector)
    @assert nbatches(A) == nbatches(b)
    b = copy(b_)
    getrs_strided_batched!('N', A.factors, A.pivot_array, b.data)
    return b
end

## --------
## Direct \
## --------

# Based on https://github.com/JuliaGPU/CUDA.jl/blob/dcd0970aea794acb81d8097485710b25986eac4f/lib/cusolver/linalg.jl#L54
function LinearAlgebra.:\(A::CuBatchedMatrix{T1}, b::CuBatchedVector{T2}) where {T1, T2}
    T = promote_type(T1, T2)
    return (T != T1 ? T.(A) : A) \ (T != T2 ? T.(b) : b)
end

function LinearAlgebra.ldiv!(bX::CuBatchedVector{T}, bA::CuBatchedMatrix{T},
        bb::CuBatchedVector{T}) where {T <: CuBlasFloat}
    X, A, b = bX.data, copy(bA.data), copy(bb.data)
    n, m = size(A)
    if n < m
        # Underdetermined System: Use LQ
        error("Not yet implemented!")
    elseif n == m
        # LU with Pivoting
        p, _, F = CUBLAS.getrf_strided_batched!(A, true)
        copyto!(X, b)
        getrs_strided_batched!('N', F, p, X)
    else
        # Overdetermined System: Use QR
        error("Not yet implemented!")
    end
    return X
end

function LinearAlgebra.:\(A::CuBatchedMatrix{T},
        b::CuBatchedVector{T}) where {T <: CuBlasFloat}
    X = similar(A, T, size(A, 1))
    ldiv!(X, A, b)
    return X
end

## -------------------------------------
## Low Level Wrappers (to be upstreamed)
## -------------------------------------
for (fname, elty) in ((:cublasDgetrsBatched, :Float64),
    (:cublasSgetrsBatched, :Float32),
    (:cublasZgetrsBatched, :ComplexF64),
    (:cublasCgetrsBatched, :ComplexF32))
    @eval begin
        function getrs_batched!(trans::Char, n, nrhs, Aptrs::CuVector{CuPtr{$elty}}, lda, p,
                Bptrs::CuVector{CuPtr{$elty}}, ldb)
            batchSize = length(Aptrs)
            info = Array{Cint}(undef, batchSize)
            CUBLAS.$fname(CUBLAS.handle(), trans, n, nrhs, Aptrs, lda, p, Bptrs, ldb, info,
                batchSize)
            CUDA.unsafe_free!(Aptrs)
            CUDA.unsafe_free!(Bptrs)

            return info
        end
    end
end

function getrs_strided_batched!(trans::Char, F::DenseCuArray{<:Any, 3}, p::DenseCuMatrix,
        B::Union{DenseCuArray{<:Any, 3}, DenseCuMatrix})
    m, n = size(F, 1), size(F, 2)
    if m != n
        throw(DimensionMismatch("All matrices must be square!"))
    end
    lda = max(1, stride(F, 2))
    ldb = max(1, stride(B, 2))
    nrhs = ifelse(ndims(B) == 2, 1, size(B, 2))

    Fptrs = CUBLAS.unsafe_strided_batch(F)
    Bptrs = CUBLAS.unsafe_strided_batch(B)
    info = getrs_batched!(trans, n, nrhs, Fptrs, lda, p, Bptrs, ldb)

    return B, info
end

end
