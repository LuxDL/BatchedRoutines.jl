# Common Operations
## ---------------------
## Matrix Multiplication
## ---------------------
## For CPUs there is probably some advantage of threading when the batch dimension is large
## but for now we just use a loop
function LinearAlgebra.mul!(C::BatchedVecOrMat, A::BatchedVecOrMat, B::BatchedVecOrMat)
    _batched_mul!(C, A, B)
    return C
end

function LinearAlgebra.mul!(C::BatchedVecOrMat, A::BatchedVecOrMat, B::BatchedVecOrMat,
        α::Number, β::Number)
    _batched_mul!(C, A, B, α, β)
    return C
end

Base.:*(A::BatchedVecOrMat, B::BatchedVecOrMat) = _batched_mul(A, B)

## -------------------
## Transpose / Adjoint
## -------------------
function Base.adjoint(A::BatchedMatrix{T}) where {T}
    T <: Real || error("`adjoint` for Complex valued Batched Matrices not implemented!")
    return BatchedArray{eltype(A), nbatches(A)}(PermutedDimsArray(A.data, (2, 1, 3)))
end

function Base.transpose(A::BatchedMatrix{T}) where {T}
    return BatchedArray{eltype(A), nbatches(A)}(PermutedDimsArray(A.data, (2, 1, 3)))
end

# Factorizations
## -----------------
## QR Implementation
## -----------------
@concrete struct GenericBatchedQR{T}
    qr_vals
    size
end

nbatches(QR::GenericBatchedQR) = length(QR.qr_vals)
batchview(QR::GenericBatchedQR) = QR.qr_vals
batchview(QR::GenericBatchedQR, idx::Int) = QR.qr_vals[idx]
Base.size(QR::GenericBatchedQR) = QR.size
Base.size(QR::GenericBatchedQR, i::Integer) = QR.size[i]

function Base.show(io::IO, QR::GenericBatchedQR)
    return print(io, "GenericBatchedQR() with $(_batch_print(nbatches(QR)))")
end

## Generic Version using a loop over the batch
LinearAlgebra.qr(A::BatchedMatrix, args...; kwargs...) = qr!(copy(A), args...; kwargs...)

function LinearAlgebra.qr!(A::BatchedMatrix, args...; kwargs...)
    return GenericBatchedQR{eltype(A)}(map(batchview(A)) do Aᵢ
            return qr!(Aᵢ, args...; kwargs...)
        end, size(A)[1:(end - 1)])
end

function LinearAlgebra.qr!(A::BatchedVector, args...; kwargs...)
    return qr!(reshape(A, :, 1), args...; kwargs...)
end

function LinearAlgebra.ldiv!(A::GenericBatchedQR, b::BatchedVector)
    @assert nbatches(A) == nbatches(b)
    for i in 1:nbatches(A)
        ldiv!(batchview(A, i), batchview(b, i))
    end
    return b
end

function LinearAlgebra.:\(A::GenericBatchedQR{T1}, b_::BatchedVector{T2}) where {T1, T2}
    b = copy(b_)
    @assert nbatches(A) == nbatches(b)
    X = similar(b, promote_type(T1, T2), size(A, 1))
    for i in 1:nbatches(A)
        ldiv!(batchview(X, i), batchview(A, i), batchview(b, i))
    end
    return X
end

## -----------------
## LU Implementation
## -----------------
@concrete struct GenericBatchedLU{T}
    lu_vals
    size
end

nbatches(LU::GenericBatchedLU) = length(LU.lu_vals)
batchview(LU::GenericBatchedLU) = LU.lu_vals
batchview(LU::GenericBatchedLU, idx::Int) = LU.lu_vals[idx]
Base.size(LU::GenericBatchedLU) = LU.size
Base.size(LU::GenericBatchedLU, i::Integer) = LU.size[i]

function Base.show(io::IO, LU::GenericBatchedLU)
    return print(io, "GenericBatchedLU() with $(_batch_print(nbatches(LU)))")
end

## Generic Version using a loop over the batch
LinearAlgebra.lu(A::BatchedMatrix, args...; kwargs...) = lu!(copy(A), args...; kwargs...)

function LinearAlgebra.lu!(A::BatchedMatrix, args...; kwargs...)
    return GenericBatchedLU{eltype(A)}(map(batchview(A)) do Aᵢ
            return lu!(Aᵢ, args...; kwargs...)
        end, size(A)[1:(end - 1)])
end

function LinearAlgebra.ldiv!(A::GenericBatchedLU, b::BatchedVector)
    @assert nbatches(A) == nbatches(b)
    for i in 1:nbatches(A)
        ldiv!(batchview(A, i), batchview(b, i))
    end
    return b
end

function LinearAlgebra.:\(A::GenericBatchedLU{T1}, b_::BatchedVector{T2}) where {T1, T2}
    b = copy(b_)
    @assert nbatches(A) == nbatches(b)
    X = similar(b, promote_type(T1, T2), size(A, 1))
    for i in 1:nbatches(A)
        ldiv!(batchview(X, i), batchview(A, i), batchview(b, i))
    end
    return X
end

## --------
## Direct \
## --------
function LinearAlgebra.:\(A::BatchedMatrix{T1}, b::BatchedVector{T2}) where {T1, T2}
    @assert nbatches(A) == nbatches(b)
    X = similar(b, promote_type(T1, T2), size(A, 1))
    for i in 1:nbatches(A)
        batchview(X, i) .= batchview(A, i) \ batchview(b, i)
    end
    return X
end

function LinearAlgebra.ldiv!(X::BatchedVector, A::BatchedMatrix, b::BatchedVector)
    @assert nbatches(A) == nbatches(b) == nbatches(X)
    for i in 1:nbatches(A)
        batchview(X, i) .= batchview(A, i) \ batchview(b, i)
    end
    return X
end
