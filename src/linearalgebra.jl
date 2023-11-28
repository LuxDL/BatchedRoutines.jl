# Common Operations
## ---------------------
## Matrix Multiplication
## ---------------------
## For CPUs there is probably some advantage of threading when the batch dimension is large
## but for now we just use a loop
for aType in (:BatchedVector, :BatchedMatrix)
    @eval begin
        function LinearAlgebra.mul!(C::$(aType), A::BatchedVecOrMat, B::$(aType))
            _batched_mul!(C, A, B)
            return C
        end

        function LinearAlgebra.mul!(C::$(aType), A::BatchedVecOrMat, B::$(aType),
                α::Number, β::Number)
            _batched_mul!(C, A, B, α, β)
            return C
        end
    end
end

function LinearAlgebra.mul!(C::BatchedVector, A::AbstractMatrix, B::BatchedVector)
    mul!(C.data, A, B.data)
    return C
end

function LinearAlgebra.mul!(C::BatchedVector, A::AbstractMatrix, B::BatchedVector,
        α::Number, β::Number)
    mul!(C.data, A, B.data, α, β)
    return C
end

Base.:*(A::BatchedMatrix, B::BatchedMatrix) = _batched_mul(A, B)
Base.:*(A::BatchedMatrix, B::BatchedVector) = _batched_mul(A, B)

function Base.:*(A::AbstractMatrix, B::BatchedVector)
    X = A * B.data
    return BatchedArray{eltype(X), nbatches(B)}(X)
end

# TODO: Non-allocating version for some array types
function LinearAlgebra.dot(A::BatchedVector, B::BatchedVector)
    res = sum(A.data .* B.data; dims=1)
    return BatchedArray{promote_type(eltype(A), eltype(B)), nbatches(A)}(res)
end

## ------------------------------------
## BatchedVector a.k.a Batch of Scalars
## ------------------------------------

for op in (:/, :*)
    @eval Base.$(op)(A::BatchedVector, B::BatchedVector) = broadcast($op, A, B)
end

## -------------------
## Transpose / Adjoint
## -------------------
function Base.adjoint(A::BatchedMatrix{T}) where {T}
    T <: Real || error("`adjoint` for Complex valued Batched Matrices not implemented!")
    return BatchedArray{T, nbatches(A)}(PermutedDimsArray(A.data, (2, 1, 3)))
end
Base.adjoint(A::BatchedVector) = adjoint(reshape(A, :, 1))

function Base.transpose(A::BatchedMatrix{T}) where {T}
    return BatchedArray{T, nbatches(A)}(PermutedDimsArray(A.data, (2, 1, 3)))
end
Base.transpose(A::BatchedVector) = transpose(reshape(A, :, 1))

# --------------
# Factorizations
# --------------
abstract type AbstractBatchedMatrixFactorization end

struct GenericBatchedFactorization{A, F} <: AbstractBatchedMatrixFactorization
    alg::A
    fact::Vector{F}

    function GenericBatchedFactorization(alg::A, fact::Vector{F}) where {A, F}
        return new{A, F}(alg, fact)
    end
end

nbatches(F::GenericBatchedFactorization) = length(F.fact)
batchview(F::GenericBatchedFactorization) = F.fact
batchview(F::GenericBatchedFactorization, idx::Int) = F.fact[idx]
Base.size(F::GenericBatchedFactorization) = size(first(F.fact))
Base.size(F::GenericBatchedFactorization, i::Integer) = size(first(F.fact), i)
Base.eltype(F::GenericBatchedFactorization) = eltype(first(F.fact))

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, F::GenericBatchedFactorization)
    println(io, "GenericBatchedFactorization() with $(_batch_print(nbatches(F)))")
    Base.printstyled(io, "Factorization Function: "; color=:green)
    show(io, mime, F.alg)
    Base.printstyled(io, "\nPrototype Factorization: "; color=:green)
    show(io, mime, first(F.fact))
    return nothing
end

const PIVOT_TYPES = Dict(:qr => (:NoPivot, :ColumnNorm),
    :lu => (:NoPivot, :RowMaximum, :RowNonZero),
    :cholesky => (:NoPivot, :RowMaximum))

for fact in (:qr, :lu, :cholesky)
    fact! = Symbol("$(fact)!")
    @eval begin
        function LinearAlgebra.$(fact)(A::BatchedMatrix, args...; kwargs...)
            return $(fact!)(copy(A), args...; kwargs...)
        end
    end

    for pType in PIVOT_TYPES[fact]
        @eval begin
            function LinearAlgebra.$(fact!)(A::BatchedMatrix, pivot::$pType; kwargs...)
                fact = map(Aᵢ -> $(fact!)(Aᵢ, pivot; kwargs...), batchview(A))
                return GenericBatchedFactorization($(fact!), fact)
            end
        end
    end
end

function LinearAlgebra.ldiv!(A::GenericBatchedFactorization, b::BatchedVector)
    @assert nbatches(A) == nbatches(b)
    for i in 1:nbatches(A)
        ldiv!(batchview(A, i), batchview(b, i))
    end
    return b
end

function LinearAlgebra.ldiv!(X::BatchedVector, A::GenericBatchedFactorization,
        b::BatchedVector)
    @assert nbatches(A) == nbatches(b) == nbatches(X)
    for i in 1:nbatches(A)
        ldiv!(batchview(X, i), batchview(A, i), batchview(b, i))
    end
    return X
end

function LinearAlgebra.:\(A::GenericBatchedFactorization, b::BatchedVector)
    X = similar(b, promote_type(eltype(A), eltype(b)), size(A, 2))
    ldiv!(X, A, copy(b))
    return X
end

## Extra Methods for certain factorizations

function LinearAlgebra.qr!(A::BatchedVector, args...; kwargs...)
    return qr!(reshape(A, :, 1), args...; kwargs...)
end

## --------
## Direct \
## --------
function LinearAlgebra.:\(A::BatchedMatrix, b::BatchedVector)
    @assert nbatches(A) == nbatches(b)
    X = similar(b, promote_type(eltype(A), eltype(b)), size(A, 2))
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
