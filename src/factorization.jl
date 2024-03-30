abstract type AbstractBatchedMatrixFactorization{T} <: LinearAlgebra.Factorization{T} end

const AdjointAbstractBatchedMatrixFactorization{T} = LinearAlgebra.AdjointFactorization{
    T, <:AbstractBatchedMatrixFactorization{T}}
const TransposeAbstractBatchedMatrixFactorization{T} = LinearAlgebra.TransposeFactorization{
    T, <:AbstractBatchedMatrixFactorization{T}}
const AdjointOrTransposeAbstractBatchedMatrixFactorization{T} = Union{
    AdjointAbstractBatchedMatrixFactorization{T},
    TransposeAbstractBatchedMatrixFactorization{T}}

const AllAbstractBatchedMatrixFactorization{T} = Union{
    AbstractBatchedMatrixFactorization{T},
    AdjointOrTransposeAbstractBatchedMatrixFactorization{T}}

nbatches(f::AdjointOrTransposeAbstractBatchedMatrixFactorization) = nbatches(parent(f))
batchview(f::AdjointOrTransposeAbstractBatchedMatrixFactorization) = batchview(parent(f))
function batchview(f::AdjointOrTransposeAbstractBatchedMatrixFactorization, idx::Int)
    return batchview(parent(f), idx)
end

# First we take inputs and standardize them
function LinearAlgebra.ldiv!(A::AllAbstractBatchedMatrixFactorization, b::AbstractVector)
    LinearAlgebra.ldiv!(A, reshape(b, :, nbatches(A)))
    return b
end

function LinearAlgebra.ldiv!(
        X::AbstractVector, A::AllAbstractBatchedMatrixFactorization, b::AbstractVector)
    LinearAlgebra.ldiv!(reshape(X, :, nbatches(A)), A, reshape(b, :, nbatches(A)))
    return X
end

function Base.:\(A::AllAbstractBatchedMatrixFactorization, b::AbstractVector)
    X = similar(b, promote_type(eltype(A), eltype(b)), size(A, 1))
    LinearAlgebra.ldiv!(X, A, b)
    return X
end

function Base.:\(A::AllAbstractBatchedMatrixFactorization, b::AbstractMatrix)
    X = similar(b, promote_type(eltype(A), eltype(b)), size(A, 1))
    LinearAlgebra.ldiv!(X, A, vec(b))
    return reshape(X, :, nbatches(b))
end

# Now we implement the actual factorizations
## This just loops over the batches and calls the factorization on each, mostly used where
## we don't have native batched factorizations
struct GenericBatchedFactorization{T, A, F} <: AbstractBatchedMatrixFactorization{T}
    alg::A
    fact::Vector{F}

    function GenericBatchedFactorization(alg, fact)
        return GenericBatchedFactorization{eltype(first(fact))}(alg, fact)
    end

    function GenericBatchedFactorization{T}(alg::A, fact::Vector{F}) where {T, A, F}
        return new{T, A, F}(alg, fact)
    end
end

nbatches(F::GenericBatchedFactorization) = length(F.fact)
batchview(F::GenericBatchedFactorization) = F.fact
batchview(F::GenericBatchedFactorization, idx::Int) = F.fact[idx]
Base.size(F::GenericBatchedFactorization) = size(first(F.fact)) .* length(F.fact)
function Base.size(F::GenericBatchedFactorization, i::Integer)
    return size(first(F.fact), i) * length(F.fact)
end

function LinearAlgebra.issuccess(fact::GenericBatchedFactorization)
    return all(LinearAlgebra.issuccess, fact.fact)
end

function Base.adjoint(fact::GenericBatchedFactorization{T}) where {T}
    return GenericBatchedFactorization{T}(fact.alg, adjoint.(fact.fact))
end

function Base.show(io::IO, mime::MIME"text/plain", F::GenericBatchedFactorization)
    println(io, "GenericBatchedFactorization() with Batch Count: $(nbatches(F))")
    Base.printstyled(io, "Factorization Function: "; color=:green)
    show(io, mime, F.alg)
    Base.printstyled(io, "\nPrototype Factorization: "; color=:green)
    show(io, mime, first(F.fact))
end

for fact in (:qr, :lu, :cholesky, :generic_lufact, :svd)
    fact! = Symbol(fact, :!)
    if isdefined(LinearAlgebra, fact)
        @eval function LinearAlgebra.$(fact)(
                op::UniformBlockDiagonalOperator, args...; kwargs...)
            return LinearAlgebra.$(fact!)(copy(op), args...; kwargs...)
        end
    end

    @eval function LinearAlgebra.$(fact!)(
            op::UniformBlockDiagonalOperator, args...; kwargs...)
        fact = map(Aᵢ -> LinearAlgebra.$(fact!)(Aᵢ, args...; kwargs...), batchview(op))
        return GenericBatchedFactorization(LinearAlgebra.$(fact!), fact)
    end
end

function LinearAlgebra.ldiv!(A::GenericBatchedFactorization, b::AbstractMatrix)
    @assert nbatches(A) == nbatches(b)
    for i in 1:nbatches(A)
        LinearAlgebra.ldiv!(batchview(A, i), batchview(b, i))
    end
    return b
end

function LinearAlgebra.ldiv!(
        X::AbstractMatrix, A::GenericBatchedFactorization, b::AbstractMatrix)
    @assert nbatches(A) == nbatches(b) == nbatches(X)
    for i in 1:nbatches(A)
        LinearAlgebra.ldiv!(batchview(X, i), batchview(A, i), batchview(b, i))
    end
    return X
end
