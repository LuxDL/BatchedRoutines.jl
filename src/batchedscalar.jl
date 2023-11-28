"""
    BatchedScalar{T, B, D <: AbstractVector{T}} <: Number

Batched Scalar behaves like a scalar but contains a vector of values. Users should not
construct this directly in most cases. However, reduction operations on `BatchedArray`s
will return a `BatchedScalar` with the correct number of batches.
"""
struct BatchedScalar{T, B, D <: AbstractVector{T}} <: Number
    data::D
end

nbatches(::BatchedScalar{T, B}) where {T, B} = B
batchview(x::BatchedScalar{T, B}) where {T, B} = x.data
batchview(x::BatchedScalar{T, B}, i::Integer) where {T, B} = x.data[i]
nbatches(::Number) = 1
batchview(x::Number) = x
batchview(x::Number, ::Integer) = x

Base.eltype(::BatchedScalar{T}) where {T} = T
Base.length(x::BatchedScalar) = nbatches(x)

function Base.show(io::IO, x::BatchedScalar)
    print(io, "BatchedScalar(", x.data, ") with $(_batch_print(nbatches(x)))")
end
function Base.show(io::IO, m::MIME"text/plain", B::BatchedScalar)
    print(io, "BatchedScalar with $(_batch_print(nbatches(B)))")
    print(io, " storing ")
    show(io, m, B.data)
    return
end

function BatchedScalar(data::AbstractVector)
    return BatchedScalar{eltype(data), length(data), typeof(data)}(data)
end
function BatchedScalar{B}(data::AbstractVector) where {B}
    @assert length(data) == B
    return BatchedScalar{eltype(data), B, typeof(data)}(data)
end

function Base.fill(v::BatchedScalar, dims::Vararg{<:Union{<:Integer, <:AbstractUnitRange}})
    data = repeat(reshape(v.data, ntuple(_ -> 1, length(dims))..., :); inner=(dims..., 1))
    return BatchedArray{eltype(data), nbatches(v)}(data)
end

function Base.setindex!(B::BatchedArray, v::BatchedScalar, args...)
    Bᵢ = view(B.data, args..., 1:nbatches(B))
    copyto!(Bᵢ, v.data)
    return B
end

## -----------------
## Common Operations
## -----------------
for op in (:+, :-, :*, :/, :^, :isless), T1 in (:BatchedScalar, :Number),
    T2 in (:BatchedScalar, :Number)

    T1 == :Number && T2 == :Number && continue

    @eval function Base.$(op)(x::$T1, y::$T2)
        res = $(op).(batchview(x), batchview(y))
        return BatchedScalar{max(nbatches(x), nbatches(y))}(res)
    end
end

# Ambiguities
function Base.:^(x::BatchedScalar, p::Integer)
    return BatchedScalar{nbatches(x)}(x.data .^ p)
end

for op in (:<, :>, :(==), :≥, :≤),  T1 in (:BatchedScalar, :Number),
    T2 in (:BatchedScalar, :Number)

    T1 == :Number && T2 == :Number && continue

    # TODO: Nonallocating version if possible
    @eval function Base.$(op)(x::$T1, y::$T2)
        return all($(op).(batchview(x), batchview(y)))
    end
end

function Base.ifelse(c::BatchedScalar, x::BatchedScalar, y::BatchedScalar)
    res = ifelse.(batchview(c), batchview(x), batchview(y))
    return BatchedScalar{max(nbatches(c), nbatches(x), nbatches(y))}(res)
end

for op in (:any, :all)
    @eval begin
        Base.$(op)(f::F, x::BatchedScalar) where {F} = $(op)(f, batchview(x))
        Base.$(op)(x::BatchedScalar) = $(op)(batchview(x))
    end
end

for op in (:sqrt, :sign, :abs, :abs2)
    @eval Base.$(op)(x::BatchedScalar) = BatchedScalar{nbatches(x)}($(op).(batchview(x)))
end

## ------------
## Broadcasting
## ------------
@inline __unwrap_barray(x::BatchedScalar) = reshape(x.data, 1, :)

@inline __extract_nbatches(x::BatchedScalar) = nbatches(x)
