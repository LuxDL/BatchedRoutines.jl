struct BatchedArray{T <: Number, N, D <: AbstractArray{T, N}} <: AbstractArray{T, N}
    data::D

    function BatchedArray{T}(data::AbstractArray{T}) where {T}
        ndims(data) == 1 && (data = reshape(data, :, 1))
        return new{T, ndims(data), typeof(data)}(data)
    end

    BatchedArray{T}(data::AbstractArray) where {T} = BatchedArray{T}(T.(data))

    function BatchedArray(data::AbstractArray)
        ndims(data) == 1 && (data = reshape(data, :, 1))
        return new{eltype(data), ndims(data), typeof(data)}(data)
    end
end

const BatchedVector = BatchedArray{T, 2} where {T}
const BatchedMatrix = BatchedArray{T, 3} where {T}
const BatchedVecOrMat = Union{BatchedVector, BatchedMatrix}

Base.size(b::BatchedArray) = size(b.data)
Base.size(b::BatchedArray, i::Integer) = size(b.data, i)
Base.eltype(::BatchedArray{T}) where {T} = T
Base.ndims(::BatchedArray{T, N}) where {T, N} = N

Base.getindex(b::BatchedArray, args...) = getindex(b.data, args...)
Base.setindex!(b::BatchedArray, args...) = setindex!(b.data, args...)

nbatches(B::BatchedArray) = size(B, ndims(B))
batchview(B::BatchedArray) = eachslice(B.data; dims=ndims(B))
batchview(B::BatchedArray, idx::Int) = selectdim(B.data, ndims(B), idx)
batchview(B::AbstractArray) = eachslice(B; dims=ndims(B))
batchview(B::AbstractArray, idx::Int) = selectdim(B, ndims(B), idx)

# Pretty Printing
_batch_print(N) = ifelse(N == 1, "1 batch", "$(N) batches")
function _batched_summary(io, B::BatchedArray{T, N}, inds) where {T, N}
    print(io, Base.dims2string(length.(inds)), " BatchedArray{$T, $N} with ")
    return Base.printstyled(io, _batch_print(nbatches(B)); italic=true, underline=true)
end
function Base.array_summary(io::IO, B::BatchedArray, inds::Tuple{Vararg{Base.OneTo}})
    _batched_summary(io, B, inds)
    print(io, " with data ")
    return summary(io, B.data)
end
function Base.array_summary(io::IO, B::BatchedArray, inds)
    _batched_summary(io, B, inds)
    print(io, " with data ")
    summary(io, B.data)
    return print(io, " with indices ", Base.inds2string(inds))
end

# Reshaping
function Base.reshape(B::BatchedArray, dims::Tuple{Vararg{Union{Colon, Int}}})
    dims_ = Base._reshape_uncolon(B.data, (dims..., nbatches(B)))
    return reshape(B, dims_[1:(end - 1)])
end

function Base.reshape(B::BatchedArray, dims::Dims)
    return BatchedArray(reshape(B.data, (dims..., nbatches(B))))
end

Base.vec(B::BatchedArray) = reshape(B, :)
Base.vec(B::BatchedVector) = B

Base.similar(B::BatchedArray) = BatchedArray(similar(B.data))
Base.similar(B::BatchedArray, ::Type{T}) where {T} = BatchedArray(similar(B.data, T))
function Base.similar(B::BatchedArray, dims::Dims)
    return BatchedArray(similar(B.data, (dims..., nbatches(B))))
end
function Base.similar(B::BatchedArray, ::Type{T}, dims::Dims) where {T}
    return BatchedArray(similar(B.data, T, (dims..., nbatches(B))))
end

# Broadcasting
## Implementation taken from https://github.com/SciML/RecursiveArrayTools.jl/blob/master/src/vector_of_array.jl
struct BatchedArrayStyle{N} <: Broadcast.AbstractArrayStyle{N} end
BatchedArrayStyle(::Val{N}) where {N} = BatchedArrayStyle{N}()

Broadcast.BroadcastStyle(a::BatchedArrayStyle, ::Base.Broadcast.DefaultArrayStyle{0}) = a
function Broadcast.BroadcastStyle(::BatchedArrayStyle{N},
        a::Broadcast.DefaultArrayStyle{M}) where {M, N}
    return Broadcast.DefaultArrayStyle(Val(max(M, N)))
end
function Broadcast.BroadcastStyle(::BatchedArrayStyle{N},
        a::Broadcast.AbstractArrayStyle{M}) where {M, N}
    return typeof(a)(Val(max(M, N)))
end
function Broadcast.BroadcastStyle(::BatchedArrayStyle{M},
        ::BatchedArrayStyle{N}) where {M, N}
    return BatchedArrayStyle(Val(max(M, N)))
end
function Base.BroadcastStyle(::Type{<:BatchedArray{T, N, A}}) where {T, N, A}
    return BatchedArrayStyle{N}()
end

@inline function Base.copy(bc::Broadcast.Broadcasted{<:BatchedArrayStyle})
    bc = Broadcast.flatten(bc)
    return BatchedArray(bc.f.(__unwrap_barray(bc.args)...))
end

@inline function Base.copyto!(dest::BatchedArray,
        bc::Broadcast.Broadcasted{<:BatchedArrayStyle})
    bc = Broadcast.flatten(bc)
    dest.data .= bc.f.(__unwrap_barray(bc.args)...)
    return dest
end

__unwrap_barray(args::Tuple) = map(__unwrap_barray, args)
__unwrap_barray(x::BatchedArray) = x.data
__unwrap_barray(x) = x
