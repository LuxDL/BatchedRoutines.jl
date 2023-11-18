# Note that an N-dimensional BatchedArray is an N-1-dimensional array
# This is done to trick a lot of SciML algorithms to think that a 3D Array is a matrix which
# for the puposes of a BatchedArray is true.
# NOTE: Creation of a BatchedArray will be type unstable. But we need to store the
# batchsize in the type to allow construction via `undef` -- heavily used in Krylov.jl
# This assumes that we can store the batched array in a contiguous fashion. Support for
# non-contiguous structured arrays is planned but not very high priority.
struct BatchedArray{T <: Number, N, D <: AbstractArray{T}, B} <: AbstractArray{T, N}
    data::D
end

function BatchedArray{T, B}(data::AbstractArray{ T, N}) where {T, B, N}
    @assert N > 0
    N == 1 && (data = reshape(data, :, 1))
    return BatchedArray{T, N - 1, typeof(data), B}(data)
end
function BatchedArray{T}(data::AbstractArray{T, N}) where {T, N}
    return BatchedArray{T, size(data, N)}(data)
end
BatchedArray{T}(data::AbstractArray) where {T} = BatchedArray{T}(T.(data))
BatchedArray(data::AbstractArray) = BatchedArray{eltype(data)}(data)

const BatchedVector = BatchedArray{T, 1} where {T}
const BatchedMatrix = BatchedArray{T, 2} where {T}
const BatchedVecOrMat = Union{BatchedVector, BatchedMatrix}

nbatches(::Type{<:BatchedArray{T, N, D, B}}) where {T, N, D, B} = B
nbatches(::BatchedArray{T, N, D, B}) where {T, N, D, B} = B
nbatches(A::AbstractArray) = size(A, ndims(A))
batchview(B::BatchedArray) = eachslice(B.data; dims=ndims(B) + 1)
batchview(B::BatchedArray, idx::Int) = selectdim(B.data, ndims(B) + 1, idx)
batchview(B::AbstractArray) = eachslice(B; dims=ndims(B))
batchview(B::AbstractArray, idx::Int) = selectdim(B, ndims(B), idx)

Base.size(B::BatchedArray) = size(B.data)[1:(end - 1)]
Base.size(B::BatchedArray, i::Integer) = size(B.data, i)
Base.eltype(::BatchedArray{T}) where {T} = T
Base.ndims(::BatchedArray{T, N}) where {T, N} = N

function Base.getindex(B::BatchedArray, args...)
    length(args) == ndims(B) + 1 && return getindex(B.data, args...)
    return BatchedArray{eltype(B), nbatches(B)}(reshape(getindex(B.data, args..., :), 1, :))
end
function Base.setindex!(B::BatchedArray, v, args...)
    length(args) == ndims(B) + 1 && (setindex!(B.data, v, args...); return B)
    for Bᵢ in batchview(B)
        setindex!(Bᵢ, v, args...)
    end
    return B
end

function Base.fill!(B::BatchedArray, args...)
    return BatchedArray{eltype(B), nbatches(B)}(fill!(B.data, args...))
end

Base.copy(B::BatchedArray) = BatchedArray{eltype(B), nbatches(B)}(copy(B.data))

Base.similar(B::BatchedArray) = BatchedArray{eltype(B), nbatches(B)}(similar(B.data))
function Base.similar(B::BatchedArray, ::Type{T}) where {T}
    return BatchedArray{T, nbatches(B)}(similar(B.data, T))
end
function Base.similar(B::BatchedArray, dims::Dims)
    return BatchedArray{eltype(B), nbatches(B)}(similar(B.data, (dims..., nbatches(B))))
end
function Base.similar(B::BatchedArray, ::Type{T}, dims::Dims) where {T}
    return BatchedArray{T, nbatches(B)}(similar(B.data, T, (dims..., nbatches(B))))
end

function BatchedArray{T, N, D, B}(::UndefInitializer, dims...) where {T, N, D, B}
    return BatchedArray{T, B}(D(undef, (dims..., B)))
end

# ---------------
# Pretty Printing
# ---------------
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

function Base.show(io::IO, m::MIME"text/plain", B::BatchedArray)
    _batched_summary(io, B, axes(B))
    print(io, " with data ")
    show(io, m, B.data)
    return
end

# ---------
# Reshaping
# ---------
# Reshapes are performed without specifying the number of batches
function Base.reshape(B::BatchedArray, dims::Tuple{Vararg{Union{Colon, Int}}})
    dims_ = Base._reshape_uncolon(B.data, (dims..., nbatches(B)))
    return reshape(B, dims_[1:(end - 1)])
end

function Base.reshape(B::BatchedArray, dims::Dims)
    return BatchedArray{eltype(B), nbatches(B)}(reshape(B.data, (dims..., nbatches(B))))
end

Base.vec(B::BatchedArray) = reshape(B, :)
Base.vec(B::BatchedVector) = B

function Base.permutedims(B::BatchedArray, perm)
    L, N = length(perm), ndims(B)
    if length(perm) == N
        return BatchedArray{eltype(B), nbatches(B)}(permutedims(B.data, (perm..., N + 1)))
    elseif length(perm) == ndims(B) + 1
        @assert last(perm)==L "For BatchedArrays, the last dimension must be the batch \
                               dimension!"
        return BatchedArray{eltype(B), nbatches(B)}(permutedims(B.data, perm))
    else
        error("Cannot permute a $(N) dimensional BatchedArray to $(L) dimensions")
    end
end

# -------------------
# Adapt Compatibility
# -------------------
function Adapt.adapt_structure(to, B::BatchedArray)
    data = Adapt.adapt(to, B.data)
    return BatchedArray{eltype(data), nbatches(B)}(data)
end

# ------------
# Broadcasting
# ------------
## Implementation taken from https://github.com/SciML/RecursiveArrayTools.jl/blob/master/src/vector_of_array.jl
struct BatchedArrayStyle{N} <: Broadcast.AbstractArrayStyle{N} end
BatchedArrayStyle(::Val{N}) where {N} = BatchedArrayStyle{N}()

Broadcast.BroadcastStyle(a::BatchedArrayStyle, ::Base.Broadcast.DefaultArrayStyle{0}) = a
function Broadcast.BroadcastStyle(::BatchedArrayStyle{N},
        a::Broadcast.DefaultArrayStyle{M}) where {M, N}
    return BatchedArrayStyle(Val(max(M, N)))
end
function Broadcast.BroadcastStyle(::BatchedArrayStyle{N},
        a::Broadcast.AbstractArrayStyle{M}) where {M, N}
    return BatchedArrayStyle(Val(max(M, N)))
end
function Broadcast.BroadcastStyle(::BatchedArrayStyle{M},
        ::BatchedArrayStyle{N}) where {M, N}
    return BatchedArrayStyle(Val(max(M, N)))
end
function Base.BroadcastStyle(::Type{<:BatchedArray{T, N, A}}) where {T, N, A}
    return BatchedArrayStyle{N + 1}()
end

@inline function Base.copy(bc::Broadcast.Broadcasted{<:BatchedArrayStyle})
    bc = Broadcast.flatten(bc)
    T = __extract_eltype(bc.args)
    N = __extract_nbatches(bc.args)
    return BatchedArray{T, _unwrap_val(N)}(bc.f.(__unwrap_barray(bc.args)...))
end

@inline function Base.copyto!(dest::BatchedArray,
        bc::Broadcast.Broadcasted{<:BatchedArrayStyle})
    bc = Broadcast.flatten(bc)
    dest.data .= bc.f.(__unwrap_barray(bc.args)...)
    return dest
end

@inline _unwrap_val(x) = x
@inline _unwrap_val(::Val{X}) where {X} = X

@inline __unwrap_barray(args::Tuple) = map(__unwrap_barray, args)
@inline __unwrap_barray(x::BatchedArray) = x.data
@inline __unwrap_barray(x) = x

@inline __extract_eltype(args::Tuple) = promote_type(map(__extract_eltype, args)...)
@inline __extract_eltype(x) = eltype(x)

@inline __extract_nbatches(args::Tuple) = Val(maximum(map(__extract_nbatches, args)))
@inline __extract_nbatches(x::BatchedArray) = nbatches(x)
@inline __extract_nbatches(x::AbstractArray) = 0
@inline __extract_nbatches(x) = 0
