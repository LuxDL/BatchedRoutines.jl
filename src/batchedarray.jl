"""
    BatchedArray{T, batchsize}(data)
    BatchedArray{T}(data)
    BatchedArray(data)

Construct a BatchedArray from an AbstractArray. The batchsize is inferred from the last
dimension of the array. If the provided array is 1D, then the batchsize is set to the length
of the array.

A `N` dimensional `BatchedArray` stores an `N + 1` dimensional abstract array internally.
This means that if your code was designed to work for an `N` dimensional array, a
`BatchedArray` pretends to be a `N` dimensional array.

::: note

This Array Type expects that the Batched Version of the underlying matrix can be stored in
a contiguous fashion, which is not true for several structed matrices. We plan to support
on a non-contiguous Batched Array in future releases!

:::

::: warning

If `batchsize` isn't specified, construction of `BatchedArray` is type unstable.

:::
"""
struct BatchedArray{T <: Number, N, D <: AbstractArray{T}, B} <: AbstractArray{T, N}
    data::D
end

function BatchedArray{T, B}(data::AbstractArray{T, N}) where {T, B, N}
    @assert N > 0
    N == 1 && (data = reshape(data, :, 1))
    return BatchedArray{T, N - 1, typeof(data), B}(data)
end
BatchedArray{T, B}(data::AbstractArray) where {T, B} = BatchedArray{T, B}(T.(data))
function BatchedArray{T}(data::AbstractArray{T, N}) where {T, N}
    return BatchedArray{T, size(data, N)}(data)
end
BatchedArray{T}(data::AbstractArray) where {T} = BatchedArray{T}(T.(data))
BatchedArray(data::AbstractArray) = BatchedArray{eltype(data)}(data)

const BatchedVector = BatchedArray{T, 1} where {T}
const BatchedMatrix = BatchedArray{T, 2} where {T}
const BatchedVecOrMat{T} = Union{BatchedVector{T}, BatchedMatrix{T}} where {T}

nbatches(::Type{<:BatchedArray{T, N, D, B}}) where {T, N, D, B} = B
nbatches(::BatchedArray{T, N, D, B}) where {T, N, D, B} = B
nbatches(A::AbstractArray) = size(A, ndims(A))
# NOTE: Don't use eachslice here, because it always returns a view whereas for contiguous
#       types we want to return an array and not a subarray
function batchview(B::BatchedArray)
    return map(i -> view(B.data, ntuple(_ -> Colon(), ndims(B))..., i), 1:nbatches(B))
end
batchview(B::BatchedArray, idx::Int) = view(B.data, ntuple(_ -> Colon(), ndims(B))..., idx)
function batchview(B::AbstractArray)
    return map(i -> view(B, ntuple(_ -> Colon(), ndims(B) - 1)..., i), 1:nbatches(B))
end
batchview(B::AbstractArray, idx::Int) = view(B, ntuple(_ -> Colon(), ndims(B) - 1)..., idx)
batchview(B::AbstractVector{<:AbstractArray}) = B
batchview(B::AbstractVector{<:AbstractArray}, idx::Int) = B[idx]

Base.size(B::BatchedArray) = size(B.data)[1:(end - 1)]
Base.size(B::BatchedArray, i::Integer) = size(B.data, i)
Base.eltype(::BatchedArray{T}) where {T} = T
Base.ndims(::BatchedArray{T, N}) where {T, N} = N

Base.strides(B::BatchedArray) = strides(B.data)[1:(end - 1)]
function Base.unsafe_convert(::Type{Ptr{T}}, B::BatchedArray) where {T}
    return Base.unsafe_convert(Ptr{T}, B.data)
end

function Base.getindex(B::BatchedArray, args...)
    length(args) == ndims(B) + 1 && return getindex(B.data, args...)
    return BatchedArray{eltype(B), nbatches(B)}(reshape(getindex(B.data, args..., :), 1, :))
end
# Happens when indexing with `diagind`
function Base.getindex(B::BatchedArray, idx::StepRange)
    final_idxs = Vector{Int}(undef, length(idx) * nbatches(B))
    for i in 1:nbatches(B)
        final_idxs[((i - 1) * length(idx) + 1):(i * length(idx))] = idx .+
                                                                    (i - 1) * length(B)
    end
    return B.data[final_idxs]
end

function Base.setindex!(B::BatchedArray, v, args...)
    length(args) == ndims(B) + 1 && (setindex!(B.data, v, args...); return B)
    for Bᵢ in batchview(B)
        setindex!(Bᵢ, v, args...)
    end
    return B
end

ArrayInterface.can_setindex(B::BatchedArray) = ArrayInterface.can_setindex(B.data)

function Base.fill!(B::BatchedArray, args...)
    return BatchedArray{eltype(B), nbatches(B)}(fill!(B.data, args...))
end

Base.copy(B::BatchedArray) = BatchedArray{eltype(B), nbatches(B)}(copy(B.data))
function Base.copyto!(B::BatchedArray, A::BatchedArray)
    copyto!(B.data, A.data)
    return B
end

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

function Base.convert(::Type{BatchedArray}, A::AbstractArray)
    return BatchedArray{eltype(A), nbatches(A)}(A)
end

function Base.view(B::BatchedArray, args...)
    return BatchedArray{eltype(B), nbatches(B)}(view(B.data, args..., 1:nbatches(B)))
end
# TODO: Generalize beyond step-range to linear indexing
function Base.view(B::BatchedArray, idx::StepRange)
    final_idxs = Vector{Int}(undef, length(idx) * nbatches(B))
    for i in 1:nbatches(B)
        final_idxs[((i - 1) * length(idx) + 1):(i * length(idx))] = idx .+
                                                                    (i - 1) * length(B)
    end
    return view(B.data, final_idxs)
end

function Base.clamp(A::BatchedVector, lo, hi)
    return BatchedArray{eltype(A), nbatches(A)}(clamp.(A.data, lo, hi))
end

function Base.map!(f::F, dest::BatchedArray, src::BatchedArray) where {F}
    for (destᵢ, srcᵢ) in zip(batchview(dest), batchview(src))
        map!(f, destᵢ, srcᵢ)
    end
end

Base.iszero(B::BatchedArray) = iszero(B.data)

# ---------
# Reduction
# ---------
# Forward Reduction Operations to the internal array?
function Base.mapreduce(f::F, op::OP, B::BatchedArray; dims=Colon(),
        kwargs...) where {F, OP}
    dims_internal = dims === Colon() ? ntuple(identity, ndims(B)) : dims
    y = mapreduce(f, op, B.data; dims=dims_internal, kwargs...)
    if dims === Colon()
        return BatchedScalar{nbatches(B)}(dropdims(y; dims=dims_internal))
    else
        return BatchedArray{eltype(y), nbatches(B)}(y)
    end
end

# -------------
# Concatenation
# -------------
for op in (:vcat, :hcat, :hvcat)
    @eval begin
        function Base.$(op)(Bs::BatchedArray...)
            Ns = nbatches.(Bs)
            @assert all(==(first(Ns)), Ns)
            T = promote_type(eltype.(Bs)...)
            return BatchedArray{T, first(Ns)}(mapreduce(Base.Fix2(getproperty, :data),
                $(op), Bs))
        end
    end
end

# ---------------
# Pretty Printing
# ---------------
_batch_print(N) = ifelse(N == 1, "1 batch", "$(N) batches")
function _batch_array_print(T, N)
    return ifelse(N == 1, "BatchedVector{$T}",
        ifelse(N == 2, "BatchedMatrix{$T}", "BatchedArray{$T, $N}"))
end
function _batched_summary(io, B::BatchedArray{T, N}, inds) where {T, N}
    print(io, Base.dims2string(length.(inds)), " $(_batch_array_print(T, N)) with ")
    if VERSION ≥ v"1.10-"
        return Base.printstyled(io, _batch_print(nbatches(B)); italic=true, underline=true)
    else
        return Base.printstyled(io, _batch_print(nbatches(B)); underline=true)
    end
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

# TODO: Make this a bit better, but currently atleast lets `@show` work
function Base.show_vector(io::IO, v::BatchedVector, opn='[', cls=']')
    prefix, implicit = Base.typeinfo_prefix(io, v)
    print(io, prefix)
    # directly or indirectly, the context now knows about eltype(v)
    if !implicit
        io = IOContext(io, :typeinfo => eltype(v))
    end
    limited = get(io, :limit, false)::Bool

    if limited && length(v) > 20
        axs1 = axes1(v)
        f, l = first(axs1), last(axs1)
        Base.show_delim_array(io, v.data, opn, ",", "", false, f, f + 9)
        print(io, "  …  ")
        Base.show_delim_array(io, v.data, "", ",", cls, false, l - 9, l)
    else
        Base.show_delim_array(io, v.data, opn, ",", cls, false)
    end
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
    N = __extract_nbatches(bc.args)
    X = bc.f.(__unwrap_barray(bc.args)...)
    return BatchedArray{eltype(X), _unwrap_val(N)}(X)
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

@inline __extract_nbatches(args::Tuple) = Val(maximum(map(__extract_nbatches, args)))
@inline __extract_nbatches(x::BatchedArray) = nbatches(x)
@inline __extract_nbatches(x::AbstractArray) = 0
@inline __extract_nbatches(x) = 0
