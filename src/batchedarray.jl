@concrete struct BatchedArray{T, N} <: AbstractArray{T, N}
    data
end

const BatchedVector = BatchedArray{T, 2} where T
const BatchedMatrix = BatchedArray{T, 3} where T
const BatchedVecOrMat = Union{BatchedVector, BatchedMatrix}

Base.size(b::BatchedArray) = size(b.data)
Base.size(b::BatchedArray, i::Integer) = size(b.data, i)
Base.eltype(::BatchedArray{T}) where {T} = T
Base.ndims(::BatchedArray{T, N}) where {T, N} = N

Base.getindex(b::BatchedArray, args...) = getindex(b.data, args...)
Base.setindex!(b::BatchedArray, args...) = setindex!(b.data, args...)

nbatches(B::BatchedArray) = size(B, ndims(B))

function _batched_summary(io, B::BatchedArray{T, N}, inds) where {T, N}
    print(io, Base.dims2string(length.(inds)),
        " BatchedArray{$T, $N} with $(nbatches(B)) batches")
end
function Base.array_summary(io::IO, B::BatchedArray, inds::Tuple{Vararg{Base.OneTo}})
    _batched_summary(io, B, inds)
    print(io, " with data: ")
    summary(io, B.data)
end
function Base.array_summary(io::IO, B::BatchedArray, inds)
    _batched_summary(io, B, inds)
    print(io, " with data: ")
    summary(io, B.data)
    print(io, " with indices ", Base.inds2string(inds))
end

BatchedArray(data::AbstractArray{T, N}) where {T, N} = BatchedArray{T, N}(data)
