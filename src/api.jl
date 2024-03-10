"""
    batched_jacobian(ad, f::F, x) where {F}

Use the backend `ad` to compute the Jacobian of `f` at `x` in batched mode. Returns a
[`UniformBlockDiagonalMatrix`](@ref) as the Jacobian.
"""
function batched_jacobian end

@inline function batched_jacobian(ad, f::F, u::AbstractArray) where {F}
    B = size(u, ndims(u))
    f_mat = @closure x -> reshape(f(reshape(x, size(u))), :, B)
    return batched_jacobian(ad, f_mat, reshape(u, :, B))
end

"""
    batched_pickchunksize(X::AbstractArray, n::Int)
    batched_pickchunksize(N::Int, n::Int)

Pick a chunk size for ForwardDiff ignoring the batch dimension.
"""
function batched_pickchunksize end

"""
    batched_mul(A, B)

TODO: Needs Documentation (take from NNlib.jl)
"""
batched_mul(A, B) = _batched_mul(A, B)

batched_mul!(C, A, B) = _batched_mul!(C, A, B)

"""
    batched_transpose(X::AbstractArray{T, 3}) where {T}

Transpose the first two dimensions of `X`.
"""
batched_transpose(X::BatchedMatrix) = PermutedDimsArray(X, (2, 1, 3))

"""
    batched_adjoint(X::AbstractArray{T, 3}) where {T}

Adjoint the first two dimensions of `X`.
"""
batched_adjoint(X::BatchedMatrix{<:Real}) = batched_transpose(X)
function batched_adjoint(X::BatchedMatrix)
    return mapfoldl(adjoint, (x, y) -> cat(x, y; dims=Val(3)), batchview(X))
end

"""
    nbatches(A::AbstractArray)

Return the number of batches in `A`, which is the size of the last dimension. If `A` is the
vector, then this returns 1.
"""
nbatches(A::AbstractArray) = size(A, ndims(A))
nbatches(A::AbstractVector) = 1

"""
    batchview(A::AbstractArray, idx::Int)
    batchview(A::AbstractArray)

Return a view of the `idx`-th batch of `A`. If `idx` is not supplied an iterator of the
batches is returned.
"""
batchview(A::AbstractArray, idx::Int) = selectdim(A, ndims(A), idx)
function batchview(A::AbstractVector, idx::Int)
    return idx ≥ 2 && throw(BoundsError(batchview(A), idx))
end
batchview(A::AbstractArray) = eachslice(A; dims=ndims(A))
batchview(A::AbstractVector) = (A,)
