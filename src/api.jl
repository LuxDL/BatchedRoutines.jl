"""
    batched_jacobian(ad, f::F, x) where {F}
    batched_jacobian(ad, f::F, x, p) where {F}

Use the backend `ad` to compute the Jacobian of `f` at `x` in batched mode. Returns a
[`UniformBlockDiagonalMatrix`](@ref) as the Jacobian.

!!! warning

    If the batches interact among themselves, then the Jacobian is not block diagonal and
    this function will not work as expected.
"""
function batched_jacobian end

@inline function batched_jacobian(ad, f::F, u::AbstractArray) where {F}
    B = size(u, ndims(u))
    f_mat = @closure x -> reshape(f(reshape(x, size(u))), :, B)
    return batched_jacobian(ad, f_mat, reshape(u, :, B))
end

@inline batched_jacobian(ad, f::F, u, p) where {F} = batched_jacobian(
    ad, Base.Fix2(f, p), u)

"""
    batched_gradient(ad, f::F, x) where {F}
    batched_gradient(ad, f::F, x, p) where {F}

Use the backend `ad` to compute the batched_gradient of `f` at `x`. For the forward pass this is no
different from calling the batched_gradient function in the backend. This exists to efficiently swap
backends for computing the `batched_gradient` of the `batched_gradient`.
"""
function batched_gradient end

function batched_gradient(ad, f::F, u::AbstractVector) where {F}
    return vec(batched_gradient(ad, f, reshape(u, 1, :)))
end

function batched_gradient(ad, f::F, u::AbstractArray) where {F}
    B = size(u, ndims(u))
    f_mat = @closure x -> reshape(f(reshape(x, size(u))), :, B)
    return reshape(batched_gradient(ad, f_mat, reshape(u, :, B)), size(u))
end

@inline batched_gradient(ad, f::F, u, p) where {F} = batched_gradient(ad, Base.Fix2(f, p), u)

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
batched_adjoint(X::BatchedMatrix) = mapfoldl(adjoint, _cat3, batchview(X))

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

"""
    batched_pinv(A::AbstractArray{T, 3}) where {T}
    batched_pinv(A::UniformBlockDiagonalMatrix)

Compute the pseudo-inverse of `A` in batched mode.
"""
@inline batched_pinv(x::AbstractArray{T, 3}) where {T} = _batched_map(pinv, x)

"""
    batched_inv(A::AbstractArray{T, 3}) where {T}
    batched_inv(A::UniformBlockDiagonalMatrix)

Compute the inverse of `A` in batched mode.
"""
@inline batched_inv(x::AbstractArray{T, 3}) where {T} = _batched_map(inv, x)

"""
    batched_vec(x::AbstractArray)

Reshape `x` into a matrix with the batch dimension as the last dimension.
"""
@inline batched_vec(x::AbstractArray) = batched_reshape(x, :)

"""
    batched_reshape(x::AbstractArray, dims...)

Reshape `x` into an array with the batch dimension as the last dimension.
"""
@inline batched_reshape(x::AbstractArray, dims...) = reshape(x, dims..., nbatches(x))
