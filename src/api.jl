"""
    batched_jacobian(ad, f::F, x) where {F}

Use the backend `ad` to compute the Jacobian of `f` at `x` in batched mode. If `x` is a
vector then this behaves like usual jacobian and returns a matrix, else we return a 3D array
where the last dimension is the batch size.
"""
function batched_jacobian end

"""
    batched_pickchunksize(X::AbstractArray, n::Int)
    batched_pickchunksize(N::Int, n::Int)

Pick a chunk size for ForwardDiff ignoring the batch dimension.
"""
function batched_pickchunksize end

"""
    batched_mul(A, B)

TODO: Needs Documentation
"""
batched_mul(A, B) = _batched_mul(A, B)

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
    idx ≥ 2 && throw(BoundsError(batchview(A), idx))
end
batchview(A::AbstractArray) = eachslice(A; dims=ndims(A))
batchview(A::AbstractVector) = (A,)
