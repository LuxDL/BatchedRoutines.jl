module BatchedRoutinesFiniteDiffExt

using ADTypes: AutoFiniteDiff
using ArrayInterface: matrix_colors, parameterless_type
using BatchedRoutines: BatchedRoutines, UniformBlockDiagonalMatrix, _assert_type
using FastClosures: @closure
using FiniteDiff: FiniteDiff

@inline BatchedRoutines._is_extension_loaded(::Val{:FiniteDiff}) = true

# api.jl
## Exposed API
@inline function BatchedRoutines._batched_jacobian(
        ad::AutoFiniteDiff, f::F, x::AbstractVector{T}) where {F, T}
    J = FiniteDiff.finite_difference_jacobian(f, x, ad.fdjtype)
    (_assert_type(f) && _assert_type(x) && Base.issingletontype(F)) &&
        (return UniformBlockDiagonalMatrix(J::parameterless_type(x){T, 2}))
    return UniformBlockDiagonalMatrix(J)
end

@inline function BatchedRoutines._batched_jacobian(
        ad::AutoFiniteDiff, f::F, x::AbstractMatrix) where {F}
    f! = @closure (y, x_) -> copyto!(y, f(x_))
    fx = f(x)
    J = UniformBlockDiagonalMatrix(similar(
        x, promote_type(eltype(fx), eltype(x)), size(fx, 1), size(x, 1), size(x, 2)))
    sparsecache = FiniteDiff.JacobianCache(
        x, fx, ad.fdjtype; colorvec=matrix_colors(J), sparsity=J)
    FiniteDiff.finite_difference_jacobian!(J, f!, x, sparsecache)
    return J
end

# NOTE: This doesn't exploit batching
@inline function BatchedRoutines._batched_gradient(ad::AutoFiniteDiff, f::F, x) where {F}
    return FiniteDiff.finite_difference_batched_gradient(f, x, ad.fdjtype)
end

end
