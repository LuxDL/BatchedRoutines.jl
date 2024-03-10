module BatchedRoutinesFiniteDiffExt

using ADTypes: AutoFiniteDiff
using ArrayInterface: parameterless_type
using BatchedRoutines: BatchedRoutines, _assert_type
using FiniteDiff: FiniteDiff

# api.jl
## Exposed API
@inline function BatchedRoutines.batched_jacobian(
        ad::AutoFiniteDiff, f::F, u::AbstractVector{T}) where {F, T}
    J = FiniteDiff.finite_difference_jacobian(f, u, ad.fdjtype)
    (_assert_type(f) && _assert_type(u) && Base.issingletontype(F)) &&
        return J::parameterless_type(u){T, 2}
    return J
end

@inline function BatchedRoutines.batched_jacobian(
        ad::AutoFiniteDiff, f::F, u::AbstractMatrix) where {F}
    return error(2)
end

end