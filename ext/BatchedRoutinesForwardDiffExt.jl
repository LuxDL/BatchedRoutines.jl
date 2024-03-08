module BatchedRoutinesForwardDiffExt

using ADTypes: AutoForwardDiff
using BatchedRoutines: BatchedRoutines
using ForwardDiff: ForwardDiff

function BatchedRoutines.batched_pickchunksize(
        X::AbstractArray{T, N}, n::Int=ForwardDiff.DEFAULT_CHUNK_THRESHOLD) where {T, N}
    return batched_pickchunksize(N == 1 ? length(X) : prod(size(X)[1:(N - 1)]), n)
end
function BatchedRoutines.batched_pickchunksize(
        N::Int, n::Int=ForwardDiff.DEFAULT_CHUNK_THRESHOLD)
    return ForwardDiff.pickchunksize(N, n)
end

end
