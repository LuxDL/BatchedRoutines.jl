module BatchedRoutinesForwardDiffExt

using ADTypes: AutoForwardDiff
using ArrayInterface: parameterless_type
using BatchedRoutines: BatchedRoutines, batched_jacobian, batched_mul,
                       batched_pickchunksize, _assert_type
using ChainRulesCore: ChainRulesCore, HasReverseMode, NoTangent, RuleConfig
using FastClosures: @closure
using ForwardDiff: ForwardDiff

# api.jl
function BatchedRoutines.batched_pickchunksize(
        X::AbstractArray{T, N}, n::Int=ForwardDiff.DEFAULT_CHUNK_THRESHOLD) where {T, N}
    return batched_pickchunksize(N == 1 ? length(X) : prod(size(X)[1:(N - 1)]), n)
end
function BatchedRoutines.batched_pickchunksize(
        N::Int, n::Int=ForwardDiff.DEFAULT_CHUNK_THRESHOLD)
    return ForwardDiff.pickchunksize(N, n)
end

## Internal Details
@views function __batched_forwarddiff_value_and_jacobian_chunk(
        idx::Int, ::Val{chunksize}, ::Type{Tag}, ::Type{Dual}, ::Type{Partials},
        f::F, u::AbstractMatrix{T}) where {chunksize, Tag, Dual, Partials, F, T}
    N, B = size(u)
    idxs = idx:min(idx + chunksize - 1, N)
    idxs_prev = 1:(idx - 1)
    idxs_next = (idx + chunksize):N

    partials = map(
        𝒾 -> Partials(ntuple(𝒿 -> ifelse(𝒾 == 𝒿, oneunit(T), zero(T)), chunksize)),
        1:length(idxs))
    u_part_duals = Dual.(u[idxs, :], partials)

    if length(idxs_prev) == 0
        u_part_prev = similar(u_part_duals, 0, B)
    else
        u_part_prev = Dual.(u[idxs_prev, :],
            Partials.(map(𝒾 -> ntuple(_ -> zero(T), chunksize), 1:length(idxs_prev))))
    end

    if length(idxs_next) == 0
        u_part_next = similar(u_part_duals, 0, B)
    else
        u_part_next = Dual.(u[idxs_next, :],
            Partials.(map(𝒾 -> ntuple(_ -> zero(T), chunksize), 1:length(idxs_next))))
    end

    u_duals = vcat(u_part_prev, u_part_duals, u_part_next)
    y_duals = f(u_duals)

    J_partial = mapreduce(hcat, 1:chunksize) do i
        return reshape(ForwardDiff.partials.(y_duals, i), :, 1, B)
    end

    return ForwardDiff.value.(y_duals), J_partial
end

function __batched_value_and_jacobian(ad::AutoForwardDiff, f::F, u::AbstractMatrix{T},
        ck::Val{chunksize}) where {F, T, chunksize}
    N, B = size(u)

    nchunks, remainder = divrem(N, chunksize)

    # Cannot use the jacobian functions from ForwardDiff since they are mutating
    Tag = ad.tag === nothing ? typeof(ForwardDiff.Tag(f, T)) : typeof(ad.tag)
    Dual = ForwardDiff.Dual{Tag, T, chunksize}
    Partials = ForwardDiff.Partials{chunksize, T}

    y, J_first = __batched_forwarddiff_value_and_jacobian_chunk(
        1, ck, Tag, Dual, Partials, f, u)
    if nchunks == 1
        remainder == 0 && return y, J_first
        J = similar(J_first, size(y, 1), N, B)
        J[:, 1:chunksize, :] .= J_first
    else
        J = similar(J_first, size(y, 1), N, B)
        J[:, 1:chunksize, :] .= J_first
        for i in 2:nchunks
            J[:, ((i - 1) * chunksize + 1):(i * chunksize), :] .= last(__batched_forwarddiff_value_and_jacobian_chunk(
                (i - 1) * chunksize + 1, ck, Tag, Dual, Partials, f, u))
        end
    end

    if remainder > 0
        Dual_rem = ForwardDiff.Dual{Tag, T, remainder}
        Partials_rem = ForwardDiff.Partials{remainder, T}
        _, J_last = __batched_forwarddiff_value_and_jacobian_chunk(
            nchunks * chunksize + 1, Val(remainder), Tag, Dual_rem, Partials_rem, f, u)
        J[:, (nchunks * chunksize + 1):end, :] .= J_last
        return y, J
    end

    return y, J
end

@generated function __batched_value_and_jacobian(
        ad::AutoForwardDiff{CK}, f::F, u::AbstractMatrix{T}) where {CK, F, T}
    if CK === nothing || CK ≤ 0
        if _assert_type(u) && Base.issingletontype(F)
            rType = Tuple{u, parameterless_type(u){T, 3}}
            return :(__batched_value_and_jacobian(
                ad, f, u, Val(batched_pickchunksize(u)))::$(rType))
        else # Cases like ReverseDiff over ForwardDiff
            return :(__batched_value_and_jacobian(ad, f, u, Val(batched_pickchunksize(u))))
        end
    end
    return :(__batched_value_and_jacobian(ad, f, u, $(Val(CK))))
end

## Exposed API
@inline function BatchedRoutines.batched_jacobian(
        ad::AutoForwardDiff, f::F, u::AbstractVector{T}) where {F, T}
    tag = ad.tag === nothing ? ForwardDiff.Tag{F, eltype(u)}() : ad.tag
    cfg = ForwardDiff.JacobianConfig(
        f, u, ForwardDiff.Chunk{batched_pickchunksize(u)}(), tag)
    J = ForwardDiff.jacobian(f, u, cfg)
    (_assert_type(f) && _assert_type(u) && Base.issingletontype(F)) &&
        return J::parameterless_type(u){T, 2}
    return J
end

@inline function BatchedRoutines.batched_jacobian(
        ad::AutoForwardDiff, f::F, u::AbstractMatrix) where {F}
    return last(__batched_value_and_jacobian(ad, f, u))
end

## Reverse over Forward: Just construct Hessian for now
function ChainRulesCore.rrule(::RuleConfig{>:HasReverseMode}, ::typeof(batched_jacobian),
        ad::AutoForwardDiff, f::F, x::AbstractMatrix) where {F}
    N, B = size(x)
    J, H = __batched_value_and_jacobian(
        ad, @closure(y->reshape(batched_jacobian(ad, f, y), :, B)), x)

    function ∇batched_jacobian(Δ)
        ∂x = reshape(batched_mul(reshape(Δ, 1, :, B), H), N, B)
        return NoTangent(), NoTangent(), NoTangent(), ∂x
    end

    return reshape(J, :, N, B), ∇batched_jacobian
end

# helpers.jl
Base.@assume_effects :total BatchedRoutines._assert_type(::Type{<:AbstractArray{<:ForwardDiff.Dual}})=false

end
