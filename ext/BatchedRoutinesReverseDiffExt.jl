module BatchedRoutinesReverseDiffExt

using ADTypes: AutoReverseDiff, AutoForwardDiff
using ArrayInterface: ArrayInterface
using BatchedRoutines: BatchedRoutines, batched_pickchunksize, _assert_type
using ChainRulesCore: ChainRulesCore, NoTangent
using ConcreteStructs: @concrete
using FastClosures: @closure
using ReverseDiff: ReverseDiff

const CRC = ChainRulesCore

@inline BatchedRoutines._is_extension_loaded(::Val{:ReverseDiff}) = true

Base.@assume_effects :total BatchedRoutines._assert_type(::Type{<:ReverseDiff.TrackedArray})=false
Base.@assume_effects :total BatchedRoutines._assert_type(::Type{<:AbstractArray{<:ReverseDiff.TrackedReal}})=false

function BatchedRoutines._batched_gradient(::AutoReverseDiff, f::F, u) where {F}
    return ReverseDiff.gradient(f, u)
end

# Chain rules integration
function BatchedRoutines.batched_jacobian(
        ad, f::F, x::AbstractMatrix{<:ReverseDiff.TrackedReal}) where {F}
    return BatchedRoutines.batched_jacobian(ad, f, ArrayInterface.aos_to_soa(x))
end

ReverseDiff.@grad_from_chainrules BatchedRoutines.batched_jacobian(
    ad, f, x::ReverseDiff.TrackedArray)

function BatchedRoutines.batched_jacobian(
        ad, f::F, x::AbstractArray{<:ReverseDiff.TrackedReal},
        p::AbstractArray{<:ReverseDiff.TrackedReal}) where {F}
    return BatchedRoutines.batched_jacobian(
        ad, f, ArrayInterface.aos_to_soa(x), ArrayInterface.aos_to_soa(p))
end

function BatchedRoutines.batched_jacobian(
        ad, f::F, x::AbstractArray, p::AbstractArray{<:ReverseDiff.TrackedReal}) where {F}
    return BatchedRoutines.batched_jacobian(ad, f, x, ArrayInterface.aos_to_soa(p))
end

function BatchedRoutines.batched_jacobian(
        ad, f::F, x::AbstractArray{<:ReverseDiff.TrackedReal}, p::AbstractArray) where {F}
    return BatchedRoutines.batched_jacobian(ad, f, ArrayInterface.aos_to_soa(x), p)
end

ReverseDiff.@grad_from_chainrules BatchedRoutines.batched_jacobian(
    ad, f, x::ReverseDiff.TrackedArray, p::ReverseDiff.TrackedArray)

ReverseDiff.@grad_from_chainrules BatchedRoutines.batched_jacobian(
    ad, f, x, p::ReverseDiff.TrackedArray)

ReverseDiff.@grad_from_chainrules BatchedRoutines.batched_jacobian(
    ad, f, x::ReverseDiff.TrackedArray, p)

function BatchedRoutines.batched_gradient(
        ad, f::F, x::AbstractArray{<:ReverseDiff.TrackedReal}) where {F}
    return BatchedRoutines.batched_gradient(ad, f, ArrayInterface.aos_to_soa(x))
end

ReverseDiff.@grad_from_chainrules BatchedRoutines.batched_gradient(
    ad, f, x::ReverseDiff.TrackedArray)

function BatchedRoutines.batched_gradient(
        ad, f::F, x::AbstractArray{<:ReverseDiff.TrackedReal},
        p::AbstractArray{<:ReverseDiff.TrackedReal}) where {F}
    return BatchedRoutines.batched_gradient(
        ad, f, ArrayInterface.aos_to_soa(x), ArrayInterface.aos_to_soa(p))
end

function BatchedRoutines.batched_gradient(
        ad, f::F, x::AbstractArray, p::AbstractArray{<:ReverseDiff.TrackedReal}) where {F}
    return BatchedRoutines.batched_gradient(ad, f, x, ArrayInterface.aos_to_soa(p))
end

function BatchedRoutines.batched_gradient(
        ad, f::F, x::AbstractArray{<:ReverseDiff.TrackedReal}, p::AbstractArray) where {F}
    return BatchedRoutines.batched_gradient(ad, f, ArrayInterface.aos_to_soa(x), p)
end

ReverseDiff.@grad_from_chainrules BatchedRoutines.batched_gradient(
    ad, f, x::ReverseDiff.TrackedArray, p::ReverseDiff.TrackedArray)

ReverseDiff.@grad_from_chainrules BatchedRoutines.batched_gradient(
    ad, f, x, p::ReverseDiff.TrackedArray)

ReverseDiff.@grad_from_chainrules BatchedRoutines.batched_gradient(
    ad, f, x::ReverseDiff.TrackedArray, p)

end
