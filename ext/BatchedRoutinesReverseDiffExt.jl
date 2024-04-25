module BatchedRoutinesReverseDiffExt

using ADTypes: AutoReverseDiff, AutoForwardDiff
using ArrayInterface: ArrayInterface
using BatchedRoutines: BatchedRoutines, batched_pickchunksize, _assert_type,
                       UniformBlockDiagonalOperator, getdata
using ChainRulesCore: ChainRulesCore, NoTangent
using ConcreteStructs: @concrete
using FastClosures: @closure
using ReverseDiff: ReverseDiff

const CRC = ChainRulesCore

@inline BatchedRoutines._is_extension_loaded(::Val{:ReverseDiff}) = true

Base.@assume_effects :total BatchedRoutines._assert_type(::Type{<:ReverseDiff.TrackedArray})=false
Base.@assume_effects :total BatchedRoutines._assert_type(::Type{<:AbstractArray{<:ReverseDiff.TrackedReal}})=false

function BatchedRoutines._batched_gradient(::AutoReverseDiff, f::F, u) where {F}
    Base.issingletontype(f) && return ReverseDiff.gradient(f, u)

    ∂u = similar(u, first(BatchedRoutines._resolve_gradient_type(f, f, u, Val(1))))
    fill!(∂u, false)

    tape = ReverseDiff.InstructionTape()
    u_tracked = ReverseDiff.TrackedArray(u, ∂u, tape)
    y_tracked = f(u_tracked)
    y_tracked.deriv = true
    ReverseDiff.reverse_pass!(tape)

    return ∂u
end

# ReverseDiff compatible `UniformBlockDiagonalOperator`
@inline function ReverseDiff.track(
        op::UniformBlockDiagonalOperator, tp::ReverseDiff.InstructionTape)
    return UniformBlockDiagonalOperator(ReverseDiff.track(getdata(op), tp))
end

@inline function ReverseDiff.deriv(x::UniformBlockDiagonalOperator)
    return UniformBlockDiagonalOperator(ReverseDiff.deriv(getdata(x)))
end

@inline function ReverseDiff.value!(
        op::UniformBlockDiagonalOperator, val::UniformBlockDiagonalOperator)
    ReverseDiff.value!(getdata(op), getdata(val))
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
    ad, f, x::AbstractArray, p::ReverseDiff.TrackedArray)

ReverseDiff.@grad_from_chainrules BatchedRoutines.batched_jacobian(
    ad, f, x::ReverseDiff.TrackedArray, p::AbstractArray)

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
    ad, f, x::AbstractArray, p::ReverseDiff.TrackedArray)

ReverseDiff.@grad_from_chainrules BatchedRoutines.batched_gradient(
    ad, f, x::ReverseDiff.TrackedArray, p::AbstractArray)

end
