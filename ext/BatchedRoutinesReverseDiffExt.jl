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

@concrete struct ReverseDiffPullbackFunction <: Function
    tape
    ∂input
    output
end

function (pb_f::ReverseDiffPullbackFunction)(Δ)
    if pb_f.output isa AbstractArray{<:ReverseDiff.TrackedReal}
        @inbounds for (oᵢ, Δᵢ) in zip(vec(pb_f.output), vec(Δ))
            oᵢ.deriv = Δᵢ
        end
    else
        vec(pb_f.output.deriv) .= vec(Δ)
    end
    ReverseDiff.reverse_pass!(pb_f.tape)
    return pb_f.∂input
end

function BatchedRoutines._value_and_pullback(::AutoReverseDiff, f::F, x) where {F}
    tape = ReverseDiff.InstructionTape()
    ∂x = zero(x)
    x_tracked = ReverseDiff.TrackedArray(x, ∂x, tape)
    y_tracked = f(x_tracked)

    if y_tracked isa AbstractArray{<:ReverseDiff.TrackedReal}
        y = ReverseDiff.value.(y_tracked)
    else
        y = ReverseDiff.value(y_tracked)
    end

    return y, ReverseDiffPullbackFunction(tape, ∂x, y_tracked)
end

end
