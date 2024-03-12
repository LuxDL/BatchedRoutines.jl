module BatchedRoutinesReverseDiffExt

using ADTypes: AutoReverseDiff, AutoForwardDiff
using BatchedRoutines: BatchedRoutines, batched_pickchunksize, _assert_type
using ChainRulesCore: ChainRulesCore, NoTangent
using ConcreteStructs: @concrete
using FastClosures: @closure
using ReverseDiff: ReverseDiff

const CRC = ChainRulesCore

@inline BatchedRoutines._is_extension_loaded(::Val{:ReverseDiff}) = true

Base.@assume_effects :total BatchedRoutines._assert_type(::Type{<:ReverseDiff.TrackedArray})=false
Base.@assume_effects :total BatchedRoutines._assert_type(::Type{<:AbstractArray{<:ReverseDiff.TrackedReal}})=false

function BatchedRoutines.batched_gradient(
        ::AutoReverseDiff, f::F, u::AbstractMatrix) where {F}
    return ReverseDiff.gradient(f, u)
end

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

function _value_and_pullback(f::F, x) where {F}
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

function CRC.rrule(::typeof(BatchedRoutines.batched_gradient),
        ad::AutoReverseDiff, f::F, x::AbstractMatrix) where {F}
    if BatchedRoutines._is_extension_loaded(Val(:ForwardDiff))
        dx = BatchedRoutines.batched_gradient(ad, f, x)
        # Use Forward Over Reverse to compute the Hessian Vector Product
        ∇batched_gradient = @closure Δ -> begin
            ∂x = BatchedRoutines._jacobian_vector_product(
                AutoForwardDiff(), @closure(x->BatchedRoutines.batched_gradient(ad, f, x)),
                x, reshape(Δ, size(x)))
            return NoTangent(), NoTangent(), NoTangent(), ∂x
        end
        return dx, ∇batched_gradient
    end

    dx, pb_f = _value_and_pullback(Base.Fix1(ReverseDiff.gradient, f), x)
    ∇batched_gradient = @closure Δ -> (NoTangent(), NoTangent(), NoTangent(), pb_f(Δ))
    return dx, ∇batched_gradient
end

end
