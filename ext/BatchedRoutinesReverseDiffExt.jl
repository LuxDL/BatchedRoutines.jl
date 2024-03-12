module BatchedRoutinesReverseDiffExt

using ADTypes: AutoReverseDiff, AutoForwardDiff
using BatchedRoutines: BatchedRoutines, batched_pickchunksize, _assert_type
using ChainRulesCore: ChainRulesCore, NoTangent
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

    tape = ReverseDiff.InstructionTape()
    ∂x = zero(x)
    x_tracked = ReverseDiff.TrackedArray(x, ∂x, tape)
    y_tracked = ReverseDiff.gradient(f, x_tracked)

    if y_tracked isa AbstractArray{<:ReverseDiff.TrackedReal}
        dx = ReverseDiff.value.(y_tracked)
    else
        dx = ReverseDiff.value(y_tracked)
    end

    ∇batched_gradient = @closure Δ -> begin
        if y_tracked isa AbstractArray{<:ReverseDiff.TrackedReal}
            @inbounds for (oᵢ, Δᵢ) in zip(vec(y_tracked), vec(Δ))
                oᵢ.deriv = Δᵢ
            end
        else
            vec(y_tracked.deriv) .= vec(Δ)
        end
        ReverseDiff.reverse_pass!(tape)
        return NoTangent(), NoTangent(), NoTangent(), ∂x
    end

    return dx, ∇batched_gradient
end

end
