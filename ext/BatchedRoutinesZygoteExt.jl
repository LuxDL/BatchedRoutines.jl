module BatchedRoutinesZygoteExt

using ADTypes: AutoForwardDiff, AutoZygote
using BatchedRoutines: BatchedRoutines
using ChainRulesCore: ChainRulesCore, NoTangent
using FastClosures: @closure
using Zygote: Zygote

const CRC = ChainRulesCore

@inline BatchedRoutines._is_extension_loaded(::Val{:Zygote}) = true

function BatchedRoutines.batched_gradient(::AutoZygote, f::F, u::AbstractMatrix) where {F}
    return only(Zygote.gradient(f, u))
end

function CRC.rrule(::typeof(BatchedRoutines.batched_gradient),
        ad::AutoZygote, f::F, x::AbstractMatrix) where {F}
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

    dx, pb_f = Zygote.pullback(@closure(x->only(Zygote.gradient(f, x))), x)
    ∇batched_gradient = @closure Δ -> begin
        ∂x = only(pb_f(Δ)) # Else we have to do Zygote over Zygote
        return NoTangent(), NoTangent(), NoTangent(), ∂x
    end
    return dx, ∇batched_gradient
end

end
