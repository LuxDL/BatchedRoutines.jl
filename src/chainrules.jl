function __batched_value_and_jacobian(ad, f::F, x::AbstractMatrix) where {F}
    J = batched_jacobian(ad, f, x)
    return f(x), J
end

# Reverse over Forward: Just construct Hessian for now
function ChainRulesCore.rrule(::RuleConfig{>:HasReverseMode}, ::typeof(batched_jacobian),
        ad, f::F, x::AbstractMatrix) where {F}
    N, B = size(x)
    J, H = __batched_value_and_jacobian(
        ad, @closure(y->reshape(batched_jacobian(ad, f, y).data, :, B)), x)

    function ∇batched_jacobian(Δ)
        ∂x = reshape(batched_mul(Δ, H).data, :, nbatches(Δ))
        return NoTangent(), NoTangent(), NoTangent(), ∂x
    end

    return UniformBlockDiagonalMatrix(reshape(J, :, N, B)), ∇batched_jacobian
end
