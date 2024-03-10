function __batched_value_and_jacobian(ad, f::F, x) where {F}
    J = batched_jacobian(ad, f, x)
    return f(x), J
end

# Reverse over Forward: Just construct Hessian for now
function CRC.rrule(::typeof(batched_jacobian), ad, f::F, x::AbstractMatrix) where {F}
    N, B = size(x)
    J, H = __batched_value_and_jacobian(
        ad, @closure(y->reshape(batched_jacobian(ad, f, y).data, :, B)), x)

    function ∇batched_jacobian(Δ)
        ∂x = reshape(
            batched_mul(reshape(Δ.data, 1, :, nbatches(Δ)), H.data), :, nbatches(Δ))
        return NoTangent(), NoTangent(), NoTangent(), ∂x
    end

    return UniformBlockDiagonalMatrix(reshape(J, :, N, B)), ∇batched_jacobian
end

function CRC.rrule(::typeof(batched_jacobian), ad, f::F, x, p) where {F}
    N, B = size(x)
    J, H = __batched_value_and_jacobian(
        ad, @closure(y->reshape(batched_jacobian(ad, f, y, p).data, :, B)), x)

    p_size = size(p)
    _, Jₚ_ = __batched_value_and_jacobian(
        ad, @closure(p->reshape(batched_jacobian(ad, f, x, reshape(p, p_size)).data, :, B)),
        vec(p))
    Jₚ = dropdims(Jₚ_.data; dims=3)

    function ∇batched_jacobian(Δ)
        ∂x = reshape(
            batched_mul(reshape(Δ.data, 1, :, nbatches(Δ)), H.data), :, nbatches(Δ))
        ∂p = reshape(reshape(Δ.data, 1, :) * Jₚ, p_size)
        return NoTangent(), NoTangent(), NoTangent(), ∂x, ∂p
    end

    return UniformBlockDiagonalMatrix(reshape(J, :, N, B)), ∇batched_jacobian
end

# batched_mul rrule
function CRC.rrule(::typeof(batched_mul), A::AbstractArray{T1, 3},
        B::AbstractArray{T2, 3}) where {T1, T2}
    function ∇batched_mul(_Δ)
        Δ = CRC.unthunk(_Δ)
        Athunk = CRC.@thunk begin
            tmp = batched_mul(Δ, batched_adjoint(B))
            size(A, 3) == 1 ? sum(tmp; dims=3) : tmp
        end
        Bthunk = CRC.@thunk begin
            tmp = batched_mul(batched_adjoint(A), Δ)
            size(B, 3) == 1 ? sum(tmp; dims=3) : tmp
        end
        return (NoTangent(), Athunk, Bthunk)
    end
    return batched_mul(A, B), ∇batched_mul
end

# constructor
function CRC.rrule(::Type{<:UniformBlockDiagonalMatrix}, data)
    ∇UniformBlockDiagonalMatrix(Δ) = (NoTangent(), Δ.data)
    return UniformBlockDiagonalMatrix(data), ∇UniformBlockDiagonalMatrix
end

function CRC.rrule(::typeof(getproperty), A::UniformBlockDiagonalMatrix, x::Symbol)
    @assert x === :data
    ∇getproperty(Δ) = (NoTangent(), UniformBlockDiagonalMatrix(Δ))
    return A.data, ∇getproperty
end
