function __batched_value_and_jacobian(ad, f::F, x) where {F}
    J = batched_jacobian(ad, f, x)
    return f(x), J
end

# FIXME: Gradient of jacobians is really in-efficient here
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

function CRC.rrule(::typeof(batched_gradient), ad, f::F, x) where {F}
    BatchedRoutines._is_extension_loaded(Val(:ForwardDiff)) ||
        throw(ArgumentError("`ForwardDiff.jl` needs to be loaded to compute the gradient \
                             of `batched_gradient`."))

    dx = BatchedRoutines.batched_gradient(ad, f, x)
    ∇batched_gradient = @closure Δ -> begin
        ∂x = _jacobian_vector_product(
            AutoForwardDiff(), @closure(x->BatchedRoutines.batched_gradient(ad, f, x)),
            x, reshape(Δ, size(x)))
        return NoTangent(), NoTangent(), NoTangent(), ∂x
    end
    return dx, ∇batched_gradient
end

function CRC.rrule(::typeof(batched_gradient), ad, f::F, x, p) where {F}
    BatchedRoutines._is_extension_loaded(Val(:ForwardDiff)) ||
        throw(ArgumentError("`ForwardDiff.jl` needs to be loaded to compute the gradient \
                             of `batched_gradient`."))

    dx = BatchedRoutines.batched_gradient(ad, f, x, p)
    ∇batched_gradient = @closure Δ -> begin
        ∂x = _jacobian_vector_product(AutoForwardDiff(),
            @closure(x->BatchedRoutines.batched_gradient(ad, Base.Fix2(f, p), x)),
            x, reshape(Δ, size(x)))
        ∂p = _jacobian_vector_product(AutoForwardDiff(),
            @closure((x, p)->BatchedRoutines.batched_gradient(ad, Base.Fix1(f, x), p)),
            x, reshape(Δ, size(x)), p)
        return NoTangent(), NoTangent(), NoTangent(), ∂x, ∂p
    end
    return dx, ∇batched_gradient
end

# batched_mul rrule
function CRC.rrule(::typeof(_batched_mul), A::AbstractArray{T1, 3},
        B::AbstractArray{T2, 3}) where {T1, T2}
    function ∇batched_mul(_Δ)
        Δ = CRC.unthunk(_Δ)
        ∂A = CRC.@thunk begin
            tmp = batched_mul(Δ, batched_adjoint(B))
            size(A, 3) == 1 ? sum(tmp; dims=3) : tmp
        end
        ∂B = CRC.@thunk begin
            tmp = batched_mul(batched_adjoint(A), Δ)
            size(B, 3) == 1 ? sum(tmp; dims=3) : tmp
        end
        return (NoTangent(), ∂A, ∂B)
    end
    return batched_mul(A, B), ∇batched_mul
end

function CRC.rrule(::typeof(*), X::UniformBlockDiagonalMatrix{<:Union{Real, Complex}},
        Y::AbstractMatrix{<:Union{Real, Complex}})
    function ∇times(_Δ)
        Δ = CRC.unthunk(_Δ)
        ∂X = CRC.@thunk(Δ*batched_adjoint(batched_reshape(Y, :, 1)))
        ∂Y = CRC.@thunk begin
            res = (X' * Δ)
            Y isa UniformBlockDiagonalMatrix ? res : dropdims(res.data; dims=2)
        end
        return (NoTangent(), ∂X, ∂Y)
    end
    return X * Y, ∇times
end

# constructor
function CRC.rrule(::Type{<:UniformBlockDiagonalMatrix}, data)
    function ∇UniformBlockDiagonalMatrix(Δ)
        ∂data = Δ isa UniformBlockDiagonalMatrix ? Δ.data :
                (Δ isa NoTangent ? NoTangent() : Δ)
        return (NoTangent(), ∂data)
    end
    return UniformBlockDiagonalMatrix(data), ∇UniformBlockDiagonalMatrix
end

function CRC.rrule(::typeof(getproperty), A::UniformBlockDiagonalMatrix, x::Symbol)
    @assert x === :data
    ∇getproperty(Δ) = (NoTangent(), UniformBlockDiagonalMatrix(Δ))
    return A.data, ∇getproperty
end
