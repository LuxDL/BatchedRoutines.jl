function __batched_value_and_jacobian(ad, f::F, x) where {F}
    J = batched_jacobian(ad, f, x)
    return f(x), J
end

# TODO: Use OneElement for this
function CRC.rrule(::typeof(batched_jacobian), ad, f::F, x::AbstractMatrix) where {F}
    if !_is_extension_loaded(Val(:ForwardDiff)) || !_is_extension_loaded(Val(:Zygote))
        throw(ArgumentError("`ForwardDiff.jl` and `Zygote.jl` needs to be loaded to \
                             compute the gradient of `batched_jacobian`."))
    end

    J = batched_jacobian(ad, f, x)

    ∇batched_jacobian = Δ -> begin
        gradient_ad = AutoZygote()
        _map_fnₓ = ((i, Δᵢ),) -> _jacobian_vector_product(AutoForwardDiff(),
            x -> batched_gradient(gradient_ad, x_ -> sum(vec(f(x_))[i:i]), x),
            x, reshape(Δᵢ, size(x)))
        ∂x = reshape(mapreduce(_map_fnₓ, +, enumerate(eachrow(Δ))), size(x))
        return NoTangent(), NoTangent(), NoTangent(), ∂x
    end
    return J, ∇batched_jacobian
end

function CRC.rrule(::typeof(batched_jacobian), ad, f::F, x, p) where {F}
    if !_is_extension_loaded(Val(:ForwardDiff)) || !_is_extension_loaded(Val(:Zygote))
        throw(ArgumentError("`ForwardDiff.jl` and `Zygote.jl` needs to be loaded to \
                             compute the gradient of `batched_jacobian`."))
    end

    J = batched_jacobian(ad, f, x, p)

    ∇batched_jacobian = Δ -> begin
        _map_fnₓ = ((i, Δᵢ),) -> _jacobian_vector_product(AutoForwardDiff(),
            x -> batched_gradient(AutoZygote(), x_ -> sum(vec(f(x_, p))[i:i]), x),
            x, reshape(Δᵢ, size(x)))

        ∂x = reshape(mapreduce(_map_fnₓ, +, enumerate(eachrow(Δ))), size(x))

        _map_fnₚ = ((i, Δᵢ),) -> _jacobian_vector_product(AutoForwardDiff(),
            (x, p_) -> batched_gradient(AutoZygote(), p__ -> sum(vec(f(x, p__))[i:i]), p_),
            x, reshape(Δᵢ, size(x)), p)

        ∂p = reshape(mapreduce(_map_fnₚ, +, enumerate(eachrow(Δ))), size(p))

        return NoTangent(), NoTangent(), NoTangent(), ∂x, ∂p
    end

    return J, ∇batched_jacobian
end

function CRC.rrule(::typeof(batched_gradient), ad, f::F, x) where {F}
    _is_extension_loaded(Val(:ForwardDiff)) ||
        throw(ArgumentError("`ForwardDiff.jl` needs to be loaded to compute the gradient \
                             of `batched_gradient`."))

    dx = batched_gradient(ad, f, x)
    ∇batched_gradient = @closure Δ -> begin
        ∂x = _jacobian_vector_product(
            AutoForwardDiff(), @closure(x->batched_gradient(ad, f, x)),
            x, reshape(Δ, size(x)))
        return NoTangent(), NoTangent(), NoTangent(), ∂x
    end
    return dx, ∇batched_gradient
end

function CRC.rrule(::typeof(batched_gradient), ad, f::F, x, p) where {F}
    _is_extension_loaded(Val(:ForwardDiff)) ||
        throw(ArgumentError("`ForwardDiff.jl` needs to be loaded to compute the gradient \
                             of `batched_gradient`."))

    dx = batched_gradient(ad, f, x, p)
    ∇batched_gradient = @closure Δ -> begin
        ∂x = _jacobian_vector_product(
            AutoForwardDiff(), @closure(x->batched_gradient(ad, Base.Fix2(f, p), x)),
            x, reshape(Δ, size(x)))
        ∂p = _jacobian_vector_product(
            AutoForwardDiff(), @closure((x, p)->batched_gradient(ad, Base.Fix1(f, x), p)),
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
