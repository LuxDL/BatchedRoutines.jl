function CRC.rrule(::typeof(batched_jacobian), ad, f::F, x::AbstractMatrix) where {F}
    if !_is_extension_loaded(Val(:ForwardDiff)) || !_is_extension_loaded(Val(:Zygote))
        throw(ArgumentError("`ForwardDiff.jl` and `Zygote.jl` needs to be loaded to \
                             compute the gradient of `batched_jacobian`."))
    end

    J = batched_jacobian(ad, f, x)

    ∇batched_jacobian = Δ -> begin
        gradient_ad = AutoZygote()
        _map_fnₓ = ((i, Δᵢ),) -> _jacobian_vector_product(AutoForwardDiff(),
            x -> batched_gradient(gradient_ad, x_ -> sum(vec(f(x_))[i:i]), x), x, Δᵢ)
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
            x -> batched_gradient(AutoZygote(), x_ -> sum(vec(f(x_, p))[i:i]), x), x, Δᵢ)

        ∂x = reshape(mapreduce(_map_fnₓ, +, enumerate(eachrow(Δ))), size(x))

        _map_fnₚ = ((i, Δᵢ),) -> _jacobian_vector_product(AutoForwardDiff(),
            (x, p_) -> batched_gradient(AutoZygote(), p__ -> sum(vec(f(x, p__))[i:i]), p_),
            x, Δᵢ, p)

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

    if ad isa AutoForwardDiff && get_device(x) isa LuxDeviceUtils.AbstractLuxGPUDevice
        @warn "`rrule` of `batched_gradient($(ad))` might fail on GPU. Consider using \
               `AutoZygote` instead." maxlog=1
    end

    dx = batched_gradient(ad, f, x, p)
    ∇batched_gradient = @closure Δ -> begin
        ∂x = _jacobian_vector_product(
            AutoForwardDiff(), @closure(x->batched_gradient(ad, Base.Fix2(f, p), x)),
            x, reshape(Δ, size(x)))
        ∂p = _jacobian_vector_product(AutoForwardDiff(),
            @closure((x, p)->batched_gradient(
                _maybe_remove_chunksize(ad, p), Base.Fix1(f, x), p)),
            x,
            reshape(Δ, size(x)),
            p)
        return NoTangent(), NoTangent(), NoTangent(), ∂x, ∂p
    end
    return dx, ∇batched_gradient
end

# batched_mul rrule
function CRC.rrule(::typeof(_batched_mul), A::AbstractArray{T1, 3},
        B::AbstractArray{T2, 3}) where {T1, T2}
    ∇batched_mul = @closure _Δ -> begin
        Δ = CRC.unthunk(_Δ)
        tmpA = batched_mul(Δ, batched_adjoint(B))
        ∂A = size(A, 3) == 1 ? sum(tmpA; dims=3) : tmpA
        tmpB = batched_mul(batched_adjoint(A), Δ)
        ∂B = size(B, 3) == 1 ? sum(tmpB; dims=3) : tmpB
        return (NoTangent(), ∂A, ∂B)
    end
    return batched_mul(A, B), ∇batched_mul
end

# constructor
function CRC.rrule(::Type{<:UniformBlockDiagonalOperator}, data)
    ∇UniformBlockDiagonalOperator = @closure Δ -> begin
        ∂data = Δ isa UniformBlockDiagonalOperator ? getdata(Δ) :
                (Δ isa NoTangent ? NoTangent() : Δ)
        return (NoTangent(), ∂data)
    end
    return UniformBlockDiagonalOperator(data), ∇UniformBlockDiagonalOperator
end

function CRC.rrule(::typeof(getproperty), op::UniformBlockDiagonalOperator, x::Symbol)
    @assert x === :data
    ∇getproperty = @closure Δ -> (NoTangent(), UniformBlockDiagonalOperator(Δ))
    return op.data, ∇getproperty
end

# mapreduce fallback rules for UniformBlockDiagonalOperator
@inline _unsum(x, dy, dims) = broadcast(last ∘ tuple, x, dy)
@inline _unsum(x, dy, ::Colon) = broadcast(last ∘ tuple, x, Ref(dy))

function CRC.rrule(::typeof(sum), ::typeof(abs2), op::UniformBlockDiagonalOperator{T};
        dims=:) where {T <: Union{Real, Complex}}
    y = sum(abs2, op; dims)
    ∇sum_abs2 = @closure Δ -> begin
        ∂op = if dims isa Colon
            UniformBlockDiagonalOperator(2 .* real.(Δ) .* getdata(op))
        else
            UniformBlockDiagonalOperator(2 .* real.(getdata(Δ)) .* getdata(op))
        end
        return NoTangent(), NoTangent(), ∂op
    end
    return y, ∇sum_abs2
end

function CRC.rrule(::typeof(sum), ::typeof(identity), op::UniformBlockDiagonalOperator{T};
        dims=:) where {T <: Union{Real, Complex}}
    y = sum(abs2, op; dims)
    project = CRC.ProjectTo(getdata(op))
    ∇sum_abs2 = @closure Δ -> begin
        ∂op = project(_unsum(getdata(op), getdata(Δ), dims))
        return NoTangent(), NoTangent(), UniformBlockDiagonalOperator(∂op)
    end
    return y, ∇sum_abs2
end

# Direct Ldiv
function CRC.rrule(cfg::CRC.RuleConfig{>:CRC.HasReverseMode}, ::typeof(\),
        op::UniformBlockDiagonalOperator, b::AbstractMatrix)
    # We haven't implemented the rrule for least squares yet, direct AD through the code
    size(op, 1) != size(op, 2) && return CRC.rrule_via_ad(cfg, __internal_backslash, op, b)
    # TODO: reuse the factorization once, `factorize(op)` has been implemented
    u = op \ b
    proj_A = CRC.ProjectTo(getdata(op))
    proj_b = CRC.ProjectTo(b)
    ∇backslash = @closure ∂u -> begin
        λ = op' \ ∂u
        ∂A = -batched_mul(λ, batched_adjoint(reshape(u, :, 1, nbatches(u))))
        return NoTangent(), UniformBlockDiagonalOperator(proj_A(∂A)), proj_b(λ)
    end
    return u, ∇backslash
end
