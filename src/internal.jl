# We are mostly focused on Neural Networks and as such most of the mutating functions are
# considered internal API
batched_mul!(C, A, B) = _batched_mul!(C, A, B)

@inline function batched_dot!(A::AbstractVector, B::AbstractMatrix, C::AbstractMatrix)
    if ArrayInterface.fast_scalar_indexing(A)
        @inbounds for (i, (Bᵢ, Cᵢ)) in enumerate(zip(eachcol(B), eachcol(C)))
            A[i] = LinearAlgebra.dot(Bᵢ, Cᵢ)
        end
    else
        sum!(A', B .* C)
    end
    return A
end

@inline function batched_axpy!(
        α_mul::Number, α::AbstractVector, x::AbstractMatrix, y::AbstractMatrix)
    if ArrayInterface.fast_scalar_indexing(y)
        for (i, (xᵢ, yᵢ)) in enumerate(zip(eachcol(x), eachcol(y)))
            axpy!(α_mul * α[i], xᵢ, yᵢ)
        end
    else
        y .+= α_mul .* reshape(α, 1, :) * x
    end
end
