# Some of these are based off
# https://github.com/JaneliaSciComp/BatchedBLAS.jl/blob/master/src/BatchedBLAS.jl
## https://github.com/JuliaLang/julia/issues/40469
@inline _maybe_cast(::Type, x) = x
@inline function _maybe_cast(::Type{T}, x::AbstractFloat) where {T <: Integer}
    return round(T, clamp(x, typemin(T), typemax(T)))
end

@inline function batched_dot_kernel!(::Type{T}, o, x, y) where {T}
    k = CUDA.threadIdx().x + (CUDA.blockIdx().x - 1) * CUDA.blockDim().x
    @inbounds if k ≤ size(x, 2)
        tmp = T(0)
        for i in 1:size(x, 1)
            tmp += x[i, k] * T(y[i, k])
        end
        o[k] = _maybe_cast(eltype(o), tmp)
    end
    return nothing
end

function BatchedRoutines.batched_dot!(A::CuVector, B::CuMatrix, C::CuMatrix)
    T = promote_type(eltype(A), eltype(B), eltype(C))
    CUDA.@cuda name="batched_dot!" launch=true batched_dot_kernel!(T, A, B, C)
    return A
end

# TODO: batched_axpy!
