# Most of this code is from https://github.com/FluxML/NNlib.jl/blob/master/src/batched/batchedmul.jl
function __batched_mul(::Type, A::BatchedMatrix, B::BatchedMatrix)
    T = promote_type(eltype(A), eltype(B))
    C = similar(A, T, size(A, 1), size(B, 2))
    __batched_mul!(C, A, B)
    return C
end

# function __batched_mul(::Type{<})
    
# end