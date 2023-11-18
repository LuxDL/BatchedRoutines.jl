module BatchedArraysTullioExt

using BatchedArrays, Tullio
import BatchedArrays: __batched_gemm_tullio!

# ---------------------
# Matrix Multiplication
# ---------------------
function __batched_gemm_tullio!(C_::BatchedMatrix, A_::BatchedMatrix, B_::BatchedMatrix,
        α::Number, β::Number)
    C, A, B = C_.data, A_.data, B_.data
    if iszero(β)
        if isone(α)
            @tullio C[i, j, k] = A[i, l, k] * B[l, j, k]
        else
            @tullio C[i, j, k] = α * A[i, l, k] * B[l, j, k]
        end
    else
        if isone(α)
            @tullio C[i, j, k] = A[i, l, k] * B[l, j, k] + β * C[i, j, k]
        else
            @tullio C[i, j, k] = α * A[i, l, k] * B[l, j, k] + β * C[i, j, k]
        end
    end
    return C_
end

end
