function BatchedRoutines.__batched_gemm!(::Type{<:CuArray{<:CuBlasFloat}}, transA::Char,
        transB::Char, α::Number, A, org_A, B, org_B, β::Number, C)
    CUBLAS.gemm_strided_batched!(transA, transB, α, A, B, β, C)
    return C
end
