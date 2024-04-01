# Low Level Wrappers
for (fname, elty) in ((:cublasDgetrsBatched, :Float64), (:cublasSgetrsBatched, :Float32),
    (:cublasZgetrsBatched, :ComplexF64), (:cublasCgetrsBatched, :ComplexF32))
    @eval begin
        function getrs_batched!(trans::Char, n, nrhs, Aptrs::CuVector{CuPtr{$elty}},
                lda, p, Bptrs::CuVector{CuPtr{$elty}}, ldb)
            batchSize = length(Aptrs)
            info = Array{Cint}(undef, batchSize)
            CUBLAS.$fname(
                CUBLAS.handle(), trans, n, nrhs, Aptrs, lda, p, Bptrs, ldb, info, batchSize)
            CUDA.unsafe_free!(Aptrs)
            CUDA.unsafe_free!(Bptrs)
            return info
        end
    end
end

function getrs_strided_batched!(trans::Char, F::DenseCuArray{<:Any, 3}, p::DenseCuMatrix,
        B::Union{DenseCuArray{<:Any, 3}, DenseCuMatrix})
    m, n = size(F, 1), size(F, 2)
    m != n && throw(DimensionMismatch("All matrices must be square!"))
    lda = max(1, stride(F, 2))
    ldb = max(1, stride(B, 2))
    nrhs = ifelse(ndims(B) == 2, 1, size(B, 2))

    Fptrs = CUBLAS.unsafe_strided_batch(F)
    Bptrs = CUBLAS.unsafe_strided_batch(B)
    info = getrs_batched!(trans, n, nrhs, Fptrs, lda, p, Bptrs, ldb)

    return B, info
end
