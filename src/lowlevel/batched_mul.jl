# Most of this code is from https://github.com/FluxML/NNlib.jl/blob/master/src/batched/batchedmul.jl
# Entry Point
function _batched_mul(A::BatchedArray{T1}, B::BatchedArray{T2}) where {T1, T2}
    @assert ndims(A) ≤ 2 && ndims(B) ≤ 2
    # TODO: Implement this!!!
    error(1)
end

function _batched_mul(A::BatchedMatrix{T1}, B::BatchedMatrix{T2}) where {T1, T2}
    if nbatches(A) != nbatches(B) && (nbatches(A) != 1 || nbatches(B) != 1)
        throw(DimensionMismatch("Batch dimensions must match or either must be 1."))
    end
    return __batched_mul(__storage_typejoin(A, B), A, B)
end

# TODO: Implement this for non matrix Batched Matvec!!!

function _batched_mul!(C::BatchedMatrix{T}, A::BatchedMatrix, B::BatchedMatrix,
        α::Number=one(T), β::Number=zero(T)) where {T}
    __batched_mul!(__storage_typejoin(C, A, B), C, A, B, α, β)
    return C
end

# Checks fo potentialr BLAS/LAPACK dispatch
function __batched_mul(::Type, A::BatchedMatrix, B::BatchedMatrix)
    T = promote_type(eltype(A), eltype(B))
    C = similar(A, T, size(A, 1), size(B, 2))
    _batched_mul!(C, A, B)
    return C
end

function __batched_mul(::Type{<:DenseArray{T}}, A::BatchedMatrix,
        B::BatchedMatrix) where {T}
    C = similar(A, T, size(A, 1), size(B, 2))
    _batched_mul!(C, __copy_if_strided(A), __copy_if_strided(B))
    return C
end

function __batched_mul!(::Type, C::BatchedMatrix, A::BatchedMatrix, B::BatchedMatrix,
        α::Number, β::Number)
    __batched_mul_generic!(C, A, B, α, β)
    return C
end

function __batched_mul!(::Type{DT}, C::BatchedMatrix, A::BatchedMatrix, B::BatchedMatrix,
        α::Number, β::Number) where {DT <: DenseArray{<:BlasFloat}}
    T = eltype(DT)
    α, β = promote(α, β, T(0))

    # If these don't match in types then we need to use the generic implementation
    (α isa T && β isa T) || return __batched_mul_generic!(C, A, B, α, β)

    # If any of them are strided, we need to use the generic implementation
    (__is_strided(A) || __is_strided(B) || __is_strided(C)) ||
        return __batched_mul_generic!(C, A, B, α, β)

    # TODO: Implement this!!!
    # blasA, transA = if A isa BatchedAdjoint && T <: Complex
    #     Base.stride(parent(A),1) == 1 || return batched_mul_generic!(C, A, B, α, β)
    #     parent(A), 'C'
    # elseif Base.stride(A,2) == 1 && size(A,1) > 1
    #     batched_transpose(A), 'T'
    # elseif Base.stride(A,1) == 1
    #     A, 'N'
    # elseif Base.stride(A,2) == 1  # This is awful, but exhaustively tested. Issues 268, 282.
    #     batched_transpose(A), 'T'
    # else
    #     return batched_mul_generic!(C, A, B, α, β)
    # end

    # blasB, transB = if B isa BatchedAdjoint && T <: Complex
    #     Base.stride(parent(B),1) == 1 || return batched_mul_generic!(C, A, B, α, β)
    #     parent(B), 'C'
    # elseif Base.stride(B,2) == 1 && size(B,1) > 1
    #     batched_transpose(B), 'T'
    # elseif Base.stride(B,1) == 1
    #     B, 'N'
    # elseif Base.stride(B,2) == 1
    #     batched_transpose(B), 'T'
    # else
    #     return batched_mul_generic!(C, A, B, α, β)
    # end

    # _batched_gemm!(DT, transA, transB, alpha, blasA, blasB, beta, C)
    # C
end

function __batched_gemm!(::Type{<:Array}, transA::Char, transB::Char, α::Number, A, B,
        β::Number, C)
    # TODO: We should use a polyalgorithm to decide between Looped BLAS & Threaded BLAS vs
    # TODO: MKL Batched vs Tullio/LV
    return __batched_gemm_cpu!(transA, transB, α, A, B, β, C)
end

# Core Implementation
function __batched_mul_generic!(C::BatchedMatrix, A::BatchedMatrix, B::BatchedMatrix, α, β)
    for i in 1:nbatches(C)
        Cᵢ = batchview(C, i)
        Aᵢ = batchview(A, min(i, nbatches(A)))
        Bᵢ = batchview(B, min(i, nbatches(B)))
        mul!(Cᵢ, Aᵢ, Bᵢ, α, β)
    end
    return C
end

## TODO: Use LoopVectorization/Tullio for small matrices
function __batched_gemm_cpu_tullio! end

## TODO: Use MKL batched blas routines if MKL is loaded
function __batched_gemm_cpu_mkl! end

function __batched_gemm_cpu!(transA::AbstractChar, transB::AbstractChar, α::T,
        A::BatchedMatrix{T}, B::BatchedMatrix{T}, β::T,
        C::BatchedMatrix{T}) where {T <: BlasFloat}
    Base.require_one_based_indexing(A.data)
    Base.require_one_based_indexing(B.data)
    Base.require_one_based_indexing(C.data)

    m = size(A, transA == 'N' ? 1 : 2)
    ka = size(A, transA == 'N' ? 2 : 1)
    kb = size(B, transB == 'N' ? 1 : 2)
    n = size(B, transB == 'N' ? 2 : 1)

    if ka != kb || m != size(C, 1) || n != size(C, 2)
        throw(DimensionMismatch("A1 has size ($m, $ka), B1 has size ($kb, $n), C1 has size $(size(C))"))
    end
    LinearAlgebra.BLAS.chkstride1(A.data)
    LinearAlgebra.BLAS.chkstride1(B.data)
    LinearAlgebra.BLAS.chkstride1(C.data)

    n_threads = min(Threads.maxthreadid(),
        1 + max(length(A) * nbatches(A), length(B) * nbatches(B)) ÷ 8000)

    if n_threads > 1
        old_threads = BLAS.get_num_threads()
        BLAS.set_num_threads(1)
        Threads.@sync for ks in Iterators.partition(1:nbatches(C),
            cld(nbatches(C), n_threads))
            Threads.@spawn for i in ks
                Cᵢ = batchview(C, i)
                Aᵢ = batchview(A, i)
                Bᵢ = batchview(B, i)
                BLAS.gemm!(transA, transB, α, Aᵢ, Bᵢ, β, Cᵢ)
            end
        end
        for (Cᵢ, Aᵢ, Bᵢ) in zip(batchview(C), batchview(A), batchview(B))
            BLAS.gemm!(transA, transB, α, Aᵢ, Bᵢ, β, Cᵢ)
        end
        BLAS.set_num_threads(old_threads)
    else
        for (Cᵢ, Aᵢ, Bᵢ) in zip(batchview(C), batchview(A), batchview(B))
            BLAS.gemm!(transA, transB, α, Aᵢ, Bᵢ, β, Cᵢ)
        end
    end
end
