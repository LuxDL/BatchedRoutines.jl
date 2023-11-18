# Most of this code is from https://github.com/FluxML/NNlib.jl/blob/master/src/batched/batchedmul.jl
# Entry Point
function _batched_mul(A::BatchedArray{T1}, B::BatchedArray{T2}) where {T1, T2}
    @assert ndims(A) ≤ 2 && ndims(B) ≤ 2
    # TODO (high-priority): Implement this!!!
    error(1)
end

function _batched_mul(A::BatchedMatrix{T1}, B::BatchedMatrix{T2}) where {T1, T2}
    if nbatches(A) != nbatches(B) && (nbatches(A) != 1 || nbatches(B) != 1)
        throw(DimensionMismatch("Batch dimensions must match or either must be 1."))
    end
    return __batched_mul(__storage_typejoin(A, B), A, B)
end

# TODO (high-priority): Implement this for non matrix Batched Matvec!!!

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

function __batched_mul!(::Type{T}, C::BatchedMatrix, A::BatchedMatrix, B::BatchedMatrix,
        α::Number, β::Number) where {T}
    @debug "`__batched_mul!` got non Concrete Type $T. Using generic fallback!"
    __batched_mul_generic!(C, A, B, α, β)
    return C
end

# FIXME (low-priority): Inconsistency between BlasFloat and cuBlasFloat here for Float16.
function __batched_mul!(::Type{DT}, C::BatchedMatrix, A::BatchedMatrix,
        B::BatchedMatrix, α::Number, β::Number) where {DT <: DenseArray{<:BlasFloat}}
    __batched_mul_try_gemm!(DT, C, A, B, α, β)
    return C
end

function __batched_mul_try_gemm!(::Type{DT}, C::BatchedMatrix, A::BatchedMatrix,
        B::BatchedMatrix, α::Number, β::Number) where {DT <: DenseArray{<:BlasFloat}}
    T = eltype(DT)
    α, β = promote(α, β, T(0))

    # If these don't match in types then we need to use the generic implementation
    (α isa T && β isa T) || return __batched_mul_generic!(C, A, B, α, β)

    # If any of them are strided, we need to use the generic implementation
    (__is_strided(A) || __is_strided(B) || __is_strided(C)) ||
        return __batched_mul_generic!(C, A, B, α, β)

    A_data, B_data = A.data, B.data

    blasA, transA = if A_data isa PermutedDimsArray{<:AbstractArray, (2, 1, 3)} &&
                       T <: Complex
        stride(parent(A_data), 1) == 1 || return batched_mul_generic!(C, A, B, α, β)
        BatchedArray{T, nbatches(A)}(parent(A_data)), 'C'
    elseif stride(A_data, 2) == 1 && size(A_data, 1) > 1
        transpose(A), 'T'
    elseif stride(A_data, 1) == 1
        A, 'N'
    elseif stride(A_data, 2) == 1
        transpose(A), 'T'
    else
        return __batched_mul_generic!(C, A, B, α, β)
    end

    blasB, transB = if B_data isa PermutedDimsArray{<:AbstractArray, (2, 1, 3)} &&
                       T <: Complex
        stride(parent(B_data), 1) == 1 || return batched_mul_generic!(C, A, B, α, β)
        BatchedArray{T, nbatches(B)}(parent(B_data)), 'C'
    elseif stride(B_data, 2) == 1 && size(B_data, 1) > 1
        transpose(B), 'T'
    elseif stride(B_data, 1) == 1
        B, 'N'
    elseif stride(B_data, 2) == 1
        transpose(B), 'T'
    else
        return __batched_mul_generic!(C, A, B, α, β)
    end

    __batched_gemm!(DT, transA, transB, α, blasA, A, blasB, B, β, C)
    return C
end

# FIXME (med-priority): For Static Arrays force using LV implementation

function __batched_gemm!(::Type{<:Array}, transA::Char, transB::Char, α::Number, A, org_A,
        B, org_B, β::Number, C)
    # TODO (medium-priority): We should use a polyalgorithm to decide between Looped BLAS &
    # TODO (medium-priority): Threaded BLAS vs MKL Batched vs Tullio/LV
    return __batched_gemm_cpu!(transA, transB, α, A, B, β, C)
end

function __batched_gemm!(::Type{T}, transA::Char, transB::Char, α::Number, A, org_A,
        B, org_B, β::Number, C) where {T}
    # If we don't have a specific blas dispatch implemented, just use the generic matmul
    # instead of failing!
    @debug "Tried using `__batched_gemm!` for $(T) but no direct dispatch found!"
    return __batched_mul_generic!(C, org_A, org_B, α, β)
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

function __batched_gemm_tullio! end

## TODO (medium-priority): Use MKL batched blas routines if MKL is loaded
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
        BLAS.set_num_threads(old_threads)
    else
        for (Cᵢ, Aᵢ, Bᵢ) in zip(batchview(C), batchview(A), batchview(B))
            BLAS.gemm!(transA, transB, α, Aᵢ, Bᵢ, β, Cᵢ)
        end
    end
end
