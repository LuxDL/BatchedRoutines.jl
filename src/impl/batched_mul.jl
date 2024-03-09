# Most of this code is from https://github.com/FluxML/NNlib.jl/blob/master/src/batched/batchedmul.jl
# Entry Point
@inline _batched_mul(A::BatchedMatrix, B::BatchedVector) = vec(_batched_mul(
    A, reshape(B, :, 1)))
@inline _batched_mul(A::BatchedVector, B::BatchedMatrix) = _batched_mul(reshape(A, :, 1), B)

@inline function _batched_mul(A::BatchedMatrix{T1}, B::BatchedMatrix{T2}) where {T1, T2}
    if (nbatches(A) != nbatches(B)) && (nbatches(A) != 1 && nbatches(B) != 1)
        throw(DimensionMismatch("Batch dimensions must match or either must be 1."))
    end
    return __batched_mul(_storage_typejoin(A, B), A, B)
end

@inline function _batched_mul!(C::BatchedVector{T}, A::BatchedMatrix, B::BatchedVector,
        α::Number=one(T), β::Number=zero(T)) where {T}
    _batched_mul!(reshape(C, :, 1), A, reshape(B, :, 1), α, β)
    return C
end
@inline function _batched_mul!(C::BatchedMatrix{T}, A::BatchedVector, B::BatchedMatrix,
        α::Number=one(T), β::Number=zero(T)) where {T}
    _batched_mul!(C, reshape(A, :, 1), B, α, β)
    return C
end

@inline function _batched_mul!(C::BatchedMatrix{T}, A::BatchedMatrix, B::BatchedMatrix,
        α::Number=one(T), β::Number=zero(T)) where {T}
    if (nbatches(A) != nbatches(B)) && (nbatches(A) != 1 && nbatches(B) != 1)
        throw(DimensionMismatch("Batch dimensions must match or either must be 1."))
    end
    @assert nbatches(C) == max(nbatches(A), nbatches(B))
    __batched_mul!(_storage_typejoin(C, A, B), C, A, B, α, β)
    return C
end

# Checks fo potential BLAS dispatch
@inline function __batched_mul(::Type, A::BatchedMatrix, B::BatchedMatrix)
    T = promote_type(eltype(A), eltype(B))
    C = similar(A, T, size(A, 1), size(B, 2), max(nbatches(A), nbatches(B)))
    _batched_mul!(C, A, B)
    return C
end

@inline function __batched_mul(
        ::Type{<:DenseArray{T}}, A::BatchedMatrix, B::BatchedMatrix) where {T}
    C = similar(A, T, size(A, 1), size(B, 2), max(nbatches(A), nbatches(B)))
    _batched_mul!(C, _copy_if_strided(A), _copy_if_strided(B))
    return C
end

@inline function __batched_mul!(::Type{T}, C::BatchedMatrix, A::BatchedMatrix,
        B::BatchedMatrix, α::Number, β::Number) where {T}
    @debug "`__batched_mul!` got non Concrete Type $T. Using generic fallback!"
    __batched_mul_generic!(C, A, B, α, β)
    return C
end

@inline function __batched_mul!(
        ::Type{DT}, C::BatchedMatrix, A::BatchedMatrix, B::BatchedMatrix,
        α::Number, β::Number) where {DT <: DenseArray{<:LinearAlgebra.BlasFloat}}
    __batched_mul_try_gemm!(DT, C, A, B, α, β)
    return C
end

function __batched_mul_try_gemm!(
        ::Type{DT}, C::BatchedMatrix, A::BatchedMatrix, B::BatchedMatrix,
        α::Number, β::Number) where {DT <: DenseArray{<:LinearAlgebra.BlasFloat}}
    T = eltype(DT)
    α, β = promote(α, β, T(0))

    # If these don't match in types then we need to use the generic implementation
    (α isa T && β isa T) || return __batched_mul_generic!(C, A, B, α, β)

    (_is_strided(A) || _is_strided(B)) || return __batched_mul_generic!(C, A, B, α, β)
    C isa StridedArray || return __batched_mul_generic!(C, A, B, α, β)

    blasA, transA = if A isa PermutedDimsArray{<:AbstractArray, (2, 1, 3)} && T <: Complex
        stride(parent(A), 1) == 1 || return batched_mul_generic!(C, A, B, α, β)
        parent(A), 'C'
    elseif stride(A, 2) == 1 && size(A, 1) > 1
        transpose(A), 'T'
    elseif stride(A, 1) == 1
        A, 'N'
    elseif stride(A, 2) == 1
        transpose(A), 'T'
    else
        return __batched_mul_generic!(C, A, B, α, β)
    end

    blasB, transB = if B isa PermutedDimsArray{<:AbstractArray, (2, 1, 3)} && T <: Complex
        stride(parent(B), 1) == 1 || return batched_mul_generic!(C, A, B, α, β)
        parent(B), 'C'
    elseif stride(B, 2) == 1 && size(B, 1) > 1
        transpose(B), 'T'
    elseif stride(B, 1) == 1
        B, 'N'
    elseif stride(B, 2) == 1
        transpose(B), 'T'
    else
        return __batched_mul_generic!(C, A, B, α, β)
    end

    __batched_gemm!(DT, transA, transB, α, blasA, A, blasB, B, β, C)
    return C
end

@inline function __batched_gemm!(::Type{<:Array}, transA::Char, transB::Char,
        α::Number, A, org_A, B, org_B, β::Number, C)
    return __batched_gemm_cpu!(transA, transB, α, A, B, β, C)
end

@inline function __batched_gemm!(::Type{T}, transA::Char, transB::Char, α::Number,
        A, org_A, B, org_B, β::Number, C) where {T}
    # If we don't have a specific blas dispatch implemented, just use the generic matmul
    # instead of failing!
    @debug "Tried using `__batched_gemm!` for $(T) but no direct dispatch found!"
    return __batched_mul_generic!(C, org_A, org_B, α, β)
end

# Core Implementation
@inline function __batched_mul_generic!(
        C::BatchedMatrix, A::BatchedMatrix, B::BatchedMatrix, α, β)
    @show 11
    for i in 1:nbatches(C)
        Cᵢ = batchview(C, i)
        Aᵢ = batchview(A, min(i, nbatches(A)))
        Bᵢ = batchview(B, min(i, nbatches(B)))
        mul!(Cᵢ, Aᵢ, Bᵢ, α, β)
    end
    return C
end

@inline function __batched_gemm_cpu!(transA::AbstractChar, transB::AbstractChar, α::T,
        A::BatchedMatrix{T}, B::BatchedMatrix{T}, β::T,
        C::BatchedMatrix{T}) where {T <: LinearAlgebra.BlasFloat}
    Base.require_one_based_indexing(A)
    Base.require_one_based_indexing(B)
    Base.require_one_based_indexing(C)

    m = size(A, transA == 'N' ? 1 : 2)
    ka = size(A, transA == 'N' ? 2 : 1)
    kb = size(B, transB == 'N' ? 1 : 2)
    n = size(B, transB == 'N' ? 2 : 1)

    if ka != kb || m != size(C, 1) || n != size(C, 2)
        throw(DimensionMismatch("A1 has size ($m, $ka), B1 has size ($kb, $n), C1 has \
                                 size $(size(C))"))
    end
    LinearAlgebra.BLAS.chkstride1(A)
    LinearAlgebra.BLAS.chkstride1(B)
    LinearAlgebra.BLAS.chkstride1(C)

    n_threads = min(Threads.maxthreadid(),
        1 + max(length(A) * nbatches(A), length(B) * nbatches(B)) ÷ 8000)

    if n_threads > 1
        old_threads = BLAS.get_num_threads()
        BLAS.set_num_threads(1)

        Threads.@sync for ks in Iterators.partition(
            1:nbatches(C), cld(nbatches(C), n_threads))
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
