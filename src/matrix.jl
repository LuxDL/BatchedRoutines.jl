struct UniformBlockDiagonalMatrix{T, D <: AbstractArray{T, 3}} <: AbstractMatrix{T}
    data::D
end

function UniformBlockDiagonalMatrix(X::AbstractMatrix)
    return UniformBlockDiagonalMatrix(reshape(X, size(X, 1), size(X, 2), 1))
end

nbatches(A::UniformBlockDiagonalMatrix) = size(A.data, 3)
batchview(A::UniformBlockDiagonalMatrix) = batchview(A.data)
batchview(A::UniformBlockDiagonalMatrix, i::Int) = batchview(A.data, i)

function batched_mul(A::UniformBlockDiagonalMatrix, B::UniformBlockDiagonalMatrix)
    return UniformBlockDiagonalMatrix(batched_mul(A.data, B.data))
end

function batched_transpose(X::UniformBlockDiagonalMatrix)
    return UniformBlockDiagonalMatrix(batched_transpose(X.data))
end

Base.transpose(A::UniformBlockDiagonalMatrix) = batched_transpose(A)

function batched_adjoint(X::UniformBlockDiagonalMatrix)
    return UniformBlockDiagonalMatrix(batched_adjoint(X.data))
end

Base.adjoint(A::UniformBlockDiagonalMatrix) = batched_adjoint(A)

function batched_pinv(A::UniformBlockDiagonalMatrix)
    return UniformBlockDiagonalMatrix(batched_pinv(A.data))
end

# Adapt
function Adapt.adapt_structure(to, x::UniformBlockDiagonalMatrix)
    return UniformBlockDiagonalMatrix(Adapt.adapt(to, parent(x)))
end

# ArrayInterface
ArrayInterface.fast_matrix_colors(::Type{<:UniformBlockDiagonalMatrix}) = true
function ArrayInterface.fast_scalar_indexing(::Type{<:UniformBlockDiagonalMatrix{
        T, D}}) where {T, D}
    return ArrayInterface.fast_scalar_indexing(D)
end
function ArrayInterface.can_setindex(::Type{<:UniformBlockDiagonalMatrix{
        T, D}}) where {T, D}
    return ArrayInterface.can_setindex(D)
end

function ArrayInterface.matrix_colors(A::UniformBlockDiagonalMatrix)
    return repeat(1:size(A.data, 2), size(A.data, 3))
end

function ArrayInterface.findstructralnz(A::UniformBlockDiagonalMatrix)
    I, J, K = size(A.data)
    L = I * J * K
    i_idxs, j_idxs = Vector{Int}(undef, L), Vector{Int}(undef, L)

    @inbounds for (idx, (i, j, k)) in enumerate(Iterators.product(1:I, 1:J, 1:K))
        i_idxs[idx] = i + (k - 1) * I
        j_idxs[idx] = j + (k - 1) * J
    end

    return i_idxs, j_idxs
end

ArrayInterface.has_sparsestruct(::Type{<:UniformBlockDiagonalMatrix}) = true

# Βase
function Base.size(A::UniformBlockDiagonalMatrix)
    return (size(A.data, 1) * size(A.data, 3), size(A.data, 2) * size(A.data, 3))
end
Base.size(A::UniformBlockDiagonalMatrix, i::Int) = (size(A.data, i) * size(A.data, 3))

Base.parent(A::UniformBlockDiagonalMatrix) = A.data

Base.@propagate_inbounds function Base.getindex(
        A::UniformBlockDiagonalMatrix, i::Int, j::Int)
    i_, j_, k = _block_indices(A, i, j)
    k == -1 && return zero(eltype(A))
    return A.data[i_, j_, k]
end

Base.@propagate_inbounds function Base.setindex!(
        A::UniformBlockDiagonalMatrix, v, i::Int, j::Int)
    i_, j_, k = _block_indices(A, i, j)
    k == -1 &&
        !iszero(v) &&
        throw(ArgumentError("cannot set non-zero value outside of block."))
    A.data[i_, j_, k] = v
    return v
end

function _block_indices(A::UniformBlockDiagonalMatrix, i::Int, j::Int)
    all((0, 0) .< (i, j) .<= size(A)) || throw(BoundsError(A, (i, j)))

    M, N, _ = size(A.data)

    i_div = div(i - 1, M) + 1
    !((i_div - 1) * N + 1 ≤ j ≤ i_div * N) && return -1, -1, -1

    return mod1(i, M), mod1(j, N), i_div
end

function Base.Matrix(A::UniformBlockDiagonalMatrix)
    M = Matrix{eltype(A)}(undef, size(A, 1), size(A, 2))
    L1, L2, _ = size(A.data)
    fill!(M, false)
    for (i, Aᵢ) in enumerate(batchview(A))
        M[((i - 1) * L1 + 1):(i * L1), ((i - 1) * L2 + 1):(i * L2)] .= Aᵢ
    end
    return M
end

Base.collect(A::UniformBlockDiagonalMatrix) = Matrix(A)

function Base.similar(A::UniformBlockDiagonalMatrix, ::Type{T}) where {T}
    return UniformBlockDiagonalMatrix(similar(A.data, T))
end

Base.copy(A::UniformBlockDiagonalMatrix) = UniformBlockDiagonalMatrix(copy(A.data))

function Base.copyto!(dest::UniformBlockDiagonalMatrix, src::UniformBlockDiagonalMatrix)
    copyto!(dest.data, src.data)
    return dest
end

function Base.fill!(A::UniformBlockDiagonalMatrix, v)
    fill!(A.data, v)
    return A
end

# Broadcasting
struct UniformBlockDiagonalMatrixStyle{N} <: Broadcast.AbstractArrayStyle{2} end

function Broadcast.BroadcastStyle(
        ::UniformBlockDiagonalMatrixStyle{N}, ::Broadcast.DefaultArrayStyle{M}) where {N, M}
    return UniformBlockDiagonalMatrixStyle{max(N, M)}()
end
function Broadcast.BroadcastStyle(::Broadcast.AbstractArrayStyle{M},
        ::UniformBlockDiagonalMatrixStyle{N}) where {M, N}
    return UniformBlockDiagonalMatrixStyle{max(M, N)}()
end
function Broadcast.BroadcastStyle(::UniformBlockDiagonalMatrixStyle{M},
        ::UniformBlockDiagonalMatrixStyle{N}) where {M, N}
    return UniformBlockDiagonalMatrixStyle{max(M, N)}()
end
function Base.BroadcastStyle(::Type{<:UniformBlockDiagonalMatrix{T}}) where {T}
    return UniformBlockDiagonalMatrixStyle{-1}()
end

@inline function Base.copy(bc::Broadcast.Broadcasted{<:UniformBlockDiagonalMatrixStyle{N}}) where {N}
    bc = Broadcast.flatten(bc)
    return UniformBlockDiagonalMatrix(bc.f.(__standardize_broadcast_args(
        Val(N), bc.axes, bc.args)...))
end

@inline function Base.copyto!(dest::UniformBlockDiagonalMatrix,
        bc::Broadcast.Broadcasted{<:UniformBlockDiagonalMatrixStyle{N}}) where {N}
    bc = Broadcast.flatten(bc)
    dest.data .= bc.f.(__standardize_broadcast_args(Val(N), bc.axes, bc.args)...)
    return dest
end

@inline function Broadcast.instantiate(bc::Broadcast.Broadcasted{<:UniformBlockDiagonalMatrixStyle{N}}) where {N}
    bc = Broadcast.flatten(bc)
    axes_bc = Broadcast.combine_axes(getfield.(
        filter(Base.Fix2(isa, UniformBlockDiagonalMatrix), bc.args), :data)...)
    args = __standardize_broadcast_args(Val(N), axes_bc, bc.args)
    if bc.axes isa Nothing # Not done via dispatch to make it easier to extend instantiate(::Broadcasted{Style})
        axes = Broadcast.combine_axes(args...)
    else
        axes = bc.axes
        Broadcast.check_broadcast_axes(axes, args...)
    end
    return Broadcast.Broadcasted(bc.style, bc.f, args, axes)
end

@inline __standardize_broadcast_args(N::Val, new_axes, args::Tuple) = __standardize_broadcast_args.(
    (N,), (new_axes,), args)
for N in -1:1:3
    @eval @inline __standardize_broadcast_args(::Val{$(N)}, _, x::UniformBlockDiagonalMatrix) = x.data
end
@inline __standardize_broadcast_args(::Val{3}, _, x::AbstractArray{T, 3}) where {T} = x
@inline function __standardize_broadcast_args(
        ::Val{3}, new_axes, x::AbstractArray{T, 2}) where {T}
    I, J, K = __standardize_axes(new_axes)
    ((I == size(x, 1) || I == 1) && (J == size(x, 2) || J == 1)) &&
        return reshape(x, size(x, 1), size(x, 2), 1)
    return __standardize_broadcast_args(Val(2), new_axes, x)
end
@inline __standardize_broadcast_args(::Val, _, x::AbstractArray{T, 1}) where {T} = reshape(
    x, 1, 1, length(x))
@inline function __standardize_broadcast_args(
        ::Val{2}, new_axes, x::AbstractArray{T, 2}) where {T}
    I, J, K = __standardize_axes(new_axes)
    @assert I * K == size(x, 1) && J * K == size(x, 2)
    return mapfoldl(_cat3, 1:K; init=_init_array_prototype(x, I, J, 0)) do k
        return view(x, ((k - 1) * I + 1):(k * I), ((k - 1) * J + 1):(k * J))
    end
end
@inline function __standardize_broadcast_args(::Val{2}, new_axes, x::Fill{T, 2}) where {T}
    I, J, K = __standardize_axes(new_axes)
    @assert I * K == size(x, 1) && J * K == size(x, 2)
    return Fill(x.value, I, J, K)
end
@inline __standardize_broadcast_args(::Val, _, x) = x

@inline __standardize_axes(axes::Tuple) = __standardize_axes.(axes)
@inline __standardize_axes(axes::Base.OneTo) = axes.stop
@inline function __standardize_axes(axes::StepRange)
    @assert axes.start == 1 && axes.step == 1
    return axes.stop
end

# Common Math Operations
function Base.mapreduce(
        f::F, op::OP, A::UniformBlockDiagonalMatrix; dims=Colon(), kwargs...) where {F, OP}
    res = mapreduce(f, op, A.data; dims, kwargs...)
    dims isa Colon && return res
    return UniformBlockDiagonalMatrix(res)
end

function Base.:*(X::UniformBlockDiagonalMatrix, Y::UniformBlockDiagonalMatrix)
    return UniformBlockDiagonalMatrix(batched_mul(X.data, Y.data))
end

function Base.:*(X::UniformBlockDiagonalMatrix, Y::AbstractVector)
    return (X * reshape(Y, :, 1, nbatches(X))).data |> vec
end

function Base.:*(X::UniformBlockDiagonalMatrix, Y::AbstractMatrix)
    return X * reshape(Y, :, 1, nbatches(Y))
end

function Base.:*(X::UniformBlockDiagonalMatrix, Y::AbstractArray{T, 3}) where {T}
    return UniformBlockDiagonalMatrix(batched_mul(X.data, Y))
end

function Base.:*(X::AbstractArray{T, 3}, Y::UniformBlockDiagonalMatrix) where {T}
    return UniformBlockDiagonalMatrix(batched_mul(X, Y.data))
end

# LinearAlgebra
abstract type AbstractBatchedMatrixFactorization end

function LinearAlgebra.ldiv!(A::AbstractBatchedMatrixFactorization, b::AbstractVector)
    return LinearAlgebra.ldiv!(A, reshape(b, :, nbatches(A)))
end

function LinearAlgebra.ldiv!(
        X::AbstractVector, A::AbstractBatchedMatrixFactorization, b::AbstractVector)
    LinearAlgebra.ldiv!(reshape(X, :, nbatches(A)), A, reshape(b, :, nbatches(A)))
    return X
end

function LinearAlgebra.:\(A::AbstractBatchedMatrixFactorization, b::AbstractVector)
    X = similar(b, promote_type(eltype(A), eltype(b)), size(A, 1))
    LinearAlgebra.ldiv!(X, A, b)
    return X
end

struct GenericBatchedFactorization{A, F} <: AbstractBatchedMatrixFactorization
    alg::A
    fact::Vector{F}

    function GenericBatchedFactorization(alg::A, fact::Vector{F}) where {A, F}
        return new{A, F}(alg, fact)
    end
end

nbatches(F::GenericBatchedFactorization) = length(F.fact)
batchview(F::GenericBatchedFactorization) = F.fact
batchview(F::GenericBatchedFactorization, idx::Int) = F.fact[idx]
Base.size(F::GenericBatchedFactorization) = size(first(F.fact)) .* length(F.fact)
function Base.size(F::GenericBatchedFactorization, i::Integer)
    return size(first(F.fact), i) * length(F.fact)
end
Base.eltype(F::GenericBatchedFactorization) = eltype(first(F.fact))

LinearAlgebra.issuccess(fact::GenericBatchedFactorization) = all(issuccess, fact.fact)

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, F::GenericBatchedFactorization)
    println(io, "GenericBatchedFactorization() with Batch Count: $(nbatches(F))")
    Base.printstyled(io, "Factorization Function: "; color=:green)
    show(io, mime, F.alg)
    Base.printstyled(io, "\nPrototype Factorization: "; color=:green)
    show(io, mime, first(F.fact))
    return nothing
end

const PIVOT_TYPES = Dict(
    :qr => (:NoPivot, :ColumnNorm), :lu => (:NoPivot, :RowMaximum, :RowNonZero),
    :cholesky => (:NoPivot, :RowMaximum))

for fact in (:qr, :lu, :cholesky)
    fact! = Symbol("$(fact)!")
    @eval begin
        function LinearAlgebra.$(fact)(A::UniformBlockDiagonalMatrix, args...; kwargs...)
            return LinearAlgebra.$(fact!)(copy(A), args...; kwargs...)
        end
    end

    for pType in PIVOT_TYPES[fact]
        @eval begin
            function LinearAlgebra.$(fact!)(
                    A::UniformBlockDiagonalMatrix, pivot::$pType; kwargs...)
                fact = map(Aᵢ -> LinearAlgebra.$(fact!)(Aᵢ, pivot; kwargs...), batchview(A))
                return GenericBatchedFactorization(LinearAlgebra.$(fact!), fact)
            end

            # Needed to prevent method ambiguities
            function LinearAlgebra.$(fact)(
                    A::UniformBlockDiagonalMatrix, pivot::$pType; kwargs...)
                return LinearAlgebra.$(fact!)(copy(A), pivot; kwargs...)
            end
        end
    end
end

function LinearAlgebra.ldiv!(A::GenericBatchedFactorization, b::AbstractMatrix)
    @assert nbatches(A) == nbatches(b)
    for i in 1:nbatches(A)
        ldiv!(batchview(A, i), batchview(b, i))
    end
    return b
end

function LinearAlgebra.ldiv!(
        X::AbstractMatrix, A::GenericBatchedFactorization, b::AbstractMatrix)
    @assert nbatches(A) == nbatches(b) == nbatches(X)
    for i in 1:nbatches(A)
        LinearAlgebra.ldiv!(batchview(X, i), batchview(A, i), batchview(b, i))
    end
    return X
end

function LinearAlgebra.mul!(
        C::AbstractVector, A::UniformBlockDiagonalMatrix, B::AbstractVector)
    LinearAlgebra.mul!(reshape(C, :, 1, nbatches(A)), A, reshape(B, :, 1, nbatches(A)))
    return C
end

function LinearAlgebra.mul!(C::AbstractArray{T1, 3}, A::UniformBlockDiagonalMatrix,
        B::AbstractArray{T2, 3}) where {T1, T2}
    batched_mul!(C, A.data, B)
    return C
end
