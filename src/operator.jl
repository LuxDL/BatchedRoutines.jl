struct UniformBlockDiagonalOperator{T, D <: AbstractArray{T, 3}} <: AbstractSciMLOperator{T}
    data::D
end

function UniformBlockDiagonalOperator(X::AbstractMatrix)
    return UniformBlockDiagonalOperator(reshape(X, size(X, 1), size(X, 2), 1))
end

# SciMLOperators Interface
## Even though it is easily convertible, it is helpful to get warnings
SciMLOperators.isconvertible(::UniformBlockDiagonalOperator) = false

# BatchedRoutines API
getdata(op::UniformBlockDiagonalOperator) = op.data
getdata(x) = x
nbatches(op::UniformBlockDiagonalOperator) = size(op.data, 3)
batchview(op::UniformBlockDiagonalOperator) = batchview(op.data)
batchview(op::UniformBlockDiagonalOperator, i::Int) = batchview(op.data, i)

function batched_mul(op1::UniformBlockDiagonalOperator, op2::UniformBlockDiagonalOperator)
    return UniformBlockDiagonalOperator(batched_mul(op1.data, op2.data))
end

for f in (
    :batched_transpose, :batched_adjoint, :batched_inv, :batched_pinv, :batched_reshape)
    @eval function $(f)(op::UniformBlockDiagonalOperator, args...)
        return UniformBlockDiagonalOperator($(f)(op.data, args...))
    end
end

## Matrix Multiplies
@inline function Base.:*(
        op1::UniformBlockDiagonalOperator, op2::UniformBlockDiagonalOperator)
    return batched_mul(op1, op2)
end

@inline function Base.:*(op::UniformBlockDiagonalOperator, x::AbstractVector)
    return (op * reshape(x, :, 1, nbatches(op))) |> vec
end

@inline function Base.:*(op::UniformBlockDiagonalOperator, x::AbstractMatrix)
    return dropdims(op * reshape(x, :, 1, nbatches(x)); dims=2)
end

@inline function Base.:*(op::UniformBlockDiagonalOperator, x::AbstractArray{T, 3}) where {T}
    return (op * UniformBlockDiagonalOperator(x)) |> getdata
end

@inline function Base.:*(x::AbstractVector, op::UniformBlockDiagonalOperator)
    return (reshape(x, :, 1, nbatches(op)) * op) |> vec
end

@inline function Base.:*(x::AbstractMatrix, op::UniformBlockDiagonalOperator)
    return dropdims(reshape(x, :, 1, nbatches(x)) * op; dims=1)
end

@inline function Base.:*(x::AbstractArray{T, 3}, op::UniformBlockDiagonalOperator) where {T}
    return (UniformBlockDiagonalOperator(x) * op) |> getdata
end

for f in (:transpose, :adjoint)
    batched_f = Symbol("batched_", f)
    @eval (Base.$(f))(op::UniformBlockDiagonalOperator) = $(batched_f)(op)
end

@inline function Base.size(op::UniformBlockDiagonalOperator)
    N, M, B = size(op.data)
    return N * B, M * B
end
@inline Base.size(op::UniformBlockDiagonalOperator, i::Int) = size(op.data, i) *
                                                              size(op.data, 3)

@inline Base.length(op::UniformBlockDiagonalOperator) = prod(size(op))

function Base.show(io::IO, mime::MIME"text/plain", op::UniformBlockDiagonalOperator)
    print(io, "UniformBlockDiagonalOperator{$(eltype(op.data))} storing ")
    show(io, mime, op.data)
end

function Base.mapreduce(f::F, op::OP, A::UniformBlockDiagonalOperator;
        dims=Colon(), kwargs...) where {F, OP}
    res = mapreduce(f, op, getdata(A); dims, kwargs...)
    dims isa Colon && return res
    return UniformBlockDiagonalOperator(res)
end

function Base.fill!(op::UniformBlockDiagonalOperator, v)
    fill!(getdata(op), v)
    return op
end

## getindex and setindex! are supported mostly to allow finitediff to compute the jacobian
Base.@propagate_inbounds function Base.getindex(
        A::UniformBlockDiagonalOperator, i::Int, j::Int)
    i_, j_, k = _block_indices(A, i, j)
    k == -1 && return zero(eltype(A))
    return A.data[i_, j_, k]
end

Base.@propagate_inbounds function Base.getindex(A::UniformBlockDiagonalOperator, idx::Int)
    return getindex(A, mod1(idx, size(A, 1)), (idx - 1) ÷ size(A, 1) + 1)
end

Base.@propagate_inbounds function Base.setindex!(
        A::UniformBlockDiagonalOperator, v, i::Int, j::Int)
    i_, j_, k = _block_indices(A, i, j)
    k == -1 &&
        !iszero(v) &&
        throw(ArgumentError("cannot set non-zero value outside of block."))
    A.data[i_, j_, k] = v
    return v
end

Base.@propagate_inbounds function Base.setindex!(
        A::UniformBlockDiagonalOperator, v, idx::Int)
    return setindex!(A, v, mod1(idx, size(A, 1)), (idx - 1) ÷ size(A, 1) + 1)
end

function _block_indices(A::UniformBlockDiagonalOperator, i::Int, j::Int)
    all((0, 0) .< (i, j) .<= size(A)) || throw(BoundsError(A, (i, j)))

    M, N, _ = size(A.data)

    i_div = div(i - 1, M) + 1
    !((i_div - 1) * N + 1 ≤ j ≤ i_div * N) && return -1, -1, -1

    return mod1(i, M), mod1(j, N), i_div
end

@inline function Base.eachrow(X::UniformBlockDiagonalOperator)
    row_fn = @closure i -> begin
        M, N, K = size(X.data)
        k = (i - 1) ÷ M + 1
        i_ = mod1(i, M)
        data = view(X.data, i_, :, k)
        if k == 1
            return vcat(data, zeros_like(data, N * (K - 1)))
        elseif k == K
            return vcat(zeros_like(data, N * (K - 1)), data)
        else
            return vcat(zeros_like(data, N * (k - 1)), data, zeros_like(data, N * (K - k)))
        end
    end
    return map(row_fn, 1:size(X, 1))
end

## Operator --> AbstractArray
function __copyto!(A::AbstractMatrix, op::UniformBlockDiagonalOperator)
    N, M, B = size(getdata(op))
    @assert size(A) == (N * B, M * B)
    fill!(A, zero(eltype(op)))
    for (i, Aᵢ) in enumerate(batchview(op))
        A[((i - 1) * N + 1):(i * N), ((i - 1) * M + 1):(i * M)] .= convert(
            AbstractMatrix, Aᵢ)
    end
end

function Base.convert(
        ::Type{C}, op::UniformBlockDiagonalOperator) where {C <: AbstractArray}
    A = similar(op.data, size(op))
    __copyto!(A, op)
    return convert(C, A)
end

Base.Matrix(op::UniformBlockDiagonalOperator) = convert(Matrix, op)
Base.Array(op::UniformBlockDiagonalOperator) = Matrix(op)
Base.collect(op::UniformBlockDiagonalOperator) = convert(AbstractMatrix, op)

function Base.copyto!(A::AbstractArray, op::UniformBlockDiagonalOperator)
    @assert length(A) ≥ length(op)
    A_ = reshape(view(vec(A), 1:length(op)), size(op))
    __copyto!(A_, op)
    return A
end

@inline function Base.copy(op::UniformBlockDiagonalOperator)
    return UniformBlockDiagonalOperator(copy(getdata(op)))
end

## Define some of the common operations like `sum` directly since SciMLOperators doesn't
## use a very nice implemented
@inline function Base.sum(op::UniformBlockDiagonalOperator; kwargs...)
    return sum(identity, op; kwargs...)
end

@inline function Base.sum(f::F, op::UniformBlockDiagonalOperator; dims=Colon()) where {F}
    return mapreduce(f, +, op; dims)
end

## Common Operations
function Base.:+(op1::UniformBlockDiagonalOperator, op2::UniformBlockDiagonalOperator)
    return UniformBlockDiagonalOperator(getdata(op1) + getdata(op2))
end

function Base.:-(op1::UniformBlockDiagonalOperator, op2::UniformBlockDiagonalOperator)
    return UniformBlockDiagonalOperator(getdata(op1) - getdata(op2))
end

function Base.isapprox(
        op1::UniformBlockDiagonalOperator, op2::UniformBlockDiagonalOperator; kwargs...)
    return isapprox(getdata(op1), getdata(op2); kwargs...)
end

# Adapt
@inline function Adapt.adapt_structure(to, op::UniformBlockDiagonalOperator)
    return UniformBlockDiagonalOperator(Adapt.adapt(to, getdata(op)))
end

# ArrayInterface
ArrayInterface.fast_matrix_colors(::Type{<:UniformBlockDiagonalOperator}) = true
function ArrayInterface.fast_scalar_indexing(::Type{<:UniformBlockDiagonalOperator{
        T, D}}) where {T, D}
    return ArrayInterface.fast_scalar_indexing(D)
end
function ArrayInterface.can_setindex(::Type{<:UniformBlockDiagonalOperator{
        T, D}}) where {T, D}
    return ArrayInterface.can_setindex(D)
end

function ArrayInterface.matrix_colors(A::UniformBlockDiagonalOperator)
    return repeat(1:size(A.data, 2), size(A.data, 3))
end

function ArrayInterface.findstructralnz(A::UniformBlockDiagonalOperator)
    I, J, K = size(A.data)
    L = I * J * K
    i_idxs, j_idxs = Vector{Int}(undef, L), Vector{Int}(undef, L)

    @inbounds for (idx, (i, j, k)) in enumerate(Iterators.product(1:I, 1:J, 1:K))
        i_idxs[idx] = i + (k - 1) * I
        j_idxs[idx] = j + (k - 1) * J
    end

    return i_idxs, j_idxs
end

ArrayInterface.has_sparsestruct(::Type{<:UniformBlockDiagonalOperator}) = true

# Linear Algebra Routines
function LinearAlgebra.mul!(
        A::Union{AbstractMatrix, UniformBlockDiagonalOperator}, B::AbstractMatrix,
        C::UniformBlockDiagonalOperator, α::Number=true, β::Number=false)
    A_ = A isa AbstractArray ? reshape(A, :, 1, nbatches(A)) : getdata(A)
    B_ = reshape(B, :, 1, nbatches(B))
    batched_mul!(A_, B_, getdata(C), α, β)
    return A
end

function LinearAlgebra.mul!(A::Union{AbstractMatrix, UniformBlockDiagonalOperator},
        B::UniformBlockDiagonalOperator,
        C::AbstractMatrix, α::Number=true, β::Number=false)
    A_ = A isa AbstractArray ? reshape(A, :, 1, nbatches(A)) : getdata(A)
    C_ = reshape(C, :, 1, nbatches(C))
    batched_mul!(A_, getdata(B), C_, α, β)
    return A
end

function LinearAlgebra.mul!(A::Union{AbstractMatrix, UniformBlockDiagonalOperator},
        B::UniformBlockDiagonalOperator,
        C::UniformBlockDiagonalOperator, α::Number=true, β::Number=false)
    A_ = A isa AbstractArray ? reshape(A, :, 1, nbatches(A)) : getdata(A)
    batched_mul!(A_, getdata(B), getdata(C), α, β)
    return A
end

function LinearAlgebra.mul!(
        C::AbstractVector, A::UniformBlockDiagonalOperator, B::AbstractVector)
    LinearAlgebra.mul!(reshape(C, :, 1, nbatches(A)), A, reshape(B, :, 1, nbatches(A)))
    return C
end

function LinearAlgebra.mul!(C::AbstractArray{T1, 3}, A::UniformBlockDiagonalOperator,
        B::AbstractArray{T2, 3}) where {T1, T2}
    batched_mul!(C, getdata(A), B)
    return C
end

# Direct \ operator
function Base.:\(op::UniformBlockDiagonalOperator, b::AbstractVector)
    return vec(op \ reshape(b, :, nbatches(op)))
end
Base.:\(op::UniformBlockDiagonalOperator, b::AbstractMatrix) = __internal_backslash(op, b)

## This exists to allow a direct autodiff through the code. eg, for non-square systems
@inline function __internal_backslash(op::UniformBlockDiagonalOperator, b::AbstractMatrix)
    size(op, 1) != length(b) && throw(DimensionMismatch("size(op, 1) != length(b)"))
    return mapfoldl(((Aᵢ, bᵢ),) -> Aᵢ \ bᵢ, hcat, zip(batchview(op), batchview(b)))
end
