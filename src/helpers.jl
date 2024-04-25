@generated function _get_chunksize(ad::AutoAllForwardDiff{CK}, u::AbstractArray) where {CK}
    (CK === nothing || CK ≤ 0) && return :(batched_pickchunksize(u))
    return :($(CK))
end

Base.@assume_effects :total _assert_type(x)=_assert_type(typeof(x))
Base.@assume_effects :total _assert_type(::Type)=true

"""
    _storage_type(A) -> Type

Removes all wrappers to return the `Array` or `CuArray` (or whatever) type within.

```julia-repl
julia> view(reshape(ones(10)', 2, 5), :, 3:4) |> _storage_type
Vector{Float64} (alias for Array{Float64, 1})

julia> reshape(sparse(rand(10)), 5, 2) |> _storage_type
SparseVector{Float64, Int64}
```
"""
function _storage_type(A::AbstractArray)
    P = parent(A)
    return typeof(A) === typeof(P) ? typeof(A) : _storage_type(P)
end
_storage_type(A) = typeof(A)

"""
    _storage_typejoin(A, B, C, ...) -> Type

Reduces with `Base.promote_typejoin`, in order that this conveys useful information for
dispatching to BLAS. It does not tell you what container to allocate:

```julia-repl
julia> _storage_typejoin(rand(2), rand(Float32, 2))
Vector (alias for Array{T, 1} where T)

julia> eltype(ans) <: LinearAlgebra.BlasFloat
false

julia> _storage_typejoin(rand(2), rand(2, 3), rand(2, 3, 4))
Array{Float64}
```
"""
_storage_typejoin(A, Bs...) = Base.promote_typejoin(
    _storage_type(A), _storage_typejoin(Bs...))
_storage_typejoin(A) = _storage_type(A)

"""
    _is_strided(A::AbstractArray) -> Bool

This generalises `A isa StridedArray` to treat wrappers like `A::PermutedDimsArray`, for
which it returns `is_strided(parent(A))`.

It returns `true` for `CuArray`s, and `PermutedDimsArray`s of those.

Other wrappers (defined outside Base, LinearAlgebra) are assumed not to break strided-ness,
and hence also return `is_strided(parent(A))`. This correctly handles things like
`NamedDimsArray` wihch don't alter indexing. However, it's a little pessimistic in that e.g.
a `view` of such a container will return `false`, even in cases where the same `view` of
`parent(A)` would be a `StridedArray`.
"""
_is_strided(::StridedArray) = true
_is_strided(A) = false
function _is_strided(A::AbstractArray)
    M = parentmodule(typeof(A))
    if parent(A) === A # SparseMatrix, StaticArray, etc
        return false
    elseif M === Base || M === Core || M === LinearAlgebra
        # bad reshapes, etc, plus Diagonal, UpperTriangular, etc.
        return false
    else
        return _is_strided(parent(A)) # PermutedDimsArray, NamedDimsArray
    end
end
_is_strided(A::LinearAlgebra.Transpose) = _is_strided(parent(A))
_is_strided(A::LinearAlgebra.Adjoint) = eltype(A) <: Real && _is_strided(parent(A))

"""
    __copy_if_strided(X::AbstractArray) -> AbstractArray

Make a contiguous copy of `X` if it is strided. This is useful for BLAS and LAPACK
dispatches that require contiguous arrays.
"""
function _copy_if_strided(X::Union{BatchedMatrix{T}, BatchedVector{T}}) where {T}
    _is_strided(X) || return X
    if stride(X, 3) == 1 && stride(X, 1) != 1
        @debug "copying to avoid generic strided fallbacks!" typeof(X) size(X) strides(X)
        return copy(X)
    end
    return X
end

@inline _init_array_prototype(X::AbstractArray, lengths::Int...) = similar(X, lengths...)

@inline _cat1(x, y) = vcat(x, y)
@inline _cat2(x, y) = hcat(x, y)
for i in 3:10
    fname = Symbol("_cat", i)
    @eval @inline $(fname)(x, y) = cat(x, y; dims=$(Val(i)))
end

@generated function _batched_map(f::F, X::AbstractArray{T, N}) where {F, T, N}
    _cat_fn = Symbol("_cat", N)
    return quote
        X₁, Xᵣ = Iterators.peel(batchview(X))
        _proto = f(X₁)
        proto = reshape(_proto, size(_proto)..., 1)
        return mapfoldl(f, $(_cat_fn), Xᵣ; init=proto)
    end
end

@inline _array_on_cpu(x) = _array_on_cpu(get_device(x))
@inline _array_on_cpu(x::LuxDeviceUtils.LuxCPUDevice) = true
@inline _array_on_cpu(x::LuxDeviceUtils.AbstractLuxGPUDevice) = false

# Useful for computing the gradient of a gradient
function _jacobian_vector_product end
function _vector_jacobian_product end
function _batched_jacobian end
function _batched_gradient end
function _construct_jvp_duals end

@inline _restructure(y, x) = reshape(y, size(x))

# Test Loaded AD Backend
_assert_loaded_backend(::AutoForwardDiff) = @assert _is_extension_loaded(Val(:ForwardDiff))
_assert_loaded_backend(::AutoReverseDiff) = @assert _is_extension_loaded(Val(:ReverseDiff))
_assert_loaded_backend(::AutoFiniteDiff) = @assert _is_extension_loaded(Val(:FiniteDiff))
_assert_loaded_backend(::AutoZygote) = @assert _is_extension_loaded(Val(:Zygote))

CRC.@non_differentiable _assert_loaded_backend(::Any...)

# Chunksize remove
_maybe_remove_chunksize(ad, x) = ad
function _maybe_remove_chunksize(ad::AutoAllForwardDiff{CK}, x) where {CK}
    (CK === nothing || CK ≤ 0 || CK ≤ length(x)) && return ad
    return parameterless_type(ad)()
end

# Figure out the type of the gradient
@inline function _resolve_gradient_type(f::F, g::G, x, ::Val{depth}) where {F, G, depth}
    Base.issingletontype(f) && return (eltype(x), false)
    return promote_type(eltype(x), eltype(g(x))), true
end
@inline function _resolve_gradient_type(
        f::Union{Base.Fix1, Base.Fix2}, g::G, x, ::Val{depth}) where {G, depth}
    depth ≥ 5 && return promote_type(eltype(x), eltype(f(x))), true
    T, resolved = _resolve_gradient_type(f.f, g, x, Val(depth + 1))
    resolved && return T, true
    return promote_type(T, eltype(f.x)), false
end

# MLUtils.jl has too many unwanted dependencies
@inline fill_like(x::AbstractArray, v, ::Type{T}, dims...) where {T} = fill!(
    similar(x, T, dims...), v)
@inline fill_like(x::AbstractArray, v, dims...) = fill_like(x, v, eltype(x), dims...)

@inline zeros_like(x::AbstractArray, ::Type{T}, dims...) where {T} = fill_like(
    x, zero(T), T, dims...)
@inline zeros_like(x::AbstractArray, dims...) = zeros_like(x, eltype(x), dims...)

@inline ones_like(x::AbstractArray, ::Type{T}, dims...) where {T} = fill_like(
    x, one(T), T, dims...)
@inline ones_like(x::AbstractArray, dims...) = ones_like(x, eltype(x), dims...)

CRC.@non_differentiable fill_like(::Any...)
CRC.@non_differentiable zeros_like(::Any...)
CRC.@non_differentiable ones_like(::Any...)

@inline _wrap_batched_operator(x::AbstractArray{T, 3}) where {T} = UniformBlockDiagonalOperator(x)
@inline _wrap_batched_operator(x::UniformBlockDiagonalOperator) = x
