"""
    __storage_type(A) -> Type

Removes all wrappers to return the `Array` or `CuArray` (or whatever) type within.

```julia-repl
julia> view(reshape(ones(10)', 2, 5), :, 3:4) |> __storage_type
Vector{Float64} (alias for Array{Float64, 1})

julia> reshape(sparse(rand(10)), 5, 2) |> __storage_type
SparseVector{Float64, Int64}
```
"""
function __storage_type(A::AbstractArray)
    P = parent(A)
    return typeof(A) === typeof(P) ? typeof(A) : __storage_type(P)
end
__storage_type(A) = typeof(A)
__storage_type(A::BatchedArray) = __storage_type(A.data)

"""
    __storage_typejoin(A, B, C, ...) -> Type

Reduces with `Base.promote_typejoin`, in order that this conveys useful information for
dispatching to BLAS. It does not tell you what container to allocate:

```julia-repl
julia> __storage_typejoin(rand(2), rand(Float32, 2))
Vector (alias for Array{T, 1} where T)

julia> eltype(ans) <: LinearAlgebra.BlasFloat
false

julia> __storage_typejoin(rand(2), rand(2, 3), rand(2, 3, 4))
Array{Float64}
```
"""
__storage_typejoin(A, Bs...) = Base.promote_typejoin(__storage_type(A),
    __storage_typejoin(Bs...))
__storage_typejoin(A) = __storage_type(A)

"""
    __is_strided(A::AbstractArray) -> Bool

This generalises `A isa StridedArray` to treat wrappers like `A::PermutedDimsArray`, for
which it returns `is_strided(parent(A))`.

It returns `true` for `CuArray`s, and `PermutedDimsArray`s of those.

Other wrappers (defined outside Base, LinearAlgebra) are assumed not to break strided-ness,
and hence also return `is_strided(parent(A))`. This correctly handles things like
`NamedDimsArray` wihch don't alter indexing. However, it's a little pessimistic in that e.g.
a `view` of such a container will return `false`, even in cases where the same `view` of
`parent(A)` would be a `StridedArray`.
"""
__is_strided(::StridedArray) = true
__is_strided(A) = false
function __is_strided(A::AbstractArray)
    M = parentmodule(typeof(A))
    if parent(A) === A # SparseMatrix, StaticArray, etc
        return false
    elseif M === Base || M === Core || M === LinearAlgebra
        # bad reshapes, etc, plus Diagonal, UpperTriangular, etc.
        return false
    else
        return __is_strided(parent(A)) # PermutedDimsArray, NamedDimsArray
    end
end
__is_strided(A::BatchedArray) = __is_strided(A.data)
__is_strided(A::LinearAlgebra.Transpose) = __is_strided(parent(A))
__is_strided(A::LinearAlgebra.Adjoint) = eltype(A) <: Real && __is_strided(parent(A))

"""
    __copy_if_strided(X::AbstractArray) -> AbstractArray

Make a contiguous copy of `X` if it is strided. This is useful for BLAS and LAPACK
dispatches that require contiguous arrays.
"""
function __copy_if_strided(X::Union{AbstractArray{T, 3}, BatchedMatrix{T}}) where {T}
    __is_strided(X) || return X
    if stride(X, 3) == 1 && stride(X, 1) != 1
        @debug "copying to avoid generic strided fallbacks!" typeof(X) size(X) strides(X)
        return copy(X)
    end
    return X
end
