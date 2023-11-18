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
