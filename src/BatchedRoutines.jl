module BatchedRoutines

import PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using ADTypes: AutoFiniteDiff, AutoForwardDiff, AutoSparseForwardDiff,
                   AutoSparsePolyesterForwardDiff, AutoPolyesterForwardDiff
    using Adapt: Adapt
    using ArrayInterface: ArrayInterface, parameterless_type
    using ChainRulesCore: ChainRulesCore
    using ConcreteStructs: @concrete
    using FastClosures: @closure
    using LinearAlgebra: BLAS, LinearAlgebra, mul!
end

const CRC = ChainRulesCore

const AutoAllForwardDiff{CK} = Union{<:AutoForwardDiff{CK}, <:AutoSparseForwardDiff{CK},
    <:AutoSparsePolyesterForwardDiff{CK}, <:AutoPolyesterForwardDiff{CK}}

const BatchedVector{T} = AbstractMatrix{T}
const BatchedMatrix{T} = AbstractArray{T, 3}

@inline _is_extension_loaded(::Val) = false

include("api.jl")
include("helpers.jl")
include("matrix.jl")

include("impl/batched_mul.jl")

export AutoFiniteDiff, AutoForwardDiff
export batched_jacobian, batched_pickchunksize, batched_mul
export batchview, nbatches
export UniformBlockDiagonalMatrix

end
