module BatchedRoutines

import PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using ADTypes: AutoFiniteDiff, AutoForwardDiff, AutoReverseDiff, AutoSparseForwardDiff,
                   AutoSparsePolyesterForwardDiff, AutoPolyesterForwardDiff, AutoZygote
    using Adapt: Adapt
    using ArrayInterface: ArrayInterface, parameterless_type
    using ChainRulesCore: ChainRulesCore, HasReverseMode, NoTangent, RuleConfig
    using ConcreteStructs: @concrete
    using FastClosures: @closure
    using FillArrays: Fill, OneElement
    using LinearAlgebra: BLAS, ColumnNorm, I, LinearAlgebra, NoPivot, RowMaximum,
                         RowNonZero, axpby!, axpy!, mul!, norm, pinv
    using LuxDeviceUtils: LuxDeviceUtils, get_device
    using Printf: @printf
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

include("internal.jl")

include("impl/batched_mul.jl")
include("impl/batched_gmres.jl")

include("chainrules.jl")

export AutoFiniteDiff, AutoForwardDiff, AutoReverseDiff, AutoZygote
export batched_adjoint, batched_gradient, batched_jacobian, batched_pickchunksize,
       batched_mul, batched_norm, batched_pinv, batched_transpose
export batchview, nbatches
export UniformBlockDiagonalMatrix

# Special Solvers
export BatchedGmresSolver, batched_gmres, batched_gmres!

end
