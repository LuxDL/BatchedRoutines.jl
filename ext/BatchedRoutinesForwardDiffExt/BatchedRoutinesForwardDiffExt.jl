module BatchedRoutinesForwardDiffExt

using ADTypes: AutoForwardDiff
using ArrayInterface: parameterless_type
using BatchedRoutines: BatchedRoutines, AbstractBatchedNonlinearAlgorithm,
                       UniformBlockDiagonalOperator, batched_jacobian, batched_mul,
                       batched_pickchunksize, _assert_type
using ChainRulesCore: ChainRulesCore
using FastClosures: @closure
using ForwardDiff: ForwardDiff, Dual
using LinearAlgebra: LinearAlgebra
using LuxDeviceUtils: LuxDeviceUtils, get_device
using SciMLBase: SciMLBase, NonlinearProblem

const CRC = ChainRulesCore

@inline BatchedRoutines._is_extension_loaded(::Val{:ForwardDiff}) = true

@inline BatchedRoutines.__can_forwarddiff_dual(::Type{T}) where {T} = ForwardDiff.can_dual(T)

include("jacobian.jl")
include("nonlinearsolve_ad.jl")

end
