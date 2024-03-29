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
    using LinearAlgebra: BLAS, ColumnNorm, LinearAlgebra, NoPivot, RowMaximum, RowNonZero,
                         mul!, pinv
    using LuxDeviceUtils: LuxDeviceUtils, get_device
    using SciMLOperators: AbstractSciMLOperator
end

function __init__()
    @static if isdefined(Base.Experimental, :register_error_hint)
        Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
            if any(Base.Fix2(isa, UniformBlockDiagonalOperator), exc.args)
                print(io, "\nHINT: ")
                printstyled(
                    io, "`UniformBlockDiagonalOperator` doesn't support AbstractArray \
                         operations. If you want this supported open an issue at \
                         https://github.com/LuxDL/BatchedRoutines.jl to discuss it.";
                    color=:cyan)
            end
        end
    end
end

const CRC = ChainRulesCore

const AutoAllForwardDiff{CK} = Union{<:AutoForwardDiff{CK}, <:AutoSparseForwardDiff{CK},
    <:AutoSparsePolyesterForwardDiff{CK}, <:AutoPolyesterForwardDiff{CK}}

const BatchedVector{T} = AbstractMatrix{T}
const BatchedMatrix{T} = AbstractArray{T, 3}

@inline _is_extension_loaded(::Val) = false

include("api.jl")
include("helpers.jl")

include("operator.jl")
include("factorization.jl")

include("impl/batched_mul.jl")
include("impl/batched_gmres.jl")

include("chainrules.jl")

export AutoFiniteDiff, AutoForwardDiff, AutoReverseDiff, AutoZygote
export batched_adjoint, batched_gradient, batched_jacobian, batched_pickchunksize,
       batched_mul, batched_pinv, batched_transpose
export batchview, nbatches
export UniformBlockDiagonalOperator

end
