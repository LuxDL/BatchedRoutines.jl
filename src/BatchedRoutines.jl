module BatchedRoutines

import PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using ADTypes: AutoFiniteDiff, AutoForwardDiff, AutoSparseForwardDiff,
                   AutoSparsePolyesterForwardDiff, AutoPolyesterForwardDiff
    using ArrayInterface: ArrayInterface, parameterless_type
    using FastClosures: @closure
    using LinearAlgebra: LinearAlgebra
end

const AutoAllForwardDiff{CK} = Union{<:AutoForwardDiff{CK}, <:AutoSparseForwardDiff{CK},
    <:AutoSparsePolyesterForwardDiff{CK}, <:AutoPolyesterForwardDiff{CK}}

include("extensions.jl")  # Functions that will be defined in extensions
include("helpers.jl")

export batched_jacobian
export AutoFiniteDiff, AutoForwardDiff

end
