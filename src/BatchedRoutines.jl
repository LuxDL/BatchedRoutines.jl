module BatchedRoutines

import PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using ADTypes: AutoForwardDiff
    using ArrayInterface: ArrayInterface, parameterless_type
    using FastClosures: @closure
    using LinearAlgebra: LinearAlgebra
end

include("extensions.jl")  # Functions that will be defined in extensions

export AutoForwardDiff

end
