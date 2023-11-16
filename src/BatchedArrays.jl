module BatchedArrays

import PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using ConcreteStructs, LinearAlgebra
end

include("batchedarray.jl")
include("linearalgebra.jl")

export BatchedArray

end
