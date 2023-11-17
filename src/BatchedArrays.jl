module BatchedArrays

import PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using ArrayInterface, LinearAlgebra

    import Adapt
    import ConcreteStructs: @concrete
end

include("batchedarray.jl")
include("linearalgebra.jl")

export BatchedArray, BatchedVector, BatchedMatrix, BatchedVecOrMat
export nbatches, batchview

end
