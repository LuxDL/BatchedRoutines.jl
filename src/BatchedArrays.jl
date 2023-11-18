module BatchedArrays

import PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using ArrayInterface, LinearAlgebra

    import Adapt
    import ConcreteStructs: @concrete
    import LinearAlgebra: BlasFloat
end

include("utils.jl")

# Low-Level possibly direct BLAS/LAPACK code
include("lowlevel/batched_mul.jl")

include("batchedarray.jl")
include("linearalgebra.jl")

export BatchedArray, BatchedVector, BatchedMatrix, BatchedVecOrMat
export nbatches, batchview

end
