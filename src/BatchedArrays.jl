module BatchedArrays

import PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using ArrayInterface, LinearAlgebra, Statistics

    import Adapt
    import ConcreteStructs: @concrete
    import LinearAlgebra: BlasFloat, BlasInt
end

include("batchedarray.jl")
include("batchedscalar.jl")
include("utils.jl")

# Low-Level possibly direct BLAS/LAPACK code
include("lowlevel/batched_mul.jl")

include("linearalgebra.jl")

export BatchedArray, BatchedVector, BatchedMatrix, BatchedVecOrMat
export BatchedScalar
export nbatches, batchview

end
