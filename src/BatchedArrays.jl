module BatchedArrays

import PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using ArrayInterface, ConcreteStructs, LinearAlgebra

    import ArrayInterface: qr_instance
    import Polyester: @batch
end

include("batchedarray.jl")
include("linearalgebra.jl")

export BatchedArray
export nbatches, batchview

end
