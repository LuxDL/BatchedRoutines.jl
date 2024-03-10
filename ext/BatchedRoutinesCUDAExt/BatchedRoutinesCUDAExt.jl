module BatchedRoutinesCUDAExt

using BatchedRoutines: BatchedRoutines
using CUDA: CUBLAS, CUDA, CuArray

const CuBlasFloat = Union{Float16, Float32, Float64, ComplexF32, ComplexF64}

include("batched_mul.jl")

end
