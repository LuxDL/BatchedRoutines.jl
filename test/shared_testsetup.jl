@testsetup module SharedTestSetup

using LuxCUDA, LuxDeviceUtils, Random, StableRNGs
import LuxTestUtils: @jet, @test_gradients, check_approx

const GROUP = get(ENV, "GROUP", "All")

CUDA.allowscalar(false)

cpu_testing() = GROUP == "All" || GROUP == "CPU"
cuda_testing() = (GROUP == "All" || GROUP == "CUDA") && CUDA.functional()

const MODES = begin
    # Mode, Array Type, Device Function, GPU?
    cpu_mode = ("CPU", Array, LuxCPUDevice(), false)
    cuda_mode = ("CUDA", CuArray, LuxCUDADevice(), true)

    modes = []
    cpu_testing() && push!(modes, cpu_mode)
    cuda_testing() && push!(modes, cuda_mode)

    modes
end

# Some Helper Functions
function get_default_rng(mode::String)
    dev = mode == "CPU" ? LuxCPUDevice() :
          mode == "CUDA" ? LuxCUDADevice() : mode == "AMDGPU" ? LuxAMDGPUDevice() : nothing
    rng = default_device_rng(dev)
    return rng isa TaskLocalRNG ? copy(rng) : deepcopy(rng)
end

get_stable_rng(seed=12345) = StableRNG(seed)

# SVD Helper till https://github.com/SciML/LinearSolve.jl/issues/488 is resolved
using LinearSolve: LinearSolve

function svd_factorization(mode)
    mode == "CPU" && return LinearSolve.SVDFactorization()
    mode == "CUDA" &&
        return LinearSolve.SVDFactorization(true, CUDA.CUSOLVER.JacobiAlgorithm())
    error("Unsupported mode: $mode")
end

export @jet, @test_gradients, check_approx
export GROUP, MODES, cpu_testing, cuda_testing, get_default_rng, get_stable_rng
export svd_factorization

end
