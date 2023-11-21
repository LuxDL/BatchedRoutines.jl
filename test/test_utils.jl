using BatchedArrays, LuxCUDA, LuxDeviceUtils
using LuxTestUtils: @jet, check_approx

CUDA.allowscalar(false)

if !@isdefined(GROUP)
    const GROUP = get(ENV, "GROUP", "All")

    cpu_testing() = GROUP == "All" || GROUP == "CPU"
    cuda_testing() = (GROUP == "All" || GROUP == "CUDA") && LuxCUDA.functional()

    const MODES = begin
        # Mode, Array Type, Device Function, GPU?
        cpu_mode = ("CPU", Array, LuxCPUDevice(), false)
        cuda_mode = ("CUDA", CuArray, LuxCUDADevice(), true)

        modes = []
        cpu_testing() && push!(modes, cpu_mode)
        cuda_testing() && push!(modes, cuda_mode)

        modes
    end
end
