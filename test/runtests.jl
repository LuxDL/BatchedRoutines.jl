using SafeTestsets, Test

@testset "BatchedArrays.jl Tests" begin
    @testset "Applications" begin
        @safetestset "Linear Solve" begin
            include("applications/linear_solve.jl")
        end
    end
end
