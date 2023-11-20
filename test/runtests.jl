using SafeTestsets, Test

# TODO: Device agnostic tests like lux
# TODO: JET Testing
# TODO: Use Groups for faster testing

@testset "BatchedArrays.jl Tests" begin
    @testset "Applications" begin
        @safetestset "Linear Solve" begin
            include("applications/linear_solve.jl")
        end
    end
end
