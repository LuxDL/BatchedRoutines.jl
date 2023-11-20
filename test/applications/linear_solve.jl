using BatchedArrays, LinearAlgebra, LinearSolve, StableRNGs, Test

function test_inf_norm(A::BatchedArray, b::BatchedArray, alg)
    prob = LinearProblem(A, b)
    sol = solve(prob, alg)
    @test sol.retcode == ReturnCode.Default || SciMLBase.successful_retcode(sol.retcode)
    @test sol.u isa BatchedArray
    @test norm((A * sol.u .- b).data, Inf) ≤ ifelse(eltype(A) == Float32, 1.0f-5, 1e-9)
end

@testset "Basic Linear Solve: $(T)" for T in (Float32, Float64)
    A = rand(StableRNG(0), T, 4, 4, 16)
    b = rand(StableRNG(0), T, 4, 16)

    @testset "solver: $(nameof(typeof(alg)))" for alg in (DirectLdiv!(), QRFactorization(),
        LUFactorization(), nothing)
        test_inf_norm(BatchedArray(A), BatchedArray(b), alg)
    end
end
