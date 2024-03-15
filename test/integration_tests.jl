@testitem "LinearSolve" setup=[SharedTestSetup] begin
    using LinearSolve

    rng = get_stable_rng(1001)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        A1 = UniformBlockDiagonalMatrix(rand(rng, 32, 32, 8)) |> dev
        A2 = Matrix(A1) |> dev
        b = rand(rng, size(A1, 2)) |> dev

        prob1 = LinearProblem(A1, b)
        prob2 = LinearProblem(A2, b)

        @test solve(prob1, LUFactorization()).u ≈ solve(prob2, LUFactorization()).u
        @test solve(prob1, QRFactorization()).u ≈ solve(prob2, QRFactorization()).u
        @test solve(prob1, KrylovJL_GMRES()).u ≈ solve(prob2, KrylovJL_GMRES()).u
    end
end
