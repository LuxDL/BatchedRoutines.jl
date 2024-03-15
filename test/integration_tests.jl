@testitem "LinearSolve" setup=[SharedTestSetup] begin
    using LinearSolve

    rng = get_stable_rng(1001)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        for dims in ((8, 8, 2), (5, 3, 2))
            A1 = UniformBlockDiagonalMatrix(rand(rng, dims...)) |> dev
            A2 = Matrix(A1) |> dev
            b = rand(rng, size(A1, 1)) |> dev
           
            prob1 = LinearProblem(A1, b)
            prob2 = LinearProblem(A2, b)

            if dims[1] == dims[2]
                solvers = [LUFactorization(), QRFactorization(), KrylovJL_GMRES()]
            else
                solvers = [QRFactorization(), KrylovJL_LSMR()]
            end

            @testset "solver: $(solver)" for solver in solvers
                x1 = solve(prob1, solver)
                x2 = solve(prob2, solver)
                @test x1.u ≈ x2.u
            end
        end
    end
end
