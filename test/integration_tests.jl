@testitem "LinearSolve" setup=[SharedTestSetup] begin
    using FiniteDiff, LinearSolve, Zygote

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

                test_adjoint = function (A, b)
                    sol = solve(LinearProblem(A, b), solver)
                    return sum(abs2, sol.u)
                end

                dims[1] != dims[2] && continue

                ∂A_fd = FiniteDiff.finite_difference_gradient(
                    x -> test_adjoint(x, Array(b)), Array(A1))
                ∂b_fd = FiniteDiff.finite_difference_gradient(
                    x -> test_adjoint(Array(A1), x), Array(b))

                if solver isa QRFactorization && ongpu
                    @test_broken begin
                        ∂A, ∂b = Zygote.gradient(test_adjoint, A1, b)

                        @test Array(∂A)≈∂A_fd atol=1e-1 rtol=1e-1
                        @test Array(∂b)≈∂b_fd atol=1e-1 rtol=1e-1
                    end
                else
                    ∂A, ∂b = Zygote.gradient(test_adjoint, A1, b)

                    @test Array(∂A)≈∂A_fd atol=1e-1 rtol=1e-1
                    @test Array(∂b)≈∂b_fd atol=1e-1 rtol=1e-1
                end
            end
        end
    end
end
