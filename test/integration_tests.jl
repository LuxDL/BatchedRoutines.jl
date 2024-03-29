@testitem "LinearSolve" setup=[SharedTestSetup] begin
    using FiniteDiff, LinearSolve, Zygote

    rng = get_stable_rng(1001)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        for dims in ((8, 8, 2), (5, 3, 2))
            A1 = UniformBlockDiagonalOperator(rand(rng, dims...)) |> dev
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

@testitem "Simple Lux Integration" setup=[SharedTestSetup] begin
    using ComponentArrays, ForwardDiff, Lux, Random, Zygote

    rng = get_stable_rng(1001)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        model = Chain(Dense(4 => 6, tanh), Dense(6 => 3))
        ps, st = Lux.setup(rng, model)
        ps = ComponentArray(ps) |> dev
        st = st |> dev

        x = randn(rng, 4, 3) |> dev
        y = randn(rng, 4, 3) |> dev
        target_jac = batched_jacobian(
            AutoForwardDiff(; chunksize=4), StatefulLuxLayer(model, nothing, st), y, ps)

        loss_function = (model, x, target_jac, ps, st) -> begin
            m = StatefulLuxLayer(model, nothing, st)
            jac_full = batched_jacobian(AutoForwardDiff(; chunksize=4), m, x, ps)
            return sum(abs2, jac_full .- target_jac)
        end

        @test loss_function(model, x, target_jac, ps, st) isa Number
        @test !iszero(loss_function(model, x, target_jac, ps, st))

        cdev = cpu_device()
        _fn_x = x -> loss_function(model, x, target_jac |> cdev, ps |> cdev, st)
        _fn_ps = p -> loss_function(
            model, x |> cdev, target_jac |> cdev, ComponentArray(p, getaxes(ps)), st)

        ∂x_fdiff = ForwardDiff.gradient(_fn_x, cdev(x))
        ∂ps_fdiff = ForwardDiff.gradient(_fn_ps, cdev(ps))

        _, ∂x, _, ∂ps, _ = Zygote.gradient(loss_function, model, x, target_jac, ps, st)

        @test cdev(∂x) ≈ ∂x_fdiff
        @test cdev(∂ps) ≈ ∂ps_fdiff

        loss_function2 = (model, x, target_jac, ps, st) -> begin
            m = StatefulLuxLayer(model, ps, st)
            jac_full = batched_jacobian(AutoForwardDiff(; chunksize=4), m, x)
            return sum(abs2, jac_full .- target_jac)
        end

        @test loss_function2(model, x, target_jac, ps, st) isa Number
        @test !iszero(loss_function2(model, x, target_jac, ps, st))

        _fn_x = x -> loss_function2(model, x, cdev(target_jac), cdev(ps), st)

        ∂x_fdiff = ForwardDiff.gradient(_fn_x, cdev(x))

        _, ∂x, _, _, _ = Zygote.gradient(loss_function2, model, x, target_jac, ps, st)

        @test cdev(∂x) ≈ ∂x_fdiff
    end
end
