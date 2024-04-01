@testitem "LinearSolve" setup=[SharedTestSetup] begin
    using LinearAlgebra, LinearSolve, Zygote

    rng = get_stable_rng(1001)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        for dims in ((8, 8, 2), (5, 3, 2))
            A1 = UniformBlockDiagonalOperator(rand(rng, dims...)) |> dev
            A2 = collect(A1)
            b = rand(rng, size(A1, 1)) |> dev

            prob1 = LinearProblem(A1, b)
            prob2 = LinearProblem(A2, b)

            if dims[1] == dims[2]
                solvers = [LUFactorization(), QRFactorization(),
                    KrylovJL_GMRES(), svd_factorization(mode), nothing]
            else
                solvers = [
                    QRFactorization(), KrylovJL_LSMR(), NormalCholeskyFactorization(),
                    QRFactorization(LinearAlgebra.ColumnNorm()),
                    svd_factorization(mode), nothing]
            end

            if dims[1] == dims[2]
                test_chainrules_adjoint = (A, b) -> sum(abs2, A \ b)

                ∂A_cr, ∂b_cr = Zygote.gradient(test_chainrules_adjoint, A1, b)
            else
                ∂A_cr, ∂b_cr = nothing, nothing
            end

            @testset "solver: $(nameof(typeof(solver)))" for solver in solvers
                # FIXME: SVD doesn't define ldiv on CUDA side
                if mode == "CUDA"
                    if solver isa SVDFactorization || (solver isa QRFactorization &&
                        solver.pivot isa LinearAlgebra.ColumnNorm)
                        # ColumnNorm is not implemented on CUDA
                        continue
                    end
                end

                x1 = solve(prob1, solver)
                if !ongpu && !(solver isa NormalCholeskyFactorization)
                    x2 = solve(prob2, solver)
                    @test x1.u ≈ x2.u
                end

                dims[1] != dims[2] && continue

                test_adjoint = function (A, b)
                    sol = solve(LinearProblem(A, b), solver)
                    return sum(abs2, sol.u)
                end

                if solver isa QRFactorization && ongpu
                    @test_broken begin
                        ∂A, ∂b = Zygote.gradient(test_adjoint, A1, b)

                        @test ∂A ≈ ∂A_cr
                        @test ∂b ≈ ∂b_cr
                    end
                else
                    ∂A, ∂b = Zygote.gradient(test_adjoint, A1, b)

                    @test ∂A ≈ ∂A_cr
                    @test ∂b ≈ ∂b_cr
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
            return sum(abs2, jac_full - target_jac)
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
            return sum(abs2, jac_full - target_jac)
        end

        @test loss_function2(model, x, target_jac, ps, st) isa Number
        @test !iszero(loss_function2(model, x, target_jac, ps, st))

        _fn_x = x -> loss_function2(model, x, cdev(target_jac), cdev(ps), st)

        ∂x_fdiff = ForwardDiff.gradient(_fn_x, cdev(x))

        _, ∂x, _, _, _ = Zygote.gradient(loss_function2, model, x, target_jac, ps, st)

        @test cdev(∂x) ≈ ∂x_fdiff
    end
end
