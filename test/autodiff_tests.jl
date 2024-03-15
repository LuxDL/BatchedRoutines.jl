@testitem "Batched Jacobians" setup=[SharedTestSetup] begin
    using FiniteDiff, ForwardDiff, ReverseDiff, Zygote

    rng = get_stable_rng(1001)

    @testset "$mode" for (mode, aType, device, ongpu) in MODES
        simple_batched_function = function (X, p)
            X_ = reshape(X, :, nbatches(X))
            return sum(abs2, X_ .* p; dims=1) .- sum(abs, X_ .* p; dims=1) .+ p .^ 2
        end

        Xs = (aType(randn(rng, 3, 2, 4)), aType(randn(rng, 2, 4)), aType(randn(rng, 3)))
        ps = (aType(randn(rng, 6)), aType(randn(rng, 2)), aType(randn(rng, 3)))

        for (X, p) in zip(Xs, ps)
            J_fdiff = batched_jacobian(
                AutoFiniteDiff(), simple_batched_function, Array(X), Array(p))
            J_fwdiff = batched_jacobian(AutoForwardDiff(), simple_batched_function, X, p)
            J_fwdiff2 = batched_jacobian(
                AutoForwardDiff(; chunksize=2), simple_batched_function, X, p)

            @test Matrix(J_fdiff)≈Matrix(J_fwdiff) atol=1e-3
            @test Matrix(J_fwdiff) ≈ Matrix(J_fwdiff2)
        end
    end
end

@testitem "Gradient" setup=[SharedTestSetup] begin
    using FiniteDiff, ForwardDiff, ReverseDiff, Zygote

    rng = get_stable_rng(1001)

    @testset "$mode" for (mode, aType, device, ongpu) in MODES
        simple_batched_function = function (X, p)
            X_ = reshape(X, :, nbatches(X))
            return sum(
                abs2, sum(abs2, X_ .* p; dims=1) .- sum(abs, X_ .* p; dims=1) .+ p .^ 2)
        end

        Xs = (aType(randn(rng, 3, 2, 4)), aType(randn(rng, 2, 4)), aType(randn(rng, 3)))
        ps = (aType(randn(rng, 6)), aType(randn(rng, 2)), aType(randn(rng, 3)))

        for (X, p) in zip(Xs, ps)
            gs_fdiff = batched_gradient(
                AutoFiniteDiff(), simple_batched_function, Array(X), Array(p))
            gs_fwdiff = batched_gradient(AutoForwardDiff(), simple_batched_function, X, p)
            gs_rdiff = batched_gradient(
                AutoReverseDiff(), simple_batched_function, Array(X), Array(p))
            gs_zygote = batched_gradient(AutoZygote(), simple_batched_function, X, p)

            @test Array(gs_fdiff)≈Array(gs_fwdiff) atol=1e-3
            @test Array(gs_fwdiff)≈Array(gs_rdiff) atol=1e-3
            @test Array(gs_rdiff)≈Array(gs_zygote) atol=1e-3
        end

        @testset "Gradient of Gradient" begin
            for (X, p) in zip(Xs, ps)
                gs_fwddiff_x = ForwardDiff.gradient(
                    x -> sum(abs2,
                        batched_gradient(
                            AutoZygote(), simple_batched_function, x, Array(p))),
                    Array(X))
                gs_fwddiff_p = ForwardDiff.gradient(
                    p -> sum(abs2,
                        batched_gradient(
                            AutoZygote(), simple_batched_function, Array(X), p)),
                    Array(p))

                @testset "backend: $(backend)" for backend in (
                    AutoFiniteDiff(), AutoForwardDiff(),
                    AutoForwardDiff(; chunksize=3), AutoReverseDiff(), AutoZygote())
                    (!(backend isa AutoZygote) && ongpu) && continue
                    atol = backend isa AutoFiniteDiff ? 1e-1 : 1e-3
                    rtol = backend isa AutoFiniteDiff ? 1e-1 : 1e-3

                    __f = (x, p) -> sum(
                        abs2, batched_gradient(backend, simple_batched_function, x, p))

                    gs_zyg = Zygote.gradient(__f, X, p)
                    gs_rdiff = ReverseDiff.gradient(__f, (Array(X), Array(p)))

                    @test Array(gs_fwddiff_x)≈Array(gs_zyg[1]) atol=atol rtol=rtol
                    @test Array(gs_fwddiff_p)≈Array(gs_zyg[2]) atol=atol rtol=rtol
                    @test Array(gs_fwddiff_x)≈Array(gs_rdiff[1]) atol=atol rtol=rtol
                    @test Array(gs_fwddiff_p)≈Array(gs_rdiff[2]) atol=atol rtol=rtol

                    __f1 = x -> sum(
                        abs2, batched_gradient(backend, simple_batched_function, x, p))
                    __f2 = x -> sum(abs2,
                        batched_gradient(backend, simple_batched_function, x, Array(p)))

                    gs_zyg_x = only(Zygote.gradient(__f1, X))
                    gs_rdiff_x = ReverseDiff.gradient(__f2, Array(X))

                    @test Array(gs_zyg_x)≈Array(gs_fwddiff_x) atol=atol rtol=rtol
                    @test Array(gs_rdiff_x)≈Array(gs_fwddiff_x) atol=atol rtol=rtol
                end
            end
        end
    end
end
