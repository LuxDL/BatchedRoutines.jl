@testitem "Batched Jacobians" setup=[SharedTestSetup] begin
    using FiniteDiff, ForwardDiff, ReverseDiff, Zygote

    rng = get_stable_rng(1001)

    @testset "$mode" for (mode, aType, device, ongpu) in MODES
        simple_batched_function = function (X, p)
            X_ = reshape(X, :, nbatches(X))
            return sum(abs2, X_ .* p; dims=1) .- sum(abs, X_ .* p; dims=1) .+ p .^ 2
        end

        X = randn(rng, 3, 2, 4) |> aType
        p = randn(rng, 6) |> aType

        J_fdiff = batched_jacobian(
            AutoFiniteDiff(), simple_batched_function, Array(X), Array(p))
        J_fwdiff = batched_jacobian(AutoForwardDiff(), simple_batched_function, X, p)

        @test Matrix(J_fdiff)≈Matrix(J_fwdiff) atol=1e-3

        X = randn(rng, 2, 4) |> aType
        p = randn(rng, 2) |> aType

        J_fdiff = batched_jacobian(
            AutoFiniteDiff(), simple_batched_function, Array(X), Array(p))
        J_fwdiff = batched_jacobian(AutoForwardDiff(), simple_batched_function, X, p)

        @test Matrix(J_fdiff)≈Matrix(J_fwdiff) atol=1e-3

        X = randn(rng, 3) |> aType
        p = randn(rng, 3) |> aType

        J_fdiff = batched_jacobian(
            AutoFiniteDiff(), simple_batched_function, Array(X), Array(p))
        J_fwdiff = batched_jacobian(AutoForwardDiff(), simple_batched_function, X, p)

        @test Matrix(J_fdiff)≈Matrix(J_fwdiff) atol=1e-3
    end
end
