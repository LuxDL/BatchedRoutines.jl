@testitem "Batched Nonlinear Solvers" setup=[SharedTestSetup] begin
    using Chairmarks, ForwardDiff, SciMLBase, SimpleNonlinearSolve, Statistics, Zygote

    testing_f(u, p) = u .^ 2 .+ u .^ 3 .- u .- p

    u0 = rand(3, 128)
    p = rand(1, 128)

    prob = NonlinearProblem(testing_f, u0, p)

    sol_nlsolve = solve(prob, SimpleNewtonRaphson())
    sol_batched = solve(prob, BatchedSimpleNewtonRaphson())

    @test abs.(sol_nlsolve.u) ≈ abs.(sol_batched.u)

    nlsolve_timing = @be solve($prob, $SimpleNewtonRaphson())
    batched_timing = @be solve($prob, $BatchedSimpleNewtonRaphson())

    @info "SimpleNonlinearSolve Timing: $(median(nlsolve_timing))."
    @info "BatchedSimpleNewtonRaphson Timing: $(median(batched_timing))."

    ∂p1 = ForwardDiff.gradient(p) do p
        prob = NonlinearProblem(testing_f, u0, p)
        return sum(abs2, solve(prob, SimpleNewtonRaphson()).u)
    end

    ∂p2 = ForwardDiff.gradient(p) do p
        prob = NonlinearProblem(testing_f, u0, p)
        return sum(abs2, solve(prob, BatchedSimpleNewtonRaphson()).u)
    end

    @test ∂p1 ≈ ∂p2

    fwdiff_nlsolve_timing = @be ForwardDiff.gradient($p) do p
        prob = NonlinearProblem(testing_f, u0, p)
        return sum(abs2, solve(prob, SimpleNewtonRaphson()).u)
    end

    fwdiff_batched_timing = @be ForwardDiff.gradient($p) do p
        prob = NonlinearProblem(testing_f, u0, p)
        return sum(abs2, solve(prob, BatchedSimpleNewtonRaphson()).u)
    end

    @info "ForwardDiff SimpleNonlinearSolve Timing: $(median(fwdiff_nlsolve_timing))."
    @info "ForwardDiff BatchedNonlinearSolve Timing: $(median(fwdiff_batched_timing))."
end
