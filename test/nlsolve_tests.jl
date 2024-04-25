@testitem "Batched Nonlinear Solvers" setup=[SharedTestSetup] begin
    using Chairmarks, ForwardDiff, LinearSolve, SciMLBase, SciMLSensitivity,
          SimpleNonlinearSolve, Statistics, Zygote

    testing_f(u, p) = u .^ 2 .+ u .^ 3 .- u .- p

    u0 = rand(3, 128)
    p = rand(1, 128)

    prob = NonlinearProblem(testing_f, u0, p)

    sol_nlsolve = solve(prob, SimpleNewtonRaphson())
    sol_batched = solve(prob, BatchedSimpleNewtonRaphson())

    @test abs.(sol_nlsolve.u) ≈ abs.(sol_batched.u)

    nlsolve_timing = @be solve($prob, $SimpleNewtonRaphson())
    batched_timing = @be solve($prob, $BatchedSimpleNewtonRaphson())

    println("SimpleNonlinearSolve Timing: $(median(nlsolve_timing)).")
    println("BatchedSimpleNewtonRaphson Timing: $(median(batched_timing)).")

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

    println("ForwardDiff SimpleNonlinearSolve Timing: $(median(fwdiff_nlsolve_timing)).")
    println("ForwardDiff BatchedNonlinearSolve Timing: $(median(fwdiff_batched_timing)).")

    ∂p3 = only(Zygote.gradient(p) do p
        prob = NonlinearProblem(testing_f, u0, p)
        return sum(abs2, solve(prob, SimpleNewtonRaphson()).u)
    end)

    ∂p4 = only(Zygote.gradient(p) do p
        prob = NonlinearProblem(testing_f, u0, p)
        return sum(abs2, solve(prob, BatchedSimpleNewtonRaphson()).u)
    end)

    ∂p5 = only(Zygote.gradient(p) do p
        prob = NonlinearProblem(testing_f, u0, p)
        sensealg = SteadyStateAdjoint(; linsolve=KrylovJL_GMRES())
        return sum(abs2, solve(prob, BatchedSimpleNewtonRaphson(); sensealg).u)
    end)

    @test ∂p1≈∂p3 atol=1e-5
    @test ∂p3≈∂p4 atol=1e-5
    @test ∂p4≈∂p5 atol=1e-5

    zygote_nlsolve_timing = @be Zygote.gradient($p) do p
        prob = NonlinearProblem(testing_f, u0, p)
        return sum(abs2, solve(prob, SimpleNewtonRaphson()).u)
    end

    zygote_batched_timing = @be Zygote.gradient($p) do p
        prob = NonlinearProblem(testing_f, u0, p)
        return sum(abs2, solve(prob, BatchedSimpleNewtonRaphson()).u)
    end

    println("Zygote SimpleNonlinearSolve Timing: $(median(zygote_nlsolve_timing)).")
    println("Zygote BatchedNonlinearSolve Timing: $(median(zygote_batched_timing)).")
end
