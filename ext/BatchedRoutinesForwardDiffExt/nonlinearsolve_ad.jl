function SciMLBase.solve(
        prob::NonlinearProblem{<:AbstractArray, iip,
            <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}}},
        alg::AbstractBatchedNonlinearAlgorithm,
        args...;
        kwargs...) where {T, V, P, iip}
    sol, partials = __nlsolve_ad(prob, alg, args...; kwargs...)
    dual_soln = __nlsolve_dual_soln(sol.u, partials, prob.p)
    return SciMLBase.build_solution(
        prob, alg, dual_soln, sol.resid; sol.retcode, sol.stats, sol.original)
end

function __nlsolve_ad(prob::NonlinearProblem, alg, args...; kwargs...)
    p = ForwardDiff.value.(prob.p)
    u0 = ForwardDiff.value.(prob.u0)
    newprob = NonlinearProblem(prob.f, u0, p; prob.kwargs...)

    sol = SciMLBase.solve(newprob, alg, args...; kwargs...)

    uu = sol.u
    Jₚ = ForwardDiff.jacobian(Base.Fix1(prob.f, uu), p)
    Jᵤ = if prob.f.jac === nothing
        BatchedRoutines.batched_jacobian(AutoForwardDiff(), prob.f, uu, p)
    else
        BatchedRoutines._wrap_batched_operator(prob.f.jac(uu, p))
    end

    Jᵤ_fact = LinearAlgebra.lu!(Jᵤ)

    map_fn = @closure zp -> begin
        Jₚᵢ, p = zp
        LinearAlgebra.ldiv!(Jᵤ_fact, Jₚᵢ)
        Jₚᵢ .*= -1
        return map(Base.Fix2(*, ForwardDiff.partials(p)), Jₚᵢ)
    end

    return sol, sum(map_fn, zip(eachcol(Jₚ), prob.p))
end

@inline function __nlsolve_dual_soln(u::AbstractArray, partials,
        ::Union{<:AbstractArray{<:Dual{T, V, P}}, Dual{T, V, P}}) where {T, V, P}
    _partials = reshape(partials, size(u))
    return map(((uᵢ, pᵢ),) -> Dual{T, V, P}(uᵢ, pᵢ), zip(u, _partials))
end
