import SciMLSensitivity: SteadyStateAdjointProblem, SteadyStateAdjointSensitivityFunction

function SteadyStateAdjointProblem(
        sol::BatchedNonlinearSolution, sensealg::SteadyStateAdjoint, alg,
        dgdu::DG1=nothing, dgdp::DG2=nothing, g::G=nothing; kwargs...) where {DG1, DG2, G}
    @assert sol.prob isa NonlinearProblem
    (; f, p, u0) = sol.prob
    f = SciMLBase.ODEFunction(f)

    @assert !SciMLBase.isinplace(sol.prob) "Adjoint for Batched Problems does not support \
                                            inplace problems."
    @assert ndims(u0)==2 "u0 must be a matrix."
    @assert dgdu!==nothing "`dgdu` must be specified. Automatic differentiation is not \
                            currently implemented for this part."
    @assert sensealg.autojacvec isa ZygoteVJP

    dgdu === nothing &&
        dgdp === nothing &&
        g === nothing &&
        error("Either `dgdu`, `dgdp`, or `g` must be specified.")

    needs_jac = ifelse(SciMLBase.has_adjoint(f),
        false,
        ifelse(sensealg.linsolve === nothing, size(u0, 1) ≤ 50,
            SciMLSensitivity.__needs_concrete_A(sensealg.linsolve)))

    p === SciMLBase.NullParameters() &&
        error("Your model does not have parameters, and thus it is impossible to calculate \
               the derivative of the solution with respect to the parameters. Your model \
               must have parameters to use parameter sensitivity calculations!")

    y = sol.u

    if needs_jac
        if SciMLBase.has_jac(f)
            J = BatchedRoutines._wrap_batched_operator(f.jac(y, p, nothing))
        else
            uf = SciMLBase.UJacobianWrapper(f, nothing, p)
            if SciMLSensitivity.alg_autodiff(sensealg)
                J = BatchedRoutines.batched_jacobian(AutoFiniteDiff(), uf, y)
            else
                J = BatchedRoutines.batched_jacobian(AutoForwardDiff(), uf, y)
            end
        end
    end

    if dgdp === nothing && g === nothing
        dgdu_val = similar(u0, length(u0))
        dgdp_val = nothing
    else
        dgdu_val, dgdp_val = similar(u0, length(u0)), similar(u0, length(p))
    end

    if dgdu !== nothing
        dgdu(dgdu_val, y, p, nothing, nothing)
    else
        error("Not implemented yet")
    end

    if !needs_jac # Construct an operator and use Jacobian-Free Linear Solve
        linsolve = if sensealg.linsolve === nothing
            LinearSolve.SimpleGMRES(; blocksize=size(u0, 1))
        else
            sensealg.linsolve
        end
        usize = size(y)
        __f = @closure y -> vec(f(reshape(y, usize), p, nothing))
        operator = SciMLSensitivity.VecJac(__f, vec(y);
            autodiff=SciMLSensitivity.get_autodiff_from_vjp(sensealg.autojacvec))
        linear_problem = SciMLBase.LinearProblem(operator, dgdu_val)
        linsol = SciMLBase.solve(
            linear_problem, linsolve; alias_A=true, sensealg.linsolve_kwargs...)
    else
        linear_problem = SciMLBase.LinearProblem(J', dgdu_val)
        linsol = SciMLBase.solve(
            linear_problem, sensealg.linsolve; alias_A=true, sensealg.linsolve_kwargs...)
    end
    λ = linsol.u

    _, pb_f = Zygote.pullback(@closure(p->vec(f(y, p, nothing))), p)
    ∂p = only(pb_f(λ))
    ∂p === nothing &&
        !sensealg.autojacvec.allow_nothing &&
        throw(SciMLSensitivity.ZygoteVJPNothingError())

    if g !== nothing || dgdp !== nothing
        error("Not implemented yet")
    else
        SciMLSensitivity.recursive_neg!(∂p)
        return ∂p
    end
end
