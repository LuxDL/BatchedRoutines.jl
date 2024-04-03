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

    # sense = SteadyStateAdjointSensitivityFunction(g, sensealg, alg, sol, dgdu, dgdp,
    #     f, f.colorvec, false) # Dont allocate the Jacobian yet in diffcache
    # @show sense.vjp
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
        # TODO: Implement this part
        error("Not implemented yet")
        #     if g !== nothing
        #         if dgdp_val !== nothing
        #             gradient!(vec(dgdu_val), diffcache.g[1], y, sensealg,
        #                 diffcache.g_grad_config[1])
        #         else
        #             gradient!(vec(dgdu_val), diffcache.g, y, sensealg, diffcache.g_grad_config)
        #         end
        #     end
    end

    if !needs_jac # Construct an operator and use Jacobian-Free Linear Solve
        error("Todo Jacobian Free Linear Solve")
    #     usize = size(y)
    #     __f = y -> vec(f(reshape(y, usize), p, nothing))
    #     operator = VecJac(__f, vec(y);
    #         autodiff = get_autodiff_from_vjp(sensealg.autojacvec))
    #     linear_problem = LinearProblem(operator, vec(dgdu_val); u0 = vec(λ))
    #     solve(linear_problem, linsolve; alias_A = true, sensealg.linsolve_kwargs...) # u is vec(λ)
    else
        linear_problem = SciMLBase.LinearProblem(J', dgdu_val)
        linsol = SciMLBase.solve(
            linear_problem, sensealg.linsolve; alias_A=true, sensealg.linsolve_kwargs...)
        λ = linsol.u
    end

    _, pb_f = Zygote.pullback(@closure(p->vec(f(y, p, nothing))), p)
    ∂p = only(pb_f(λ))
    ∂p === nothing &&
        !sensealg.autojacvec.allow_nothing &&
        throw(SciMLSensitivity.ZygoteVJPNothingError())

    if g !== nothing || dgdp !== nothing
        error("Not implemented yet")
        # compute del g/del p
        # if dgdp !== nothing
        #     dgdp(dgdp_val, y, p, nothing, nothing)
        # else
        #     @unpack g_grad_config = diffcache
        #     gradient!(dgdp_val, diffcache.g[2], p, sensealg, g_grad_config[2])
        # end
        # recursive_sub!(dgdp_val, vjp)
        # return dgdp_val
    else
        SciMLSensitivity.recursive_neg!(∂p)
        return ∂p
    end
end
