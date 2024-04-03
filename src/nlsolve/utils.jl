@inline __get_concrete_autodiff(prob, ad::AbstractADType; kwargs...) = ad
@inline function __get_concrete_autodiff(prob, ::Nothing; kwargs...)
    prob.f.jac !== nothing && return nothing
    if _is_extension_loaded(Val(:ForwardDiff)) && __can_forwarddiff_dual(eltype(prob.u0))
        return AutoForwardDiff()
    elseif _is_extension_loaded(Val(:FiniteDiff))
        return AutoFiniteDiff()
    else
        error("No AD backend loaded. Please load an AD backend first.")
    end
end

function __can_forwarddiff_dual end

@inline function __value_and_jacobian(prob, x, autodiff)
    if prob.f.jac === nothing
        return prob.f(x, prob.p), batched_jacobian(autodiff, prob.f, x, prob.p)
    else
        return prob.f(x, prob.p), _wrap_batched_operator(prob.f.jac(x, prob.p))
    end
end

@inline __get_tolerance(abstol, u0) = __get_tolerance(abstol, eltype(u0))
@inline function __get_tolerance(abstol, ::Type{T}) where {T}
    return abstol === nothing ? real(oneunit(T)) * (eps(real(one(T))))^(4 // 5) : abstol
end

const BatchedNonlinearSolution{T, N, uType, R, P, O, uType2, S, Tr} = NonlinearSolution{
    T, N, uType, R, P, <:AbstractBatchedNonlinearAlgorithm, O, uType2, S, Tr}
