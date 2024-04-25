@kwdef @concrete struct BatchedSimpleNewtonRaphson <: AbstractBatchedNonlinearAlgorithm
    autodiff = nothing
end

const BatchedSimpleGaussNewton = BatchedSimpleNewtonRaphson

function SciMLBase.__solve(prob::Union{NonlinearProblem, NonlinearLeastSquaresProblem},
        alg::BatchedSimpleNewtonRaphson, args...;
        abstol=nothing, maxiters::Int=1000, kwargs...)
    @assert !SciMLBase.isinplace(prob) "BatchedSimpleNewtonRaphson does not support inplace."

    x = deepcopy(prob.u0)
    fx = prob.f(x, prob.p)
    @assert (ndims(x) == 2)&&(ndims(fx) == 2) "BatchedSimpleNewtonRaphson only supports matrices."

    autodiff = __get_concrete_autodiff(prob, alg.autodiff)
    abstol = __get_tolerance(abstol, x)

    maximum(abs, fx) < abstol &&
        return SciMLBase.build_solution(prob, alg, x, fx; retcode=ReturnCode.Success)

    for _ in 1:maxiters
        fx, J = __value_and_jacobian(prob, x, autodiff)
        δx = J \ fx

        maximum(abs, fx) < abstol &&
            return SciMLBase.build_solution(prob, alg, x, fx; retcode=ReturnCode.Success)

        @. x -= δx
    end

    return SciMLBase.build_solution(prob, alg, x, fx; retcode=ReturnCode.MaxIters)
end
