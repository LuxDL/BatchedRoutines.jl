module BatchedArraysLinearSolveExt

using BatchedArrays, LinearSolve
import LinearSolve: defaultalg,
    do_factorization, init_cacheval, DefaultLinearSolver, DefaultAlgorithmChoice

function defaultalg(::BatchedMatrix, ::BatchedVector, oa::OperatorAssumptions)
    return DefaultLinearSolver(DefaultAlgorithmChoice.DirectLdiv!)
end

for alg in (:SVDFactorization, :MKLLUFactorization, :DiagonalFactorization,
    :SparspakFactorization, :KLUFactorization, :UMFPACKFactorization,
    :GenericLUFactorization, :RFLUFactorization, :BunchKaufmanFactorization,
    :CHOLMODFactorization, :NormalCholeskyFactorization, :LDLtFactorization,
    :AppleAccelerateLUFactorization)
    @eval begin
        function init_cacheval(::$(alg), ::BatchedArray, b, u, Pl, Pr, maxiters::Int,
                abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
            return nothing
        end
    end
end

end
