module BatchedRoutinesLinearSolveExt

using ArrayInterface: ArrayInterface
using BatchedRoutines: BatchedRoutines, UniformBlockDiagonalOperator, getdata
using LinearAlgebra: LinearAlgebra
using LinearSolve: LinearSolve

# Overload LinearProblem, else causing problems in the adjoint code
function LinearSolve.LinearProblem(op::UniformBlockDiagonalOperator, b, args...; kwargs...)
    return LinearSolve.LinearProblem{true}(op, b, args...; kwargs...)
end

# Default Algorithm
function LinearSolve.defaultalg(
        op::UniformBlockDiagonalOperator, b, assump::LinearSolve.OperatorAssumptions{Bool})
    alg = if assump.issq
        LinearSolve.DefaultAlgorithmChoice.LUFactorization
    elseif assump.condition === LinearSolve.OperatorCondition.WellConditioned
        LinearSolve.DefaultAlgorithmChoice.NormalCholeskyFactorization
    elseif assump.condition === LinearSolve.OperatorCondition.IllConditioned
        if LinearSolve.is_underdetermined(op)
            LinearSolve.DefaultAlgorithmChoice.QRFactorizationPivoted
        else
            LinearSolve.DefaultAlgorithmChoice.QRFactorization
        end
    elseif assump.condition === LinearSolve.OperatorCondition.VeryIllConditioned
        if LinearSolve.is_underdetermined(op)
            LinearSolve.DefaultAlgorithmChoice.QRFactorizationPivoted
        else
            LinearSolve.DefaultAlgorithmChoice.QRFactorization
        end
    elseif assump.condition === LinearSolve.OperatorCondition.SuperIllConditioned
        LinearSolve.DefaultAlgorithmChoice.SVDFactorization
    else
        error("Special factorization not handled in current default algorithm.")
    end
    return LinearSolve.DefaultLinearSolver(alg)
end

# GenericLUFactorization
function LinearSolve.init_cacheval(alg::LinearSolve.GenericLUFactorization,
        A::UniformBlockDiagonalOperator, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol, verbose::Bool, ::LinearSolve.OperatorAssumptions)
    A_ = UniformBlockDiagonalOperator(similar(getdata(A), 0, 0, 1))
    return LinearAlgebra.generic_lufact!(A_, alg.pivot; check=false)
end

function LinearSolve.do_factorization(
        alg::LinearSolve.GenericLUFactorization, A::UniformBlockDiagonalOperator, b, u)
    return LinearAlgebra.generic_lufact!(A, alg.pivot; check=false)
end

# LUFactorization
function LinearSolve.init_cacheval(
        alg::LinearSolve.LUFactorization, A::UniformBlockDiagonalOperator, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool, ::LinearSolve.OperatorAssumptions)
    A_ = UniformBlockDiagonalOperator(similar(getdata(A), 0, 0, 1))
    return LinearAlgebra.lu!(A_, alg.pivot; check=false)
end

function LinearSolve.do_factorization(
        alg::LinearSolve.LUFactorization, A::UniformBlockDiagonalOperator, b, u)
    return LinearAlgebra.lu!(A, alg.pivot; check=false)
end

# QRFactorization
function LinearSolve.init_cacheval(
        alg::LinearSolve.QRFactorization, A::UniformBlockDiagonalOperator, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool, ::LinearSolve.OperatorAssumptions)
    A_ = UniformBlockDiagonalOperator(similar(getdata(A), 0, 0, 1))
    return LinearAlgebra.qr!(A_, alg.pivot)
end

function LinearSolve.do_factorization(
        alg::LinearSolve.QRFactorization, A::UniformBlockDiagonalOperator, b, u)
    alg.inplace && return LinearAlgebra.qr!(A, alg.pivot)
    return LinearAlgebra.qr(A, alg.pivot)
end

# CholeskyFactorization
function LinearSolve.init_cacheval(alg::LinearSolve.CholeskyFactorization,
        A::UniformBlockDiagonalOperator, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol, verbose::Bool, ::LinearSolve.OperatorAssumptions)
    A_ = UniformBlockDiagonalOperator(similar(getdata(A), 0, 0, 1))
    return ArrayInterface.cholesky_instance(A_, alg.pivot)
end

function LinearSolve.do_factorization(
        alg::LinearSolve.CholeskyFactorization, A::UniformBlockDiagonalOperator, b, u)
    return LinearAlgebra.cholesky!(A, alg.pivot; check=false)
end

# NormalCholeskyFactorization
function LinearSolve.init_cacheval(alg::LinearSolve.NormalCholeskyFactorization,
        A::UniformBlockDiagonalOperator, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol, verbose::Bool, ::LinearSolve.OperatorAssumptions)
    A_ = UniformBlockDiagonalOperator(similar(getdata(A), 0, 0, 1))
    return ArrayInterface.cholesky_instance(A_, alg.pivot)
end

function LinearSolve.solve!(cache::LinearSolve.LinearCache{<:UniformBlockDiagonalOperator},
        alg::LinearSolve.NormalCholeskyFactorization; kwargs...)
    A = cache.A
    if cache.isfresh
        fact = LinearAlgebra.cholesky!(A' * A, alg.pivot; check=false)
        cache.cacheval = fact
        cache.isfresh = false
    end
    y = LinearAlgebra.ldiv!(
        cache.u, LinearSolve.@get_cacheval(cache, :NormalCholeskyFactorization),
        A' * cache.b)
    return LinearSolve.SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

# SVDFactorization
function LinearSolve.init_cacheval(
        alg::LinearSolve.SVDFactorization, A::UniformBlockDiagonalOperator, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool, ::LinearSolve.OperatorAssumptions)
    A_ = UniformBlockDiagonalOperator(similar(getdata(A), 0, 0, 1))
    return ArrayInterface.svd_instance(A_)
end

function LinearSolve.do_factorization(
        alg::LinearSolve.SVDFactorization, A::UniformBlockDiagonalOperator, b, u)
    return LinearAlgebra.svd!(A; alg.full, alg.alg)
end

end
