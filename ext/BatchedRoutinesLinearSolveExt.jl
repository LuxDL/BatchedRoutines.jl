module BatchedRoutinesLinearSolveExt

using ArrayInterface: ArrayInterface
using BatchedRoutines: BatchedRoutines, UniformBlockDiagonalOperator, getdata
using ChainRulesCore: ChainRulesCore, NoTangent
using FastClosures: @closure
using LinearAlgebra: LinearAlgebra
using LinearSolve: LinearSolve
using SciMLBase: SciMLBase

const CRC = ChainRulesCore

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
function SciMLBase.solve!(cache::LinearSolve.LinearCache{<:UniformBlockDiagonalOperator},
        alg::LinearSolve.LUFactorization; kwargs...)
    A = cache.A
    if cache.isfresh
        fact = LinearAlgebra.lu!(A; check=false)
        cache.cacheval = fact

        if !LinearAlgebra.issuccess(fact)
            return SciMLBase.build_linear_solution(
                alg, cache.u, nothing, cache; retcode=ReturnCode.Failure)
        end

        cache.isfresh = false
    end

    F = LinearSolve.@get_cacheval(cache, :LUFactorization)
    y = LinearSolve._ldiv!(cache.u, F, cache.b)
    return SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

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
    return SciMLBase.build_linear_solution(alg, y, nothing, cache)
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

# We need a custom rrule here to prevent spurios gradients for zero blocks
# Copied from https://github.com/SciML/LinearSolve.jl/blob/7911113c6b14b6897cc356e277ccd5a98faa7dd7/src/adjoint.jl#L31 except the Lazy Arrays part
function CRC.rrule(::typeof(SciMLBase.solve),
        prob::SciMLBase.LinearProblem{T1, T2, <:UniformBlockDiagonalOperator},
        alg::LinearSolve.SciMLLinearSolveAlgorithm, args...;
        alias_A=LinearSolve.default_alias_A(alg, prob.A, prob.b), kwargs...) where {T1, T2}
    cache = SciMLBase.init(prob, alg, args...; kwargs...)
    (; A, sensealg) = cache

    @assert sensealg isa LinearSolve.LinearSolveAdjoint "Currently only `LinearSolveAdjoint` is supported for adjoint sensitivity analysis."

    # Decide if we need to cache `A` and `b` for the reverse pass
    A_ = A
    if sensealg.linsolve === missing
        # We can reuse the factorization so no copy is needed
        # Krylov Methods don't modify `A`, so it's safe to just reuse it
        # No Copy is needed even for the default case
        if !(alg isa LinearSolve.AbstractFactorization ||
             alg isa LinearSolve.AbstractKrylovSubspaceMethod ||
             alg isa LinearSolve.DefaultLinearSolver)
            A_ = alias_A ? deepcopy(A) : A
        end
    else
        A_ = deepcopy(A)
    end

    sol = SciMLBase.solve!(cache)

    proj_A = CRC.ProjectTo(getdata(A))
    proj_b = CRC.ProjectTo(prob.b)

    ∇linear_solve = @closure ∂sol -> begin
        ∂u = ∂sol.u
        if sensealg.linsolve === missing
            λ = if cache.cacheval isa LinearAlgebra.Factorization
                cache.cacheval' \ ∂u
            elseif cache.cacheval isa Tuple &&
                   cache.cacheval[1] isa LinearAlgebra.Factorization
                first(cache.cacheval)' \ ∂u
            elseif alg isa LinearSolve.AbstractKrylovSubspaceMethod
                invprob = SciMLBase.LinearProblem(transpose(cache.A), ∂u)
                SciMLBase.solve(invprob, alg; cache.abstol, cache.reltol, cache.verbose).u
            elseif alg isa LinearSolve.DefaultLinearSolver
                LinearSolve.defaultalg_adjoint_eval(cache, ∂u)
            else
                invprob = SciMLBase.LinearProblem(transpose(A_), ∂u) # We cached `A`
                SciMLBase.solve(invprob, alg; cache.abstol, cache.reltol, cache.verbose).u
            end
        else
            invprob = SciMLBase.LinearProblem(transpose(A_), ∂u) # We cached `A`
            λ = SciMLBase.solve(
                invprob, sensealg.linsolve; cache.abstol, cache.reltol, cache.verbose).u
        end

        uᵀ = reshape(sol.u, 1, :, BatchedRoutines.nbatches(A))
        ∂A = UniformBlockDiagonalOperator(proj_A(BatchedRoutines.batched_mul(
            reshape(λ, :, 1, BatchedRoutines.nbatches(A)), -uᵀ)))
        ∂b = proj_b(λ)
        ∂prob = SciMLBase.LinearProblem(∂A, ∂b, NoTangent())

        return (
            NoTangent(), ∂prob, NoTangent(), ntuple(Returns(NoTangent()), length(args))...)
    end

    return sol, ∇linear_solve
end

end
