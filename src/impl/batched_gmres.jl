# This is modelled very close to Krylov.jl
abstract type AbstractBatchedKrylovSolver{T, FC, S} end

"""
    SimpleStats{T}

Type for statistics returned by the majority of Krylov solvers, the attributes are:

  - niter
  - solved
  - inconsistent
  - residuals
  - Aresiduals
  - Acond
  - timer
  - status
"""
mutable struct SimpleStats{T}
    niter::Int
    solved::Bool
    inconsistent::Bool
    residuals::Vector{T}
    Aresiduals::Vector{T}
    Acond::Vector{T}
    status::Symbol
end

function _reset!(stats::SimpleStats{T}) where {T}
    stats.niter = 0
    stats.solved = false
    stats.inconsistent = false
    resize!(stats.residuals, 0)
    resize!(stats.Aresiduals, 0)
    resize!(stats.Acond, 0)
    stats.status = :unknown
    return
end

mutable struct BatchedGmresSolver{T, FC, S} <: AbstractBatchedKrylovSolver{T, FC, S}
    m::Int
    n::Int
    b::Int
    Δx::S
    x::S
    w::S
    p::S
    q::S
    V::Vector{S}
    c::Vector{T}
    s::Vector{FC}
    z::Vector{FC}
    R::Vector{FC}
    warm_start::Bool
    inner_iter::Int
    stats::SimpleStats{T}
end

function BatchedGmresSolver(m::Int, n::Int, B::Int, memory::Int, b; zeroinit::Bool=false)
    memory = min(m, memory)
    zeroinit && ((m, n, B, memory) = (0, 0, 0, 0))
    FC = eltype(b)
    T = real(FC)
    Δx = similar(b, 0, B)
    x = similar(b, n, B)
    w = similar(b, n, B)
    p = similar(b, 0, B)
    q = similar(b, 0, B)
    V = [similar(b, n, B) for _ in 1:memory]
    c = Vector{T}(undef, memory)
    s = [similar(b, B) for _ in 1:memory] # Vector{FC}(undef, memory)
    z = [similar(b, B) for _ in 1:memory] # Vector{FC}(undef, memory)
    R = [similar(b, B) for _ in 1:(memory * (memory + 1) ÷ 2)] # Vector{FC}(undef, div(memory * (memory + 1), 2))
    stats = SimpleStats(0, false, false, T[], T[], T[], :unkwown)
    return BatchedGmresSolver{T, eltype(s), typeof(Δx)}(
        m, n, B, Δx, x, w, p, q, V, c, s, z, R, false, 0, stats)
end

@inline function _zero!(x::AbstractArray{T}) where {T}
    return T <: AbstractArray ? _zero!.(x) : (x .= zero(T))
end

@inline function BatchedGmresSolver(A::AbstractArray{T, 3}, b::AbstractMatrix,
        memory=20; zeroinit::Bool=false) where {T}
    m, n, B = size(A)
    @assert m==n "BatchedGMRES requires a square matrix."
    @assert m == size(b, 1)
    @assert nbatches(A) == nbatches(b)
    return BatchedGmresSolver(m, n, B, memory, b; zeroinit)
end

function batched_gmres(A::AbstractArray{T, 3}, b::AbstractMatrix,
        x0::Union{AbstractMatrix, Nothing}=nothing; memory::Int=20,
        M=I, N=I, ldiv::Bool=false, restart::Bool=false, atol::T=(√(eps(T))),
        rtol::T=(√(eps(T))), iostream::IO=stdout, itmax::Int=0, verbose::Int=0,
        history::Bool=false, callback::C=Returns(false)) where {T, C}
    solver = BatchedGmresSolver(A, b, memory)
    if x0 !== nothing
        _allocate_if(true, solver, :Δx, size(x0)...)
        copyto!(solver.Δx, x0)
        solver.warm_start = true
    end
    batched_gmres!(solver, A, b; M, N, ldiv, restart, atol, rtol,
        itmax, verbose, history, iostream, callback)
    return solver.x, solver.stats
end

function batched_gmres!(solver::BatchedGmresSolver, A::AbstractArray{T, 3}, b; M=I, N=I,
        ldiv::Bool=false, restart::Bool=false, atol::T=(√(eps(T))),
        rtol::T=(√(eps(T))), iostream::IO=stdout, itmax::Int=0, verbose::Int=0,
        history::Bool=false, callback::C=Returns(false)) where {T, C}
    m, n, B = size(A)
    (m == solver.m && n == solver.n) ||
        error("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($m, $n)")
    m == n || error("System must be square")
    length(b) == m * B || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "BatchedGMRES: system of size %d x %d\n", n, B)

    # Check M = Iₙ and N = Iₙ
    MisI = (M === I)
    NisI = (N === I)

    # Setup the workspace
    _allocate_if(!MisI, solver, :q, n, B)
    _allocate_if(!NisI, solver, :p, n, B)
    _allocate_if(restart, solver, :Δx, n, B)

    (; Δx, x, w, V, z, c, s, R, stats, warm_start) = solver
    _reset!(stats)
    rNorms = stats.residuals

    q = MisI ? w : solver.q
    r₀ = NisI ? w : solver.p
    xr = restart ? Δx : x

    # Initial solution x₀.
    _zero!(x)

    # Initial residual r₀.
    if warm_start
        batched_mul!(w, A, Δx)
        axpby!(true, b, -one(eltype(w)), w)
        restart && axpy!(true, Δx, x)
    else
        w .= b
    end
    MisI || _mulorldiv!(r₀, M, w, ldiv)  # r₀ = M(b - Ax₀)
    β = batched_norm(r₀)                 # β = ‖r₀‖₂

    rNorm = β
    history && push!(rNorms, norm(rNorm, 2))
    ε = @. atol + rtol * rNorm

    if all(iszero, β)
        stats.niter = 0
        stats.solved, stats.inconsistent = true, false
        stats.status = :initial_solution
        solver.warm_start = false
        return solver
    end

    mem = length(c)  # Memory
    npass = 0        # Number of pass

    iter = 0        # Cumulative number of iterations
    inner_iter = 0  # Number of iterations in a pass

    itmax == 0 && (itmax = 2 * n)
    inner_itmax = itmax

    if (verbose > 0)
        @printf(iostream, "%5s  %5s  %7s  %7s\n", "pass", "k", "‖rₖ‖", "hₖ₊₁.ₖ")
        mod(iter, verbose) == 0 &&
            @printf(iostream, "%5d  %5d  %7.1e  %7s\n", npass, iter, norm(rNorm, 2),
                "✗ ✗ ✗ ✗")
    end

    # Tolerance for breakdown detection.
    btol = eps(T)^(3 / 4)

    # Stopping criterion
    breakdown = false
    inconsistent = false
    solved_cache = rNorm .≤ ε
    solved = all(solved_cache)
    tired = iter ≥ itmax
    inner_tired = inner_iter ≥ inner_itmax
    status = :unknown
    user_requested_exit = false

    while !(solved || tired || breakdown || user_requested_exit)

        # Initialize workspace.
        nr = 0  # Number of coefficients stored in Rₖ.
        _zero!(V)  # Orthogonal basis of Kₖ(MAN, Mr₀).
        _zero!(s)  # Givens sines used for the factorization QₖRₖ = Hₖ₊₁.ₖ.
        _zero!(c)  # Givens cosines used for the factorization QₖRₖ = Hₖ₊₁.ₖ.
        _zero!(R)  # Upper triangular matrix Rₖ.
        _zero!(z)  # Right-hand of the least squares problem min ‖Hₖ₊₁.ₖyₖ - βe₁‖₂.

        if restart
            _zero!(xr)  # xr === Δx when restart is set to true
            if npass ≥ 1
                batched_mul!(w, A, x)
                axpby!(true, b, -one(eltype(w)), w)
                MisI || _mulorldiv!(r₀, M, w, ldiv)
            end
        end

        # Initial ζ₁ and V₁
        β = batched_norm(r₀)  # FIXME: allocations
        z[1] = β
        V[1] .= r₀ ./ reshape(rNorm, 1, :)

        npass = npass + 1
        solver.inner_iter = 0
        inner_tired = false

        while !(solved || inner_tired || breakdown || user_requested_exit)
            # Update iteration index
            solver.inner_iter = solver.inner_iter + 1
            inner_iter = solver.inner_iter

            # Update workspace if more storage is required and restart is set to false
            if !restart && (inner_iter > mem)
                for i in 1:inner_iter
                    push!(R, zero(first(R)))
                end
                push!(s, zero(first(s)))
                push!(c, zero(first(c)))
            end

            # Continue the Arnoldi process.
            p = NisI ? V[inner_iter] : solver.p
            NisI || _mulorldiv!(p, N, V[inner_iter], ldiv)  # p ← Nvₖ
            batched_mul!(w, A, p)                           # w ← ANvₖ
            MisI || _mulorldiv!(q, M, w, ldiv)              # q ← MANvₖ
            for i in 1:inner_iter
                batched_dot!(R[nr + i], V[i], q)            # hᵢₖ = (vᵢ)ᴴq
                batched_axpy!(-1, R[nr + i], V[i], q)       # q ← q - hᵢₖvᵢ
            end

            # Compute hₖ₊₁.ₖ
            Hbis = batched_norm(q)  # hₖ₊₁.ₖ = ‖vₖ₊₁‖₂  # FIXME: allocations

            # Update the QR factorization of Hₖ₊₁.ₖ.
            # Apply previous Givens reflections Ωᵢ.
            # [cᵢ  sᵢ] [ r̄ᵢ.ₖ ] = [ rᵢ.ₖ ]
            # [s̄ᵢ -cᵢ] [rᵢ₊₁.ₖ]   [r̄ᵢ₊₁.ₖ]
            for i in 1:(inner_iter - 1)
                #             Rtmp = c[i] * R[nr + i] + s[i] * R[nr + i + 1]
                #             R[nr + i + 1] = conj(s[i]) * R[nr + i] - c[i] * R[nr + i + 1]
                #             R[nr + i] = Rtmp
            end

            # Compute and apply current Givens reflection Ωₖ.
            # [cₖ  sₖ] [ r̄ₖ.ₖ ] = [rₖ.ₖ]
            # [s̄ₖ -cₖ] [hₖ₊₁.ₖ]   [ 0  ]
            #         (c[inner_iter], s[inner_iter], R[nr + inner_iter]) = sym_givens(
            #             R[nr + inner_iter], Hbis)

            # Update zₖ = (Qₖ)ᴴβe₁
            #         ζₖ₊₁ = conj(s[inner_iter]) * z[inner_iter]
            #         z[inner_iter] = c[inner_iter] * z[inner_iter]

            # Update residual norm estimate.
            # ‖ M(b - Axₖ) ‖₂ = |ζₖ₊₁|
            #         rNorm = abs(ζₖ₊₁)
            #         history && push!(rNorms, rNorm)

            # Update the number of coefficients in Rₖ
            nr = nr + inner_iter

            # Stopping conditions that do not depend on user input.
            # This is to guard against tolerances that are unreasonably small.
            #         resid_decrease_mach = (rNorm + one(T) ≤ one(T))

            # Update stopping criterion.
            user_requested_exit = callback(solver)::Bool
            #         resid_decrease_lim = rNorm ≤ ε
            #         breakdown = Hbis ≤ btol
            #         solved = resid_decrease_lim || resid_decrease_mach
            #         inner_tired = restart ? inner_iter ≥ min(mem, inner_itmax) :
            #                       inner_iter ≥ inner_itmax
            #         timer = time_ns() - start_time
            #         overtimed = timer > timemax_ns
            #         kdisplay(iter + inner_iter, verbose) &&
            #             @printf(iostream, "%5d  %5d  %7.1e  %7.1e  %.2fs\n", npass,
            #                 iter+inner_iter, rNorm, Hbis, ktimer(start_time))

            # Compute vₖ₊₁.
            #         if !(solved || inner_tired || breakdown || user_requested_exit || overtimed)
            #             if !restart && (inner_iter ≥ mem)
            #                 push!(V, S(undef, n))
            #                 push!(z, zero(FC))
            #             end
            #             @. V[inner_iter + 1] = q / Hbis  # hₖ₊₁.ₖvₖ₊₁ = q
            #             z[inner_iter + 1] = ζₖ₊₁
            #         end

            error(1)
        end

        # Compute yₖ by solving Rₖyₖ = zₖ with backward substitution.
        #     y = z  # yᵢ = zᵢ
        #     for i in inner_iter:-1:1
        #         pos = nr + i - inner_iter      # position of rᵢ.ₖ
        #         for j in inner_iter:-1:(i + 1)
        #             y[i] = y[i] - R[pos] * y[j]  # yᵢ ← yᵢ - rᵢⱼyⱼ
        #             pos = pos - j + 1            # position of rᵢ.ⱼ₋₁
        #         end
        #         # Rₖ can be singular if the system is inconsistent
        #         if abs(R[pos]) ≤ btol
        #             y[i] = zero(FC)
        #             inconsistent = true
        #         else
        #             y[i] = y[i] / R[pos]  # yᵢ ← yᵢ / rᵢᵢ
        #         end
        #     end

        # Form xₖ = NVₖyₖ
        #     for i in 1:inner_iter
        #         @kaxpy!(n, y[i], V[i], xr)
        #     end
        #     if !NisI
        #         solver.p .= xr
        #         mulorldiv!(xr, N, solver.p, ldiv)
        #     end
        #     restart && @kaxpy!(n, one(FC), xr, x)

        # Update inner_itmax, iter, tired and overtimed variables.
        inner_itmax = inner_itmax - inner_iter
        iter = iter + inner_iter
        tired = iter ≥ itmax
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Termination status
    tired && (status = :maxiters)
    solved && (status = :converged)
    inconsistent && (status = :approx_ls_soln)
    user_requested_exit && (status = :user_requested_exit)

    # Update x
    warm_start && !restart && axpy!(true, Δx, x)
    solver.warm_start = false

    # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.inconsistent = inconsistent
    stats.status = status
    return solver
end

@inline function _allocate_if(cond, solver, sym::Symbol, dims...)
    if cond
        x = getfield(solver, sym)
        if length(x) != prod(dims)
            setfield!(solver, sym, similar(x, dims...))
        end
    end
end

@inline _mulorldiv!(y, P, x, ldiv::Bool) = ldiv ? ldiv!(y, P, x) : batched_mul!(y, P, x)
