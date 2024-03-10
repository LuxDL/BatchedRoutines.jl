using BatchedRoutines, CUDA, LinearSolve

# Batched Factorization
A1 = UniformBlockDiagonalMatrix(rand(32, 32, 8))
A2 = Matrix(A1)
b = rand(size(A1, 2))

prob1 = LinearProblem(A1, b)
prob2 = LinearProblem(A2, b)

@benchmark solve(prob1, LUFactorization())
@benchmark solve(prob2, LUFactorization())

@benchmark solve(prob1, QRFactorization())
@benchmark solve(prob2, QRFactorization())

# Batched GMRES
@benchmark solve(prob1, KrylovJL_GMRES())
@benchmark solve(prob2, KrylovJL_GMRES())
@benchmark solve(prob2, SimpleGMRES(; blocksize = size(A1.data, 1)))

# CUDA
cuA1 = cu(A1);
cuA2 = cu(A2);
cub = cu(b);

prob1 = LinearProblem(cuA1, cub)
prob2 = LinearProblem(cuA2, cub)

@benchmark CUDA.@sync solve(prob1, LUFactorization())
@benchmark CUDA.@sync solve(prob2, LUFactorization())

@benchmark CUDA.@sync solve(prob1, QRFactorization())
@benchmark CUDA.@sync solve(prob2, QRFactorization())

@benchmark CUDA.@sync solve(prob1, KrylovJL_GMRES())
@benchmark CUDA.@sync solve(prob2, KrylovJL_GMRES())
@benchmark CUDA.@sync solve(prob2, SimpleGMRES(; blocksize = size(A1.data, 1)))

