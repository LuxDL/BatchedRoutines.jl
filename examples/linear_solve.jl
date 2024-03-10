using BatchedRoutines, CUDA, LinearSolve

# Batched Factorization
A1 = UniformBlockDiagonalMatrix(rand(8, 8, 32))
A2 = Matrix(A1)
b = rand(256)

prob1 = LinearProblem(A1, b)
prob2 = LinearProblem(A2, b)

@benchmark solve(prob1, LUFactorization())
@benchmark solve(prob2, LUFactorization())

@benchmark solve(prob1, QRFactorization())
@benchmark solve(prob2, QRFactorization())

# Batched GMRES
