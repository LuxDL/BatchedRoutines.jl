module BatchedArraysBlockDiagonalsExt

using BatchedArrays, BlockDiagonals

BlockDiagonals.BlockDiagonal(B::BatchedMatrix) = BlockDiagonal(batchview(B))

end
