module BatchedArraysBlockDiagonalsExt

using BatchedArrays, BlockDiagonals

BlockDiagonals.BlockDiagonal(B::BatchedMatrix) = BlockDiagonal(collect(batchview(B)))

end
