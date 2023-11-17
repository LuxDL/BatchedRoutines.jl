module BatchedArraysBlockDiagonalsExt

import PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using BatchedArrays, BlockDiagonals
end

BlockDiagonals.BlockDiagonal(B::BatchedMatrix) = BlockDiagonal(collect(batchview(B)))

end