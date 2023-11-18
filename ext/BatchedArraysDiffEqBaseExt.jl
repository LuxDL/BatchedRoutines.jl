module BatchedArraysDiffEqBaseExt

using BatchedArrays, DiffEqBase
import DiffEqBase: recursive_length

recursive_length(B::BatchedArray) = recursive_length(B.data)

end
