module BatchedRoutinesReverseDiffExt

using BatchedRoutines: BatchedRoutines, batched_pickchunksize, _assert_type
using ReverseDiff: ReverseDiff

Base.@assume_effects :total BatchedRoutines._assert_type(::Type{<:ReverseDiff.TrackedArray})=false
Base.@assume_effects :total BatchedRoutines._assert_type(::Type{<:AbstractArray{<:ReverseDiff.TrackedReal}})=false

end