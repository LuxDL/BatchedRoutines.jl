module BatchedRoutinesComponentArraysForwardDiffExt

using BatchedRoutines: BatchedRoutines
using ComponentArrays: ComponentArrays, ComponentArray
using ForwardDiff: ForwardDiff

@inline function BatchedRoutines._restructure(y, x::ComponentArray)
    x_data = ComponentArrays.getdata(x)
    return ComponentArray(reshape(y, size(x_data)), ComponentArrays.getaxes(x))
end

end
