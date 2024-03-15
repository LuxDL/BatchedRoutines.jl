module BatchedRoutinesZygoteExt

using ADTypes: AutoForwardDiff, AutoZygote
using BatchedRoutines: BatchedRoutines
using ChainRulesCore: ChainRulesCore, NoTangent
using FastClosures: @closure
using Zygote: Zygote

const CRC = ChainRulesCore

@inline BatchedRoutines._is_extension_loaded(::Val{:Zygote}) = true

function BatchedRoutines._batched_gradient(::AutoZygote, f::F, u) where {F}
    return only(Zygote.gradient(f, u))
end

end
