@generated function _get_chunksize(ad::AutoAllForwardDiff{CK}, u::AbstractArray) where {CK}
    (CK === nothing || CK ≤ 0) && return :(batched_pickchunksize(u))
    return :($(CK))
end

Base.@assume_effects :total _assert_type(x) = _assert_type(typeof(x))
Base.@assume_effects :total _assert_type(::Type) = true
