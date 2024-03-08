"""
    batched_jacobian(ad, f::F, x) where {F}

Use the backend `ad` to compute the Jacobian of `f` at `x` in batched mode. If `x` is a
vector then this behaves like usual jacobian and returns a matrix, else we return a 3D array
where the last dimension is the batch size.
"""
function batched_jacobian end
