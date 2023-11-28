module BatchedArraysForwardDiffExt

using BatchedArrays, ForwardDiff
import ForwardDiff: Dual,
    Partials, extract_jacobian!, extract_jacobian_chunk!, partials, seed!, value

# For BatchedArrays there is a faster way to do gradient and jacobian. The duals for
# multiple batches can be propagated together.
# There is a potential correctness issue if there is an inter-batch operation -- but this
# can only happen if someone is defining a direct dispatch on Batched Arrays

function value(v::BatchedVector)
    x = value.(v.data)
    return BatchedArray{eltype(x), nbatches(v)}(x)
end

## -------
## Seeding
## -------
function seed!(duals::BatchedArray{Dual{T, V, N}}, x,
        seed::Partials{N, V}=zero(Partials{N, V})) where {T, V, N}
    for (dᵢ, xᵢ) in zip(batchview(duals), batchview(x))
        dᵢ .= Dual{T, V, N}.(xᵢ, Ref(seed))
    end
    return duals
end

function seed!(duals::BatchedArray{Dual{T, V, N}}, x,
        seeds::NTuple{N, Partials{N, V}}) where {T, V, N}
    dual_inds = 1:N
    for (dᵢ, xᵢ) in zip(batchview(duals), batchview(x))
        dᵢ[dual_inds] .= Dual{T, V, N}.(view(xᵢ, dual_inds), seeds)
    end
    return duals
end

function seed!(duals::BatchedArray{Dual{T, V, N}}, x, index,
        seed::Partials{N, V}=zero(Partials{N, V})) where {T, V, N}
    offset = index - 1
    dual_inds = (1:N) .+ offset
    for (dᵢ, xᵢ) in zip(batchview(duals), batchview(x))
        dᵢ[dual_inds] .= Dual{T, V, N}.(view(xᵢ, dual_inds), Ref(seed))
    end
    return duals
end

function seed!(duals::BatchedArray{Dual{T, V, N}}, x, index,
        seeds::NTuple{N, Partials{N, V}}, chunksize=N) where {T, V, N}
    offset = index - 1
    seed_inds = 1:chunksize
    dual_inds = seed_inds .+ offset
    for (dᵢ, xᵢ) in zip(batchview(duals), batchview(x))
        dᵢ[dual_inds] .= Dual{
            T, V, N}.(view(xᵢ, dual_inds), getindex.(Ref(seeds), seed_inds))
    end
    return duals
end

## --------
## Jacobian
## --------
function extract_jacobian!(::Type{T}, result::BatchedArray, ydual::BatchedArray,
        n) where {T}
    partials_wrap(ydual, nrange) = partials(T, ydual, nrange)
    for (resultᵢ, ydualᵢ) in zip(batchview(result), batchview(ydual))
        out_reshaped = reshape(resultᵢ, length(ydualᵢ), n)
        ydual_reshaped = vec(ydualᵢ)
        out_reshaped .= partials_wrap.(ydual_reshaped, transpose(1:n))
    end
    return result
end

function extract_jacobian_chunk!(::Type{T}, result::BatchedArray, ydual::BatchedArray,
        index, chunksize) where {T}
    partials_wrap(ydual, nrange) = partials(T, ydual, nrange)
    offset = index - 1
    irange = 1:chunksize
    col = irange .+ offset
    for (resultᵢ, ydualᵢ) in zip(batchview(result), batchview(ydual))
        ydual_reshaped = vec(ydualᵢ)
        resultᵢ[:, col] .= partials_wrap.(ydual_reshaped, transpose(irange))
    end
    return result
end

end
