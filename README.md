# BatchedArrays.jl

[![Join the chat at https://julialang.zulipchat.com #machine-learning](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/machine-learning)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](http://lux.csail.mit.edu/dev/api/)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](http://lux.csail.mit.edu/stable/api/)

[![CI](https://github.com/LuxDL/BatchedArrays.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/LuxDL/BatchedArrays.jl/actions/workflows/CI.yml)
[![Build status](https://img.shields.io/buildkite/ba1f9622add5978c2d7b194563fd9327113c9c21e5734be20e/main.svg?label=gpu)](https://buildkite.com/julialang/lux-dot-jl)
[![codecov](https://codecov.io/gh/LuxDL/BatchedArrays.jl/branch/main/graph/badge.svg?token=IMqBM1e3hz)](https://codecov.io/gh/LuxDL/BatchedArrays.jl.jl)
[![Package Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/BatchedArrays)](https://pkgs.genieframework.com?packages=BatchedArrays)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

`BatchedArrays.jl` provides a way to convert regular SciML solvers into batched solvers,
i.e., solving multiple independent problems simultaneously. This is mostly useful in the
machine learning context.

For people familiar with `jax` and `functorch`, this is similar to a poor man's `vmap`. It
effectively takes your code written for an ND-Array and tries to generalize it to
(N+1)D-Arrays.

> [!WARNING]
> This package is currently experimental and it is highly advisable not to directly depend
> on it. Support for SciML applications is available only after loading this package
> explicitly.

## Common Problems

* If you have a code where `if <x>; ...; end` block and `x` happens to be a `BatchedScalar`,
  this will lead to an error. To generically support batching:
  1. Avoid Conditionals
  2. Write you code using `ifelse(x, ...)` instead of `if <x>; ...; else ...; end`.
  3. If you are very confident that you know what you are doing, you can use `Bool(x)` to
     convert a `BatchedScalar` to a `Bool`. This returns `true` if all the elements in the
     batch are `true` and `false` otherwise. Note that doing this means the computation
     relies on all the batches which might not be what you want.
