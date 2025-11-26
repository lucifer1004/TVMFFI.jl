# TVMFFI Benchmark Suite

This directory contains benchmarks for measuring FFI overhead and performance characteristics using [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl).

## Prerequisites

The benchmarks have their own `Project.toml` to avoid adding BenchmarkTools as a main dependency.

```bash
cd TVMFFI/benchmarks
julia --project=. -e 'using Pkg; Pkg.develop(path=".."); Pkg.instantiate()'
```

## Running Benchmarks

### Quick Run (FFI Overhead)

```bash
cd TVMFFI/benchmarks
julia --project=. ffi_overhead.jl
```

### Detailed Micro-benchmarks

```bash
cd TVMFFI/benchmarks
julia --project=. microbenchmarks.jl
```

## Benchmark Files

| File | Description |
|------|-------------|
| `ffi_overhead.jl` | End-to-end FFI overhead measurement |
| `microbenchmarks.jl` | Fine-grained component timing |

## Interpreting Results

### FFI Overhead Categories

1. **Pure FFI Tax** (~500-2000 ns)
   - ccall overhead
   - GC.@preserve management
   - TVMAny wrapper allocation

2. **Type Conversion** (~50-200 ns per argument)
   - POD types (Int, Float, Bool): cheapest
   - Objects (String, Function): require refcounting
   - Arrays: require TensorView + DLTensor metadata

3. **Callback Dispatch** (~1000-3000 ns)
   - Dict lookup in callback_registry
   - copy_value for borrowed arguments
   - Exception handling setup

### When FFI Overhead Matters

| Scenario | FFI Overhead Impact |
|----------|---------------------|
| Inference (large models) | Negligible (<0.1%) |
| Small tensor ops (1000s/sec) | Noticeable (1-5%) |
| Tight loops (100k+/sec) | Significant (10%+) |

### Optimization Tips

1. **Batch operations** - Reduce FFI call frequency
2. **Reuse TensorView** - Avoid repeated metadata creation
3. **Pre-allocate** - Use mutable output buffers
4. **Raw ccall** - For extreme hot paths (advanced)

## Expected Performance

On a typical modern CPU (2020+):

```
Julia empty function:     ~2-5 ns
TVM empty function:       ~500-2000 ns
TVM scalar identity:      ~800-3000 ns
TVM array identity:       ~2000-10000 ns (size dependent)
```

FFI overhead is typically **100-500x** a pure Julia call, but:
- Actual compute dominates in real workloads
- 1μs overhead is trivial for 1ms+ inference calls

