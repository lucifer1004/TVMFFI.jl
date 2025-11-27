# TVMFFI Benchmark Suite

This directory contains benchmarks for measuring FFI overhead and performance characteristics using [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl).

## Prerequisites

The benchmarks have their own `Project.toml` to avoid adding BenchmarkTools as a main dependency.

```bash
cd TVMFFI/benchmarks
julia --project=. -e 'using Pkg; Pkg.develop(path=".."); Pkg.instantiate()'
```

## Running Benchmarks

### Julia vs Python Comparison

Compare Julia FFI performance against Python `bench_example.py`:

```bash
julia --project=. bench_example.jl
```

### FFI Overhead Analysis

End-to-end FFI overhead measurement with different array sizes:

```bash
julia --project=. ffi_overhead.jl
```

### Component Breakdown

Fine-grained timing of individual FFI components:

```bash
julia --project=. microbenchmarks.jl
```

### CUDA Analysis

GPU FFI overhead breakdown (requires CUDA.jl):

```bash
julia --project=. bench_cuda_breakdown.jl
```

## Benchmark Files

| File | Description |
|------|-------------|
| `bench_example.jl` | Julia vs Python comparison (mirrors `bench_example.py`) |
| `ffi_overhead.jl` | End-to-end FFI overhead measurement |
| `microbenchmarks.jl` | Fine-grained component timing |
| `bench_cuda_breakdown.jl` | CUDA FFI overhead breakdown |
| `example_cuda.jl` | CUDA functionality examples (not a benchmark) |

## Julia vs Python Performance Summary

Based on benchmarks run on the same machine:

| Operation | Julia | Python | Speedup |
|:---|:---:|:---:|:---:|
| Broadcast add (CPU) | 3 ns | 207 ns | **64x faster** |
| Broadcast add (CUDA) | 15 µs | 15 µs | Equal |
| TVM NOP (no args) | 12 ns | 72 ns | **6x faster** |
| TVM NOP (pre-converted) | 70 ns | 72 ns | Equal |
| TVM autodlpack (CPU) | 32 ns | 298 ns | **~10x faster** |
| TVM GPU autodlpack | 1.8 µs | 902 ns | 2x slower |
| CUDA stream query | 22 ns | 85 ns | **4x faster** |

## Interpreting Results

### FFI Overhead Categories

1. **Pure FFI Tax** (~20-70 ns optimized, ~500-2000 ns fallback)
   - ccall overhead
   - GC.@preserve management
   - TVMAny wrapper allocation

2. **Type Conversion** (~5-50 ns per argument)
   - POD types (Int, Float, Bool): cheapest (~5 ns)
   - Objects (String, Function): require refcounting (~50-100 ns)
   - Arrays: require TensorView + DLTensor metadata (~30-100 ns)

3. **Callback Dispatch** (~100-300 ns)
   - Dict lookup in callback_registry
   - copy_value for borrowed arguments
   - Exception handling setup

### When FFI Overhead Matters

| Scenario | FFI Overhead Impact |
|----------|---------------------|
| Inference (large models) | Negligible (<0.01%) |
| Small tensor ops (1000s/sec) | Minimal (<1%) |
| Tight loops (100k+/sec) | Noticeable (5-10%) |

### Optimization Tips

1. **Batch operations** - Reduce FFI call frequency
2. **Reuse TensorView** - Avoid repeated metadata creation
3. **Pre-convert TVMTensor** - For repeated GPU operations
4. **1-10 args** - Use specialized methods (auto-selected)

## Expected Performance

On a typical modern CPU (2020+):

```
Julia empty function:        ~2-5 ns
TVM func() (no args):        ~12 ns  (6x python)
TVM func(Int64):             ~20 ns  
TVM func(Array) autodlpack:  ~30 ns  (10x python)
TVM func(Array) TVMTensor:   ~400 ns (with conversion)
```

Julia FFI is typically **2-10x faster** than Python FFI for most operations!
