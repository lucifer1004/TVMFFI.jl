# Agent Guide: TVMFFI.jl

Technical guide for AI agents and developers working on the TVMFFI.jl project.

## Quick Start

### Version Control: Jujutsu (`jj`)

This repository uses **Jujutsu (jj)** instead of git.

```bash
jj new                      # Create new change
jj describe -m "message"    # Add commit message
jj st                       # View status
jj log                      # View history
jj git push                 # Push to remote
```

### Development Environment

```bash
julia --project=.           # Activate environment
```

```julia
using Pkg; Pkg.instantiate()  # Install dependencies
using Pkg; Pkg.test()         # Run tests
using JuliaFormatter; format(".")  # Format code
```

---

## Project Architecture

### Directory Structure

```
src/
├── LibTVMFFI.jl       # C API bindings (low-level)
├── TVMFFI.jl          # Main entry point
├── any.jl             # TVMAny/TVMAnyView ownership containers
├── conversion.jl      # ABI boundary layer (to_tvm_any, take_value, copy_value)
├── function.jl        # Function wrappers
├── object.jl          # Object wrappers
├── tensor.jl          # TensorView implementation
├── dlpack.jl          # DLPack zero-copy exchange
└── gpuarrays_support.jl  # GPU array support

docs/
├── src/
│   ├── index.md        # Symlink to ../../README.md (single source of truth)
│   └── api.md          # API reference documentation
└── make.jl            # Documentation build configuration

ext/
├── CUDAExt.jl         # NVIDIA CUDA support (device detection, sync callback, tensor view)
├── AMDGPUExt.jl       # AMD ROCm support
└── MetalExt.jl        # Apple Metal support
```

**Note**: `docs/src/index.md` is a symbolic link to `README.md` to maintain a single source of truth for documentation.

### Core Types

| Type | Purpose | type_index |
|------|---------|------------|
| `TVMObject` | Generic TVM object wrapper | varies |
| `TVMFunction` | Callable function | 16 |
| `TVMTensor` | Reference-counted tensor | 70 |
| `TensorView` | Lightweight pointer view | 7 |
| `TVMAny` | Owned value container | - |
| `TVMAnyView` | Borrowed value view | - |

---

## Memory Safety Guidelines

### 1. GC Safety: Pointer Passing Requires `GC.@preserve`

```julia
# ❌ Wrong - GC may collect str during C call
str = "hello"
byte_array = LibTVMFFI.TVMFFIByteArray(pointer(str), sizeof(str))
ret = some_c_function(byte_array)  # Crash!

# ✅ Correct
str = "hello"
GC.@preserve str begin
    byte_array = LibTVMFFI.TVMFFIByteArray(
        Ptr{UInt8}(pointer(str)), UInt(sizeof(str))
    )
    ret = some_c_function(byte_array)
end
```

**Note**: `GC.@preserve obj.field` doesn't work; extract to local variable first.

### 2. Reference Counting: Ownership Model

**Golden Rule**: Every `IncRef` must have a matching `DecRef`.

```julia
# Scenario 1: Take ownership (C returns new reference)
TVMObject(handle; borrowed=false)  # Don't IncRef, finalizer will DecRef

# Scenario 2: Borrow (C lends to us)
TVMObject(handle; borrowed=true)   # IncRef, finalizer will DecRef
```

**C API Return Semantics**:

| C API Function | Return Type | Julia `borrowed` |
|----------------|-------------|------------------|
| `TVMFFIFunctionGetGlobal` | New reference | `false` |
| `TVMFFIFunctionCall` result | New reference | `false` |
| Callback arguments | Borrowed | `true` |

> **Design Decision**: `borrowed` parameter has **no default value**. Forced explicit specification prevents misuse.

### 3. TVMAny / TVMAnyView Type System

```julia
# TVMAnyView - Borrowed view (for callback arguments)
view = TVMAnyView(raw_any)
value = copy_value(view)  # Copy reference (object gets IncRef)

# TVMAny - Owned (for function returns)
owned = TVMAny(raw_any)
value = take_value(owned)  # Take ownership, owned becomes invalid
```

### 4. Code Review Checklist

- [ ] Every `pointer(x)` is inside `GC.@preserve x`
- [ ] Every `TVMFFIByteArray` construction protects source data
- [ ] Every `IncRef` has corresponding `DecRef`
- [ ] `borrowed` parameter matches reference origin
- [ ] All heap objects have registered finalizers

---

## DLPack Tensor Exchange

### API

```julia
# Julia → TVM (zero-copy)
arr = rand(Float32, 3, 4)
tensor = TVMTensor(arr)

# TVM → Julia (zero-copy)
arr2 = from_dlpack(tensor)

# Lightweight view (manual lifetime management)
view = TensorView(arr)
GC.@preserve arr begin
    tvm_func(view)
end
```

### Type Comparison

| Type | type_index | Ref Counting | Use Case |
|------|------------|--------------|----------|
| `TVMTensor` | 70 | ✅ Yes | `from_dlpack` returns, long-term cross-boundary holding |
| `TensorView` | 7 | ❌ No | **CPU/GPU arrays**, FFI call arguments (recommended) |

> **Note**: FFI calls now uniformly use `TensorView` for both CPU and GPU. `GC.@preserve` ensures data validity during call.

### GPU Support

| Backend | Extension | Array Type |
|---------|-----------|------------|
| NVIDIA CUDA | TVMFFI/CUDAExt | CuArray |
| Apple Metal | TVMFFI/MetalExt | MtlArray |
| AMD ROCm | TVMFFI/AMDGPUExt | ROCArray |

---

## Known Limitations

### 1. BenchmarkTools + GPU Arrays

**Problem**: `@benchmark` calls `GC.gc()` between iterations, causing segfaults when returned GPU arrays are collected.

```julia
# ❌ Crashes
@benchmark my_gpu_func($arr)

# ✅ Use manual timing
n = 10000
t_start = time_ns()
for _ in 1:n
    func(arr)
end
CUDA.synchronize()
t_avg = (time_ns() - t_start) / n
```

**Cause**: BenchmarkTools' `gcscrub()` interaction with CUDA.jl finalizers, not a TVMFFI bug.

### 2. GPU Array Performance (Optimized)

GPU arrays now use lightweight `TensorView` (same as CPU), skipping expensive `TVMFFITensorFromDLPack` C API calls:

| Operation | Julia | Python | Comparison |
|-----------|-------|--------|------------|
| GPU autodlpack (1 arg) | ~200 ns | ~900 ns | **4.5x faster** |
| GPU autodlpack (3 args) | ~580 ns | ~900 ns | **1.6x faster** |
| GPU identity | ~210 ns | - | Zero-copy |

GPU extensions automatically register `atexit` cleanup hooks to ensure TVM finalizers execute before GPU context destruction.

---

## Design Principles

### 1. Eliminate Special Cases

```julia
# ❌ String matching and module navigation hacks
function detect_backend(arr)
    type_name = string(typeof(arr).name.name)
    if occursin("Cu", type_name)
        ...
    end
end

# ✅ Use type dispatch
function _dlpack_to_tvm_device(arr)
    dlpack_dev = dldevice(arr)  # Dispatch handles everything
    return DLDevice(Int32(dlpack_dev.device_type), Int32(dlpack_dev.device_id))
end
```

### 2. Direct Mapping

```julia
# Julia struct layout must exactly match C
struct TVMFFIObject
    combined_ref_count::UInt64
    type_index::Int32
    __padding::UInt32
    deleter::Ptr{Cvoid}
end
```

### 3. Pragmatism

- Don't hack features not supported by C API
- If existing code works, verify and document (don't rewrite)
- Defer advanced features until truly needed

---

## Future Work (Optional)

### GC Pooling

Currently using `Dict{Ptr{Cvoid}, Any}` as callback registry. If bottlenecks appear in these scenarios, consider implementing slot pool:

- Large numbers of short-lived callbacks
- High-frequency function registration/unregistration

Core idea: Replace Dict with `Vector{Any}` + freelist, expose integer indices instead of pointers.

**Current Status**: Not needed. Existing implementation is efficient enough for global function registration.

---

## Performance Optimization Records

### Successful Optimizations

#### 1. TensorView NTuple Optimization (2024-11)

**Problem**: TensorView's `shape` and `strides` used `Vector{Int64}`, heap allocation on every creation.

**Solution**:
- `TensorView{T, S}` → `TensorView{T, S, N}`, N is dimensionality
- `shape::Vector{Int64}` → `_shape::NTuple{N, Int64}` (inline storage)
- `strides::Vector{Int64}` → `_strides::NTuple{N, Int64}`
- DLTensor shape/strides pointers point to inline storage via `pointer_from_objref + fieldoffset`

**Result**: ~128 bytes saved per TensorView

**Files**: `tensor.jl`, `gpuarrays_support.jl`

#### 2. FFI Call Path Optimization (2024-11)

**Problem**: `TVMFunction` calls had redundant allocations.

**Solution**:
- Merge `args_any` into `args_raw`: Store `TVMFFIAny` directly instead of `TVMAny` wrapper
- `Dict{Ptr, Tuple}` → `Vector{Tuple{Ptr, Any}}`: Linear search faster for small N with zero allocation

**Result**: `func(Float32[64])` from 1200B→688B (-43%), ~461ns→~365ns (-21%)

**Files**: `function.jl`

#### 3. Small Argument Count Specialization (2024-11)

**Problem**: Generic `(func::TVMFunction)(args...)` uses `Vector{Any}` to store GC refs and argument data.

**Solution**:
- Generate specialized methods for 1-4 arguments using named variables instead of `Vector`
- Extract common logic to `_convert_arg` helper function
- Use `Ref((raw1, raw2, ...))` instead of `Vector{TVMFFIAny}`
- Inline identity optimization checks, avoid `filter` allocation

**Result**:
- 2 arguments: 94 bytes/call (vs 5+ args 494 bytes)
- Single array argument: 158 bytes/call (vs generic path 432 bytes)

**Inspiration**:
- Python Cython: Cache setter function pointers by type
- Rust: `const STACK_LEN = 4` stack-allocated small array optimization

**Files**: `function.jl`

#### 4. GPU TensorView Optimization (2024-11)

**Problem**: GPU arrays passed via `TVMTensor` (requiring `TVMFFITensorFromDLPack` C API), 8x slower than CPU arrays.

**Solution**:
- GPU arrays also use `TensorView` (type_index=7), skipping C API call
- Add `_wrap_gpu_dltensor_view` interface to convert DLTensor to native GPU arrays (CuArray/MtlArray/ROCArray) in callbacks
- GPU extensions register sync callback + atexit hook, ensuring correct cleanup order

**Result**:
- GPU autodlpack: ~1600 ns → ~200 ns (**8x improvement**)
- Now 4.5x faster than Python (previously 1.8x slower than Python)

**Files**: `function.jl`, `dlpack.jl`, `tensor.jl`, `ext/CUDAExt.jl`, `ext/MetalExt.jl`, `ext/AMDGPUExt.jl`

#### 5. Unified atexit Cleanup (2024-11)

**Problem**: Multiple atexit hooks with overlapping functionality, GPU exit could segfault (finalizer order issues).

**Solution**:
- Add `GC.gc()` in `_cleanup_wrapped_arrays_at_exit` to force cleanup of all objects
- GPU extensions only call `_ensure_cleanup_atexit_registered()` to ensure registration
- Cleanup order: GPU sync → GC.gc() → GPU sync → clear handles

**Result**: Eliminated GPU exit segfaults, simplified code

**Files**: `dlpack.jl`, `ext/CUDAExt.jl`, `ext/MetalExt.jl`, `ext/AMDGPUExt.jl`

#### 6. 1-10 Argument Macro-Generated Specialization (2024-11)

**Problem**: Manually writing 1-4 argument specialized methods was repetitive.

**Solution**:
- Use `@_define_narg_method N` macro to generate specialized methods for 1-10 arguments
- Use `quote` blocks and `esc()` for clean metaprogramming
- Arguments packed as `NTuple` (stack-allocated), avoiding `Vector` heap allocation

**Result**: 80% code reduction, covers 1-10 arguments, same performance

**Files**: `function.jl`

---

### Failed Optimization Attempts

The following optimization attempts did not improve performance, documented to avoid repetition:

#### 1. ntuple in safe_call ❌

**Attempt**: Change `julia_args` and `arg_raws` from `Vector` to `ntuple`

**Result**: Memory reduced but time increased 30-50%

**Cause**: `ntuple(n) do i ... end` with runtime `n` triggers dynamic compilation, JIT overhead exceeds saved allocations

**Lesson**: `ntuple` only suitable when `N` is known at compile time

#### 2. take_value_raw (if-elseif chain) ❌

**Attempt**: Extract isbits type values directly from `TVMFFIAny`, skip `TVMAny` creation

```julia
function take_value_raw(raw)
    type_idx == kTVMFFINone && return nothing
    type_idx == kTVMFFIInt && return reinterpret(Int64, raw.data)
    # ...
    return take_value(TVMAny(raw))
end
```

**Result**: `func(Int64)` slightly faster, but `func()` returning `nothing` 29% slower

**Cause**: if-elseif branch overhead exceeds saved allocations in some JIT scenarios

#### 3. take_value_raw (Val dispatch) ❌

**Attempt**: Use Julia type dispatch + `Val` for zero-overhead dispatch

```julia
@inline _extract_by_type(::Val{kTVMFFINone}, raw) = nothing
@inline _extract_by_type(::Val{kTVMFFIInt}, raw) = reinterpret(Int64, raw.data)
# ...
take_value_raw(raw) = _extract_by_type(Val(raw.type_index), raw)
```

**Result**: Memory increased 32B, time slower

**Cause**: `Val(raw.type_index)` creates Val instance at runtime, requiring heap allocation

**Lesson**: Val dispatch is zero-overhead only when type parameter is compile-time constant

#### 4. take_value_raw (function table) ❌

**Attempt**: Pre-generate extractor function table, call by index

```julia
const _EXTRACTOR_FUNCS = ntuple(128) do i
    # Return corresponding extractor function
end
take_value_raw(raw) = _EXTRACTOR_FUNCS[raw.type_index + 1](raw)
```

**Result**: Time slower, memory unchanged

**Cause**: Function table indirect call overhead cancels any potential gains

---

### Optimization Guidelines

1. **Measure First**: Always use `@allocated` and `@elapsed`, avoid gut-feeling optimization
2. **Trust the JIT**: Julia JIT optimizes simple code well, complex manual optimization may backfire
3. **Compile-time vs Runtime**: `ntuple`, `Val` etc. only effective when parameters known at compile time
4. **Microbench ≠ Macrobench**: Zero allocation when measured alone, may differ in complete call
5. **Profile.Allocs**: Precisely locating allocation sources is more effective than guessing
