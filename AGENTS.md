# Agent Guide: TVMFFI.jl

This document outlines the workflow and goals for working on the `TVMFFI.jl` Julia package.

## Version Control: Jujutsu (`jj`)

This repository uses **Jujutsu (jj)** for version control. Do **NOT** use standard `git` commands for creating commits or branches.

### Common Commands
-   **Start work**: `jj new` (creates a new anonymous change on top of the current one).
-   **Save progress**: `jj describe -m "Description of changes"` (gives the current change a commit message).
-   **View status**: `jj st` (shows the current working copy status and parent chain).
-   **View log**: `jj log` (shows the commit graph).
-   **Push**: `jj git push` (pushes changes to the remote).

**Important**: You are always in a "working copy" commit. You don't need to "stage" files. Just modify them and `jj describe` to name the commit.

## Development Workflow

### Environment
-   Activate the environment: `julia --project=.`
-   Instantiate dependencies: `using Pkg; Pkg.instantiate()`

### Testing
-   Run tests: `using Pkg; Pkg.test()`
-   Run specific test file: `include("test/runtests.jl")` or specific files in `test/`.

### Code Style
-   This project uses `.JuliaFormatter.toml`.
-   Format code: `using JuliaFormatter; format(".")`

## Roadmap & Tasks

The immediate goal is to reach feature parity with Python/Rust bindings. See `../STATUS.md` for a detailed matrix.

### Priority 1: Function Registration ✅ COMPLETE
We need to allow Julia functions to be called by TVM.
1.  ✅ **`CallbackFunctionObjImpl`**: Implemented via `safe_call` C callback with Julia function registry.
2.  ✅ **`register_global_func`**: Fully functional API that registers Julia functions into TVM global registry.
3.  ✅ **Argument Conversion**: `to_tvm_any`, `take_value`, and `copy_value` work for all major types (int, float, bool, string, objects).
4.  ✅ **Exception Handling**: Julia exceptions are properly caught and translated to TVM errors.

### Priority 2: Object Registration ⚠️ Partially Complete
Allow defining custom TVM objects in Julia.
1.  ✅ **Basic `register_object` function**: Implemented. Can register type keys and allocate type indices.
2.  ✅ **Type queries**: `get_type_index` works for looking up registered types.
3.  **TODO: `@register_object` Macro**: Create a macro to generate full boilerplate (vtable, field accessors, methods) for a Julia struct to be a complete `TVMObject`.

### Priority 3: System Library ✅ COMPLETE
1.  ✅ **`system_lib()`**: Access statically linked TVM modules.
2.  ✅ **Module introspection**: `inspect_source()`, `get_module_kind()`.
3.  ✅ **Module export**: `write_to_file()` for saving compiled modules.
4.  ✅ **Function checking**: `implements_function()` to verify function availability.

### Priority 4: Documentation ✅ COMPLETE
1.  ✅ **Basic Structure**: `index.md` and `api.md` created.
2.  ✅ **API Reference**: Comprehensive docstrings for exported types and functions.
3.  ✅ **Examples**: Practical examples in `README.md` and `examples/` directory.

## Directory Structure
-   `src/LibTVMFFI.jl`: Low-level C bindings (generated/maintained manually).
-   `src/TVMFFI.jl`: Main entry point.
-   `src/any.jl`: TVMAny/TVMAnyView ownership-aware containers.
-   `src/conversion.jl`: ABI boundary layer (to_tvm_any, take_value, copy_value).
-   `src/function.jl`: Function wrappers.
-   `src/object.jl`: Object wrappers. **Focus here for Priority 2.**
-   `src/dlpack.jl`: DLPack zero-copy tensor exchange (TVMTensor, from_dlpack).
-   `src/gpuarrays_support.jl`: GPU array integration via DLPack.jl's type dispatch.
-   `ext/CUDAExt.jl`: Placeholder (DLPack.jl provides CUDA support).
-   `ext/AMDGPUExt.jl`: AMD ROCm support (DLPack.share, dldevice for ROCArray).
-   `ext/MetalExt.jl`: Apple Metal support (DLPack.share, dldevice for MtlArray).
-   `docs/`: Documentation source and build scripts.

---

## Memory Safety Guidelines

### Critical Rules for C FFI Code

When working with C API bindings, **ALWAYS** follow these rules:

#### 1. GC Safety: Use `GC.@preserve` for ALL Pointer Passing

**Rule**: Any time you extract a pointer from a Julia object and pass it to C, you **MUST** use `GC.@preserve`.

```julia
# ❌ WRONG - GC can collect str during C call
str = "hello"
byte_array = LibTVMFFI.TVMFFIByteArray(pointer(str), sizeof(str))
ret = some_c_function(byte_array)  # CRASH!

# ✅ CORRECT
str = "hello"
GC.@preserve str begin
    byte_array = LibTVMFFI.TVMFFIByteArray(
        Ptr{UInt8}(pointer(str)), UInt(sizeof(str))
    )
    ret = some_c_function(byte_array)
end
```

**Why**: Julia's GC can run at any time, even during C calls. Without `@preserve`, the string/array may be freed while C is reading it → segfault.

**Watch out for**:
- String → `TVMFFIByteArray` conversions
- Array → pointer conversions  
- Field access: `GC.@preserve obj.field` ❌ → Extract to local first: `x = obj.field; GC.@preserve x` ✅

#### 2. Reference Counting: Understand Ownership

**The Golden Rule**: Every `IncRef` must have a matching `DecRef`.

**Ownership Model**:
```julia
# Scenario 1: Taking ownership (C gives us a reference)
function from_c_api_returns_new_ref(any::TVMFFIAny)
    handle = reinterpret(TVMFFIObjectHandle, any.data)
    return TVMObject(handle; own=false)  # Don't IncRef, we already own it
    # Finalizer will DecRef → balanced
end

# Scenario 2: Borrowing (C lends us a reference)
function from_c_api_borrowed_ref(any::TVMFFIAny)
    handle = reinterpret(TVMFFIObjectHandle, any.data)
    return TVMObject(handle; own=true)   # IncRef to copy reference
    # Finalizer will DecRef → C's reference untouched
end

# Scenario 3: Passing to C (we create a temporary reference)
function pass_to_c(obj::TVMObject)
    LibTVMFFI.TVMFFIObjectIncRef(obj.handle)  # +1 for C
    result = some_c_function(obj.handle)
    LibTVMFFI.TVMFFIObjectDecRef(obj.handle)  # -1 cleanup
    return result
end
```

**Quick Reference**:
- `borrowed=true`: IncRef immediately, DecRef in finalizer (copy reference)
- `borrowed=false`: Don't IncRef, DecRef in finalizer (take ownership)

> **Design Decision (2025-11-26)**: The `borrowed` parameter has NO DEFAULT VALUE.
> This is intentional - forces explicit semantics at every call site, preventing misuse.
> These constructors are internal API; users should not call them directly.

#### 3. Cross-Language Verified Reference Semantics

> **Audit Date**: 2025-11-26
> **Verified Against**: C++ (`object.h`, `function.h`), Python (`object.pxi`, `function.pxi`), Rust (`object.rs`, `any.rs`, `function.rs`)

**C API Return Semantics** (when Julia receives from C):

| C API Function | Returns | Julia `borrowed` | Rationale |
|----------------|---------|------------------|-----------|
| `TVMFFIFunctionGetGlobal` | New reference | `false` | Caller owns, don't IncRef |
| `TVMFFIFunctionCall` result | New reference | `false` | Caller owns the result |
| `TVMFFIErrorMoveFromRaised` | Moved ownership | `false` | "Move" = take ownership |
| `TVMFFIStringFromByteArray` | New reference | N/A | Direct use, finalizer handles |
| `TVMFFIBytesFromByteArray` | New reference | N/A | Direct use, finalizer handles |
| Callback arguments | Borrowed | `true` | Must IncRef to keep alive |

**Julia → C Passing Pattern**:

```julia
# When passing TVMObject/TVMFunction to C API:
function (func::TVMFunction)(args...)
    # 1. IncRef before passing (to_tvm_any does this)
    args_array[i] = to_tvm_any(arg)  # IncRef inside
    
    # 2. Call C API
    ret = LibTVMFFI.TVMFFIFunctionCall(...)
    
    # 3. DecRef after call (cleanup)
    for arg_any in args_array
        if is_object(arg_any)
            LibTVMFFI.TVMFFIObjectDecRef(obj_ptr)
        end
    end
end
```

**Cross-Language Equivalents**:

| Julia | C++ | Rust | Python |
|-------|-----|------|--------|
| `TVMObject(h; borrowed=false)` | `ObjectPtrFromOwned(h)` | `ObjectArc::from_raw(h)` | Direct assign to `.chandle` |
| `TVMObject(h; borrowed=true)` | `ObjectPtrFromUnowned(h)` | Clone after `from_raw` | `TVMFFIAnyViewToOwnedAny` |
| `take_value(TVMAny(raw))` | Return from function | `Any::from_raw_ffi_any` | `make_ret(result)` |
| `copy_value(TVMAnyView(raw))` | Callback arg handling | `T::copy_from_any_view` | `TVMFFIAnyViewToOwnedAny` |

#### 3.1 TVMAny/TVMAnyView Type System (NEW)

> **Added**: 2025-11-26
> **Inspired by**: Rust's `Any` and `AnyView<'a>` types

Julia now has ownership-aware value containers that make reference semantics explicit at the type level:

```julia
# TVMAnyView - Borrowed view (no lifetime management)
# Use for callback arguments, temporary access
view = TVMAnyView(raw_any)
value = copy_value(view)  # Copies reference (IncRef for objects)

# TVMAny - Owned value (manages reference count)
# Use for function returns, storing values
owned = TVMAny(raw_any)
value = take_value(owned)  # Takes ownership, invalidates `owned`

# Convert view to owned (uses TVMFFIAnyViewToOwnedAny C API)
owned = TVMAny(view)
```

**Key Design Decisions**:

1. **`take_value` invalidates `TVMAny`**: After `take_value(any)`, the TVMAny's data is cleared to prevent double-free. The returned wrapper takes over reference management.

2. **`copy_value` is always safe**: Uses `TVMFFIAnyViewToOwnedAny` C API to properly handle all type conversions (objects, raw strings, byte arrays, rvalue refs). Original view remains valid.

**Usage in function.jl**:

```julia
# Callback arguments - use TVMAnyView + copy_value
view = TVMAnyView(unsafe_load(arg_ptr))
julia_args[i] = copy_value(view)

# Function returns - use TVMAny + take_value
result_owned = TVMAny(result[])
julia_result = take_value(result_owned)
```

#### 4. Common Pitfalls

| Mistake | Impact | How to Avoid |
|---------|--------|--------------|
| Missing `GC.@preserve` | Segfault, data corruption | Always wrap `pointer()` → C calls |
| Wrong `borrowed` parameter | Double-free, use-after-free | Check if C API returns new ref or borrows |
| Forgetting `DecRef` | Memory leak | Every `IncRef` needs matching `DecRef` |
| Preserving fields directly | Syntax error | Extract to local: `x = obj.field` |

#### 5. Code Review Checklist

Before committing C FFI code, verify:

- [ ] Every `pointer(x)` is inside `GC.@preserve x`
- [ ] Every `TVMFFIByteArray` construction preserves the source data
- [ ] Every `IncRef` has a corresponding `DecRef`
- [ ] `borrowed` parameter matches reference source (new ref vs borrowed ref)
- [ ] Finalizers are registered for all heap objects
- [ ] NULL pointer checks before dereferencing

#### 6. Testing for Memory Safety

Add tests that stress memory management:

```julia
@testset "Memory Safety" begin
    # Force GC between operations
    for i in 1:1000
        result = some_function(data)
        GC.gc()  # Try to trigger use-after-free bugs
        @test result == expected
    end
    
    # Test with exception paths
    @test_throws TVMError error_function()
    GC.gc()  # Ensure error cleanup is correct
end
```

---

## Design Principles (Linus Style)

### 1. Good Taste - Eliminate Special Cases

Use existing abstractions instead of reimplementing:

```julia
# ❌ BAD: String matching and module navigation hacks
function detect_backend(arr)
    type_name = string(typeof(arr).name.name)
    if occursin("Cu", type_name)
        cuda_module = _navigate_to_root_module(arr, :CUDA)  # Ugly!
        return (:CUDA, cuda_module.deviceid(...))
    elseif occursin("ROC", type_name)
        ...
    end
end

# ✅ GOOD: Use DLPack.jl's type dispatch
function _dlpack_to_tvm_device(arr)
    dlpack_dev = DLPack.dldevice(arr)  # DLPack handles everything!
    return DLDevice(Int32(dlpack_dev.device_type), Int32(dlpack_dev.device_id))
end
```

The refactored code deleted 120 lines and added 40 - a 60% reduction by using proper abstractions.

### 2. Simplicity - Direct Mapping

```julia
# Struct layout must match C exactly
struct TVMFFIObject
    combined_ref_count::UInt64  # Match C layout
    type_index::Int32
    __padding::UInt32
    deleter::Ptr{Cvoid}
end
```

No intermediate wrappers. One Julia struct = One C struct.

### 3. Practical - Solve Real Problems

- If C API doesn't support it, don't hack around it
- If existing code works, verify and document it (don't rewrite)
- Defer advanced features until they're actually needed

---

## Future Design: GC Pooling for Julia Object Exposure

> Reference: MWORKS.Syslab团队的GC Pooling技术（同元软控）

### Problem Statement

When exposing Julia objects to external languages (like TVM/C++), we face a challenge:
- Julia GC may collect objects that are still referenced by the external language
- External language has no way to participate in Julia's GC

### Current Implementation

`function.jl` uses a simple registry approach:

```julia
const _callback_registry = Dict{Ptr{Cvoid}, Any}()

function register_global_func(name, func)
    func_ref = Ref{Any}(func)
    resource_handle = Base.unsafe_convert(Ptr{Cvoid}, func_ref)
    _callback_registry[resource_handle] = func_ref  # Keep alive
    # ... register with TVM ...
end

function callback_deleter(resource_handle)
    delete!(_callback_registry, resource_handle)  # Allow GC
end
```

**Limitations**:
- Dict uses pointer as key → hash overhead
- No slot reuse after deletion → memory fragmentation
- Pointer debugging is harder than integer indices

### GC Pooling Pattern

Core idea: Use a **slot pool** as GC root, expose **integer indices** instead of pointers.

```julia
# Object pool as GC root
const JULIA_OBJECT_POOL = Vector{Any}(undef, 1024)
const POOL_FREELIST = Vector{Int}()
const POOL_LOCK = ReentrantLock()

function pool_insert(obj)::Int
    lock(POOL_LOCK) do
        idx = if isempty(POOL_FREELIST)
            push!(JULIA_OBJECT_POOL, obj)
            length(JULIA_OBJECT_POOL)
        else
            pop!(POOL_FREELIST)
            JULIA_OBJECT_POOL[idx] = obj
            idx
        end
        return idx
    end
end

function pool_get(idx::Int)
    @inbounds JULIA_OBJECT_POOL[idx]
end

function pool_release(idx::Int)
    lock(POOL_LOCK) do
        JULIA_OBJECT_POOL[idx] = nothing  # Allow GC
        push!(POOL_FREELIST, idx)
    end
end
```

**Benefits**:
- O(1) slot reuse via freelist
- Integer indices are easier to debug
- No hash computation
- Compact memory layout

### When to Implement

**Do NOT implement now** - current `_callback_registry` works fine for:
- Global function registration (infrequent)
- Long-lived callbacks

**Implement when**:
1. **`@register_object` macro** (Priority 2) - Julia objects exposed as TVM objects need lifecycle management
2. **Callback returns Julia arrays** - If TVM needs to hold array references beyond callback scope
3. **Performance issues** - If profiling shows registry operations as bottleneck

### Integration Points

When implementing, modify:
1. `src/function.jl`: Replace `_callback_registry` Dict with pool
2. `src/object.jl`: Use pool for `@register_object` macro
3. C callbacks: Pass slot index instead of pointer

---

## DLPack Zero-Copy Tensor Exchange ✅ IMPLEMENTED

> **Implementation Date**: 2025-11-26
> **Status**: COMPLETE
> **Reference**: [DLPack.jl](https://github.com/p-zubieta/DLPack.jl)

### Problem Statement

Current `TensorView` uses `kTVMFFIDLTensorPtr` (type_index=7) which is just a raw pointer:
- **No ownership information** - TVM doesn't know when Julia will free the data
- **Must copy on receive** - Julia doesn't know when TVM will free the data
- **GC.@preserve required** - Manual lifetime management is error-prone

### Solution: Use `kTVMFFITensor` with DLPack Protocol

Replace raw `DLTensorPtr` with managed `TVMTensor` objects (type_index=70) that have:
- Reference counting for safe lifecycle management
- DLPack protocol for zero-copy data exchange

### Implementation Plan

#### Phase 1: Add DLPack.jl Dependency

```toml
# TVMFFI/Project.toml
[deps]
DLPack = "d1985e75-91ee-4ca6-9615-d8afb9e46da7"
```

#### Phase 2: Bind C API Functions

```julia
# LibTVMFFI.jl additions:

# Julia Array → TVMTensor (via DLPack)
function TVMFFITensorFromDLPackVersioned(
    from::Ptr{DLManagedTensorVersioned},
    require_alignment::Int32,
    require_contiguous::Int32
)
    out = Ref{TVMFFIObjectHandle}(C_NULL)
    ret = ccall(
        (:TVMFFITensorFromDLPackVersioned, libtvm_ffi),
        Cint,
        (Ptr{Cvoid}, Int32, Int32, Ptr{TVMFFIObjectHandle}),
        from, require_alignment, require_contiguous, out
    )
    return ret, out[]
end

# TVMTensor → Julia Array (via DLPack)
function TVMFFITensorToDLPackVersioned(from::TVMFFIObjectHandle)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    ret = ccall(
        (:TVMFFITensorToDLPackVersioned, libtvm_ffi),
        Cint,
        (TVMFFIObjectHandle, Ptr{Ptr{Cvoid}}),
        from, out
    )
    return ret, out[]
end
```

#### Phase 3: Implement Conversions ✅

```julia
# src/dlpack.jl - IMPLEMENTED

import DLPack: from_dlpack

"""
    TVMTensor(arr::StridedArray) -> TVMTensor

Convert Julia array to TVMTensor via DLPack protocol (zero-copy).
The returned TVMTensor holds a reference to the Julia array.
"""
function TVMTensor(arr::StridedArray)
    capsule = DLPack.share(arr)
    # ... setup deleter and pool ...
    ret, handle = LibTVMFFI.TVMFFITensorFromDLPack(...)
    return TVMTensor(handle; borrowed=false)
end

"""
    from_dlpack(tensor::TVMTensor) -> AbstractArray

Convert TVMTensor to Julia array via DLPack protocol (zero-copy).
Returns Array for CPU, CuArray for CUDA (when CUDA.jl loaded).
"""
function DLPack.from_dlpack(tensor::TVMTensor)
    ret, dlpack_ptr = LibTVMFFI.TVMFFITensorToDLPack(tensor.handle)
    managed = DLPack.DLManagedTensor(Ptr{DLPack.DLManagedTensor}(dlpack_ptr))
    return Base.unsafe_wrap(managed, tensor)
end
```

### Type Mapping

| Scenario | API | Result |
|----------|-----|--------|
| Julia → TVM | `TVMTensor(arr)` | `kTVMFFITensor` (refcounted) |
| TVM → Julia | `from_dlpack(tensor)` | `Array` / `CuArray` (zero-copy) |
| Manual control | `TensorView(arr)` | `kTVMFFIDLTensorPtr` (no refcount) |

### API Usage

```julia
using TVMFFI
using DLPack: from_dlpack

# Recommended usage (zero-copy, safe)
arr = rand(Float32, 3, 4)
tensor = TVMTensor(arr)        # Julia → TVM
arr2 = from_dlpack(tensor)     # TVM → Julia

# Verify zero-copy
arr[1] = 99.0f0
@assert 99.0f0 in arr2         # Same memory!

# High-performance usage (manual lifetime)
view = TensorView(arr)
GC.@preserve arr begin
    tvm_func(view)  # Direct DLTensorPtr, no refcount overhead
end
```

### Benefits

1. **Zero-copy** - No data copying in either direction
2. **Safe** - Reference counting manages lifecycle
3. **Standard API** - `from_dlpack` matches NumPy/PyTorch/JAX naming
4. **GPU support** - Automatic CuArray/MtlArray/ROCArray via extensions
5. **Backward compatible** - `TensorView` still available for performance

### GPU Support via Package Extensions

> **Updated**: 2025-11-26
> **Design**: Use DLPack.jl's type dispatch - no code duplication!

GPU support leverages Julia 1.9+ package extensions and **DLPack.jl**:

```
ext/
├── CUDAExt.jl    # Placeholder - DLPack.jl provides CUDA support
├── MetalExt.jl   # Loaded when Metal.jl is imported (TVMFFI provides)
└── AMDGPUExt.jl  # Loaded when AMDGPU.jl is imported (TVMFFI provides)
```

**Architecture (Linus-style: no duplication)**:

| Backend | DLPack Integration | Provider |
|---------|-------------------|----------|
| NVIDIA CUDA | `DLPack.dldevice`, `DLPack.share` | **DLPack.jl's CUDAExt** |
| Apple Metal | `DLPack.dldevice`, `DLPack.share` | **TVMFFI's MetalExt** |
| AMD ROCm | `DLPack.dldevice`, `DLPack.share` | **TVMFFI's AMDGPUExt** |

**Key Insight**: Device detection uses `DLPack.dldevice()` directly:

```julia
# gpuarrays_support.jl - Clean and simple!
function _dlpack_to_tvm_device(arr)
    dlpack_dev = DLPack.dldevice(arr)  # DLPack handles type dispatch
    return DLDevice(Int32(dlpack_dev.device_type), Int32(dlpack_dev.device_id))
end
```

**Deleted Code** (was redundant):
- `detect_backend()` - DLPack.dldevice() does this
- `gpu_backend_to_dldevice()` - unnecessary conversion
- `_navigate_to_root_module()` - ugly hack, no longer needed
- `_extract_cuda_device_id()` etc. - DLPack handles this

**Usage**:
```julia
using TVMFFI
using CUDA  # Triggers DLPack.jl's CUDAExt automatically!

# Now TVMTensor works with CuArrays
cu_arr = CUDA.rand(Float32, 3, 4)
tensor = TVMTensor(cu_arr)  # Zero-copy!
arr = from_dlpack(tensor)   # → CuArray
```

**Supported GPU backends**:
| Backend | Extension | Array Type | DLDevice |
|---------|-----------|------------|----------|
| NVIDIA | DLPack.jl/CUDAExt | CuArray | kDLCUDA |
| Apple | TVMFFI/MetalExt | MtlArray | kDLMetal |
| AMD | TVMFFI/AMDGPUExt | ROCArray | kDLROCM |
