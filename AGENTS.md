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
-   `src/any.jl`: TVMAny/TVMAnyView ownership-aware containers. **NEW**
-   `src/conversion.jl`: ABI boundary layer (to_tvm_any, take_value, copy_value).
-   `src/function.jl`: Function wrappers. **Focus here for Priority 1.**
-   `src/object.jl`: Object wrappers. **Focus here for Priority 2.**
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

Use proper data structures to avoid if/else chains:

```julia
# Good: Unified handling via generic type parameter
mutable struct TensorView{T, S}
    source::S  # Works for Array, CuArray, ROCArray, etc.
end

# No need for:
if source isa CuArray
    # special GPU handling
elseif source isa ROCArray
    # another special case
```

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

## Future Design: DLPack Zero-Copy Tensor Exchange

> **Design Date**: 2025-11-26
> **Status**: PLANNED
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

#### Phase 3: Implement Conversions

```julia
# conversion.jl or new dlpack.jl:

using DLPack

"""
    array_to_tvm_tensor(arr::AbstractArray) -> TVMTensor

Convert Julia array to TVMTensor via DLPack protocol (zero-copy).
The returned TVMTensor holds a reference to the Julia array.
"""
function array_to_tvm_tensor(arr::AbstractArray)::TVMTensor
    # Use DLPack.jl to create DLManagedTensorVersioned
    # Then call TVMFFITensorFromDLPackVersioned
    # Return managed TVMTensor
end

"""
    tvm_tensor_to_array(tensor::TVMTensor) -> AbstractArray

Convert TVMTensor to Julia array via DLPack protocol (zero-copy).
The returned array is a view into the TVMTensor's data.
"""
function tvm_tensor_to_array(tensor::TVMTensor)
    # Call TVMFFITensorToDLPackVersioned
    # Use DLPack.jl to wrap as Julia array
end
```

#### Phase 4: Update Function Call Path

```julia
# function.jl changes:

function (func::TVMFunction)(args...)
    for (i, arg) in enumerate(args)
        if arg isa AbstractArray && !(arg isa TensorView)
            # NEW: Convert to TVMTensor (kTVMFFITensor) instead of TensorView
            tensor = array_to_tvm_tensor(arg)
            args_any[i] = TVMAny(tensor)
            # TVMTensor manages lifecycle via refcount
        elseif arg isa TensorView
            # Keep for performance-critical code
            args_any[i] = TVMAny(arg)
        end
    end
end
```

### Type Mapping After Implementation

| Scenario | Current | After DLPack |
|----------|---------|--------------|
| Julia → TVM (args) | `TensorView` → `kTVMFFIDLTensorPtr` | `array_to_tvm_tensor` → `kTVMFFITensor` |
| TVM → Julia (return) | `kTVMFFIDLTensorPtr` → **COPY** | `kTVMFFITensor` → `tvm_tensor_to_array` (zero-copy) |
| Julia → TVM (callback return) | `TensorView` → `kTVMFFIDLTensorPtr` | `kTVMFFITensor` (zero-copy) |
| TVM → Julia (callback arg) | `kTVMFFIDLTensorPtr` → **COPY** | `kTVMFFITensor` → `TVMTensor` (zero-copy) |

### API After Implementation

```julia
# Recommended usage (zero-copy, safe)
arr = rand(Float32, 3, 4)
result = tvm_func(arr)  # Auto-converts to TVMTensor
julia_arr = tvm_tensor_to_array(result)  # Zero-copy view

# High-performance usage (zero-copy, manual lifetime)
view = TensorView(arr)
GC.@preserve arr begin
    tvm_func(view)  # Direct DLTensorPtr, no refcount overhead
end
```

### Benefits

1. **Zero-copy** - No data copying in either direction
2. **Safe** - Reference counting manages lifecycle
3. **Compatible** - Works with DLPack.jl ecosystem (PyTorch, JAX, CuPy via PyCall)
4. **Backward compatible** - `TensorView` still available for performance

### When to Implement

**Implement when**:
1. Users report performance issues due to tensor copying
2. Need interop with Python ML frameworks via DLPack
3. Large tensor sizes make copying prohibitive

**Current workaround**: Use `TVMTensor` directly for TVM-managed data, accept copy overhead for callback arguments.
